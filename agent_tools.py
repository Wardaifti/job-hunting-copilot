"""
agent_tools.py — the actual "hands" of the AI Job Hunting Copilot agent.

Each function here is a plain Python function that reads from and/or writes
to Lakebase. agent.py exposes these to the LLM as callable tools (OpenAI-style
function calling via the Databricks Foundation Model API) — the LLM decides
*which* tool to call and with *what arguments*, these functions do the actual
work against Lakebase.

Single-user demo model: this app doesn't implement login/auth (out of scope
for the capstone). One demo user is created on first run (see
get_or_create_demo_user) and every tool operates against that user_id.
"""

import os

from databricks.sdk import WorkspaceClient

import lakebase

EMBEDDING_ENDPOINT = os.environ.get("EMBEDDING_ENDPOINT", "databricks-gte-large-en")
CHAT_ENDPOINT = os.environ.get("CHAT_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")
DEMO_USER_EMAIL = os.environ.get("DEMO_USER_EMAIL", "demo@jobcopilot.local")

_w = WorkspaceClient()


# ---------------------------------------------------------------------
# User / profile
# ---------------------------------------------------------------------
def get_or_create_demo_user() -> dict:
    """Return the single demo user, creating it (with an empty profile) on
    first run. Called at the top of every request, so it's cheap and idempotent."""
    rows = lakebase.run_query("SELECT * FROM users WHERE email = %s", (DEMO_USER_EMAIL,))
    if rows:
        user = rows[0]
    else:
        rows = lakebase.run_returning(
            "INSERT INTO users (email, full_name) VALUES (%s, %s) RETURNING *",
            (DEMO_USER_EMAIL, "Demo User"),
        )
        user = rows[0]
        lakebase.run_write(
            "INSERT INTO profiles (user_id) VALUES (%s) ON CONFLICT (user_id) DO NOTHING",
            (user["user_id"],),
        )
    return user


def get_profile(user_id: int) -> dict:
    rows = lakebase.run_query("SELECT * FROM profiles WHERE user_id = %s", (user_id,))
    return rows[0] if rows else {}


def get_skills(user_id: int) -> list[dict]:
    return lakebase.run_query(
        "SELECT skill_name, years_experience FROM skills WHERE user_id = %s ORDER BY skill_name",
        (user_id,),
    )


def update_profile(user_id: int, headline: str = None, summary: str = None,
                    target_roles: list[str] = None, target_locations: list[str] = None,
                    min_salary: float = None, remote_only: bool = None) -> dict:
    """Partial update — only overwrites fields that are explicitly passed."""
    fields, params = [], []
    for col, val in [
        ("headline", headline), ("summary", summary),
        ("target_roles", target_roles), ("target_locations", target_locations),
        ("min_salary", min_salary), ("remote_only", remote_only),
    ]:
        if val is not None:
            fields.append(f"{col} = %s")
            params.append(val)
    if not fields:
        return get_profile(user_id)
    fields.append("updated_at = now()")
    params.append(user_id)
    lakebase.run_write(
        f"UPDATE profiles SET {', '.join(fields)} WHERE user_id = %s", tuple(params)
    )
    return get_profile(user_id)


def set_skills(user_id: int, skills: list[dict]) -> list[dict]:
    """Replace the user's skill list. skills = [{'skill_name': .., 'years_experience': ..}, ...]"""
    lakebase.run_write("DELETE FROM skills WHERE user_id = %s", (user_id,))
    for s in skills:
        name = (s.get("skill_name") or "").strip()
        if not name:
            continue
        lakebase.run_write(
            "INSERT INTO skills (user_id, skill_name, years_experience) VALUES (%s, %s, %s) "
            "ON CONFLICT (user_id, skill_name) DO UPDATE SET years_experience = EXCLUDED.years_experience",
            (user_id, name, s.get("years_experience")),
        )
    return get_skills(user_id)


# ---------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------
def embed_text(text: str) -> list[float]:
    response = _w.serving_endpoints.query(name=EMBEDDING_ENDPOINT, input=text)
    return response.data[0].embedding


def _chat_client():
    """OpenAI-compatible client pointed at the Databricks Foundation Model API.
    Used both here (for explanations/drafts) and in agent.py (for tool-calling)."""
    return _w.serving_endpoints.get_open_ai_client()


def _complete(system_prompt: str, user_prompt: str, max_tokens: int = 400) -> str:
    client = _chat_client()
    resp = client.chat.completions.create(
        model=CHAT_ENDPOINT,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------
# Search / rank
# ---------------------------------------------------------------------
def search_jobs(user_id: int, query_text: str, remote_only: bool = None,
                 min_salary: float = None, limit: int = 8) -> list[dict]:
    """Semantic search over job_embeddings (cosine distance), optionally
    filtered by remote/salary. Returns the best-matching chunk per job,
    deduplicated to one row per job_id, ranked by similarity."""
    query_vec = embed_text(query_text)

    filters = []
    params = {"qvec": query_vec, "limit": limit}
    if remote_only is not None:
        filters.append("j.remote = %(remote_only)s")
        params["remote_only"] = remote_only
    if min_salary is not None:
        filters.append("(j.salary_max IS NULL OR j.salary_max >= %(min_salary)s)")
        params["min_salary"] = min_salary
    where_clause = f"AND {' AND '.join(filters)}" if filters else ""

    sql = f"""
        SELECT DISTINCT ON (j.job_id)
            j.job_id, j.title, j.company, j.location, j.remote,
            j.salary_min, j.salary_max, j.url, j.tags,
            1 - (e.embedding <=> %(qvec)s::vector) AS similarity
        FROM job_embeddings e
        JOIN job_postings j ON j.job_id = e.job_id
        WHERE 1=1 {where_clause}
        ORDER BY j.job_id, e.embedding <=> %(qvec)s::vector
        LIMIT %(limit)s
    """
    rows = lakebase.run_query(sql, params)
    rows.sort(key=lambda r: r["similarity"], reverse=True)
    return rows


def explain_match(user_id: int, job_id: str) -> str:
    """LLM explains why a specific posting is or isn't a good fit for the
    user's stated profile + skills."""
    job_rows = lakebase.run_query(
        "SELECT title, company, description, tags, remote, salary_min, salary_max "
        "FROM job_postings WHERE job_id = %s", (job_id,)
    )
    if not job_rows:
        return "Couldn't find that job posting."
    job = job_rows[0]
    profile = get_profile(user_id)
    skills = get_skills(user_id)
    skills_str = ", ".join(f"{s['skill_name']} ({s['years_experience']}y)" for s in skills) or "not specified"

    prompt = f"""
Candidate profile:
- Headline: {profile.get('headline') or 'not specified'}
- Summary: {profile.get('summary') or 'not specified'}
- Target roles: {profile.get('target_roles') or 'not specified'}
- Skills: {skills_str}
- Remote only: {profile.get('remote_only')}
- Min salary: {profile.get('min_salary')}

Job posting:
- Title: {job['title']} at {job.get('company') or 'unknown company'}
- Remote: {job.get('remote')}
- Salary: {job.get('salary_min')}-{job.get('salary_max')}
- Tags: {job.get('tags')}
- Description: {(job.get('description') or '')[:1500]}

In 3-4 sentences, explain whether this is a good match and why, calling out
specific overlaps or gaps between the candidate's skills and the job's
requirements. Be direct and specific, not generic.
"""
    return _complete("You are a candid, specific career-matching assistant.", prompt)


# ---------------------------------------------------------------------
# Pipeline: save / stage / draft / notes / staleness
# ---------------------------------------------------------------------
def save_job(user_id: int, job_id: str, stage: str = "saved", match_score: float = None) -> dict:
    rows = lakebase.run_returning(
        """
        INSERT INTO saved_jobs (user_id, job_id, stage, match_score)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (user_id, job_id) DO UPDATE
            SET stage = EXCLUDED.stage, updated_at = now()
        RETURNING *
        """,
        (user_id, job_id, stage, match_score),
    )
    return rows[0]


def update_stage(user_id: int, job_id: str, new_stage: str) -> dict:
    valid = {"saved", "applied", "interviewing", "rejected", "offer"}
    if new_stage not in valid:
        raise ValueError(f"stage must be one of {valid}")
    rows = lakebase.run_returning(
        """
        UPDATE saved_jobs SET stage = %s, updated_at = now()
        WHERE user_id = %s AND job_id = %s
        RETURNING *
        """,
        (new_stage, user_id, job_id),
    )
    if not rows:
        raise ValueError("That job isn't saved to your pipeline yet — save it first.")
    return rows[0]


def list_pipeline(user_id: int, stage: str = None) -> list[dict]:
    sql = """
        SELECT sj.saved_job_id, sj.job_id, sj.stage, sj.match_score, sj.saved_at, sj.updated_at,
               j.title, j.company, j.location, j.remote, j.url
        FROM saved_jobs sj
        JOIN job_postings j ON j.job_id = sj.job_id
        WHERE sj.user_id = %s
    """
    params = [user_id]
    if stage:
        sql += " AND sj.stage = %s"
        params.append(stage)
    sql += " ORDER BY sj.updated_at DESC"
    return lakebase.run_query(sql, tuple(params))


def draft_application_material(user_id: int, job_id: str, kind: str = "cover_letter") -> dict:
    """kind: 'cover_letter' or 'resume_bullet'. Drafts with the LLM, saves it
    onto the applications row (upsert), returns the saved row."""
    job_rows = lakebase.run_query(
        "SELECT title, company, description FROM job_postings WHERE job_id = %s", (job_id,)
    )
    if not job_rows:
        raise ValueError("Job not found.")
    job = job_rows[0]
    profile = get_profile(user_id)
    skills = get_skills(user_id)
    skills_str = ", ".join(f"{s['skill_name']} ({s['years_experience']}y)" for s in skills) or "not specified"

    saved_rows = lakebase.run_query(
        "SELECT saved_job_id FROM saved_jobs WHERE user_id = %s AND job_id = %s", (user_id, job_id)
    )
    if not saved_rows:
        saved = save_job(user_id, job_id)
        saved_job_id = saved["saved_job_id"]
    else:
        saved_job_id = saved_rows[0]["saved_job_id"]

    if kind == "resume_bullet":
        prompt = (
            f"Candidate summary: {profile.get('summary') or profile.get('headline') or 'not specified'}\n"
            f"Skills: {skills_str}\n\n"
            f"Job: {job['title']} at {job.get('company')}\n"
            f"Description: {(job.get('description') or '')[:1000]}\n\n"
            "Write ONE tailored resume bullet (one line, action-verb led, quantify if plausible) "
            "that highlights the candidate's most relevant experience for this specific role."
        )
        text = _complete("You write sharp, specific resume bullets. No fluff.", prompt, max_tokens=120)
        rows = lakebase.run_returning(
            """
            INSERT INTO applications (saved_job_id, resume_bullet)
            VALUES (%s, %s)
            ON CONFLICT (saved_job_id) DO UPDATE
                SET resume_bullet = EXCLUDED.resume_bullet, last_updated_at = now()
            RETURNING *
            """,
            (saved_job_id, text),
        )
    else:
        prompt = (
            f"Candidate headline: {profile.get('headline') or 'not specified'}\n"
            f"Candidate summary: {profile.get('summary') or 'not specified'}\n"
            f"Skills: {skills_str}\n\n"
            f"Job: {job['title']} at {job.get('company')}\n"
            f"Description: {(job.get('description') or '')[:1500]}\n\n"
            "Write a short, specific 3-paragraph cover-letter snippet (not a full formal "
            "letter — just the body) connecting the candidate's real skills to this role's "
            "actual requirements. No generic filler."
        )
        text = _complete("You write concise, specific cover-letter paragraphs. No cliches.", prompt, max_tokens=400)
        rows = lakebase.run_returning(
            """
            INSERT INTO applications (saved_job_id, cover_letter)
            VALUES (%s, %s)
            ON CONFLICT (saved_job_id) DO UPDATE
                SET cover_letter = EXCLUDED.cover_letter, last_updated_at = now()
            RETURNING *
            """,
            (saved_job_id, text),
        )
    return rows[0]


def add_interview_note(user_id: int, job_id: str, note_text: str, follow_up_at: str = None) -> dict:
    saved_rows = lakebase.run_query(
        "SELECT saved_job_id FROM saved_jobs WHERE user_id = %s AND job_id = %s", (user_id, job_id)
    )
    if not saved_rows:
        raise ValueError("That job isn't saved to your pipeline yet — save it first.")
    saved_job_id = saved_rows[0]["saved_job_id"]
    rows = lakebase.run_returning(
        """
        INSERT INTO interview_notes (saved_job_id, note_text, follow_up_at)
        VALUES (%s, %s, %s)
        RETURNING *
        """,
        (saved_job_id, note_text, follow_up_at),
    )
    return rows[0]


def stale_applications(user_id: int, days: int = 7) -> list[dict]:
    """Saved jobs in an active stage ('applied' or 'interviewing') that
    haven't been updated in `days` days — surfaced for follow-up."""
    return lakebase.run_query(
        """
        SELECT sj.saved_job_id, sj.job_id, sj.stage, sj.updated_at,
               j.title, j.company
        FROM saved_jobs sj
        JOIN job_postings j ON j.job_id = sj.job_id
        WHERE sj.user_id = %s
          AND sj.stage IN ('applied', 'interviewing')
          AND sj.updated_at < now() - (%s || ' days')::interval
        ORDER BY sj.updated_at ASC
        """,
        (user_id, days),
    )
