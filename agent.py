"""
agent.py — the AI Job Hunting Copilot agent loop.

Uses the Databricks Foundation Model API's OpenAI-compatible chat completions
endpoint with standard function/tool calling (see agent_tools.py for the
actual implementations). The model decides which tool(s) to call based on
the user's message; we execute them against Lakebase and feed the results
back until the model produces a final natural-language reply.
"""

import json

import agent_tools as T

MAX_TOOL_ROUNDS = 4

SYSTEM_PROMPT = """You are the AI Job Hunting Copilot — an assistant that helps a user
search job postings, understand match quality, manage a save/apply/interview
pipeline, draft tailored application material, and track follow-ups.

Rules:
- Always use tools to search, save, update, draft, or read data. Never invent
  job postings, match explanations, or pipeline state — only report what the
  tools return.
- When you search jobs, briefly summarize the top results in your reply
  (title, company, why it's relevant) rather than just calling the tool
  silently.
- When the user asks to save/update/draft something, confirm what you did in
  plain language.
- Keep replies concise and conversational.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": "Semantic search over job postings, ranked by relevance to a natural-language query. Optionally filter by remote-only and minimum salary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {"type": "string", "description": "What kind of role the user is looking for, in their own words."},
                    "remote_only": {"type": "boolean"},
                    "min_salary": {"type": "number"},
                    "limit": {"type": "integer", "default": 8},
                },
                "required": ["query_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_match",
            "description": "Explain why a specific job posting is or isn't a good match for the user's profile and skills.",
            "parameters": {
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_job",
            "description": "Save a job posting to the user's pipeline at a given stage (default 'saved').",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "stage": {"type": "string", "enum": ["saved", "applied", "interviewing", "rejected", "offer"]},
                },
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_stage",
            "description": "Update the pipeline stage of a job the user has already saved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "new_stage": {"type": "string", "enum": ["saved", "applied", "interviewing", "rejected", "offer"]},
                },
                "required": ["job_id", "new_stage"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_pipeline",
            "description": "List the user's saved jobs, optionally filtered by stage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "stage": {"type": "string", "enum": ["saved", "applied", "interviewing", "rejected", "offer"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_application_material",
            "description": "Draft a tailored cover-letter snippet or resume bullet for a specific job posting, saving it to the application record.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "kind": {"type": "string", "enum": ["cover_letter", "resume_bullet"]},
                },
                "required": ["job_id", "kind"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_interview_note",
            "description": "Add an interview/follow-up note to a saved job, optionally with a follow-up date (ISO format).",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "note_text": {"type": "string"},
                    "follow_up_at": {"type": "string", "description": "ISO date/time, optional"},
                },
                "required": ["job_id", "note_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stale_applications",
            "description": "List applications in 'applied' or 'interviewing' stage that haven't been updated in N days (default 7) — for surfacing follow-ups.",
            "parameters": {
                "type": "object",
                "properties": {"days": {"type": "integer", "default": 7}},
            },
        },
    },
]

_TOOL_IMPL = {
    "search_jobs": T.search_jobs,
    "explain_match": T.explain_match,
    "save_job": T.save_job,
    "update_stage": T.update_stage,
    "list_pipeline": T.list_pipeline,
    "draft_application_material": T.draft_application_material,
    "add_interview_note": T.add_interview_note,
    "stale_applications": T.stale_applications,
}


def _to_jsonable(obj):
    """Best-effort conversion so tool results (which may include numpy floats
    from pgvector similarity, datetimes, etc.) serialize cleanly for the LLM."""
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def run_agent(user_id: int, user_message: str, history: list[dict] = None) -> dict:
    """Run one turn of the agent. `history` is prior [{'role', 'content'}, ...]
    (tool calls are not carried across turns, only the final text). Returns
    {'reply': str, 'actions': [tool call summaries]}."""
    client = T._chat_client()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += (history or [])
    messages.append({"role": "user", "content": user_message})

    actions = []

    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.chat.completions.create(
            model=T.CHAT_ENDPOINT,
            max_tokens=800,
            messages=messages,
            tools=TOOLS,
        )
        choice = resp.choices[0]
        msg = choice.message

        if not getattr(msg, "tool_calls", None):
            return {"reply": msg.content, "actions": actions}

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            fn = _TOOL_IMPL.get(name)
            try:
                result = fn(user_id, **args) if fn else {"error": f"unknown tool {name}"}
                actions.append({"tool": name, "args": args, "ok": True})
            except Exception as e:  # noqa: BLE001 — surface the error to the model, not a 500
                result = {"error": str(e)}
                actions.append({"tool": name, "args": args, "ok": False, "error": str(e)})

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(_to_jsonable(result)),
            })

    return {"reply": "I ran into trouble finishing that — could you rephrase or try a simpler request?", "actions": actions}
