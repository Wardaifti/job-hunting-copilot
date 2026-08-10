# 🧭 AI Job Hunting Copilot

Capstone project — DataExpert.io AI Data Engineering Bootcamp

Live app: https://job-hunting-copilot-7474644487245197.aws.databricksapps.com 
Repo: https://github.com/Wardaifti/job-hunting-copilot

An AI-powered job search assistant. Users describe their skills and target roles, then ask an agent to find matching live job postings, explain why a posting is (or isn't) a good fit, save it to a pipeline, draft tailored application material, and track follow-ups — all backed by Lakebase.

Architecture
RemoteOK API (live job listings, no key required)
        │
        ▼
Spark ingestion pipeline (notebooks/ingest_remoteok_jobs.py)
   fetch → clean/dedupe/transform in Spark → write to Lakebase (driver-side, psycopg2)
        │
        ▼
Lakebase (Postgres): job_postings, users, profiles, skills,
                      saved_jobs, applications, interview_notes, contacts
        │
        ├── Embedding pipeline (notebooks/ingest_job_embeddings.py)
        │      chunk job descriptions → Databricks Foundation Model API
        │      (databricks-gte-large-en, 1024-dim) → job_embeddings (pgvector)
        │
        ▼
AI Agent (agent.py + agent_tools.py)
   OpenAI-style tool-calling loop against the Databricks Foundation Model API
   (databricks-meta-llama-3-3-70b-instruct) — 8 tools covering search,
   explanation, and pipeline writes
        │
        ▼
Databricks App frontend (app.py, Flask)
   Chat tab · Pipeline tab · Profile tab
Why Lakebase (not a traditional analytics table)

Lakebase is a real transactional Postgres database — it supports fast single-row reads/writes, foreign keys, and pgvector similarity search all in the same store. That's what lets one app both serve live chat requests (save a job, log a note, update a stage — all instant, consistent writes) and do semantic retrieval over job descriptions, instead of needing a separate OLTP database plus a separate vector database bolted on.

Data pipeline (Spark) + third-party API

notebooks/ingest_remoteok_jobs.py pulls live listings from the RemoteOK API (no key required), loads them into a Spark DataFrame, and does the real transformation work there: dedupes by job id, drops listings with no description, derives a remote boolean from the location string, casts salary fields, parses posting dates. The final write to Lakebase happens via psycopg2 from the driver (not spark.write.jdbc, which is unreliable against this Lakebase instance, and not executor-side foreachPartition, since executors don't carry Databricks Workspace auth needed to fetch the Lakebase credential).

# Unstructured data processing (RAG)

notebooks/ingest_job_embeddings.py chunks each job posting's free-text description (800-char sliding window, 100-char overlap) and embeds each chunk via the Databricks Foundation Model API (databricks-gte-large-en, 1024 dimensions) — no local model download needed, so no risk of a cluster OOM/hang from loading a multi-GB model. Chunks are written to job_embeddings with an HNSW index (vector_cosine_ops) for fast cosine-similarity search.

#   AI agent — tools

The agent (agent.py) runs an OpenAI-style tool-calling loop (up to 4 rounds per turn) against databricks-meta-llama-3-3-70b-instruct. Its tools (implemented in agent_tools.py) are the agent's actual "hands" — plain Python functions that read from and write to Lakebase:

Tool	Type	What it does
search_jobs	read	Semantic search over job_embeddings (cosine distance), optional remote/salary filters
explain_match	read	LLM explains fit between the user's profile/skills and a specific posting — not a canned response
save_job	write	Saves a posting to the user's pipeline at a stage (saved/applied/interviewing/rejected/offer)
update_stage	write	Moves an already-saved job to a new pipeline stage
list_pipeline	read	Lists the user's saved jobs, optionally filtered by stage
draft_application_material	write	LLM drafts a tailored cover-letter snippet or resume bullet from the user's real profile + the specific job description, saved onto the application record
add_interview_note	write	Logs a free-text interview/follow-up note against a saved job
stale_applications	read	Surfaces applied/interviewing jobs untouched for N days — prompts the user to follow up

The system prompt instructs the agent to always use tools (never invent job data or match explanations), summarize search results in plain language, and confirm writes explicitly.

# Frontend (Databricks App)

Flask app (app.py) with three tabs, styled with a custom ticket-stub/paper-and-teal design (static/style.css) rather than default Bootstrap:

Chat — talk to the agent directly; each reply shows which tools fired
Pipeline — saved jobs as cards, grouped/filterable by stage, inline stage-change dropdown and note-adding form, stale-application banner
Profile — headline, summary, target roles/locations, min salary, remote-only toggle, and a dynamic skills list — this is what search ranking, match explanations, and drafted materials all run against

Uses a single demo user (DEMO_USER_EMAIL env var) — real login/multi-tenant auth was out of scope for this capstone.

# Lakebase schema

9 tables (schema.sql): users, profiles, skills, job_postings, job_embeddings, saved_jobs, applications, interview_notes, contacts. saved_jobs is the pipeline's spine — applications and interview_notes both hang off saved_job_id.

# Setup
1. Lakebase

Create a Lakebase instance, add a native-password role, run schema.sql, then grant the role access:

sql
GRANT USAGE ON SCHEMA public TO "<role>";
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO "<role>";
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO "<role>";
2. Ingest data

Run notebooks/ingest_remoteok_jobs.py (Spark-attached cluster), then notebooks/ingest_job_embeddings.py.

3. Deploy the app

Deploy app.py as a Databricks App. Attach a Database resource in the Apps UI (Resources → Add resource → Database → your Lakebase instance) — lakebase.py uses this to auto-fetch a short-lived Postgres credential for the app's own service principal via w.postgres.generate_database_credential(...), which sidesteps a Free Edition limitation where app service principals can't be granted secret scope ACLs. Then grant that service principal the same table access as step 1 (its identity is the app's DATABRICKS_CLIENT_ID, visible in the app's Environment tab).

# Reflection

What was the most difficult part? Getting Lakebase authentication right for a deployed app on Free Edition — the secret-scope-ACL pattern that worked for notebooks doesn't work for app service principals there, so lakebase.py had to fall back to exchanging the app's own identity for a short-lived credential via its attached Database resource instead.

How is Lakebase different from storing this data in a traditional analytics table? See "Why Lakebase" above — transactional reads/writes plus pgvector similarity search in one store, versus a batch-loaded, scan-optimized analytics table that can't serve either well.

What feature would you add next? A scheduled Databricks Workflow to re-run the sync + embedding pipeline automatically, and adding Adzuna as a second job source — the RemoteOK feed alone skews web-dev-heavy, which limits match quality for other role types (e.g. data science).

# Known limitations
Match relevance depends on what's in the ~100-listing RemoteOK snapshot at sync time; underrepresented role types get weaker "best available" matches. This is a data-coverage limitation, not a retrieval bug.
Single demo-user model — no real auth (out of scope for this capstone).
Only RemoteOK is wired up; Adzuna/USAJobs were not integrated.
Files
File	Purpose
schema.sql	Lakebase DDL
lakebase.py	Connection helper (resource-credential + secret-scope fallback)
setup_secrets.py	One-time notebook-auth secret setup
notebooks/ingest_remoteok_jobs.py	Spark pipeline: RemoteOK → job_postings
notebooks/ingest_job_embeddings.py	Embedding pipeline: job_postings → job_embeddings
agent_tools.py	Agent's tools — reads/writes against Lakebase + LLM calls
agent.py	Tool-calling loop + system prompt
app.py	Flask app / Databricks App entrypoint
templates/, static/	Frontend (chat, pipeline, profile)
app.yaml, requirements.txt	Databricks Apps deployment config
