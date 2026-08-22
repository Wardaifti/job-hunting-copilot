# 🧭 AI Job Hunting Copilot

**Capstone project — DataExpert.io AI Data Engineering Bootcamp**

An AI-powered job search assistant. Describe your skills and target roles,
then ask an agent to find matching live job postings, explain why a posting
is (or isn't) a good fit, save it to a pipeline, draft tailored application
material, and track follow-ups — all backed by Lakebase.

---

## Architecture

```
RemoteOK API              Adzuna API
(no key required)      (free key, real salary data)
        │                       │
        └───────────┬───────────┘
                     ▼
        Spark ingestion pipeline
   fetch → clean, dedupe, transform → write to Lakebase (driver-side, psycopg2)
                     │
                     ▼
   Lakebase (Postgres): job_postings, job_embeddings, users, profiles,
                         skills, saved_jobs, applications, interview_notes
                     │
                     ├── Embedding pipeline: chunk job descriptions →
                     │   Databricks Foundation Model API (databricks-gte-large-en,
                     │   1024-dim) → job_embeddings (pgvector, HNSW index)
                     │
                     ▼
        AI Agent (agent.py + agent_tools.py)
   Tool-calling loop against the Databricks Foundation Model API
   (databricks-meta-llama-3-3-70b-instruct) — 8 tools covering
   search, explanation, and pipeline writes
                     │
                     ▼
        Databricks App frontend (app.py, Flask)
           Chat tab · Pipeline tab · Profile tab
```

## Why Lakebase (not a traditional analytics table)

Lakebase is a real transactional Postgres database — fast single-row
reads/writes, foreign keys, and `pgvector` similarity search all in the
same store. That's what lets one app both serve live chat requests (save a
job, log a note, update a stage — instant, consistent writes) *and* do
semantic retrieval over job descriptions, instead of needing a separate
OLTP database plus a separate vector database bolted on.

## Data sources

- **[RemoteOK](https://remoteok.com/api)** — no key required. The base feed
  only ever returns the ~100 most recent listings across *all* categories,
  so the pipeline also queries several tag-filtered feeds (`python`,
  `data-science`, `backend`, `devops`, etc.) and merges/dedupes the results,
  widening the pool to several hundred candidate listings before filtering.
- **[Adzuna](https://developer.adzuna.com/)** — free API key. Real listings
  across countries with actual salary data, queried across multiple job
  titles and two countries (US, UK) to further diversify the pool beyond
  RemoteOK's remote-only, startup-skewed feed.

Both sources write into the same `job_postings` table (`source` column
distinguishes them), so search and matching work seamlessly across both.

## Data pipeline (Spark)

`notebooks/ingest_remoteok_jobs.py` loads raw listings into a Spark
DataFrame and does the real transformation work: dedupes by job id, drops
listings with no description, derives a `remote` boolean from the location
string, casts salary fields, and filters out non-tech spam/ad listings
(RemoteOK's free feed is heavily polluted with junk — filtering requires
either a real disclosed salary or 2+ recognizable tech tags plus a
substantive description). The final write to Lakebase happens via
`psycopg2` from the driver — not `spark.write.jdbc` (unreliable against
this Lakebase instance) and not executor-side `foreachPartition`
(executors don't carry the Databricks Workspace auth needed to fetch the
Lakebase credential).

`notebooks/ingest_adzuna_jobs.py` is a lighter, plain-Python companion
pipeline for the second source (the Spark requirement is already satisfied
by the RemoteOK pipeline).

## Unstructured data processing (RAG)

`notebooks/ingest_job_embeddings.py` chunks each posting's free-text
`description` (800-char sliding window, 100-char overlap) and embeds each
chunk via the Databricks Foundation Model API (`databricks-gte-large-en`,
1024 dimensions) — no local model download, so no risk of a cluster
OOM/hang from loading a multi-GB model. Chunks are written to
`job_embeddings` with an HNSW index (`vector_cosine_ops`) for fast
cosine-similarity search.

## AI agent — tools

The agent (`agent.py`) runs a tool-calling loop (up to 4 rounds per turn)
against `databricks-meta-llama-3-3-70b-instruct`, called via a raw REST
request to the Foundation Model API's `/invocations` endpoint
(`agent_tools.fm_chat`) rather than an OpenAI SDK client object — the
`get_open_ai_client()` convenience method isn't available on every
`databricks-sdk` version, so authentication is handled directly via
`WorkspaceClient().config.authenticate()`, which resolves correctly under
both PAT auth (notebook/local) and OAuth service-principal auth (deployed
Databricks App).

Its tools (`agent_tools.py`) are the agent's actual "hands" — plain Python
functions that read from and write to Lakebase:

| Tool | Type | What it does |
|---|---|---|
| `search_jobs` | read | Semantic search over `job_embeddings` (cosine distance), optional remote/salary filters |
| `explain_match` | read | LLM explains fit between the user's profile/skills and a specific posting — not a canned response |
| `save_job` | **write** | Saves a posting to the pipeline at a stage (`saved`/`applied`/`interviewing`/`rejected`/`offer`) |
| `update_stage` | **write** | Moves an already-saved job to a new pipeline stage |
| `list_pipeline` | read | Lists saved jobs, optionally filtered by stage |
| `draft_application_material` | **write** | LLM drafts a tailored cover-letter snippet or resume bullet from the real profile + job description, saved onto the application record |
| `add_interview_note` | **write** | Logs a free-text interview/follow-up note against a saved job |
| `stale_applications` | read | Surfaces `applied`/`interviewing` jobs untouched for N days |

The system prompt instructs the agent to always use tools (never invent job
data or match explanations), summarize search results in plain language
without exposing raw similarity scores, and confirm writes explicitly.

## Frontend (Databricks App)

Flask app (`app.py`) with three tabs, styled with a custom
ticket-stub/paper-and-teal design (`static/style.css`):

- **Chat** — talk to the agent directly; each reply shows which tools fired
- **Pipeline** — saved jobs as cards with a live stats bar (total + per-stage
  counts), filterable by stage, inline stage-change dropdown, note-adding
  form, drafted cover-letter/resume-bullet display, and a stale-application
  banner
- **Profile** — headline, summary, target roles/locations, min salary,
  remote-only toggle, and a dynamic skills list — what search ranking,
  match explanations, and drafted materials all run against

Uses a single demo user (`DEMO_USER_EMAIL` env var) — real
login/multi-tenant auth was out of scope for this capstone.

## Lakebase schema

9 tables (`schema.sql`): `users`, `profiles`, `skills`, `job_postings`,
`job_embeddings`, `saved_jobs`, `applications`, `interview_notes`,
`contacts`. `saved_jobs` is the pipeline's spine — `applications` and
`interview_notes` both hang off `saved_job_id`.

## Setup

### 1. Lakebase
Create a Lakebase instance, add a native-password role, run `schema.sql`,
then grant the role access:
```sql
GRANT USAGE ON SCHEMA public TO "<role>";
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO "<role>";
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO "<role>";
```

### 2. Secrets
Run `setup_secrets.py` (Lakebase URL, for notebook/local use) and
`setup_adzuna_secrets.py` (Adzuna `app_id`/`app_key`).

### 3. Ingest data
Run `notebooks/ingest_remoteok_jobs.py` (Spark-attached cluster), then
`notebooks/ingest_adzuna_jobs.py`, then `notebooks/ingest_job_embeddings.py`.

### 4. Deploy the app
Deploy `app.py` as a Databricks App. **Attach a Database resource** in the
Apps UI (Resources → Add resource → Database → your Lakebase instance) —
`lakebase.py` uses this to auto-fetch a short-lived Postgres credential for
the app's own service principal via
`w.postgres.generate_database_credential(...)`, which sidesteps a Free
Edition limitation where app service principals can't be granted secret
scope ACLs. Then grant that service principal the same table access as
step 1 (its identity is the app's `DATABRICKS_CLIENT_ID`, visible in the
app's Environment tab).

## Reflection

**What was the most difficult part?**
Two things, both authentication-shaped: getting Lakebase credentials right
for a *deployed* app on Free Edition (secret-scope ACLs don't work for app
service principals there — `lakebase.py` falls back to exchanging the
app's own identity for a short-lived credential via its attached Database
resource instead), and getting the agent's LLM calls right when the
`databricks-sdk`'s `get_open_ai_client()` helper wasn't available — solved
by calling the Foundation Model API's REST endpoint directly and
authenticating via `WorkspaceClient().config.authenticate()`.

**How is Lakebase different from storing this data in a traditional
analytics table?**
See "Why Lakebase" above — transactional reads/writes plus `pgvector`
similarity search in one store, versus a batch-loaded, scan-optimized
analytics table that can't serve either well.

**What feature would you add next?**
A scheduled Databricks Workflow to re-run the sync + embedding pipelines
automatically instead of triggering them manually, and a richer
`predict`-style tool that factors in things like commute/timezone overlap
for non-remote roles.

## Known limitations

- Match relevance still depends on what's actually in the synced pool at
  ingestion time; very niche role types can get weaker "best available"
  matches even with two combined sources.
- Single demo-user model — no real auth (out of scope for this capstone).
- No Databricks Workflow schedule yet — both ingestion pipelines and the
  embedding pipeline are triggered manually.

## Files

| File | Purpose |
|---|---|
| `schema.sql` | Lakebase DDL |
| `lakebase.py` | Connection helper (resource-credential + secret-scope fallback) |
| `setup_secrets.py`, `setup_adzuna_secrets.py` | One-time secret setup scripts |
| `notebooks/ingest_remoteok_jobs.py` | Spark pipeline: RemoteOK → `job_postings` |
| `notebooks/ingest_adzuna_jobs.py` | Plain-Python pipeline: Adzuna → `job_postings` |
| `notebooks/ingest_job_embeddings.py` | Embedding pipeline: `job_postings` → `job_embeddings` |
| `agent_tools.py` | Agent's tools — reads/writes against Lakebase + Foundation Model API calls |
| `agent.py` | Tool-calling loop + system prompt |
| `app.py` | Flask app / Databricks App entrypoint |
| `templates/`, `static/` | Frontend (chat, pipeline, profile) |
| `app.yaml`, `requirements.txt` | Databricks Apps deployment config |
