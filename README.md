# AI Job Hunting Copilot — Capstone Project

A job-search assistant that ingests live job listings, embeds them for
semantic search, and (coming next) gives an AI agent tools to search,
save, and track applications — all backed by Lakebase.

## Architecture

```
RemoteOK API (no key required)
        │
        ▼
Spark ingestion pipeline (notebooks/ingest_remoteok_jobs.py)
   fetch → clean/dedupe/transform in Spark → write to Lakebase (driver-side, psycopg2)
        │
        ▼
Lakebase: job_postings
        │
        ▼
Embedding pipeline (notebooks/ingest_job_embeddings.py)
   chunk job descriptions → Databricks Foundation Model API (databricks-gte-large-en, 1024-dim)
   → job_embeddings (pgvector)
        │
        ▼
[NEXT] Agent (search/rank/explain/save/draft/track) + Databricks App frontend
```

## Status

- ✅ Lakebase schema (`schema.sql`) — all 8 required tables (`users`,
  `profiles`, `skills`, `job_postings`, `applications`, `saved_jobs`,
  `interview_notes`, `contacts`) plus `job_embeddings` for RAG
- ✅ Spark ingestion pipeline for RemoteOK — fetches, cleans, dedupes, and
  writes live listings into `job_postings`
- ✅ Embedding pipeline — chunks job descriptions and embeds them via
  Databricks' own Foundation Model API (no local model download needed)
- ⏳ Agent (search/rank/save/draft/track tools) — not built yet
- ⏳ Databricks App frontend — not built yet
- ⏳ Adzuna / USAJobs as additional sources — not built yet (RemoteOK only
  so far)

## Setup

### 1. Lakebase

Create a Lakebase Postgres instance, add a native-password role (no special
checkboxes needed — plain read/write is enough), and run:

```sql
\i schema.sql
```

Then grant the app role access (replace `<role>` with your actual role name):

```sql
GRANT USAGE ON SCHEMA public TO "<role>";
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO "<role>";
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO "<role>";
GRANT USAGE ON TYPE vector TO "<role>";

ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT ALL PRIVILEGES ON TABLES TO "<role>";
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT ALL PRIVILEGES ON SEQUENCES TO "<role>";
```

### 2. Secret

Run `setup_secrets.py` from a Databricks notebook/terminal and paste your
Lakebase connection URL (`postgresql://role:password@host/databricks_postgres?sslmode=require`)
when prompted. Stored under secret scope `database`, key `lakebase-url-capstone`.

### 3. Ingest job postings

Run `notebooks/ingest_remoteok_jobs.py` as a Databricks notebook (needs a
Spark-attached cluster). Fetches ~100 live RemoteOK listings and writes
them to `job_postings`.

### 4. Generate embeddings

Run `notebooks/ingest_job_embeddings.py`. Uses the Databricks Foundation
Model API (`databricks-gte-large-en`) — no extra package installs needed
beyond what's already on a Databricks cluster (`databricks-sdk`,
`psycopg2-binary`).

## Files

| File | Purpose |
|---|---|
| `schema.sql` | Lakebase DDL — all tables |
| `lakebase.py` | Connection helper (single `LAKEBASE_URL` secret pattern) |
| `setup_secrets.py` | One-time script to store the Lakebase secret |
| `notebooks/ingest_remoteok_jobs.py` | Spark pipeline: RemoteOK → `job_postings` |
| `notebooks/ingest_job_embeddings.py` | Embedding pipeline: `job_postings` → `job_embeddings` |
