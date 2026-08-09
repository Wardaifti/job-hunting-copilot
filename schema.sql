-- AI Job Hunting Copilot — Lakebase schema
-- Run once against your Lakebase Postgres instance (SQL editor or psql).

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------
-- Core identity + profile
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    user_id     SERIAL PRIMARY KEY,
    email       TEXT UNIQUE NOT NULL,
    full_name   TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS profiles (
    profile_id       SERIAL PRIMARY KEY,
    user_id          INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    headline         TEXT,                -- e.g. "Backend engineer, 4 yrs, Python/Go"
    summary          TEXT,                -- free-text resume summary (gets embedded)
    target_roles     TEXT[],              -- e.g. {'Backend Engineer','Platform Engineer'}
    target_locations TEXT[],              -- e.g. {'Remote','Austin, TX'}
    min_salary       NUMERIC,
    remote_only      BOOLEAN NOT NULL DEFAULT false,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id)
);

CREATE TABLE IF NOT EXISTS skills (
    skill_id    SERIAL PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    skill_name  TEXT NOT NULL,
    years_experience NUMERIC,
    UNIQUE (user_id, skill_name)
);

-- ---------------------------------------------------------------------
-- Job postings (synced from RemoteOK / Adzuna / USAJobs) + embeddings
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS job_postings (
    job_id          TEXT PRIMARY KEY,     -- stable id from the source API
    source          TEXT NOT NULL,        -- 'remoteok' | 'adzuna' | 'usajobs'
    title           TEXT NOT NULL,
    company         TEXT,
    location        TEXT,
    remote          BOOLEAN,
    salary_min      NUMERIC,
    salary_max      NUMERIC,
    description     TEXT,                 -- full text, gets chunked + embedded
    tags            TEXT[],
    url             TEXT,
    posted_at       TIMESTAMPTZ,
    payload         JSONB NOT NULL,       -- raw API response, for provenance
    synced_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_job_postings_source ON job_postings (source);
CREATE INDEX IF NOT EXISTS idx_job_postings_remote ON job_postings (remote);

CREATE TABLE IF NOT EXISTS job_embeddings (
    id           SERIAL PRIMARY KEY,
    job_id       TEXT NOT NULL REFERENCES job_postings(job_id) ON DELETE CASCADE,
    chunk_index  INT NOT NULL,
    chunk_text   TEXT NOT NULL,
    embedding    vector(1024) NOT NULL,   -- Databricks Foundation Model API: databricks-gte-large-en
    model_name   TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (job_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_job_embeddings_cosine
    ON job_embeddings USING hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------------
-- Pipeline: saved jobs, applications, interview notes, contacts
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS saved_jobs (
    saved_job_id SERIAL PRIMARY KEY,
    user_id      INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    job_id       TEXT NOT NULL REFERENCES job_postings(job_id) ON DELETE CASCADE,
    stage        TEXT NOT NULL DEFAULT 'saved'
                 CHECK (stage IN ('saved', 'applied', 'interviewing', 'rejected', 'offer')),
    match_score  NUMERIC,                 -- similarity score at time of save, optional
    saved_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, job_id)
);
CREATE INDEX IF NOT EXISTS idx_saved_jobs_stage ON saved_jobs (stage);

CREATE TABLE IF NOT EXISTS applications (
    application_id  SERIAL PRIMARY KEY,
    saved_job_id    INTEGER NOT NULL REFERENCES saved_jobs(saved_job_id) ON DELETE CASCADE,
    cover_letter    TEXT,                 -- agent-drafted or user-edited
    resume_bullet   TEXT,                 -- agent-drafted tailored bullet
    applied_at      TIMESTAMPTZ,
    last_updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS interview_notes (
    note_id       SERIAL PRIMARY KEY,
    saved_job_id  INTEGER NOT NULL REFERENCES saved_jobs(saved_job_id) ON DELETE CASCADE,
    note_text     TEXT NOT NULL,
    follow_up_at  TIMESTAMPTZ,            -- next follow-up date, used for staleness checks
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS contacts (
    contact_id    SERIAL PRIMARY KEY,
    saved_job_id  INTEGER REFERENCES saved_jobs(saved_job_id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    role          TEXT,                   -- e.g. "Recruiter", "Hiring Manager"
    email         TEXT,
    notes         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
