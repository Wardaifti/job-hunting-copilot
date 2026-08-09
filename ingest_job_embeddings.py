"""
notebooks/ingest_job_embeddings.py

Embeds job_postings.description into job_embeddings for semantic retrieval
(RAG). Uses Databricks' own Foundation Model API (databricks-gte-large-en,
1024-dim) via WorkspaceClient().serving_endpoints.query(...) instead of a
locally-loaded sentence-transformers model — this needs only the
databricks-sdk package (already on every Databricks cluster), so there's no
multi-GB torch install and no risk of the kernel hanging/OOM-ing on a small
cluster, which is what happened with the local-model approach.

Run as a Databricks notebook, or standalone:
    python notebooks/ingest_job_embeddings.py
"""

import os
import sys

try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _this_dir = os.getcwd()
sys.path.insert(0, os.path.dirname(_this_dir))

from databricks.sdk import WorkspaceClient
from psycopg2.extras import execute_values

import lakebase

EMBEDDING_ENDPOINT = os.environ.get("EMBEDDING_ENDPOINT", "databricks-gte-large-en")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 100))

_w = WorkspaceClient()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sliding-window character chunking. Job descriptions run longer than
    weather text, so chunking matters more here than it did in Day 2."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += step
    return chunks


def embed_text(text: str) -> list[float]:
    """Call the Databricks Foundation Model API embedding endpoint for one chunk."""
    response = _w.serving_endpoints.query(name=EMBEDDING_ENDPOINT, input=text)
    return response.data[0].embedding


def fetch_unembedded_jobs() -> list[dict]:
    """job_postings rows with no rows yet in job_embeddings."""
    return lakebase.run_query(
        """
        SELECT j.job_id, j.description
        FROM job_postings j
        LEFT JOIN job_embeddings e ON e.job_id = j.job_id
        WHERE e.id IS NULL AND j.description IS NOT NULL AND j.description <> ''
        """
    )


def main():
    jobs = fetch_unembedded_jobs()
    print(f"Found {len(jobs)} job posting(s) to embed.")
    if not jobs:
        return

    rows_to_insert = []
    for job in jobs:
        chunks = chunk_text(job["description"])
        for idx, chunk in enumerate(chunks):
            embedding = embed_text(chunk)
            rows_to_insert.append((
                job["job_id"],
                idx,
                chunk,
                embedding,
                EMBEDDING_ENDPOINT,
            ))

    if not rows_to_insert:
        print("No chunks produced.")
        return

    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO job_embeddings
                    (job_id, chunk_index, chunk_text, embedding, model_name)
                VALUES %s
                ON CONFLICT (job_id, chunk_index) DO UPDATE
                SET chunk_text = EXCLUDED.chunk_text,
                    embedding = EXCLUDED.embedding,
                    model_name = EXCLUDED.model_name
                """,
                rows_to_insert,
                template="(%s, %s, %s, %s::vector, %s)",
            )
        conn.commit()

    print(f"Wrote {len(rows_to_insert)} embedding row(s) for {len(jobs)} job posting(s).")


if __name__ == "__main__":
    main()
