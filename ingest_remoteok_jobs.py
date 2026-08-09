"""
notebooks/ingest_remoteok_jobs.py

Spark ingestion pipeline for the AI Job Hunting Copilot capstone.

Pulls live listings from the RemoteOK API (no key required), loads them into
a Spark DataFrame for cleaning/transformation (this is the "data pipeline in
Spark" the capstone requires), then writes the cleaned rows into Lakebase's
job_postings table.

IMPORTANT: writes go through psycopg2 from the DRIVER, NOT spark.write.jdbc
(the Day 2 project already established JDBC writes are unreliable against
this Lakebase instance) and NOT executor-side foreachPartition (executors
don't carry Databricks Workspace auth, so WorkspaceClient() fails there).
Spark still does the real transformation work — parsing, cleaning,
deduping, filtering — only the final write step happens on the driver.

Run as a Databricks notebook, or standalone:
    python notebooks/ingest_remoteok_jobs.py
"""

import json
import os
import sys
from datetime import datetime, timezone

import requests
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    StringType,
    StructField,
    StructType,
)

REMOTEOK_URL = "https://remoteok.com/api"

# Allow running this file directly (not just as a notebook) so `import lakebase` works.
# __file__ isn't defined when running inside a Databricks notebook (as opposed
# to a plain .py script), so fall back to the current working directory.
try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _this_dir = os.getcwd()
sys.path.insert(0, os.path.dirname(_this_dir))


def fetch_raw_listings() -> list[dict]:
    """
    Fetch the raw RemoteOK feed. RemoteOK requires a real User-Agent or it
    returns a 403. The very first element of the response is a legal/notice
    object (no 'id' field), not a job — it's filtered out below.
    """
    resp = requests.get(
        REMOTEOK_URL,
        headers={"User-Agent": "job-hunting-copilot (student project; contact: support@example.com)"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    return [item for item in data if isinstance(item, dict) and item.get("id")]


RAW_SCHEMA = StructType([
    StructField("id", StringType(), True),
    StructField("slug", StringType(), True),
    StructField("company", StringType(), True),
    StructField("position", StringType(), True),
    StructField("location", StringType(), True),
    StructField("tags", ArrayType(StringType()), True),
    StructField("description", StringType(), True),
    StructField("url", StringType(), True),
    StructField("date", StringType(), True),
    StructField("salary_min", StringType(), True),   # raw API sometimes sends numbers as strings
    StructField("salary_max", StringType(), True),
])


def build_spark_dataframe(spark: SparkSession, raw_listings: list[dict]):
    """
    Load raw RemoteOK dicts into a Spark DataFrame, keeping only the fields
    we care about (extra/inconsistent fields in the raw API response are
    dropped here rather than fighting schema inference on all of them).
    """
    rows = [
        {
            "id": str(item.get("id")),
            "slug": item.get("slug"),
            "company": item.get("company"),
            "position": item.get("position"),
            "location": item.get("location"),
            "tags": item.get("tags") or [],
            "description": item.get("description"),
            "url": item.get("url"),
            "date": item.get("date"),
            "salary_min": str(item.get("salary_min")) if item.get("salary_min") is not None else None,
            "salary_max": str(item.get("salary_max")) if item.get("salary_max") is not None else None,
        }
        for item in raw_listings
    ]
    return spark.createDataFrame(rows, schema=RAW_SCHEMA)


def clean_and_transform(df):
    """
    The actual Spark transformation work:
      - dedupe by job id (RemoteOK occasionally repeats listings across pages)
      - drop rows with no description (nothing to embed downstream, not useful)
      - normalize location -> remote boolean
      - cast salary strings to numeric, tolerating nulls/garbage
      - parse the ISO date string into a real timestamp
      - trim/clean the free-text position and description fields
    """
    df = df.dropDuplicates(["id"])
    df = df.filter(F.col("description").isNotNull() & (F.length("description") > 0))

    df = df.withColumn(
        "remote",
        F.when(
            F.lower(F.coalesce(F.col("location"), F.lit(""))).contains("remote")
            | F.col("location").isNull(),
            F.lit(True),
        ).otherwise(F.lit(False)),
    )

    df = df.withColumn("salary_min_num", F.col("salary_min").cast("double"))
    df = df.withColumn("salary_max_num", F.col("salary_max").cast("double"))

    df = df.withColumn("posted_at", F.to_timestamp("date"))

    df = df.withColumn("title", F.trim(F.col("position")))
    df = df.withColumn("description_clean", F.trim(F.col("description")))

    return df.select(
        F.col("id").alias("job_id"),
        F.lit("remoteok").alias("source"),
        "title",
        "company",
        "location",
        "remote",
        F.col("salary_min_num").alias("salary_min"),
        F.col("salary_max_num").alias("salary_max"),
        F.col("description_clean").alias("description"),
        "tags",
        "url",
        "posted_at",
    )


def write_rows_to_lakebase(rows: list) -> int:
    """
    Writes a list of Spark Row objects to Lakebase from the DRIVER (not
    executors). This replaces an earlier foreachPartition-based approach:
    executors don't carry Databricks Workspace auth context, so
    WorkspaceClient() (used inside lakebase.py to fetch the connection
    secret) fails with "cannot configure default credentials" when run on
    an executor. Collecting to the driver first avoids that entirely — for
    a dataset this size (~100 rows/run) this is simple and fast; only much
    larger datasets would need genuine distributed writes.
    """
    import lakebase  # noqa: E402  (import here, not at module top, to mirror the actions available on the driver)

    count = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for row in rows:
                cur.execute(
                    """
                    INSERT INTO job_postings (
                        job_id, source, title, company, location, remote,
                        salary_min, salary_max, description, tags, url,
                        posted_at, payload, synced_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (job_id) DO UPDATE
                    SET title = EXCLUDED.title,
                        company = EXCLUDED.company,
                        location = EXCLUDED.location,
                        remote = EXCLUDED.remote,
                        salary_min = EXCLUDED.salary_min,
                        salary_max = EXCLUDED.salary_max,
                        description = EXCLUDED.description,
                        tags = EXCLUDED.tags,
                        url = EXCLUDED.url,
                        posted_at = EXCLUDED.posted_at,
                        payload = EXCLUDED.payload,
                        synced_at = EXCLUDED.synced_at
                    """,
                    (
                        row["job_id"],
                        row["source"],
                        row["title"],
                        row["company"],
                        row["location"],
                        row["remote"],
                        row["salary_min"],
                        row["salary_max"],
                        row["description"],
                        row["tags"],
                        row["url"],
                        row["posted_at"],
                        json.dumps(row.asDict(recursive=True), default=str),
                    ),
                )
                count += 1
        conn.commit()
    return count


def main():
    spark = SparkSession.builder.appName("ingest_remoteok_jobs").getOrCreate()

    print("Fetching RemoteOK listings...")
    raw = fetch_raw_listings()
    print(f"Fetched {len(raw)} raw listings.")

    df = build_spark_dataframe(spark, raw)
    clean_df = clean_and_transform(df)
    count = clean_df.count()
    print(f"{count} listings after cleaning/deduping (dropped listings with no description).")

    print("Collecting cleaned rows to the driver for the Lakebase write...")
    rows = clean_df.collect()
    written = write_rows_to_lakebase(rows)
    print(f"Wrote/updated {written} job_postings rows in Lakebase.")


if __name__ == "__main__":
    main()
