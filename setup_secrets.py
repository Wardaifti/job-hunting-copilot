"""
One-time setup script: stores the Lakebase connection URL for the Job
Hunting Copilot capstone in a Databricks secret scope. Run this once from a
Databricks notebook or terminal — never commit the resulting secret value.

Usage:
    python setup_secrets.py
"""
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace
import getpass

w = WorkspaceClient()

# The "database" scope already exists from Day 1/2/3 — don't recreate it.
# If this is a fresh workspace and it doesn't exist yet, uncomment:
# w.secrets.create_scope(scope="database")

# Using a capstone-specific key so this doesn't overwrite Day 1/2/3's secrets.
w.secrets.put_secret(
    scope="database",
    key="lakebase-url-capstone",
    string_value=getpass.getpass("Paste your capstone Lakebase URL: ")
)

w.secrets.put_acl(
    scope="database",
    principal="users",
    permission=workspace.AclPermission.READ,
)
