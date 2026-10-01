"""The probing results as a queryable database (DuckDB over Parquet).

    uv run python -m prob.db ingest        # after copying results from the cluster

    from prob.db import connect, load_runs, load_predictions

See prob/db/README.md for the schema and how to extend it.
"""
from prob.db.ingestion import (
    IngestResult,
    Source,
    default_db_dir,
    discover_sources,
    ingest,
    ingest_source,
)
from prob.db.schema import ISSUE_KINDS, SCHEMA_VERSION, TABLES, TABLES_BY_NAME
from prob.db.store import connect, load_predictions, load_runs

__all__ = [
    "ingest", "ingest_source", "discover_sources", "default_db_dir",
    "IngestResult", "Source",
    "connect", "load_runs", "load_predictions",
    "TABLES", "TABLES_BY_NAME", "SCHEMA_VERSION", "ISSUE_KINDS",
]
