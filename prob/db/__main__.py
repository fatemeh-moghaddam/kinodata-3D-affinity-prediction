"""Command line for the probing database.

    uv run python -m prob.db ingest                  # all sources, skip unchanged
    uv run python -m prob.db ingest current --force  # rebuild data/probing
    uv run python -m prob.db status                  # what is stored
    uv run python -m prob.db issues --severity error # what did not load
    uv run python -m prob.db sql "SELECT target, avg(r2) FROM runs GROUP BY ALL"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

from prob.db.ingestion import ingest
from prob.db.store import connect


def _print(frame: pd.DataFrame) -> None:
    if frame.empty:
        print("(no rows)")
        return
    with pd.option_context("display.max_rows", 200, "display.max_columns", 30,
                           "display.width", 200, "display.max_colwidth", 90):
        print(frame.to_string(index=False))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m prob.db", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db-dir", type=Path, default=None,
                   help="database folder (default: data/probing_db)")
    sub = p.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="build or refresh sources from the files on disk")
    ing.add_argument("sources", nargs="*",
                     help="'current' and/or archive folder names (default: all)")
    ing.add_argument("--force", action="store_true", help="rebuild even if unchanged")
    ing.add_argument("--prune", action="store_true",
                     help="drop stored sources whose directory is gone")
    ing.add_argument("--root", type=Path, default=None,
                     help="live probing directory (default: data/probing)")
    ing.add_argument("--archive-root", type=Path, default=None,
                     help="archive snapshots folder (default: data/probing_archive)")
    ing.add_argument("--no-archives", action="store_true", help="only the live sweep")
    ing.add_argument("--workers", type=int, default=8, help="parallel file readers")

    sub.add_parser("status", help="one line per stored source")

    iss = sub.add_parser("issues", help="files the ingest flagged")
    iss.add_argument("--source", default=None)
    iss.add_argument("--severity", choices=("error", "warning"), default=None)
    iss.add_argument("--kind", default=None)
    iss.add_argument("--summary", action="store_true", help="counts per kind instead of rows")

    sql = sub.add_parser("sql", help="run one query against the views and print it")
    sql.add_argument("query")
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.command == "ingest":
        print(f"Ingesting into {args.db_dir or 'data/probing_db'}")
        results = ingest(args.sources or None, root=args.root, archive_root=args.archive_root,
                         archives=not args.no_archives, db_dir=args.db_dir, force=args.force,
                         prune=args.prune, workers=args.workers)
        if any(r.n_errors for r in results):
            print("Some data was left out. See: python -m prob.db issues --severity error")
        return 0

    con = connect(args.db_dir)
    if args.command == "status":
        _print(con.sql("""
            SELECT source, kind, label, n_runs, n_predictions, n_errors, n_warnings,
                   ingested_at, root
            FROM sources ORDER BY kind, source DESC""").df())
    elif args.command == "issues":
        where, params = [], []
        for col in ("source", "severity", "kind"):
            if getattr(args, col):
                where.append(f"{col} = ?")
                params.append(getattr(args, col))
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        if args.summary:
            sql = (f"SELECT source, severity, kind, count(*) AS n FROM issues{clause} "
                   "GROUP BY ALL ORDER BY source, severity, n DESC")
        else:
            sql = (f"SELECT source, severity, kind, rel_path, detail FROM issues{clause} "
                   "ORDER BY source, severity, kind, rel_path")
        _print(con.execute(sql, params).df())
    elif args.command == "sql":
        _print(con.sql(args.query).df())
    return 0


if __name__ == "__main__":
    sys.exit(main())
