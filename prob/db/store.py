"""Query the probing database.

`connect()` returns an in-memory DuckDB connection with one view per table,
reading the Parquet files of every ingested source. Nothing is loaded until a
query runs, and the files are never locked, so any number of notebooks can
query while `ingest` rebuilds a source.

    from prob.db import connect, load_runs, load_predictions

    con = connect()
    con.sql("SELECT source, count(*) FROM runs GROUP BY ALL").df()

    runs = load_runs(source="20260929_160507", target="affinity", layer=[0, 3])
    preds = load_predictions(runs.run_id[0], source="20260929_160507")

Views (all carry `source` as their first column):

    runs         the stored table, plus <metric>_ckpt_lower/_upper (mean -/+ sd)
                 and absolute paths: exp_root, predictions_path, summary_path,
                 figures_dir -- the same columns find_probe_runs returns
    predictions  folds  figures (+ path)  issues  sources (+ root_path)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import duckdb
import pandas as pd

from prob.db.ingestion import PARTITION_PREFIX, _sql_str, default_db_dir
from prob.db.schema import TABLES, Table
from prob.paths_and_io import (
    EXP_DIR_ARTIFACTS,
    EXP_DIR_FIGURES,
    EXP_DIR_REPORTS,
    PRED_SUFFIX,
    SUMMARY_SUFFIX,
)

#: Filters accepted by `load_runs`, the same as find_probe_runs takes.
FACTOR_FILTERS = ("gnn_model_type", "rmsd_threshold", "split_type", "target",
                  "prob_model", "layer")

_RUN_ORDER = ("source", "gnn_model_type", "rmsd_threshold", "split_type", "target",
              "is_baseline", "prob_model", "layer")


def _base_view(con: duckdb.DuckDBPyConnection, table: Table, db_dir: Path) -> None:
    """`_<table>`: the union of every source's file, or an empty typed table."""
    if any(db_dir.glob(f"{PARTITION_PREFIX}*/{table.name}.parquet")):
        pattern = db_dir / f"{PARTITION_PREFIX}*" / f"{table.name}.parquet"
        con.execute(
            f'CREATE VIEW "_{table.name}" AS SELECT * FROM read_parquet({_sql_str(pattern)}, '
            "hive_partitioning = true, hive_types = {'source': VARCHAR}, union_by_name = true)")
    else:
        empty = Table(f"_{table.name}", table.doc, table.columns)
        con.execute(empty.ddl({"source": "VARCHAR"}, temp=True))


def connect(db_dir: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    """A connection with the views listed in the module docstring."""
    db_dir = Path(db_dir) if db_dir is not None else default_db_dir()
    data_dir = db_dir.resolve().parent   # roots are stored relative to data/
    con = duckdb.connect()
    for table in TABLES:
        _base_view(con, table, db_dir)

    con.execute(f"""
        CREATE VIEW sources AS
        SELECT source, * EXCLUDE (source),
               CASE WHEN starts_with(root, '/') THEN root
                    ELSE {_sql_str(data_dir)} || '/' || root END AS root_path
        FROM _sources""")

    cols = [r[0] for r in con.execute("DESCRIBE _runs").fetchall()]
    bounds = [f'"{m}_ckpt_mean" - "{m}_ckpt_sd" AS "{m}_ckpt_lower", '
              f'"{m}_ckpt_mean" + "{m}_ckpt_sd" AS "{m}_ckpt_upper"'
              for m in (c[: -len("_ckpt_mean")] for c in cols if c.endswith("_ckpt_mean"))
              if f"{m}_ckpt_sd" in cols]
    con.execute(f"""
        CREATE VIEW runs AS
        SELECT r.source, r.* EXCLUDE (source),
               {''.join(b + ', ' for b in bounds)}
               s.root_path || '/' || r.rel_dir AS exp_root,
               exp_root || '/{EXP_DIR_ARTIFACTS}/' || r.prob_model || '{PRED_SUFFIX}' AS predictions_path,
               exp_root || '/{EXP_DIR_REPORTS}/' || r.prob_model || '{SUMMARY_SUFFIX}' AS summary_path,
               exp_root || '/{EXP_DIR_FIGURES}' AS figures_dir
        FROM _runs r LEFT JOIN sources s USING (source)""")
    con.execute("""
        CREATE VIEW figures AS
        SELECT f.source, f.* EXCLUDE (source), s.root_path || '/' || f.rel_path AS path
        FROM _figures f LEFT JOIN sources s USING (source)""")
    for name in ("predictions", "folds", "issues"):
        con.execute(f'CREATE VIEW {name} AS SELECT source, * EXCLUDE (source) FROM "_{name}"')
    return con


def _as_list(value: Any) -> list | None:
    if value is None:
        return None
    if isinstance(value, (str, int, float)):
        return [value]
    return list(value)


def load_runs(*, source: str | Iterable[str] | None = None,
              include_baselines: bool = True,
              con: duckdb.DuckDBPyConnection | None = None,
              db_dir: str | Path | None = None,
              **filters: Any) -> pd.DataFrame:
    """Runs as a DataFrame, filtered like find_probe_runs (scalar or list per factor).

    The frame has find_probe_runs' columns and attach_run_metrics' metric
    columns, plus `source`, so it drops into existing plotting code.
    """
    unknown = set(filters) - set(FACTOR_FILTERS)
    if unknown:
        raise TypeError(f"unknown filter(s) {sorted(unknown)}; use {FACTOR_FILTERS}")
    con = con or connect(db_dir)
    where, params = [], []
    for column, wanted in {"source": source, **filters}.items():
        values = _as_list(wanted)
        if values is None:
            continue
        if column == "rmsd_threshold":
            values = [float(v) for v in values]
        where.append(f"list_contains(?, {column})")
        params.append(values)
    if not include_baselines:
        where.append("NOT is_baseline")
    sql = (f"SELECT * FROM runs{' WHERE ' + ' AND '.join(where) if where else ''} "
           f"ORDER BY {', '.join(_RUN_ORDER)}")
    runs = con.execute(sql, params).df()
    runs["layer"] = runs["layer"].astype("Int64")
    return runs


def load_predictions(run_id: str | Iterable[str], *, source: str,
                     con: duckdb.DuckDBPyConnection | None = None,
                     db_dir: str | Path | None = None) -> pd.DataFrame:
    """Predictions of one run (or several, told apart by `run_id`), in file order."""
    con = con or connect(db_dir)
    ids = _as_list(run_id)
    return con.execute(
        "SELECT run_id, row, ident, fold, y_true, y_pred FROM predictions "
        "WHERE source = ? AND list_contains(?, run_id) ORDER BY run_id, row",
        [source, ids]).df()
