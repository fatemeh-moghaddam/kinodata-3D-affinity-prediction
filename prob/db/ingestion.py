"""Build the probing database from the files the probe runs wrote.

The files under data/probing (and each archive snapshot) stay the source of
truth: cluster jobs keep writing them in parallel, and nothing here writes back.
`ingest` reads one source at a time and replaces that source's folder under
data/probing_db as a whole, so a source is either fully its old version or
fully its new one.

A source whose input files are unchanged since its last ingest (same paths,
sizes and mtimes, same SCHEMA_VERSION) is skipped, so re-running `ingest` after
copying new results from the cluster only rebuilds what changed.

    uv run python -m prob.db ingest            # every source, skipping unchanged ones
    uv run python -m prob.db ingest current    # just data/probing
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import duckdb
import numpy as np
import pandas as pd

from prob.db.schema import (
    FIGURES,
    FOLDS,
    ISSUE_KINDS,
    ISSUES,
    PREDICTIONS,
    RUNS,
    SCHEMA_VERSION,
    SOURCES,
    TABLES,
    Table,
)
from prob.paths_and_io import (
    EXP_DIR_ARTIFACTS,
    EXP_DIR_FIGURES,
    EXP_DIR_REPORTS,
    PRED_SUFFIX,
    SUMMARY_SUFFIX,
    _decode_pred_path,
    get_data_dir,
)

DB_DIRNAME = "probing_db"
ARCHIVE_DIRNAME = "probing_archive"
ARCHIVE_STAMP = "%Y%m%d_%H%M%S"       # the name archive_probe_results.sh gives a snapshot
CURRENT = "current"
BEST_PARAMS_SUFFIX = "_best_params.json"
BASELINE_TAG = "shuffled_ident"
PARTITION_PREFIX = "source="


# ─────────────────────────────────────────────────────────────
# Sources
# ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Source:
    """One probing directory: the live sweep or one archive snapshot."""

    name: str      # partition name; "current" or the archive folder name
    kind: str      # "current" | "archive"
    root: Path
    label: str


def default_db_dir() -> Path:
    return get_data_dir(prob=False) / DB_DIRNAME


def _archive_label(name: str) -> str:
    """'20260929_160507' -> '2026-09-29 16:05:07'; any other name as-is."""
    try:
        return datetime.strptime(name, ARCHIVE_STAMP).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return name


def discover_sources(root: str | Path | None = None, *,
                     archive_root: str | Path | None = None,
                     archives: bool = True) -> list[Source]:
    """The live sweep plus every archive snapshot, newest archive first.

    Folders in the archive root starting with "_" or "." are not snapshots
    (_old_data, _chembl data, ...) and are left out.
    """
    root = Path(root) if root is not None else get_data_dir(prob=True)
    sources = [Source(CURRENT, "current", root, "Current")]
    if archives:
        a_root = (Path(archive_root) if archive_root is not None
                  else get_data_dir(prob=False) / ARCHIVE_DIRNAME)
        if a_root.is_dir():
            snaps = sorted((p for p in a_root.iterdir()
                            if p.is_dir() and not p.name.startswith(("_", "."))),
                           key=lambda p: p.name, reverse=True)
            sources += [Source(_safe_name(p.name), "archive", p, _archive_label(p.name))
                        for p in snaps]
    return sources


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text)).strip("-") or "x"


# ─────────────────────────────────────────────────────────────
# Finding the runs in a source
# ─────────────────────────────────────────────────────────────


@dataclass
class RunFiles:
    """The files that make up one run, before anything is read."""

    exp_root: Path
    probe: str
    summary: Path | None = None
    predictions: Path | None = None
    best_params: Path | None = None
    figures: list[Path] = field(default_factory=list)

    def all_files(self) -> list[Path]:
        return [p for p in (self.summary, self.predictions, self.best_params) if p] + self.figures


def scan_source(root: Path) -> list[RunFiles]:
    """Every run under `root`, found by its summary JSON or its predictions CSV.

    A run with only one of the two is still a run (the missing half becomes an
    issue), which is why discovery is not anchored on the CSV alone.
    """
    found: dict[tuple[Path, str], RunFiles] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        here = Path(dirpath)
        if here.name == EXP_DIR_ARTIFACTS:
            suffix, slot = PRED_SUFFIX, "predictions"
        elif here.name == EXP_DIR_REPORTS:
            suffix, slot = SUMMARY_SUFFIX, "summary"
        else:
            continue
        for name in sorted(filenames):
            if name.endswith(suffix) and len(name) > len(suffix):
                probe = name[: -len(suffix)]
                run = found.setdefault((here.parent, probe), RunFiles(here.parent, probe))
                setattr(run, slot, here / name)

    for run in found.values():
        bp = run.exp_root / EXP_DIR_REPORTS / f"{run.probe}{BEST_PARAMS_SUFFIX}"
        run.best_params = bp if bp.is_file() else None
        run.figures = sorted((run.exp_root / EXP_DIR_FIGURES).glob(f"{run.probe}_*.png"))
    return [found[k] for k in sorted(found, key=lambda k: (str(k[0]), k[1]))]


def fingerprint(runs: Iterable[RunFiles], root: Path) -> str:
    """Hash of every input file's path, size and mtime: changes iff an input did."""
    h = hashlib.sha1()
    for run in runs:
        for p in run.all_files():
            st = p.stat()
            h.update(f"{p.relative_to(root)}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────
# Reading one run
# ─────────────────────────────────────────────────────────────


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def _clean_json(value: Any) -> Any:
    """NaN/inf -> None, recursively: Python's json writes NaN, which is not JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    return value


def _column_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def flatten_summary(summary: dict) -> dict[str, Any]:
    """The summary JSON's numbers as flat columns.

    Names match `attach_run_metrics` (r2, r2_ci_lower, r2_ckpt_mean, ...). Any
    numeric field is kept, declared or not, so a new metric needs no code here.
    """
    out: dict[str, Any] = {}
    for k, v in (summary.get("metrics_on_unseen_data") or {}).items():
        if _is_number(v):
            out[_column_name(k)] = float(v)
    for name, test in (summary.get("statistical_tests") or {}).items():
        if not isinstance(test, dict):
            continue
        for k, v in test.items():
            if k == "confidence" and _is_number(v):
                out["ci_confidence"] = float(v)
            elif k == "point_estimate" and _is_number(v):
                # Older runs report pearson only here, not in metrics_on_unseen_data.
                metric = test.get("metric")
                if isinstance(metric, str) and _column_name(metric) not in out:
                    out[_column_name(metric)] = float(v)
            elif _is_number(v):
                out[_column_name(f"{name}_{k}")] = float(v)
    for name, stat in (summary.get("across_checkpoints") or {}).items():
        if isinstance(stat, dict):
            for k in ("mean", "sd"):
                if _is_number(stat.get(k)):
                    out[_column_name(f"{name}_ckpt_{k}")] = float(stat[k])
    for k in ("n_samples", "n_train_samples", "n_test_samples", "n_features", "n_splits_cv"):
        if _is_number(summary.get(k)):
            out[k] = int(summary[k])
    for k in ("probe_mode", "best_params_from"):
        if summary.get(k) is not None:
            out[k] = str(summary[k])
    # The summary's run_id names its run manifest; the table's own run_id is a different key.
    if summary.get("run_id") is not None:
        out["manifest_id"] = str(summary["run_id"])
    return out


def flatten_folds(summary: dict) -> list[dict[str, Any]]:
    rows = []
    for entry in summary.get("per_fold") or []:
        if not isinstance(entry, dict) or not _is_number(entry.get("fold")):
            continue
        row: dict[str, Any] = {}
        for k, v in entry.items():
            if k == "best_params":
                row[k] = json.dumps(_clean_json(v))
            elif k == "params_source":
                row[k] = None if v is None else str(v)
            elif _is_number(v):
                row[_column_name(k)] = v
        row["fold"] = int(entry["fold"])
        rows.append(row)
    return rows


def _read_json(path: Path) -> tuple[Any, str | None]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return None, f"cannot read: {exc}"
    if not text.strip():
        return None, "file is empty"
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"


_PRED_COLUMNS = ("ident", "fold", "y_true", "y_pred")


def read_predictions(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    """A validated predictions frame, or (None, why it was rejected).

    All or nothing: a file with any non-numeric value is rejected whole, since
    a byte-damaged CSV cannot be trusted in the rows that happen to parse.
    """
    try:
        num = pd.read_csv(path, encoding_errors="replace")
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        return None, f"cannot parse: {str(exc).splitlines()[0]}"
    if not isinstance(num.index, pd.RangeIndex):
        # pandas turns a leading extra field on every row into an index, silently.
        return None, "cannot parse: rows have more fields than the header"
    missing = {"y_true", "y_pred"} - set(num.columns)
    if missing:
        return None, f"missing column(s) {sorted(missing)}"

    cols = [c for c in _PRED_COLUMNS if c in num.columns]
    if not all(pd.api.types.is_numeric_dtype(num[c]) for c in cols):
        # Something did not parse as a number: say how much, then reject.
        coerced = num[cols].apply(pd.to_numeric, errors="coerce")
        rows = np.flatnonzero((coerced.isna() & num[cols].notna()).to_numpy().any(axis=1))
        return None, (f"{len(rows)} of {len(num)} rows hold non-numeric values "
                      f"(first at CSV line {rows[0] + 2})")

    out = pd.DataFrame({"row": np.arange(len(num), dtype=np.int32)})
    for c in ("ident", "fold"):
        out[c] = (num[c].round().astype("Int64") if c in num else
                  pd.arrays.IntegerArray(np.zeros(len(num), np.int64), np.ones(len(num), bool)))
    out["y_true"] = num["y_true"].astype(float)
    out["y_pred"] = num["y_pred"].astype(float)
    return out, None


@dataclass
class RunResult:
    run: dict[str, Any]
    predictions: pd.DataFrame | None
    folds: list[dict[str, Any]]
    figures: list[dict[str, Any]]
    issues: list[dict[str, Any]]


def _issue(kind: str, rel_path: Path | str, detail: str = "", run_id: str | None = None) -> dict:
    severity, meaning = ISSUE_KINDS[kind]
    return {"run_id": run_id, "rel_path": str(rel_path), "severity": severity,
            "kind": kind, "detail": detail or meaning}


def run_id_for(decoded: dict) -> str:
    """Stable, file-safe id from the factors. For real runs it equals the
    explorer's run_id, so parity sidecars keep their names."""
    layer = decoded["layer"]
    parts = [decoded["gnn_model_type"], f"rmsd{float(decoded['rmsd_threshold']):g}",
             decoded["split_type"], decoded["target_full"], decoded["prob_model"],
             f"L{'na' if layer is None else layer}"]
    return _safe_name("__".join(str(p) for p in parts))


def read_run(files: RunFiles, decoded: dict, run_id: str, root: Path) -> RunResult:
    rel = lambda p: p.relative_to(root)  # noqa: E731
    issues: list[dict] = []
    row: dict[str, Any] = {
        "run_id": run_id,
        **{k: decoded[k] for k in ("gnn_model_type", "rmsd_threshold", "split_type", "target",
                                   "target_full", "is_baseline", "prob_model", "layer")},
        "rel_dir": str(rel(files.exp_root)),
        "has_summary": False,
        "has_predictions": False,
        "n_predictions": None,
        "has_ident": False,
    }
    if decoded["layer"] is None:
        issues.append(_issue("legacy_layout", rel(files.exp_root), run_id=run_id))

    folds: list[dict] = []
    summary = None
    if files.summary is None:
        issues.append(_issue("summary_missing", rel(files.exp_root), run_id=run_id))
    else:
        summary, err = _read_json(files.summary)
        if err or not isinstance(summary, dict):
            issues.append(_issue("summary_unreadable", rel(files.summary),
                                 err or "not a JSON object", run_id))
            summary = None
        else:
            row["has_summary"] = True
            row["summary"] = json.dumps(_clean_json(summary))
            row.update(flatten_summary(summary))
            folds = [{"run_id": run_id, **f} for f in flatten_folds(summary)]

    if files.best_params is not None:
        params, err = _read_json(files.best_params)
        if err:
            issues.append(_issue("best_params_unreadable", rel(files.best_params), err, run_id))
        else:
            row["best_params"] = json.dumps(_clean_json(params))

    preds = None
    if files.predictions is None:
        issues.append(_issue("predictions_missing", rel(files.exp_root), run_id=run_id))
    else:
        preds, err = read_predictions(files.predictions)
        if err:
            issues.append(_issue("predictions_corrupt", rel(files.predictions), err, run_id))
        else:
            preds.insert(0, "run_id", run_id)
            row.update(has_predictions=True, n_predictions=len(preds),
                       has_ident=bool(preds["ident"].notna().any()))
            expected = row.get("n_test_samples")
            if expected is not None and expected != len(preds):
                issues.append(_issue("predictions_count_mismatch", rel(files.predictions),
                                     f"{len(preds)} rows, summary says n_test_samples={expected}",
                                     run_id))

    figures = [{"run_id": run_id, "kind": p.stem[len(files.probe) + 1:], "rel_path": str(rel(p))}
               for p in files.figures]
    return RunResult(row, preds, folds, figures, issues)


# ─────────────────────────────────────────────────────────────
# Writing a source
# ─────────────────────────────────────────────────────────────


def _sql_str(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _create(con: duckdb.DuckDBPyConnection, table: Table, frame: pd.DataFrame | None = None) -> None:
    """Create `table` (plus open columns found in `frame`) and load `frame` into it."""
    extra: dict[str, str] = {}
    if frame is not None and table.open_type:
        extra = {c: table.open_type for c in frame.columns if c not in table.names}
    con.execute(table.ddl(extra))
    if frame is not None and len(frame):
        frame = frame[[c for c in frame.columns if c in table.names or c in extra]]
        # Rows built from dicts leave NaN where a key was absent; make that NULL
        # (NaN in a string column, or in an INTEGER one, does not cast).
        _insert(con, table.name, frame.astype(object).where(frame.notna(), None))


def _insert(con: duckdb.DuckDBPyConnection, name: str, frame: pd.DataFrame) -> None:
    con.register("_frame", frame)
    try:
        con.execute(f'INSERT INTO "{name}" BY NAME SELECT * FROM _frame')
    finally:
        con.unregister("_frame")


def _relative_to_data(root: Path) -> str:
    data = get_data_dir(prob=False).resolve()
    try:
        return str(root.resolve().relative_to(data))
    except ValueError:
        return str(root.resolve())


def partition_dir(db_dir: Path, source_name: str) -> Path:
    return db_dir / f"{PARTITION_PREFIX}{source_name}"


def stored_fingerprint(db_dir: Path, source_name: str) -> tuple[str | None, int | None]:
    """(fingerprint, schema_version) recorded at the source's last ingest."""
    path = partition_dir(db_dir, source_name) / f"{SOURCES.name}.parquet"
    if not path.is_file():
        return None, None
    try:
        row = duckdb.sql(f"SELECT fingerprint, schema_version FROM read_parquet({_sql_str(path)})"
                         ).fetchone()
    except duckdb.Error:
        return None, None
    return (row[0], row[1]) if row else (None, None)


#: Predictions are inserted in batches of about this many rows: one insert per
#: run is slow, one for the whole source holds every prediction in memory twice.
_INSERT_BATCH_ROWS = 1_000_000


@dataclass
class IngestResult:
    source: str
    status: str            # "ingested" | "unchanged" | "empty"
    n_runs: int = 0
    n_predictions: int = 0
    n_errors: int = 0
    n_warnings: int = 0
    seconds: float = 0.0


def ingest_source(source: Source, db_dir: Path, *, force: bool = False,
                  workers: int = 8) -> IngestResult:
    """(Re)build one source's partition, unless its inputs are unchanged."""
    t0 = time.perf_counter()
    root = source.root
    runs = scan_source(root) if root.is_dir() else []
    if not runs and source.kind == "archive":
        return IngestResult(source.name, "empty")

    fp = fingerprint(runs, root)
    if not force and stored_fingerprint(db_dir, source.name) == (fp, SCHEMA_VERSION):
        return IngestResult(source.name, "unchanged")

    # Decode every run's factors first, so run_ids are settled before the reads.
    issues: list[dict] = []
    planned: list[tuple[RunFiles, dict, str]] = []
    seen: dict[str, int] = {}
    for files in runs:
        probe_csv = files.exp_root / EXP_DIR_ARTIFACTS / f"{files.probe}{PRED_SUFFIX}"
        decoded = _decode_pred_path(probe_csv, root, BASELINE_TAG)
        if decoded is None:
            for p in (files.summary, files.predictions):
                if p is not None:
                    issues.append(_issue("unrecognised_path", p.relative_to(root)))
            continue
        rid = run_id_for(decoded)
        seen[rid] = seen.get(rid, 0) + 1
        if seen[rid] > 1:
            first = rid
            rid = f"{rid}-{seen[rid]}"
            issues.append(_issue("duplicate_run", files.exp_root.relative_to(root),
                                 f"same factors as {first}; stored as {rid}", rid))
        planned.append((files, decoded, rid))

    con = duckdb.connect()
    for table in (PREDICTIONS, FIGURES):
        _create(con, table)
    run_rows: list[dict] = []
    fold_rows: list[dict] = []
    figure_rows: list[dict] = []
    pending: list[pd.DataFrame] = []
    n_predictions = n_pending = 0

    def flush() -> None:
        nonlocal n_pending
        if pending:
            _insert(con, PREDICTIONS.name, pd.concat(pending, ignore_index=True))
            pending.clear()
            n_pending = 0

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for res in pool.map(lambda job: read_run(*job, root), planned):
            run_rows.append(res.run)
            fold_rows += res.folds
            figure_rows += res.figures
            issues += res.issues
            if res.predictions is not None:
                pending.append(res.predictions)
                n_pending += len(res.predictions)
                n_predictions += len(res.predictions)
                if n_pending >= _INSERT_BATCH_ROWS:
                    flush()
    flush()
    if figure_rows:
        _insert(con, FIGURES.name, pd.DataFrame(figure_rows))
    _create(con, RUNS, pd.DataFrame(run_rows) if run_rows else None)
    _create(con, FOLDS, pd.DataFrame(fold_rows) if fold_rows else None)
    _create(con, ISSUES, pd.DataFrame(issues) if issues else None)

    n_errors = sum(i["severity"] == "error" for i in issues)
    n_warnings = len(issues) - n_errors
    seconds = time.perf_counter() - t0
    _create(con, SOURCES, pd.DataFrame([{
        "kind": source.kind, "label": source.label, "root": _relative_to_data(root),
        "ingested_at": datetime.now().replace(microsecond=0), "schema_version": SCHEMA_VERSION,
        "fingerprint": fp, "n_runs": len(run_rows), "n_predictions": n_predictions,
        "n_errors": n_errors, "n_warnings": n_warnings, "ingest_seconds": round(seconds, 2),
    }]))

    _write_partition(con, db_dir, source.name)
    return IngestResult(source.name, "ingested", len(run_rows), n_predictions,
                        n_errors, n_warnings, round(time.perf_counter() - t0, 2))


#: Sort order inside each file: keeps one run's rows together, so reading a
#: single run touches only the row groups whose run_id range covers it.
_ORDER = {"runs": "run_id", "predictions": "run_id, row", "folds": "run_id, fold",
          "figures": "run_id, kind", "issues": "severity, kind, rel_path"}


def _write_partition(con: duckdb.DuckDBPyConnection, db_dir: Path, name: str) -> None:
    """Write every table to a hidden temp folder, then swap it in."""
    db_dir.mkdir(parents=True, exist_ok=True)
    final = partition_dir(db_dir, name)
    tmp = db_dir / f".tmp-{final.name}-{os.getpid()}"
    old = db_dir / f".old-{final.name}-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        for table in TABLES:
            order = f" ORDER BY {_ORDER[table.name]}" if table.name in _ORDER else ""
            con.execute(f'COPY (SELECT * FROM "{table.name}"{order}) '
                        f"TO {_sql_str(tmp / f'{table.name}.parquet')} "
                        "(FORMAT parquet, COMPRESSION zstd)")
        if final.exists():
            final.rename(old)
        tmp.rename(final)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(old, ignore_errors=True)


def stored_sources(db_dir: Path) -> list[str]:
    if not db_dir.is_dir():
        return []
    return sorted(p.name[len(PARTITION_PREFIX):] for p in db_dir.iterdir()
                  if p.is_dir() and p.name.startswith(PARTITION_PREFIX))


def ingest(names: Iterable[str] | None = None, *,
           root: str | Path | None = None,
           archive_root: str | Path | None = None,
           archives: bool = True,
           db_dir: str | Path | None = None,
           force: bool = False,
           prune: bool = False,
           workers: int = 8,
           log: Callable[[str], None] | None = print) -> list[IngestResult]:
    """Bring the database up to date with the files on disk.

    names   only these sources ("current" or archive folder names); default all.
    force   rebuild even sources whose inputs are unchanged.
    prune   drop stored sources whose directory no longer exists. Without it they
            are kept and reported.
    """
    db_dir = Path(db_dir) if db_dir is not None else default_db_dir()
    sources = discover_sources(root, archive_root=archive_root, archives=archives)
    if names is not None:
        wanted = set(names)
        unknown = wanted - {s.name for s in sources}
        if unknown:
            raise ValueError(f"unknown source(s) {sorted(unknown)}; "
                             f"found {[s.name for s in sources]}")
        sources = [s for s in sources if s.name in wanted]

    results = []
    for source in sources:
        res = ingest_source(source, db_dir, force=force, workers=workers)
        results.append(res)
        if log:
            log(_describe(source, res))

    if names is None and archives:
        known = {s.name for s in discover_sources(root, archive_root=archive_root)}
        for orphan in sorted(set(stored_sources(db_dir)) - known):
            if prune:
                shutil.rmtree(partition_dir(db_dir, orphan))
                if log:
                    log(f"  {orphan:<22} pruned (source directory is gone)")
            elif log:
                log(f"  {orphan:<22} stale: source directory is gone (ingest --prune drops it)")
    return results


def _describe(source: Source, res: IngestResult) -> str:
    head = f"  {source.name:<22}"
    if res.status != "ingested":
        return f"{head} {res.status}"
    flags = []
    if res.n_errors:
        flags.append(f"{res.n_errors} error{'s' * (res.n_errors != 1)}")
    if res.n_warnings:
        flags.append(f"{res.n_warnings} warning{'s' * (res.n_warnings != 1)}")
    tail = f"  [{', '.join(flags)}]" if flags else ""
    return (f"{head} {res.n_runs:>5} runs  {res.n_predictions:>10,} predictions  "
            f"{res.seconds:5.1f}s{tail}")
