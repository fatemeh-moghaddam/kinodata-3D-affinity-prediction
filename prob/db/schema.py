"""The probing database's tables, declared once.

Everything else -- the ingest, the query views and the README -- reads these
declarations, so a new column or table is added here and nowhere else.

Storage is one folder per *source* (the live sweep or one archive snapshot):

    data/probing_db/source=<name>/<table>.parquet

`source` is not a column inside the files. It comes from the folder name
(Hive partitioning), so a source can be rebuilt or deleted on its own and the
query layer sees the union of all of them. Every table is keyed by
(source, run_id) or belongs to the source as a whole.

Open columns: `runs` and `folds` accept numeric columns that are not declared
here. A summary JSON that starts carrying a new metric (say `spearman`) lands
as a DOUBLE column without a schema change, and sources written before it read
it back as NULL. Declare it below once you rely on it, so it is typed and always
present.
"""
from __future__ import annotations

from dataclasses import dataclass

#: Bump when a declared column changes meaning or type. Sources ingested under
#: an older version are rebuilt on the next `ingest`, even if no file changed.
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Col:
    name: str
    type: str          # DuckDB type
    doc: str = ""


@dataclass(frozen=True)
class Table:
    name: str
    doc: str
    columns: tuple[Col, ...]
    #: DuckDB type for undeclared columns the ingest may add, or None to drop them.
    open_type: str | None = None

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    def ddl(self, extra: dict[str, str] | None = None, *, temp: bool = False) -> str:
        cols = [(c.name, c.type) for c in self.columns] + list((extra or {}).items())
        body = ",\n  ".join(f'"{name}" {type_}' for name, type_ in cols)
        return f'CREATE {"TEMP " if temp else ""}TABLE "{self.name}" (\n  {body}\n)'


def _metric_cols(names: tuple[str, ...], *, ci: bool, ckpt: bool) -> tuple[Col, ...]:
    cols: list[Col] = []
    for m in names:
        cols.append(Col(m, "DOUBLE", f"{m} over all test rows"))
        if ci:
            cols += [Col(f"{m}_ci_lower", "DOUBLE", f"bootstrap CI of {m}"),
                     Col(f"{m}_ci_upper", "DOUBLE", f"bootstrap CI of {m}")]
        if ckpt:
            cols += [Col(f"{m}_ckpt_mean", "DOUBLE", f"mean {m} over GNN checkpoints"),
                     Col(f"{m}_ckpt_sd", "DOUBLE", f"sd of {m} over GNN checkpoints")]
    return tuple(cols)


#: Metrics every summary is expected to report. Others are still kept (open columns).
CORE_METRICS = ("r2", "rmse", "mae", "pearson")

#: The experimental factors, decoded from the run's directory. Same names as
#: `find_probe_runs` and the explorer's FACTORS, so frames move between them.
FACTOR_COLUMNS = (
    Col("gnn_model_type", "VARCHAR", "CGNN-3D, CGNN, DTI, ..."),
    Col("rmsd_threshold", "DOUBLE", "docking RMSD cutoff in Å"),
    Col("split_type", "VARCHAR", "random-k-fold, scaffold-k-fold, pocket-k-fold"),
    Col("target", "VARCHAR", "probed property, without the baseline tag"),
    Col("target_full", "VARCHAR", "directory name, e.g. affinity_shuffled_ident"),
    Col("is_baseline", "BOOLEAN", "shuffled-ident control run"),
    Col("prob_model", "VARCHAR", "ridge, lasso, mlp, random_forest, ..."),
    Col("layer", "INTEGER", "GNN layer probed; NULL for the legacy layout"),
)

RUNS = Table(
    "runs",
    "One probe run: the factors that define it, the numbers its summary JSON "
    "reported, and where its files live.",
    (
        Col("run_id", "VARCHAR", "stable id from the factors; unique within a source"),
        *FACTOR_COLUMNS,
        Col("probe_mode", "VARCHAR", "'per_checkpoint' for runs with a folds table; NULL before"),
        Col("best_params_from", "VARCHAR", "file the params were copied from, for fixed-param runs"),
        Col("manifest_id", "VARCHAR", "run_id of the prob_orchestrate run that wrote it: "
                                      "<target>/experiments/run_manifests/<manifest_id>.json; NULL before"),
        Col("n_samples", "INTEGER"),
        Col("n_train_samples", "INTEGER"),
        Col("n_test_samples", "INTEGER"),
        Col("n_features", "INTEGER"),
        Col("n_splits_cv", "INTEGER"),
        *_metric_cols(CORE_METRICS, ci=True, ckpt=True),
        Col("fit_seconds", "DOUBLE"),
        Col("ci_confidence", "DOUBLE", "confidence level of the *_ci_* columns"),
        Col("best_params", "JSON", "reports/<probe>_best_params.json as written"),
        Col("summary", "JSON", "reports/<probe>_summary.json as written -- the escape hatch "
                               "for any field not flattened into a column"),
        Col("rel_dir", "VARCHAR", "run directory relative to the source root"),
        Col("has_summary", "BOOLEAN"),
        Col("has_predictions", "BOOLEAN", "false when the CSV is missing or failed validation"),
        Col("n_predictions", "INTEGER"),
        Col("has_ident", "BOOLEAN", "predictions carry idents, so runs can be paired"),
    ),
    open_type="DOUBLE",
)

PREDICTIONS = Table(
    "predictions",
    "Every test-set prediction of every run, in file order.",
    (
        Col("run_id", "VARCHAR"),
        Col("row", "INTEGER", "0-based row in the predictions CSV"),
        Col("ident", "BIGINT", "complex ident; NULL in runs that predate it"),
        Col("fold", "INTEGER", "CV fold / GNN checkpoint; NULL in runs that predate it"),
        Col("y_true", "DOUBLE"),
        Col("y_pred", "DOUBLE"),
    ),
)

FOLDS = Table(
    "folds",
    "Per-checkpoint runs only: one row per CV fold (the summary's per_fold list).",
    (
        Col("run_id", "VARCHAR"),
        Col("fold", "INTEGER"),
        Col("n_train_samples", "INTEGER"),
        Col("n_test_samples", "INTEGER"),
        *_metric_cols(CORE_METRICS, ci=False, ckpt=False),
        Col("params_source", "VARCHAR", "fixed / saved / shared / tuned / default"),
        Col("best_params", "JSON"),
    ),
    open_type="DOUBLE",
)

FIGURES = Table(
    "figures",
    "The PNGs a run wrote. The files stay on disk; this is their index.",
    (
        Col("run_id", "VARCHAR"),
        Col("kind", "VARCHAR", "parity, residuals, ... (file name minus the probe prefix)"),
        Col("rel_path", "VARCHAR", "relative to the source root"),
    ),
)

ISSUES = Table(
    "issues",
    "Everything the ingest could not take at face value, so nothing is dropped silently.",
    (
        Col("run_id", "VARCHAR", "NULL when the file could not be tied to a run"),
        Col("rel_path", "VARCHAR", "relative to the source root"),
        Col("severity", "VARCHAR", "'error': data left out; 'warning': kept, but look"),
        Col("kind", "VARCHAR", "one of ISSUE_KINDS"),
        Col("detail", "VARCHAR"),
    ),
)

SOURCES = Table(
    "sources",
    "One row per source: where it came from and when it was ingested.",
    (
        Col("kind", "VARCHAR", "'current' (data/probing) or 'archive'"),
        Col("label", "VARCHAR", "display name, e.g. 2026-09-29 16:05:07"),
        Col("root", "VARCHAR", "source directory, relative to data/ when inside it"),
        Col("ingested_at", "TIMESTAMP"),
        Col("schema_version", "INTEGER"),
        Col("fingerprint", "VARCHAR", "hash of the input files' paths, sizes and mtimes"),
        Col("n_runs", "INTEGER"),
        Col("n_predictions", "BIGINT"),
        Col("n_errors", "INTEGER"),
        Col("n_warnings", "INTEGER"),
        Col("ingest_seconds", "DOUBLE"),
    ),
)

TABLES: tuple[Table, ...] = (RUNS, PREDICTIONS, FOLDS, FIGURES, ISSUES, SOURCES)
TABLES_BY_NAME = {t.name: t for t in TABLES}

#: kind -> (severity, what it means)
ISSUE_KINDS: dict[str, tuple[str, str]] = {
    "predictions_corrupt": ("error", "CSV unreadable or holds non-numeric values; predictions left out"),
    "summary_unreadable": ("error", "summary JSON empty or invalid; metrics are NULL"),
    "best_params_unreadable": ("warning", "best-params JSON empty or invalid"),
    "predictions_missing": ("warning", "run has a summary but no predictions CSV"),
    "summary_missing": ("warning", "run has predictions but no summary JSON; metrics are NULL"),
    "predictions_count_mismatch": ("warning", "CSV rows differ from the summary's n_test_samples"),
    "legacy_layout": ("warning", "path lacks the <layer> level; layer is NULL"),
    "duplicate_run": ("warning", "two files decode to the same factors; run_id got a suffix"),
    "unrecognised_path": ("warning", "file sits where no run layout is expected; skipped"),
}
