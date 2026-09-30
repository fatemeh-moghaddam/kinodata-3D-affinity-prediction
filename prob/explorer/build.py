"""Build the standalone Probe Sweep Explorer page from the runs on disk.

The page is one HTML file: `template.html` with one JSON payload injected into
it. There is no server, no build toolchain and no CDN -- open the output with a
browser.

The payload carries one run table per *source*: the live sweep under
data/probing and every timestamped snapshot under data/probing_archive (see
prob/cluster/archive_probe_results.sh). The page switches between them.

Parity plots need every prediction, which is far too much to inline, so each
run's (y_true, y_pred) goes into its own small script next to the page,
<page stem>_parity/<source>/<run id>.js, loaded only when that run is plotted.
A <script src> works from file:// where fetch() does not, which is what keeps
this serverless. Move the page and that folder together.

Everything the page knows about the sweep comes from that payload, so the
factors, the metrics, the table columns and the opening filter are declared
*here*, in Python:

    FACTORS        the experimental axes (filters, matrix rows/cols, colour-by)
    METRICS        what can be plotted, and how each one behaves
    EXTRA_COLUMNS  extra per-run numbers to carry into the table + tooltips
    DEFAULT_FILTER the slice the page opens on

Adding an axis or a metric is a one-line change to those lists plus, if it is
not already in the runs frame, a column in `collect_runs`.

Usage
-----
    uv run python -m prob.explorer
    uv run python -m prob.explorer --out /tmp/affinity.html --target affinity
    uv run python -m prob.explorer --gnn CGNN-3D --split random-k-fold --open
    uv run python -m prob.explorer --no-archives --no-parity

From a notebook:

    from prob.explorer import build_explorer
    build_explorer("explorer.html", target="affinity")
"""
from __future__ import annotations

import argparse
import json
import math
import re
import webbrowser
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from prob.paths_and_io import (
    attach_run_metrics,
    find_probe_runs,
    get_data_dir,
    load_run_predictions,
)

TEMPLATE_PATH = Path(__file__).with_name("template.html")
PLACEHOLDER = "__RUNS_JSON__"

DEFAULT_OUT = "probe_explorer.html"

ARCHIVE_DIRNAME = "probing_archive"   # data/probing_archive/<timestamp>/
ARCHIVE_STAMP = "%Y%m%d_%H%M%S"        # the name archive_probe_results.sh gives it

PARITY_SUFFIX = "_parity"              # <page stem>_parity/<source>/<run id>.js
PARITY_CALLBACK = "__probeParity"      # the global each sidecar calls


# ─────────────────────────────────────────────────────────────
# What the page is about: declare it once, here
# ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Factor:
    """One experimental axis. `key` must be a column of the runs frame."""

    key: str
    label: str
    prefix: str = ""   # rendered before the level, e.g. "L" -> "L3"
    suffix: str = ""   # rendered after it, e.g. " Å" -> "≤ 2 Å"

    def as_dict(self) -> dict:
        return {"key": self.key, "label": self.label,
                "prefix": self.prefix, "suffix": self.suffix}


@dataclass(frozen=True)
class Metric:
    """One plottable quantity.

    higher_better  which direction is good (drives ranking and the colour ramp)
    cross_target   is it comparable between targets? RMSE/MAE are not -- they
                   carry each target's own units -- and the page warns when a
                   non-comparable metric is shown across several targets.
    ci_lower/upper column names of a precomputed interval, if there is one.
                   Where present the depth curves draw a band and the ranked
                   view draws a whisker.
    baseline       column holding the paired control value, if there is one.
    """

    key: str
    label: str
    higher_better: bool = True
    cross_target: bool = True
    decimals: int = 3
    ci_lower: str | None = None
    ci_upper: str | None = None
    baseline: str | None = None

    def as_dict(self) -> dict:
        return {"label": self.label,
                "higherBetter": self.higher_better,
                "crossTarget": self.cross_target,
                "dec": self.decimals,
                "ciLower": self.ci_lower,
                "ciUpper": self.ci_upper,
                "baseline": self.baseline}


@dataclass(frozen=True)
class Column:
    """An extra numeric column for the table and the tooltips."""

    key: str
    label: str
    numeric: bool = True
    decimals: int | None = None   # None -> print as-is (counts, ids)

    def as_dict(self) -> dict:
        return {"key": self.key, "label": self.label,
                "num": self.numeric, "dec": self.decimals}


FACTORS: list[Factor] = [
    Factor("target", "Target"),
    Factor("gnn_model_type", "GNN"),
    Factor("layer", "Layer", prefix="L"),
    Factor("prob_model", "Probe"),
    Factor("rmsd_threshold", "RMSD", prefix="≤ ", suffix=" Å"),
    Factor("split_type", "Split"),
]

METRICS: list[Metric] = [
    Metric("r2", "R²", decimals=3,
           ci_lower="r2_ci_lower", ci_upper="r2_ci_upper", baseline="r2_baseline"),
    Metric("r2_delta", "ΔR² vs shuffled", decimals=3),
    Metric("rmse", "RMSE", higher_better=False, cross_target=False, decimals=3,
           ci_lower="rmse_ci_lower", ci_upper="rmse_ci_upper"),
    Metric("mae", "MAE", higher_better=False, cross_target=False, decimals=3),
    Metric("pearson", "Pearson r", decimals=3,
           ci_lower="pearson_ci_lower", ci_upper="pearson_ci_upper"),
    # Band = mean +/- sd over the GNN checkpoints, not a bootstrap CI.
    Metric("r2_ckpt_mean", "R² (checkpoint mean ± sd)", decimals=3,
           ci_lower="r2_ckpt_lower", ci_upper="r2_ckpt_upper"),
    Metric("pearson_ckpt_mean", "Pearson r (checkpoint mean ± sd)", decimals=3,
           ci_lower="pearson_ckpt_lower", ci_upper="pearson_ckpt_upper"),
]

EXTRA_COLUMNS: list[Column] = [
    Column("r2_baseline", "Baseline", decimals=3),
    Column("r2_ckpt_sd", "R² sd (ckpts)", decimals=3),
    Column("n_test_samples", "Test n"),
    Column("n_features", "Features"),
]

# The page opens on a readable slice rather than all runs at once; every filter
# is still one click away. Values are matched as strings against the levels, and
# a default naming a level that is not present is ignored rather than emptying
# the page.
DEFAULT_FILTER: dict[str, list[str]] = {
    "prob_model": ["mlp"],
    "split_type": ["random-k-fold"],
    # "rmsd_threshold": ["2"],
    "layer": ["0", "1", "2", "3"],
    "target": ["affinity"],
}

DEFAULT_METRIC = "r2"

# Which factor sits where when the page first opens. Values must be FACTORS
# keys. The two view groups below are independent -- the depth curves and the
# coverage matrix each have their own axes, and neither reads the other's.
DEFAULT_VIEW = {
    # ---- Depth curves tab -------------------------------------------------
    # Four channels put a factor somewhere: the x axis, colour, marker shape,
    # and the two panel-grid axes. Anything left over collides on one mark and
    # the page says so rather than averaging it (see `aggregate`).
    "depthAxis": "layer",         # x axis
    "colourBy": "gnn_model_type",
    "shapeBy": "",                # second channel in the same panel; "" = off
    # The panel grid: rows x columns. Leave one unset ("") for a single wrapped
    # strip of panels, or set both for a 2-D facet grid.
    "facetRowBy": "target",
    "facetColBy": "rmsd_threshold",
    "panelsPerRow": "auto",       # "auto" | 1..4; only used with one facet set
    "yScale": "free",             # "free" (per panel) | "row" | "shared"

    # ---- Coverage matrix tab ---------------------------------------------
    # Nothing to do with the depth-curve grid above: this is the heatmap's own
    # pair of axes. Both are required -- a matrix with one axis is a list -- so
    # "" is not meaningful here and falls back to the first factor.
    "matrixRow": "target",
    "matrixCol": "layer",

    # ---- Parity tab -------------------------------------------------------
    # One run per panel, so every factor not on a panel axis has to be pinned
    # to a single level; the page asks when one is not. "" = axis unset.
    "parityRow": "gnn_model_type",
    "parityCol": "layer",
    "parityDraw": "density",      # "density" (dots coloured by density) | "points"
    "parityScale": "target",      # "target" (shared within a target) | "panel"

    # ---- Depth curves and coverage matrix ---------------------------------
    # What to do when several runs land on one mark because a factor was left
    # off every channel. "split" draws them separately and averages nothing;
    # "break" refuses to place a value; "mean"/"median" collapse but stay
    # flagged with a red star. Never silently averaged.
    "aggregate": "split",         # "split" | "mean" | "median" | "break"
}


# ─────────────────────────────────────────────────────────────
# Collect
# ─────────────────────────────────────────────────────────────


def collect_runs(**filters: Any) -> pd.DataFrame:
    """Index the sweep and attach every number the page plots.

    Baselines are not rows in their own right here: each `shuffled_ident` run is
    joined onto the real run that shares its factors, as `r2_baseline`, and the
    difference becomes `r2_delta`. That keeps one row per experiment, which is
    what every view assumes.

    `filters` are forwarded to `find_probe_runs` (gnn_model_type, target, ...),
    so a page can be built for one slice of the sweep as easily as for all of it.
    """
    runs = attach_run_metrics(find_probe_runs(include_baselines=True, **filters))
    if runs.empty:
        return runs

    keys = [f.key for f in FACTORS]
    real = runs[~runs["is_baseline"]].copy()
    base = runs[runs["is_baseline"]]

    if not base.empty:
        paired = (base.groupby(keys, dropna=False)["r2"].first()
                      .rename("r2_baseline").reset_index())
        real = real.merge(paired, on=keys, how="left")
    else:
        real["r2_baseline"] = np.nan

    # With no paired control the honest delta is the raw score: the shuffled
    # baseline sits at R² ≈ 0 across this sweep, so 0 is the right stand-in.
    real["r2_delta"] = real["r2"] - real["r2_baseline"].fillna(0.0)
    return real


def drop_incomplete(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off runs that are missing a factor value.

    A run with no value on some axis cannot be placed by a page that filters on
    every axis -- it would be silently invisible while still counting toward the
    total. `find_probe_runs` yields these for the handful of legacy paths that
    predate the `<layer>` directory level. Returns (usable, dropped) so callers
    can say out loud how many were set aside.
    """
    if runs.empty:
        return runs, runs
    keys = [f.key for f in FACTORS]
    incomplete = runs[keys].isna().any(axis=1)
    return runs[~incomplete].copy(), runs[incomplete].copy()


@dataclass(frozen=True)
class Source:
    """One probing directory the page can switch to: the live sweep or an archive.

    `runs` is already through `drop_incomplete`; `n_dropped` says how many were
    set aside so the page can repeat it.
    """

    key: str        # stable id, also the sidecar sub-folder name
    label: str      # what the picker shows
    kind: str       # "current" | "archive"
    path: Path
    runs: pd.DataFrame
    n_dropped: int = 0


def default_archive_root() -> Path:
    return get_data_dir(prob=False) / ARCHIVE_DIRNAME


def list_archives(archive_root: str | Path) -> list[Path]:
    """Archive snapshots, newest first (the timestamp names sort by time)."""
    archive_root = Path(archive_root)
    if not archive_root.is_dir():
        return []
    return sorted((p for p in archive_root.iterdir() if p.is_dir()),
                  key=lambda p: p.name, reverse=True)


def _archive_label(name: str) -> str:
    """'20260929_160507' -> '2026-09-29 16:05:07'; any other name as-is."""
    try:
        return datetime.strptime(name, ARCHIVE_STAMP).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return name


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text)).strip("-") or "x"


def collect_sources(
        root: str | Path | None = None,
        *,
        archive_root: str | Path | None = None,
        archives: bool = True,
        **filters: Any,
    ) -> list[Source]:
    """The live sweep plus every archive snapshot, each indexed with `collect_runs`.

    The live source is always listed -- empty right after an archive run, which
    the page shows rather than hides -- while an archive with nothing matching
    `filters` is left out.
    """
    root = Path(root) if root is not None else get_data_dir(prob=True)
    places = [("current", "Current", "current", root)]
    if archives:
        a_root = Path(archive_root) if archive_root is not None else default_archive_root()
        places += [(_safe_name(p.name), _archive_label(p.name), "archive", p)
                   for p in list_archives(a_root)]

    sources = []
    for key, label, kind, path in places:
        runs = collect_runs(root=path, **filters) if path.is_dir() else pd.DataFrame()
        usable, dropped = drop_incomplete(runs)
        if kind == "archive" and usable.empty:
            continue
        sources.append(Source(key, label, kind, path, usable, len(dropped)))
    return sources


# ─────────────────────────────────────────────────────────────
# Parity sidecars
# ─────────────────────────────────────────────────────────────


def run_id(row: Any) -> str:
    """A file-safe id from a run's factors, stable across rebuilds."""
    rmsd = row["rmsd_threshold"]
    parts = [row["gnn_model_type"], f"rmsd{float(rmsd):g}", row["split_type"],
             row["target"], row["prob_model"], f"L{row['layer']}"]
    return _safe_name("__".join(str(p) for p in parts))


def _parity_js(key: str, y_true: np.ndarray, y_pred: np.ndarray) -> str:
    # 4 significant figures: well below a pixel on any panel, and it halves the
    # file size against full float repr.
    def numbers(a: np.ndarray) -> str:
        return ",".join(f"{v:.4g}" for v in a.tolist())

    return (f"window.{PARITY_CALLBACK}({json.dumps(key)},"
            f'{{"t":[{numbers(y_true)}],"p":[{numbers(y_pred)}]}});\n')


def write_parity_sidecars(runs: pd.DataFrame, out_dir: str | Path,
                          source_key: str) -> pd.Series:
    """Write <out_dir>/<run id>.js for every run with a readable predictions CSV.

    Returns the run id per row (None where no sidecar exists), for the page to
    find the file by. A sidecar at least as new as its CSV is left alone, so
    rebuilding over an archive that never changes costs nothing. Sidecars in
    `out_dir` that no longer belong to a run are removed.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = pd.Series([None] * len(runs), index=runs.index, dtype=object)
    if "predictions_path" not in runs.columns:
        return ids

    taken: dict[str, int] = {}
    for idx, row in runs.iterrows():
        raw = row["predictions_path"]
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            continue
        csv = Path(raw)
        if not csv.is_file():
            continue

        rid = run_id(row)
        taken[rid] = taken.get(rid, 0) + 1
        if taken[rid] > 1:   # two CSVs decoding to the same factors
            rid = f"{rid}-{taken[rid]}"

        target = out_dir / f"{rid}.js"
        if not (target.exists() and target.stat().st_mtime >= csv.stat().st_mtime):
            try:
                y_true, y_pred = load_run_predictions(csv)
            except (OSError, ValueError, pd.errors.ParserError):
                continue
            keep = np.isfinite(y_true) & np.isfinite(y_pred)
            target.write_text(_parity_js(f"{source_key}/{rid}", y_true[keep], y_pred[keep]),
                              encoding="utf-8")
        ids[idx] = rid

    written = {f"{rid}.js" for rid in ids.dropna()}
    for stale in out_dir.glob("*.js"):
        if stale.name not in written:
            stale.unlink()
    return ids


# ─────────────────────────────────────────────────────────────
# Serialise
# ─────────────────────────────────────────────────────────────


def _jsonable(value: Any) -> Any:
    """Numpy/pandas scalar -> plain JSON value, with NaN collapsed to null."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if math.isnan(f) else f
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _round(value: Any, places: int = 6) -> Any:
    v = _jsonable(value)
    return round(v, places) if isinstance(v, float) else v


def build_payload(runs: pd.DataFrame, *, source: str | Path | None = None,
                  n_dropped: int = 0) -> dict:
    """The single JSON blob the page reads: its config, then its rows."""
    metric_keys = [m.key for m in METRICS]
    ci_keys = [k for m in METRICS for k in (m.ci_lower, m.ci_upper) if k]
    wanted = ([f.key for f in FACTORS] + metric_keys + ci_keys
              + [c.key for c in EXTRA_COLUMNS])

    missing = [c for c in wanted if c not in runs.columns]
    for column in missing:
        runs = runs.assign(**{column: np.nan})
    # which parity sidecar belongs to the run; absent when none were written
    if "parity_id" in runs.columns:
        wanted.append("parity_id")

    integer_cols = {"layer", "rmsd_threshold", "n_test_samples", "n_features"}
    records = []
    for row in runs[wanted].itertuples(index=False):
        record = {}
        for key, value in zip(wanted, row):
            v = _round(value)
            if key in integer_cols and isinstance(v, float):
                v = int(v)
            record[key] = v
        records.append(record)

    config = {
        "factors": [f.as_dict() for f in FACTORS],
        "metrics": {m.key: m.as_dict() for m in METRICS},
        "metricOrder": metric_keys,
        "extraColumns": [c.as_dict() for c in EXTRA_COLUMNS],
        "defaults": dict(DEFAULT_VIEW, metric=DEFAULT_METRIC, filter=DEFAULT_FILTER),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": str(source) if source is not None else "",
        "nRuns": len(records),
        "nDropped": int(n_dropped),
    }
    if missing:
        config["missingColumns"] = missing
    return {"config": config, "runs": records}


def build_sources_payload(sources: Sequence[Source], *,
                          parity_dir: str | None = None) -> dict:
    """The payload for a page that switches between sources.

    The config (factors, metrics, defaults) is shared; each source carries its
    own runs and provenance. The page opens on the live sweep when it has runs,
    otherwise on the newest archive.
    """
    if not sources:
        raise ValueError("no sources to build a page from")
    parts = [build_payload(s.runs, source=s.path, n_dropped=s.n_dropped) for s in sources]

    config = dict(parts[0]["config"])
    for key in ("source", "nRuns", "nDropped", "missingColumns"):
        config.pop(key, None)
    config["parityDir"] = parity_dir or ""

    entries = []
    for src, part in zip(sources, parts):
        c = part["config"]
        entries.append({
            "key": src.key,
            "label": src.label,
            "kind": src.kind,
            "path": c["source"],
            "nRuns": c["nRuns"],
            "nDropped": c["nDropped"],
            "runs": part["runs"],
        })
    default = next((e["key"] for e in entries if e["nRuns"]), entries[0]["key"])
    return {"config": config, "sources": entries, "defaultSource": default}


def render_html(payload: dict, template: str | Path = TEMPLATE_PATH) -> str:
    """Inject the payload into the template.

    `</` is escaped because the payload sits inside a <script> element, where an
    unescaped closing tag anywhere in the data would end the block early.
    """
    html = Path(template).read_text(encoding="utf-8")
    if PLACEHOLDER not in html:
        raise ValueError(f"{template} has no {PLACEHOLDER} placeholder")
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return html.replace(PLACEHOLDER, blob.replace("</", "<\\/"))


# ─────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────


def build_explorer(
        out_path: str | Path = DEFAULT_OUT,
        *,
        runs: pd.DataFrame | None = None,
        sources: Sequence[Source] | None = None,
        template: str | Path = TEMPLATE_PATH,
        root: str | Path | None = None,
        archive_root: str | Path | None = None,
        archives: bool = True,
        parity: bool = True,
        **filters: Any,
    ) -> Path:
    """Write the explorer page (plus its parity sidecars) and return its path.

    runs: a prepared frame (from `collect_runs`, or your own, as long as it has
        the factor and metric columns). The page then shows that frame alone.
    sources: prepared sources (from `collect_sources`). Takes precedence.
    Omit both to index data/probing and, with `archives`, every snapshot under
    `archive_root` (default data/probing_archive).
    parity: write the per-run sidecars the Parity tab reads. Runs without a
        `predictions_path` column simply have no parity plot.
    **filters: forwarded to `find_probe_runs` when indexing from disk.
    """
    if sources is None:
        if runs is not None:
            usable, dropped = drop_incomplete(runs)
            path = Path(root) if root is not None else get_data_dir(prob=True)
            sources = [Source("current", "Current", "current", path, usable, len(dropped))]
        else:
            sources = collect_sources(root, archive_root=archive_root,
                                      archives=archives, **filters)
    if not any(len(s.runs) for s in sources):
        where = ", ".join(str(s.path) for s in sources) or str(root)
        raise ValueError(f"No probe runs found under {where} for filters {filters or '{}'}.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parity_dir = None
    if parity:
        parity_dir = f"{out_path.stem}{PARITY_SUFFIX}"
        sources = [
            s if s.runs.empty else replace(s, runs=s.runs.assign(
                parity_id=write_parity_sidecars(s.runs, out_path.parent / parity_dir / s.key,
                                                s.key)))
            for s in sources
        ]

    payload = build_sources_payload(sources, parity_dir=parity_dir)
    out_path.write_text(render_html(payload, template), encoding="utf-8")
    return out_path


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────


def _as_list(values: Sequence[str] | None) -> list[str] | None:
    return list(values) if values else None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m prob.explorer",
        description="Build the standalone Probe Sweep Explorer page.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage\n-----\n", 1)[-1],
    )
    p.add_argument("--out", "-o", default=None, type=Path,
                   help=f"output HTML path (default: <data>/probing/{DEFAULT_OUT})")
    p.add_argument("--root", default=None, type=Path,
                   help="probing data directory to index (default: data/probing)")
    p.add_argument("--archive-root", default=None, type=Path,
                   help=f"folder of archive snapshots (default: data/{ARCHIVE_DIRNAME})")
    p.add_argument("--no-archives", action="store_true",
                   help="index only --root, not the archive snapshots")
    p.add_argument("--no-parity", action="store_true",
                   help="skip the per-run parity sidecars (the Parity tab stays empty)")
    p.add_argument("--open", dest="open_after", action="store_true",
                   help="open the page in a browser when it is written")

    g = p.add_argument_group("slice (all repeatable; default is the whole sweep)")
    g.add_argument("--gnn", nargs="+", metavar="NAME", help="gnn_model_type")
    g.add_argument("--target", nargs="+", metavar="NAME")
    g.add_argument("--probe", nargs="+", metavar="NAME", help="prob_model")
    g.add_argument("--split", nargs="+", metavar="NAME", help="split_type")
    g.add_argument("--rmsd", nargs="+", type=float, metavar="X", help="rmsd_threshold")
    g.add_argument("--layer", nargs="+", type=int, metavar="N")
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)

    root = args.root if args.root is not None else get_data_dir(prob=True)
    out = args.out if args.out is not None else Path(root) / DEFAULT_OUT

    filters = {
        "gnn_model_type": _as_list(args.gnn),
        "target": _as_list(args.target),
        "prob_model": _as_list(args.probe),
        "split_type": _as_list(args.split),
        "rmsd_threshold": _as_list(args.rmsd),
        "layer": _as_list(args.layer),
    }
    filters = {k: v for k, v in filters.items() if v is not None}

    sources = collect_sources(root, archive_root=args.archive_root,
                              archives=not args.no_archives, **filters)
    for s in sources:
        print(f"  {s.label:<21} {len(s.runs):>5} runs  {s.path}")
        if s.n_dropped:
            print(f"  [warn] {s.n_dropped} run(s) in {s.label} set aside -- missing a factor "
                  "value. These predate the current output layout.")
    if not any(len(s.runs) for s in sources):
        print(f"No probe runs found for {filters or 'the whole sweep'}.")
        return 1

    if not args.no_parity:
        print("  writing parity sidecars (only runs whose CSV changed are rewritten) ...")
    path = build_explorer(out, sources=sources, parity=not args.no_parity)
    size_kb = path.stat().st_size / 1024
    total = sum(len(s.runs) for s in sources)
    print(f"[ok] {total} runs in {len(sources)} source(s) -> {path}  ({size_kb:.0f} KB)")
    if not args.no_parity:
        side = path.parent / f"{path.stem}{PARITY_SUFFIX}"
        side_mb = sum(f.stat().st_size for f in side.rglob("*.js")) / 1e6
        print(f"     parity sidecars: {side}  ({side_mb:.0f} MB) -- keep it next to the page")
    print(f"     open it with:  open {path}")
    if args.open_after:
        webbrowser.open(path.resolve().as_uri())
    return 0
