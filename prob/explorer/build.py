"""Build the standalone Probe Sweep Explorer page from the runs on disk.

The page is a single self-contained HTML file: `template.html` with one JSON
payload injected into it. There is no server, no build toolchain and no CDN --
open the output with a browser.

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

From a notebook:

    from prob.explorer import build_explorer
    build_explorer("explorer.html", target="affinity")
"""
from __future__ import annotations

import argparse
import json
import math
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from prob.paths_and_io import attach_run_metrics, find_probe_runs, get_data_dir

TEMPLATE_PATH = Path(__file__).with_name("template.html")
PLACEHOLDER = "__RUNS_JSON__"

DEFAULT_OUT = "probe_explorer.html"


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
]

EXTRA_COLUMNS: list[Column] = [
    Column("r2_baseline", "Baseline", decimals=3),
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
    "rmsd_threshold": ["2"],
}

DEFAULT_METRIC = "r2"

# Which factor sits where when the page first opens. Each must be a FACTORS key;
# an unknown name falls back to a positional default rather than breaking a view.
DEFAULT_VIEW = {
    "depthAxis": "layer",         # x axis of the depth curves
    "colourBy": "gnn_model_type",
    # Depth curves lay out as a panel grid: rows x columns. Leave one unset ("")
    # for a single wrapped strip of panels, or set both for a 2-D facet grid.
    "facetRowBy": "",
    "facetColBy": "target",
    "panelsPerRow": "auto",       # "auto" | 1..4; only used with one facet set
    "yScale": "free",             # "free" (per panel) | "row" | "shared"
    # What to do when several runs land on one mark because a factor was left
    # off every axis. "split" draws them separately and averages nothing;
    # "break" refuses to place a value; "mean"/"median" collapse but stay
    # flagged with a red star. Never silently averaged.
    "aggregate": "split",         # "split" | "mean" | "median" | "break"
    "matrixRow": "target",
    "matrixCol": "layer",
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
        template: str | Path = TEMPLATE_PATH,
        root: str | Path | None = None,
        **filters: Any,
    ) -> Path:
    """Write a self-contained explorer page and return its path.

    runs: a prepared frame (from `collect_runs`, or your own, as long as it has
        the factor and metric columns). Omit it to index the sweep on disk.
    **filters: forwarded to `find_probe_runs` when `runs` is not given.
    """
    source = Path(root) if root is not None else get_data_dir(prob=True)
    if runs is None:
        runs = collect_runs(root=source, **filters)
    if runs.empty:
        raise ValueError(
            f"No probe runs found under {source} for filters {filters or '{}'}."
        )

    runs, dropped = drop_incomplete(runs)
    if runs.empty:
        raise ValueError(
            f"Every run found under {source} is missing at least one factor value."
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(runs, source=source, n_dropped=len(dropped))
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

    runs = collect_runs(root=root, **filters)
    if runs.empty:
        print(f"No probe runs found under {root} for {filters or 'the whole sweep'}.")
        return 1

    usable, dropped = drop_incomplete(runs)
    if len(dropped):
        keys = [f.key for f in FACTORS]
        print(f"[warn] {len(dropped)} run(s) set aside -- missing a factor value "
              f"({', '.join(sorted(dropped[keys].columns[dropped[keys].isna().any()]))}). "
              "These predate the current output layout; re-run them to include them.")

    path = build_explorer(out, runs=runs, root=root)
    size_kb = path.stat().st_size / 1024
    print(f"[ok] {len(usable)} runs -> {path}  ({size_kb:.0f} KB)")
    print(f"     open it with:  open {path}")
    if args.open_after:
        webbrowser.open(path.resolve().as_uri())
    return 0
