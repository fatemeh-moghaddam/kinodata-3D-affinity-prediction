"""
Backfill bootstrap CIs into already-written probe summaries.

Why this exists
---------------
`run_probe_statistical_tests` bootstraps every metric in `prob_stats.METRIC_FNS`
(r2, rmse, mae, pearson), but summaries written before a metric was added to that
registry only carry the CIs that existed at fit time. A CI needs nothing but
(y_true, y_pred), and both are already saved in each run's predictions CSV -- so
the missing CIs are recoverable without refitting a single probe.

Runs are found with `find_probe_runs()` rather than by enumerating probe/layer
registries and testing paths, so only experiments that actually ran are visited.

By default this adds the *missing* CIs and leaves existing ones untouched: the
r2/rmse CIs on disk were bootstrapped with the fit-time seed, and silently
recomputing them under a different seed would move numbers that have already
been reported. Pass --overwrite to recompute every CI under one seed instead.

Usage
-----
    # look first -- no writes
    python prob/backfill_stats.py --dry-run

    # backfill everything on disk
    python prob/backfill_stats.py

    # or just one slice
    python prob/backfill_stats.py --gnn_model_type CGNN-3D --target hydrogen_bonds
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from prob.paths_and_io import find_probe_runs, load_run_predictions
from prob.prob_stats import METRIC_FNS, bootstrap_ci

# Matches prob_orchestrate.RANDOM_STATE (imported as a literal to keep this
# script free of the orchestrator's heavy dataset/config imports).
RANDOM_STATE = 96


def backfill_summary(
        run,
        metrics: list[str],
        *,
        n_bootstrap: int,
        confidence: float,
        random_state: int,
        overwrite: bool,
        dry_run: bool,
    ) -> tuple[str, list[str]]:
    """Add missing metric CIs to one run's summary JSON. Returns (status, added)."""
    path = Path(run.summary_path)
    if not path.exists():
        return "no summary", []
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return "unreadable", []

    stats = payload.setdefault("statistical_tests", {})
    todo = [m for m in metrics if overwrite or f"{m}_ci" not in stats]
    if not todo:
        return "up to date", []

    y_true, y_pred = load_run_predictions(run)
    for metric in todo:
        stats[f"{metric}_ci"] = bootstrap_ci(
            y_true, y_pred,
            metric_fn=METRIC_FNS[metric],
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            confidence=confidence,
            name=metric,
        )
    if not dry_run:
        path.write_text(json.dumps(payload, indent=2))
    return "updated", todo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gnn_model_type", nargs="*", default=None)
    parser.add_argument("--split_type", nargs="*", default=None)
    parser.add_argument("--rmsd_threshold", nargs="*", type=float, default=None)
    parser.add_argument("--target", nargs="*", default=None)
    parser.add_argument("--prob_model", nargs="*", default=None)
    parser.add_argument("--layer", nargs="*", type=int, default=None)
    parser.add_argument("--metrics", nargs="*", default=list(METRIC_FNS),
                        choices=list(METRIC_FNS), help="which CIs to ensure (default: all)")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute CIs that already exist, under --random-state")
    parser.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    args = parser.parse_args()

    runs = find_probe_runs(
        gnn_model_type=args.gnn_model_type,
        rmsd_threshold=args.rmsd_threshold,
        split_type=args.split_type,
        target=args.target,
        prob_model=args.prob_model,
        layer=args.layer,
    )
    print(f"{len(runs)} probe runs matched"
          f"{' (dry run -- nothing will be written)' if args.dry_run else ''}\n")

    status_counts: Counter[str] = Counter()
    added_counts: Counter[str] = Counter()
    for run in runs.itertuples():
        status, added = backfill_summary(
            run, args.metrics,
            n_bootstrap=args.n_bootstrap,
            confidence=args.confidence,
            random_state=args.random_state,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        status_counts[status] += 1
        added_counts.update(added)

    for status, count in status_counts.most_common():
        print(f"  {status:<12} {count}")
    if added_counts:
        print("\nCIs " + ("recomputed" if args.overwrite else "added") + ":")
        for metric, count in added_counts.most_common():
            print(f"  {metric:<12} {count}")


if __name__ == "__main__":
    main()
