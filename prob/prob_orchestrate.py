"""
Orchestrate probing experiments: load X/y, run registered linear and non-linear
probes per layer, and aggregate results.

To add a probe: append an entry to LINEAR_PROBES or NONLINEAR_PROBES in
prob_models (or pass a custom list to run_probes). Statistical analysis
lives in prob_stats; metrics in prob_metrics; CV pipeline in prob_run.
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import kinodata.configuration as cfg

from prob.paths_and_io import get_exp_dirs, load_X_from_pt, load_y_by_ids
from prob.prob_config import get_ds_load_config
from prob.prob_run import run_cv_search, run_probe, tune_probe

# Probe registry: add entries here to run new linear or non-linear probes.
# For metrics/stats use prob.prob_metrics and prob.prob_stats.
from prob.prob_models import LINEAR_PROBES, NONLINEAR_PROBES

import wandb


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

RANDOM_STATE = 96
N_SPLITS_CV = 5
TEST_SIZE = 0.1
TARGET_FILE = None
_ROOT = Path(os.environ.get("HOME_PROJ_DIR", Path(__file__).resolve().parents[1]))


# ─────────────────────────────────────────────────────────────
# Resource helpers
# ─────────────────────────────────────────────────────────────


def _cpu_budget(default: int = 16, sub_file: Optional[Path] = None) -> int:
    """Infer job CPU count from scheduler env vars, submission file, or default."""
    for key in ("SLURM_CPUS_PER_TASK", "NSLOTS", "OMP_NUM_THREADS"):
        if key in os.environ and os.environ[key].isdigit():
            return int(os.environ[key])
    if sub_file is not None and Path(sub_file).exists():
        text = Path(sub_file).read_text()
        match = re.search(r"request_CPUs\s*=\s*(\d+)", text, flags=re.I)
        if match:
            return int(match.group(1))
    return default


# ─────────────────────────────────────────────────────────────
# Probe runner (uses registry)
# ─────────────────────────────────────────────────────────────


def run_probes(
    prob_config: cfg.Config,
    X: np.ndarray,
    y: np.ndarray,
    probe_entries: List[Dict[str, Any]],
    n_jobs: int,
    reuse_best_params: bool = False,
    best_params_cache_dir: Optional[Path] = None,
    target_name_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    For each probe entry: if param_grid is present, tune then run (best estimator)
    and write tuning + run artifacts (including statistical_tests in summary).
    If param_grid is missing or empty, run_probe only with the given estimator.
    """
    target_name = target_name_override or Path(prob_config.get("target_file", TARGET_FILE)).stem
    layer = prob_config.layer_num
    all_runs: List[Dict[str, Any]] = []

    for entry in probe_entries:
        name = entry["name"]
        estimator = entry["estimator"]
        param_grid = entry.get("param_grid")
        exp_dirs = get_exp_dirs(
            prob_config.output_dir,
            target=target_name,
            prob_model=name,
            layer_num=layer,
        )

        if param_grid:
            search, metrics, _ = run_cv_search(
                X, y, estimator, param_grid,
                n_splits=N_SPLITS_CV,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                n_jobs=n_jobs,
                exp_dirs=exp_dirs,
                model_name=name,
                run_stats=True,
                reuse_best_params=reuse_best_params,
                best_params_cache_dir=best_params_cache_dir,
            )
            run_dict = {
                "experiment": f"{target_name}_{name}",
                "layer": layer,
                **metrics,
                **search.best_params_,
            }
        else:
            metrics, _, _ = run_probe(
                X, y, estimator,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                exp_dirs=exp_dirs,
                model_name=name,
                run_stats=True,
            )
            run_dict = {
                "experiment": f"{target_name}_{name}",
                "layer": layer,
                **metrics,
            }
        all_runs.append(run_dict)

    return all_runs


def linear_models(
    prob_config: cfg.Config,
    X: np.ndarray,
    y: np.ndarray,
    n_jobs: int = -1,
    reuse_best_params: bool = False,
    best_params_cache_dir: Optional[Path] = None,
    target_name_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run all registered linear probes. Use LINEAR_PROBES in prob_models to add more."""
    if n_jobs == -1:
        n_jobs = _cpu_budget()
    return run_probes(
        prob_config,
        X,
        y,
        LINEAR_PROBES,
        n_jobs,
        reuse_best_params=reuse_best_params,
        best_params_cache_dir=best_params_cache_dir,
        target_name_override=target_name_override,
    )


def linear_models_shuffled_ident_baseline(
    prob_config: cfg.Config,
    X: np.ndarray,
    *,
    n_jobs: int = -1,
    random_state: int = RANDOM_STATE,
    baseline_tag: str = "shuffled_ident",
    target_file: Optional[str] = None,
    reuse_best_params: bool = False,
    best_params_cache_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Run linear probes on a shuffled-ident baseline target assignment.

    This permutes id->target mapping before loading y, producing a random-label
    control with the same marginal target distribution.
    """
    if n_jobs == -1:
        n_jobs = _cpu_budget()

    target_file = target_file or prob_config.get("target_file", TARGET_FILE)
    y_shuffled = load_y_by_ids(
        prob_config.output_dir,
        target_dir=prob_config.target_dir,
        targets_file=target_file,
        shuffle_idents=True,
        random_state=random_state,
    )
    baseline_target_name = f"{Path(target_file).stem}_{baseline_tag}"

    return linear_models(
        prob_config,
        X,
        y_shuffled,
        n_jobs=n_jobs,
        reuse_best_params=reuse_best_params,
        best_params_cache_dir=best_params_cache_dir,
        target_name_override=baseline_target_name,
    )


def non_linear_models(
    prob_config: cfg.Config,
    X: np.ndarray,
    y: np.ndarray,
    n_jobs: int = -1,
    reuse_best_params: bool = False,
    best_params_cache_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Run all registered non-linear probes. Use NONLINEAR_PROBES in prob_models to add more."""
    if n_jobs == -1:
        n_jobs = _cpu_budget()
    return run_probes(
        prob_config,
        X,
        y,
        NONLINEAR_PROBES,
        n_jobs,
        reuse_best_params=reuse_best_params,
        best_params_cache_dir=best_params_cache_dir,
    )


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────


def main(prob_config: cfg.Config) -> List[Dict[str, Any]]:
    """
    Load targets and layer representations, run linear (and optionally non-linear)
    probes per layer, write per-model artifacts and a single summary CSV.
    """
    all_runs: List[Dict[str, Any]] = []
    baseline_runs: List[Dict[str, Any]] = []
    layer_nums = [1, 2, 3]
    target_file = prob_config.get("target_file", TARGET_FILE)

    def _flag(env_key: str, cfg_key: str, default: bool) -> bool:
        raw = os.getenv(env_key)
        if raw is not None:
            return raw.lower() in {"1", "true", "yes"}
        return bool(int(prob_config.get(cfg_key, int(default))))

    reuse_best_params = _flag("PROB_REUSE_BEST_PARAMS", "reuse_best_params", True)
    best_params_cache_dir_env = os.getenv("PROB_BEST_PARAMS_CACHE_DIR")
    best_params_cache_dir = (
        Path(best_params_cache_dir_env) if best_params_cache_dir_env else None
    )
    run_shuffled_baseline = _flag("PROB_RUN_SHUFFLED_BASELINE", "run_shuffled_baseline", True)
    run_non_linear_models = _flag("PROB_RUN_NON_LINEAR_MODELS", "run_non_linear_models", False)
    baseline_tag = str(
        prob_config.get(
            "baseline_tag",
            os.getenv("PROB_BASELINE_TAG", "shuffled_ident"),
        )
    )

    sub_file = _ROOT / "prob" / "cluster" / "run_prob.sub"
    n_jobs = _cpu_budget(sub_file=sub_file)

    y = load_y_by_ids(
        prob_config.output_dir,
        target_dir=prob_config.target_dir,
        targets_file=target_file,
    )
    target_name = Path(target_file).stem
    baseline_target_name = f"{target_name}_{baseline_tag}"

    if run_shuffled_baseline:
        y_shuffled = load_y_by_ids(
            prob_config.output_dir,
            target_dir=prob_config.target_dir,
            targets_file=target_file,
            shuffle_idents=True,
            random_state=RANDOM_STATE,
        )

    for layer in layer_nums:
        X = load_X_from_pt(prob_config.output_dir, layer_num=layer)
        prob_config.update({"layer_num": layer}, allow_duplicates=True)

        all_runs.extend(
            linear_models(
                prob_config,
                X,
                y,
                n_jobs=n_jobs,
                reuse_best_params=reuse_best_params,
                best_params_cache_dir=best_params_cache_dir,
            )
        )
        if run_non_linear_models:
            all_runs.extend(
                non_linear_models(
                    prob_config,
                    X,
                    y,
                    n_jobs=n_jobs,
                    reuse_best_params=reuse_best_params,
                    best_params_cache_dir=best_params_cache_dir,
                )
            )

        if run_shuffled_baseline:
            baseline_runs.extend(
                linear_models(
                    prob_config,
                    X,
                    y_shuffled,
                    n_jobs=n_jobs,
                    reuse_best_params=reuse_best_params,
                    best_params_cache_dir=best_params_cache_dir,
                    target_name_override=baseline_target_name,
                )
            )

    # Single summary CSV after all layers
    summary_df = pd.DataFrame(all_runs)
    summary_dir = Path(prob_config.output_dir) / target_name / "experiments"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = summary_dir / "summary_runs.csv"
    summary_df.to_csv(summary_csv, index=False)

    if baseline_runs:
        baseline_summary_df = pd.DataFrame(baseline_runs)
        baseline_summary_dir = Path(prob_config.output_dir) / baseline_target_name / "experiments"
        baseline_summary_dir.mkdir(parents=True, exist_ok=True)
        baseline_summary_csv = baseline_summary_dir / "summary_runs.csv"
        baseline_summary_df.to_csv(baseline_summary_csv, index=False)

    return all_runs


if __name__ == "__main__":
    ds_load_config = get_ds_load_config()
    target_file = ds_load_config.get("target_file", TARGET_FILE)

    wandb.init(
        project="probing",
        name=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            **ds_load_config,
            "random_state": RANDOM_STATE,
            "target_file": target_file,
        },
    )
    main(ds_load_config)
