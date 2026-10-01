"""
Orchestrate probing experiments: load X/y, run registered linear and non-linear
probes per layer, and aggregate results.

To add a probe: append an entry to LINEAR_PROBES or NONLINEAR_PROBES in
prob_models (or pass a custom list to run_probes). Statistical analysis
lives in prob_stats; metrics in prob_metrics; CV pipeline in prob_run.
"""
from __future__ import annotations

import inspect
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import kinodata.configuration as cfg

from prob.paths_and_io import (
    EXP_DIR_REPORTS,
    get_exp_dirs,
    load_fold_index,
    load_out_tensor,
    load_X_from_pt,
    load_y_by_ids,
)
from prob import prob_models, prob_run, run_manifest
from prob.prob_config import get_ds_load_config
from prob.prob_run import load_best_params_by_fold, run_cv_search, run_probe_per_checkpoint

# Probe registry: add entries here to run new linear or non-linear probes.
# For metrics/stats use prob.prob_metrics and prob.prob_stats.
from prob.prob_models import LINEAR_PROBES, NONLINEAR_PROBES

import wandb


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

RANDOM_STATE = 96
# Inner CV folds of the per-checkpoint grid search (hyperparameter choice only;
# test metrics come from the fixed probe split).
N_SPLITS_CV = 3
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


def _resolve_layer_nums(prob_config: cfg.Config) -> List[int]:
    """
    Which layers to probe, in order.

    By default: whatever aggregated representations actually exist in output_dir.
    A model whose `layer_0.pt` has been aggregated gets layer 0 probed with no
    config change; one whose hasn't is simply left out instead of crashing. DTI's
    per-tower artifacts (`ligand_layer_*.pt` / `pocket_layer_*.pt`) deliberately do
    not match the glob -- only the joint `layer_*.pt` representations are probed.

    PROB_LAYERS overrides this with an explicit comma-separated list. That is how
    you backfill a single layer into a target that has already been probed, without
    recomputing the rest: PROB_LAYERS=0.
    """
    override = os.getenv("PROB_LAYERS", "")
    if override.strip():
        return [int(part) for part in override.split(",") if part.strip()]

    layer_nums = sorted(
        int(path.stem.split("_")[1])
        for path in Path(prob_config.output_dir).glob("layer_*.pt")
    )
    if not layer_nums:
        raise FileNotFoundError(
            f"No aggregated layer_*.pt found under {prob_config.output_dir}. "
            "Run the extraction (prob/run_extraction.py) for this model first."
        )
    return layer_nums


# ─────────────────────────────────────────────────────────────
# Probe runner (uses registry)
# ─────────────────────────────────────────────────────────────


def _source_best_params_file(
    prob_config: cfg.Config, target: str, prob_model: str, layer: int
) -> Path:
    """Where `target`'s run of `prob_model` at `layer` saved its best params."""
    dirs = get_exp_dirs(
        prob_config.output_dir, target=target, prob_model=prob_model, layer_num=layer, create=False,
    )
    return dirs[EXP_DIR_REPORTS] / f"{prob_model}_best_params.json"


def _log_run(run_dict: Dict[str, Any]) -> Dict[str, Any]:
    if wandb.run is not None:
        wandb.log(run_dict)
    return run_dict


def run_probes(
    prob_config: cfg.Config,
    X: np.ndarray,
    y: np.ndarray,
    idents: np.ndarray,
    folds: np.ndarray,
    probe_entries: List[Dict[str, Any]],
    n_jobs: int,
    layer_num: int,
    reuse_best_params: bool = False,
    best_params_cache_dir: Optional[Path] = None,
    target_name_override: Optional[str] = None,
    params_from_target: Optional[str] = None,
    per_ckpt: bool = True,
    pooled: bool = False,
    run_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    For each probe entry, train one probe per GNN checkpoint (see
    prob_run.run_probe_per_checkpoint): tuned with GridSearchCV per fold if the
    entry has a param_grid, else fit as given. Writes one experiment per probe
    and layer, with per-fold results inside.

    `idents` and `folds` are aligned row-for-row with X and y: the probe
    train/test split is looked up by ident, and `folds` says which checkpoint
    (CV fold) produced each row. reuse_best_params shares each fold's tuned
    params across the layers of a probe instead of tuning every layer.

    params_from_target skips tuning and fits every fold with the params tuned
    for that target (same probe, same layer), which must have run already. The
    shuffled-ident baseline uses it with the real target, so the control runs
    the same probe as the real task.

    per_ckpt / pooled choose what runs (PROB_PER_CKPT / PROB_POOLED in main).
    pooled also fits one probe over all folds' rows (prob_run.run_cv_search), the
    pre-per-checkpoint setup, kept for comparison. It is written as probe
    "<name>_pooled" next to "<name>", so the explorer shows it as its own probe.

    run_id (the run manifest's id, see prob.run_manifest) is written into every
    probe summary JSON and summary row, linking each result to its manifest.
    """
    run_fields = {"run_id": run_id} if run_id else {}
    target_name = target_name_override or Path(prob_config.get("target_file", TARGET_FILE)).stem
    layer = layer_num
    all_runs: List[Dict[str, Any]] = []

    for entry in probe_entries:
        name = entry["name"]
        estimator = entry["estimator"]
        param_grid = entry.get("param_grid")
        entry_n_jobs = entry.get("n_jobs") or n_jobs

        if per_ckpt:
            exp_dirs = get_exp_dirs(
                prob_config.output_dir, target=target_name, prob_model=name, layer_num=layer,
            )
            # One line per experiment, so the .out file shows progress and the exact
            # directory each probe's artifacts/reports/figures are written to.
            print(f"[prob] {name} layer={layer} -> {exp_dirs['root']}", flush=True)

            fixed_params, fixed_source = None, None
            if params_from_target is not None:
                fixed_source = _source_best_params_file(prob_config, params_from_target, name, layer)
                fixed_params = load_best_params_by_fold(fixed_source, param_grid)
                if not fixed_params:
                    raise FileNotFoundError(
                        f"No per-fold best params for {name} layer {layer} of target "
                        f"'{params_from_target}' at {fixed_source} (tuned over the current grid); "
                        "run the real target before its baseline"
                    )

            _, metrics, _ = run_probe_per_checkpoint(
                X, y, estimator, param_grid,
                idents=idents,
                folds=folds,
                n_splits=N_SPLITS_CV,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                n_jobs=entry_n_jobs,
                exp_dirs=exp_dirs,
                model_name=name,
                run_stats=True,
                share_best_params_across_layers=reuse_best_params,
                best_params_cache_dir=best_params_cache_dir,
                fixed_best_params=fixed_params,
                fixed_best_params_source=str(fixed_source) if fixed_source is not None else None,
                extra_summary=run_fields,
            )
            across = metrics.pop("across_checkpoints")
            all_runs.append(_log_run({
                "experiment": f"{target_name}_{name}",
                "layer": layer,
                **run_fields,
                **metrics,
                **{f"{m}_ckpt_{stat}": v for m, st in across.items() for stat, v in st.items()},
            }))

        if pooled:
            pooled_name = f"{name}_pooled"
            exp_dirs = get_exp_dirs(
                prob_config.output_dir, target=target_name, prob_model=pooled_name, layer_num=layer,
            )
            print(f"[prob] {pooled_name} layer={layer} -> {exp_dirs['root']}", flush=True)

            fixed_params, fixed_source = None, None
            if params_from_target is not None:
                fixed_source = _source_best_params_file(prob_config, params_from_target, pooled_name, layer)
                if not fixed_source.exists():
                    raise FileNotFoundError(
                        f"No best params for {pooled_name} layer {layer} of target "
                        f"'{params_from_target}' at {fixed_source}; run the real target before its baseline"
                    )
                fixed_params = json.loads(fixed_source.read_text())

            search, metrics, _ = run_cv_search(
                X, y, estimator, param_grid or {},
                idents=idents,
                n_splits=N_SPLITS_CV,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                n_jobs=entry_n_jobs,
                exp_dirs=exp_dirs,
                model_name=pooled_name,
                run_stats=True,
                reuse_best_params=reuse_best_params,
                best_params_cache_dir=best_params_cache_dir,
                fixed_best_params=fixed_params,
                fixed_best_params_source=str(fixed_source) if fixed_source is not None else None,
                extra_summary=run_fields,
            )
            all_runs.append(_log_run({
                "experiment": f"{target_name}_{pooled_name}",
                "layer": layer,
                **run_fields,
                **metrics,
            }))

    return all_runs


def linear_models(
    prob_config: cfg.Config,
    X: np.ndarray,
    y: np.ndarray,
    idents: np.ndarray,
    folds: np.ndarray,
    layer_num: int,
    n_jobs: int = -1,
    reuse_best_params: bool = False,
    best_params_cache_dir: Optional[Path] = None,
    target_name_override: Optional[str] = None,
    params_from_target: Optional[str] = None,
    per_ckpt: bool = True,
    pooled: bool = False,
    run_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run all registered linear probes. Use LINEAR_PROBES in prob_models to add more."""
    if n_jobs == -1:
        n_jobs = _cpu_budget()
    return run_probes(
        prob_config,
        X,
        y,
        idents,
        folds,
        LINEAR_PROBES,
        n_jobs,
        layer_num=layer_num,
        reuse_best_params=reuse_best_params,
        best_params_cache_dir=best_params_cache_dir,
        target_name_override=target_name_override,
        params_from_target=params_from_target,
        per_ckpt=per_ckpt,
        pooled=pooled,
        run_id=run_id,
    )


def linear_models_shuffled_ident_baseline(
    prob_config: cfg.Config,
    X: np.ndarray,
    idents: np.ndarray,
    folds: np.ndarray,
    layer_num: int,
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
    control with the same marginal target distribution. `idents` are the real
    (unshuffled) row idents from ids.pt: only y is shuffled, X rows keep theirs.
    Uses the real target's tuned params, so the real target must have run first.
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
        idents,
        folds,
        layer_num=layer_num,
        n_jobs=n_jobs,
        reuse_best_params=reuse_best_params,
        best_params_cache_dir=best_params_cache_dir,
        target_name_override=baseline_target_name,
        params_from_target=Path(target_file).stem,
    )


def _select_probes(
    probe_entries: List[Dict[str, Any]], names_csv: str
) -> List[Dict[str, Any]]:
    """Filter probe_entries by a comma-separated list of names (e.g. "mlp,random_forest").

    Empty/unset names_csv means "run all". Raises if the filter matches nothing, so a
    typo'd name fails loudly instead of silently running zero probes.
    """
    names = {name.strip() for name in names_csv.split(",") if name.strip()}
    if not names:
        return probe_entries
    filtered = [entry for entry in probe_entries if entry["name"] in names]
    if not filtered:
        available = [entry["name"] for entry in probe_entries]
        raise ValueError(f"No probes matched {sorted(names)}; available: {available}")
    return filtered


def non_linear_models(
    prob_config: cfg.Config,
    X: np.ndarray,
    y: np.ndarray,
    idents: np.ndarray,
    folds: np.ndarray,
    layer_num: int,
    n_jobs: int = -1,
    reuse_best_params: bool = False,
    best_params_cache_dir: Optional[Path] = None,
    target_name_override: Optional[str] = None,
    probe_entries: Optional[List[Dict[str, Any]]] = None,
    params_from_target: Optional[str] = None,
    per_ckpt: bool = True,
    pooled: bool = False,
    run_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run registered non-linear probes (all of NONLINEAR_PROBES, or probe_entries if given)."""
    if n_jobs == -1:
        n_jobs = _cpu_budget()
    return run_probes(
        prob_config,
        X,
        y,
        idents,
        folds,
        probe_entries if probe_entries is not None else NONLINEAR_PROBES,
        n_jobs,
        layer_num=layer_num,
        reuse_best_params=reuse_best_params,
        best_params_cache_dir=best_params_cache_dir,
        target_name_override=target_name_override,
        params_from_target=params_from_target,
        per_ckpt=per_ckpt,
        pooled=pooled,
        run_id=run_id,
    )


def _write_summary(runs: List[Dict[str, Any]], summary_dir: Path) -> Path:
    """
    Write summary_runs.csv, merging with whatever is already there.

    A run that probes only some layers -- PROB_LAYERS=0 to backfill layer 0 into a
    target that was already probed for layers 1..3 -- must not erase the rows for
    the layers it did not touch. Rows are keyed by (experiment, layer): re-running
    a pair replaces its row, every other row is kept.
    """
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = summary_dir / "summary_runs.csv"
    summary_df = pd.DataFrame(runs)

    if summary_csv.exists():
        previous = pd.read_csv(summary_csv)
        keys = ["experiment", "layer"]
        if set(keys) <= set(previous.columns) and set(keys) <= set(summary_df.columns):
            replaced = set(zip(summary_df["experiment"], summary_df["layer"]))
            keep = [
                pair not in replaced
                for pair in zip(previous["experiment"], previous["layer"])
            ]
            summary_df = pd.concat([previous[keep], summary_df], ignore_index=True)
            summary_df = summary_df.sort_values(["layer", "experiment"]).reset_index(drop=True)
        else:
            summary_df = pd.concat([previous, summary_df], ignore_index=True)

    summary_df.to_csv(summary_csv, index=False)
    return summary_csv


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────


def _run_constants() -> Dict[str, Any]:
    """Fixed choices that live in code rather than in any config, for the run manifest."""
    defaults = inspect.signature(run_probe_per_checkpoint).parameters
    return {
        "seeds": {
            "probe_estimators": prob_models.RANDOM_STATE,
            "inner_cv_shuffle": RANDOM_STATE,
            "bootstrap": RANDOM_STATE,
            "baseline_permutation": RANDOM_STATE,
            "probe_split_if_rebuilt": prob_run.DEFAULT_PROBE_SPLIT_SEED,
        },
        "inner_cv_folds": N_SPLITS_CV,
        "refit_metric": prob_run.DEFAULT_REFIT,
        "probe_test_size_if_split_rebuilt": TEST_SIZE,
        "bootstrap": {
            "n": defaults["n_bootstrap"].default,
            "confidence": defaults["confidence"].default,
            "resampled": "probe test rows",
        },
        "max_test_fraction_deviation": {
            "pooled": prob_run.MAX_TEST_FRACTION_DEVIATION,
            "per_checkpoint": prob_run.MAX_TEST_FRACTION_DEVIATION_PER_FOLD,
        },
        "feature_scaling": "StandardScaler inside the probe pipeline, fit on probe-train rows",
    }


def _cpu_budget_source(sub_file: Path) -> str:
    """Which input _cpu_budget took its value from (same order as _cpu_budget)."""
    for key in ("SLURM_CPUS_PER_TASK", "NSLOTS", "OMP_NUM_THREADS"):
        if key in os.environ and os.environ[key].isdigit():
            return f"env {key}"
    if sub_file.exists() and re.search(r"request_CPUs\s*=\s*(\d+)", sub_file.read_text(), flags=re.I):
        return f"request_CPUs in {sub_file}"
    return "code default"


def _extraction_spec(output_dir: Path) -> Optional[Dict[str, Any]]:
    """The extraction manifest's spec and date, copied so the probe manifest reads alone."""
    path = Path(output_dir) / "manifest.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"unreadable": str(path)}
    return {
        "created_at": payload.get("created_at"),
        "git_commit": payload.get("git_commit"),
        "spec": payload.get("spec"),
    }


def main(prob_config: cfg.Config, use_wandb: bool = False) -> List[Dict[str, Any]]:
    """
    Load targets and layer representations, run linear (and optionally non-linear)
    probes per layer, write per-model artifacts and a single summary CSV.

    Also writes a run manifest (see prob.run_manifest) under
    <target>/experiments/run_manifests/<run_id>.json: every resolved setting and
    where it came from, the seeds and fixed constants, the probes and grids, and
    the sha256 of every input file. Every summary this run writes carries run_id.
    """
    started = time.time()
    run_id = run_manifest.new_run_id()
    all_runs: List[Dict[str, Any]] = []
    baseline_runs: List[Dict[str, Any]] = []
    layer_nums = _resolve_layer_nums(prob_config)
    target_file = prob_config.get("target_file", TARGET_FILE) or None
    if not target_file:
        raise ValueError(
            "target_file must be set (e.g. --target_file affinity.pt). "
            "Check that it is passed as a CLI argument and registered in prob_config defaults as a str."
        )

    # Every run setting with its resolved value and where that value came from.
    settings: Dict[str, Dict[str, Any]] = {}

    def _flag(env_key: str, cfg_key: str, default: bool) -> bool:
        raw = os.getenv(env_key)
        if raw is not None:
            value, source = raw.lower() in {"1", "true", "yes"}, f"env {env_key}={raw!r}"
        elif cfg_key in prob_config:
            value, source = bool(int(prob_config[cfg_key])), f"config {cfg_key}"
        else:
            value, source = default, "code default"
        settings[cfg_key] = {"value": value, "source": source}
        return value

    # 1 = share each fold's tuned params across layers (faster, but deeper layers
    # are then probed with params tuned on the first layer). Default: tune every layer.
    reuse_best_params = _flag("PROB_REUSE_BEST_PARAMS", "reuse_best_params", False)
    best_params_cache_dir_env = os.getenv("PROB_BEST_PARAMS_CACHE_DIR")
    best_params_cache_dir = (
        Path(best_params_cache_dir_env) if best_params_cache_dir_env else None
    )
    settings["best_params_cache_dir"] = {
        "value": best_params_cache_dir,
        "source": "env PROB_BEST_PARAMS_CACHE_DIR" if best_params_cache_dir_env
        else "code default (<target>/<probe>/shared_best_params; used only with reuse_best_params)",
    }
    run_shuffled_baseline = _flag("PROB_RUN_SHUFFLED_BASELINE", "run_shuffled_baseline", True)
    # One probe per GNN checkpoint and one pooled probe over all folds; both on by default.
    run_per_ckpt = _flag("PROB_PER_CKPT", "per_ckpt", True)
    run_pooled = _flag("PROB_POOLED", "pooled", True)
    if not (run_per_ckpt or run_pooled):
        raise ValueError("PROB_PER_CKPT and PROB_POOLED are both off; nothing to run")
    run_linear_models = _flag("PROB_RUN_LINEAR_MODELS", "run_linear_models", True)
    run_non_linear_models = _flag("PROB_RUN_NON_LINEAR_MODELS", "run_non_linear_models", False)
    nonlinear_models_csv = os.getenv("PROB_NONLINEAR_MODELS", "")
    nonlinear_probe_entries = _select_probes(NONLINEAR_PROBES, nonlinear_models_csv)
    settings["nonlinear_models"] = {
        "value": [e["name"] for e in nonlinear_probe_entries],
        "source": f"env PROB_NONLINEAR_MODELS={nonlinear_models_csv!r}" if nonlinear_models_csv.strip()
        else "code default (all registered)",
    }
    settings["layers"] = {
        "value": layer_nums,
        "source": f"env PROB_LAYERS={os.getenv('PROB_LAYERS')!r}" if os.getenv("PROB_LAYERS", "").strip()
        else "discovered: every aggregated layer_*.pt in output_dir",
    }

    # Non-linear probe estimators (TorchMLPRegressor, HybridRandomForestRegressor)
    # expose a `.device` attribute; move them onto prob_config.device. On cuda,
    # cap each one's GridSearchCV to n_jobs=1 -- parallel worker processes would
    # each open their own CUDA context on the same GPU and contend for memory
    # instead of speeding anything up.
    device = str(prob_config.get("device", "cpu")) or "cpu"
    settings["device"] = {"value": device, "source": "config device"}
    for entry in nonlinear_probe_entries:
        if hasattr(entry["estimator"], "device"):
            entry["estimator"].device = device
            if device == "cuda":
                entry["n_jobs"] = 1
    baseline_tag = str(
        prob_config.get(
            "baseline_tag",
            os.getenv("PROB_BASELINE_TAG", "shuffled_ident"),
        )
    )
    settings["baseline_tag"] = {
        "value": baseline_tag,
        "source": "config baseline_tag" if "baseline_tag" in prob_config
        else ("env PROB_BASELINE_TAG" if os.getenv("PROB_BASELINE_TAG") else "code default"),
    }

    sub_file = _ROOT / "prob" / "cluster" / "run_prob.sub"
    n_jobs = _cpu_budget(sub_file=sub_file)
    settings["n_jobs"] = {"value": n_jobs, "source": _cpu_budget_source(sub_file)}

    # Echo the resolved run identity + paths up front, so the condor .out file
    # records which model's representations this job actually read and where it
    # will write -- flush=True because stdout redirected to a file is block
    # buffered and would otherwise show nothing until the job ends.
    target_name = Path(target_file).stem
    baseline_target_name = f"{target_name}_{baseline_tag}"
    print(
        "[prob] run: "
        f"gnn={prob_config.get('gnn_model_type')} "
        f"rmsd={prob_config.get('filter_rmsd_max_value')} "
        f"split={prob_config.get('split_type')} "
        f"target={target_name} layers={layer_nums}\n"
        f"[prob] X / results root : {prob_config.output_dir}\n"
        f"[prob] y (targets) from : {Path(prob_config.target_dir) / target_file}\n"
        f"[prob] probes: linear={run_linear_models} "
        f"non_linear={run_non_linear_models} "
        f"({[e['name'] for e in nonlinear_probe_entries] if run_non_linear_models else []}) "
        f"baseline={run_shuffled_baseline} per_ckpt={run_per_ckpt} pooled={run_pooled} n_jobs={n_jobs}",
        flush=True,
    )

    if use_wandb:
        target_name_for_run = Path(target_file).stem
        tags = [target_name_for_run, str(prob_config.get("gnn_model_type", ""))]
        if run_linear_models:
            tags += [entry["name"] for entry in LINEAR_PROBES]
        if run_non_linear_models:
            tags += [entry["name"] for entry in nonlinear_probe_entries]
        if run_shuffled_baseline:
            tags.append(baseline_tag)
        wandb.init(
            project="probing",
            name=f"{target_name_for_run}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags=[tag for tag in tags if tag],
            config={**prob_config, "random_state": RANDOM_STATE, "run_id": run_id},
        )

    # valid_mask marks non-NaN targets; index X and y by it together (never y
    # alone) to drop invalid rows without desyncing the two.
    y, valid_mask = load_y_by_ids(
        prob_config.output_dir,
        target_dir=prob_config.target_dir,
        targets_file=target_file,
        return_mask=True,
    )

    # Real idents of the rows of X (same order as ids.pt, which load_y_by_ids and
    # load_X_from_pt both follow). The probe train/test split is looked up by these,
    # so they must be masked exactly like X and y. The shuffled baseline also uses
    # them unshuffled: only its y is permuted, X rows keep their own idents.
    idents = load_out_tensor(prob_config.output_dir, "ids.pt").detach().cpu().numpy().astype(int)
    # Which GNN checkpoint (CV fold) produced each row: one probe is trained per checkpoint.
    folds = load_fold_index(prob_config.output_dir)

    if run_shuffled_baseline:
        y_shuffled, valid_mask_shuffled = load_y_by_ids(
            prob_config.output_dir,
            target_dir=prob_config.target_dir,
            targets_file=target_file,
            shuffle_idents=True,
            random_state=RANDOM_STATE,
            return_mask=True,
        )

    # ── Run manifest: written now (status "running") and again at the end. ──
    output_dir = Path(prob_config.output_dir)
    manifest_dirs = [output_dir / target_name / "experiments"]
    if run_shuffled_baseline:
        manifest_dirs.append(output_dir / baseline_target_name / "experiments")
    probe_split_path = prob_run.default_probe_split_path()
    probe_split_existed = probe_split_path.is_file()

    def _input_files() -> Dict[str, Any]:
        return {
            "extraction_manifest": run_manifest.file_record(output_dir / "manifest.json"),
            "ids": run_manifest.file_record(output_dir / "ids.pt"),
            "layers": {str(n): run_manifest.file_record(output_dir / f"layer_{n}.pt") for n in layer_nums},
            "target": run_manifest.file_record(Path(prob_config.target_dir) / target_file),
            "probe_split": {
                **run_manifest.file_record(probe_split_path),
                "existed_before_run": probe_split_existed,
            },
        }

    def _rows_by_fold(mask: np.ndarray) -> Dict[str, int]:
        return {str(int(k)): int(mask[folds == k].sum()) for k in np.unique(folds)}

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "started_at": run_manifest.utc_now(),
        "host": run_manifest.run_host(),
        "request": {
            "gnn_model_type": prob_config.get("gnn_model_type"),
            "filter_rmsd_max_value": prob_config.get("filter_rmsd_max_value"),
            "split_type": prob_config.get("split_type"),
            "target_file": target_file,
            "output_dir": output_dir,
            "target_dir": prob_config.target_dir,
        },
        "settings": settings,
        "constants": _run_constants(),
        "probes": {
            "linear": [run_manifest.probe_record(e, n_jobs) for e in LINEAR_PROBES] if run_linear_models else [],
            "non_linear": [run_manifest.probe_record(e, n_jobs) for e in nonlinear_probe_entries]
            if run_non_linear_models else [],
        },
        "data": {
            "rows_in_ids": int(len(idents)),
            "rows_with_target": int(valid_mask.sum()),
            "rows_dropped_no_target": int((~valid_mask).sum()),
            "rows_with_target_by_fold": _rows_by_fold(valid_mask),
            "baseline_rows_with_target": int(valid_mask_shuffled.sum()) if run_shuffled_baseline else None,
        },
        "extraction": _extraction_spec(output_dir),
        "inputs": _input_files(),
        "code": {"git": run_manifest.git_info(_ROOT), "versions": run_manifest.package_versions()},
        "env": run_manifest.recorded_env(),
        "config": dict(prob_config),
    }
    manifest_paths = run_manifest.write_manifest(manifest, manifest_dirs)
    print(f"[prob] run manifest: {manifest_paths[0]}", flush=True)

    try:
        for layer in layer_nums:
            X = load_X_from_pt(prob_config.output_dir, layer_num=layer)

            if run_linear_models:
                all_runs.extend(
                    linear_models(
                        prob_config,
                        X[valid_mask],
                        y[valid_mask],
                        idents[valid_mask],
                        folds[valid_mask],
                        layer_num=layer,
                        n_jobs=n_jobs,
                        reuse_best_params=reuse_best_params,
                        best_params_cache_dir=best_params_cache_dir,
                        per_ckpt=run_per_ckpt,
                        pooled=run_pooled,
                        run_id=run_id,
                    )
                )
            if run_non_linear_models:
                all_runs.extend(
                    non_linear_models(
                        prob_config,
                        X[valid_mask],
                        y[valid_mask],
                        idents[valid_mask],
                        folds[valid_mask],
                        layer_num=layer,
                        n_jobs=n_jobs,
                        reuse_best_params=reuse_best_params,
                        best_params_cache_dir=best_params_cache_dir,
                        per_ckpt=run_per_ckpt,
                        pooled=run_pooled,
                        probe_entries=nonlinear_probe_entries,
                        run_id=run_id,
                    )
                )

            if run_shuffled_baseline:
                if run_linear_models:
                    baseline_runs.extend(
                        linear_models(
                            prob_config,
                            X[valid_mask_shuffled],
                            y_shuffled[valid_mask_shuffled],
                            idents[valid_mask_shuffled],
                            folds[valid_mask_shuffled],
                            layer_num=layer,
                            n_jobs=n_jobs,
                            reuse_best_params=reuse_best_params,
                            best_params_cache_dir=best_params_cache_dir,
                            per_ckpt=run_per_ckpt,
                            pooled=run_pooled,
                            target_name_override=baseline_target_name,
                            params_from_target=target_name,
                            run_id=run_id,
                        )
                    )
                if run_non_linear_models:
                    baseline_runs.extend(
                        non_linear_models(
                            prob_config,
                            X[valid_mask_shuffled],
                            y_shuffled[valid_mask_shuffled],
                            idents[valid_mask_shuffled],
                            folds[valid_mask_shuffled],
                            layer_num=layer,
                            n_jobs=n_jobs,
                            reuse_best_params=reuse_best_params,
                            best_params_cache_dir=best_params_cache_dir,
                            per_ckpt=run_per_ckpt,
                            pooled=run_pooled,
                            probe_entries=nonlinear_probe_entries,
                            target_name_override=baseline_target_name,
                            params_from_target=target_name,
                            run_id=run_id,
                        )
                    )

        # Single summary CSV after all layers
        summary_csv = _write_summary(
            all_runs, Path(prob_config.output_dir) / target_name / "experiments"
        )

        written = [f"[prob] DONE {len(all_runs)} run(s) -> {summary_csv}"]
        baseline_summary_csv = None

        if baseline_runs:
            baseline_summary_csv = _write_summary(
                baseline_runs,
                Path(prob_config.output_dir) / baseline_target_name / "experiments",
            )
            written.append(
                f"[prob] DONE {len(baseline_runs)} baseline run(s) -> {baseline_summary_csv}"
            )
    except BaseException as err:
        manifest.update(
            status="failed",
            finished_at=run_manifest.utc_now(),
            duration_seconds=round(time.time() - started, 1),
            error=f"{type(err).__name__}: {err}",
            results={"runs_finished": len(all_runs), "baseline_runs_finished": len(baseline_runs)},
            inputs=_input_files(),
        )
        run_manifest.write_manifest(manifest, manifest_dirs)
        raise

    manifest.update(
        status="finished",
        finished_at=run_manifest.utc_now(),
        duration_seconds=round(time.time() - started, 1),
        results={
            "runs": len(all_runs),
            "baseline_runs": len(baseline_runs),
            "summary_csv": summary_csv,
            "baseline_summary_csv": baseline_summary_csv,
            "experiments": sorted({(r["experiment"], r["layer"]) for r in all_runs + baseline_runs}),
        },
        # Re-hashed: the probe split may have been created during the run.
        inputs=_input_files(),
    )
    run_manifest.write_manifest(manifest, manifest_dirs)

    # Per-experiment artifacts/reports/figures live one level deeper, under
    # <target>/<probe>/<layer>/ -- run_probes prints each of those as it goes.
    written.append(f"[prob] all outputs under: {prob_config.output_dir}")
    print("\n".join(written), flush=True)

    return all_runs


if __name__ == "__main__":
    ds_load_config = get_ds_load_config()
    main(ds_load_config, use_wandb=True)
