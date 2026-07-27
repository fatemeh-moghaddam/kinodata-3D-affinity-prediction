"""
Regenerate parity/residual plots from already-saved prediction CSVs.

Configuration
-------------
Uses the exact same config plumbing as the probing pipeline. Invoke it with
the same CLI arguments `run_prob.sh` forwards to `prob_orchestrate.py`
(--gnn_model_type, --split_type, --filter_rmsd_max_value, --target_file, ...);
`get_ds_load_config()` builds the config and resolves `output_dir` via
`get_out_dir()`. Point a new `.sub` file at this script the same way
`run_prob.sub` points at `prob_orchestrate`.

The (target, prob_model, layer) experiments to visit mirror the pipeline: probe
names come from the LINEAR_PROBES / NONLINEAR_PROBES registries, layers are
[1, 2, 3], and each experiment's directories are resolved with `get_exp_dirs()`.
Experiments that were not run (no predictions CSV) are simply skipped.

Usage
-----
    python prob/replot_from_predictions.py \
        --split_type random-k-fold --filter_rmsd_max_value 2 \
        --gnn_model_type CGNN-3D --target_file affinity.pt
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import kinodata.configuration as cfg

from prob.paths_and_io import EXP_DIR_ARTIFACTS, EXP_DIR_FIGURES, get_exp_dirs
from prob.prob_config import get_ds_load_config
from prob.prob_models import LINEAR_PROBES, NONLINEAR_PROBES
from prob.prob_plots import plot_parity, plot_residuals

# Same layers the orchestrator probes; all registered probe names.
LAYER_NUMS = [1, 2, 3]
PROBE_NAMES = [entry["name"] for entry in (*LINEAR_PROBES, *NONLINEAR_PROBES)]


def replot_experiment(out_root: Path, target: str, prob_model: str, layer_num: int) -> bool:
    """Replot one experiment's parity + residual figures from its predictions CSV.

    Returns True if a CSV was found and plotted, False if the experiment was not
    run (no predictions CSV) so the caller can keep a count.
    """
    exp_dirs = get_exp_dirs(out_root, target=target, prob_model=prob_model, layer_num=layer_num)
    artifacts_dir = exp_dirs[EXP_DIR_ARTIFACTS]
    figures_dir = exp_dirs[EXP_DIR_FIGURES]

    csv_path = artifacts_dir / f"{prob_model}_predictions.csv"
    if not csv_path.exists():
        return False

    df = pd.read_csv(csv_path)
    if not {"y_true", "y_pred"}.issubset(df.columns):
        print(f"[skip] {csv_path} missing y_true/y_pred columns ({list(df.columns)})")
        return False

    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()

    plot_parity(
        y_true, y_pred,
        title=f"{prob_model} Parity — {target}, layer {layer_num}",
        save_path=figures_dir / f"{prob_model}_parity.png",
        show=False,
    )
    plot_residuals(
        y_true, y_pred,
        title=f"{prob_model} Residuals — {target}, layer {layer_num}",
        save_path=figures_dir / f"{prob_model}_residuals.png",
        show=False,
    )
    print(f"[ok] {target} / {prob_model} / layer {layer_num}  ->  {figures_dir}")
    return True


def main(prob_config: cfg.Config) -> int:
    """Replot every existing experiment for the config's target (and its baseline)."""
    out_root = Path(prob_config.output_dir)
    target_file = prob_config.get("target_file", None)
    if not target_file:
        raise ValueError(
            "target_file must be set (e.g. --target_file affinity.pt), matching "
            "how run_prob.sub forwards it to the probing pipeline."
        )

    target_name = Path(target_file).stem
    baseline_tag = str(prob_config.get("baseline_tag", "shuffled_ident"))
    targets = [target_name, f"{target_name}_{baseline_tag}"]

    total = 0
    for target in targets:
        for prob_model in PROBE_NAMES:
            for layer_num in LAYER_NUMS:
                if replot_experiment(out_root, target, prob_model, layer_num):
                    total += 1

    print(f"\nDone. Replotted {total} experiment(s) for target '{target_name}'.")
    return total


if __name__ == "__main__":
    ds_load_config = get_ds_load_config()
    main(ds_load_config)
