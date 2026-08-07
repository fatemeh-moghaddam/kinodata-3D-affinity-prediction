from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch






# _ROOT = Path(__file__).resolve().parent.parent
_ROOT = Path(os.environ.get("HOME_PROJ_DIR", Path(__file__).resolve().parents[1])) # to allow setting a different root via env variable


_MARKERS = ("setup.py", ".git", "README.md")

def _find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if any((p / m).exists() for m in _MARKERS):
            return p
    raise RuntimeError(f"Could not find project root from {start}")

@lru_cache(maxsize=1)
def get_project_root() -> Path:
    env = os.getenv("HOME_PROJ_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return _find_root(Path.cwd().resolve())

@lru_cache(maxsize=1)
def get_data_dir(prob: bool = True) -> Path:
    if prob:
        return get_project_root() / "data/probing"
    return get_project_root() / "data"
# ─────────────────────────────────────────────────────────────
# Directory helpers
# ─────────────────────────────────────────────────────────────
def get_model_dir(
                rmsd_threshold: int, 
                  split_type: str, 
                  split_fold: int, 
                  model_type: str,
                  root: Path = _ROOT
                  ) -> Path:
    p = root / "models" / f"rmsd_cutoff_{rmsd_threshold}" / split_type / str(split_fold) / model_type
    if not p.exists():
        p.mkdir(parents=True)
    return p


def get_model_ckpt(model_dir: Path) -> Path:
    cks = list(model_dir.glob("**/*.ckpt"))
    if not cks:
        raise FileNotFoundError(f"No .ckpt found under {model_dir}")
    return cks[0]


def get_gnn_config_path(model_dir: Path) -> Path:
    return model_dir / "config.json"


def get_split_file(
        split_type: str,
        split_fold: int,
        rmsd_threshold: int = None,
        k_fold: int = 5,
        root: Path = _ROOT
    ) -> Path:
    """ Get the path to the split file, which has a different pattern than model directory.
    Args:
        split_type (str): The type of split, e.g., "random-k-fold", "scaffold-k-fold".
        split_fold (int): The fold number.
        rmsd_threshold (int, optional): The RMSD threshold. Defaults to 0.
        k_fold (int, optional): Total number of folds. Defaults to 5.
        root (Path, optional): The root path. Defaults to the parent directory of this file.
    Returns:
        Path: The path to the split csv file.
    """
    p = root / "data/processed"
    if rmsd_threshold is None:
        p = p / split_type
    else:
        p = p / f"filter_predicted_rmsd_le{rmsd_threshold}.00" / split_type
    candidates = list(p.glob(f"{split_fold + 1}*{k_fold}.csv"))
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No split file found for fold {split_fold + 1}/{k_fold} in {p}"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Ambiguous split files for fold {split_fold + 1}/{k_fold} in {p}: {candidates}"
        )
    return candidates[0]


def get_out_dir(
        gnn_model_type: str,
        rmsd_threshold: int,
        split_type: str,
        split_fold: int | None,
        *,
        root: Path = _ROOT,
        create: bool = True,
    ) -> Path:
    """
    This is the output directory per fold.
    The concatenated one would be in the parent directory of this.
    And the layered separation would be in the naming.

    Pass create=False from read-only callers (plotting, analysis notebooks) so
    that merely asking for a path does not litter data/probing with empty dirs.
    """
    p = root / "data/probing" / gnn_model_type/ f"rmsd_cutoff_{rmsd_threshold}" / split_type
    if split_fold is not None:   # <- avoids dropping fold==0
        p = p / str(split_fold)
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


EXP_DIR_FIGURES = "figures"
EXP_DIR_ARTIFACTS = "artifacts"
EXP_DIR_REPORTS = "reports"


def get_exp_dirs(
        out_root: str | Path,
        target: str,
        prob_model: str,
        layer_num: int,
        *,
        create: bool = True,
    ) -> dict[str, Path]:
    """Create directories for storing results, figures, and artifacts.
    Experiments are created based on their X, y, and prob model.
    Returns a dict with paths.

    create=False just computes the paths (see get_out_dir).
    """
    exp_root = Path(out_root)  # e.g. ~/kinodata-3D-affinity-prediction/data/probing/CGNN-3D/rmsd_cutoff_2/random-k-fold
    exp_root = exp_root / target / prob_model / str(layer_num)
    dirs = {
        "root": exp_root,
        EXP_DIR_FIGURES: exp_root / EXP_DIR_FIGURES,
        EXP_DIR_ARTIFACTS: exp_root / EXP_DIR_ARTIFACTS,
        EXP_DIR_REPORTS: exp_root / EXP_DIR_REPORTS,
    }
    if create:
        for p in dirs.values():
            p.mkdir(parents=True, exist_ok=True)
    return dirs


# ─────────────────────────────────────────────────────────────
# Discovery: the inverse of get_out_dir() / get_exp_dirs()
# ─────────────────────────────────────────────────────────────
#
# get_out_dir/get_exp_dirs encode experimental factors *into* a path. Analysis
# code needs the other direction: walk what actually ran and decode the factors
# back out. Doing that with an ad-hoc glob + string split re-implements the
# layout a third time and silently drifts from it; these helpers read the same
# structure the writers produce.

PRED_SUFFIX = "_predictions.csv"
SUMMARY_SUFFIX = "_summary.json"

# <gnn>/rmsd_cutoff_<x>/<split>/<target>/<probe>/<layer>/artifacts/<probe>_predictions.csv
_CANONICAL_DEPTH = 8
_LEGACY_DEPTH = 7  # same, minus the <layer> level (a few early runs)


def _parse_rmsd(token: str) -> float | None:
    """'rmsd_cutoff_2' -> 2.0. Returns None if the token is not recognised."""
    if not token.startswith("rmsd_cutoff_"):
        return None
    value = token[len("rmsd_cutoff_"):]
    try:
        return float(value)
    except ValueError:
        return None


def _decode_pred_path(path: Path, data_dir: Path, baseline_tag: str) -> dict | None:
    """Turn one predictions CSV path back into the factors that produced it."""
    parts = path.relative_to(data_dir).parts
    if len(parts) == _CANONICAL_DEPTH:
        gnn, rmsd_tok, split_type, target_full, prob_model, layer_tok = parts[:6]
    elif len(parts) == _LEGACY_DEPTH:
        gnn, rmsd_tok, split_type, target_full, prob_model = parts[:5]
        layer_tok = None
    else:
        return None

    rmsd = _parse_rmsd(rmsd_tok)
    if rmsd is None:
        return None

    # The probe name is authoritative from the filename; the directory level is
    # a copy of it and has been out of sync in hand-moved runs.
    prob_model = path.name[: -len(PRED_SUFFIX)] or prob_model

    is_baseline = target_full.endswith(f"_{baseline_tag}")
    exp_root = path.parents[1]
    return {
        "gnn_model_type": gnn,
        "rmsd_threshold": rmsd,
        "split_type": split_type,
        "target": target_full[: -len(f"_{baseline_tag}")] if is_baseline else target_full,
        "target_full": target_full,
        "is_baseline": is_baseline,
        "prob_model": prob_model,
        "layer": int(layer_tok) if layer_tok is not None and layer_tok.isdigit() else None,
        "exp_root": exp_root,
        "predictions_path": path,
        "summary_path": exp_root / EXP_DIR_REPORTS / f"{prob_model}{SUMMARY_SUFFIX}",
        "figures_dir": exp_root / EXP_DIR_FIGURES,
    }


def find_probe_runs(
        *,
        gnn_model_type: str | Iterable[str] | None = None,
        rmsd_threshold: float | Iterable[float] | None = None,
        split_type: str | Iterable[str] | None = None,
        target: str | Iterable[str] | None = None,
        prob_model: str | Iterable[str] | None = None,
        layer: int | Iterable[int] | None = None,
        include_baselines: bool = True,
        baseline_tag: str = "shuffled_ident",
        root: Path | None = None,
    ) -> pd.DataFrame:
    """Index every probe run that has a saved predictions CSV on disk.

    One row per (gnn, rmsd, split, target, probe, layer) experiment, with the
    factors decoded from the path and the artifact paths attached. Nothing is
    read into memory beyond the index itself -- use `load_run_predictions` /
    `attach_run_metrics` on the rows you keep.

    Every keyword filter accepts a scalar or any iterable of values. Only
    experiments that actually ran show up; there is no need to enumerate probe
    registries or layer lists and check existence per candidate.
    """
    data_dir = Path(root) if root is not None else get_data_dir(prob=True)
    if not data_dir.exists():
        raise FileNotFoundError(f"No probing data directory at {data_dir}")

    # Anchored on the artifacts level: this excludes the model-level
    # data/probing/<gnn>/<rmsd>/<split>/predictions.csv (the GNN's own affinity
    # predictions), which is a different quantity and must not be mixed in.
    rows = [
        decoded
        for p in data_dir.glob(f"**/{EXP_DIR_ARTIFACTS}/*{PRED_SUFFIX}")
        if (decoded := _decode_pred_path(p, data_dir, baseline_tag)) is not None
    ]
    runs = pd.DataFrame(rows, columns=[
        "gnn_model_type", "rmsd_threshold", "split_type", "target", "target_full",
        "is_baseline", "prob_model", "layer", "exp_root", "predictions_path",
        "summary_path", "figures_dir",
    ])
    if runs.empty:
        return runs

    runs["layer"] = runs["layer"].astype("Int64")

    filters = {
        "gnn_model_type": gnn_model_type,
        "rmsd_threshold": rmsd_threshold,
        "split_type": split_type,
        "target": target,
        "prob_model": prob_model,
        "layer": layer,
    }
    for column, wanted in filters.items():
        if wanted is None:
            continue
        wanted = [wanted] if isinstance(wanted, (str, int, float)) else list(wanted)
        if column == "rmsd_threshold":
            wanted = [float(v) for v in wanted]
        runs = runs[runs[column].isin(wanted)]

    if not include_baselines:
        runs = runs[~runs["is_baseline"]]

    sort_cols = ["gnn_model_type", "rmsd_threshold", "split_type", "target",
                 "is_baseline", "prob_model", "layer"]
    return runs.sort_values(sort_cols).reset_index(drop=True)


def load_run_predictions(run) -> tuple[np.ndarray, np.ndarray]:
    """(y_true, y_pred) for one row of `find_probe_runs` (or a path to its CSV).

    Accepts a path, a dict, a Series row, or an itertuples namedtuple.
    """
    if isinstance(run, (str, Path)):
        path = Path(run)
    elif isinstance(run, Mapping):
        path = Path(run["predictions_path"])
    else:  # Series / namedtuple from .itertuples()
        path = Path(run.predictions_path)
    df = pd.read_csv(path)
    missing = {"y_true", "y_pred"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s) {sorted(missing)}")
    return df["y_true"].to_numpy(float), df["y_pred"].to_numpy(float)


def attach_run_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    """Add the metrics each run already wrote to reports/<probe>_summary.json.

    These are the numbers the pipeline itself reported (plus n_samples and the
    bootstrap CIs), so box plots built from them agree with the per-experiment
    figures by construction rather than by re-deriving r2 from the CSVs.
    Missing or unreadable summaries yield NaN columns for that row.
    """
    if runs.empty:
        return runs.copy()

    records = []
    for summary_path in runs["summary_path"]:
        record: dict[str, float] = {}
        path = Path(summary_path)
        if path.exists():
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError:
                payload = {}
            record.update(payload.get("metrics_on_unseen_data", {}))
            for key in ("n_samples", "n_test_samples", "n_features"):
                if key in payload:
                    record[key] = payload[key]
            for name, ci in (payload.get("statistical_tests") or {}).items():
                if isinstance(ci, dict) and "lower" in ci:
                    record[f"{name}_lower"] = ci["lower"]
                    record[f"{name}_upper"] = ci["upper"]
        records.append(record)

    metrics = pd.DataFrame(records, index=runs.index)
    return pd.concat([runs, metrics], axis=1)


# ─────────────────────────────────────────────────────────────
# Load/Save helpers
# ─────────────────────────────────────────────────────────────


def save_out_tensor(tensor: torch.Tensor, output_dir: Path, filename: str):
    """
    Save a tensor to a file in the output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / filename
    torch.save(tensor, out_file)
    return out_file

def load_out_tensor(output_dir: Path, filename: str) -> torch.Tensor:
    """
    Load a tensor from a file in the output directory.
    """
    out_file = output_dir / filename
    if not out_file.exists():
        raise FileNotFoundError(f"File {out_file} does not exist.")
    return torch.load(out_file)


def load_X_from_pt(
        in_dir: str | Path,
        layer_num: int = 1,
        ) -> np.ndarray:
    """ Load a representation tensor from a .pt file and return it as a Numpy array."""
    file_name = f"layer_{layer_num}.pt"
    X_tensor = load_out_tensor(in_dir, file_name)
    return X_tensor.detach().cpu().numpy()


def load_y_by_ids(
        in_dir: str | Path,
        target_dir: str | Path,
        targets_file: str = None,
        ids_file: str = "ids.pt",
        default_value: float = np.nan,
        shuffle_idents: bool = False,
        random_state: int = 96,
        return_mask: bool = False,
        ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Load target values corresponding to ids from a targets .pt file.

    If shuffle_idents=True, ids are permuted before lookup to create a
    random-ident baseline target assignment while preserving y distribution.

    Missing idents (and, for source data like docking_score that encodes a
    failed/missing value as a huge sentinel rather than NaN, any target this
    was already cleaned to NaN at generation time) get default_value (np.nan
    by default). Callers that need integer targets can cast afterwards.

    If return_mask=True, also returns a boolean `valid_mask` of the same
    length and row order as this y (and therefore as the X loaded for the
    same ids/split, since load_X_from_pt preserves ids.pt's row order and
    shuffle_idents only permutes values, not length/position). Index both
    X and y with this same mask to drop NaN rows without desynchronizing
    them -- do not drop from y alone, or the two will no longer line up.
    """
    if targets_file is None:
        raise ValueError("targets_file must be provided")
    ids = load_out_tensor(in_dir, ids_file)
    full_targets = load_out_tensor(target_dir, targets_file)
    ids = ids.detach().cpu().numpy().astype(int)
    if shuffle_idents:
        rng = np.random.default_rng(random_state)
        ids = rng.permutation(ids)
    y = np.array(
        [full_targets.get(int(ident), default_value) for ident in ids],
        dtype=float,
    )
    if return_mask:
        return y, ~np.isnan(y)
    return y

# ─────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     ds = build_kd_ds(split_path="/home/fatemeh/thesis/kinodata-3D-affinity-prediction/data/processed/filter_predicted_rmsd_le2.00/random-k-fold/1:5.csv")
#     # ds = build_kd_ds(split_path=Path("data/processed/random-k-fold/1:5.csv"))
#     # ds = build_kd_ds()
#     print(len(ds))
