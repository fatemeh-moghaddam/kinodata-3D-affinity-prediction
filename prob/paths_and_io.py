from functools import lru_cache
import os
from pathlib import Path

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
        root: Path = _ROOT
    ) -> Path:
    """
    This is the output directory per fold. 
    The concatenated one would be in the parent directory of this.
    And the layered separation would be in the naming.
    """
    p = root / "data/probing" / gnn_model_type/ f"rmsd_cutoff_{rmsd_threshold}" / split_type
    if split_fold is not None:   # <- avoids dropping fold==0
        p = p / str(split_fold)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_exp_dirs(out_root: str | Path, target: str , prob_model: str, layer_num: int) -> dict[str, Path]:
    """Create directories for storing results, figures, and artifacts.
    Experiments are created based on their X, y, and prob model.
    Returns a dict with paths.
    """
    exp_root = Path(out_root)  # e.g. ~/kinodata-3D-affinity-prediction/data/probing/CGNN-3D/rmsd_cutoff_2/random-k-fold
    exp_root = exp_root / target / prob_model / str(layer_num)
    dirs = {
        "root": exp_root,
        "figures": exp_root / "figures",
        "artifacts": exp_root / "artifacts",
        "reports": exp_root / "reports",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


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
        default_value: int = 0,
        shuffle_idents: bool = False,
        random_state: int = 96,
        ) -> np.ndarray:
    """
    Load target values corresponding to ids from a targets .pt file.

    If shuffle_idents=True, ids are permuted before lookup to create a
    random-ident baseline target assignment while preserving y distribution.
    """
    if targets_file is None:
        raise ValueError("targets_file must be provided")
    ids = load_out_tensor(in_dir, ids_file)
    full_targets = load_out_tensor(target_dir, targets_file)
    # for easier indexing
    ids = ids.detach().cpu().numpy().astype(int)
    if shuffle_idents:
        rng = np.random.default_rng(random_state)
        ids = rng.permutation(ids)
    y = np.array(
        [full_targets.get(int(ident), default_value) for ident in ids],
        dtype="int64",
    )
    return y

# ─────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     ds = build_kd_ds(split_path="/home/fatemeh/thesis/kinodata-3D-affinity-prediction/data/processed/filter_predicted_rmsd_le2.00/random-k-fold/1:5.csv")
#     # ds = build_kd_ds(split_path=Path("data/processed/random-k-fold/1:5.csv"))
#     # ds = build_kd_ds()
#     print(len(ds))
