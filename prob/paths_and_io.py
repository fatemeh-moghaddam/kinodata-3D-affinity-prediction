import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch






# _ROOT = Path(__file__).resolve().parent.parent
_ROOT = Path(os.environ.get("HOME_PROJ_DIR", Path(__file__).resolve().parents[1])) # to allow setting a different root via env variable


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
        root: Path = _ROOT
    ) -> Path:
    """ Get the path to the split file, which has a different pattern than model directory.
    Args:
        split_type (str): The type of split, e.g., "random-k-fold", "scaffold-k-fold".
        split_fold (int): The fold number.
        rmsd_threshold (int, optional): The RMSD threshold. Defaults to 0.
        root (Path, optional): The root path. Defaults to the parent directory of this file.
    Returns:
        Path: The path to the split csv file.
    """
    p = root / "data/processed"
    if rmsd_threshold is None:
        p = p / split_type
    else:
        p = p / f"filter_predicted_rmsd_le{rmsd_threshold}.00" / split_type
    p = p / f"{split_fold + 1}:5.csv"
    if not p.exists():
        raise FileNotFoundError(f"Split file not found: {p}")
    return p


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


def get_exp_dirs(out_root: str | Path, target: str , prob_model: str) -> dict[str, Path]:
    """Create directories for storing results, figures, and artifacts.
    Experiments are created based on their X, y, and prob model.
    Returns a dict with paths.
    """
    exp_root = Path(out_root)  # e.g. ~/kinodata-3D-affinity-prediction/data/probing/CGNN-3D/rmsd_cutoff_2/random-k-fold
    exp_root = exp_root / target / prob_model
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
        ) -> np.ndarray:
    """ Load target values corresponding to the given ids from a targets .pt file."""
    if targets_file is None:
        raise ValueError("targets_file must be provided")
    ids = load_out_tensor(in_dir, ids_file)
    full_targets = load_out_tensor(target_dir, targets_file)
    # for easier indexing
    ids = ids.detach().cpu().numpy().astype(int)
    target_df = pd.Series(full_targets, dtype="int64")
    # slice
    target_df = target_df[ids]
    return target_df.to_numpy()

# ─────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     ds = build_kd_ds(split_path="/home/fatemeh/thesis/kinodata-3D-affinity-prediction/data/processed/filter_predicted_rmsd_le2.00/random-k-fold/1:5.csv")
#     # ds = build_kd_ds(split_path=Path("data/processed/random-k-fold/1:5.csv"))
#     # ds = build_kd_ds()
#     print(len(ds))
