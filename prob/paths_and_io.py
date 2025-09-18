import os
from pathlib import Path

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




# ─────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     ds = build_kd_ds(split_path="/home/fatemeh/thesis/kinodata-3D-affinity-prediction/data/processed/filter_predicted_rmsd_le2.00/random-k-fold/1:5.csv")
#     # ds = build_kd_ds(split_path=Path("data/processed/random-k-fold/1:5.csv"))
#     # ds = build_kd_ds()
#     print(len(ds))
