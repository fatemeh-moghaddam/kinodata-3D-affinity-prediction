import os
import numpy as np
from pathlib import Path
from typing import List, Literal, Any, Union
import json
import colorama
from tqdm import tqdm
import gc

import torch

from kinodata.data import KinodataDocked
from kinodata.data.data_split import Split
from kinodata.transform import TransformToComplexGraph
from kinodata.model.complex_transformer import RegressionModel
from kinodata.model.complex_transformer import make_model as make_complex_transformer
from kinodata.model.dti import make_model as make_dti_baseline
import kinodata.configuration as cfg




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
# Load/Save/Concatenate helpers
# ─────────────────────────────────────────────────────────────
# Load the model config from a JSON file
# This is needed instead of cfg.update_from_file() because JSON files have a different structure
# 
def load_config(config_file: Path) -> dict[str, Any]:
    with open(config_file, "r") as f_config:
        config = json.load(f_config)
    config = {str(key): value["value"] for key, value in config.items()}
    # cfg.Config is a subclass of dict, it's a dictionary with some extra methods
    # it is used when make_model is called
    return cfg.Config(config) 




def load_model_from_checkpoint(model: RegressionModel, model_ckpt: str) -> RegressionModel:
    ckp = torch.load(model_ckpt, map_location="cpu")
    assert isinstance(model, RegressionModel)
    model.load_state_dict(ckp["state_dict"])
    return model



# ─────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────

def build_kd_ds(split_path: Union[str, Path, None] = None) -> KinodataDocked:
    """
    Build the subset of KinodataDocked dataset based on the specific Split, 
    combining test and validation of a split type and fold.

    This will take a bit because it loads KinodataDocked
    """
    if split_path is None:
        raise ValueError("split_path must be provided")
    if isinstance(split_path, str):
        split_path = Path(split_path)
    if split_path.exists():
        split = Split.from_csv(split_path)
    else:
        raise FileNotFoundError(f"Split file not found: {split_path}")

    full_ds = KinodataDocked(transform=TransformToComplexGraph(remove_heterogeneous_representation=False),
                      use_multiprocessing=True,
                      num_processes= os.cpu_count())
    ds = full_ds[[*split.test_split, *split.val_split]]

    del full_ds  # free memory
    gc.collect()
    
    return ds



def build_gnn_model(cfg: cfg.Config) -> RegressionModel:
    """
    Build and load a GNN model based on the provided configuration: model type and model checkpoint.
    """
    gnn_type = cfg.gnn_model_type
    if gnn_type not in ["DTI", "CGNN", "CGNN-3D"]:
        raise ValueError(f"Unknown GNN model type: {gnn_type}. Expected one of ['DTI', 'CGNN', 'CGNN-3D']")
    gnn_maker = {
    "DTI": make_dti_baseline,
    "CGNN": make_complex_transformer,
    "CGNN-3D": make_complex_transformer
    }
    gnn = gnn_maker[gnn_type](cfg)
    assert isinstance(gnn, RegressionModel), f"Expected a RegressionModel, got {type(gnn)}"
    gnn_ckpt = cfg.model_ckpt
    if not gnn_ckpt:
        raise ValueError(f"Model checkpoint not found for GNN type: {gnn_type}")
    loaded_gnn = load_model_from_checkpoint(gnn, gnn_ckpt)
    return loaded_gnn


# ─────────────────────────────────────────────────────────────
# Transform helpers
# ─────────────────────────────────────────────────────────────

def get_np_X(
        graphs: List['GraphReprs'],
        layer_name: str,
        level: Literal["graph", "node"] = "graph"
        ) -> np.ndarray:
    """
    Extract feature matrix X from GraphReprs objects for a given layer and level.

    Returns:
        X: np.ndarray, shape [n_samples, hidden_dim]
    """
    X_list = []

    for g in graphs:
        if level == "graph":
            x = g.graph_repr[layer_name].detach().cpu().numpy()
            X_list.append(x)
        elif level == "node":
            x = g.node_repr[layer_name].detach().cpu().numpy()
            X_list.append(x)
        else:
            raise ValueError("level must be 'graph' or 'node'")

    return np.vstack(X_list)



def get_np_y(
        graphs: List['GraphReprs'],
        target: str,
        level: Literal["graph", "node"] = "graph",
        ) -> np.ndarray:
    """
    Extract target array y from GraphReprs objects for a given level.

    Returns:
        y: np.ndarray, shape [n_samples]
    """
    y_list = []

    for g in graphs:
        y = g.get_property(target)
        if level == "graph":
            y_list.append(y)
        elif level == "node":
            # y should be a 2D array with shape (NUM_GRAPHS, NUM_NODES)
            pass 
        else:
            raise ValueError("level must be 'graph' or 'node'")

    return np.array(y_list)


# ─────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     ds = build_kd_ds(split_path="/home/fatemeh/thesis/kinodata-3D-affinity-prediction/data/processed/filter_predicted_rmsd_le2.00/random-k-fold/1:5.csv")
#     # ds = build_kd_ds(split_path=Path("data/processed/random-k-fold/1:5.csv"))
#     # ds = build_kd_ds()
#     print(len(ds))
