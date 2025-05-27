import numpy as np
from pathlib import Path
from typing import List, Literal, Any
import json

import torch
from kinodata.model.complex_transformer import RegressionModel
import kinodata.configuration as cfg

from .prob_dataset import GraphReprs




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


def get_path_to_model(model_dir: Path,
                rmsd_threshold: int, 
                  split_type: str, 
                  split_fold: int, 
                  model_type: str
                  ) -> Path:
    p = model_dir / f"rmsd_cutoff_{rmsd_threshold}" / split_type / str(split_fold) / model_type
    if not p.exists():
        p.mkdir(parents=True)
    return p


def load_model_config(config_file: Path) -> dict[str, Any]:
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