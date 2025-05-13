import kinodata
from kinodata.data import KinodataDocked
from kinodata.transform import TransformToComplexGraph
from kinodata.types import *
import kinodata.configuration as cfg
from kinodata.model import ComplexTransformer, RegressionModel
from kinodata.model.complex_transformer import make_model as make_complex_transformer


import torch
from torch_geometric.loader import DataLoader

import json
from pathlib import Path
import os
from typing import Any
from functools import partial
from collections import defaultdict


# import matplotlib.pyplot as plt
# import seaborn as sns

import pandas as pd
import numpy as np
from tqdm import tqdm
import random

# from rdkit.Chem import PandasTools
# from rdkit import Chem
# from rdkit.Chem import Descriptors

_DATA = Path.cwd().parent / "data"
_PROBING_DATA = _DATA / "probing"
CPU_COUNT = 12

model_dir = Path("models")
assert model_dir.exists()

def path_to_model(rmsd_threshold: int, 
                  split_type: str, 
                  split_fold: int, 
                  model_type: str
                  ) -> Path:
    p = model_dir / f"rmsd_cutoff_{rmsd_threshold}" / split_type / str(split_fold) / model_type
    if not p.exists():
        p.mkdir(parents=True)
    return p


def load_config(config_file: Path) -> dict[str, Any]:
    with open(config_file, "r") as f_config:
        config = json.load(f_config)
    config = {str(key): value["value"] for key, value in config.items()}
    # cfg.Config is a subclass of dict, it's a dictionary with some extra methods
    # it is used when make_model is called
    return cfg.Config(config) 


def load_from_checkpoint(model: RegressionModel, model_ckpt: str) -> RegressionModel:
    ckp = torch.load(model_ckpt, map_location="cpu")
    assert isinstance(model, RegressionModel)
    model.load_state_dict(ckp["state_dict"])
    return model






if __name__ == "__main__":
    # let's have it deterministic!
    torch.manual_seed(123)
    np.random.seed(123)
    
    # Load the model
    cgnn_3d_path = path_to_model(rmsd_threshold=2, split_type="scaffold-k-fold", split_fold=0, model_type="CGNN-3D")
    # Load the model checkpoint
    cgnn_3d_ckpt = list(cgnn_3d_path.glob("**/*.ckpt"))[0]
    assert cgnn_3d_ckpt.exists()
    # Load the model config for this checkpoint
    cgnn_3d_config_path = cgnn_3d_path / "config.json"
    config = load_config(cgnn_3d_config_path)
    # Make and Load the model
    cgnn_3d = make_complex_transformer(config)
    cgnn_3d_loaded = load_from_checkpoint(model = cgnn_3d, model_ckpt=cgnn_3d_ckpt)
    
    # Load the orignial dataset
    orig_data = KinodataDocked(transform=TransformToComplexGraph(remove_heterogeneous_representation=False),
                      use_multiprocessing=True,
                      num_processes= CPU_COUNT)
    
    # Load the probing dataset
    X = make_probing_x(dataset=orig_data, model=cgnn_3d_loaded, num_workers=CPU_COUNT)
    y = make_probing_y(data_index, mapper=mapper)

    # Initialize the probe
    ## train-test split
    ## CV for parameters?
    prob_results  = prob(X, y, probe_model)
    # Load probing dataset

    # Probe

    # Save the results