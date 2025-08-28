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

from kinodata.data.data_split import Split

from .utils import get_path_to_model, load_model_config, load_model_from_checkpoint, make_probing_x, make_probing_y
from . prob_dataset import ProbingDataset, GraphReprs
from .utils import get_split_file


# from rdkit.Chem import PandasTools
# from rdkit import Chem
# from rdkit.Chem import Descriptors

_ROOT = Path(kinodata.__file__).parent.parent
_MODEL = _ROOT / "models"
_DATA = _ROOT / "data"
_PROBING_DATA = _DATA / "probing"
CPU_COUNT = 12










# if __name__ == "__main__":
#     # let's have it deterministic!
#     torch.manual_seed(123)
#     np.random.seed(123)
    
#     # The experiment setup
#     rmsd_threshold = 2
#     split_type = "random-k-fold"
#     split_fold = 0
#     gnn_model_type = "CGNN-3D"


#     # Load the model
#     gnn_model_path = get_path_to_model(rmsd_threshold=rmsd_threshold, split_type=split_type, split_fold=split_fold, model_type=gnn_model_type)
#     # Load the model checkpoint
#     gnn_model_ckpt = list(gnn_model_path.glob("**/*.ckpt"))[0]
#     assert gnn_model_ckpt.exists()
#     # Load the model config for this checkpoint
#     gnn_model_config_path = gnn_model_path / "config.json"
#     config = load_model_config(gnn_model_config_path)
#     # Make and Load the model
#     gnn_model = make_complex_transformer(config)
#     gnn_model_loaded = load_model_from_checkpoint(model = gnn_model, model_ckpt=gnn_model_ckpt)

    
#     # Load the orignial dataset
#     orig_data = KinodataDocked(transform=TransformToComplexGraph(remove_heterogeneous_representation=False),
#                       use_multiprocessing=True,
#                       num_processes= CPU_COUNT)
#     # Load the split:
#     split_file = get_split_file(split_type=split_type, split_fold=split_fold, rmsd_threshold=rmsd_threshold, root=_ROOT)
#     split_obj = Split.from_csv(split_file)
#     # Get the test and validation datasets
#     ds = orig_data[[*split_obj.test_split, *split_obj.val_split]]


#     graph_list = ProbingDataset(dataset=ds,
#                                 model=gnn_model_loaded,
#                                 num_workers=CPU_COUNT)

    
#     ####
#     # Load the probing dataset
#     X = make_probing_x(dataset=orig_data, model=gnn_model_loaded, num_workers=CPU_COUNT)
#     y = make_probing_y(data_index, mapper=mapper)

#     # Initialize the probe
#     ## train-test split
#     ## CV for parameters?
#     prob_results  = prob(X, y, probe_model)
#     # Load probing dataset

#     # Probe

#     # Save the results




def main():
    # Load their config, then allow YAML/CLI to override (same behavior they already use)
    cfg.register("probing_ds"
                 
                 )
    cfg = get("data", "training", "probe").update_from_file().update_from_args()

    # Support either single values or lists:
    model_types = as_list(cfg.probe_model_types) or [cfg.probe_model_type]
    split_types = as_list(cfg.probe_split_types) or [cfg.split_type]
    rmsds       = as_list(cfg.probe_rmsd_cutoffs) or [cfg.filter_rmsd_max_value]

    for m in model_types:
        cfg["probe_model_type"] = m
        for s in split_types:
            cfg["split_type"] = s
            for r in rmsds:
                cfg["filter_rmsd_max_value"] = r
                print(f"\n=== Running setting: model={m}, split={s}, rmsd={r} ===")
                run_one_setting(cfg)

if __name__ == "__main__":
    main()