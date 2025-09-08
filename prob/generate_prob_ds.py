import os
from tqdm import tqdm

import wandb
import torch

import kinodata.configuration as cfg

from prob.utils import get_model_dir, get_model_ckpt, get_gnn_config_path, get_split_file, get_out_dir
from prob.utils import build_kd_ds, build_gnn_model, load_config
from prob.prob_ds_helpers import run_fold, aggregate_folds, aggregate_ids

'''
This generate the prob dataset for a specific GNN model and split
    It can be updated from arguments, but for now, just using CGNN-3D and random-k-fold
To get the probing dataset, written:
    1. Set the config: 
        - which GNN model? CGNN-3D
        - which split? random-k-fold
        - which RMSD threshold? 2
    2. Using config, get the directories for:
        - model config -> update config
        - model ckpt
        - splits (from processed data)
        - output_dir
    fold changes the dataset and the model
    3. Build dataset subset
    4. Build model
    5. Run model for each fold
    X. Concatenate folds?
'''



def set_probing_config(**kwargs) -> cfg.Config:
    """ Set the probing configuration based on the provided arguments.
    Args:
        **kwargs: Keyword arguments to set the configuration.
    Returns:
        cfg.Config: The updated configuration object.
    """
    # defaults
    defaults = dict(
        gnn_model_type="CGNN-3D",
        split_type="random-k-fold",
        filter_rmsd_max_value=2,
        graph_level=True,
        split_index=0,
        dtype_out=None,  # None means no dtype conversion
    )

    # validate kwargs
    if kwargs.keys() - defaults.keys() != set():
        raise ValueError(f"Invalid arguments: {kwargs.keys() - defaults.keys()}")

    if "gnn_model_type" in kwargs:
        assert kwargs["gnn_model_type"] in ["CGNN-3D", "CGNN", "DTI"], "Invalid GNN model type"
    if "split_type" in kwargs:
        assert kwargs["split_type"] in ["random-k-fold", "scaffold-k-fold", "pocket-k-fold"], "Invalid split type"
    if "filter_rmsd_max_value" in kwargs:
        assert kwargs["filter_rmsd_max_value"] in set({2, 4, 6, 2.00, 4.00, 6.00, None}), "Invalid RMSD threshold"
    if "split_index" in kwargs:
        assert isinstance(kwargs["split_index"], int) and kwargs["split_index"] >= 0, "Split index must be a non-negative integer"

    # merge: kwargs overrides defaults
    config_args = {**defaults, **kwargs}
    # Initialize the config with the defaults and kwargs

    cfg.register("probing_ds", **config_args)

    prob_config = cfg.get("probing_ds").update_from_args() # this activates the argparse itself


    # Get the addresses; each follows a pattern
    #   - model config and model ckpt follow: root/models/rmsd_cutoff_<rmsd_threshold>/<split_type>/<fold>/<model_name>
    #   - splits follows: root/data/processed/filter_predicted_rmsd_le<rmsd_threshold>.00/<split_type>/<fold>_5.csv
    #   - output_dir: root/data/probing/<model_name>/<split_type>/<fold>/output
    model_dir = get_model_dir(rmsd_threshold = prob_config.filter_rmsd_max_value,
                                  split_type = prob_config.split_type,
                                  split_fold = prob_config.split_index,
                                  model_type = prob_config.gnn_model_type
                                  )
    model_ckpt = get_model_ckpt(model_dir)
    split_file_path = get_split_file(prob_config.split_type,
                                 prob_config.split_index,
                                 prob_config.filter_rmsd_max_value)
    
    ##### do I change this to output_dir?
    output_dir = get_out_dir(prob_config.gnn_model_type,
                              prob_config.split_type,
                              split_fold=None)
    

    # Update the config from config file for GNN settings
    # prob_config.update_from_file(get_gnn_config_path(model_dir))  This doesn't work for json, only for yaml
    model_config = load_config(get_gnn_config_path(model_dir))
    prob_config.update(
        {  **model_config,
            'model_ckpt': model_ckpt,
            'split_file': split_file_path,
            # 'output_fold_dir': output_fold_dir,
            # 'output_agg_dir': output_agg_dir
            'output_dir': output_dir
        }, allow_duplicates=True
    )

    # print(f"Model checkpoint: {prob_config.model_ckpt}")
    # print(f"Split file: {prob_config.split_file}")
    print(f"Output directory: {prob_config.output_dir}")

    return prob_config

    





if __name__ == "__main__":
    # Set the random seed for reproducibility
    torch.manual_seed(123)
    # wandb.init(project="kinodata", config=prob_config)
    wandb.init(mode="disabled")  # Disable W&B for now, can be enabled later

    k_fold = 5  # for now, just hardcoding this

    for fold in range(k_fold):
        # prob_config.update({'split_index': fold})
        # prob_config.split_index = fold
        # need to have fold index for model ckpt
        prob_config = set_probing_config(split_index = fold)

        gnn_model = build_gnn_model(prob_config).eval()
        assert gnn_model is not None, "Failed to build GNN model"
        # print(gnn_model)  # Set to eval mode
        
        ds = build_kd_ds(split_path=prob_config.split_file)
        assert len(ds) > 0, "Prob dataset is empty"

        run_fold(ds, gnn_model, prob_config)

    prob_config.split_index = None
    num_layers = prob_config.get('num_attention_blocks', 3)

    # Aggregate folds for each layer
    for i in range(num_layers):
        layer_name = f"layer_{i+1}"
        aggregate_folds(prob_config, layer_name)

    # Aggregate ids across folds for downstream mapping
    aggregate_ids(prob_config)

