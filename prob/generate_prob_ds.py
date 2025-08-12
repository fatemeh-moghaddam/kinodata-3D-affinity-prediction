import os
import wandb
import torch

from kinodata.data.data_split import Split
import kinodata.configuration as cfg

from prob.utils import get_model_dir, get_model_ckpt, get_gnn_config_path, get_split_file, get_out_dir
from prob.utils import build_kd_ds, build_gnn_model, load_config

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
    3. Build model
    4. Run model for each fold -> fold changes the dataset
    5. Build dataset
    6. Concatenate folds
'''
if __name__ == "__main__":
    # I might need to set some of these with arguments later
    # Initializing config, filling it up as we go
    cfg.register("probing_ds",
                 gnn_model_type="CGNN-3D",
                 split_type="random-k-fold",
                 filter_rmsd_max_value=2,
                 graph_level=True,
                 split_index=0
                 ) 
    prob_config = cfg.get("probing_ds").update_from_args() # this activates the argparse itself

    # wandb.init(project="kinodata", config=prob_config)
    wandb.init(mode="disabled")  # Disable W&B for now, can be enabled later

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
    output_fold_dir = get_out_dir(prob_config.gnn_model_type,
                              prob_config.split_type,
                              prob_config.split_index)
    
    # Update the config from config file for GNN settings
    # prob_config.update_from_file(get_gnn_config_path(model_dir))  This doesn't work for json, only for yaml
    model_config = load_config(get_gnn_config_path(model_dir))
    prob_config.update(
        {  **model_config,
            'model_ckpt': model_ckpt,
            'split_file': split_file_path,
            'output_fold_dir': output_fold_dir
        }, allow_duplicates=True
    )

    # print(f"Model checkpoint: {prob_config.model_ckpt}")
    # print(f"Split file: {prob_config.split_file}")
    # print(f"Output fold directory: {prob_config.output_fold_dir}")

    # ds = build_kd_ds(split_path=split_file_path)
    # assert len(ds) > 0, "Prob dataset is empty"

    gnn_model = build_gnn_model(prob_config).eval()
    assert gnn_model is not None, "Failed to build GNN model"
    # print(gnn_model)  # Set to eval mode


    # with torch.no_grad():
    #     run_fold(gnn_model, ds, output_fold_dir)