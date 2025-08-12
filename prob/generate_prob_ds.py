

from kinodata.data.data_split import Split
import kinodata.configuration as cfg

from .utils import get_model_dir, get_model_ckpt, get_gnn_config_path, get_split_file

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
    prob_config = cfg.get("probing_ds").update_from_args()

    # Get the addresses; each follows a pattern
    #   - model config and model ckpt follow: root/models/rmsd_cutoff_<rmsd_threshold>/<split_type>/<fold>/<model_name>
    #   - splits follows: root/data/processed/filter_predicted_rmsd_le<rmsd_threshold>.00/<split_type>/<fold>_5.csv
    #   - output_dir: root/data/probing/<model_name>/<split_type>/<fold>/output
    model_dir = get_model_dir(rmsd_threshold = prob_config.filter_rmsd_max_value,
                                  split_type = prob_config.split_type,
                                  split_fold = prob_config.split_index,
                                  model_type = prob_config.gnn_model_type
                                  )
    # Update the config from config file for GNN settings
    prob_config.update_from_file(get_gnn_config_path(model_dir))
    prob_config['model_ckpt'] = get_model_ckpt(model_dir)

    prob_config['split_file'] = get_split_file(prob_config.split_type, 
                                               prob_config.split_index, 
                                               prob_config.filter_rmsd_max_value)
    

    