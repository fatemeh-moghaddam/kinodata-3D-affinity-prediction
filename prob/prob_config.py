from pathlib import Path
import sys

import kinodata.configuration as cfg

from prob.paths_and_io import get_out_dir


TARGET_FILE = None
# ─────────────────────────────────────────────────────────────
# General Config
# ─────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────
# DS Generation Config
# ─────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────
# Probing Config
# ─────────────────────────────────────────────────────────────

def build_experiment_name(ds_cfg: cfg.Config, layer_num: int) -> str:
    if not ds_cfg.target_file:
        raise ValueError("target_file must be set before building an experiment name")
    parts = [
        f"gnn={ds_cfg.gnn_model_type}",
        f"rmsd={ds_cfg.filter_rmsd_max_value}",
        f"split={ds_cfg.split_type}",
        f"layer={layer_num}",
        f"target={Path(ds_cfg.target_file).stem}",
    ]
    return "_".join(parts)


def get_ds_load_config(**kwargs):
    # default config values
    defaults = dict(
            gnn_model_type="CGNN-3D",
            split_type="random-k-fold",
            filter_rmsd_max_value=2,
            graph_level=True,
            split_index=None,
            dtype_out=None,  # None means no dtype conversion
            device="cpu",
            target_file=TARGET_FILE or "",  # empty str so argparser registers it as str type
            run_shuffled_baseline=0,
            baseline_tag="shuffled_ident",
        )

    allowed_keys = set(defaults.keys()) | {"config_name"}
    invalid = kwargs.keys() - allowed_keys
    if invalid:
        raise ValueError(f"Invalid arguments: {invalid}")

    if "gnn_model_type" in kwargs:
        assert kwargs["gnn_model_type"] in ["CGNN-3D", "CGNN", "DTI"], "Invalid GNN model type"
    if "split_type" in kwargs:
        assert kwargs["split_type"] in ["random-k-fold", "scaffold-k-fold", "pocket-k-fold"], "Invalid split type"
    if "filter_rmsd_max_value" in kwargs:
        assert kwargs["filter_rmsd_max_value"] in set({2, 4, 6, 2.00, 4.00, 6.00, None}), "Invalid RMSD threshold"
    if "split_index" in kwargs:
        assert kwargs["split_index"] in set({0, 1, 2, 3, 4, None}), "Invalid split index"
    config_name = kwargs.get("config_name", "prob_ds_load")
    target_file = kwargs.get("target_file", defaults["target_file"])

    # merge: kwargs overrides defaults
    config_args = {**defaults, **{k: v for k, v in kwargs.items() if k != "config_name"}}
    # Initialize the config with the defaults and kwargs
    output_dir = get_out_dir(config_args["gnn_model_type"],
                                config_args["filter_rmsd_max_value"],
                                config_args["split_type"],
                                split_fold=None)
    
    target_dir = output_dir.parents[2] / "targets"

    cfg.register(config_name, **{**config_args, "output_dir": output_dir, "target_dir": target_dir})
    prob_ds_config = cfg.get(config_name)

    # In notebooks, argparse sees Jupyter kernel args and can crash.
    # Keep CLI behavior unchanged: only parse argv outside ipykernel.
    if "ipykernel" in sys.modules:
        return prob_ds_config
    return prob_ds_config.update_from_args()
