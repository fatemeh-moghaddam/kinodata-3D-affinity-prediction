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

    # merge: kwargs overrides defaults
    config_args = {**defaults, **{k: v for k, v in kwargs.items() if k != "config_name"}}
    cfg.register(config_name, **config_args)
    prob_ds_config = cfg.get(config_name)

    # In notebooks, argparse sees Jupyter kernel args and can crash.
    # Keep CLI behavior unchanged: only parse argv outside ipykernel.
    if "ipykernel" not in sys.modules:
        prob_ds_config = prob_ds_config.update_from_args()

    # Resolve paths only AFTER the CLI overrides are applied: gnn_model_type,
    # filter_rmsd_max_value and split_type are all part of output_dir, so
    # resolving them from `defaults` would pin every CLI run to the *default*
    # model's directory -- e.g. a `--gnn_model_type CGNN` job would silently
    # read X from and write its results into data/probing/CGNN-3D/...
    gnn_model_type = prob_ds_config["gnn_model_type"]
    assert gnn_model_type in ["CGNN-3D", "CGNN", "DTI"], (
        f"Invalid GNN model type: {gnn_model_type!r}"
    )
    output_dir = get_out_dir(
        gnn_model_type,
        prob_ds_config["filter_rmsd_max_value"],
        prob_ds_config["split_type"],
        split_fold=None,
    )
    # target_dir stays model-agnostic: data/probing/targets
    return prob_ds_config.update(
        {"output_dir": output_dir, "target_dir": output_dir.parents[2] / "targets"}
    )
