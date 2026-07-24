from typing import List, Any
import json
from pathlib import Path
import colorama

import pandas as pd
import torch

from kinodata.model.complex_transformer import RegressionModel
import kinodata.configuration as cfg






# ─────────────────────────────────────────────────────────────
# Resolvers
# ─────────────────────────────────────────────────────────────
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


def dtype_resolve(dtype: torch.dtype | str | None) -> torch.dtype | None:
    """
    Resolve the dtype from a string or return None.
    """
    if isinstance(dtype, str):
        _map = {"float32": torch.float32, "fp32": torch.float32,
                "float16": torch.float16, "fp16": torch.float16,
                "bfloat16": torch.bfloat16}
        dtype_out = _map.get(dtype.lower(), None)
    elif isinstance(dtype, torch.dtype):
        dtype_out = dtype
    else:
        dtype_out = None
    return dtype_out




# ─────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────


def aggregate_folds(config: cfg.Config, layer_name: str) -> None:
    '''
    Aggregate tensors from all folds for a specific layer, and writes them to config.output_dir.
    out:
        Saved aggregated tensors.
        The naming pattern of the saved tensors is: {layer_name}.pt
    '''
    # I'm assuming this directory is the parent of the fold directories
    output_dir = config.output_dir
    # TODO: check if this is the correct directory

    k_fold = config.get('k_fold', 5)

    fold_tensors: List[torch.Tensor] = []
    

    # Load tensors from each fold
    for fold in range(k_fold):  # Assuming 5 folds
        fold_dir = output_dir / str(fold)
        tensor_path = fold_dir / f"{layer_name}_{fold}.pt"
        if tensor_path.exists():
            fold_tensors.append(torch.load(tensor_path))
        else:
            print(f"Warning: {tensor_path} does not exist.")

    if not fold_tensors:
        print(colorama.Fore.RED + "No tensors found for aggregation." + colorama.Style.RESET_ALL)
        return

    # Concatenate tensors
    aggregated_tensor = torch.cat(fold_tensors, dim=0)

    # Save aggregated tensor
    aggregated_path = output_dir / f"{layer_name}.pt"
    torch.save(aggregated_tensor, aggregated_path)
    print(f"Aggregated tensor saved to {aggregated_path}")

    return


def aggregate_ids(config: cfg.Config) -> None:
    """
    Aggregate ids_<fold>.pt across all folds and save to config.output_dir/ids.pt
    """
    output_dir = config.output_dir
    k_fold = config.get('k_fold', 5)

    id_tensors: List[torch.Tensor] = []
    for fold in range(k_fold):
        fold_dir = output_dir / str(fold)
        ids_path = fold_dir / f"ids_{fold}.pt"
        if ids_path.exists():
            id_tensors.append(torch.load(ids_path))
        else:
            print(f"Warning: {ids_path} does not exist.")

    if not id_tensors:
        print(colorama.Fore.RED + "No ids found for aggregation." + colorama.Style.RESET_ALL)
        return

    aggregated_ids = torch.cat(id_tensors, dim=0)
    aggregated_path = output_dir / "ids.pt"
    torch.save(aggregated_ids, aggregated_path)
    print(f"Aggregated ids saved to {aggregated_path}")

    return


def aggregate_predictions(config: cfg.Config) -> None:
    """
    Aggregate preds_<fold>.pt / y_true_<fold>.pt across all folds (written by
    run_fold when save_predictions=True) and save as a single
    config.output_dir/predictions.csv with columns y_true, y_pred.
    """
    output_dir = config.output_dir
    k_fold = config.get('k_fold', 5)

    pred_tensors: List[torch.Tensor] = []
    y_true_tensors: List[torch.Tensor] = []
    for fold in range(k_fold):
        fold_dir = output_dir / str(fold)
        preds_path = fold_dir / f"preds_{fold}.pt"
        y_true_path = fold_dir / f"y_true_{fold}.pt"
        if preds_path.exists() and y_true_path.exists():
            pred_tensors.append(torch.load(preds_path))
            y_true_tensors.append(torch.load(y_true_path))
        else:
            print(f"Warning: {preds_path} or {y_true_path} does not exist.")

    if not pred_tensors:
        print(colorama.Fore.RED + "No predictions found for aggregation." + colorama.Style.RESET_ALL)
        return

    y_pred = torch.cat(pred_tensors, dim=0).numpy()
    y_true = torch.cat(y_true_tensors, dim=0).numpy()

    aggregated_path = output_dir / "predictions.csv"
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(aggregated_path, index=False)
    print(f"Aggregated predictions saved to {aggregated_path}")

    return

# ─────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────


