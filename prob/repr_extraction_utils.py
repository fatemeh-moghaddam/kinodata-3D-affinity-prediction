"""
Build and run probing pipeline: dataset subset, GNN model, and per-fold representation extraction.

Public API:
  - build_kd_ds(split_path) -> KinodataDocked
  - build_gnn_model(config) -> RegressionModel
  - run_fold(ds, gnn_model, config) -> None

Run contract (what `config` must provide for run_fold):
  - batch_size: int
  - output_dir: Path (root under which fold subdirs are created)
  - split_index: int
  - graph_level: bool (must be True; node-level not implemented)
  - device: str (e.g. "cpu", "cuda")
  - dtype_out: str | None (optional, e.g. "float32", "float16")
"""
from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Dict, List, Union

import torch
from torch_geometric.loader import DataLoader

import kinodata.configuration as cfg
from kinodata.configuration import Config
from kinodata.data import KinodataDocked
from kinodata.data.data_split import Split
from kinodata.model.complex_transformer import RegressionModel
from kinodata.model.complex_transformer import make_model as make_complex_transformer
from kinodata.model.dti import make_model as make_dti_baseline
from kinodata.transform import TransformToComplexGraph
from tqdm import tqdm

from prob.paths_and_io import save_out_tensor
from prob.resloves_and_transforms import dtype_resolve, load_model_from_checkpoint

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

_ROOT = Path(os.environ.get("HOME_PROJ_DIR", Path(__file__).resolve().parents[1]))
CPU_COUNT = int(os.environ.get("CPU_COUNT", "16"))

GNN_MAKERS = {
    "DTI": make_dti_baseline,
    "CGNN": make_complex_transformer,
    "CGNN-3D": make_complex_transformer,
}


# ─────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────


def build_kd_ds(split_path: Union[str, Path, None] = None) -> KinodataDocked:
    """
    Build the subset of KinodataDocked for a given split (test + val indices).

    Loads the full dataset, indexes by split, then drops the full dataset to free memory.
    """
    if split_path is None:
        raise ValueError("split_path must be provided")
    split_path = Path(split_path) if isinstance(split_path, str) else split_path
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    split = Split.from_csv(split_path)
    full_ds = KinodataDocked(
        transform=TransformToComplexGraph(remove_heterogeneous_representation=False),
        use_multiprocessing=True,
        num_processes=CPU_COUNT,
    )
    ds = full_ds[[*split.test_split, *split.val_split]]
    del full_ds
    gc.collect()
    return ds


def build_gnn_model(config: cfg.Config) -> RegressionModel:
    """
    Build and load a GNN from config (model type + checkpoint).
    """
    gnn_type = config.gnn_model_type
    if gnn_type not in GNN_MAKERS:
        raise ValueError(
            f"Unknown GNN model type: {gnn_type}. Expected one of {list(GNN_MAKERS)}"
        )

    maker = GNN_MAKERS[gnn_type]
    model = maker(config)
    if not isinstance(model, RegressionModel):
        raise TypeError(f"Expected RegressionModel, got {type(model)}")

    ckpt = config.model_ckpt
    if not ckpt:
        raise ValueError(f"Model checkpoint not set for GNN type: {gnn_type}")

    return load_model_from_checkpoint(model, ckpt)


# ─────────────────────────────────────────────────────────────
# Fold runner (internal helpers + public run_fold)
# ─────────────────────────────────────────────────────────────


def _resolve_device(config: Config) -> torch.device:
    return torch.device(config.device or "cpu")


def _compute_fold_representations(
    ds: KinodataDocked,
    gnn_model: RegressionModel,
    config: Config,
    device: torch.device,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run model in eval mode over the dataset and collect per-layer graph representations,
    sample IDs, and the model's predictions/targets (the latter two are cheap byproducts
    of the same forward pass, so they're always collected; callers decide whether to
    persist them). Returns (layer_name -> tensor of shape [N, d], ids_tensor of shape [N],
    preds_tensor of shape [N], y_true_tensor of shape [N]).
    """
    if not config.graph_level:
        raise NotImplementedError("Only graph-level probing is implemented.")

    layer_bufs: Dict[str, List[torch.Tensor]] = {}
    idents: List[int] = []
    preds: List[torch.Tensor] = []
    y_true: List[torch.Tensor] = []
    dtype_out = dtype_resolve(config.dtype_out)

    gnn_model.eval().to(device)
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing graph representations"):
            idents.extend(batch.ident.tolist())
            batch = batch.to(device)
            pred, intermediate_node_reprs, _, _ = gnn_model(batch)
            preds.append(pred.detach().flatten().cpu())
            y_true.append(batch.y.detach().flatten().cpu())

            for layer_name, (node_repr, batch_index) in intermediate_node_reprs.items():
                graph_repr = gnn_model.aggr(node_repr, batch_index).detach().cpu()
                if dtype_out is not None:
                    graph_repr = graph_repr.to(dtype_out)
                layer_bufs.setdefault(layer_name, []).append(graph_repr)

    layers_cat = {k: torch.cat(v, dim=0) for k, v in layer_bufs.items()}
    ids_tensor = torch.tensor(idents, dtype=torch.long)
    preds_tensor = torch.cat(preds, dim=0)
    y_true_tensor = torch.cat(y_true, dim=0)
    return layers_cat, ids_tensor, preds_tensor, y_true_tensor


def _validate_fold_shapes(
    fold_size: int,
    layers_cat: Dict[str, torch.Tensor],
    ids_tensor: torch.Tensor,
    preds_tensor: torch.Tensor,
    y_true_tensor: torch.Tensor,
) -> None:
    if ids_tensor.shape[0] != fold_size:
        raise RuntimeError(
            f"IDs tensor size {ids_tensor.shape[0]} does not match fold size {fold_size}"
        )
    if preds_tensor.shape[0] != fold_size or y_true_tensor.shape[0] != fold_size:
        raise RuntimeError(
            f"Predictions/targets size {preds_tensor.shape[0]}/{y_true_tensor.shape[0]} "
            f"does not match fold size {fold_size}"
        )
    for layer_name, t in layers_cat.items():
        if t.shape[0] != fold_size:
            raise RuntimeError(
                f"Layer {layer_name} tensor size {t.shape[0]} does not match fold size {fold_size}"
            )


def _write_fold_artifacts(
    config: Config,
    layers_cat: Dict[str, torch.Tensor],
    ids_tensor: torch.Tensor,
) -> None:
    fold_dir = Path(config.output_dir) / str(config.split_index)
    for layer_name, t in layers_cat.items():
        save_out_tensor(t, fold_dir, f"{layer_name}_{config.split_index}.pt")
    save_out_tensor(ids_tensor, fold_dir, f"ids_{config.split_index}.pt")


def _write_fold_predictions(
    config: Config,
    preds_tensor: torch.Tensor,
    y_true_tensor: torch.Tensor,
) -> None:
    fold_dir = Path(config.output_dir) / str(config.split_index)
    save_out_tensor(preds_tensor, fold_dir, f"preds_{config.split_index}.pt")
    save_out_tensor(y_true_tensor, fold_dir, f"y_true_{config.split_index}.pt")


def run_fold(
    ds: KinodataDocked,
    gnn_model: RegressionModel,
    config: Config,
    save_representations: bool = True,
    save_predictions: bool = False,
) -> None:
    """
    Run one fold: compute per-layer graph representations, IDs, and predictions/targets
    for `ds` in a single forward pass, then write out whichever artifacts are requested.

    - save_representations (default True, matches prior behavior): write layer tensors
      and ids_<fold>.pt to config.output_dir / config.split_index.
    - save_predictions (default False): write preds_<fold>.pt and y_true_<fold>.pt to
      the same fold directory. These are new, separate files -- enabling this never
      touches the layer/ids artifacts, and disabling save_representations lets you
      collect predictions only, without rewriting the (possibly already computed)
      representation artifacts.
    """
    fold_size = len(ds)
    device = _resolve_device(config)

    layers_cat, ids_tensor, preds_tensor, y_true_tensor = _compute_fold_representations(
        ds, gnn_model, config, device
    )
    _validate_fold_shapes(fold_size, layers_cat, ids_tensor, preds_tensor, y_true_tensor)
    if save_representations:
        _write_fold_artifacts(config, layers_cat, ids_tensor)
    if save_predictions:
        _write_fold_predictions(config, preds_tensor, y_true_tensor)
