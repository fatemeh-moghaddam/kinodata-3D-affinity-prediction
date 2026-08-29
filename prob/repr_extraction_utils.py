"""
Build and run probing pipeline: dataset subset, GNN model, and per-fold representation extraction.

Public API:
  - build_kd_ds(split_path, filter_rmsd_max_value, include_val) -> KinodataDocked
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
from kinodata.data.dataset import Filtered
from kinodata.model.complex_transformer import RegressionModel
from kinodata.model.complex_transformer import make_model as make_complex_transformer
from kinodata.model.dti import make_model as make_dti_baseline
from kinodata.transform import FilterDockingRMSD, TransformToComplexGraph
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
    "DTI-soft": make_dti_baseline,
    "CGNN": make_complex_transformer,
    "CGNN-3D": make_complex_transformer,
}

# Config a model type implies by its *name* rather than by its checkpoint. "DTI-soft"
# loads the same weights as "DTI" (see `checkpoint_model_type`) and differs only in
# how the probing readout pools them, so the difference has to be injected here --
# the checkpoint's own config.json knows nothing about it.
GNN_CONFIG_OVERRIDES: Dict[str, Dict[str, object]] = {
    "DTI-soft": {"repr_aggr": "softmax"},
}


# ─────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────


def _assert_split_matches_filter(
    split_path: Path, filter_rmsd_max_value: float | None
) -> None:
    """
    Guard the index space of a split file against the dataset it will index.

    A split CSV holds *positional indices into the dataset it was generated from*,
    even though `Split.to_data_frame` names the column "ident". `generate_splits.py`
    generates the ones under data/processed/filter_predicted_rmsd_le<x>.00/ against
    the `Filtered` dataset, so indexing the unfiltered `KinodataDocked` with them
    silently addresses entirely different complexes -- no error, no shape mismatch,
    just the wrong molecules. Fail loudly instead.
    """
    expected = (
        None
        if filter_rmsd_max_value is None
        else f"filter_predicted_rmsd_le{float(filter_rmsd_max_value):.2f}"
    )
    found = next(
        (p for p in split_path.parts if p.startswith("filter_predicted_rmsd_le")), None
    )
    if found != expected:
        raise ValueError(
            f"Split file {split_path} was generated against "
            f"{found or 'the unfiltered dataset'}, but the dataset is being built with "
            f"filter_rmsd_max_value={filter_rmsd_max_value} ({expected or 'no filter'}). "
            "The split's indices are positional and would address the wrong complexes."
        )


def build_kd_ds(
    split_path: Union[str, Path, None] = None,
    filter_rmsd_max_value: float | None = None,
    include_val: bool = False,
) -> KinodataDocked:
    """
    Build the evaluation subset of KinodataDocked for one CV fold.

    `filter_rmsd_max_value` must match the split file's RMSD cutoff: the split's
    indices are positions in the *filtered* dataset, so the same `Filtered` wrapper
    the training pipeline used (see `make_kinodata_module`) has to be applied here
    before indexing. `_assert_split_matches_filter` enforces this.

    include_val=False (default) evaluates the fold's test split only. Checkpoints were
    selected on min val/mae, so val predictions are not clean held-out data and must
    not go into affinity metrics compared against the paper. Set include_val=True to
    recover the larger test+val sample for representation probing.

    Loads the dataset, indexes by split, then drops the full dataset to free memory.
    """
    if split_path is None:
        raise ValueError("split_path must be provided")
    split_path = Path(split_path) if isinstance(split_path, str) else split_path
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    _assert_split_matches_filter(split_path, filter_rmsd_max_value)

    split = Split.from_csv(split_path)
    transform = TransformToComplexGraph(remove_heterogeneous_representation=False)
    base_ds = KinodataDocked(
        use_multiprocessing=True,
        num_processes=CPU_COUNT,
    )
    if filter_rmsd_max_value is None:
        base_ds.transform = transform
        full_ds = base_ds
    else:
        full_ds = Filtered(base_ds, FilterDockingRMSD(filter_rmsd_max_value))(
            transform=transform
        )

    indices = [*split.test_split, *(split.val_split if include_val else [])]
    out_of_range = [i for i in indices if not 0 <= i < len(full_ds)]
    if out_of_range:
        raise IndexError(
            f"{len(out_of_range)} index/indices from {split_path} fall outside the "
            f"{len(full_ds)}-sample dataset (e.g. {out_of_range[:5]}). The split was "
            "generated against a different dataset than the one being indexed."
        )

    ds = full_ds[indices]
    del full_ds, base_ds
    gc.collect()
    return ds


def build_gnn_model(config: cfg.Config) -> RegressionModel:
    """
    Build and load a GNN from config (model type + checkpoint).

    Any `GNN_CONFIG_OVERRIDES` for the type are applied first, so a readout variant
    like "DTI-soft" is configured before the maker reads the config.
    """
    gnn_type = config.gnn_model_type
    if gnn_type not in GNN_MAKERS:
        raise ValueError(
            f"Unknown GNN model type: {gnn_type}. Expected one of {list(GNN_MAKERS)}"
        )

    config.update(GNN_CONFIG_OVERRIDES.get(gnn_type, {}))
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
    requested = config.device or "auto"
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def _build_eval_loader(ds: KinodataDocked, config: Config, device: torch.device) -> DataLoader:
    """
    Inference loader. Defaults to the training `batch_size`, which is known to fit.

    Do not assume no_grad allows a bigger batch here: extraction keeps every layer's
    intermediate node representations resident on the device at once, so peak memory
    per batch is *higher* than a plain forward pass, not lower. A 4x batch OOMs on a
    16 GB card. Override via `infer_batch_size` only if you have headroom to spare;
    batch size does not affect outputs (eval mode, per-graph pooling via `batch_index`).

    Worker count is deliberately small rather than `CPU_COUNT`. Each worker forks the
    in-memory dataset, and the prefetch buffer holds `num_workers * prefetch_factor *
    batch_size` graphs, so a high count costs RAM and oversubscribes the CPUs the job
    actually requested. A handful of workers is enough to stop the main process from
    being the bottleneck.
    """
    batch_size = int(config.get("infer_batch_size", 0) or config.batch_size)
    on_gpu = device.type == "cuda"
    num_workers = min(int(config.get("eval_num_workers", 4) or 0), max(CPU_COUNT - 1, 0)) if on_gpu else 0

    logger.info(
        "Eval loader: batch_size=%s num_workers=%s device=%s", batch_size, num_workers, device
    )
    kwargs = {}
    if num_workers > 0:
        kwargs.update(persistent_workers=True)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=on_gpu,
        **kwargs,
    )


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
    loader = _build_eval_loader(ds, config, device)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing graph representations"):
            idents.extend(batch.ident.tolist())
            batch = batch.to(device, non_blocking=True)
            pred, intermediate_node_reprs, _, _ = gnn_model(batch)
            preds.append(pred.detach().flatten().cpu())
            y_true.append(batch.y.detach().flatten().cpu())

            for layer_name, (node_repr, batch_index) in intermediate_node_reprs.items():
                # A `None` batch index means the model already pooled to one row per
                # complex. Models whose towers pool differently (DTI) have to do that
                # themselves; ComplexTransformer always passes a real batch index, so
                # its path is unaffected.
                if batch_index is None:
                    graph_repr = node_repr.detach().cpu()
                else:
                    graph_repr = gnn_model.aggr(node_repr, batch_index).detach().cpu()
                if dtype_out is not None:
                    graph_repr = graph_repr.to(dtype_out)
                layer_bufs.setdefault(layer_name, []).append(graph_repr)

            # Every layer's node representations are held on-device at once and are the
            # bulk of the peak. Drop them now rather than at the next loop iteration,
            # where they would still be live while the next batch is being allocated.
            del intermediate_node_reprs, pred, batch

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
