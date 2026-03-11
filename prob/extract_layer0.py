"""
Standalone extraction of pre-network graph representations as layer_0.

This script does not modify or call `_compute_fold_representations`; it reruns
the model over each fold and writes fold artifacts in the existing style:
  - layer_0_<fold>.pt
  - ids_<fold>.pt
Then aggregates to:
  - layer_0.pt
  - ids.pt
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import kinodata.configuration as cfg
from kinodata.types import NodeType
from prob.builds_and_runs import build_gnn_model, build_kd_ds
from prob.generate_prob_ds import ProbingJobSpec, _validate_spec, set_probing_config
from prob.paths_and_io import get_out_dir, save_out_tensor
from prob.resloves_and_transforms import aggregate_folds, aggregate_ids, dtype_resolve

logger = logging.getLogger(__name__)


def _extract_layer0_fold(prob_config: cfg.Config) -> None:
    device = torch.device(prob_config.device or "cpu")
    dtype_out = dtype_resolve(prob_config.dtype_out)

    gnn_model = build_gnn_model(prob_config).eval().to(device)
    ds = build_kd_ds(split_path=prob_config.split_file)
    loader = DataLoader(ds, batch_size=prob_config.batch_size, shuffle=False)

    required = ("atomic_num_embedding", "lin_atom_features", "act", "aggr")
    if not all(hasattr(gnn_model, attr) for attr in required):
        raise NotImplementedError(
            "layer_0 extraction currently supports ComplexTransformer-style models only."
        )

    fold = prob_config.split_index
    layer0_bufs = []
    idents = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extracting layer_0 fold {fold}"):
            idents.extend(batch.ident.tolist())
            batch = batch.to(device)

            node_store = batch[NodeType.Complex]
            node_repr = gnn_model.act(
                gnn_model.atomic_num_embedding(node_store.z)
                + gnn_model.lin_atom_features(node_store.x)
            )
            graph_repr = gnn_model.aggr(node_repr, node_store.batch).detach().cpu()
            if dtype_out is not None:
                graph_repr = graph_repr.to(dtype_out)
            layer0_bufs.append(graph_repr)

    layer0 = torch.cat(layer0_bufs, dim=0)
    ids_tensor = torch.tensor(idents, dtype=torch.long)

    if layer0.shape[0] != len(ds):
        raise RuntimeError(
            f"layer_0 size {layer0.shape[0]} does not match fold size {len(ds)} for fold {fold}"
        )
    if ids_tensor.shape[0] != len(ds):
        raise RuntimeError(
            f"ids size {ids_tensor.shape[0]} does not match fold size {len(ds)} for fold {fold}"
        )

    fold_dir = Path(prob_config.output_dir) / str(fold)
    save_out_tensor(layer0, fold_dir, f"layer_0_{fold}.pt")
    save_out_tensor(ids_tensor, fold_dir, f"ids_{fold}.pt")
    logger.info("Saved fold %s layer_0 and ids to %s", fold, fold_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract layer_0 representations")
    parser.add_argument("--gnn_model_type", default="CGNN-3D", choices=["CGNN-3D", "CGNN", "DTI"])
    parser.add_argument(
        "--split_type",
        default="random-k-fold",
        choices=["random-k-fold", "scaffold-k-fold", "pocket-k-fold"],
    )
    parser.add_argument("--filter_rmsd_max_value", default=2, type=float)
    parser.add_argument("--k_fold", default=5, type=int)
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--dtype_out", default=None, type=str)
    args = parser.parse_args()

    spec = ProbingJobSpec(
        gnn_model_type=args.gnn_model_type,
        split_type=args.split_type,
        filter_rmsd_max_value=int(args.filter_rmsd_max_value)
        if args.filter_rmsd_max_value is not None
        else None,
        graph_level=True,
        device=args.device,
        dtype_out=args.dtype_out,
        k_fold=args.k_fold,
        seed=args.seed,
    )
    _validate_spec(spec)

    output_root_dir = get_out_dir(
        spec.gnn_model_type,
        spec.filter_rmsd_max_value,
        spec.split_type,
        split_fold=None,
    )

    for fold in range(spec.k_fold):
        logger.info("Extracting fold %s/%s", fold + 1, spec.k_fold)
        prob_config = set_probing_config(
            gnn_model_type=spec.gnn_model_type,
            split_type=spec.split_type,
            filter_rmsd_max_value=spec.filter_rmsd_max_value,
            graph_level=True,
            split_index=fold,
            dtype_out=spec.dtype_out,
            device=spec.device,
            k_fold=spec.k_fold,
            seed=spec.seed,
        )
        _extract_layer0_fold(prob_config)

    agg_cfg = cfg.Config({"output_dir": output_root_dir, "k_fold": spec.k_fold})
    aggregate_folds(agg_cfg, "layer_0")
    aggregate_ids(agg_cfg)
    logger.info("Aggregated layer_0 and ids under %s", output_root_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
