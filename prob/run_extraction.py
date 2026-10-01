"""
Generate the probing dataset: run a trained GNN over each CV fold's test
molecules and save its per-layer graph representations (and, optionally, its
affinity predictions), then aggregate the folds.

What a job does is prob_config.ExtractionSpec -- every setting, its default and
its command-line flag, and the fixed values (test molecules only, 5 folds, seed).
This module only carries it out:

- per fold: find the fold's one checkpoint and its config.json, check that the
  checkpoint was trained for this split / fold / cutoff, build the model and the
  fold's test set, write the fold's artifacts (resuming: folds already on disk
  are skipped unless overwrite);
- then concatenate the folds in fold order and write manifest.json: the spec
  with each setting's source, every fold's checkpoint, config and split file
  with their sha256, the code version and package versions.

    python prob/run_extraction.py --gnn_model_type CGNN-3D \\
        --split_type random-k-fold --filter_rmsd_max_value 2 --device cuda
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import wandb

import kinodata.configuration as cfg

from prob import run_manifest
from prob.paths_and_io import (
    checkpoint_model_type,
    get_gnn_config_path,
    get_model_ckpt,
    get_model_dir,
    get_out_dir,
    get_split_file,
)
from prob.prob_config import (
    ExtractionComputeSettings,
    ExtractionDataSettings,
    ExtractionSpec,
    resolve_extraction_spec,
    spec_record,
)
from prob.repr_extraction_utils import build_gnn_model, build_kd_ds, run_fold
from prob.resloves_and_transforms import (
    aggregate_folds,
    aggregate_ids,
    aggregate_predictions,
    load_config,
)

logger = logging.getLogger(__name__)

_ROOT = Path(os.environ.get("HOME_PROJ_DIR", Path(__file__).resolve().parents[1]))


# ─────────────────────────────────────────────────────────────
# Per fold: paths, checkpoint, config
# ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResolvedPaths:
    model_dir: Path
    model_ckpt: Path
    split_file: Path
    output_root_dir: Path
    gnn_config_path: Path


def resolve_paths(spec: ExtractionSpec, fold: int) -> ResolvedPaths:
    # Weights come from `checkpoint_model_type` (a readout variant reuses another
    # model's checkpoint); outputs still go to this type's own tree.
    data = spec.data
    model_dir = get_model_dir(
        rmsd_threshold=data.filter_rmsd_max_value,
        split_type=data.split_type,
        split_fold=fold,
        model_type=checkpoint_model_type(data.gnn_model_type),
    )
    return ResolvedPaths(
        model_dir=model_dir,
        model_ckpt=get_model_ckpt(model_dir),
        split_file=get_split_file(data.split_type, fold, data.filter_rmsd_max_value),
        output_root_dir=get_out_dir(
            data.gnn_model_type, data.filter_rmsd_max_value, data.split_type, split_fold=None,
        ),
        gnn_config_path=get_gnn_config_path(model_dir),
    )


def _same(a: Any, b: Any) -> bool:
    """Equal, treating 2 / 2.0 / "2" alike and "None" as None (config.json's spelling)."""
    a = None if a == "None" else a
    b = None if b == "None" else b
    if a is None or b is None:
        return a is b
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return a == b


def check_checkpoint_config(model_config: dict, spec: ExtractionSpec, fold: int, source: Path) -> None:
    """
    The checkpoint must have been trained for the condition being extracted.
    It only *describes* the model (architecture, training seed, batch size); it
    never overrides what the job asked for. A mismatch means a checkpoint sits in
    the wrong directory, which would otherwise silently decide what is extracted.
    """
    expected = {
        "split_type": spec.data.split_type,
        "split_index": fold,
        "k_fold": ExtractionDataSettings.K_FOLD,
        "filter_rmsd_max_value": spec.data.filter_rmsd_max_value,
    }
    mismatches = {
        key: {"checkpoint": model_config[key], "requested": want}
        for key, want in expected.items()
        if key in model_config and not _same(model_config[key], want)
    }
    if mismatches:
        raise ValueError(f"Checkpoint config {source} does not match the extraction: {mismatches}")


def build_fold_config(spec: ExtractionSpec, fold: int, device: str) -> cfg.Config:
    """
    The config run_fold and build_gnn_model read for one fold: the checkpoint's
    config.json (architecture, batch_size, ...), checked against the spec, then the
    extraction's own choices on top.
    """
    paths = resolve_paths(spec, fold)
    model_config = load_config(paths.gnn_config_path)
    check_checkpoint_config(model_config, spec, fold, paths.gnn_config_path)
    return cfg.Config(dict(model_config)).update({
        "gnn_model_type": spec.data.gnn_model_type,
        "split_index": fold,
        "graph_level": ExtractionDataSettings.GRAPH_LEVEL,
        "emit_tower_reprs": spec.outputs.emit_tower_reprs,
        "dtype_out": spec.outputs.dtype_out,
        "device": device,
        "infer_batch_size": spec.compute.infer_batch_size,
        "eval_num_workers": spec.compute.eval_num_workers,
        "num_processes": spec.compute.num_processes,
        "model_ckpt": paths.model_ckpt,
        "split_file": paths.split_file,
        "output_dir": paths.output_root_dir,
    })


# ─────────────────────────────────────────────────────────────
# What a fold writes
# ─────────────────────────────────────────────────────────────


def probe_layer_names(prob_config: cfg.Config, gnn_model_type: str) -> list[str]:
    """
    Names of the graph-representation artifacts a fold writes, for this model.

    Every model reports `layer_0` -- the embedding before any message passing.
    CGNN/CGNN-3D then report one representation per attention block, up to `layer_N`.

    DTI has two towers of different depth. It reports a depth-aligned joint
    representation per ligand-tower layer, which is what the downstream probing reads
    and is directly comparable to the CGNN layers. The per-tower representations
    behind them are written only when `emit_tower_reprs` is set, for branch-resolved
    analysis; this must stay in step with `DTIModel._forward_prob`, since a name
    listed here that the model never emits makes the resume check wait on a file that
    will never appear.
    """
    if checkpoint_model_type(gnn_model_type) == "DTI":
        num_ligand = int(prob_config.get("num_layers", 3))
        num_pocket = int(prob_config.get("num_attention_blocks", 2))
        names = [f"layer_{i}" for i in range(num_ligand + 1)]
        if not prob_config.get("emit_tower_reprs", False):
            return names
        return (
            names
            + [f"ligand_layer_{i}" for i in range(num_ligand + 1)]
            + [f"pocket_layer_{i}" for i in range(num_pocket + 1)]
        )
    num_layers = int(prob_config.get("num_attention_blocks", 3))
    return [f"layer_{i}" for i in range(num_layers + 1)]


def expected_fold_artifacts(fold_config: cfg.Config, spec: ExtractionSpec) -> list[Path]:
    """
    Files `run_fold` will write for this fold, given what the spec asks to save.
    Used to decide whether a fold is already done and can be skipped on a re-run.
    """
    fold = int(fold_config.split_index)
    fold_dir = Path(fold_config.output_dir) / str(fold)
    paths: list[Path] = []
    if spec.outputs.save_representations:
        layer_names = probe_layer_names(fold_config, spec.data.gnn_model_type)
        paths += [fold_dir / f"{name}_{fold}.pt" for name in layer_names]
        paths.append(fold_dir / f"ids_{fold}.pt")
    if spec.outputs.save_predictions:
        paths += [fold_dir / f"preds_{fold}.pt", fold_dir / f"y_true_{fold}.pt"]
    return paths


# ─────────────────────────────────────────────────────────────
# Run helpers
# ─────────────────────────────────────────────────────────────


def resolve_device(requested: str | None) -> str:
    """
    Turn a requested device string into a concrete one.

    "auto" (or None) picks CUDA when it is actually available, else CPU. An explicit
    "cuda" request that cannot be honoured is a hard error: silently falling back to
    CPU is how a GPU job ends up running for hours on the wrong device.
    """
    if requested in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={requested!r} requested but torch.cuda.is_available() is False "
            f"(torch {getattr(torch, '__version__', '?')}). Check that the job was "
            "granted a GPU and that the image's CUDA build matches the host driver."
        )
    return requested


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # optional dependency in some environments

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_manifest(output_root_dir: Path, payload: dict[str, Any]) -> Path:
    output_root_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    return manifest_path


def _fold_record(paths: ResolvedPaths, fold_dir: Path, num_samples: int, reused: bool) -> dict[str, Any]:
    return {
        "model_dir": str(paths.model_dir),
        "checkpoint": run_manifest.file_record(paths.model_ckpt),
        "checkpoint_config": run_manifest.file_record(paths.gnn_config_path),
        "split_file": run_manifest.file_record(paths.split_file),
        "fold_output_dir": str(fold_dir),
        "num_samples": num_samples,
        "reused": reused,
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────


def main(argv: Optional[Sequence[str]] = None) -> Path:
    """Run one extraction job from its command line; returns the manifest path."""
    spec, sources = resolve_extraction_spec(argv)
    data, outputs = spec.data, spec.outputs

    device = resolve_device(spec.compute.device)
    if device.startswith("cuda"):
        logger.info(
            "Using GPU: %s (torch %s, CUDA %s)",
            torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda,
        )
    else:
        logger.warning("Running on CPU (device=%s); extraction will be slow.", device)

    _seed_everything(ExtractionComputeSettings.SEED)
    wandb.init(mode=spec.compute.wandb_mode)

    output_root_dir = get_out_dir(
        data.gnn_model_type, data.filter_rmsd_max_value, data.split_type, split_fold=None,
    )
    git = run_manifest.git_info(_ROOT)
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git["commit"],
        "spec": spec_record(spec, sources),
        "resolved": {"device": device},
        "folds": {},
        "artifacts": {"output_root_dir": str(output_root_dir)},
        "code": {"git": git, "versions": run_manifest.package_versions()},
        "env": run_manifest.recorded_env(),
        "argv": list(argv) if argv is not None else None,
    }

    fold_config = None
    for fold in range(ExtractionDataSettings.K_FOLD):
        fold_config = build_fold_config(spec, fold, device)
        paths = resolve_paths(spec, fold)
        fold_dir = Path(fold_config.output_dir) / str(fold)

        logger.info("Fold %s/%s", fold + 1, ExtractionDataSettings.K_FOLD)
        logger.info("Split file: %s", fold_config.split_file)
        logger.info("Checkpoint: %s", fold_config.model_ckpt)
        logger.info("Output root: %s", fold_config.output_dir)

        # Resume support: skip folds already on disk. Checked before build_kd_ds so a
        # skipped fold never pays the (large) dataset load.
        expected = expected_fold_artifacts(fold_config, spec)
        if not outputs.overwrite and expected and all(p.exists() for p in expected):
            n_done = int(torch.load(expected[0], map_location="cpu").shape[0])
            logger.info(
                "Fold %s: artifacts already present (%s samples), skipping. "
                "Pass --overwrite 1 to recompute.", fold, n_done,
            )
            manifest["folds"][str(fold)] = _fold_record(paths, fold_dir, n_done, reused=True)
            continue

        gnn_model = build_gnn_model(fold_config).eval()
        ds = build_kd_ds(
            split_path=fold_config.split_file,
            filter_rmsd_max_value=data.filter_rmsd_max_value,
            include_val=ExtractionDataSettings.INCLUDE_VAL,
            num_processes=spec.compute.num_processes,
        )
        if len(ds) == 0:
            raise ValueError("Probing dataset is empty")

        run_fold(
            ds,
            gnn_model,
            fold_config,
            save_representations=outputs.save_representations,
            save_predictions=outputs.save_predictions,
        )
        manifest["folds"][str(fold)] = _fold_record(paths, fold_dir, int(len(ds)), reused=False)

    agg_cfg = cfg.Config({"output_dir": output_root_dir, "k_fold": ExtractionDataSettings.K_FOLD})
    layer_names = probe_layer_names(fold_config, data.gnn_model_type)

    if outputs.save_representations:
        for layer_name in layer_names:
            aggregate_folds(agg_cfg, layer_name)
        aggregate_ids(agg_cfg)
        manifest["artifacts"].update(
            {
                "aggregated_ids": str(output_root_dir / "ids.pt"),
                "aggregated_layers": [str(output_root_dir / f"{name}.pt") for name in layer_names],
            }
        )

    if outputs.save_predictions:
        aggregate_predictions(agg_cfg)
        manifest["artifacts"].update({"predictions": str(output_root_dir / "predictions.csv")})

    return write_manifest(output_root_dir, manifest)


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main(sys.argv[1:])
