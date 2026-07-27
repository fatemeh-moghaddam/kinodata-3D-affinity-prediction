from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import wandb

import kinodata.configuration as cfg

from prob.repr_extraction_utils import build_gnn_model, build_kd_ds, run_fold
from prob.paths_and_io import (
    get_gnn_config_path,
    get_model_ckpt,
    get_model_dir,
    get_out_dir,
    get_split_file,
)
from prob.resloves_and_transforms import (
    aggregate_folds,
    aggregate_ids,
    aggregate_predictions,
    load_config,
)

"""
Generate probing dataset representations for a trained GNN across CV folds.

Key design goals:
- Explicit "job spec" (human-chosen settings) vs derived paths/config.
- Deterministic, reproducible runs with a manifest for later baselines/stats.
- Minimal hidden side-effects: CLI parsing happens only in `__main__`.
"""

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbingJobSpec:
    gnn_model_type: str = "CGNN-3D"
    split_type: str = "random-k-fold"
    filter_rmsd_max_value: int | float | None = 2
    graph_level: bool = True
    device: str = "auto"
    dtype_out: str | None = None
    k_fold: int = 5
    seed: int = 96
    wandb_mode: str = "disabled"
    save_representations: bool = True
    save_predictions: bool = False


@dataclass(frozen=True)
class ResolvedPaths:
    model_dir: Path
    model_ckpt: Path
    split_file: Path
    output_root_dir: Path
    gnn_config_path: Path


def _validate_spec(spec: ProbingJobSpec) -> None:
    if spec.gnn_model_type not in {"CGNN-3D", "CGNN", "DTI"}:
        raise ValueError(f"Invalid gnn_model_type: {spec.gnn_model_type}")
    if spec.split_type not in {"random-k-fold", "scaffold-k-fold", "pocket-k-fold"}:
        raise ValueError(f"Invalid split_type: {spec.split_type}")
    if spec.filter_rmsd_max_value not in {2, 4, 6, 2.00, 4.00, 6.00, None}:
        raise ValueError(f"Invalid filter_rmsd_max_value: {spec.filter_rmsd_max_value}")
    if spec.k_fold <= 0:
        raise ValueError("k_fold must be positive")


def resolve_paths(spec: ProbingJobSpec, fold: int) -> ResolvedPaths:
    model_dir = get_model_dir(
        rmsd_threshold=spec.filter_rmsd_max_value,
        split_type=spec.split_type,
        split_fold=fold,
        model_type=spec.gnn_model_type,
    )
    model_ckpt = get_model_ckpt(model_dir)
    split_file_path = get_split_file(spec.split_type, fold, spec.filter_rmsd_max_value)
    output_root_dir = get_out_dir(
        spec.gnn_model_type,
        spec.filter_rmsd_max_value,
        spec.split_type,
        split_fold=None,
    )
    gnn_config_path = get_gnn_config_path(model_dir)
    return ResolvedPaths(
        model_dir=model_dir,
        model_ckpt=model_ckpt,
        split_file=split_file_path,
        output_root_dir=output_root_dir,
        gnn_config_path=gnn_config_path,
    )


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


def expected_fold_artifacts(prob_config: cfg.Config, spec: ProbingJobSpec) -> list[Path]:
    """
    Files `run_fold` will write for this fold, given what the spec asks to save.
    Used to decide whether a fold is already done and can be skipped on a re-run.
    """
    fold = int(prob_config.split_index)
    fold_dir = Path(prob_config.output_dir) / str(fold)
    paths: list[Path] = []
    if spec.save_representations:
        num_layers = int(prob_config.get("num_attention_blocks", 3))
        paths += [fold_dir / f"layer_{i+1}_{fold}.pt" for i in range(num_layers)]
        paths.append(fold_dir / f"ids_{fold}.pt")
    if spec.save_predictions:
        paths += [fold_dir / f"preds_{fold}.pt", fold_dir / f"y_true_{fold}.pt"]
    return paths


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


def _get_git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def write_manifest(output_root_dir: Path, payload: dict[str, Any]) -> Path:
    output_root_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    return manifest_path



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
        device="auto",
        k_fold=5,
        seed=96,
        parse_args=False,  # keep CLI parsing out of this function by default
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
    if "config_name" in kwargs:
        config_name = kwargs["config_name"]
    else:
        config_name = "probing_ds_generation"

    # merge: kwargs overrides defaults
    config_args = {**defaults, **kwargs}
    cfg.register(config_name, **config_args)
    prob_config = cfg.get(config_name)
    if prob_config.parse_args:
        prob_config = prob_config.update_from_args()

    spec = ProbingJobSpec(
        gnn_model_type=prob_config.gnn_model_type,
        split_type=prob_config.split_type,
        filter_rmsd_max_value=prob_config.filter_rmsd_max_value,
        graph_level=prob_config.graph_level,
        device=prob_config.device,
        dtype_out=prob_config.dtype_out,
        k_fold=prob_config.k_fold,
        seed=prob_config.seed,
    )
    _validate_spec(spec)

    paths = resolve_paths(spec, fold=prob_config.split_index)
    model_config = load_config(paths.gnn_config_path)
    prob_config.update(
        {
            **model_config,
            "model_ckpt": paths.model_ckpt,
            "split_file": paths.split_file,
            "output_dir": paths.output_root_dir,
        },
        allow_duplicates=True,
    )
    return prob_config

    





if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Generate probing dataset representations for a trained GNN.")
    parser.add_argument("--gnn_model_type", default="CGNN-3D", choices=["CGNN-3D", "CGNN", "DTI"])
    parser.add_argument(
        "--split_type",
        default="random-k-fold",
        choices=["random-k-fold", "scaffold-k-fold", "pocket-k-fold"],
    )
    parser.add_argument("--filter_rmsd_max_value", default=2, type=float)
    parser.add_argument("--k_fold", default=5, type=int)
    parser.add_argument("--wandb_mode", default="disabled", choices=["disabled", "online", "offline"])
    parser.add_argument("--seed", default=96, type=int)
    parser.add_argument("--device", default="auto", type=str, help='"auto" (default), "cpu", "cuda", "cuda:0", ...')
    parser.add_argument("--dtype_out", default=None, type=str)
    parser.add_argument("--save_representations", default=1, type=int, choices=[0, 1])
    parser.add_argument("--save_predictions", default=0, type=int, choices=[0, 1])
    parser.add_argument(
        "--overwrite",
        default=0,
        type=int,
        choices=[0, 1],
        help="recompute folds whose artifacts already exist (default: skip them)",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    if device.startswith("cuda"):
        logger.info(
            "Using GPU: %s (torch %s, CUDA %s)",
            torch.cuda.get_device_name(0),
            torch.__version__,
            torch.version.cuda,
        )
    else:
        logger.warning("Running on CPU (device=%s); extraction will be slow.", device)

    spec = ProbingJobSpec(
        gnn_model_type=args.gnn_model_type,
        split_type=args.split_type,
        filter_rmsd_max_value=int(args.filter_rmsd_max_value) if args.filter_rmsd_max_value is not None else None,
        graph_level=True,
        device=device,
        dtype_out=args.dtype_out,
        k_fold=args.k_fold,
        seed=args.seed,
        save_representations=bool(args.save_representations),
        save_predictions=bool(args.save_predictions),
    )
    _validate_spec(spec)

    _seed_everything(spec.seed)
    wandb.init(mode=spec.wandb_mode)

    output_root_dir = get_out_dir(spec.gnn_model_type, spec.filter_rmsd_max_value, spec.split_type, split_fold=None)

    manifest_payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _get_git_commit(),
        "spec": asdict(spec),
        "folds": {},
        "artifacts": {"output_root_dir": str(output_root_dir)},
        "versions": {
            "python": os.sys.version,
            "torch": getattr(torch, "__version__", None),
        },
    }

    for fold in range(spec.k_fold):
        prob_config = set_probing_config(
            gnn_model_type=spec.gnn_model_type,
            split_type=spec.split_type,
            filter_rmsd_max_value=spec.filter_rmsd_max_value,
            graph_level=spec.graph_level,
            split_index=fold,
            dtype_out=spec.dtype_out,
            device=spec.device,
            k_fold=spec.k_fold,
            seed=spec.seed,
        )

        logger.info("Fold %s/%s", fold + 1, spec.k_fold)
        logger.info("Split file: %s", prob_config.split_file)
        logger.info("Checkpoint: %s", prob_config.model_ckpt)
        logger.info("Output root: %s", prob_config.output_dir)

        # Resume support: skip folds already on disk. Checked before build_kd_ds so a
        # skipped fold never pays the (large) dataset load.
        expected = expected_fold_artifacts(prob_config, spec)
        if not args.overwrite and expected and all(p.exists() for p in expected):
            fold_dir = Path(prob_config.output_dir) / str(fold)
            n_done = int(torch.load(expected[0], map_location="cpu").shape[0])
            logger.info(
                "Fold %s: artifacts already present (%s samples), skipping. "
                "Pass --overwrite 1 to recompute.", fold, n_done,
            )
            fold_paths = resolve_paths(spec, fold)
            manifest_payload["folds"][str(fold)] = {
                "model_dir": str(fold_paths.model_dir),
                "split_file": str(prob_config.split_file),
                "model_ckpt": str(prob_config.model_ckpt),
                "fold_output_dir": str(fold_dir),
                "num_samples": n_done,
                "reused": True,
            }
            continue

        gnn_model = build_gnn_model(prob_config).eval()
        if gnn_model is None:
            raise RuntimeError("Failed to build GNN model")

        ds = build_kd_ds(split_path=prob_config.split_file)
        if len(ds) == 0:
            raise ValueError("Probing dataset is empty")

        run_fold(
            ds,
            gnn_model,
            prob_config,
            save_representations=spec.save_representations,
            save_predictions=spec.save_predictions,
        )

        fold_paths = resolve_paths(spec, fold)
        manifest_payload["folds"][str(fold)] = {
            "model_dir": str(fold_paths.model_dir),
            "split_file": str(prob_config.split_file),
            "model_ckpt": str(prob_config.model_ckpt),
            "fold_output_dir": str(Path(prob_config.output_dir) / str(fold)),
            "num_samples": int(len(ds)),
        }

    agg_cfg = cfg.Config({"output_dir": output_root_dir, "k_fold": spec.k_fold})
    num_layers = int(prob_config.get("num_attention_blocks", 3))
    layer_names = [f"layer_{i+1}" for i in range(num_layers)]

    if spec.save_representations:
        for layer_name in layer_names:
            aggregate_folds(agg_cfg, layer_name)
        aggregate_ids(agg_cfg)
        manifest_payload["artifacts"].update(
            {
                "aggregated_ids": str(output_root_dir / "ids.pt"),
                "aggregated_layers": [str(output_root_dir / f"{layer_name}.pt") for layer_name in layer_names],
            }
        )

    if spec.save_predictions:
        aggregate_predictions(agg_cfg)
        manifest_payload["artifacts"].update(
            {"predictions": str(output_root_dir / "predictions.csv")}
        )

    write_manifest(output_root_dir, manifest_payload)

