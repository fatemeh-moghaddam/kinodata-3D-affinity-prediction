"""
Aggregate `layer_0` for extraction runs that predate it being aggregated.

`ComplexTransformer` has always reported `layer_0` (the atom embedding, before any
attention block) and `_write_fold_artifacts` saves every layer it is given, so every
fold directory of every finished run already holds `layer_0_<fold>.pt`. Only the
concatenation step was skipped: `run_extraction.py` aggregated `layer_1..N` and left
`layer_0.pt` unwritten.

This script does that concatenation and nothing else. It does not load the dataset,
build a model, or need a GPU -- unlike `extract_layer0.py`, which recomputes the
representations from scratch and is unnecessary when the fold files are on disk.

    python prob/aggregate_layer0.py              # every run under data/probing
    python prob/aggregate_layer0.py --output_dir data/probing/CGNN/rmsd_cutoff_2/random-k-fold
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch

import kinodata.configuration as cfg

from prob.paths_and_io import get_data_dir
from prob.resloves_and_transforms import aggregate_folds

logger = logging.getLogger(__name__)


def find_run_dirs(probing_root: Path) -> list[Path]:
    """Extraction output roots, i.e. data/probing/<model>/rmsd_cutoff_<r>/<split>."""
    return sorted(probing_root.glob("*/rmsd_cutoff_*/*-k-fold"))


def backfill_layer(
    output_dir: Path,
    layer_name: str = "layer_0",
    k_fold: int = 5,
    overwrite: bool = False,
) -> str:
    """
    Concatenate `<layer_name>_<fold>.pt` across folds into `<layer_name>.pt`.

    Refuses to write a partial tensor: every fold file must be present, and the
    result must have as many rows as `ids.pt`. A short `layer_0.pt` would silently
    misalign X against the y that `load_y_by_ids` builds from those same ids.
    """
    target = output_dir / f"{layer_name}.pt"
    if target.exists() and not overwrite:
        return "already aggregated"

    fold_files = [output_dir / str(fold) / f"{layer_name}_{fold}.pt" for fold in range(k_fold)]
    missing = [p for p in fold_files if not p.exists()]
    if missing:
        return f"skipped, {len(missing)}/{k_fold} fold files missing"

    aggregate_folds(cfg.Config({"output_dir": output_dir, "k_fold": k_fold}), layer_name)

    n_rows = int(torch.load(target, map_location="cpu").shape[0])
    ids_path = output_dir / "ids.pt"
    if ids_path.exists():
        n_ids = int(torch.load(ids_path, map_location="cpu").shape[0])
        if n_rows != n_ids:
            target.unlink()
            raise RuntimeError(
                f"{target} has {n_rows} rows but ids.pt has {n_ids}; refusing to leave "
                "a tensor that does not line up with the ids. Check the fold files."
            )
    return f"aggregated {n_rows} rows"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate layer_0 from existing fold artifacts")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="single extraction output root; default is every run under data/probing",
    )
    parser.add_argument("--layer_name", default="layer_0")
    parser.add_argument("--k_fold", default=5, type=int)
    parser.add_argument("--overwrite", default=0, type=int, choices=[0, 1])
    args = parser.parse_args()

    if args.output_dir:
        run_dirs = [Path(args.output_dir)]
    else:
        run_dirs = find_run_dirs(get_data_dir(prob=True))
        if not run_dirs:
            logger.warning("No extraction runs found under %s", get_data_dir(prob=True))

    for run_dir in run_dirs:
        status = backfill_layer(run_dir, args.layer_name, args.k_fold, bool(args.overwrite))
        print(f"[layer0] {run_dir}: {status}", flush=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
