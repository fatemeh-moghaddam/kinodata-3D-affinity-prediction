"""
Build the skyline probing targets -- affinity (graph.y) and docking_score --
in the same {ident: value} format and ident space as the other targets.

Keyed by graph.ident from the cached KinodataDocked, i.e. the same source as
ids.pt and nitrogen_counts. (The old skyline notebook used
KinodataDockedAgnostic, whose live `ident = df.index` numbering does not match
ids.pt, which silently shuffled the affinity target.)
"""

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

from kinodata.data import KinodataDocked
from kinodata.transform import TransformToComplexGraph

from prob.paths_and_io import get_data_dir, save_out_tensor


# docking.chemgauss_score encodes a failed/missing run as ~float32-max instead of NaN.
DOCKING_SENTINEL = np.float32(3.4028235e38)

AFFINITY_FILE = "affinity.pt"
DOCKING_SCORE_FILE = "docking_score.pt"


def _scalar(value) -> float:
    """Coerce a per-graph attribute (tensor[1], python scalar, str) to float."""
    if isinstance(value, torch.Tensor):
        return float(value.reshape(-1)[0].item())
    return float(value)


def extract_skyline_targets(
    dataset: Iterable[HeteroData],
    affinity_key: str = "y",
    docking_key: str = "docking_score",
) -> tuple[Dict[int, float], Dict[int, float]]:
    """
    Iterate the dataset once and collect, keyed by graph.ident:
        - affinity        (graph.y, the trained regression target)
        - docking_score   (graph.docking_score, sentinel cleaned to NaN)
    """
    affinity: Dict[int, float] = {}
    docking: Dict[int, float] = {}
    n_sentinel = 0

    for graph in tqdm(dataset, desc="extracting skyline targets"):
        ident = int(graph.ident.item())

        if hasattr(graph, affinity_key) and graph[affinity_key] is not None:
            affinity[ident] = _scalar(graph[affinity_key])

        if hasattr(graph, docking_key) and graph[docking_key] is not None:
            score = _scalar(graph[docking_key])
            if abs(score) >= DOCKING_SENTINEL:
                score = np.nan
                n_sentinel += 1
            docking[ident] = score

    print(
        f"Extracted {len(affinity)} affinity and {len(docking)} docking_score "
        f"values ({n_sentinel} docking sentinels -> NaN)."
    )
    return affinity, docking


def calculate_save_skyline_targets(
    dataset: Iterable[HeteroData],
    output_dir: Optional[Path] = None,
) -> tuple[Path, Path]:
    """
    Extract affinity + docking_score and save them, in the standard
    {ident: value} format, next to the other probing targets.

    output_dir defaults to data/probing/targets (the dir load_y_by_ids reads).
    """
    if output_dir is None:
        output_dir = get_data_dir(prob=True) / "targets"

    affinity, docking = extract_skyline_targets(dataset)

    aff_path = save_out_tensor(affinity, output_dir=output_dir, filename=AFFINITY_FILE)
    dock_path = save_out_tensor(
        docking, output_dir=output_dir, filename=DOCKING_SCORE_FILE
    )
    print(f"Saved affinity      -> {aff_path}")
    print(f"Saved docking_score -> {dock_path}")
    return aff_path, dock_path


if __name__ == "__main__":
    dataset = KinodataDocked(
        transform=TransformToComplexGraph(remove_heterogeneous_representation=False),
        use_multiprocessing=True,
        num_processes=16,
    )
    calculate_save_skyline_targets(dataset)
