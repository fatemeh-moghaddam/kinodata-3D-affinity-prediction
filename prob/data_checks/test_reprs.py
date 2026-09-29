"""
Checks of the extracted representations under data/probing. These read the
files on disk rather than test code: run them after an extraction and after
copying representations, before probing.

    pytest prob/data_checks                        # every condition found
    pytest prob/data_checks -k rmsd2               # one cutoff
    pytest prob/data_checks -k "DTI and scaffold"  # one model and split type
    pytest prob/data_checks -m "not folds"         # without the per-fold comparisons

Per condition (<gnn>/rmsd_cutoff_<x>/<split>/): the manifest, ids.pt, every
layer_*.pt, and (marker `folds`) that the aggregated files equal their per-fold
files concatenated in fold order, which per-checkpoint probing relies on.
Across conditions: every model and split type holds the same idents at a cutoff,
and the cutoffs nest.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
PROBING = ROOT / "data" / "probing"
PROCESSED = ROOT / "data" / "processed"
CATALOGUE = ROOT / "data" / "ident_to_activity_id.csv"
RMSD2_MAPPING = ROOT / "data" / "ident_activity_id_index_mapping.csv"
#: Size of each cutoff's filtered dataset, used when its split CSV is not on disk.
KNOWN_ROWS = {2: 41238, 4: 58300, 6: 93406}
#: A row whose largest |value| is this many times the median row's counts as
#: extreme (damaged copies showed values ~15x the normal range).
EXTREME_FACTOR = 5.0


@dataclass(frozen=True)
class Condition:
    gnn: str
    rmsd: int
    split: str
    path: Path

    @property
    def id(self) -> str:
        return f"{self.gnn}-rmsd{self.rmsd}-{self.split.removesuffix('-k-fold')}"


def _discover() -> list[Condition]:
    found = []
    for d in sorted(PROBING.glob("*/rmsd_cutoff_*/*-k-fold")):
        if (d / "ids.pt").exists():
            rmsd = int(float(d.parent.name.removeprefix("rmsd_cutoff_")))
            found.append(Condition(d.parent.parent.name, rmsd, d.name, d))
    return found


CONDITIONS = _discover()
LAYERS = [
    (c, int(p.stem.split("_")[1]))
    for c in CONDITIONS
    for p in sorted(c.path.glob("layer_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
]
CUTOFFS = sorted({c.rmsd for c in CONDITIONS})

pytestmark = pytest.mark.skipif(not CONDITIONS, reason=f"no extracted representations under {PROBING}")
by_condition = pytest.mark.parametrize("cond", CONDITIONS, ids=lambda c: c.id)
by_layer = pytest.mark.parametrize(
    "cond,layer", LAYERS, ids=[f"{c.id}-L{layer}" for c, layer in LAYERS]
)


# ─────────────────────────────────────────────────────────────
# Loading (cached: several tests read the same files)
# ─────────────────────────────────────────────────────────────


def _load(path: Path) -> np.ndarray:
    return torch.load(path, map_location="cpu").detach().cpu().numpy()


@lru_cache(maxsize=None)
def ids_of(path: Path) -> np.ndarray:
    return _load(path / "ids.pt").astype(np.int64).ravel()


def fold_dirs(path: Path) -> list[int]:
    return sorted(int(p.name) for p in path.iterdir() if p.is_dir() and p.name.isdigit())


@lru_cache(maxsize=None)
def catalogue() -> np.ndarray:
    return pd.read_csv(CATALOGUE, usecols=["ident_processed"])["ident_processed"].to_numpy()


@lru_cache(maxsize=None)
def expected_rows(split: str, rmsd: int) -> int | None:
    """Size of the cutoff's filtered dataset: from its split CSV (named like
    get_split_file expects, e.g. 1_5.csv or 1:5.csv), else KNOWN_ROWS."""
    csvs = sorted((PROCESSED / f"filter_predicted_rmsd_le{rmsd:.2f}" / split).glob("1*5.csv"))
    if csvs:
        return sum(1 for line in csvs[0].read_bytes().split(b"\n")[1:] if line.strip())
    return KNOWN_ROWS.get(rmsd)


# ─────────────────────────────────────────────────────────────
# Per condition
# ─────────────────────────────────────────────────────────────


@by_condition
def test_manifest_was_extracted_with_val(cond: Condition):
    manifest = cond.path / "manifest.json"
    assert manifest.exists(), "no manifest.json: the extraction did not finish"
    spec = json.loads(manifest.read_text()).get("spec", {})
    assert spec.get("include_val") is True, (
        f"include_val={spec.get('include_val')}: only the GNN folds' test molecules were "
        "extracted (half the cutoff), so split types hold different molecules"
    )


@by_condition
def test_ids_are_unique_and_non_negative(cond: Condition):
    ids = ids_of(cond.path)
    assert len(np.unique(ids)) == len(ids), f"{len(ids) - len(np.unique(ids))} duplicate idents"
    assert (ids >= 0).all(), f"{(ids < 0).sum()} negative idents"


@by_condition
def test_ids_cover_the_whole_cutoff(cond: Condition):
    expected = expected_rows(cond.split, cond.rmsd)
    if expected is None:
        pytest.skip(f"no split CSV and no known size for rmsd {cond.rmsd}")
    assert len(ids_of(cond.path)) == expected


@by_condition
def test_ids_are_not_the_positional_indexing_bug(cond: Condition):
    """The pre-fix extraction indexed the unfiltered dataset with filtered positions,
    which yields exactly its first N idents."""
    ids = ids_of(cond.path)
    first_n = catalogue()[: len(ids)]
    assert set(ids.tolist()) != set(first_n.tolist())


@by_condition
def test_ids_are_inside_the_rmsd2_dataset(cond: Condition):
    if cond.rmsd != 2 or not RMSD2_MAPPING.exists():
        pytest.skip("only checkable at rmsd 2")
    allowed = set(pd.read_csv(RMSD2_MAPPING, usecols=["ident_processed"])["ident_processed"])
    outside = set(ids_of(cond.path).tolist()) - allowed
    assert not outside, f"{len(outside)} idents outside the RMSD <= 2 dataset"


@pytest.mark.folds
@by_condition
def test_ids_equal_concatenated_fold_ids(cond: Condition):
    folds = fold_dirs(cond.path)
    assert folds, "no fold dirs; copy them with INCLUDE_FOLDS=1 bash prob/local/copy_reprs.sh"
    fold_ids = np.concatenate([_load(cond.path / str(k) / f"ids_{k}.pt").ravel() for k in folds])
    assert np.array_equal(fold_ids, ids_of(cond.path))


# ─────────────────────────────────────────────────────────────
# Per layer
# ─────────────────────────────────────────────────────────────


@by_layer
def test_layer_rows_are_finite_and_in_range(cond: Condition, layer: int):
    X = _load(cond.path / f"layer_{layer}.pt")
    assert X.shape[0] == len(ids_of(cond.path)), "one row per ident in ids.pt"
    finite = np.isfinite(X)
    assert finite.all(), f"{(~finite.all(1)).sum()} rows with NaN/inf"
    row_max = np.abs(X).max(1)
    extreme = row_max > EXTREME_FACTOR * max(float(np.median(row_max)), 1e-12)
    assert not extreme.any(), (
        f"{extreme.sum()} extreme rows (max |x| {row_max.max():.3g}, median {np.median(row_max):.3g})"
    )


@pytest.mark.folds
@by_layer
def test_layer_equals_concatenated_fold_files(cond: Condition, layer: int):
    folds = fold_dirs(cond.path)
    assert folds, "no fold dirs; copy them with INCLUDE_FOLDS=1 bash prob/local/copy_reprs.sh"
    stacked = np.concatenate([_load(cond.path / str(k) / f"layer_{layer}_{k}.pt") for k in folds])
    assert np.array_equal(stacked, _load(cond.path / f"layer_{layer}.pt"), equal_nan=True)


# ─────────────────────────────────────────────────────────────
# Across conditions
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("rmsd", CUTOFFS, ids=lambda r: f"rmsd{r}")
def test_same_idents_for_every_model_and_split(rmsd: int):
    sets = {c.id: set(ids_of(c.path).tolist()) for c in CONDITIONS if c.rmsd == rmsd}
    reference_id, reference = next(iter(sets.items()))
    differ = [cid for cid, s in sets.items() if s != reference]
    assert not differ, (
        f"{differ} hold different idents than {reference_id} "
        f"(shared by all {len(sets)}: {len(set.intersection(*sets.values()))})"
    )


@pytest.mark.parametrize(
    "lower,higher", list(combinations(CUTOFFS, 2)), ids=lambda r: f"rmsd{r}"
)
def test_cutoffs_nest(lower: int, higher: int):
    lower_ids = set().union(*(ids_of(c.path).tolist() for c in CONDITIONS if c.rmsd == lower))
    higher_ids = set().union(*(ids_of(c.path).tolist() for c in CONDITIONS if c.rmsd == higher))
    missing = lower_ids - higher_ids
    assert not missing, f"{len(missing)} rmsd-{lower} idents are missing at rmsd {higher}"
