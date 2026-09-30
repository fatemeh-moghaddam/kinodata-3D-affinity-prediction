"""
Shared fixtures: a synthetic extraction output that looks like one condition's
data/probing/<gnn>/rmsd_cutoff_<x>/<split>/ directory, and the probe split file
that goes with it.

Nothing here reads or writes the real data/ tree. The autouse guard points the
default probe split at tmp_path and makes the catalogue unreadable, so a test
that forgets to set up its own split fails instead of silently creating
data/probing/probe_split.csv.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import torch

#: Env vars prob_orchestrate reads; cleared so the caller's shell cannot steer a test.
PROB_ENV = (
    "PROB_LAYERS",
    "PROB_REUSE_BEST_PARAMS",
    "PROB_BEST_PARAMS_CACHE_DIR",
    "PROB_RUN_SHUFFLED_BASELINE",
    "PROB_RUN_LINEAR_MODELS",
    "PROB_RUN_NON_LINEAR_MODELS",
    "PROB_NONLINEAR_MODELS",
    "PROB_BASELINE_TAG",
)


@pytest.fixture(autouse=True)
def isolate_from_real_data(tmp_path, monkeypatch):
    for key in PROB_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        "prob.prob_run.default_probe_split_path", lambda: tmp_path / "default_probe_split.csv"
    )

    def _no_catalogue(*_args, **_kwargs):
        raise RuntimeError("tests must not read the real ident catalogue; write a probe split first")

    monkeypatch.setattr("prob.prob_run.catalogue_idents", _no_catalogue)


# ─────────────────────────────────────────────────────────────
# Synthetic condition
# ─────────────────────────────────────────────────────────────


@dataclass
class World:
    """One synthetic condition on disk plus the ground truth it was built from."""

    out_dir: Path              # like data/probing/<gnn>/rmsd_cutoff_<x>/<split>
    target_dir: Path           # like data/probing/targets
    target_file: str
    probe_split_path: Path
    idents: np.ndarray         # row order of ids.pt
    folds: np.ndarray          # checkpoint of each row
    layers: dict[int, np.ndarray]
    targets: dict[int, float]  # what affinity.pt holds (missing idents are absent)
    test_idents: set[int]      # probe test idents, over the whole catalogue

    def layer(self, n: int) -> np.ndarray:
        return self.layers[n]

    @property
    def y(self) -> np.ndarray:
        return np.array([self.targets.get(int(i), np.nan) for i in self.idents])


def build_world(
    root: Path,
    *,
    n_folds: int = 5,
    rows_per_fold: int = 200,
    n_features: int = 8,
    missing_targets: int = 2,
    seed: int = 0,
) -> World:
    """
    Write a condition whose layer_1 carries the target and whose layer_0 does not.

    Every checkpoint sees the same latent z through its own random rotation, as
    separately trained GNNs do: layer_1 of fold k is z @ R_k. y = z @ w, so a
    probe per checkpoint recovers y and one probe over all folds cannot.

    Idents are spaced (not row positions) and shuffled inside each fold, so any
    code that splits or aligns by position rather than by ident gets caught.
    Exactly 10% of each fold is in the probe test set, which keeps
    split_by_ident's fraction check satisfied after a couple of NaN targets.
    """
    rng = np.random.default_rng(seed)
    n = n_folds * rows_per_fold
    out_dir = root / "probing" / "CGNN-3D" / "rmsd_cutoff_2" / "random-k-fold"
    target_dir = root / "probing" / "targets"
    out_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)

    positions = np.arange(n)
    idents_sorted = 1000 + 7 * positions
    is_test = positions % 10 == 0
    folds = positions // rows_per_fold

    # Shuffle rows inside each fold: ids.pt is not sorted in real extractions either.
    order = np.concatenate([
        rng.permutation(np.flatnonzero(folds == k)) for k in range(n_folds)
    ])
    idents = idents_sorted[order]
    folds = folds[order]

    z = rng.normal(size=(n, n_features))
    w = rng.normal(size=n_features)
    y = z @ w + 0.05 * rng.normal(size=n)

    layer_1 = np.empty_like(z)
    for k in range(n_folds):
        rotation, _ = np.linalg.qr(rng.normal(size=(n_features, n_features)))
        rows = folds == k
        layer_1[rows] = z[rows] @ rotation
    layers = {0: rng.normal(size=(n, n_features)), 1: layer_1}

    targets = {int(i): float(v) for i, v in zip(idents, y)}
    # Drop a few targets (NaN after load_y_by_ids): one test and train ident first.
    test_rows = np.flatnonzero(np.isin(idents, idents_sorted[is_test]))
    train_rows = np.flatnonzero(~np.isin(idents, idents_sorted[is_test]))
    for row in [*test_rows[:missing_targets // 2], *train_rows[:missing_targets - missing_targets // 2]]:
        targets.pop(int(idents[row]))

    for k in range(n_folds):
        rows = folds == k
        fold_dir = out_dir / str(k)
        fold_dir.mkdir()
        torch.save(torch.as_tensor(idents[rows], dtype=torch.long), fold_dir / f"ids_{k}.pt")
        for num, X in layers.items():
            torch.save(torch.as_tensor(X[rows], dtype=torch.float32), fold_dir / f"layer_{num}_{k}.pt")
    torch.save(torch.as_tensor(idents, dtype=torch.long), out_dir / "ids.pt")
    for num, X in layers.items():
        torch.save(torch.as_tensor(X, dtype=torch.float32), out_dir / f"layer_{num}.pt")
    torch.save(targets, target_dir / "affinity.pt")

    # The split file covers more than this condition, like the real catalogue does.
    extra = 1000 + 7 * np.arange(n, n + 100)
    catalogue = np.concatenate([idents_sorted, extra])
    catalogue_test = np.concatenate([is_test, np.arange(100) % 10 == 0])
    probe_split_path = root / "probing" / "probe_split.csv"
    pd.DataFrame({
        "ident": catalogue,
        "probe_split": np.where(catalogue_test, "test", "train"),
    }).to_csv(probe_split_path, index=False)

    return World(
        out_dir=out_dir,
        target_dir=target_dir,
        target_file="affinity.pt",
        probe_split_path=probe_split_path,
        idents=idents,
        folds=folds,
        layers={num: X.astype(np.float32) for num, X in layers.items()},
        targets=targets,
        test_idents=set(catalogue[catalogue_test].tolist()),
    )


@pytest.fixture
def world(tmp_path, monkeypatch) -> World:
    """A synthetic condition, with its split installed as the default probe split."""
    w = build_world(tmp_path)
    monkeypatch.setattr("prob.prob_run.default_probe_split_path", lambda: w.probe_split_path)
    return w
