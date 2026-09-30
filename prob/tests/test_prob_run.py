"""
prob_run: the fixed probe train/test split by ident, and per-checkpoint probing
(one probe per GNN checkpoint, best params saved and reused per fold).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from prob import prob_run
from prob.paths_and_io import get_exp_dirs
from prob.prob_run import (
    _read_best_params_by_fold,
    load_probe_split,
    make_probe_split,
    probe_split_provenance,
    run_probe_per_checkpoint,
    split_by_ident,
)

GRID = {"model__alpha": [0.01, 1.0, 100.0]}


# ─────────────────────────────────────────────────────────────
# Probe split file
# ─────────────────────────────────────────────────────────────


def test_make_probe_split_is_deterministic_and_deduplicated(tmp_path):
    idents = np.concatenate([np.arange(1000), np.arange(10)])  # duplicates collapse
    a = make_probe_split(idents, tmp_path / "a.csv", random_state=3)
    b = make_probe_split(idents[::-1], tmp_path / "b.csv", random_state=3)

    assert a["ident"].is_unique and len(a) == 1000
    assert (a["probe_split"] == "test").sum() == 100
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "a.csv"), a)


def test_make_probe_split_refuses_to_overwrite(tmp_path):
    path = tmp_path / "split.csv"
    first = make_probe_split(np.arange(100), path, random_state=0)
    with pytest.raises(FileExistsError):
        make_probe_split(np.arange(100), path, random_state=1)
    pd.testing.assert_frame_equal(pd.read_csv(path), first)

    make_probe_split(np.arange(100), path, random_state=1, overwrite=True)
    assert not pd.read_csv(path).equals(first)


def test_load_probe_split_builds_from_catalogue_once(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(prob_run, "catalogue_idents", lambda: calls.append(1) or np.arange(50))
    path = tmp_path / "split.csv"

    first = load_probe_split(path)
    second = load_probe_split(path)

    assert calls == [1]
    pd.testing.assert_frame_equal(first, second)


def test_provenance_hashes_the_file(tmp_path):
    path = tmp_path / "split.csv"
    make_probe_split(np.arange(20), path)
    info = probe_split_provenance(path)
    assert info == {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


# ─────────────────────────────────────────────────────────────
# split_by_ident
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def split_frame():
    idents = np.arange(1000)
    return pd.DataFrame({"ident": idents, "probe_split": np.where(idents % 10 == 0, "test", "train")})


def test_split_by_ident_keeps_rows_with_their_idents(split_frame):
    rng = np.random.default_rng(0)
    idents = rng.permutation(1000)
    X = idents[:, None] * np.ones((1, 3))   # row content encodes its ident
    y = idents * 2.0

    X_tr, X_te, y_tr, y_te, id_tr, id_te = split_by_ident(X, y, idents, split_frame)

    assert (id_te % 10 == 0).all() and (id_tr % 10 != 0).all()
    assert list(id_te) == sorted(id_te) and list(id_tr) == sorted(id_tr)
    np.testing.assert_array_equal(X_te[:, 0], id_te)
    np.testing.assert_array_equal(y_tr, id_tr * 2.0)


def test_split_by_ident_does_not_depend_on_row_order(split_frame):
    idents = np.arange(1000)
    X, y = np.random.default_rng(0).normal(size=(1000, 2)), np.arange(1000.0)
    perm = np.random.default_rng(1).permutation(1000)

    a = split_by_ident(X, y, idents, split_frame)
    b = split_by_ident(X[perm], y[perm], idents[perm], split_frame)
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left, right)


def test_split_by_ident_rejects_unknown_idents(split_frame):
    idents = np.arange(995, 1005)
    with pytest.raises(ValueError, match="not in the probe split"):
        split_by_ident(np.zeros((10, 1)), np.zeros(10), idents, split_frame)


def test_split_by_ident_rejects_misaligned_inputs(split_frame):
    with pytest.raises(ValueError, match="same length"):
        split_by_ident(np.zeros((10, 1)), np.zeros(9), np.arange(10), split_frame)


def test_split_by_ident_rejects_a_skewed_subset(split_frame):
    """A subset chosen with the labels (here: over-sampling test idents) must not
    be probed on a quietly different test fraction."""
    idents = np.concatenate([np.arange(0, 1000, 10), np.arange(1, 400)])
    idents = idents[~np.isin(idents, np.arange(10, 400, 10))]
    with pytest.raises(ValueError, match="test fraction"):
        split_by_ident(np.zeros((len(idents), 1)), np.zeros(len(idents)), idents, split_frame)


# ─────────────────────────────────────────────────────────────
# run_probe_per_checkpoint
# ─────────────────────────────────────────────────────────────


def _valid(world, layer=1):
    ok = np.isfinite(world.y)
    return world.layer(layer)[ok], world.y[ok], world.idents[ok], world.folds[ok]


def _run(world, exp_dirs=None, **kw):
    X, y, idents, folds = _valid(world)
    return run_probe_per_checkpoint(
        X, y, Ridge(), kw.pop("param_grid", GRID),
        idents=idents, folds=folds, random_state=0, n_jobs=1,
        exp_dirs=exp_dirs, model_name="ridge", n_bootstrap=50, **kw,
    )


def test_one_probe_per_checkpoint_beats_one_probe_over_all(world):
    """The reason the pipeline probes per checkpoint: each fold's embedding is its
    own rotation of the same signal, which a single pooled probe cannot undo."""
    _, per_ckpt, _ = _run(world)

    X, y, idents, _ = _valid(world)
    pooled, *_ = prob_run.run_probe(
        X, y, Ridge(), random_state=0, idents=idents, run_stats=False,
    )
    assert per_ckpt["r2"] > 0.95
    assert pooled["r2"] < per_ckpt["r2"] - 0.3


def test_per_checkpoint_predictions_and_metrics(world):
    best, metrics, pred = _run(world)

    assert sorted(best) == [0, 1, 2, 3, 4]
    assert all("model__alpha" in p for p in best.values())
    assert list(pred["ident"]) == sorted(pred["ident"])
    assert set(pred["ident"]) <= world.test_idents

    across = metrics["across_checkpoints"]
    assert set(across) == {"r2", "rmse", "mae", "pearson"}
    fold_r2 = [
        prob_run.evaluate_predictions(g["y_true"], g["y_pred"])["r2"] for _, g in pred.groupby("fold")
    ]
    assert across["r2"]["mean"] == pytest.approx(np.mean(fold_r2))
    assert across["r2"]["sd"] == pytest.approx(np.std(fold_r2, ddof=1))


def test_per_checkpoint_without_grid_fits_as_given(world):
    best, metrics, _ = _run(world, param_grid=None)
    assert best == {k: {} for k in range(5)}
    assert np.isfinite(metrics["r2"])


def test_per_checkpoint_rejects_misaligned_folds(world):
    X, y, idents, folds = _valid(world)
    with pytest.raises(ValueError, match="same length"):
        run_probe_per_checkpoint(
            X, y, Ridge(), GRID, idents=idents, folds=folds[:-1], random_state=0,
        )


def test_per_checkpoint_writes_artifacts(world, tmp_path):
    exp_dirs = get_exp_dirs(tmp_path, "affinity", "ridge", 1)
    _run(world, exp_dirs=exp_dirs)

    cv = pd.read_csv(exp_dirs["artifacts"] / "ridge_cv_results.csv")
    assert sorted(cv["fold"].unique()) == [0, 1, 2, 3, 4]
    saved = json.loads((exp_dirs["reports"] / "ridge_best_params.json").read_text())
    assert saved["param_grid"] == {"model__alpha": [0.01, 1.0, 100.0]}
    assert sorted(saved["folds"]) == ["0", "1", "2", "3", "4"]

    summary = json.loads((exp_dirs["reports"] / "ridge_summary.json").read_text())
    assert summary["bootstrap"] == {"n_bootstrap": 50, "confidence": 0.95, "random_state": 0}
    assert len(summary["per_fold"]) == 5
    assert sum(f["n_test_samples"] for f in summary["per_fold"]) == summary["n_test_samples"]


def test_rerun_reuses_saved_params_unless_the_grid_changed(world, tmp_path, monkeypatch):
    exp_dirs = get_exp_dirs(tmp_path, "affinity", "ridge", 1)
    first, *_ = _run(world, exp_dirs=exp_dirs)

    def no_tuning(*_a, **_k):
        raise AssertionError("tuned again although params were saved for this grid")

    monkeypatch.setattr(prob_run, "tune_probe", no_tuning)
    again, *_ = _run(world, exp_dirs=exp_dirs)
    assert again == first

    with pytest.raises(AssertionError, match="tuned again"):
        _run(world, exp_dirs=exp_dirs, param_grid={"model__alpha": [0.5]})


def test_shared_params_carry_over_to_other_layers(world, tmp_path, monkeypatch):
    layer1 = get_exp_dirs(tmp_path, "affinity", "ridge", 1)
    first, *_ = _run(world, exp_dirs=layer1, share_best_params_across_layers=True)
    assert (tmp_path / "affinity" / "ridge" / "shared_best_params" / "ridge_best_params.json").exists()

    monkeypatch.setattr(prob_run, "tune_probe", lambda *a, **k: pytest.fail("layer 2 retuned"))
    layer2 = get_exp_dirs(tmp_path, "affinity", "ridge", 2)
    second, *_ = _run(world, exp_dirs=layer2, share_best_params_across_layers=True)
    assert second == first


def test_best_params_reader_ignores_old_or_foreign_files(tmp_path):
    path = tmp_path / "p.json"
    assert _read_best_params_by_fold(path, GRID) == {}

    path.write_text(json.dumps({"model__alpha": 1.0}))  # old single-params format
    assert _read_best_params_by_fold(path, GRID) == {}

    grid = {"model__hidden_layer_sizes": [[128], [256, 128]]}
    path.write_text(json.dumps({"param_grid": grid, "folds": {"0": {"model__hidden_layer_sizes": [256, 128]}}}))
    assert _read_best_params_by_fold(path, grid) == {0: {"model__hidden_layer_sizes": (256, 128)}}
