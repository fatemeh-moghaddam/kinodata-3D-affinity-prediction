"""
paths_and_io: the directory layout the extraction and probes write, its inverse
(find_probe_runs / attach_run_metrics), and the loaders prob_orchestrate.main
reads X, y, idents and folds with.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from prob.paths_and_io import (
    GNN_MODEL_TYPES,
    attach_run_metrics,
    checkpoint_model_type,
    find_probe_runs,
    get_exp_dirs,
    get_out_dir,
    get_split_file,
    load_fold_index,
    load_run_predictions,
    load_run_predictions_frame,
    load_X_from_pt,
    load_y_by_ids,
)


# ─────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────


def test_model_type_names_are_safe_path_and_name_tokens():
    assert all("_" not in name for name in GNN_MODEL_TYPES)


def test_dti_soft_loads_dti_weights():
    assert checkpoint_model_type("DTI-soft") == "DTI"
    for name in ("CGNN-3D", "CGNN", "DTI"):
        assert checkpoint_model_type(name) == name


def test_out_dir_keeps_fold_zero_and_can_stay_read_only(tmp_path):
    base = get_out_dir("CGNN-3D", 2, "random-k-fold", None, root=tmp_path, create=False)
    assert base == tmp_path / "data/probing/CGNN-3D/rmsd_cutoff_2/random-k-fold"
    assert not base.exists()
    assert get_out_dir("CGNN-3D", 2, "random-k-fold", 0, root=tmp_path) == base / "0"
    assert (base / "0").is_dir()


def test_exp_dirs_layout(tmp_path):
    dirs = get_exp_dirs(tmp_path, "affinity", "ridge", 3, create=False)
    assert dirs["root"] == tmp_path / "affinity" / "ridge" / "3"
    assert {k: v.name for k, v in dirs.items() if k != "root"} == {
        "figures": "figures", "artifacts": "artifacts", "reports": "reports",
    }
    assert not dirs["root"].exists()
    get_exp_dirs(tmp_path, "affinity", "ridge", 3)
    assert all(p.is_dir() for p in dirs.values())


@pytest.mark.parametrize("name", ["1_5.csv", "1:5.csv"])
def test_split_file_accepts_both_separators(tmp_path, name):
    d = tmp_path / "data/processed/filter_predicted_rmsd_le2.00/scaffold-k-fold"
    d.mkdir(parents=True)
    (d / name).touch()
    assert get_split_file("scaffold-k-fold", 0, 2, root=tmp_path) == d / name


def test_split_file_missing_or_ambiguous(tmp_path):
    d = tmp_path / "data/processed/filter_predicted_rmsd_le2.00/random-k-fold"
    d.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        get_split_file("random-k-fold", 0, 2, root=tmp_path)
    (d / "1_5.csv").touch()
    (d / "1:5.csv").touch()
    with pytest.raises(RuntimeError, match="Ambiguous"):
        get_split_file("random-k-fold", 0, 2, root=tmp_path)


# ─────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────


def _write_run(root: Path, gnn, rmsd, split, target, probe, layer, summary=None):
    exp = root / gnn / f"rmsd_cutoff_{rmsd}" / split / target / probe
    if layer is not None:
        exp = exp / str(layer)
    (exp / "artifacts").mkdir(parents=True)
    (exp / "reports").mkdir()
    pd.DataFrame({"ident": [1, 2], "fold": [0, 1], "y_true": [1.0, 2.0], "y_pred": [1.5, 2.5]}).to_csv(
        exp / "artifacts" / f"{probe}_predictions.csv", index=False
    )
    if summary is not None:
        (exp / "reports" / f"{probe}_summary.json").write_text(
            summary if isinstance(summary, str) else json.dumps(summary)
        )
    return exp


@pytest.fixture
def sweep(tmp_path):
    root = tmp_path / "probing"
    summary = {
        "metrics_on_unseen_data": {"r2": 0.5, "rmse": 1.0},
        "n_test_samples": 2,
        "statistical_tests": {"r2_ci": {"lower": 0.4, "upper": 0.6}},
        "across_checkpoints": {"r2": {"mean": 0.45, "sd": 0.05}},
    }
    _write_run(root, "CGNN-3D", 2, "random-k-fold", "affinity", "ridge", 1, summary)
    _write_run(root, "CGNN-3D", 2, "random-k-fold", "affinity", "ridge", 0, "{not json")
    _write_run(root, "CGNN-3D", 2, "random-k-fold", "affinity_shuffled_ident", "ridge", 1)
    _write_run(root, "DTI", 4, "scaffold-k-fold", "affinity", "mlp", 2)
    _write_run(root, "CGNN", 6, "random-k-fold", "affinity", "ridge", None)  # legacy, no layer level
    # Must be ignored: the GNN's own predictions, and a directory that is not a cutoff.
    (root / "CGNN-3D" / "rmsd_cutoff_2" / "random-k-fold" / "predictions.csv").write_text("fold,y_true,y_pred\n")
    _write_run(root, "CGNN-3D", "x", "random-k-fold", "affinity", "ridge", 1)
    return root


def test_find_probe_runs_decodes_every_factor(sweep):
    runs = find_probe_runs(root=sweep)
    assert len(runs) == 5
    row = runs[(runs.gnn_model_type == "DTI")].iloc[0]
    assert (row.rmsd_threshold, row.split_type, row.target, row.prob_model, row.layer) == (
        4.0, "scaffold-k-fold", "affinity", "mlp", 2,
    )
    assert runs["layer"].dtype == "Int64"
    assert runs.loc[runs.gnn_model_type == "CGNN", "layer"].isna().all()


def test_find_probe_runs_tags_baselines_and_filters(sweep):
    baseline = find_probe_runs(root=sweep, target="affinity", layer=1, gnn_model_type="CGNN-3D")
    assert baseline["is_baseline"].tolist() == [False, True]
    assert set(baseline["target"]) == {"affinity"}
    assert set(baseline["target_full"]) == {"affinity", "affinity_shuffled_ident"}

    assert len(find_probe_runs(root=sweep, include_baselines=False)) == 4
    assert len(find_probe_runs(root=sweep, rmsd_threshold=[2, 4])) == 4
    assert find_probe_runs(root=sweep, prob_model="lasso").empty


def test_find_probe_runs_without_data_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_probe_runs(root=tmp_path / "nope")


def test_attach_run_metrics_reads_summaries_and_checkpoint_spread(sweep):
    runs = attach_run_metrics(find_probe_runs(root=sweep, gnn_model_type="CGNN-3D", include_baselines=False))
    layer1 = runs[runs.layer == 1].iloc[0]
    assert layer1.r2 == 0.5 and layer1.n_test_samples == 2
    assert (layer1.r2_ci_lower, layer1.r2_ci_upper) == (0.4, 0.6)
    assert layer1.r2_ckpt_mean == 0.45
    assert layer1.r2_ckpt_lower == pytest.approx(0.40) and layer1.r2_ckpt_upper == pytest.approx(0.50)
    assert np.isnan(runs[runs.layer == 0].iloc[0].r2)  # unreadable summary -> NaN, not a crash


def test_load_run_predictions_accepts_path_mapping_and_row(sweep):
    runs = find_probe_runs(root=sweep, gnn_model_type="DTI")
    row = next(runs.itertuples())
    for run in (row.predictions_path, str(row.predictions_path), runs.iloc[0].to_dict(), row):
        y_true, y_pred = load_run_predictions(run)
        np.testing.assert_array_equal(y_true, [1.0, 2.0])
    assert list(load_run_predictions_frame(row).columns) == ["ident", "fold", "y_true", "y_pred"]


def test_load_run_predictions_missing_column(tmp_path):
    path = tmp_path / "p.csv"
    pd.DataFrame({"y_true": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="y_pred"):
        load_run_predictions(path)


# ─────────────────────────────────────────────────────────────
# Loaders used by prob_orchestrate.main
# ─────────────────────────────────────────────────────────────


def test_fold_index_matches_the_fold_files(world):
    np.testing.assert_array_equal(load_fold_index(world.out_dir), world.folds)


def test_fold_index_ignores_unrelated_digit_dirs(world):
    stray = world.out_dir / "7"
    stray.mkdir()
    torch.save(torch.tensor([1, 2]), stray / "ids_3.pt")  # wrong name for its folder
    np.testing.assert_array_equal(load_fold_index(world.out_dir), world.folds)


def test_fold_index_refuses_a_stale_aggregate(world):
    torch.save(torch.as_tensor(np.sort(world.idents)), world.out_dir / "ids.pt")
    with pytest.raises(ValueError, match="concatenation"):
        load_fold_index(world.out_dir)


def test_fold_index_needs_fold_files(tmp_path):
    torch.save(torch.tensor([1]), tmp_path / "ids.pt")
    with pytest.raises(FileNotFoundError, match="ids_"):
        load_fold_index(tmp_path)


def test_load_x_keeps_ids_row_order(world):
    np.testing.assert_array_equal(load_X_from_pt(world.out_dir, layer_num=1), world.layer(1))


def test_load_y_follows_ids_and_marks_missing_targets(world):
    y, mask = load_y_by_ids(world.out_dir, world.target_dir, world.target_file, return_mask=True)
    np.testing.assert_allclose(y[mask], world.y[mask], rtol=1e-6)
    np.testing.assert_array_equal(mask, np.isfinite(world.y))
    assert (~mask).sum() == 2


def test_shuffled_y_is_a_seeded_permutation(world):
    y = load_y_by_ids(world.out_dir, world.target_dir, world.target_file)
    a = load_y_by_ids(world.out_dir, world.target_dir, world.target_file, shuffle_idents=True, random_state=1)
    b = load_y_by_ids(world.out_dir, world.target_dir, world.target_file, shuffle_idents=True, random_state=1)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(np.sort(a), np.sort(y))  # NaNs sort last in both
    assert not np.array_equal(a, y, equal_nan=True)


def test_load_y_requires_a_target_file(world):
    with pytest.raises(ValueError, match="targets_file"):
        load_y_by_ids(world.out_dir, world.target_dir)
