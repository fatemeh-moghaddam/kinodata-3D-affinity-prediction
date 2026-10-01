"""
prob_orchestrate: which layers and probes a run picks up, how the summary CSV is
merged, and -- end to end on the synthetic condition -- that main() reads X, y,
idents and folds consistently and writes what the explorer and plots read.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone
from sklearn.linear_model import Ridge

import kinodata.configuration as cfg
from prob import prob_orchestrate as orch
from prob.prob_models import HybridRandomForestRegressor
from prob.prob_run import probe_split_provenance


def make_config(world, **overrides) -> cfg.Config:
    return cfg.Config({
        "gnn_model_type": "CGNN-3D",
        "filter_rmsd_max_value": 2,
        "split_type": "random-k-fold",
        "output_dir": world.out_dir,
        "target_dir": world.target_dir,
        "target_file": world.target_file,
        "device": "cpu",
        "run_shuffled_baseline": 0,
        "baseline_tag": "shuffled_ident",
        **overrides,
    })


# ─────────────────────────────────────────────────────────────
# _cpu_budget
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def no_scheduler_env(monkeypatch):
    for key in ("SLURM_CPUS_PER_TASK", "NSLOTS", "OMP_NUM_THREADS"):
        monkeypatch.delenv(key, raising=False)


def test_cpu_budget_prefers_scheduler_env_in_order(no_scheduler_env, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("NSLOTS", "8")
    assert orch._cpu_budget() == 8
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    assert orch._cpu_budget() == 4


def test_cpu_budget_ignores_non_numeric_env(no_scheduler_env, monkeypatch):
    monkeypatch.setenv("NSLOTS", "auto")
    assert orch._cpu_budget(default=3) == 3


def test_cpu_budget_reads_condor_submit_file(no_scheduler_env, tmp_path):
    sub = tmp_path / "run_prob.sub"
    sub.write_text("universe = docker\nrequest_cpus = 12\nrequest_gpus = 1\n")
    assert orch._cpu_budget(default=1, sub_file=sub) == 12
    assert orch._cpu_budget(default=5, sub_file=tmp_path / "missing.sub") == 5


# ─────────────────────────────────────────────────────────────
# _resolve_layer_nums
# ─────────────────────────────────────────────────────────────


def test_layers_are_discovered_numerically_and_tower_files_ignored(tmp_path):
    for name in ("layer_0", "layer_2", "layer_10", "ligand_layer_1", "pocket_layer_0", "ids"):
        (tmp_path / f"{name}.pt").touch()
    config = cfg.Config({"output_dir": tmp_path})
    assert orch._resolve_layer_nums(config) == [0, 2, 10]


def test_prob_layers_env_overrides_discovery(tmp_path, monkeypatch):
    (tmp_path / "layer_1.pt").touch()
    monkeypatch.setenv("PROB_LAYERS", " 0, 3,")
    assert orch._resolve_layer_nums(cfg.Config({"output_dir": tmp_path})) == [0, 3]


def test_no_layers_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_extraction"):
        orch._resolve_layer_nums(cfg.Config({"output_dir": tmp_path}))


# ─────────────────────────────────────────────────────────────
# _select_probes
# ─────────────────────────────────────────────────────────────

ENTRIES = [{"name": "random_forest"}, {"name": "mlp"}]


@pytest.mark.parametrize("names_csv", ["", " ", ","])
def test_select_probes_empty_means_all(names_csv):
    assert orch._select_probes(ENTRIES, names_csv) == ENTRIES


def test_select_probes_filters_by_name():
    assert orch._select_probes(ENTRIES, " mlp ") == [{"name": "mlp"}]


def test_select_probes_typo_fails_loudly():
    with pytest.raises(ValueError, match="available"):
        orch._select_probes(ENTRIES, "mpl")


# ─────────────────────────────────────────────────────────────
# _write_summary
# ─────────────────────────────────────────────────────────────


def test_write_summary_creates_file(tmp_path):
    runs = [{"experiment": "affinity_ridge", "layer": 1, "r2": 0.5}]
    path = orch._write_summary(runs, tmp_path / "affinity" / "experiments")
    assert path.name == "summary_runs.csv"
    pd.testing.assert_frame_equal(pd.read_csv(path), pd.DataFrame(runs))


def test_write_summary_backfill_keeps_untouched_layers(tmp_path):
    """PROB_LAYERS=0 into an already probed target: layer 0 is added, the rerun
    (experiment, layer) pair is replaced, and every other row survives."""
    summary_dir = tmp_path / "experiments"
    orch._write_summary(
        [
            {"experiment": "affinity_ridge", "layer": 1, "r2": 0.1},
            {"experiment": "affinity_ridge", "layer": 2, "r2": 0.2},
            {"experiment": "affinity_mlp", "layer": 1, "r2": 0.3},
        ],
        summary_dir,
    )
    path = orch._write_summary(
        [
            {"experiment": "affinity_ridge", "layer": 0, "r2": 0.0},
            {"experiment": "affinity_ridge", "layer": 1, "r2": 0.9},
        ],
        summary_dir,
    )
    out = pd.read_csv(path)
    got = {(e, int(l)): r for e, l, r in out[["experiment", "layer", "r2"]].itertuples(index=False)}
    assert got == {
        ("affinity_ridge", 0): 0.0,
        ("affinity_ridge", 1): 0.9,
        ("affinity_ridge", 2): 0.2,
        ("affinity_mlp", 1): 0.3,
    }
    assert list(out["layer"]) == sorted(out["layer"])


# ─────────────────────────────────────────────────────────────
# main(): flag and registry wiring (probes stubbed out)
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def recorded_probes(world, monkeypatch):
    """Replace the probe runner with a recorder and give main a private registry,
    since main mutates the non-linear entries (device, n_jobs) in place."""
    calls = []

    def fake_run_probes(prob_config, X, y, idents, folds, probe_entries, n_jobs, layer_num, **kw):
        calls.append({
            "names": [e["name"] for e in probe_entries],
            "entries": probe_entries,
            "layer": layer_num,
            "target": kw.get("target_name_override"),
            "n_rows": len(y),
            "has_nan": bool(np.isnan(y).any()),
            "aligned": len(X) == len(y) == len(idents) == len(folds),
        })
        return []

    monkeypatch.setattr(orch, "run_probes", fake_run_probes)
    monkeypatch.setattr(orch, "LINEAR_PROBES", [{"name": "ridge", "estimator": Ridge(), "param_grid": {}}])
    monkeypatch.setattr(orch, "NONLINEAR_PROBES", [
        {"name": "random_forest", "estimator": HybridRandomForestRegressor(), "param_grid": {}},
        {"name": "mlp", "estimator": HybridRandomForestRegressor(), "param_grid": {}},
    ])
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    return calls


def test_main_defaults_run_linear_only_on_every_layer(world, recorded_probes):
    orch.main(make_config(world))
    assert [(c["names"], c["layer"], c["target"]) for c in recorded_probes] == [
        (["ridge"], 0, None),
        (["ridge"], 1, None),
    ]


def test_main_masks_nan_targets_out_of_x_y_idents_and_folds(world, recorded_probes):
    orch.main(make_config(world))
    n_valid = int(np.isfinite(world.y).sum())
    assert n_valid < len(world.idents)
    for call in recorded_probes:
        assert call["aligned"]
        assert call["n_rows"] == n_valid
        assert not call["has_nan"]


def test_main_env_flags_override_config(world, recorded_probes, monkeypatch):
    monkeypatch.setenv("PROB_RUN_LINEAR_MODELS", "0")
    monkeypatch.setenv("PROB_RUN_NON_LINEAR_MODELS", "true")
    monkeypatch.setenv("PROB_NONLINEAR_MODELS", "mlp")
    monkeypatch.setenv("PROB_RUN_SHUFFLED_BASELINE", "1")
    monkeypatch.setenv("PROB_LAYERS", "1")

    orch.main(make_config(world, run_shuffled_baseline=0))

    assert [(c["names"], c["layer"], c["target"]) for c in recorded_probes] == [
        (["mlp"], 1, None),
        (["mlp"], 1, "affinity_shuffled_ident"),
    ]


def test_main_baseline_tag_comes_from_config(world, recorded_probes):
    orch.main(make_config(world, run_shuffled_baseline=1, baseline_tag="perm"))
    assert {c["target"] for c in recorded_probes} == {None, "affinity_perm"}


def test_main_on_cuda_moves_nonlinear_estimators_and_caps_n_jobs(world, recorded_probes, monkeypatch):
    monkeypatch.setenv("PROB_RUN_LINEAR_MODELS", "0")
    monkeypatch.setenv("PROB_RUN_NON_LINEAR_MODELS", "1")
    monkeypatch.setenv("PROB_LAYERS", "1")

    orch.main(make_config(world, device="cuda"))

    (call,) = recorded_probes
    assert [e["estimator"].device for e in call["entries"]] == ["cuda", "cuda"]
    assert [e.get("n_jobs") for e in call["entries"]] == [1, 1]


def test_main_requires_a_target_file(world, recorded_probes):
    with pytest.raises(ValueError, match="target_file"):
        orch.main(make_config(world, target_file=""))


# ─────────────────────────────────────────────────────────────
# run_probes and main(): real probes on the synthetic condition
# ─────────────────────────────────────────────────────────────

CHEAP_RIDGE = [{"name": "ridge", "estimator": Ridge(), "param_grid": {"model__alpha": [0.01, 1.0, 100.0]}}]


@pytest.fixture
def cheap_registry(monkeypatch):
    monkeypatch.setattr(orch, "LINEAR_PROBES", [dict(e, estimator=clone(e["estimator"])) for e in CHEAP_RIDGE])
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")


@pytest.mark.slow
def test_run_probes_returns_one_row_per_probe_with_checkpoint_spread(world):
    y = world.y
    ok = np.isfinite(y)
    runs = orch.run_probes(
        make_config(world),
        world.layer(1)[ok], y[ok], world.idents[ok], world.folds[ok],
        CHEAP_RIDGE,
        n_jobs=1,
        layer_num=1,
    )
    (run,) = runs
    assert run["experiment"] == "affinity_ridge"
    assert run["layer"] == 1
    for metric in ("r2", "rmse", "mae", "pearson"):
        assert np.isfinite(run[metric])
        assert f"{metric}_ckpt_mean" in run and f"{metric}_ckpt_sd" in run
    assert "across_checkpoints" not in run
    assert (world.out_dir / "affinity" / "ridge" / "1" / "reports" / "ridge_summary.json").exists()


@pytest.mark.slow
def test_main_end_to_end(world, cheap_registry):
    runs = orch.main(make_config(world, run_shuffled_baseline=1))

    # Summary CSV: one row per (probe, layer) per mode (per checkpoint and pooled
    # are both on by default); the signal is in layer 1 only.
    summary = pd.read_csv(world.out_dir / "affinity" / "experiments" / "summary_runs.csv")
    assert list(zip(summary["experiment"], summary["layer"])) == [
        ("affinity_ridge", 0), ("affinity_ridge_pooled", 0),
        ("affinity_ridge", 1), ("affinity_ridge_pooled", 1),
    ]
    per_ckpt = summary[summary["experiment"] == "affinity_ridge"]
    pooled = summary[summary["experiment"] == "affinity_ridge_pooled"]
    r2 = dict(zip(per_ckpt["layer"], per_ckpt["r2"]))
    assert r2[1] > 0.95, "one probe per checkpoint should decode a rotated linear signal"
    assert r2[0] < 0.1, "layer 0 is noise"
    assert pooled.loc[pooled["layer"] == 1, "r2"].item() < r2[1], (
        "one pooled probe cannot undo a different rotation per checkpoint"
    )
    assert {"r2_ckpt_mean", "r2_ckpt_sd", "pearson_ckpt_mean"} <= set(summary.columns)
    assert len(runs) == 4

    # The shuffled baseline has its own target dir and learns nothing.
    baseline = pd.read_csv(world.out_dir / "affinity_shuffled_ident" / "experiments" / "summary_runs.csv")
    assert sorted(baseline["layer"]) == [0, 0, 1, 1]
    assert (baseline["r2"] < 0.1).all()

    # Predictions: the probe test idents that have a target, sorted, each tagged
    # with the checkpoint it came from and its own y.
    exp = world.out_dir / "affinity" / "ridge" / "1"
    pred = pd.read_csv(exp / "artifacts" / "ridge_predictions.csv")
    assert list(pred.columns) == ["ident", "fold", "y_true", "y_pred"]
    expected = sorted(i for i in world.idents.tolist() if i in world.test_idents and i in world.targets)
    assert pred["ident"].tolist() == expected
    fold_of = dict(zip(world.idents.tolist(), world.folds.tolist()))
    assert pred["fold"].tolist() == [fold_of[i] for i in expected]
    np.testing.assert_allclose(pred["y_true"], [world.targets[i] for i in expected], rtol=1e-6)

    # Summary JSON: per checkpoint, traced to the exact split file.
    report = json.loads((exp / "reports" / "ridge_summary.json").read_text())
    assert report["probe_mode"] == "per_checkpoint"
    assert [f["fold"] for f in report["per_fold"]] == [0, 1, 2, 3, 4]
    assert report["n_test_samples"] == len(expected)
    assert report["n_samples"] == int(np.isfinite(world.y).sum())
    assert report["probe_split"] == probe_split_provenance(world.probe_split_path)
    assert set(report["statistical_tests"]) == {"r2_ci", "rmse_ci", "mae_ci", "pearson_ci"}
    for f in ("ridge_parity.png", "ridge_residuals.png"):
        assert (exp / "figures" / f).exists()


@pytest.mark.slow
def test_main_layer_backfill_keeps_other_layers_in_summary(world, cheap_registry, monkeypatch):
    config = make_config(world)
    monkeypatch.setenv("PROB_LAYERS", "1")
    orch.main(config)
    monkeypatch.setenv("PROB_LAYERS", "0")
    orch.main(config)

    summary = pd.read_csv(world.out_dir / "affinity" / "experiments" / "summary_runs.csv")
    for experiment in ("affinity_ridge", "affinity_ridge_pooled"):
        assert sorted(summary.loc[summary["experiment"] == experiment, "layer"]) == [0, 1]


@pytest.mark.slow
def test_main_pooled_switch_off_runs_only_per_checkpoint(world, cheap_registry, monkeypatch):
    monkeypatch.setenv("PROB_POOLED", "0")
    orch.main(make_config(world))

    summary = pd.read_csv(world.out_dir / "affinity" / "experiments" / "summary_runs.csv")
    assert set(summary["experiment"]) == {"affinity_ridge"}
    assert not (world.out_dir / "affinity" / "ridge_pooled").exists()


@pytest.mark.slow
def test_main_with_both_switches_off_raises(world, cheap_registry, monkeypatch):
    monkeypatch.setenv("PROB_POOLED", "0")
    monkeypatch.setenv("PROB_PER_CKPT", "0")
    with pytest.raises(ValueError, match="nothing to run"):
        orch.main(make_config(world))


@pytest.mark.slow
def test_main_refuses_ids_that_are_not_the_fold_concatenation(world, cheap_registry, monkeypatch):
    """A layer file whose rows disagree with ids.pt must not be probed silently:
    load_fold_index refuses when ids.pt is not the per-fold concatenation."""
    torch.save(torch.as_tensor(world.idents[::-1].copy()), world.out_dir / "ids.pt")
    with pytest.raises(ValueError, match="concatenation"):
        orch.main(make_config(world))
