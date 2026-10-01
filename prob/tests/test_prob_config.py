"""prob_config.get_ds_load_config: the config prob_orchestrate.main runs with."""
from __future__ import annotations

import sys

import pytest

from prob import prob_config
from prob.paths_and_io import get_out_dir


@pytest.fixture
def tmp_root(tmp_path, monkeypatch):
    """Resolve output_dir under tmp_path instead of the real data/probing."""
    monkeypatch.setattr(
        prob_config, "get_out_dir",
        lambda *args, **kwargs: get_out_dir(*args, **kwargs, root=tmp_path),
    )
    monkeypatch.setattr(sys, "argv", ["prob_orchestrate.py"])
    return tmp_path


def test_defaults(tmp_root):
    config = prob_config.get_ds_load_config(config_name="test_defaults")
    assert config.gnn_model_type == "CGNN-3D"
    assert config.output_dir == tmp_root / "data/probing/CGNN-3D/rmsd_cutoff_2/random-k-fold"
    assert config.target_dir == tmp_root / "data/probing/targets"
    # Run switches are not part of the condition config: their defaults live in the spec.
    assert "run_shuffled_baseline" not in config and "baseline_tag" not in config


def test_cli_overrides_move_output_dir(tmp_root, monkeypatch):
    """run_prob.sh passes the model on the command line; output_dir has to follow it,
    or every job reads and writes the default model's directory."""
    monkeypatch.setattr(sys, "argv", [
        "prob_orchestrate.py", "--gnn_model_type", "DTI", "--split_type", "scaffold-k-fold",
        "--filter_rmsd_max_value", "4", "--target_file", "affinity.pt",
    ])
    config = prob_config.get_ds_load_config(config_name="test_cli")
    assert config.output_dir == tmp_root / "data/probing/DTI/rmsd_cutoff_4/scaffold-k-fold"
    assert config.target_dir == tmp_root / "data/probing/targets"
    assert config.target_file == "affinity.pt"


def test_rejects_unknown_and_invalid_arguments(tmp_root):
    with pytest.raises(ValueError, match="Invalid arguments"):
        prob_config.get_ds_load_config(layer=3)
    with pytest.raises(AssertionError):
        prob_config.get_ds_load_config(gnn_model_type="GIN")
    with pytest.raises(AssertionError):
        prob_config.get_ds_load_config(filter_rmsd_max_value=3)


def test_rejects_invalid_model_from_cli(tmp_root, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--gnn_model_type", "GIN"])
    with pytest.raises(AssertionError, match="GIN"):
        prob_config.get_ds_load_config(config_name="test_cli_bad")


def test_experiment_name():
    config = {"gnn_model_type": "DTI-soft", "filter_rmsd_max_value": 2, "split_type": "random-k-fold",
              "target_file": "targets/hb_score.pt"}
    from kinodata.configuration import Config
    name = prob_config.build_experiment_name(Config(config), layer_num=2)
    assert name == "gnn=DTI-soft_rmsd=2_split=random-k-fold_layer=2_target=hb_score"
    with pytest.raises(ValueError):
        prob_config.build_experiment_name(Config({**config, "target_file": ""}), 0)


# ─────────────────────────────────────────────────────────────
# ProbingExperimentSpec and resolve_experiment_spec
# ─────────────────────────────────────────────────────────────

CONDITION = {"gnn_model_type": "DTI", "split_type": "scaffold-k-fold",
             "filter_rmsd_max_value": 4, "target_file": "hb_score.pt"}


def test_defaults_come_from_the_spec_and_say_so():
    spec, sources = prob_config.resolve_experiment_spec(CONDITION, environ={})
    assert spec.data.gnn_model_type == "DTI" and spec.target.target_file == "hb_score.pt"
    assert spec.target.run_shuffled_baseline is True
    assert (spec.model.per_ckpt, spec.model.pooled, spec.model.run_non_linear_models) == (True, True, False)
    assert spec.data.layers is None and spec.compute.n_jobs is None
    assert sources["target.run_shuffled_baseline"] == "default"
    assert sources["data.gnn_model_type"] == "config"


def test_precedence_is_cli_then_config_then_env_then_default():
    env = {"PROB_POOLED": "0", "PROB_LAYERS": "2", "PROB_RUN_LINEAR_MODELS": "0"}
    config = {**CONDITION, "layers": "1", "run_linear_models": 1}
    spec, sources = prob_config.resolve_experiment_spec(
        config, argv=["--target_file", "x.pt", "--run_linear_models", "0"], environ=env,
    )
    assert spec.model.run_linear_models is False and sources["model.run_linear_models"] == "cli"
    assert spec.data.layers == (1,) and sources["data.layers"] == "config"
    assert spec.model.pooled is False and sources["model.pooled"] == "env PROB_POOLED='0'"
    assert spec.model.per_ckpt is True and sources["model.per_ckpt"] == "default"
    # Condition fields come from the config (get_ds_load_config parsed the CLI into
    # it already); the source says the command line set them.
    assert spec.target.target_file == "hb_score.pt" and sources["target.target_file"] == "cli"


def test_fixed_values_cannot_be_set_per_run():
    with pytest.raises(TypeError):
        prob_config.ProbeModelSettings(INNER_CV_FOLDS=5)
    spec, _ = prob_config.resolve_experiment_spec(
        CONDITION, argv=["--BOOTSTRAP_N", "10"], environ={},
    )
    assert spec.evaluation.BOOTSTRAP_N == prob_config.ProbeEvalSettings.BOOTSTRAP_N


@pytest.mark.parametrize("env, message", [
    ({"PROB_POOLED": "maybe"}, "model.pooled"),
    ({"PROB_POOLD": "0"}, "PROB_POOLD"),
    ({"PROB_POOLED": "0", "PROB_PER_CKPT": "0"}, "nothing to run"),
])
def test_bad_values_fail_loudly(env, message):
    with pytest.raises(ValueError, match=message):
        prob_config.resolve_experiment_spec(CONDITION, environ=env)


@pytest.mark.parametrize("bad", [
    {"gnn_model_type": "GIN"}, {"split_type": "time-split"},
    {"filter_rmsd_max_value": 3}, {"target_file": ""},
])
def test_invalid_condition_fails(bad):
    with pytest.raises(ValueError):
        prob_config.resolve_experiment_spec({**CONDITION, **bad}, environ={})


def test_spec_record_lists_every_setting_and_fixed_value():
    spec, sources = prob_config.resolve_experiment_spec(CONDITION, environ={"PROB_LAYERS": "0,3"})
    record = prob_config.spec_record(spec, sources)
    assert record["data"]["settings"]["layers"] == {"value": (0, 3), "source": "env PROB_LAYERS='0,3'"}
    assert record["evaluation"]["fixed"] == {
        "BOOTSTRAP_CONFIDENCE": 0.95, "BOOTSTRAP_N": 1000, "BOOTSTRAP_SEED": 96,
        "MAX_TEST_FRACTION_DEVIATION": 0.01, "MAX_TEST_FRACTION_DEVIATION_PER_FOLD": 0.015,
        "PROBE_SPLIT_SEED": 0, "PROBE_TEST_SIZE": 0.1,
    }
    assert record["compute"]["fixed"] == {}


def test_probe_modules_read_their_fixed_values_from_the_spec():
    from prob import prob_models, prob_run
    assert prob_models.RANDOM_STATE == prob_config.ProbeModelSettings.ESTIMATOR_SEED
    assert prob_run.DEFAULT_N_SPLITS_CV == prob_config.ProbeModelSettings.INNER_CV_FOLDS
    assert prob_run.DEFAULT_N_BOOTSTRAP == prob_config.ProbeEvalSettings.BOOTSTRAP_N
    assert prob_run.MAX_TEST_FRACTION_DEVIATION_PER_FOLD == (
        prob_config.ProbeEvalSettings.MAX_TEST_FRACTION_DEVIATION_PER_FOLD
    )
