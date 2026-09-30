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
    assert config.run_shuffled_baseline == 0


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
