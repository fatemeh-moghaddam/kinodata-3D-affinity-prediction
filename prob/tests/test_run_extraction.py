"""
run_extraction and its aggregation helpers: the job spec contract, what a fold
is expected to write, and that the aggregates are the fold files concatenated in
fold order -- the layout prob_orchestrate.main reads back with load_fold_index.

Importing these modules pulls in kinodata.model (torch_scatter) and colorama, so
the whole file is skipped where those are not installed.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

run_extraction = pytest.importorskip("prob.run_extraction")
aggregation = pytest.importorskip("prob.resloves_and_transforms")

import kinodata.configuration as cfg
from prob.paths_and_io import load_fold_index
from prob.run_extraction import (
    ProbingJobSpec,
    ResolvedPaths,
    _validate_spec,
    expected_fold_artifacts,
    probe_layer_names,
    resolve_device,
    resolve_paths,
    set_probing_config,
    write_manifest,
)


# ─────────────────────────────────────────────────────────────
# Job spec
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", [
    {"gnn_model_type": "BAD"},
    {"split_type": "BAD"},
    {"filter_rmsd_max_value": 123},
    {"k_fold": 0},
])
def test_validate_spec_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        _validate_spec(ProbingJobSpec(**bad))


@pytest.mark.parametrize("rmsd", [2, 4, 6, 2.0, 4.0, 6.0, None])
def test_validate_spec_accepts_supported_rmsd(rmsd):
    _validate_spec(ProbingJobSpec(filter_rmsd_max_value=rmsd))


def test_spec_defaults_extract_val_too():
    """data_checks requires include_val: without it split types hold different molecules."""
    assert ProbingJobSpec().include_val is True


def test_resolve_paths_loads_dti_soft_from_dti(tmp_path, monkeypatch):
    seen = {}

    def fake_model_dir(**kw):
        seen.update(kw)
        return tmp_path / kw["model_type"]

    monkeypatch.setattr(run_extraction, "get_model_dir", fake_model_dir)
    monkeypatch.setattr(run_extraction, "get_model_ckpt", lambda d: d / "x.ckpt")
    monkeypatch.setattr(run_extraction, "get_split_file", lambda *a, **k: tmp_path / "1_5.csv")
    monkeypatch.setattr(run_extraction, "get_out_dir", lambda gnn, *a, **k: tmp_path / "out" / gnn)

    paths = resolve_paths(ProbingJobSpec(gnn_model_type="DTI-soft"), fold=3)
    assert seen["model_type"] == "DTI" and seen["split_fold"] == 3
    assert paths.model_ckpt == tmp_path / "DTI" / "x.ckpt"
    assert paths.gnn_config_path == tmp_path / "DTI" / "config.json"
    assert paths.output_root_dir == tmp_path / "out" / "DTI-soft"


def test_set_probing_config_keeps_extraction_choices_over_checkpoint_config(tmp_path, monkeypatch):
    paths = ResolvedPaths(
        model_dir=tmp_path, model_ckpt=tmp_path / "x.ckpt", split_file=tmp_path / "1_5.csv",
        output_root_dir=tmp_path / "out", gnn_config_path=tmp_path / "config.json",
    )
    monkeypatch.setattr(run_extraction, "resolve_paths", lambda *a, **k: paths)
    monkeypatch.setattr(run_extraction, "load_config", lambda _: cfg.Config({
        "gnn_model_type": "CGNN", "emit_tower_reprs": True, "num_attention_blocks": 3,
    }))
    monkeypatch.setattr(cfg.Config, "update_from_args", lambda self: pytest.fail("parsed CLI args"))

    config = set_probing_config(gnn_model_type="DTI-soft", split_index=2)
    assert config.gnn_model_type == "DTI-soft"
    assert config.emit_tower_reprs is False
    assert config.num_attention_blocks == 3
    assert config.output_dir == paths.output_root_dir
    assert config.model_ckpt == paths.model_ckpt

    with pytest.raises(ValueError, match="Invalid arguments"):
        set_probing_config(layer=1)


def test_resolve_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    assert resolve_device(None) == "cpu"
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(RuntimeError, match="cuda"):
        resolve_device("cuda:0")


# ─────────────────────────────────────────────────────────────
# What a fold writes
# ─────────────────────────────────────────────────────────────


def test_layer_names_cgnn():
    config = cfg.Config({"num_attention_blocks": 3})
    assert probe_layer_names(config, "CGNN-3D") == ["layer_0", "layer_1", "layer_2", "layer_3"]


@pytest.mark.parametrize("gnn", ["DTI", "DTI-soft"])
def test_layer_names_dti(gnn):
    config = cfg.Config({"num_layers": 2, "num_attention_blocks": 1})
    assert probe_layer_names(config, gnn) == ["layer_0", "layer_1", "layer_2"]
    config["emit_tower_reprs"] = True
    assert probe_layer_names(config, gnn) == [
        "layer_0", "layer_1", "layer_2",
        "ligand_layer_0", "ligand_layer_1", "ligand_layer_2",
        "pocket_layer_0", "pocket_layer_1",
    ]


def test_expected_fold_artifacts(tmp_path):
    config = cfg.Config({"split_index": 1, "output_dir": tmp_path, "num_attention_blocks": 1})
    got = expected_fold_artifacts(config, ProbingJobSpec(save_predictions=False))
    assert [p.relative_to(tmp_path).as_posix() for p in got] == ["1/layer_0_1.pt", "1/layer_1_1.pt", "1/ids_1.pt"]

    got = expected_fold_artifacts(config, ProbingJobSpec(save_representations=False))
    assert [p.name for p in got] == ["preds_1.pt", "y_true_1.pt"]


def test_write_manifest(tmp_path):
    path = write_manifest(tmp_path / "out", {"spec": {"include_val": True}, "created": tmp_path})
    assert path.name == "manifest.json"
    assert json.loads(path.read_text()) == {"spec": {"include_val": True}, "created": str(tmp_path)}


# ─────────────────────────────────────────────────────────────
# Aggregation <-> load_fold_index
# ─────────────────────────────────────────────────────────────


def test_aggregates_are_fold_order_concatenations(tmp_path):
    rng = np.random.default_rng(0)
    sizes = [3, 5, 4]
    for k, n in enumerate(sizes):
        d = tmp_path / str(k)
        d.mkdir()
        torch.save(torch.as_tensor(rng.permutation(100)[:n] + 100 * k), d / f"ids_{k}.pt")
        torch.save(torch.as_tensor(rng.normal(size=(n, 2)), dtype=torch.float32), d / f"layer_1_{k}.pt")
        torch.save(torch.as_tensor(rng.normal(size=n)), d / f"preds_{k}.pt")
        torch.save(torch.as_tensor(rng.normal(size=n)), d / f"y_true_{k}.pt")

    config = cfg.Config({"output_dir": tmp_path, "k_fold": len(sizes)})
    aggregation.aggregate_ids(config)
    aggregation.aggregate_folds(config, "layer_1")
    aggregation.aggregate_predictions(config)

    ids = torch.load(tmp_path / "ids.pt")
    assert torch.equal(ids, torch.cat([torch.load(tmp_path / str(k) / f"ids_{k}.pt") for k in range(3)]))
    assert torch.load(tmp_path / "layer_1.pt").shape == (sum(sizes), 2)
    np.testing.assert_array_equal(load_fold_index(tmp_path), np.repeat([0, 1, 2], sizes))
    assert pd.read_csv(tmp_path / "predictions.csv")["fold"].tolist() == np.repeat([0, 1, 2], sizes).tolist()
