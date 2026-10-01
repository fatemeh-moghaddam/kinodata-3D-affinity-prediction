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
from prob.prob_config import ExtractionDataSettings, ExtractionSpec, resolve_extraction_spec
from prob.run_extraction import (
    ResolvedPaths,
    build_fold_config,
    check_checkpoint_config,
    expected_fold_artifacts,
    probe_layer_names,
    resolve_device,
    resolve_paths,
    write_manifest,
)

CONDITION_ARGS = ["--gnn_model_type", "CGNN-3D", "--split_type", "random-k-fold", "--filter_rmsd_max_value", "2"]


def make_spec(*extra: str, gnn: str = "CGNN-3D") -> ExtractionSpec:
    args = ["--gnn_model_type", gnn, *CONDITION_ARGS[2:], *extra]
    return resolve_extraction_spec(args, environ={})[0]


# ─────────────────────────────────────────────────────────────
# Spec: what a job is allowed to choose
# ─────────────────────────────────────────────────────────────


def test_condition_is_required_and_validated():
    with pytest.raises(ValueError, match="data.gnn_model_type: required"):
        resolve_extraction_spec(CONDITION_ARGS[2:], environ={})
    for bad in (["--gnn_model_type", "BAD"], ["--split_type", "BAD"], ["--filter_rmsd_max_value", "3"]):
        args = dict(zip(CONDITION_ARGS[::2], CONDITION_ARGS[1::2]))
        args[bad[0]] = bad[1]
        with pytest.raises(ValueError):
            resolve_extraction_spec([x for kv in args.items() for x in kv], environ={})


@pytest.mark.parametrize("text, value", [("2", 2), ("4.0", 4), ("6", 6), ("none", None)])
def test_rmsd_flag_parses_to_the_cutoff(text, value):
    args = [*CONDITION_ARGS[:4], "--filter_rmsd_max_value", text]
    assert resolve_extraction_spec(args, environ={})[0].data.filter_rmsd_max_value == value


def test_extraction_is_test_molecules_only_and_that_is_fixed():
    assert ExtractionDataSettings.INCLUDE_VAL is False
    with pytest.raises(ValueError, match="--include_val is fixed"):
        resolve_extraction_spec([*CONDITION_ARGS, "--include_val", "1"], environ={})
    with pytest.raises(ValueError, match="--seed is fixed"):
        resolve_extraction_spec([*CONDITION_ARGS, "--seed=1"], environ={})


def test_defaults_sources_and_env_alias():
    spec, sources = resolve_extraction_spec([*CONDITION_ARGS, "--overwrite", "1"], environ={"CPU_COUNT": "8"})
    assert (spec.outputs.save_representations, spec.outputs.save_predictions) == (True, True)
    assert spec.outputs.overwrite is True and sources["outputs.overwrite"] == "cli"
    assert spec.compute.num_processes == 8 and sources["compute.num_processes"] == "env CPU_COUNT='8'"
    assert spec.compute.device == "auto" and sources["compute.device"] == "default"
    assert sources["data.split_type"] == "cli"


@pytest.mark.parametrize("extra", [
    ["--save_representations", "0", "--save_predictions", "0"],
    ["--dtype_out", "int8"],
    ["--wandb_mode", "loud"],
])
def test_bad_settings_fail(extra):
    with pytest.raises(ValueError):
        resolve_extraction_spec([*CONDITION_ARGS, *extra], environ={})


def test_unknown_flags_fail():
    with pytest.raises(SystemExit):
        resolve_extraction_spec([*CONDITION_ARGS, "--k_folds", "3"], environ={})


# ─────────────────────────────────────────────────────────────
# Per fold: paths and the checkpoint's config
# ─────────────────────────────────────────────────────────────


def test_resolve_paths_loads_dti_soft_from_dti(tmp_path, monkeypatch):
    seen = {}

    def fake_model_dir(**kw):
        seen.update(kw)
        return tmp_path / kw["model_type"]

    monkeypatch.setattr(run_extraction, "get_model_dir", fake_model_dir)
    monkeypatch.setattr(run_extraction, "get_model_ckpt", lambda d: d / "x.ckpt")
    monkeypatch.setattr(run_extraction, "get_split_file", lambda *a, **k: tmp_path / "1_5.csv")
    monkeypatch.setattr(run_extraction, "get_out_dir", lambda gnn, *a, **k: tmp_path / "out" / gnn)

    paths = resolve_paths(make_spec(gnn="DTI-soft"), fold=3)
    assert seen["model_type"] == "DTI" and seen["split_fold"] == 3
    assert paths.model_ckpt == tmp_path / "DTI" / "x.ckpt"
    assert paths.gnn_config_path == tmp_path / "DTI" / "config.json"
    assert paths.output_root_dir == tmp_path / "out" / "DTI-soft"


CHECKPOINT_CONFIG = {
    "split_type": "random-k-fold", "split_index": 2, "k_fold": 5, "filter_rmsd_max_value": 2.0,
    "seed": 420, "batch_size": 32, "num_attention_blocks": 3,
    "gnn_model_type": "CGNN", "emit_tower_reprs": True,
}


@pytest.fixture
def fake_fold_paths(tmp_path, monkeypatch):
    paths = ResolvedPaths(
        model_dir=tmp_path, model_ckpt=tmp_path / "x.ckpt", split_file=tmp_path / "3_5.csv",
        output_root_dir=tmp_path / "out", gnn_config_path=tmp_path / "config.json",
    )
    monkeypatch.setattr(run_extraction, "resolve_paths", lambda *a, **k: paths)
    monkeypatch.setattr(run_extraction, "load_config", lambda _: cfg.Config(dict(CHECKPOINT_CONFIG)))
    return paths


def test_fold_config_keeps_extraction_choices_over_checkpoint_config(fake_fold_paths):
    spec = make_spec("--dtype_out", "float16", gnn="DTI-soft")
    config = build_fold_config(spec, fold=2, device="cpu")
    # The extraction decides these ...
    assert config.gnn_model_type == "DTI-soft" and config.emit_tower_reprs is False
    assert (config.split_index, config.device, config.dtype_out) == (2, "cpu", "float16")
    assert config.graph_level is True and config.num_processes == 16
    # ... the checkpoint describes the model.
    assert config.num_attention_blocks == 3 and config.batch_size == 32
    assert config.output_dir == fake_fold_paths.output_root_dir
    assert config.model_ckpt == fake_fold_paths.model_ckpt


@pytest.mark.parametrize("key, wrong", [
    ("split_index", 1), ("split_type", "scaffold-k-fold"), ("k_fold", 10), ("filter_rmsd_max_value", 4),
])
def test_checkpoint_trained_for_another_condition_is_refused(key, wrong, tmp_path):
    spec = make_spec()
    check_checkpoint_config(CHECKPOINT_CONFIG, spec, fold=2, source=tmp_path)  # matches
    with pytest.raises(ValueError, match=key):
        check_checkpoint_config({**CHECKPOINT_CONFIG, key: wrong}, spec, fold=2, source=tmp_path)


def test_unfiltered_condition_matches_none_spelled_as_text(tmp_path):
    spec = make_spec()
    spec = resolve_extraction_spec([*CONDITION_ARGS[:4], "--filter_rmsd_max_value", "none"], environ={})[0]
    check_checkpoint_config({**CHECKPOINT_CONFIG, "filter_rmsd_max_value": "None"}, spec, 2, tmp_path)


def test_more_than_one_checkpoint_is_refused(tmp_path):
    from prob.paths_and_io import get_model_ckpt
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "model.ckpt").touch()
    assert get_model_ckpt(tmp_path) == tmp_path / "a" / "model.ckpt"
    (tmp_path / "last.ckpt").touch()
    with pytest.raises(RuntimeError, match="2 checkpoints"):
        get_model_ckpt(tmp_path)


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
    got = expected_fold_artifacts(config, make_spec("--save_predictions", "0"))
    assert [p.relative_to(tmp_path).as_posix() for p in got] == ["1/layer_0_1.pt", "1/layer_1_1.pt", "1/ids_1.pt"]

    got = expected_fold_artifacts(config, make_spec("--save_representations", "0"))
    assert [p.name for p in got] == ["preds_1.pt", "y_true_1.pt"]


def test_write_manifest(tmp_path):
    path = write_manifest(tmp_path / "out", {"spec": {"include_val": True}, "created": tmp_path})
    assert path.name == "manifest.json"
    assert json.loads(path.read_text()) == {"spec": {"include_val": True}, "created": str(tmp_path)}


# ─────────────────────────────────────────────────────────────
# main(): one job end to end, with the model and dataset faked
# ─────────────────────────────────────────────────────────────


def test_main_extracts_test_molecules_of_every_fold_and_records_it(tmp_path, monkeypatch):
    out = tmp_path / "out"
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "model.ckpt").write_bytes(b"weights")
    (model_dir / "config.json").write_text("{}")
    built = []

    def fake_paths(spec, fold):
        return ResolvedPaths(model_dir, model_dir / "model.ckpt", tmp_path / f"{fold + 1}_5.csv", out,
                             model_dir / "config.json")

    def fake_ds(split_path, filter_rmsd_max_value, include_val, num_processes):
        built.append((split_path.name, include_val, num_processes))
        return list(range(3))

    def fake_run_fold(ds, model, config, save_representations, save_predictions):
        fold_dir = out / str(config.split_index)
        fold_dir.mkdir(parents=True, exist_ok=True)
        k = config.split_index
        torch.save(torch.arange(3) + 10 * k, fold_dir / f"ids_{k}.pt")
        for n in range(2):
            torch.save(torch.zeros(3, 2), fold_dir / f"layer_{n}_{k}.pt")
        torch.save(torch.zeros(3), fold_dir / f"preds_{k}.pt")
        torch.save(torch.zeros(3), fold_dir / f"y_true_{k}.pt")

    monkeypatch.setattr(run_extraction, "resolve_paths", fake_paths)
    monkeypatch.setattr(run_extraction, "get_out_dir", lambda *a, **k: out)
    monkeypatch.setattr(run_extraction, "load_config", lambda _: cfg.Config(
        {"split_type": "random-k-fold", "k_fold": 5, "num_attention_blocks": 1, "batch_size": 4}))
    monkeypatch.setattr(run_extraction, "build_gnn_model", lambda config: torch.nn.Identity())
    monkeypatch.setattr(run_extraction, "build_kd_ds", fake_ds)
    monkeypatch.setattr(run_extraction, "run_fold", fake_run_fold)
    monkeypatch.setattr(run_extraction.wandb, "init", lambda **kw: None)

    manifest_path = run_extraction.main([*CONDITION_ARGS, "--device", "cpu"])

    assert built == [(f"{k + 1}_5.csv", False, 16) for k in range(5)]
    assert torch.equal(torch.load(out / "ids.pt"), torch.cat([torch.arange(3) + 10 * k for k in range(5)]))
    m = json.loads(manifest_path.read_text())
    assert m["spec"]["data"]["fixed"]["INCLUDE_VAL"] is False
    assert m["spec"]["compute"]["settings"]["device"] == {"value": "cpu", "source": "cli"}
    assert m["resolved"]["device"] == "cpu"
    fold0 = m["folds"]["0"]
    assert fold0["checkpoint"]["sha256"] and fold0["checkpoint_config"]["exists"]
    assert fold0["num_samples"] == 3 and fold0["reused"] is False

    # A second run resumes: every fold is on disk, nothing is rebuilt.
    built.clear()
    m = json.loads(run_extraction.main([*CONDITION_ARGS, "--device", "cpu"]).read_text())
    assert built == [] and all(f["reused"] for f in m["folds"].values())


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
