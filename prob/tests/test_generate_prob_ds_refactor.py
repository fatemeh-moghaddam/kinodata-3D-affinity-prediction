import json
from pathlib import Path

import pytest


def test_resolve_paths_is_patchable(tmp_path, monkeypatch):
    """
    `resolve_paths` should be a small, deterministic derivation step.
    This test patches its dependencies so it doesn't require real data/model files.
    """
    pytest.importorskip("prob.run_extraction")

    from prob.run_extraction import ProbingJobSpec, resolve_paths

    model_dir = tmp_path / "models" / "m"
    model_ckpt = model_dir / "x.ckpt"
    split_file = tmp_path / "split.csv"
    output_root = tmp_path / "out"
    gnn_cfg = model_dir / "config.json"

    monkeypatch.setattr("prob.run_extraction.get_model_dir", lambda **_: model_dir)
    monkeypatch.setattr("prob.run_extraction.get_model_ckpt", lambda _: model_ckpt)
    monkeypatch.setattr("prob.run_extraction.get_split_file", lambda *_args, **_kwargs: split_file)
    monkeypatch.setattr("prob.run_extraction.get_out_dir", lambda *_args, **_kwargs: output_root)
    monkeypatch.setattr("prob.run_extraction.get_gnn_config_path", lambda _: gnn_cfg)

    spec = ProbingJobSpec()
    paths = resolve_paths(spec, fold=0)

    assert paths.model_dir == model_dir
    assert paths.model_ckpt == model_ckpt
    assert paths.split_file == split_file
    assert paths.output_root_dir == output_root
    assert paths.gnn_config_path == gnn_cfg


def test_write_manifest_writes_valid_json(tmp_path):
    pytest.importorskip("prob.run_extraction")
    from prob.run_extraction import write_manifest

    payload = {"spec": {"gnn_model_type": "CGNN-3D"}, "folds": {"0": {"num_samples": 10}}}
    p = write_manifest(tmp_path, payload)

    assert p.exists()
    assert p.name == "manifest.json"
    loaded = json.loads(p.read_text(encoding="utf-8"))
    assert loaded == payload


def test_set_probing_config_does_not_parse_args_by_default(monkeypatch, tmp_path):
    """
    The refactor goal is: no hidden CLI parsing unless explicitly enabled.
    We stub kinodata.configuration `register/get` to avoid touching the real config system.
    """
    pytest.importorskip("prob.run_extraction")
    from prob import run_extraction as mod

    class FakeCfg(dict):
        # minimal attribute access used by set_probing_config
        def __getattr__(self, k):
            return self[k]

        def update_from_args(self):
            raise AssertionError("update_from_args should NOT be called when parse_args=False")

        def update(self, mapping, allow_duplicates=False):
            super().update(mapping)
            return self

    fake = FakeCfg(
        {
            "gnn_model_type": "CGNN-3D",
            "split_type": "random-k-fold",
            "filter_rmsd_max_value": 2,
            "graph_level": True,
            "split_index": 0,
            "dtype_out": None,
            "device": "cpu",
            "k_fold": 5,
            "seed": 123,
            "parse_args": False,
        }
    )

    monkeypatch.setattr(mod.cfg, "register", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod.cfg, "get", lambda *_args, **_kwargs: fake)

    # Patch downstream I/O so the function stays pure and fast.
    paths = mod.ResolvedPaths(
        model_dir=tmp_path / "m",
        model_ckpt=tmp_path / "m" / "x.ckpt",
        split_file=tmp_path / "split.csv",
        output_root_dir=tmp_path / "out",
        gnn_config_path=tmp_path / "m" / "config.json",
    )
    monkeypatch.setattr(mod, "resolve_paths", lambda *_args, **_kwargs: paths)
    monkeypatch.setattr(mod, "load_config", lambda *_args, **_kwargs: mod.cfg.Config({"batch_size": 1}))

    out = mod.set_probing_config()
    assert out["parse_args"] is False
    assert Path(out["output_dir"]) == paths.output_root_dir


def test_seed_everything_works_without_numpy(monkeypatch):
    """
    Environments may run tests without NumPy installed.
    `_seed_everything` should still work (it treats NumPy as optional).
    """
    pytest.importorskip("torch")
    pytest.importorskip("prob.run_extraction")

    # Ensure importing numpy fails inside the function.
    monkeypatch.setitem(__import__("sys").modules, "numpy", None)

    from prob.run_extraction import _seed_everything

    _seed_everything(123)  # should not raise

