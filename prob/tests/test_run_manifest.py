"""prob.run_manifest helpers: file hashes, the code version without a git binary,
and what a probe registry entry is recorded as."""
from __future__ import annotations

import hashlib
import json

import numpy as np
from sklearn.linear_model import Ridge

from prob import run_manifest as rm
from prob.prob_models import HybridRandomForestRegressor


def test_file_record_hashes_existing_files_and_flags_missing_ones(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"abc" * 1000)
    rec = rm.file_record(f)
    assert rec["exists"] and rec["bytes"] == 3000
    assert rec["sha256"] == hashlib.sha256(b"abc" * 1000).hexdigest()
    assert rm.file_record(tmp_path / "nope") == {"path": str(tmp_path / "nope"), "exists": False}
    assert rm.file_record(None) == {"path": None, "exists": False}


def _fake_repo(root, head, refs=None, packed=None):
    git = root / ".git"
    git.mkdir()
    (git / "HEAD").write_text(head)
    for ref, sha in (refs or {}).items():
        (git / ref).parent.mkdir(parents=True, exist_ok=True)
        (git / ref).write_text(sha + "\n")
    if packed:
        (git / "packed-refs").write_text(packed)


def test_git_commit_from_loose_ref_without_git_binary(tmp_path, monkeypatch):
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    _fake_repo(tmp_path, "ref: refs/heads/probing\n", refs={"refs/heads/probing": "a" * 40})
    # tmp_path is no real repository, so the git binary fails and the files are read.
    assert rm.git_info(tmp_path) == {"commit": "a" * 40, "source": ".git files", "dirty": None}


def test_git_commit_from_packed_refs_and_detached_head(tmp_path, monkeypatch):
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    packed = tmp_path / "packed"
    packed.mkdir()
    _fake_repo(packed, "ref: refs/heads/main\n", packed=f"# pack-refs\n{'b' * 40} refs/heads/main\n")
    assert rm.git_info(packed)["commit"] == "b" * 40

    detached = tmp_path / "detached"
    detached.mkdir()
    _fake_repo(detached, "c" * 40 + "\n")
    assert rm.git_info(detached)["commit"] == "c" * 40


def test_git_commit_falls_back_to_env_then_none(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_COMMIT", "d" * 40)
    assert rm.git_info(tmp_path) == {"commit": "d" * 40, "source": "GIT_COMMIT env", "dirty": None}
    monkeypatch.delenv("GIT_COMMIT")
    assert rm.git_info(tmp_path)["commit"] is None


def test_probe_record_is_json_and_names_the_forest_backend():
    ridge = rm.probe_record({"name": "ridge", "estimator": Ridge(), "param_grid": {"model__alpha": np.logspace(-1, 1, 3)}}, 4)
    assert ridge["n_jobs"] == 4 and ridge["param_grid"] == {"model__alpha": [0.1, 1.0, 10.0]}
    json.dumps(ridge)

    gpu = rm.probe_record({"name": "random_forest", "estimator": HybridRandomForestRegressor(device="cuda"), "n_jobs": 1}, 4)
    cpu = rm.probe_record({"name": "random_forest", "estimator": HybridRandomForestRegressor()}, 4)
    assert gpu["backend"] == "xgboost random-forest mode" and gpu["n_jobs"] == 1
    assert cpu["backend"] == "sklearn RandomForestRegressor"


def test_write_manifest_puts_one_copy_in_each_dir(tmp_path):
    payload = {"run_id": "r1", "path": tmp_path, "pair": ("a", 1)}
    written = rm.write_manifest(payload, [tmp_path / "x", tmp_path / "y"])
    assert [p.relative_to(tmp_path).as_posix() for p in written] == [
        "x/run_manifests/r1.json", "y/run_manifests/r1.json",
    ]
    assert json.loads(written[0].read_text()) == {"run_id": "r1", "path": str(tmp_path), "pair": ["a", 1]}
