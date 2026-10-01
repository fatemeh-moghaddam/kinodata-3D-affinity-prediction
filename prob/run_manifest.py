"""
The probe run manifest: one JSON per prob_orchestrate run that says exactly what
the run did, so a result can be traced back to its conditions without reading
code, shell history or the cluster .out file.

Written to <output_dir>/<target>/experiments/run_manifests/<run_id>.json (and
the same file under the baseline target when the baseline runs). Every probe
summary.json the run writes carries the same run_id, which is the link back.

The manifest is written once when the run starts (status "running") and again
when it ends ("finished", or "failed" with the error). A manifest left at
"running" means the job died without Python seeing it (killed, out of memory).

What it records:
  - request: the condition (gnn, split, rmsd, target, device, baseline tag)
  - settings: every run toggle, its resolved value, and where the value came
    from (env var, config, or code default)
  - constants: seeds, inner CV folds, test size, bootstrap settings, tolerances
  - probes: each probe's estimator, fixed params, grid and n_jobs
  - inputs: path + sha256 of the extraction manifest, ids.pt, every probed
    layer file, the target file and the probe split file
  - data: how many rows were loaded, how many had no target and were dropped
  - code: git commit (and whether the tree was dirty), package versions, host
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

MANIFEST_DIRNAME = "run_manifests"

#: Environment variables recorded verbatim (in addition to every PROB_* variable).
RECORDED_ENV = (
    "HOME_PROJ_DIR",
    "CPU_COUNT",
    "NSLOTS",
    "SLURM_CPUS_PER_TASK",
    "OMP_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES",
    "WANDB_MODE",
    "GIT_COMMIT",
)

RECORDED_PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "torch", "xgboost")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    """Sortable and unique enough for one user's runs: UTC time plus the process id."""
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Optional[Path]) -> Dict[str, Any]:
    """{"path", "exists", "sha256", "bytes"} for one input file."""
    if path is None:
        return {"path": None, "exists": False}
    path = Path(path)
    if not path.is_file():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _read_git_files(root: Path) -> Optional[str]:
    """HEAD's commit read straight from .git, for machines without a git binary."""
    git_dir = root / ".git"
    head = git_dir / "HEAD"
    if not head.is_file():
        return None
    ref = head.read_text().strip()
    if not ref.startswith("ref: "):
        return ref or None  # detached HEAD holds the commit itself
    ref = ref[len("ref: "):]
    loose = git_dir / ref
    if loose.is_file():
        return loose.read_text().strip() or None
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref:
                return parts[0]
    return None


def git_info(root: Path) -> Dict[str, Any]:
    """
    Commit of the code that ran. Tries the git binary (which also tells whether the
    tree had uncommitted changes), then the .git files, then a GIT_COMMIT env var.
    """
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            stderr=subprocess.DEVNULL,
        ).decode()
        changed = [line[3:] for line in status.splitlines() if line.strip()]
        return {"commit": commit, "source": "git", "dirty": bool(changed), "changed_files": changed}
    except Exception:
        pass
    commit = _read_git_files(root)
    if commit:
        return {"commit": commit, "source": ".git files", "dirty": None}
    commit = os.environ.get("GIT_COMMIT")
    if commit:
        return {"commit": commit, "source": "GIT_COMMIT env", "dirty": None}
    return {"commit": None, "source": None, "dirty": None}


def package_versions(names: Iterable[str] = RECORDED_PACKAGES) -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {"python": sys.version.split()[0]}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def recorded_env() -> Dict[str, str]:
    keys = sorted({k for k in os.environ if k.startswith("PROB_")} | set(RECORDED_ENV))
    return {k: os.environ[k] for k in keys if k in os.environ}


def _jsonable(value: Any) -> Any:
    """Params and grids as they read back from JSON (numpy -> python, tuples -> lists)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def probe_record(entry: Dict[str, Any], default_n_jobs: int) -> Dict[str, Any]:
    """What one registry entry will run: estimator, its fixed params, grid, n_jobs."""
    estimator = entry["estimator"]
    params = estimator.get_params(deep=False) if hasattr(estimator, "get_params") else {}
    record = {
        "name": entry["name"],
        "estimator": f"{type(estimator).__module__}.{type(estimator).__name__}",
        "estimator_params": _jsonable(params),
        "param_grid": _jsonable(entry.get("param_grid") or {}),
        "n_jobs": entry.get("n_jobs") or default_n_jobs,
    }
    # The random forest is a different algorithm on GPU (see HybridRandomForestRegressor).
    if type(estimator).__name__ == "HybridRandomForestRegressor":
        record["backend"] = (
            "xgboost random-forest mode" if params.get("device") == "cuda" else "sklearn RandomForestRegressor"
        )
    return record


def write_manifest(payload: Dict[str, Any], dirs: Iterable[Path]) -> list[Path]:
    """Write the same manifest into each experiments dir; returns the files written."""
    written = []
    text = json.dumps(_jsonable(payload), indent=2)
    for experiments_dir in dirs:
        out = Path(experiments_dir) / MANIFEST_DIRNAME / f"{payload['run_id']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        written.append(out)
    return written


def run_host() -> str:
    return platform.node()
