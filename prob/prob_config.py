"""
Configuration of a probing run, in one place.

- get_ds_load_config: the condition's paths (which representations and targets
  to read, where to write). Its defaults are the default condition.
- ProbingExperimentSpec: everything a probe run does, grouped by the level it
  acts on: data, target, model (the probe), evaluation, compute.

Every level has two kinds of values:
- settings (dataclass fields, lower case): chosen per run. Each has one default,
  here, and one precedence rule shared by all of them:
      command line  >  config passed to main  >  environment variable  >  default
  The environment variables (PROB_*) exist so the cluster and local scripts can
  keep passing switches the way they always have.
- fixed values (class constants, UPPER CASE): the probing method itself -- seeds,
  inner CV, probe split, bootstrap. They cannot be overridden from the command
  line, the config or the environment, because changing one makes new results
  incomparable with old ones. Change them here, deliberately, or not at all.

Every run records the resolved spec, each setting's source and every fixed
value in its run manifest (see prob.run_manifest).
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Mapping, Optional, Sequence, Tuple

import kinodata.configuration as cfg

from prob.paths_and_io import GNN_MODEL_TYPES, get_out_dir


TARGET_FILE = None

SPLIT_TYPES = ("random-k-fold", "scaffold-k-fold", "pocket-k-fold")
RMSD_CUTOFFS = (2, 4, 6, None)


# ─────────────────────────────────────────────────────────────
# Condition: paths of the representations and targets a run reads
# ─────────────────────────────────────────────────────────────

def build_experiment_name(ds_cfg: cfg.Config, layer_num: int) -> str:
    if not ds_cfg.target_file:
        raise ValueError("target_file must be set before building an experiment name")
    parts = [
        f"gnn={ds_cfg.gnn_model_type}",
        f"rmsd={ds_cfg.filter_rmsd_max_value}",
        f"split={ds_cfg.split_type}",
        f"layer={layer_num}",
        f"target={Path(ds_cfg.target_file).stem}",
    ]
    return "_".join(parts)


def get_ds_load_config(**kwargs):
    # default config values
    defaults = dict(
            gnn_model_type="CGNN-3D",
            split_type="random-k-fold",
            filter_rmsd_max_value=2,
            graph_level=True,
            split_index=None,
            dtype_out=None,  # None means no dtype conversion
            device="cpu",
            target_file=TARGET_FILE or "",  # empty str so argparser registers it as str type
        )

    allowed_keys = set(defaults.keys()) | {"config_name"}
    invalid = kwargs.keys() - allowed_keys
    if invalid:
        raise ValueError(f"Invalid arguments: {invalid}")

    if "gnn_model_type" in kwargs:
        assert kwargs["gnn_model_type"] in GNN_MODEL_TYPES, "Invalid GNN model type"
    if "split_type" in kwargs:
        assert kwargs["split_type"] in SPLIT_TYPES, "Invalid split type"
    if "filter_rmsd_max_value" in kwargs:
        assert kwargs["filter_rmsd_max_value"] in set(RMSD_CUTOFFS), "Invalid RMSD threshold"
    if "split_index" in kwargs:
        assert kwargs["split_index"] in set({0, 1, 2, 3, 4, None}), "Invalid split index"
    config_name = kwargs.get("config_name", "prob_ds_load")

    # merge: kwargs overrides defaults
    config_args = {**defaults, **{k: v for k, v in kwargs.items() if k != "config_name"}}
    cfg.register(config_name, **config_args)
    prob_ds_config = cfg.get(config_name)

    # In notebooks, argparse sees Jupyter kernel args and can crash.
    # Keep CLI behavior unchanged: only parse argv outside ipykernel.
    if "ipykernel" not in sys.modules:
        prob_ds_config = prob_ds_config.update_from_args()

    # Resolve paths only AFTER the CLI overrides are applied: gnn_model_type,
    # filter_rmsd_max_value and split_type are all part of output_dir, so
    # resolving them from `defaults` would pin every CLI run to the *default*
    # model's directory -- e.g. a `--gnn_model_type CGNN` job would silently
    # read X from and write its results into data/probing/CGNN-3D/...
    gnn_model_type = prob_ds_config["gnn_model_type"]
    assert gnn_model_type in GNN_MODEL_TYPES, (
        f"Invalid GNN model type: {gnn_model_type!r}"
    )
    output_dir = get_out_dir(
        gnn_model_type,
        prob_ds_config["filter_rmsd_max_value"],
        prob_ds_config["split_type"],
        split_fold=None,
    )
    # target_dir stays model-agnostic: data/probing/targets
    return prob_ds_config.update(
        {"output_dir": output_dir, "target_dir": output_dir.parents[2] / "targets"}
    )


# ─────────────────────────────────────────────────────────────
# ProbingExperimentSpec: one level per class
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProbeDataSettings:
    """Which representations are probed. The condition fields have no default
    here: they come from get_ds_load_config, which also turns them into paths.
    How the representations were extracted (e.g. test only or test + val) is
    fixed at extraction and copied from its manifest into the run manifest."""

    gnn_model_type: str
    split_type: str
    filter_rmsd_max_value: Optional[float]
    #: Layers to probe. None = every aggregated layer_*.pt in the condition's dir.
    layers: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.gnn_model_type not in GNN_MODEL_TYPES:
            raise ValueError(f"gnn_model_type {self.gnn_model_type!r} not in {GNN_MODEL_TYPES}")
        if self.split_type not in SPLIT_TYPES:
            raise ValueError(f"split_type {self.split_type!r} not in {SPLIT_TYPES}")
        if self.filter_rmsd_max_value not in RMSD_CUTOFFS:
            raise ValueError(f"filter_rmsd_max_value {self.filter_rmsd_max_value!r} not in {RMSD_CUTOFFS}")
        if self.layers is not None and (not self.layers or min(self.layers) < 0):
            raise ValueError(f"layers must be non-negative and non-empty, got {self.layers}")


@dataclass(frozen=True)
class ProbeTargetSettings:
    """What the probes predict, and the shuffled-label control."""

    target_file: str
    #: Also fit every probe on targets permuted across idents (a random-label control).
    run_shuffled_baseline: bool = True
    #: Suffix of the baseline's target directory: <target>_<baseline_tag>.
    baseline_tag: str = "shuffled_ident"

    #: One permutation, the same for every target and layer.
    PERMUTATION_SEED: ClassVar[int] = 96

    def __post_init__(self):
        if not self.target_file:
            raise ValueError(
                "target_file must be set (e.g. --target_file affinity.pt). "
                "Check that it is passed as a CLI argument."
            )


@dataclass(frozen=True)
class ProbeModelSettings:
    """Which probes are trained, and how they are trained."""

    run_linear_models: bool = True
    run_non_linear_models: bool = False
    #: Subset of prob_models.NONLINEAR_PROBES by name. Empty = all registered.
    nonlinear_models: Tuple[str, ...] = ()
    #: One probe per GNN checkpoint (CV fold).
    per_ckpt: bool = True
    #: One probe over all folds' rows, written as "<probe>_pooled".
    pooled: bool = True
    #: Share each fold's tuned params across layers instead of tuning every layer.
    reuse_best_params: bool = False
    #: Where shared params live. None = <target>/<probe>/shared_best_params.
    best_params_cache_dir: Optional[str] = None

    #: GridSearchCV folds inside each probe's train rows (hyperparameters only).
    INNER_CV_FOLDS: ClassVar[int] = 3
    INNER_CV_SEED: ClassVar[int] = 96
    #: Metric GridSearchCV picks the best params by.
    REFIT_METRIC: ClassVar[str] = "r2"
    #: random_state of every probe estimator in prob_models.
    ESTIMATOR_SEED: ClassVar[int] = 96
    FEATURE_SCALING: ClassVar[str] = "StandardScaler in the probe pipeline, fit on probe-train rows"

    def __post_init__(self):
        if not (self.per_ckpt or self.pooled):
            raise ValueError("per_ckpt and pooled are both off; nothing to run")


@dataclass(frozen=True)
class ProbeEvalSettings:
    """How probe results are measured. Fixed values only: they define which
    molecules every probe is tested on and how uncertainty is estimated."""

    #: The probe train/test assignment by ident is the file data/probing/probe_split.csv.
    #: Seed and size are used only if that file has to be built.
    PROBE_SPLIT_SEED: ClassVar[int] = 0
    PROBE_TEST_SIZE: ClassVar[float] = 0.1
    #: A run fails if its test fraction drifts further than this from the file's.
    MAX_TEST_FRACTION_DEVIATION: ClassVar[float] = 0.01
    #: The same within one checkpoint's rows, where the fraction varies more.
    MAX_TEST_FRACTION_DEVIATION_PER_FOLD: ClassVar[float] = 0.015
    #: Bootstrap CIs resample the probe test rows.
    BOOTSTRAP_N: ClassVar[int] = 1000
    BOOTSTRAP_CONFIDENCE: ClassVar[float] = 0.95
    BOOTSTRAP_SEED: ClassVar[int] = 96


@dataclass(frozen=True)
class ComputeSettings:
    """Where and how fast. Changes run time, not results -- except the random
    forest probe, which is a different algorithm on cuda (see prob_models)."""

    device: str = "cpu"
    #: GridSearchCV workers. None = scheduler env (SLURM_CPUS_PER_TASK, NSLOTS,
    #: OMP_NUM_THREADS), then request_CPUs in cluster/run_prob.sub, then 16.
    n_jobs: Optional[int] = None


@dataclass(frozen=True)
class ProbingExperimentSpec:
    """Everything one probe run does, by level: spec.model.pooled,
    spec.evaluation.BOOTSTRAP_N, ..."""

    data: ProbeDataSettings
    target: ProbeTargetSettings
    model: ProbeModelSettings = ProbeModelSettings()
    evaluation: ProbeEvalSettings = ProbeEvalSettings()
    compute: ComputeSettings = ComputeSettings()


LEVELS = ("data", "target", "model", "evaluation", "compute")
_LEVEL_CLASSES = {
    "data": ProbeDataSettings,
    "target": ProbeTargetSettings,
    "model": ProbeModelSettings,
    "evaluation": ProbeEvalSettings,
    "compute": ComputeSettings,
}


# ─────────────────────────────────────────────────────────────
# Resolving the spec: command line > config > environment > default
# ─────────────────────────────────────────────────────────────

def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    raise ValueError(f"expected 1/0/true/false/yes/no, got {value!r}")


def _parse_layers(value: Any) -> Optional[Tuple[int, ...]]:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [p for p in value.split(",") if p.strip()]
        return tuple(int(p) for p in parts) or None
    return tuple(int(v) for v in value) or None


def _parse_names(value: Any) -> Tuple[str, ...]:
    if isinstance(value, str):
        return tuple(p.strip() for p in value.split(",") if p.strip())
    return tuple(value or ())


def _parse_optional_str(value: Any) -> Optional[str]:
    return str(value) if value not in (None, "") else None


def _parse_optional_int(value: Any) -> Optional[int]:
    return int(value) if value not in (None, "") else None


@dataclass(frozen=True)
class _Setting:
    level: str
    name: str
    parse: Callable[[Any], Any]
    env: Optional[str] = None


#: Every per-run setting the command line, config or environment may set.
#: Condition fields (gnn, split, rmsd, target_file) are read from the config only:
#: get_ds_load_config has already applied the command line to them.
RUN_SETTINGS: Tuple[_Setting, ...] = (
    _Setting("data", "layers", _parse_layers, "PROB_LAYERS"),
    _Setting("target", "run_shuffled_baseline", _parse_bool, "PROB_RUN_SHUFFLED_BASELINE"),
    _Setting("target", "baseline_tag", str, "PROB_BASELINE_TAG"),
    _Setting("model", "run_linear_models", _parse_bool, "PROB_RUN_LINEAR_MODELS"),
    _Setting("model", "run_non_linear_models", _parse_bool, "PROB_RUN_NON_LINEAR_MODELS"),
    _Setting("model", "nonlinear_models", _parse_names, "PROB_NONLINEAR_MODELS"),
    _Setting("model", "per_ckpt", _parse_bool, "PROB_PER_CKPT"),
    _Setting("model", "pooled", _parse_bool, "PROB_POOLED"),
    _Setting("model", "reuse_best_params", _parse_bool, "PROB_REUSE_BEST_PARAMS"),
    _Setting("model", "best_params_cache_dir", _parse_optional_str, "PROB_BEST_PARAMS_CACHE_DIR"),
    _Setting("compute", "device", str),
    _Setting("compute", "n_jobs", _parse_optional_int),
)
CONDITION_FIELDS = (
    ("data", "gnn_model_type"),
    ("data", "split_type"),
    ("data", "filter_rmsd_max_value"),
    ("target", "target_file"),
)
PROB_ENV_VARS = tuple(s.env for s in RUN_SETTINGS if s.env)


def _cli_values(argv: Optional[Sequence[str]]) -> Dict[str, str]:
    """The run settings given on the command line (and only those)."""
    if argv is None:
        return {}
    parser = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    for setting in RUN_SETTINGS:
        parser.add_argument(f"--{setting.name}", type=str)
    known, _ = parser.parse_known_args(list(argv))
    return vars(known)


def _on_cli(argv: Optional[Sequence[str]], name: str) -> bool:
    return argv is not None and any(a == f"--{name}" or a.startswith(f"--{name}=") for a in argv)


def resolve_experiment_spec(
    config: Mapping[str, Any],
    argv: Optional[Sequence[str]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Tuple[ProbingExperimentSpec, Dict[str, str]]:
    """
    Build the spec for one run and say where each setting came from.

    config: the condition config (get_ds_load_config, or a dict in tests and
    notebooks); may also carry any run setting by name. argv: the command line
    (None = not parsed). environ: defaults to os.environ.

    Returns (spec, sources) with sources["<level>.<name>"] = "cli", "config",
    "env <VAR>=<value>" or "default". Unknown PROB_* variables raise, so a typo'd
    switch fails instead of being ignored.
    """
    environ = os.environ if environ is None else environ
    unknown = sorted(k for k in environ if k.startswith("PROB_") and k not in PROB_ENV_VARS)
    if unknown:
        raise ValueError(f"Unknown environment variable(s) {unknown}; known: {list(PROB_ENV_VARS)}")

    values: Dict[str, Dict[str, Any]] = {level: {} for level in LEVELS}
    sources: Dict[str, str] = {}

    for level, name in CONDITION_FIELDS:
        values[level][name] = config.get(name)
        sources[f"{level}.{name}"] = "cli" if _on_cli(argv, name) else "config"

    cli = _cli_values(argv)
    for setting in RUN_SETTINGS:
        key = f"{setting.level}.{setting.name}"
        try:
            if setting.name in cli:
                value, source = setting.parse(cli[setting.name]), "cli"
            elif setting.name in config:
                value, source = setting.parse(config[setting.name]), "config"
            elif setting.env and setting.env in environ:
                raw = environ[setting.env]
                value, source = setting.parse(raw), f"env {setting.env}={raw!r}"
            else:
                sources[key] = "default"
                continue
        except ValueError as err:
            raise ValueError(f"{key}: {err}") from err
        values[setting.level][setting.name] = value
        sources[key] = source

    spec = ProbingExperimentSpec(**{
        level: _LEVEL_CLASSES[level](**values[level]) for level in LEVELS
    })
    return spec, sources


def fixed_values(level_cls: type) -> Dict[str, Any]:
    """A level's fixed values (its UPPER CASE class constants)."""
    return {
        name: getattr(level_cls, name)
        for name in dir(level_cls)
        if name.isupper() and not name.startswith("_")
    }


def spec_record(spec: ProbingExperimentSpec, sources: Mapping[str, str]) -> Dict[str, Any]:
    """The spec as the run manifest stores it: per level, every setting with its
    value and source, and every fixed value."""
    record = {}
    for level in LEVELS:
        settings = getattr(spec, level)
        record[level] = {
            "settings": {
                f.name: {
                    "value": getattr(settings, f.name),
                    "source": sources.get(f"{level}.{f.name}", "default"),
                }
                for f in fields(settings)
            },
            "fixed": fixed_values(type(settings)),
        }
    return record
