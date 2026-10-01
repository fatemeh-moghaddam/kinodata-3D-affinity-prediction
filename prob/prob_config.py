"""
Configuration of the probing pipeline, in one place. Read this file to know
every condition a run can have.

- ExtractionSpec (prob/run_extraction.py): which trained GNN is run over which
  molecules, what is saved, on what hardware.
- get_ds_load_config: a probing condition's paths (which representations and
  targets to read, where to write). Its defaults are the default condition.
- ProbingExperimentSpec (prob/prob_orchestrate.py): everything a probe run does,
  by level: data, target, model (the probe), evaluation, compute.

Every level has two kinds of values:
- settings (dataclass fields, lower case): chosen per run. Each has one default,
  here, and one precedence rule shared by all of them (prob/spec_tools.py):
      command line  >  config  >  environment variable  >  default
  Environment variables (PROB_*, CPU_COUNT) exist so the cluster and local
  scripts can keep passing switches the way they always have.
- fixed values (class constants, UPPER CASE): the method itself -- seeds, which
  molecules are extracted, inner CV, probe split, bootstrap. No run can override
  them, and naming one on the command line is an error, because changing one
  makes new results incomparable with old ones. Change them here, deliberately,
  or not at all.

Each run records its resolved spec, each setting's source and every fixed value
in its manifest.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence, Tuple

import kinodata.configuration as cfg

from prob.paths_and_io import GNN_MODEL_TYPES, get_out_dir
from prob.spec_tools import (
    Setting,
    add_setting_flags,
    parse_bool,
    parse_int_tuple,
    parse_names,
    parse_optional_int,
    parse_optional_str,
    parse_rmsd,
    reject_fixed_flags,
    resolve,
    spec_record,  # noqa: F401  (re-exported: the manifests' spec block)
)


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


def _check_condition(gnn_model_type: str, split_type: str, filter_rmsd_max_value: Any) -> None:
    if gnn_model_type not in GNN_MODEL_TYPES:
        raise ValueError(f"gnn_model_type {gnn_model_type!r} not in {GNN_MODEL_TYPES}")
    if split_type not in SPLIT_TYPES:
        raise ValueError(f"split_type {split_type!r} not in {SPLIT_TYPES}")
    if filter_rmsd_max_value not in RMSD_CUTOFFS:
        raise ValueError(f"filter_rmsd_max_value {filter_rmsd_max_value!r} not in {RMSD_CUTOFFS}")


# ─────────────────────────────────────────────────────────────
# ExtractionSpec: running the trained GNNs to get representations
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExtractionDataSettings:
    """Which trained GNN is run, over which molecules. The condition has no
    default: an extraction job must name it."""

    gnn_model_type: str
    split_type: str
    filter_rmsd_max_value: Optional[float]

    #: Each fold's checkpoint is run on that fold's *test* molecules only: never
    #: trained on, and not used to select the checkpoint (min val/mae). The folds'
    #: test sets together are about half of each cutoff's dataset, and they differ
    #: between split types -- so random, scaffold and pocket hold different
    #: molecules and are compared unpaired.
    INCLUDE_VAL: ClassVar[bool] = False
    #: Folds the GNNs were trained on: models/.../<fold>/, data/processed/.../<k>_5.csv.
    K_FOLD: ClassVar[int] = 5
    #: One representation per complex, pooled by the model's own readout.
    GRAPH_LEVEL: ClassVar[bool] = True

    def __post_init__(self):
        _check_condition(self.gnn_model_type, self.split_type, self.filter_rmsd_max_value)


DTYPES_OUT = (None, "float32", "fp32", "float16", "fp16", "bfloat16")


@dataclass(frozen=True)
class ExtractionOutputSettings:
    """What a fold writes to data/probing/<gnn>/rmsd_cutoff_<x>/<split>/<fold>/."""

    #: layer_<n>_<fold>.pt and ids_<fold>.pt (what probing reads).
    save_representations: bool = True
    #: preds_<fold>.pt and y_true_<fold>.pt: the GNN's own affinity predictions.
    save_predictions: bool = True
    #: DTI only: also the per-tower ligand_layer_* / pocket_layer_* behind layer_*.
    emit_tower_reprs: bool = False
    #: Cast saved representations. None = keep the model's dtype (float32).
    dtype_out: Optional[str] = None
    #: Recompute folds whose files already exist (default: skip them).
    overwrite: bool = False

    def __post_init__(self):
        if not (self.save_representations or self.save_predictions):
            raise ValueError("save_representations and save_predictions are both off; nothing to write")
        if self.dtype_out not in DTYPES_OUT:
            raise ValueError(f"dtype_out {self.dtype_out!r} not in {DTYPES_OUT}")


@dataclass(frozen=True)
class ExtractionComputeSettings:
    """Where and how fast. Changes run time, not the representations."""

    #: "auto" = cuda if available, else cpu. An explicit "cuda" that is not
    #: available is an error, not a silent fallback.
    device: str = "auto"
    #: Inference batch size. None = the checkpoint's training batch_size, known to fit.
    infer_batch_size: Optional[int] = None
    #: DataLoader workers on GPU (CPU runs load in the main process).
    eval_num_workers: int = 4
    #: Processes for loading the dataset; also caps eval_num_workers.
    num_processes: int = 16
    wandb_mode: str = "disabled"

    #: Seeds python, numpy and torch, with deterministic cuDNN. Inference itself
    #: involves no sampling; this pins anything a model might draw.
    SEED: ClassVar[int] = 96

    def __post_init__(self):
        if self.wandb_mode not in {"disabled", "online", "offline"}:
            raise ValueError(f"wandb_mode {self.wandb_mode!r} not in disabled/online/offline")


@dataclass(frozen=True)
class ExtractionSpec:
    """Everything one extraction job does, by level: spec.data.split_type,
    spec.outputs.save_predictions, spec.data.INCLUDE_VAL, ..."""

    data: ExtractionDataSettings
    outputs: ExtractionOutputSettings = ExtractionOutputSettings()
    compute: ExtractionComputeSettings = ExtractionComputeSettings()


EXTRACTION_LEVELS = {
    "data": ExtractionDataSettings,
    "outputs": ExtractionOutputSettings,
    "compute": ExtractionComputeSettings,
}

#: Every setting an extraction job may set. The condition has no default.
EXTRACTION_SETTINGS: Tuple[Setting, ...] = (
    Setting("data", "gnn_model_type", str, required=True),
    Setting("data", "split_type", str, required=True),
    Setting("data", "filter_rmsd_max_value", parse_rmsd, required=True),
    Setting("outputs", "save_representations", parse_bool),
    Setting("outputs", "save_predictions", parse_bool),
    Setting("outputs", "emit_tower_reprs", parse_bool),
    Setting("outputs", "dtype_out", parse_optional_str),
    Setting("outputs", "overwrite", parse_bool),
    Setting("compute", "device", str),
    Setting("compute", "infer_batch_size", parse_optional_int),
    Setting("compute", "eval_num_workers", int),
    Setting("compute", "num_processes", int, env="CPU_COUNT"),
    Setting("compute", "wandb_mode", str),
)


def extraction_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Run a trained GNN over each CV fold's test molecules and save its "
                    "per-layer representations (prob/run_extraction.py). Every flag is a "
                    "setting of prob_config.ExtractionSpec.",
    )
    add_setting_flags(parser, EXTRACTION_SETTINGS)
    return parser


def resolve_extraction_spec(
    argv: Optional[Sequence[str]] = None,
    environ: Optional[Mapping[str, str]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Tuple[ExtractionSpec, Dict[str, str]]:
    """
    The ExtractionSpec for one job: command line (argv) > config > environment
    (CPU_COUNT) > default. Unknown flags and fixed values on the command line are
    errors. Returns (spec, sources) like resolve_experiment_spec.
    """
    argv = list(argv or [])
    reject_fixed_flags(argv, EXTRACTION_LEVELS)
    cli = vars(extraction_arg_parser().parse_args(argv))
    return resolve(
        ExtractionSpec, EXTRACTION_LEVELS, EXTRACTION_SETTINGS,
        cli=cli, config=config or {}, environ=os.environ if environ is None else environ,
    )


# ─────────────────────────────────────────────────────────────
# ProbingExperimentSpec: one level per class
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProbeDataSettings:
    """Which representations are probed. The condition fields have no default
    here: they come from get_ds_load_config, which also turns them into paths.
    How the representations were extracted (e.g. test only or test + val) is
    fixed at extraction (ExtractionSpec) and copied from its manifest into the run
    manifest."""

    gnn_model_type: str
    split_type: str
    filter_rmsd_max_value: Optional[float]
    #: Layers to probe. None = every aggregated layer_*.pt in the condition's dir.
    layers: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        _check_condition(self.gnn_model_type, self.split_type, self.filter_rmsd_max_value)
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
# Resolving the probe spec: command line > config > environment > default
# ─────────────────────────────────────────────────────────────

#: Every per-run setting the command line, config or environment may set.
#: Condition fields (gnn, split, rmsd, target_file) are read from the config only:
#: get_ds_load_config has already applied the command line to them.
RUN_SETTINGS: Tuple[Setting, ...] = (
    Setting("data", "layers", parse_int_tuple, "PROB_LAYERS"),
    Setting("target", "run_shuffled_baseline", parse_bool, "PROB_RUN_SHUFFLED_BASELINE"),
    Setting("target", "baseline_tag", str, "PROB_BASELINE_TAG"),
    Setting("model", "run_linear_models", parse_bool, "PROB_RUN_LINEAR_MODELS"),
    Setting("model", "run_non_linear_models", parse_bool, "PROB_RUN_NON_LINEAR_MODELS"),
    Setting("model", "nonlinear_models", parse_names, "PROB_NONLINEAR_MODELS"),
    Setting("model", "per_ckpt", parse_bool, "PROB_PER_CKPT"),
    Setting("model", "pooled", parse_bool, "PROB_POOLED"),
    Setting("model", "reuse_best_params", parse_bool, "PROB_REUSE_BEST_PARAMS"),
    Setting("model", "best_params_cache_dir", parse_optional_str, "PROB_BEST_PARAMS_CACHE_DIR"),
    Setting("compute", "device", str),
    Setting("compute", "n_jobs", parse_optional_int),
)
CONDITION_FIELDS = (
    ("data", "gnn_model_type"),
    ("data", "split_type"),
    ("data", "filter_rmsd_max_value"),
    ("target", "target_file"),
)
PROB_ENV_VARS = tuple(s.env for s in RUN_SETTINGS if s.env)


def _on_cli(argv: Optional[Sequence[str]], name: str) -> bool:
    return argv is not None and any(a == f"--{name}" or a.startswith(f"--{name}=") for a in argv)


def resolve_experiment_spec(
    config: Mapping[str, Any],
    argv: Optional[Sequence[str]] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Tuple[ProbingExperimentSpec, Dict[str, str]]:
    """
    Build the spec for one probe run and say where each setting came from.

    config: the condition config (get_ds_load_config, or a dict in tests and
    notebooks); may also carry any run setting by name. argv: the command line
    (None = not parsed); flags it does not know are left to get_ds_load_config.
    environ: defaults to os.environ.

    Returns (spec, sources) with sources["<level>.<name>"] = "cli", "config",
    "env <VAR>=<value>" or "default". Unknown PROB_* variables and fixed values
    on the command line raise.
    """
    reject_fixed_flags(argv, _LEVEL_CLASSES)
    cli: Dict[str, Any] = {}
    if argv is not None:
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        add_setting_flags(parser, RUN_SETTINGS)
        cli = vars(parser.parse_known_args(list(argv))[0])
    given = {
        (level, name): (config.get(name), "cli" if _on_cli(argv, name) else "config")
        for level, name in CONDITION_FIELDS
    }
    return resolve(
        ProbingExperimentSpec, _LEVEL_CLASSES, RUN_SETTINGS,
        cli=cli, config=config, environ=os.environ if environ is None else environ,
        env_prefix="PROB_", given=given,
    )
