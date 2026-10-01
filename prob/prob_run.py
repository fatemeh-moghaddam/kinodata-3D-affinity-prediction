"""
Tuning (GridSearchCV) and running (fit + evaluate + stats) for probe models.

- tune_probe: hyperparameter search only; saves cv_results and best_params.
- run_probe: fit a single estimator on train, predict on test, compute metrics
  and statistical tests, save predictions/summary/plots.
- run_cv_search: convenience that does tune then run with the same split and
  writes all artifacts (tuning + evaluation + statistical_tests in summary).
- run_probe_per_checkpoint: what the pipeline uses. One probe per GNN
  checkpoint (CV fold), since each fold's representations come from a
  different, separately trained model; results of all folds are written as one
  experiment.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prob.paths_and_io import (
    EXP_DIR_ARTIFACTS,
    EXP_DIR_FIGURES,
    EXP_DIR_REPORTS,
    get_data_dir,
)
from prob.prob_metrics import evaluate_predictions, regression_scorers
from prob.prob_plots import plot_parity, plot_residuals
from prob.prob_stats import run_probe_statistical_tests


# Defaults
DEFAULT_TEST_SIZE = 0.1
DEFAULT_N_SPLITS_CV = 5
DEFAULT_REFIT = "r2"
DEFAULT_PROBE_SPLIT_SEED = 0
#: split_by_ident raises if a condition's test fraction is further than this
#: from the file's overall test fraction (absolute, e.g. 0.01 -> 9-11% for 10%).
MAX_TEST_FRACTION_DEVIATION = 0.01
#: The same check within one checkpoint's ~8k rows, where the fraction varies more
#: (sd ~0.3 points at RMSD <= 2).
MAX_TEST_FRACTION_DEVIATION_PER_FOLD = 0.015
#: Columns: ident, probe_split ("train" / "test"). One fixed assignment for every
#: model, split type, RMSD cutoff and layer.
PROBE_SPLIT_FILENAME = "probe_split.csv"
#: Metrics summarised across checkpoints (mean, sd) in per-checkpoint runs.
CHECKPOINT_METRICS = ("r2", "rmse", "mae", "pearson")


# ─────────────────────────────────────────────────────────────
# Fixed probe train/test assignment by ident
# ─────────────────────────────────────────────────────────────


def default_probe_split_path() -> Path:
    return get_data_dir(prob=True) / PROBE_SPLIT_FILENAME


def catalogue_idents(catalogue_path: Optional[Path] = None) -> np.ndarray:
    """
    Every ident in the unfiltered KinodataDocked dataset (~119.5k), read from
    data/ident_to_activity_id.csv. Each RMSD cutoff's dataset is a subset of it,
    so an assignment built from this covers every condition.
    """
    path = (
        Path(catalogue_path)
        if catalogue_path is not None
        else get_data_dir(prob=False) / "ident_to_activity_id.csv"
    )
    return pd.read_csv(path, usecols=["ident_processed"])["ident_processed"].to_numpy()


def make_probe_split(
    all_idents: np.ndarray,
    path: Optional[Path] = None,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_PROBE_SPLIT_SEED,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Pick test_size of `all_idents` at random as the probe test set and save the
    assignment. Build it from the full (least filtered) set of idents: an ident
    that is not in the file cannot be split later (split_by_ident raises).

    Refuses to overwrite an existing file unless overwrite=True, since changing
    the assignment makes earlier runs incomparable with new ones.
    """
    path = Path(path) if path is not None else default_probe_split_path()
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")

    idents = np.unique(np.asarray(all_idents).astype(int))
    rng = np.random.default_rng(random_state)
    n_test = int(round(test_size * len(idents)))
    test_idents = rng.choice(idents, size=n_test, replace=False)

    split_df = pd.DataFrame({
        "ident": idents,
        "probe_split": np.where(np.isin(idents, test_idents), "test", "train"),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(path, index=False)
    return split_df


def probe_split_provenance(path: Optional[Path] = None) -> Dict[str, str]:
    """Path and sha256 of the probe split file, recorded in every run summary so a
    result can be traced to the exact train/test assignment it used."""
    path = Path(path) if path is not None else default_probe_split_path()
    return {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_probe_split(
    path: Optional[Path] = None,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
) -> pd.DataFrame:
    """
    Load the saved assignment. On first use (no file yet) it is built from the
    full dataset catalogue (`catalogue_idents`) and saved, so all later runs reuse
    the same one. test_size is only used for that first build.
    """
    path = Path(path) if path is not None else default_probe_split_path()
    if not path.exists():
        make_probe_split(catalogue_idents(), path, test_size=test_size)
    return pd.read_csv(path)


def split_by_ident(
    X: np.ndarray,
    y: np.ndarray,
    idents: np.ndarray,
    probe_split: pd.DataFrame,
    max_fraction_deviation: float = MAX_TEST_FRACTION_DEVIATION,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split (X, y) by looking up each row's ident in `probe_split`. `idents` must
    be aligned row-for-row with X and y (the run's ids.pt, masked like X and y).
    Each side is sorted by ident, so CV folds inside tuning and the saved
    predictions do not depend on the row order of the run either.

    The test labels are random over the whole dataset, so any large subset gets
    close to the same test fraction (RMSD <= 2: 9.97% for a 10% file). A subset
    that lands further than `max_fraction_deviation` away -- small, or selected
    in a way that correlates with the labels -- raises instead of silently
    probing on a skewed split.

    Returns (X_train, X_test, y_train, y_test, ids_train, ids_test).
    """
    idents = np.asarray(idents).astype(int)
    if not (len(idents) == len(X) == len(y)):
        raise ValueError(
            f"idents ({len(idents)}), X ({len(X)}) and y ({len(y)}) must have the same length"
        )

    unknown = ~np.isin(idents, probe_split["ident"].to_numpy())
    if unknown.any():
        raise ValueError(
            f"{unknown.sum()} idents are not in the probe split file "
            f"(e.g. {idents[unknown][:5].tolist()}); rebuild it with make_probe_split "
            "from the full set of idents"
        )

    test_idents = probe_split.loc[probe_split["probe_split"] == "test", "ident"].to_numpy()
    is_test = np.isin(idents, test_idents)

    expected_fraction = len(test_idents) / len(probe_split)
    test_fraction = is_test.mean()
    if abs(test_fraction - expected_fraction) > max_fraction_deviation:
        raise ValueError(
            f"Probe test fraction is {test_fraction:.3f} ({is_test.sum()}/{len(idents)}), "
            f"expected {expected_fraction:.3f} +/- {max_fraction_deviation}. The subset is "
            "too small or not independent of the probe split labels."
        )

    train_idx = np.flatnonzero(~is_test)
    test_idx = np.flatnonzero(is_test)
    train_idx = train_idx[np.argsort(idents[train_idx], kind="stable")]
    test_idx = test_idx[np.argsort(idents[test_idx], kind="stable")]
    return (
        X[train_idx], X[test_idx],
        y[train_idx], y[test_idx],
        idents[train_idx], idents[test_idx],
    )


def _make_pipeline(estimator: Any) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", estimator),
    ])


def _json_default(value: Any) -> Any:
    """json.dump fallback for numpy scalars/arrays in params and grids."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value)}")


def _as_json(value: Any) -> Any:
    """`value` as it reads back from JSON (tuples -> lists, numpy -> python)."""
    return json.loads(json.dumps(value, default=_json_default))


def _normalize_loaded_best_params(loaded: Dict[str, Any]) -> Dict[str, Any]:
    # `json.dump` converts tuples to lists. Convert back for sklearn params
    # that typically expect tuples.
    normalized = dict(loaded)
    for k, v in normalized.items():
        if isinstance(v, list) and k.endswith("hidden_layer_sizes"):
            normalized[k] = tuple(v)
    return normalized


# ─────────────────────────────────────────────────────────────
# Tuning (hyperparameter search only)
# ─────────────────────────────────────────────────────────────


def tune_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    prob_model: Any,
    param_grid: Dict[str, Any],
    *,
    n_splits: int = DEFAULT_N_SPLITS_CV,
    random_state: int,
    n_jobs: int = -1,
    exp_dirs: Optional[Dict[str, Path]] = None,
    model_name: str = "model",
    refit: str = DEFAULT_REFIT,
) -> GridSearchCV:
    """
    Run GridSearchCV on (X_train, y_train) only. Saves cv_results and best_params;
    does not evaluate on a holdout or write predictions/summary/plots.

    Returns the fitted GridSearchCV (use .best_estimator_, .best_params_).
    """
    pipe = _make_pipeline(prob_model)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=regression_scorers(),
        refit=refit,
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)

    if exp_dirs is not None:
        _write_tuning_artifacts(search, exp_dirs, model_name)

    return search


def _write_tuning_artifacts(
    search: GridSearchCV,
    exp_dirs: Dict[str, Path],
    model_name: str,
) -> None:
    """Write only tuning outputs: cv_results.csv and best_params.json."""
    artifacts_dir = exp_dirs.get(EXP_DIR_ARTIFACTS)
    reports_dir = exp_dirs.get(EXP_DIR_REPORTS)

    if artifacts_dir is not None:
        cv_df = pd.DataFrame(search.cv_results_)
        cv_df.to_csv(artifacts_dir / f"{model_name}_cv_results.csv", index=False)

    if reports_dir is not None:
        with open(reports_dir / f"{model_name}_best_params.json", "w") as f:
            json.dump(search.best_params_, f, indent=2)


# ─────────────────────────────────────────────────────────────
# Run probe (fit + evaluate + statistical tests)
# ─────────────────────────────────────────────────────────────


def run_probe(
    X: np.ndarray,
    y: np.ndarray,
    estimator: Any,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int,
    exp_dirs: Optional[Dict[str, Path]] = None,
    model_name: str = "model",
    run_stats: bool = True,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    X_train: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    idents: Optional[np.ndarray] = None,
    ids_test: Optional[np.ndarray] = None,
    probe_split_info: Optional[Dict[str, str]] = None,
    extra_summary: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, Dict[str, Any]]]:
    """
    Fit a single probe pipeline on train data, predict on test, compute metrics
    and (optionally) bootstrap CIs for every metric, then save artifacts.

    If X_train, X_test, y_train, y_test are provided, use that split and do not
    split (X, y); pass ids_test to record test idents in the predictions CSV and
    probe_split_info (`probe_split_provenance`) to record the split in the summary.
    Otherwise split (X, y) by `idents` with the saved probe split
    (`load_probe_split`); test_size only matters if that file does not exist yet.

    Returns (metrics, y_pred, statistical_tests).
    """
    if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
        pass  # use provided split
    else:
        if idents is None:
            raise ValueError("idents are required to split (X, y) into probe train/test")
        X_train, X_test, y_train, y_test, _, ids_test = split_by_ident(
            X, y, idents, load_probe_split(test_size=test_size)
        )
        probe_split_info = probe_split_provenance()

    # Build pipeline: if estimator is already a pipeline with "model" step, use it; else wrap
    if hasattr(estimator, "steps") and isinstance(estimator, Pipeline):
        pipe = clone(estimator)
    else:
        pipe = _make_pipeline(clone(estimator))

    start = time.time()
    pipe.fit(X_train, y_train)
    fit_seconds = time.time() - start

    y_pred = pipe.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)
    metrics["fit_seconds"] = fit_seconds

    statistical_tests: Dict[str, Dict[str, Any]] = {}
    if run_stats:
        statistical_tests = run_probe_statistical_tests(
            y_test, y_pred,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )

    if exp_dirs is not None:
        _write_run_artifacts(
            metrics=metrics,
            statistical_tests=statistical_tests,
            y_test=y_test,
            y_pred=y_pred,
            n_train=int(X_train.shape[0]),
            n_features=int(X_train.shape[1]),
            exp_dirs=exp_dirs,
            model_name=model_name,
            ids_test=ids_test,
            bootstrap_settings=(
                {"n_bootstrap": n_bootstrap, "confidence": confidence, "random_state": random_state}
                if run_stats else None
            ),
            probe_split_info=probe_split_info,
            extra_summary=extra_summary,
        )

    return metrics, y_pred, statistical_tests


def _write_run_artifacts(
    metrics: Dict[str, Any],
    statistical_tests: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    y_pred: np.ndarray,
    n_train: int,
    n_features: int,
    exp_dirs: Dict[str, Path],
    model_name: str,
    ids_test: Optional[np.ndarray] = None,
    bootstrap_settings: Optional[Dict[str, Any]] = None,
    probe_split_info: Optional[Dict[str, str]] = None,
    folds_test: Optional[np.ndarray] = None,
    extra_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write evaluation outputs: predictions CSV, summary JSON (with stats), and figures.

    folds_test adds a fold column to the predictions CSV; extra_summary is merged
    into the summary JSON (per-checkpoint runs put their per_fold results there).
    """
    artifacts_dir = exp_dirs.get(EXP_DIR_ARTIFACTS)
    reports_dir = exp_dirs.get(EXP_DIR_REPORTS)
    figures_dir = exp_dirs.get(EXP_DIR_FIGURES)

    if artifacts_dir is not None:
        pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
        if folds_test is not None:
            pred_df.insert(0, "fold", folds_test)
        if ids_test is not None:
            pred_df.insert(0, "ident", ids_test)
        pred_df.to_csv(artifacts_dir / f"{model_name}_predictions.csv", index=False)

    if reports_dir is not None:
        summary = {
            "model": model_name,
            "metrics_on_unseen_data": metrics,
            "n_samples": int(n_train + len(y_test)),
            "n_train_samples": int(n_train),
            "n_test_samples": int(len(y_test)),
            "n_features": int(n_features),
        }
        if statistical_tests:
            summary["statistical_tests"] = statistical_tests
            if bootstrap_settings is not None:
                summary["bootstrap"] = bootstrap_settings
        if probe_split_info is not None:
            summary["probe_split"] = probe_split_info
        if extra_summary:
            summary.update(extra_summary)
        with open(reports_dir / f"{model_name}_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=_json_default)

    if figures_dir is not None:
        plot_parity(
            y_test, y_pred,
            title=f"{model_name} Parity (R2={metrics['r2']:.3f})",
            save_path=figures_dir / f"{model_name}_parity.png",
            show=False,
        )
        plot_residuals(
            y_test, y_pred,
            title=f"{model_name} Residuals (RMSE={metrics['rmse']:.3f})",
            save_path=figures_dir / f"{model_name}_residuals.png",
            show=False,
        )


# ─────────────────────────────────────────────────────────────
# Convenience: tune + run with same split (backward compatible)
# ─────────────────────────────────────────────────────────────


def run_cv_search(
    X: np.ndarray,
    y: np.ndarray,
    prob_model: Any,
    param_grid: Dict[str, Any],
    *,
    idents: np.ndarray,
    n_splits: int = DEFAULT_N_SPLITS_CV,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int,
    n_jobs: int = -1,
    exp_dirs: Optional[Dict[str, Path]] = None,
    model_name: str = "model",
    refit: str = DEFAULT_REFIT,
    run_stats: bool = True,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    reuse_best_params: bool = False,
    best_params_cache_dir: Optional[Path] = None,
    probe_split_path: Optional[Path] = None,
    fixed_best_params: Optional[Dict[str, Any]] = None,
    fixed_best_params_source: Optional[str] = None,
    extra_summary: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any], np.ndarray]:
    """
    Tune hyperparameters on a train split, then run the best estimator on the
    same split's test set (refit best pipeline on train, predict on test),
    compute metrics and statistical tests, and write all artifacts.

    The probe train/test split looks up each row's ident (`idents`, aligned
    row-for-row with X and y) in the saved probe split file, so every model,
    split type, RMSD cutoff and layer is tested on the same idents. test_size
    only matters if that file does not exist yet.

    Saves: tuning (cv_results, best_params) and run (predictions, summary with
    statistical_tests, figures). Returns (search, metrics, y_pred).

    If reuse_best_params is True, this will skip GridSearchCV when cached
    best params exist (writing only run artifacts). This is intended for
    running many similar experiments without re-tuning.

    This is the pooled probe: one probe over all CV folds' rows, although each
    fold's representations come from a different checkpoint (see
    run_probe_per_checkpoint). fixed_best_params skips tuning (the shuffled-ident
    baseline passes the real target's params); fixed_best_params_source is
    recorded in the summary. extra_summary (e.g. the run_id of the run manifest)
    is merged into the summary JSON.
    """
    X_train, X_test, y_train, y_test, _, ids_test = split_by_ident(
        X, y, idents, load_probe_split(probe_split_path, test_size=test_size)
    )
    probe_split_info = probe_split_provenance(probe_split_path)

    class _BestParamsOnly:
        """Minimal stand-in so callers can still use `.best_params_`."""

        def __init__(self, best_params: Dict[str, Any]):
            self.best_params_ = best_params

    def _best_params_file(dir_: Optional[Path]) -> Optional[Path]:
        if dir_ is None:
            return None
        return dir_ / f"{model_name}_best_params.json"

    search: Any
    loaded_best_params: Optional[Dict[str, Any]] = None
    params_source = "tuned"
    if fixed_best_params is not None:
        loaded_best_params = _normalize_loaded_best_params(fixed_best_params)
        params_source = "fixed"

    # If enabled, prefer:
    # 1) best_params written for the current exp_dirs (re-run of same experiment)
    # 2) a shared cache directory that is common across layers for the same model/target
    if loaded_best_params is None and reuse_best_params and exp_dirs is not None:
        params_source = "saved"
        current_best_params_fp = _best_params_file(exp_dirs.get(EXP_DIR_REPORTS))
        if current_best_params_fp is not None and current_best_params_fp.exists():
            with open(current_best_params_fp, "r") as f:
                loaded_best_params = _normalize_loaded_best_params(json.load(f))
        else:
            cache_dir = best_params_cache_dir
            if cache_dir is None:
                # exp_dirs["root"] ends with ".../<prob_model>/<layer_num>", so sharing
                # is achieved by writing alongside ".../<prob_model>/".
                cache_dir = exp_dirs["root"].parent / "shared_best_params"
            cache_fp = _best_params_file(cache_dir)
            if cache_fp is not None and cache_fp.exists():
                with open(cache_fp, "r") as f:
                    loaded_best_params = _normalize_loaded_best_params(json.load(f))
                params_source = "shared"

    summary_fields: Dict[str, Any] = {
        "probe_mode": "pooled", "params_source": params_source, **(extra_summary or {}),
    }
    if fixed_best_params_source is not None:
        summary_fields["best_params_from"] = fixed_best_params_source

    if loaded_best_params is not None:
        # Ensure per-exp best params exist too (keeps artifacts consistent).
        if exp_dirs is not None:
            reports_dir = exp_dirs.get(EXP_DIR_REPORTS)
            if reports_dir is not None:
                reports_dir.mkdir(parents=True, exist_ok=True)
                fp = reports_dir / f"{model_name}_best_params.json"
                if not fp.exists():
                    with open(fp, "w") as f:
                        json.dump(loaded_best_params, f, indent=2, default=_json_default)

        pipe = _make_pipeline(prob_model)
        pipe.set_params(**loaded_best_params)
        metrics, y_pred, _ = run_probe(
            X_train,
            y_train,  # unused when split provided
            pipe,
            test_size=test_size,
            random_state=random_state,
            exp_dirs=exp_dirs,
            model_name=model_name,
            run_stats=run_stats,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            ids_test=ids_test,
            probe_split_info=probe_split_info,
            extra_summary=summary_fields,
        )
        search = _BestParamsOnly(loaded_best_params)
        return search, metrics, y_pred

    search = tune_probe(
        X_train, y_train,
        prob_model,
        param_grid,
        n_splits=n_splits,
        random_state=random_state,
        n_jobs=n_jobs,
        exp_dirs=exp_dirs,
        model_name=model_name,
        refit=refit,
    )

    if reuse_best_params and exp_dirs is not None:
        cache_dir = best_params_cache_dir
        if cache_dir is None:
            cache_dir = exp_dirs["root"].parent / "shared_best_params"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_fp = cache_dir / f"{model_name}_best_params.json"
        # Avoid overwriting if already present (keeps first-tuned values stable).
        if not cache_fp.exists():
            with open(cache_fp, "w") as f:
                json.dump(search.best_params_, f, indent=2)

    # Refit best pipeline on the same train set, then evaluate on test
    metrics, y_pred, _ = run_probe(
        X_train, y_train,  # unused when split provided
        search.best_estimator_,
        test_size=test_size,
        random_state=random_state,
        exp_dirs=exp_dirs,
        model_name=model_name,
        run_stats=run_stats,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        ids_test=ids_test,
        probe_split_info=probe_split_info,
        extra_summary={**summary_fields, "params_source": "tuned", "n_splits_cv": n_splits},
    )

    return search, metrics, y_pred


# ─────────────────────────────────────────────────────────────
# Per-checkpoint probing (one probe per CV fold's GNN checkpoint)
# ─────────────────────────────────────────────────────────────


def _read_best_params_by_fold(path: Optional[Path], grid: Any) -> Dict[int, Dict[str, Any]]:
    """
    Per-fold best params saved at `path` ({"param_grid": ..., "folds": {"0": {...}}}),
    or {} if the file is missing, in the old single-params format, or was tuned
    over a different param_grid (so a changed grid always retunes).
    """
    if path is None or not path.exists():
        return {}
    with open(path) as f:
        saved = json.load(f)
    if not isinstance(saved, dict) or saved.get("param_grid") != grid or "folds" not in saved:
        return {}
    return {int(k): _normalize_loaded_best_params(v) for k, v in saved["folds"].items()}


def load_best_params_by_fold(path: Path, param_grid: Optional[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Per-fold best params a run saved at `path`, if they were tuned over
    `param_grid` ({} otherwise). Used to give a control run the same params."""
    return _read_best_params_by_fold(Path(path), _as_json(param_grid or {}))


def _write_best_params_by_fold(path: Path, grid: Any, best_params: Dict[int, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"param_grid": grid, "folds": {str(k): v for k, v in sorted(best_params.items())}}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def run_probe_per_checkpoint(
    X: np.ndarray,
    y: np.ndarray,
    prob_model: Any,
    param_grid: Optional[Dict[str, Any]],
    *,
    idents: np.ndarray,
    folds: np.ndarray,
    n_splits: int = DEFAULT_N_SPLITS_CV,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int,
    n_jobs: int = -1,
    exp_dirs: Optional[Dict[str, Path]] = None,
    model_name: str = "model",
    refit: str = DEFAULT_REFIT,
    run_stats: bool = True,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    share_best_params_across_layers: bool = False,
    best_params_cache_dir: Optional[Path] = None,
    probe_split_path: Optional[Path] = None,
    fixed_best_params: Optional[Dict[int, Dict[str, Any]]] = None,
    fixed_best_params_source: Optional[str] = None,
    extra_summary: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any], pd.DataFrame]:
    """
    Train one probe per GNN checkpoint and evaluate them as one experiment.

    Each CV fold's representations come from that fold's own checkpoint, and the
    checkpoints' embedding spaces are not aligned, so a single probe over all
    folds would have to decode five coordinate systems at once. Here every fold
    (`folds`, aligned row-for-row with X, y and idents) gets its own probe:
    split by ident with the global probe split, tuned with GridSearchCV on its
    train rows (or fit as-is if param_grid is empty), and evaluated on its test
    rows. The test rows of all folds together are the same idents for every
    model, split type and layer at a cutoff, so the pooled metrics and bootstrap
    CIs stay paired across conditions.

    Best params are saved per fold in reports/<model>_best_params.json. A rerun
    of the same experiment reuses them when they were tuned over the same
    param_grid. With share_best_params_across_layers, a fold's params are also
    shared with the other layers of the same probe and target (first layer
    tuned wins), trading per-layer tuning for time.

    fixed_best_params ({fold: params}, e.g. from load_best_params_by_fold) skips
    tuning altogether: every fold is fit with its given params. The shuffled-ident
    baseline uses this to run with the real target's params, so the control gets
    the same probe as the real task and costs one fit per fold.
    fixed_best_params_source (e.g. the file they came from) is recorded in the
    summary. Each fold's params_source (fixed / saved / shared / tuned / default)
    is recorded in per_fold. extra_summary (e.g. the run_id of the run manifest)
    is merged into the summary JSON.

    Writes: predictions CSV (ident, fold, y_true, y_pred), cv_results with a fold
    column, best params, figures, and a summary JSON whose metrics_on_unseen_data
    / statistical_tests are over all test rows, plus per_fold results and
    across_checkpoints (mean, sd over folds).

    Returns (best_params_by_fold, metrics, predictions DataFrame).
    """
    idents = np.asarray(idents).astype(int)
    folds = np.asarray(folds).astype(int)
    if not (len(folds) == len(idents) == len(X) == len(y)):
        raise ValueError(
            f"folds ({len(folds)}), idents ({len(idents)}), X ({len(X)}) and y ({len(y)}) "
            "must have the same length"
        )
    probe_split = load_probe_split(probe_split_path, test_size=test_size)
    probe_split_info = probe_split_provenance(probe_split_path)
    grid = _as_json(param_grid or {})

    reports_dir = exp_dirs.get(EXP_DIR_REPORTS) if exp_dirs is not None else None
    own_fp = reports_dir / f"{model_name}_best_params.json" if reports_dir is not None else None
    shared_fp = None
    if share_best_params_across_layers and exp_dirs is not None:
        # exp_dirs["root"] is .../<target>/<probe>/<layer>, so this is shared by all layers.
        cache_dir = best_params_cache_dir or exp_dirs["root"].parent / "shared_best_params"
        shared_fp = Path(cache_dir) / f"{model_name}_best_params.json"
    saved = _read_best_params_by_fold(own_fp, grid)
    shared = _read_best_params_by_fold(shared_fp, grid)
    if fixed_best_params is not None:
        missing = sorted(set(np.unique(folds).tolist()) - set(fixed_best_params))
        if missing:
            raise ValueError(
                f"fixed_best_params ({fixed_best_params_source}) has no params for fold(s) {missing}"
            )

    best_params: Dict[int, Dict[str, Any]] = {}
    per_fold: list = []
    pieces: list = []
    cv_frames: list = []
    n_train_total = 0
    start = time.time()

    for fold in np.unique(folds):
        rows = folds == fold
        X_train, X_test, y_train, y_test, _, ids_test = split_by_ident(
            X[rows], y[rows], idents[rows], probe_split,
            max_fraction_deviation=MAX_TEST_FRACTION_DEVIATION_PER_FOLD,
        )
        if fixed_best_params is not None:
            params, source = fixed_best_params[int(fold)], "fixed"
        elif fold in saved:
            params, source = saved[fold], "saved"
        elif fold in shared:
            params, source = shared[fold], "shared"
        else:
            params, source = None, "tuned" if param_grid else "default"
        if params is not None:
            pipe = _make_pipeline(clone(prob_model)).set_params(**params)
            pipe.fit(X_train, y_train)
        elif param_grid:
            search = tune_probe(
                X_train, y_train, prob_model, param_grid,
                n_splits=n_splits, random_state=random_state, n_jobs=n_jobs,
                refit=refit,
            )
            pipe, params = search.best_estimator_, search.best_params_
            cv_frames.append(pd.DataFrame(search.cv_results_).assign(fold=fold))
        else:
            pipe = _make_pipeline(clone(prob_model)).fit(X_train, y_train)
            params = {}

        y_pred = pipe.predict(X_test)
        best_params[int(fold)] = params
        per_fold.append({
            "fold": int(fold),
            "n_train_samples": int(len(y_train)),
            "n_test_samples": int(len(y_test)),
            **evaluate_predictions(y_test, y_pred),
            "best_params": params,
            "params_source": source,
        })
        pieces.append(pd.DataFrame({"ident": ids_test, "fold": int(fold), "y_true": y_test, "y_pred": y_pred}))
        n_train_total += len(y_train)

    pred_df = pd.concat(pieces, ignore_index=True).sort_values("ident", kind="stable").reset_index(drop=True)
    y_test_all = pred_df["y_true"].to_numpy()
    y_pred_all = pred_df["y_pred"].to_numpy()
    metrics = evaluate_predictions(y_test_all, y_pred_all)
    metrics["fit_seconds"] = time.time() - start

    statistical_tests: Dict[str, Dict[str, Any]] = {}
    if run_stats:
        statistical_tests = run_probe_statistical_tests(
            y_test_all, y_pred_all,
            confidence=confidence, n_bootstrap=n_bootstrap, random_state=random_state,
        )

    across_checkpoints = {
        name: {
            "mean": float(np.mean([f[name] for f in per_fold])),
            "sd": float(np.std([f[name] for f in per_fold], ddof=1)) if len(per_fold) > 1 else float("nan"),
        }
        for name in CHECKPOINT_METRICS
    }

    if exp_dirs is not None:
        if own_fp is not None:
            _write_best_params_by_fold(own_fp, grid, best_params)
        if shared_fp is not None:
            # First layer tuned wins: only fill folds the shared file does not have yet.
            _write_best_params_by_fold(shared_fp, grid, {**best_params, **shared})
        artifacts_dir = exp_dirs.get(EXP_DIR_ARTIFACTS)
        if cv_frames and artifacts_dir is not None:
            pd.concat(cv_frames, ignore_index=True).to_csv(
                artifacts_dir / f"{model_name}_cv_results.csv", index=False
            )
        _write_run_artifacts(
            metrics=metrics,
            statistical_tests=statistical_tests,
            y_test=y_test_all,
            y_pred=y_pred_all,
            n_train=n_train_total,
            n_features=int(X.shape[1]),
            exp_dirs=exp_dirs,
            model_name=model_name,
            ids_test=pred_df["ident"].to_numpy(),
            folds_test=pred_df["fold"].to_numpy(),
            bootstrap_settings=(
                {"n_bootstrap": n_bootstrap, "confidence": confidence, "random_state": random_state}
                if run_stats else None
            ),
            probe_split_info=probe_split_info,
            extra_summary={
                "probe_mode": "per_checkpoint",
                "n_splits_cv": n_splits,
                **({"best_params_from": fixed_best_params_source} if fixed_best_params is not None else {}),
                "across_checkpoints": across_checkpoints,
                "per_fold": per_fold,
                **(extra_summary or {}),
            },
        )

    return best_params, {**metrics, "across_checkpoints": across_checkpoints}, pred_df
