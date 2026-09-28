"""
Tuning (GridSearchCV) and running (fit + evaluate + stats) for probe models.

- tune_probe: hyperparameter search only; saves cv_results and best_params.
- run_probe: fit a single estimator on train, predict on test, compute metrics
  and statistical tests, save predictions/summary/plots.
- run_cv_search: convenience that does tune then run with the same split and
  writes all artifacts (tuning + evaluation + statistical_tests in summary).
"""
from __future__ import annotations

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
#: Columns: ident, probe_split ("train" / "test"). One fixed assignment for every
#: model, split type, RMSD cutoff and layer.
PROBE_SPLIT_FILENAME = "probe_split.csv"


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
) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, Dict[str, Any]]]:
    """
    Fit a single probe pipeline on train data, predict on test, compute metrics
    and (optionally) bootstrap CIs for R² and RMSE, then save artifacts.

    If X_train, X_test, y_train, y_test are provided, use that split and do not
    split (X, y); pass ids_test to record test idents in the predictions CSV.
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
            X=X_train if X_train is not None else X,
            exp_dirs=exp_dirs,
            model_name=model_name,
            ids_test=ids_test,
        )

    return metrics, y_pred, statistical_tests


def _write_run_artifacts(
    metrics: Dict[str, Any],
    statistical_tests: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    y_pred: np.ndarray,
    X: np.ndarray,
    exp_dirs: Dict[str, Path],
    model_name: str,
    ids_test: Optional[np.ndarray] = None,
) -> None:
    """Write evaluation outputs: predictions CSV, summary JSON (with stats), and figures."""
    artifacts_dir = exp_dirs.get(EXP_DIR_ARTIFACTS)
    reports_dir = exp_dirs.get(EXP_DIR_REPORTS)
    figures_dir = exp_dirs.get(EXP_DIR_FIGURES)

    if artifacts_dir is not None:
        pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
        if ids_test is not None:
            pred_df.insert(0, "ident", ids_test)
        pred_df.to_csv(artifacts_dir / f"{model_name}_predictions.csv", index=False)

    if reports_dir is not None:
        summary = {
            "model": model_name,
            "metrics_on_unseen_data": metrics,
            "n_samples": int(X.shape[0]),
            "n_test_samples": int(len(y_test)),
            "n_features": int(X.shape[1]),
        }
        if statistical_tests:
            summary["statistical_tests"] = statistical_tests
        with open(reports_dir / f"{model_name}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

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
    """
    X_train, X_test, y_train, y_test, _, ids_test = split_by_ident(
        X, y, idents, load_probe_split(probe_split_path, test_size=test_size)
    )

    class _BestParamsOnly:
        """Minimal stand-in so callers can still use `.best_params_`."""

        def __init__(self, best_params: Dict[str, Any]):
            self.best_params_ = best_params

    def _best_params_file(dir_: Optional[Path]) -> Optional[Path]:
        if dir_ is None:
            return None
        return dir_ / f"{model_name}_best_params.json"

    def _normalize_loaded_best_params(loaded: Dict[str, Any]) -> Dict[str, Any]:
        # `json.dump` converts tuples to lists. Convert back for sklearn params
        # that typically expect tuples.
        normalized = dict(loaded)
        for k, v in normalized.items():
            if isinstance(v, list) and k.endswith("hidden_layer_sizes"):
                normalized[k] = tuple(v)
        return normalized

    search: Any
    loaded_best_params: Optional[Dict[str, Any]] = None

    # If enabled, prefer:
    # 1) best_params written for the current exp_dirs (re-run of same experiment)
    # 2) a shared cache directory that is common across layers for the same model/target
    if reuse_best_params and exp_dirs is not None:
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

    if loaded_best_params is not None:
        # Ensure per-exp best params exist too (keeps artifacts consistent).
        if exp_dirs is not None:
            reports_dir = exp_dirs.get(EXP_DIR_REPORTS)
            if reports_dir is not None:
                reports_dir.mkdir(parents=True, exist_ok=True)
                fp = reports_dir / f"{model_name}_best_params.json"
                if not fp.exists():
                    with open(fp, "w") as f:
                        json.dump(loaded_best_params, f, indent=2)

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
    )

    return search, metrics, y_pred
