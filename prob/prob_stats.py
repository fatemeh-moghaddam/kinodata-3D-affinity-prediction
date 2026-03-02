"""
Statistical analysis helpers for probing results.

Use for confidence intervals, significance tests, and comparison across
models or layers. Extend here for bootstrap, permutation, or paired tests.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from prob.prob_metrics import evaluate_predictions


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """
    Bootstrap confidence interval for a single metric (default: R²).

    Returns dict with keys: metric, point_estimate, lower, upper, confidence.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    if metric_fn is None:
        metric_fn = r2_score

    point = float(metric_fn(y_true, y_pred))
    boot_scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_scores.append(metric_fn(y_true[idx], y_pred[idx]))

    boot_scores = np.array(boot_scores)
    alpha = 1 - confidence
    lower = float(np.percentile(boot_scores, 100 * alpha / 2))
    upper = float(np.percentile(boot_scores, 100 * (1 - alpha / 2)))

    name = getattr(metric_fn, "__name__", "metric")
    return {
        "metric": name,
        "point_estimate": point,
        "lower": lower,
        "upper": upper,
        "confidence": confidence,
    }


def run_probe_statistical_tests(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Run bootstrap CIs for R² and RMSE on (y_true, y_pred).
    Returns a dict suitable for inclusion in probe summary JSON, e.g.:
      {"r2_ci": {...}, "rmse_ci": {...}}
    """
    def rmse(y, yp):
        return float(np.sqrt(mean_squared_error(y, yp)))

    r2_ci = bootstrap_ci(
        y_true, y_pred,
        metric_fn=r2_score,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        confidence=confidence,
    )
    rmse_ci = bootstrap_ci(
        y_true, y_pred,
        metric_fn=rmse,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        confidence=confidence,
    )
    return {"r2_ci": r2_ci, "rmse_ci": rmse_ci}


def summarize_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Full metric dict (r2, rmse, mae) for reporting and downstream stats.
    Thin wrapper around evaluate_predictions for a clear stats entry point.
    """
    return evaluate_predictions(y_true, y_pred)
