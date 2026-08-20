"""
Statistical analysis helpers for probing results.

Use for confidence intervals, significance tests, and comparison across
models or layers. Extend here for bootstrap, permutation, or paired tests.
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from prob.prob_metrics import evaluate_predictions


def _rmse(y: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y, y_pred)))


def _pearson(y: np.ndarray, y_pred: np.ndarray) -> float:
    # Undefined if either side is constant (can happen in a bootstrap resample).
    if np.std(y) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y, y_pred)[0, 1])


# Metrics bootstrapped everywhere in this module: short name -> fn(y_true, y_pred).
METRIC_FNS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "r2": r2_score,
    "rmse": _rmse,
    "mae": mean_absolute_error,
    "pearson": _pearson,
}


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None,
    confidence: float = 0.95,
    name: Optional[str] = None,
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

    boot_scores = np.array(boot_scores, dtype=float)
    alpha = 1 - confidence
    lower = float(np.nanpercentile(boot_scores, 100 * alpha / 2))
    upper = float(np.nanpercentile(boot_scores, 100 * (1 - alpha / 2)))

    return {
        "metric": name or getattr(metric_fn, "__name__", "metric"),
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
    Run bootstrap CIs for every metric in METRIC_FNS on (y_true, y_pred).
    Returns a dict suitable for inclusion in probe summary JSON, e.g.:
      {"r2_ci": {...}, "rmse_ci": {...}, "mae_ci": {...}, "pearson_ci": {...}}
    """
    return {
        f"{name}_ci": bootstrap_ci(
            y_true, y_pred,
            metric_fn=fn,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            confidence=confidence,
            name=name,
        )
        for name, fn in METRIC_FNS.items()
    }


def summarize_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Full metric dict (r2, rmse, mae) for reporting and downstream stats.
    Thin wrapper around evaluate_predictions for a clear stats entry point.
    """
    return evaluate_predictions(y_true, y_pred)


def compare_two_conditions(
    condition_a_name: str,
    y_true_a: np.ndarray,
    y_pred_a: np.ndarray,
    condition_b_name: str,
    y_true_b: np.ndarray,
    y_pred_b: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None,
    confidence: float = 0.95,
) -> Dict:
    """
    Compare two paired conditions (e.g., layer_1 vs layer_2, or rmsd_2 vs rmsd_4).
    
    They must share the same y_true values in the same order (paired on test samples).
    
    Returns dict with, for each metric in METRIC_FNS (r2, rmse, mae, pearson):
      - delta_<metric>, delta_<metric>_ci_low, delta_<metric>_ci_high,
        delta_<metric>_bootstrap_pval
    plus median_delta_squared_error and wilcoxon_pval.
    """
    from scipy.stats import wilcoxon
    
    # Normalize to numpy arrays
    y_true_a = np.asarray(y_true_a)
    y_true_b = np.asarray(y_true_b)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    
    # Check pairing: y_true must be identical
    if not np.array_equal(y_true_a, y_true_b):
        raise ValueError(
            f"y_true differs between '{condition_a_name}' and '{condition_b_name}'. "
            "Conditions must be paired on the same test samples."
        )
    
    y_true = y_true_a
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    
    # Bootstrap per-metric deltas with shared resampling
    deltas = {name: np.empty(n_bootstrap, dtype=float) for name in METRIC_FNS}
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        for name, fn in METRIC_FNS.items():
            deltas[name][i] = fn(yt, y_pred_a[idx]) - fn(yt, y_pred_b[idx])

    # Wilcoxon signed-rank test on squared errors
    se_a = np.square(y_pred_a - y_true)
    se_b = np.square(y_pred_b - y_true)
    try:
        w_result = wilcoxon(se_a, se_b, alternative="two-sided")
        wilcoxon_pval = float(w_result.pvalue)
    except ValueError:
        # Occurs if all differences are zero
        wilcoxon_pval = 1.0
    
    alpha = 1 - confidence
    result = {
        "condition_a": condition_a_name,
        "condition_b": condition_b_name,
        "n_samples": int(n),
    }
    for name, fn in METRIC_FNS.items():
        d = deltas[name]
        result[f"delta_{name}"] = float(fn(y_true, y_pred_a) - fn(y_true, y_pred_b))
        result[f"delta_{name}_ci_low"] = float(np.nanpercentile(d, 100 * alpha / 2))
        result[f"delta_{name}_ci_high"] = float(np.nanpercentile(d, 100 * (1 - alpha / 2)))
        # Two-sided bootstrap p-value
        result[f"delta_{name}_bootstrap_pval"] = float(
            min(1.0, 2.0 * min(np.nanmean(d <= 0), np.nanmean(d >= 0)))
        )

    result["median_delta_squared_error"] = float(np.median(se_a - se_b))
    result["wilcoxon_pval"] = wilcoxon_pval
    return result


def _holm_bonferroni_correction(pvalues: np.ndarray) -> np.ndarray:
    """
    Apply Holm-Bonferroni correction to p-values.
    
    Returns adjusted p-values (monotonically non-decreasing).
    """
    pvalues = np.asarray(pvalues, dtype=float)
    if len(pvalues) == 0:
        return pvalues
    
    # Sort p-values and track original indices
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    m = len(pvalues)
    running_max = 0.0
    
    # Apply Holm correction: multiply by (m - rank)
    for rank, idx in enumerate(order):
        holm_p = min(1.0, (m - rank) * pvalues[idx])
        running_max = max(running_max, holm_p)
        adjusted[idx] = running_max
    
    return adjusted


def compare_multiple_conditions(
    conditions_dict: Dict[str, tuple],
    *,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Compare all pairs of conditions with Holm-Bonferroni correction.
    
    Args:
        conditions_dict: Dict mapping condition name -> (y_true, y_pred)
                        All conditions must share identical y_true values.
        n_bootstrap, random_state, confidence: Passed to compare_two_conditions
    
    Returns:
        DataFrame with one row per pairwise comparison, including:
        - condition_a, condition_b
        - delta_<metric>{,_ci_low,_ci_high,_bootstrap_pval} for each metric in
          METRIC_FNS (r2, rmse, mae, pearson)
        - median_delta_squared_error, wilcoxon_pval
        - a *_holm column per p-value column (Holm-Bonferroni corrected)
    """
    condition_names = sorted(conditions_dict.keys())
    if len(condition_names) < 2:
        raise ValueError("Need at least 2 conditions to compare.")
    
    # Validate that all have identical y_true
    y_true_ref = None
    for cond_name in condition_names:
        y_true, y_pred = conditions_dict[cond_name]
        y_true = np.asarray(y_true)
        if y_true_ref is None:
            y_true_ref = y_true
        elif not np.array_equal(y_true_ref, y_true):
            raise ValueError(
                f"y_true mismatch: condition '{cond_name}' has different y_true. "
                "All conditions must be paired on the same test samples."
            )
    
    # Run all pairwise comparisons
    results = []
    for cond_a, cond_b in combinations(condition_names, 2):
        y_true_a, y_pred_a = conditions_dict[cond_a]
        y_true_b, y_pred_b = conditions_dict[cond_b]
        
        result = compare_two_conditions(
            cond_a, y_true_a, y_pred_a,
            cond_b, y_true_b, y_pred_b,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            confidence=confidence,
        )
        results.append(result)
    
    df = pd.DataFrame(results)
    
    # Apply Holm-Bonferroni correction to all p-value columns
    pval_cols = [f"delta_{name}_bootstrap_pval" for name in METRIC_FNS] + ["wilcoxon_pval"]
    for pval_col in pval_cols:
        corrected_col = f"{pval_col}_holm"
        df[corrected_col] = _holm_bonferroni_correction(df[pval_col].to_numpy())
    
    return df
