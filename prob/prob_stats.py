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
    
    Returns dict with:
      - delta_r2, delta_r2_ci_low, delta_r2_ci_high, delta_r2_bootstrap_pval
      - delta_rmse, delta_rmse_ci_low, delta_rmse_ci_high, delta_rmse_bootstrap_pval
      - median_delta_squared_error, wilcoxon_pval
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
    
    # Bootstrap R² deltas with shared resampling
    r2_deltas = np.empty(n_bootstrap, dtype=float)
    rmse_deltas = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        r2_a = float(r2_score(y_true[idx], y_pred_a[idx]))
        r2_b = float(r2_score(y_true[idx], y_pred_b[idx]))
        r2_deltas[i] = r2_a - r2_b
        
        rmse_a = float(np.sqrt(mean_squared_error(y_true[idx], y_pred_a[idx])))
        rmse_b = float(np.sqrt(mean_squared_error(y_true[idx], y_pred_b[idx])))
        rmse_deltas[i] = rmse_a - rmse_b
    
    # Wilcoxon signed-rank test on squared errors
    se_a = np.square(y_pred_a - y_true)
    se_b = np.square(y_pred_b - y_true)
    try:
        w_result = wilcoxon(se_a, se_b, alternative="two-sided")
        wilcoxon_pval = float(w_result.pvalue)
    except ValueError:
        # Occurs if all differences are zero
        wilcoxon_pval = 1.0
    
    # Point estimates
    alpha = 1 - confidence
    r2_point = float(r2_score(y_true, y_pred_a) - r2_score(y_true, y_pred_b))
    rmse_point = float(np.sqrt(mean_squared_error(y_true, y_pred_a)) - np.sqrt(mean_squared_error(y_true, y_pred_b)))
    
    # CIs from bootstrap
    r2_ci_low = float(np.percentile(r2_deltas, 100 * alpha / 2))
    r2_ci_high = float(np.percentile(r2_deltas, 100 * (1 - alpha / 2)))
    rmse_ci_low = float(np.percentile(rmse_deltas, 100 * alpha / 2))
    rmse_ci_high = float(np.percentile(rmse_deltas, 100 * (1 - alpha / 2)))
    
    # Two-sided bootstrap p-values
    r2_pval = float(min(1.0, 2.0 * min(np.mean(r2_deltas <= 0), np.mean(r2_deltas >= 0))))
    rmse_pval = float(min(1.0, 2.0 * min(np.mean(rmse_deltas <= 0), np.mean(rmse_deltas >= 0))))
    
    # Median delta squared error
    median_delta_se = float(np.median(se_a - se_b))
    
    return {
        "condition_a": condition_a_name,
        "condition_b": condition_b_name,
        "n_samples": int(n),
        "delta_r2": r2_point,
        "delta_r2_ci_low": r2_ci_low,
        "delta_r2_ci_high": r2_ci_high,
        "delta_r2_bootstrap_pval": r2_pval,
        "delta_rmse": rmse_point,
        "delta_rmse_ci_low": rmse_ci_low,
        "delta_rmse_ci_high": rmse_ci_high,
        "delta_rmse_bootstrap_pval": rmse_pval,
        "median_delta_squared_error": median_delta_se,
        "wilcoxon_pval": wilcoxon_pval,
    }


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
        - delta_r2, delta_r2_ci_low, delta_r2_ci_high, delta_r2_bootstrap_pval
        - delta_rmse, delta_rmse_ci_low, delta_rmse_ci_high, delta_rmse_bootstrap_pval
        - median_delta_squared_error, wilcoxon_pval
        - delta_r2_bootstrap_pval_holm, delta_rmse_bootstrap_pval_holm, wilcoxon_pval_holm (corrected)
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
    for pval_col in ["delta_r2_bootstrap_pval", "delta_rmse_bootstrap_pval", "wilcoxon_pval"]:
        corrected_col = f"{pval_col}_holm"
        df[corrected_col] = _holm_bonferroni_correction(df[pval_col].to_numpy())
    
    return df
