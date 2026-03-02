"""
Regression metrics and scorers for probing experiments.

Pure functions over (y_true, y_pred) for easy testing and reuse in
statistical analysis (e.g. bootstrap, permutation tests).
"""
from __future__ import annotations

from typing import Dict, Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import make_scorer


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute R², RMSE, and MAE. Useful for reporting and downstream stats."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"r2": r2, "rmse": rmse, "mae": mae}


def regression_scorers() -> Dict[str, Any]:
    """Scorers for GridSearchCV (refit on r2, lower is better for neg_*)."""
    return {
        "r2": make_scorer(r2_score),
        "neg_rmse": make_scorer(
            lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            greater_is_better=False,
        ),
        "neg_mae": make_scorer(mean_absolute_error, greater_is_better=False),
    }
