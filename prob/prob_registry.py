"""
Registry of probe models and their GridSearchCV param grids.

Add or remove entries here to change which linear/non-linear probes run
without editing prob_models_and_run. Each entry is:
  {"name": str, "estimator": sklearn-like estimator, "param_grid": dict}
Param grid keys must use the "model__" prefix for the pipeline.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.neural_network import MLPRegressor


# Shared random state for reproducibility
RANDOM_STATE = 96


def _ridge():
    return Ridge(random_state=RANDOM_STATE)


def _lasso():
    return Lasso(random_state=RANDOM_STATE, max_iter=10000)


def _rf():
    return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)


def _mlp():
    return MLPRegressor(
        random_state=RANDOM_STATE,
        early_stopping=True,
        max_iter=200,
    )


# Linear probes: add ElasticNet, etc. by appending to this list
LINEAR_PROBES: List[Dict[str, Any]] = [
    {
        "name": "ridge",
        "estimator": _ridge(),
        "param_grid": {"model__alpha": np.logspace(-5, 4, 10)},
    },
    {
        "name": "lasso",
        "estimator": _lasso(),
        "param_grid": {"model__alpha": np.logspace(-5, 1, 7)},
    },
]

# Non-linear probes: add more by appending
NONLINEAR_PROBES: List[Dict[str, Any]] = [
    {
        "name": "random_forest",
        "estimator": _rf(),
        "param_grid": {
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_leaf": [1, 2, 4],
        },
    },
    {
        "name": "mlp",
        "estimator": _mlp(),
        "param_grid": {
            "model__hidden_layer_sizes": [(128,), (256,), (256, 128)],
            "model__activation": ["relu"],
            "model__alpha": [1e-5, 1e-4, 1e-3],
            "model__learning_rate_init": [1e-3, 3e-3],
            "model__max_iter": [200],
            "model__early_stopping": [True],
            "model__random_state": [RANDOM_STATE],
        },
    },
]
