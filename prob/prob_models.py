"""
Registry of probe models and their GridSearchCV param grids.

Add or remove entries here to change which linear/non-linear probes run
without editing prob_orchestrate. Each entry is:
  {"name": str, "estimator": sklearn-like estimator, "param_grid": dict, "n_jobs": Optional[int]}
Param grid keys must use the "model__" prefix for the pipeline. "n_jobs", if
present, overrides prob_orchestrate's default GridSearchCV parallelism for
that entry (used to cap the GPU-backed "mlp" probe to n_jobs=1).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from torch import nn


# Shared random state for reproducibility
RANDOM_STATE = 96


def _resolve_activation(name: str) -> type[nn.Module]:
    try:
        return {"relu": nn.ReLU, "tanh": nn.Tanh}[name]
    except KeyError:
        raise ValueError(f"Unsupported activation {name!r}; expected 'relu' or 'tanh'")


class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-compatible PyTorch MLP regressor, trained on `device`.

    Mirrors the subset of MLPRegressor's hyperparameters used by the "mlp"
    probe's param_grid so it plugs into the existing Pipeline/GridSearchCV
    probing code unchanged, but can train on GPU.
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (256,),
        activation: str = "relu",
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        max_iter: int = 200,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
        batch_size: int = 256,
        random_state: Optional[int] = None,
        device: str = "cpu",
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device

    def _build_network(self, n_features: int) -> nn.Sequential:
        """Stack of Linear+activation per hidden layer, ending in a Linear(., 1)."""
        activation = _resolve_activation(self.activation)
        layer_sizes = [n_features, *self.hidden_layer_sizes, 1]

        layers: List[nn.Module] = []
        for in_size, out_size in zip(layer_sizes, layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(activation())
        layers.pop()  # drop the activation after the final (output) layer

        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchMLPRegressor":
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        if self.early_stopping:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_fraction, random_state=self.random_state
            )
        else:
            X_train, y_train = X, y
            X_val = y_val = None

        net = self._build_network(X.shape[1]).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha)
        loss_fn = nn.MSELoss()

        X_train_t = torch.from_numpy(X_train).to(device)
        y_train_t = torch.from_numpy(y_train).to(device)
        if X_val is not None:
            X_val_t = torch.from_numpy(X_val).to(device)
            y_val_t = torch.from_numpy(y_val).to(device)

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_state = None
        n_samples = X_train_t.shape[0]

        for _ in range(self.max_iter):
            net.train()
            perm = torch.randperm(n_samples, device=device)
            for start in range(0, n_samples, self.batch_size):
                idx = perm[start : start + self.batch_size]
                optimizer.zero_grad()
                pred = net(X_train_t[idx])
                loss = loss_fn(pred, y_train_t[idx])
                loss.backward()
                optimizer.step()

            if self.early_stopping:
                net.eval()
                with torch.no_grad():
                    val_loss = loss_fn(net(X_val_t), y_val_t).item()
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.n_iter_no_change:
                        break

        if best_state is not None:
            net.load_state_dict(best_state)

        self.net_ = net
        self.device_ = device
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        self.net_.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X).to(self.device_)
            pred = self.net_(X_t).cpu().numpy()
        return pred.reshape(-1)


# XGBoost's GPU "random forest mode" requires an explicit max_depth (no "grow
# until pure" option like sklearn's max_depth=None), so on GPU this is the
# fallback -- the deepest value already tested elsewhere in this probe's grid.
_GPU_MAX_DEPTH_FOR_UNLIMITED = 20


class HybridRandomForestRegressor(BaseEstimator, RegressorMixin):
    """RandomForestRegressor that fits on GPU (via xgboost) or CPU (via sklearn)
    based on `device`.

    Exposes the same n_estimators/max_depth/min_samples_leaf/random_state API
    regardless of backend, so it plugs into the existing Pipeline/GridSearchCV
    probing code and param_grid unchanged. The GPU backend is xgboost's
    "random forest" mode (a single boosting round of num_parallel_tree bagged
    trees, each on a row/column subsample) rather than sklearn's bagging
    algorithm -- close in spirit, not a bit-for-bit equivalent.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
        device: str = "cpu",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.device = device

    def _build_backend(self):
        if self.device == "cuda":
            import xgboost as xgb

            max_depth = self.max_depth if self.max_depth is not None else _GPU_MAX_DEPTH_FOR_UNLIMITED
            return xgb.XGBRegressor(
                n_estimators=1,
                num_parallel_tree=self.n_estimators,
                max_depth=max_depth,
                min_child_weight=self.min_samples_leaf,
                subsample=0.8,
                colsample_bynode=0.8,
                tree_method="hist",
                device="cuda",
                random_state=self.random_state,
            )
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HybridRandomForestRegressor":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.backend_ = self._build_backend()
        self.backend_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.device == "cuda":
            # Tree building (the expensive part) already ran on GPU in fit();
            # X here is a host (numpy) array, so predicting on the still-cuda
            # booster would otherwise hit xgboost's slow mismatched-device
            # DMatrix fallback. Switch the trained booster to cpu for inference.
            self.backend_.set_params(device="cpu")
        return np.asarray(self.backend_.predict(X)).reshape(-1)


def _ridge():
    return Ridge(random_state=RANDOM_STATE)


def _lasso():
    return Lasso(random_state=RANDOM_STATE, max_iter=10000)


def _rf(device: str = "cpu"):
    return HybridRandomForestRegressor(random_state=RANDOM_STATE, device=device)


def _mlp(device: str = "cpu"):
    return TorchMLPRegressor(
        random_state=RANDOM_STATE,
        early_stopping=True,
        max_iter=200,
        device=device,
    )


# Linear probes: add ElasticNet, etc. by appending to this list
LINEAR_PROBES: List[Dict[str, Any]] = [
    {
        "name": "ridge",
        "estimator": _ridge(),
        "param_grid": {"model__alpha": np.logspace(-5, 4, 10)},
    },
    # {
    #     "name": "lasso",
    #     "estimator": _lasso(),
    #     "param_grid": {"model__alpha": np.logspace(-5, 1, 7)},
    # },
]

# Non-linear probes: add more by appending. Both estimators default to
# device="cpu"; prob_orchestrate.py overrides entry["estimator"].device (and
# caps entry["n_jobs"] to 1) when the job is configured to run on cuda, since
# many parallel GridSearchCV workers would otherwise contend for the same GPU.
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
