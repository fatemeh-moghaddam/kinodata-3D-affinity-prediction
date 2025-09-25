import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

import kinodata.configuration as cfg
from prob.paths_and_io import get_out_dir, get_exp_dirs, load_out_tensor, load_y_by_ids, load_X_from_pt
import wandb


RANDOM_STATE = 96
# Targets configuration for traceability
TARGET_FILE = "nitrogen_counts.pt"  # you can change per experiment


# ─────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────

def build_experiment_name(ds_cfg, layer_num: int) -> str:
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
        )
    
    if kwargs.keys() - defaults.keys() != set():
        raise ValueError(f"Invalid arguments: {kwargs.keys() - defaults.keys()}")

    if "gnn_model_type" in kwargs:
        assert kwargs["gnn_model_type"] in ["CGNN-3D", "CGNN", "DTI"], "Invalid GNN model type"
    if "split_type" in kwargs:
        assert kwargs["split_type"] in ["random-k-fold", "scaffold-k-fold", "pocket-k-fold"], "Invalid split type"
    if "filter_rmsd_max_value" in kwargs:
        assert kwargs["filter_rmsd_max_value"] in set({2, 4, 6, 2.00, 4.00, 6.00, None}), "Invalid RMSD threshold"
    if "split_index" in kwargs:
        assert kwargs["split_index"] in set({0, 1, 2, 3, 4, None}), "Invalid split index"
    if "config_name" in kwargs:
        config_name = kwargs["config_name"]
    else:
        config_name = "prob_ds_load"
    if "target_file" in kwargs:
        target_file = kwargs["target_file"]
    else:
        target_file = TARGET_FILE

    # merge: kwargs overrides defaults
    config_args = {**defaults, **kwargs}
    # Initialize the config with the defaults and kwargs
    output_dir = get_out_dir(config_args["gnn_model_type"],
                                config_args["filter_rmsd_max_value"],
                                config_args["split_type"],
                                split_fold=None)
    
    target_dir = output_dir.parents[2] / "targets"

    cfg.register(config_name, **{**config_args, "output_dir": output_dir, "target_dir": target_dir})
    prob_ds_config = cfg.get(config_name).update_from_args() # this activates the argparse itself
    return prob_ds_config


# ─────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────


def plot_parity(y_true, y_pred, title: str, save_path: Path | None = None):
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=y_true, y=y_pred, s=12, alpha=0.5)
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    plt.show()


def plot_residuals(y_true, y_pred, title: str, save_path: Path | None = None):
    residuals = y_pred - y_true
    plt.figure(figsize=(5.5, 4))
    sns.histplot(residuals, bins=50, kde=True)
    plt.axvline(0, color="r", linestyle="--", linewidth=1)
    plt.xlabel("Residual (y_pred - y_true)")
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    plt.show()

# ─────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────
def regression_scorers():
    """Common regression scorers for cross-validated model selection."""
    return {
        "r2": make_scorer(r2_score),
        "neg_rmse": make_scorer((lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))), greater_is_better=False),
        "neg_mae": make_scorer(mean_absolute_error, greater_is_better=False),
    }


def evaluate_predictions(y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"r2": r2, "rmse": rmse, "mae": mae}

# ─────────────────────────────────────────────────────────────
# Run CV and save artifacts
# ─────────────────────────────────────────────────────────────

def run_cv_search(X, y, base_estimator, param_grid: dict, n_splits: int = 5, n_jobs: int = -1, exp_dirs: dict | None = None, model_name: str = "model"):
    """Run GridSearchCV with a standard pipeline and return the fitted search, metrics on CV-held out predictions, and store artifacts."""
    # Standardize features; target left as-is
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", base_estimator),
    ])

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(
        estimator=pipe,
        # estimator=base_estimator,
        param_grid=param_grid,
        scoring=regression_scorers(),
        refit="r2",
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True,
    )

    start = time.time()
    search.fit(X, y)
    elapsed = time.time() - start

    # Evaluate on cross-validated predictions using best estimator
    y_pred = search.best_estimator_.predict(X)
    metrics = evaluate_predictions(y, y_pred)
    metrics["fit_seconds"] = elapsed

    # Save artifacts
    if exp_dirs is not None:
        # CV results
        cv_df = pd.DataFrame(search.cv_results_)
        cv_df.to_csv(exp_dirs["artifacts"] / f"{model_name}_cv_results.csv", index=False)
        # Best params and metrics summary
        summary = {
            "model": model_name,
            "best_params": search.best_params_,
            "best_score_r2": search.best_score_,
            "metrics_on_full_data": metrics,
            "n_samples": int(len(y)),
            "n_features": int(X.shape[1]),
        }
        with open(exp_dirs["reports"] / f"{model_name}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        # Predictions
        pred_df = pd.DataFrame({"y_true": y, "y_pred": y_pred})
        pred_df.to_csv(exp_dirs["artifacts"] / f"{model_name}_predictions.csv", index=False)

        # Plots
        plot_parity(y, y_pred, title=f"{model_name} Parity (R2={metrics['r2']:.3f})", save_path=exp_dirs["figures"] / f"{model_name}_parity.png")
        plot_residuals(y, y_pred, title=f"{model_name} Residuals (RMSE={metrics['rmse']:.3f})", save_path=exp_dirs["figures"] / f"{model_name}_residuals.png")

    return search, metrics, y_pred



def main(prob_config: cfg.Config):
    # Run experiments across selected layers for linear and non-linear models
    all_runs = []
    # Choose which layer(s) to probe
    layer_nums = [1,2,3]  # e.g., [0,1,2,3] to sweep multiple layers

    rf_grid = {
    "model__n_estimators": [200, 500],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_leaf": [1, 2, 4],
    }

    mlp_grid = {
        "model__hidden_layer_sizes": [(128,), (256,), (256, 128)],
        "model__activation": ["relu"],
        "model__alpha": [1e-5, 1e-4, 1e-3],
        "model__learning_rate_init": [1e-3, 3e-3],
        "model__max_iter": [200],
        "model__early_stopping": [True],
        "model__random_state": [RANDOM_STATE],
    }
    # For Linear models
    ridge_grid = {"model__alpha": np.logspace(-6, 6, 13)}
    lasso_grid = {"model__alpha": np.logspace(-6, 1, 8)}
    enet_grid  = {"model__alpha": np.logspace(-6, 3, 10),
                "model__l1_ratio": [0.1, 0.5, 0.9]}


    # Load target values
    y = load_y_by_ids(prob_config.output_dir, target_dir=prob_config.target_dir, targets_file=TARGET_FILE)

    for layer in layer_nums:
        exp_name = build_experiment_name(prob_config, layer)
        

        # Load data
        X = load_X_from_pt(prob_config.output_dir, layer_num=layer)
        
        # Linear models
        models_and_grids = [
            ("ridge", Ridge(random_state=RANDOM_STATE), ridge_grid),
            ("lasso", Lasso(random_state=RANDOM_STATE, max_iter=20000), lasso_grid),
            ("elasticnet", ElasticNet(random_state=RANDOM_STATE, max_iter=20000), enet_grid),
        ]

        for name, base, grid in models_and_grids:
            exp_dirs = get_exp_dirs(prob_config.output_dir, target=Path(TARGET_FILE).stem, prob_model=name)
            search, metrics, _ = run_cv_search(X, y, base, grid, n_splits=5, n_jobs=-1, exp_dirs=exp_dirs, model_name=name)
            all_runs.append({"experiment": exp_name, "model": name, **metrics, **search.best_params_})

        # Non-linear models
        nonlinear_models = [
            ("random_forest", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1), rf_grid),
            # ("mlp", MLPRegressor(random_state=RANDOM_STATE), mlp_grid),
        ]

        for name, base, grid in nonlinear_models:
            search, metrics, _ = run_cv_search(X, y, base, grid, n_splits=5, n_jobs=-1, exp_dirs=exp_dirs, model_name=name)
            all_runs.append({"experiment": exp_name, "model": name, **metrics, **search.best_params_})

    # Aggregate summary across runs
    summary_df = pd.DataFrame(all_runs)
    summary_csv = prob_config.output_dir / "experiments" / "summary_runs.csv"
    summary_df.to_csv(summary_csv, index=False)
    summary_df.sort_values(["r2"], ascending=False).head(10)


if __name__ == "__main__":
    # Configuration
    ds_load_config = get_ds_load_config()

    # Initialize Weights & Biases
    wandb.init(
        project="probing",
        name=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            **ds_load_config.to_dict(),
            "random_state": RANDOM_STATE,
            "target_file": TARGET_FILE,
        },
    )
    main(ds_load_config)