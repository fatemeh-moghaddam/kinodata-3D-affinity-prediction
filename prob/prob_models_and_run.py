import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

import kinodata.configuration as cfg
from prob.paths_and_io import get_exp_dirs, load_out_tensor, load_y_by_ids, load_X_from_pt
from prob.prob_config import get_ds_load_config, build_experiment_name

import wandb


RANDOM_STATE = 96
N_SPLITS_CV = 5
TEST_SIZE = 0.1

# Targets configuration for traceability
TARGET_FILE = "nitrogen_counts.pt"  # you can change per experiment
_ROOT = Path(os.environ.get("HOME_PROJ_DIR", Path(__file__).resolve().parents[1])) # to allow setting a different root via env variable


# ─────────────────────────────────────────────────────────────
# Resource helpers
# ─────────────────────────────────────────────────────────────
def _cpu_budget(default=16, sub_file=None):
    # 1. Trying scheduler environment variables, based on common conventions
    for key in ("SLURM_CPUS_PER_TASK", "NSLOTS", "OMP_NUM_THREADS"):
        if key in os.environ and os.environ[key].isdigit():
            return int(os.environ[key])

    # 2. parsing the submission file if provided
    if sub_file and Path(sub_file).exists():
        text = Path(sub_file).read_text()
        match = re.search(r"request_CPUs\s*=\s*(\d+)", text, flags=re.I)
        if match:
            return int(match.group(1))

    # 3. Fallback
    return default



# ─────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────


def plot_parity(y_true, y_pred, title: str, save_path: Path | None = None):
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.figure(figsize=(6, 6))
    g = sns.JointGrid(x=y_true, y=y_pred, space=0)
    g.plot(sns.scatterplot, sns.histplot, joint_kws={"alpha": 0.5, "s": 12}, marginal_kws={"bins": 30, "fill": True})
    # same as writing:
    # g.plot_joint(sns.scatterplot, alpha=0.5, s=12)
    # g.plot_marginals(sns.histplot, bins=30, fill=True)
    g.ax_joint.plot(lims, lims, "r--", linewidth=1)
    g.set_axis_labels("True", "Predicted")
    g.fig.suptitle(title, y=1.02)
    g.ax_joint.set_xlim(lims)
    g.ax_joint.set_ylim(lims)
    if save_path is not None:
        g.fig.savefig(save_path, dpi=200) # dot per inch
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


def plot_contours(X, y, model, title: str, save_path: Path | None = None):
    ...

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

def run_cv_search(X, y, prob_model, param_grid: dict, n_splits: int = N_SPLITS_CV, n_jobs: int = 0, exp_dirs: dict | None = None, model_name: str = "model"):
    """Run GridSearchCV with a standard pipeline and return the fitted search, metrics on CV-held out predictions, and store artifacts."""
    if n_jobs == 0:
        n_jobs = _cpu_budget()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Standardize features; target left as-is
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", prob_model),
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
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    # Evaluate on held-out set using best estimator
    y_pred = search.best_estimator_.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)
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
            "metrics_on_unseen_data": metrics,
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



# def main(prob_config: cfg.Config):
#     # Run experiments across selected layers for linear and non-linear models
#     all_runs = []
#     # Choose which layer(s) to probe
#     layer_nums = [1,2,3]  # e.g., [0,1,2,3] to sweep multiple layers

#     # Determine CPU budget
#     sub_path = _ROOT / "prob/cluster/"
#     sub_file = Path(sub_path / "run_prob.sub")  # change as needed
#     n_jobs = _cpu_budget(sub_file=sub_file)

#     rf_grid = {
#     "model__n_estimators": [200, 500],
#     "model__max_depth": [None, 10, 20],
#     "model__min_samples_leaf": [1, 2, 4],
#     }

#     mlp_grid = {
#         "model__hidden_layer_sizes": [(128,), (256,), (256, 128)],
#         "model__activation": ["relu"],
#         "model__alpha": [1e-5, 1e-4, 1e-3],
#         "model__learning_rate_init": [1e-3, 3e-3],
#         "model__max_iter": [200],
#         "model__early_stopping": [True],
#         "model__random_state": [RANDOM_STATE],
#     }
#     # For Linear models
#     ridge_grid = {"model__alpha": np.logspace(-5, 1, 5)}
#     lasso_grid = {"model__alpha": np.logspace(-5, 1, 5)}
#     enet_grid  = {"model__alpha": np.logspace(-5, 1, 5),
#                 "model__l1_ratio": [0.1, 0.5, 0.9]}


#     # Load target values
#     y = load_y_by_ids(prob_config.output_dir, target_dir=prob_config.target_dir, targets_file=TARGET_FILE)
#     target_name = Path(TARGET_FILE).stem
    
#     for layer in layer_nums:
#         # exp_name = build_experiment_name(prob_config, layer)
        

#         # Load data
#         X = load_X_from_pt(prob_config.output_dir, layer_num=layer)
        
#         # Linear models
#         models_and_grids = [
#             ("ridge", Ridge(random_state=RANDOM_STATE), ridge_grid),
#             ("lasso", Lasso(random_state=RANDOM_STATE, max_iter=20000), lasso_grid),
#             ("elasticnet", ElasticNet(random_state=RANDOM_STATE, max_iter=20000), enet_grid),
#         ]

#         for model_name, base, grid in models_and_grids:
#             exp_dirs = get_exp_dirs(prob_config.output_dir, target=target_name, prob_model=model_name, layer_num=layer)
#             search, metrics, _ = run_cv_search(X, y, base, grid, n_splits=N_SPLITS_CV, exp_dirs=exp_dirs, n_jobs=n_jobs, model_name=model_name)
#             all_runs.append({"experiment": f"{target_name}_{model_name}", "layer": layer, **metrics, **search.best_params_})

#         # Non-linear models
#         nonlinear_models = [
#             ("random_forest", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1), rf_grid),
#             ("mlp", MLPRegressor(random_state=RANDOM_STATE, n_jobs=1), mlp_grid),
#         ]

#         for model_name, base, grid in nonlinear_models:
#             exp_dirs = get_exp_dirs(prob_config.output_dir, target=target_name, prob_model=model_name, layer_num=layer)
#             search, metrics, _ = run_cv_search(X, y, base, grid, n_splits=N_SPLITS_CV, exp_dirs=exp_dirs, n_jobs=n_jobs, model_name=model_name)
#             all_runs.append({"experiment": f"{target_name}_{model_name}", "layer": layer, **metrics, **search.best_params_})

#         # Aggregate summary across runs per layer
#         summary_df = pd.DataFrame(all_runs)
#         summary_csv = prob_config.output_dir / target_name / "experiments" / "summary_runs.csv"
#         if not summary_csv.parent.exists():
#             summary_csv.parent.mkdir(parents=True, exist_ok=True)
#         summary_df.to_csv(summary_csv, index=False)
#         summary_df.sort_values(["r2"], ascending=False).head(10)


def linear_models(prob_config: cfg.Config, X: np.ndarray, y: np.ndarray, n_jobs: int = _cpu_budget()):
    # Define the linear models and their parameter grids
    ridge_grid = {"model__alpha": np.logspace(-5, 4, 10)}
    lasso_grid = {"model__alpha": np.logspace(-5, 1, 7)}
    # enet_grid  = {"model__alpha": np.logspace(-5, 4, 10),
    #             "model__l1_ratio": [0.1, 0.5, 0.9]}


    target_name = prob_config.target_file.stem
    layer = prob_config.layer_num

    # Run experiments for each model
    all_runs = []
    for model_name, base, grid in [("ridge", Ridge(random_state=RANDOM_STATE), ridge_grid),
                                    ("lasso", Lasso(random_state=RANDOM_STATE, max_iter=10000), lasso_grid)]:
                                    # ("elasticnet", ElasticNet(random_state=RANDOM_STATE, max_iter=10000), enet_grid)]:
        exp_dirs = get_exp_dirs(prob_config.output_dir, target=target_name, prob_model=model_name, layer_num=layer)
        search, metrics, _ = run_cv_search(X, y, base, grid, n_splits=N_SPLITS_CV, exp_dirs=exp_dirs, n_jobs=n_jobs, model_name=model_name)
        all_runs.append({"experiment": f"{target_name}_{model_name}", "layer": layer, **metrics, **search.best_params_})

    return all_runs


def non_linear_models(prob_config: cfg.Config, X: np.ndarray, y: np.ndarray, n_jobs: int = _cpu_budget()):
    # Define the non-linear models and their parameter grids
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

    target_name = prob_config.target_file.stem
    layer = prob_config.layer_num

    # Run experiments for each model
    all_runs = []
    for model_name, base, grid in [("random_forest", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1), rf_grid),
                                    ("mlp", MLPRegressor(random_state=RANDOM_STATE, n_jobs=1), mlp_grid)]:
        exp_dirs = get_exp_dirs(prob_config.output_dir, target=target_name, prob_model=model_name, layer_num=layer)
        search, metrics, _ = run_cv_search(X, y, base, grid, n_splits=N_SPLITS_CV, exp_dirs=exp_dirs, n_jobs=n_jobs, model_name=model_name)
        all_runs.append({"experiment": f"{target_name}_{model_name}", "layer": layer, **metrics, **search.best_params_})

    return all_runs


def main(prob_config: cfg.Config):
    # Run experiments across selected layers for linear and non-linear models
    all_runs = []
    # Choose which layer(s) to probe
    layer_nums = [1,2,3]  # e.g., [0,1,2,3] to sweep multiple layers

    # Determine CPU budget
    sub_path = _ROOT / "prob/cluster/"
    sub_file = Path(sub_path / "run_prob.sub")  # change as needed
    n_jobs = _cpu_budget(sub_file=sub_file)

    # Load target values
    y = load_y_by_ids(prob_config.output_dir, target_dir=prob_config.target_dir, targets_file=TARGET_FILE)
    target_name = Path(TARGET_FILE).stem
    
    for layer in layer_nums:
        X = load_X_from_pt(prob_config.output_dir, layer_num=layer)

        # Linear models
        all_runs.extend(linear_models(prob_config, X, y, n_jobs=n_jobs))

        # Non-linear models
        # all_runs.extend(non_linear_models(prob_config, X, y, n_jobs=n_jobs))

        # Aggregate summary across runs per layer
        summary_df = pd.DataFrame(all_runs)
        summary_csv = prob_config.output_dir / target_name / "experiments" / "summary_runs.csv"
        if not summary_csv.parent.exists():
            summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_csv, index=False)
        summary_df.sort_values(["r2"], ascending=False).head(10)

    return all_runs


if __name__ == "__main__":
    # Configuration
    ds_load_config = get_ds_load_config()

    # Initialize Weights & Biases
    wandb.init(
        project="probing",
        name=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            **ds_load_config,
            "random_state": RANDOM_STATE,
            "target_file": TARGET_FILE,
        },
    )
    main(ds_load_config)