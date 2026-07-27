"""
Plotting helpers for probing results (parity, residuals, distributions).

Optional dependency: matplotlib/seaborn. Safe to import; plotting is only
performed when functions are called, so headless runs can skip them.

Design goals:
- produce publication-quality, consistent figures
- make saving to experiment directories convenient and explicit
"""
from __future__ import annotations

from pathlib import Path

from prob.paths_and_io import EXP_DIR_FIGURES
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress


# ─────────────────────────────────────────────────────────────
# Global style / constants
# ─────────────────────────────────────────────────────────────

FIG_DPI = 300
FIG_SIZE_SQUARE = (6, 6)
FIG_SIZE_WIDE = (11, 4)
FIG_SIZE_SCATTER = (6, 5)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)


def _resolve_save_path(save_path: Optional[Path | str]) -> Optional[Path]:
    """
    Decide where to save a figure.

    Behaviour:
    - If save_path is None, return None (no saving).
    - Otherwise, interpret save_path as "<experiment_root>/<filename_stem>[.ext]".
      Save under a "figures" subdirectory with a ".png" suffix, unless
      save_path already points inside a "figures" directory.

    Example:
        save_path = exp_root / "parity_layer3"
        -> exp_root / "figures" / "parity_layer3.png"
    """
    if save_path is None:
        return None

    p = Path(save_path)
    parent = p.parent
    name = p.name

    if parent.name != EXP_DIR_FIGURES:
        parent = parent / EXP_DIR_FIGURES
    parent.mkdir(parents=True, exist_ok=True)

    if not name.endswith(".png"):
        name = f"{name}.png"

    return parent / name


def _save_and_show(fig, save_path: Optional[Path | str], show: bool = True) -> None:
    """Common pattern: save figure, optionally show, then close."""
    out_path = _resolve_save_path(save_path)
    if out_path is not None:
        fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _regression_metrics_text(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Compute R2 and RMSE, formatted as a multi-line annotation string."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rmse = np.sqrt(np.mean(np.square(y_pred - y_true)))
    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot = np.sum(np.square(y_true - y_true.mean()))
    r2 = 1 - ss_res / ss_tot
    mae = np.mean(np.abs(y_pred - y_true))
    corr_coef = np.corrcoef(y_true, y_pred)[0, 1]


    return f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}\nCorr = {corr_coef:.3f}"


def _annotate_metrics(ax, y_true: np.ndarray, y_pred: np.ndarray, loc: str = "lower right") -> None:
    """Draw an R2/RMSE text box in a corner of `ax` using axes-fraction coords."""
    text = _regression_metrics_text(y_true, y_pred)

    positions = {
        "lower right": dict(x=0.97, y=0.03, ha="right", va="bottom"),
        "upper right": dict(x=0.97, y=0.97, ha="right", va="top"),
        "upper left": dict(x=0.03, y=0.97, ha="left", va="top"),
        "lower left": dict(x=0.03, y=0.03, ha="left", va="bottom"),
    }
    pos = positions[loc]

    ax.text(
        pos["x"], pos["y"], text,
        transform=ax.transAxes,
        ha=pos["ha"], va=pos["va"],
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="0.7"),
    )


def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Optional[Path] = None,
    show: bool = True,
    context_overwrite: None | str = None,
    show_mean_line: bool = True,
    annotate_variance_ratio: bool = False,
    density: Optional[str] = "kde",  # None | "hexbin" | "kde",
    density_cmap: str = "Blues_r",
    show_metrics: bool = True,
) -> None:
    """Scatter of predicted vs true with diagonal reference line.

    density: if set, replaces the scatter with a density-aware plot to
        avoid overplotting artifacts at high point counts.
        "hexbin" -> 2D hexbin with count colormap
        "kde"    -> 2D kernel density estimate (filled contours)
    """
    if context_overwrite is not None:
        sns.set_context(context_overwrite)

    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]

    g = sns.JointGrid(x=y_true, y=y_pred, space=0, height=FIG_SIZE_SQUARE[0])

    if density is None:
        g.plot_joint(sns.scatterplot, alpha=0.4, s=16, edgecolor="none")
    elif density == "hexbin":
        hb = g.ax_joint.hexbin(
            y_true, y_pred, gridsize=80, cmap=density_cmap, mincnt=1, extent=lims + lims
        )
        g.fig.colorbar(hb, ax=g.ax_joint, label="count", fraction=0.046, pad=0.04)
    elif density == "kde":
        g.plot_joint(sns.kdeplot, fill=True, cmap=density_cmap, thresh=0.02, levels=20)
    else:
        raise ValueError(f"Unknown density mode: {density}")

    g.plot_marginals(sns.histplot, bins=30, fill=True, element="step")
    g.ax_joint.plot(lims, lims, "r--", linewidth=1.3)

    if show_mean_line:
        mean_true = y_true.mean()
        g.ax_joint.axhline(
            mean_true, color="gray", linestyle=":", linewidth=1.7,
            label=f"mean(y_true)={mean_true:.2f}",
        )
        g.ax_joint.legend(loc="upper left", fontsize=8, frameon=False)

    g.set_axis_labels("True", "Predicted")

    if show_metrics:
        _annotate_metrics(g.ax_joint, y_true, y_pred, loc="lower right")

    if annotate_variance_ratio:
        ratio = y_pred.std() / y_true.std()
        title = f"{title}  |  std(pred)/std(true)={ratio:.3f}"

    g.fig.suptitle(title, y=1.02)
    g.ax_joint.set_xlim(lims)
    g.ax_joint.set_ylim(lims)

    _save_and_show(g.fig, save_path, show)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Optional[Path] = None,
    show: bool = True,
    vs_true: bool = True,
    show_metrics: bool = True,
) -> None:
    """Histogram of residuals (y_pred - y_true).

    vs_true: adds a second panel plotting residuals against y_true,
        which reveals patterned/shrinkage residuals that a plain
        histogram can hide.
    """
    residuals = y_pred - y_true

    BINS = 60
    COLOR = "C0"
    ALPHA_HIST = 0.8

    if not vs_true:
        fig, ax = plt.subplots(figsize=(5.7, 4.3))
        sns.histplot(residuals, bins=BINS, kde=True, ax=ax, color=COLOR, alpha=ALPHA_HIST)
        ax.axvline(0, color="r", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Residual (y_pred - y_true)")
        ax.set_ylabel("Count")
        ax.set_title(title)
        if show_metrics:
            _annotate_metrics(ax, y_true, y_pred, loc="upper right")
        fig.tight_layout()
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3))

        sns.histplot(residuals, bins=BINS, kde=True, ax=ax1, color=COLOR, alpha=ALPHA_HIST)
        ax1.axvline(0, color="r", linestyle="--", linewidth=1.2)
        ax1.set_xlabel("Residual (y_pred - y_true)")
        ax1.set_ylabel("Count")
        if show_metrics:
            _annotate_metrics(ax1, y_true, y_pred, loc="upper right")

        ax2.scatter(y_true, residuals, alpha=0.3, s=12, edgecolor="none")
        ax2.axhline(0, color="r", linestyle="--", linewidth=1.2)
        ax2.set_xlabel("True")
        ax2.set_ylabel("Residual (y_pred - y_true)")

        fig.suptitle(title)
        fig.tight_layout()

    _save_and_show(fig, save_path, show)


def plot_conditions_box(
    conditions_dict: dict,
    *,
    metric: str = "squared_error",
    baseline_dict: Optional[dict] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Box plot of per-sample metrics across conditions.
    
    Args:
        conditions_dict: Dict mapping condition name -> (y_true, y_pred)
        metric: "squared_error" or "abs_residual"
        baseline_dict: Optional dict of baseline conditions to include
        title: Plot title
        save_path: Where to save figure
        show: Whether to display
    
    Example:
        conditions = {
            "layer_1": (y_true, y_pred_1),
            "layer_2": (y_true, y_pred_2),
            "layer_3": (y_true, y_pred_3),
        }
        baseline = {
            "shuffled": (y_true, y_pred_shuffled),
        }
        plot_conditions_box(conditions, baseline_dict=baseline, metric="squared_error")
    """
    # Determine metric function
    if metric.lower() in {"squared_error", "sq_error", "squared"}:
        metric_fn = lambda y_true, y_pred: np.square(y_pred - y_true)
        metric_label = "Squared Error"
    elif metric.lower() in {"abs_residual", "absolute_residual", "abs_error"}:
        metric_fn = lambda y_true, y_pred: np.abs(y_pred - y_true)
        metric_label = "Absolute Residual"
    else:
        raise ValueError("metric must be 'squared_error' or 'abs_residual'")
    
    # Helper to add condition data to rows
    def add_to_rows(cond_dict, cond_type):
        for cond_name, (y_true, y_pred) in cond_dict.items():
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            values = metric_fn(y_true, y_pred)
            for val in values:
                rows.append({
                    "condition": cond_name,
                    "type": cond_type,
                    "metric": float(val)
                })
    
    # Collect all data
    rows = []
    add_to_rows(conditions_dict, "real")
    if baseline_dict is not None:
        add_to_rows(baseline_dict, "baseline")
    
    if not rows:
        raise ValueError("No data to plot")
    
    df = pd.DataFrame(rows)
    
    # Sort conditions alphabetically (layer_1, layer_2, layer_3)
    condition_order = sorted(df["condition"].unique())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(8, 2 * len(condition_order)), 5.5))
    
    # Create box plot with hue for type (real vs baseline)
    sns.boxplot(
        data=df,
        x="condition",
        y="metric",
        hue="type",
        order=condition_order,
        hue_order=["real", "baseline"] if baseline_dict else None,
        palette={"real": "C0", "baseline": "C1"},
        ax=ax,
        width=0.6,
    )
    
    ax.set_xlabel("Condition")
    ax.set_ylabel(metric_label)
    ax.set_title(title or f"{metric_label} Distribution Across Conditions")
    ax.legend(title="Type", frameon=True)
    fig.tight_layout()
    
    _save_and_show(fig, save_path, show)


def plot_dist_with_log(
    df: pd.DataFrame | np.ndarray | pd.Series,
    col: str,
    *,
    bins: int = 80,
    show_ecdf: bool = False,
    kde: bool = False,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
    context_overwrite: None | str = None,
) -> None:
    """Side-by-side linear / log-count histogram of a column."""
    if context_overwrite is not None:
        sns.set_context(context_overwrite)

    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE_WIDE, sharex=True)

    # linear
    sns.histplot(
        data=df,
        x=col,
        bins=bins,
        kde=kde,
        stat="count",
        edgecolor="black",
        alpha=0.8,
        ax=axes[0],
    )
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Count")
    axes[0].set_title(title or f"{col} distribution")

    if show_ecdf:
        ax_ecdf = axes[0].twinx()
        sns.ecdfplot(data=df, x=col, ax=ax_ecdf, color="C1", linewidth=2)
        ax_ecdf.set_ylabel("ECDF")
        ax_ecdf.set_ylim(0, 1)

    # log y
    sns.histplot(
        data=df,
        x=col,
        bins=bins,
        kde=kde,
        stat="count",
        # log_scale=(False, True),
        element="bars",      # force rectangles
        fill=True,
        edgecolor="black",
        alpha=0.85,
        ax=axes[1],
    )
    axes[1].set_yscale("symlog", linthresh=20)
    axes[1].set_xlabel(col)
    axes[1].set_ylabel("Count (log scale)")
    axes[1].set_title(f"{title or col} (log-scaled)")

    if show_ecdf:
        ax_ecdf = axes[1].twinx()
        sns.ecdfplot(
            data=df,
            x=col,
            ax=ax_ecdf,
            color="C1",
            linewidth=2,
        )
        ax_ecdf.set_ylabel("ECDF")
        ax_ecdf.set_ylim(0, 1)

    fig.tight_layout()

    _save_and_show(fig, save_path, show)


def scatter_affinity_vs_bonds(
    df: pd.DataFrame,
    bond_col: str,
    y_col: str = "y_processed",
    *,
    subsample: Optional[int] = None,
    hexbin: bool = True,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    data = df[[bond_col, y_col]].dropna()
    if subsample is not None and len(data) > subsample:
        data_vis = data.sample(subsample, random_state=0)
    else:
        data_vis = data

    res = linregress(data[bond_col], data[y_col])

    fig, ax = plt.subplots(figsize=FIG_SIZE_SCATTER)
    if hexbin:
        hb = ax.hexbin(
            data_vis[bond_col],
            data_vis[y_col],
            gridsize=40,
            cmap="viridis",
            mincnt=3,
        )
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label("Count")
        x = np.linspace(data[bond_col].min(), data[bond_col].max(), 100)
        y = res.intercept + res.slope * x
        ax.plot(x, y, color="red", linewidth=2)
    else:
        sns.regplot(
            data=data_vis,
            x=bond_col,
            y=y_col,
            ax=ax,
            scatter_kws={"alpha": 0.3, "s": 20},
            line_kws={"color": "red"},
        )

    ax.set_xlabel(bond_col)
    ax.set_ylabel(y_col)
    ax.set_title(
        title
        or f"{y_col} vs {bond_col}\n"
           f"R = {res.rvalue:.3f}, p = {res.pvalue:.2e}"
    )
    fig.tight_layout()

    _save_and_show(fig, save_path, show)


def plot_transformation_mapping(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    title: Optional[str] = None,
    kde_contour: bool = True,
    add_binned_trend: bool = True,
    trend_bins: int = 40,
    save_path: Optional[Path] = None,
    show: bool = True,
    context_overwrite: None | str = None,
) -> None:
    """
    Visualize how a transformed score maps from an input column.

    Left panel:
    - scatter of (x_col, y_col)
    - optional binned-median trend line for readability

    Right panel:
    - KDE contour (default) or hexbin density

    Example:
        plot_transformation_mapping(
            hydrogen_bonds_df,
            x_col="DIST_D-A",
            y_col="s_d_gauss",
            title="Gaussian distance score mapping",
            save_path=exp_root / "s_d_gauss_vs_distance",
        )
    """
    if context_overwrite is not None:
        sns.set_context(context_overwrite)

    data = df[[x_col, y_col]].dropna().copy()
    if data.empty:
        raise ValueError(f"No non-null rows found for columns: {x_col}, {y_col}")
    data = data.sort_values(x_col)

    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE_WIDE)

    # Panel 1: direct mapping
    sns.scatterplot(
        data=data,
        x=x_col,
        y=y_col,
        s=10,
        alpha=0.22,
        edgecolor="none",
        ax=axes[0],
    )

    if add_binned_trend:
        n_bins = max(5, min(trend_bins, len(data)))
        binned = data.assign(
            _bin=pd.qcut(data[x_col], q=n_bins, duplicates="drop")
        ).groupby("_bin", observed=True).agg(
            x_mid=(x_col, "median"),
            y_mid=(y_col, "median"),
        )
        axes[0].plot(
            binned["x_mid"],
            binned["y_mid"],
            color="red",
            linewidth=2,
            label="Binned median trend",
        )
        axes[0].legend(frameon=True)

    axes[0].set_title(f"{y_col} vs {x_col}")
    axes[0].set_xlabel(x_col)
    axes[0].set_ylabel(y_col)

    # Panel 2: density view
    if kde_contour:
        sns.kdeplot(
            data=data,
            x=x_col,
            y=y_col,
            fill=True,
            levels=20,
            thresh=0.02,
            cmap="viridis",
            ax=axes[1],
        )
        sns.scatterplot(
            data=data,
            x=x_col,
            y=y_col,
            s=5,
            alpha=0.08,
            color="white",
            edgecolor="none",
            ax=axes[1],
        )
        axes[1].set_title("KDE contour density")
    else:
        hb = axes[1].hexbin(
            data[x_col],
            data[y_col],
            gridsize=40,
            cmap="viridis",
            mincnt=3,
        )
        cbar = fig.colorbar(hb, ax=axes[1])
        cbar.set_label("Count")
        axes[1].set_title("Hexbin density")

    axes[1].set_xlabel(x_col)
    axes[1].set_ylabel(y_col)

    fig.suptitle(title or f"Transformation mapping: {x_col} -> {y_col}", y=1.02)
    fig.tight_layout()

    _save_and_show(fig, save_path, show)
