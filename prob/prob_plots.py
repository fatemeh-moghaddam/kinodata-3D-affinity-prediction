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
    - Otherwise, interpret save_path as "<experiment_root>/<filename_stem>[.ext]"
      and always save under a "figures" subdirectory with a ".png" suffix.

    Example:
        save_path = exp_root / "parity_layer3"
        -> exp_root / "figures" / "parity_layer3.png"
    """
    if save_path is None:
        return None

    p = Path(save_path)
    parent = p.parent
    name = p.name

    parent = parent / "figures"
    parent.mkdir(parents=True, exist_ok=True)

    if not name.endswith(".png"):
        name = f"{name}.png"

    return parent / name


def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Optional[Path] = None,
    show: bool = True,
    context_overwrite: None | str = None,
) -> None:
    """Scatter of predicted vs true with diagonal reference line."""
    if context_overwrite is not None:
        sns.set_context(context_overwrite)

    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]

    g = sns.JointGrid(x=y_true, y=y_pred, space=0, height=FIG_SIZE_SQUARE[0])
    g.plot_joint(sns.scatterplot, alpha=0.4, s=16, edgecolor="none")
    g.plot_marginals(sns.histplot, bins=30, fill=True, element="step")
    g.ax_joint.plot(lims, lims, "r--", linewidth=1.3)
    g.set_axis_labels("True", "Predicted")
    g.fig.suptitle(title, y=1.02)
    g.ax_joint.set_xlim(lims)
    g.ax_joint.set_ylim(lims)

    out_path = _resolve_save_path(save_path)
    if out_path is not None:
        g.fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(g.fig)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Histogram of residuals (y_pred - y_true)."""
    residuals = y_pred - y_true

    fig, ax = plt.subplots(figsize=(5.7, 4.3))
    sns.histplot(residuals, bins=60, kde=True, ax=ax, color="C0", alpha=0.8)
    ax.axvline(0, color="r", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Residual (y_pred - y_true)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()

    out_path = _resolve_save_path(save_path)
    if out_path is not None:
        fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


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
    axes[1].set_title(f"{title or col} Distribution (log-scaled)")

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

    out_path = _resolve_save_path(save_path)
    if out_path is not None:
        fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


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

    out_path = _resolve_save_path(save_path)
    if out_path is not None:
        fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)