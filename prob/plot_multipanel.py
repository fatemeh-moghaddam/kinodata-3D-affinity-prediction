"""
Multi-panel figures for probing experiments, from saved predictions only.

Everything that already exists in `prob.plots` is imported, not rewritten.
The only genuinely new piece is `parity_on_ax`: `plot_parity` builds its own
Figure via seaborn's JointGrid, which cannot be placed into a subplot cell,
so the joint-panel drawing has to exist in an ax-level form.

Reuses from prob.plots:
    _annotate_metrics, _save_and_show, plot_conditions_box (per-sample boxes)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

from prob.prob_plots import _annotate_metrics, _save_and_show


# ─────────────────────────────────────────────────────────────
# Ax-level parity: the one thing plot_parity cannot give us
# ─────────────────────────────────────────────────────────────

def parity_on_ax(
    ax,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: Optional[str] = None,
    lims: Optional[Sequence[float]] = None,
    density: Optional[str] = "kde",
    density_cmap: str = "Blues_r",
    show_mean_line: bool = True,
    show_metrics: bool = True,
    metrics_loc: str = "lower right",
    marginals: bool = False,
    scatter: bool = True,
    colorbar: bool = False,
) -> None:
    """`plot_parity`'s joint panel, drawn into an existing axes.

    marginals: attach histogram axes top/right via make_axes_locatable,
        reproducing the JointGrid look inside a subplot grid.
    lims: pass shared limits so panels are directly comparable.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if lims is None:
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    lims = list(lims)

    if scatter:
        ax.scatter(y_true, y_pred, alpha=0.5, s=16,
                   facecolors="none", edgecolors="black", linewidths=0.4)

    if density == "hexbin":
        hb = ax.hexbin(y_true, y_pred, gridsize=80, cmap=density_cmap,
                       mincnt=1, extent=lims + lims, alpha=0.85)
        if colorbar:
            ax.figure.colorbar(hb, ax=ax, label="count", fraction=0.046, pad=0.04)
    elif density == "kde":
        sns.kdeplot(x=y_true, y=y_pred, fill=True, cmap=density_cmap,
                    thresh=0.02, levels=10, alpha=0.7, ax=ax)
    elif density is not None:
        raise ValueError(f"Unknown density mode: {density}")

    ax.plot(lims, lims, "r--", linewidth=1.3, zorder=5)

    if show_mean_line:
        mean_true = float(y_true.mean())
        ax.axhline(mean_true, color="gray", linestyle=":", linewidth=1.7,
                   label=f"mean(y_true)={mean_true:.2f}", zorder=5)
        ax.legend(loc="upper left", fontsize=7, frameon=False)

    if show_metrics:
        _annotate_metrics(ax, y_true, y_pred, loc=metrics_loc)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    if title:
        ax.set_title(title, fontsize=10)

    if marginals:
        div = make_axes_locatable(ax)
        ax_top = div.append_axes("top", size="18%", pad=0.05, sharex=ax)
        ax_right = div.append_axes("right", size="18%", pad=0.05, sharey=ax)
        sns.histplot(x=y_true, bins=30, element="step", fill=True, ax=ax_top)
        sns.histplot(y=y_pred, bins=30, element="step", fill=True, ax=ax_right)
        for a in (ax_top, ax_right):
            a.set_xlabel("")
            a.set_ylabel("")
            a.tick_params(labelbottom=False, labelleft=False)
            a.grid(False)
        if title:
            ax.set_title("")
            ax_top.set_title(title, fontsize=10)


def plot_parity_grid(
    panels: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    ncols: int = 3,
    panel_size: float = 4.0,
    shared_lims: bool = True,
    suptitle: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
    row_labels: Optional[Sequence[str]] = None,
    **panel_kwargs,
) -> None:
    """Grid of parity panels from saved predictions.

    panels: ordered mapping  panel_title -> (y_true, y_pred).
    shared_lims: one common range across panels, so shrinkage toward the
        mean is visually comparable between conditions.
    **panel_kwargs: forwarded to parity_on_ax.
    """
    items = list(panels.items())
    n = len(items)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    lims = None
    if shared_lims:
        lo = min(min(np.min(t), np.min(p)) for _, (t, p) in items)
        hi = max(max(np.max(t), np.max(p)) for _, (t, p) in items)
        lims = [lo, hi]

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_size * ncols, panel_size * nrows),
                             squeeze=False)

    for idx, (title, (y_true, y_pred)) in enumerate(items):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        parity_on_ax(ax, y_true, y_pred, title=title, lims=lims, **panel_kwargs)
        if c != 0:
            ax.set_ylabel("")
        if r != nrows - 1:
            ax.set_xlabel("")

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    if row_labels is not None:
        for r, lab in enumerate(row_labels[:nrows]):
            axes[r][0].annotate(lab, xy=(-0.28, 0.5), xycoords="axes fraction",
                                rotation=90, ha="center", va="center", fontsize=11)

    if suptitle:
        fig.suptitle(suptitle, y=1.0, fontsize=13)
    fig.tight_layout()
    _save_and_show(fig, save_path, show)


# ─────────────────────────────────────────────────────────────
# Per-run metric box plots (Fig-10 style: one dot per fold/seed)
# ─────────────────────────────────────────────────────────────

METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "r2": lambda t, p: float(1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2)),
    "rmse": lambda t, p: float(np.sqrt(np.mean((p - t) ** 2))),
    "mae": lambda t, p: float(np.mean(np.abs(p - t))),
    "pearson": lambda t, p: float(np.corrcoef(t, p)[0, 1]),
}


def summarize_runs(runs: pd.DataFrame, metrics=("r2", "rmse", "mae", "pearson")):
    """runs: table with y_true / y_pred object columns + factor columns.

    Swap METRICS for whatever your significance module already computes if
    you'd rather have a single source of truth for metric definitions.
    """
    out = runs.copy()
    for m in metrics:
        fn = METRICS[m]
        out[m] = [fn(np.asarray(t, float), np.asarray(p, float))
                  for t, p in zip(runs["y_true"], runs["y_pred"])]
    return out.drop(columns=["y_true", "y_pred"])


def plot_metric_box(
    summary: pd.DataFrame,
    *,
    x: str,
    y: str = "r2",
    hue: Optional[str] = None,
    col: Optional[str] = None,
    order: Optional[Sequence[str]] = None,
    hue_order: Optional[Sequence[str]] = None,
    show_points: bool = True,
    hline: Optional[float] = 0.0,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """One box per (x, hue) cell, one point per run. `col` facets sideways.

    hline: reference level, e.g. 0.0 for R2 or the shuffled-control value.
    """
    facets = [None] if col is None else list(summary[col].drop_duplicates())
    fig, axes = plt.subplots(
        1, len(facets),
        figsize=(max(4.5, 1.5 * summary[x].nunique()) * len(facets), 5.0),
        squeeze=False, sharey=True,
    )

    for i, facet in enumerate(facets):
        ax = axes[0][i]
        data = summary if facet is None else summary[summary[col] == facet]
        sns.boxplot(data=data, x=x, y=y, hue=hue, order=order,
                    hue_order=hue_order, width=0.6, fliersize=0, ax=ax)
        if show_points:
            sns.stripplot(data=data, x=x, y=y, hue=hue, order=order,
                          hue_order=hue_order, dodge=hue is not None,
                          size=4, color="0.25", alpha=0.8, ax=ax, legend=False)
        if hue is not None and i > 0 and ax.get_legend() is not None:
            ax.get_legend().remove()
        if hline is not None:
            ax.axhline(hline, color="r", linestyle="--", linewidth=1.0)
        ax.set_ylabel(y.upper() if i == 0 else "")
        ax.set_xlabel(x)
        if facet is not None:
            ax.set_title(f"{col} = {facet}", fontsize=11)

    if title:
        fig.suptitle(title, y=1.02, fontsize=13)
    fig.tight_layout()
    _save_and_show(fig, save_path, show)
