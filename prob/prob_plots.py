"""
Plotting helpers for probing results (parity, residuals).

Optional dependency: matplotlib/seaborn. Safe to import; plotting is only
performed when functions are called, so headless runs can skip them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Scatter of predicted vs true with diagonal reference line."""
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.figure(figsize=(6, 6))
    g = sns.JointGrid(x=y_true, y=y_pred, space=0)
    g.plot_joint(sns.scatterplot, alpha=0.5, s=12)
    g.plot_marginals(sns.histplot, bins=30, fill=True)
    g.ax_joint.plot(lims, lims, "r--", linewidth=1)
    g.set_axis_labels("True", "Predicted")
    g.fig.suptitle(title, y=1.02)
    g.ax_joint.set_xlim(lims)
    g.ax_joint.set_ylim(lims)
    if save_path is not None:
        g.fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Histogram of residuals (y_pred - y_true)."""
    residuals = y_pred - y_true
    plt.figure(figsize=(5.5, 4))
    sns.histplot(residuals, bins=50, kde=True)
    plt.axvline(0, color="r", linestyle="--", linewidth=1)
    plt.xlabel("Residual (y_pred - y_true)")
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close()
