"""Plotting utilities for training curves and metrics."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def plot_losses(history: Dict[str, List[float]], path: Path) -> None:
    """Plot train/validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    for split, values in history.items():
        ax.plot(values, label=split, alpha=0.7, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(metrics: Dict[str, List[float]], path: Path) -> None:
    """Plot multiple metrics curves on the same figure."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    for name, values in metrics.items():
        ax.plot(values, label=name, alpha=0.7, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
