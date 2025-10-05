"""Extended metrics utilities for multi-class pathology tasks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


@dataclass
class MetricResult:
    accuracy: float
    balanced_accuracy: float
    f1_micro: float
    f1_macro: float
    roc_auc: float | None
    pr_auc: float | None


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    labels: List[int],
) -> MetricResult:
    """Compute a set of robust metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    balanced = balanced_accuracy_score(y_true, y_pred)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    roc_auc = None
    pr_auc = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except ValueError:
            roc_auc = None
        try:
            pr_auc = average_precision_score(y_true, y_prob)
        except ValueError:
            pr_auc = None
    return MetricResult(
        accuracy=accuracy,
        balanced_accuracy=balanced,
        f1_micro=f1_micro,
        f1_macro=f1_macro,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
    )


def confusion_matrix_png(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    path: Path,
) -> None:
    """Save confusion matrix as PNG."""
    import matplotlib.pyplot as plt

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, matrix[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
