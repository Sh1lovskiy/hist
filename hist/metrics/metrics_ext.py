"""Safe metric computations for small datasets."""

from __future__ import annotations

import math
from typing import Dict, Sequence

from hist.logging import logger


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[Sequence[float]],
    labels: Sequence[int],
) -> Dict[str, float]:
    total = len(y_true)
    accuracy = sum(int(t == p) for t, p in zip(y_true, y_pred)) / total if total else 0.0
    recalls = []
    precisions = []
    f1_scores = []
    for label in labels:
        tp = sum(int(t == label and p == label) for t, p in zip(y_true, y_pred))
        fn = sum(int(t == label and p != label) for t, p in zip(y_true, y_pred))
        fp = sum(int(t != label and p == label) for t, p in zip(y_true, y_pred))
        recall = _safe_div(tp, tp + fn)
        precision = _safe_div(tp, tp + fp)
        f1 = _safe_div(2 * precision * recall, precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1)
    balanced_accuracy = sum(recalls) / len(recalls) if recalls else 0.0
    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_macro": sum(precisions) / len(precisions) if precisions else 0.0,
        "recall_macro": sum(recalls) / len(recalls) if recalls else 0.0,
        "f1_macro": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
    }

    # log-loss (safe)
    try:
        loss = 0.0
        for target, probs in zip(y_true, y_prob):
            prob = max(min(probs[target], 1 - 1e-9), 1e-9)
            loss -= math.log(prob)
        metrics["log_loss"] = loss / total if total else float("nan")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Log loss undefined: %s", exc)
        metrics["log_loss"] = float("nan")

    return metrics


def compute_confusion(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int]):
    size = len(labels)
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    for truth, pred in zip(y_true, y_pred):
        i = label_to_idx[truth]
        j = label_to_idx[pred]
        matrix[i][j] += 1
    return matrix


__all__ = ["compute_classification_metrics", "compute_confusion"]
