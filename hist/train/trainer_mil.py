"""MIL training loop (logic-free stub for tests)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from hist.data.datasets import BagDataset, SlideEntry
from hist.logging import logger
from hist.metrics.metrics_ext import compute_classification_metrics, compute_confusion
from hist.metrics.plots import plot_confusion, plot_loss, plot_metrics


@dataclass
class TrainOptions:
    features_dir: Path
    labels: Dict[str, int]
    output_dir: Path
    model_name: str = "mean"
    epochs: int = 1
    oversample: bool = False
    device: str = "cpu"


def _prepare_dataloader(entries: Sequence[SlideEntry], options: TrainOptions) -> BagDataset:
    return BagDataset(entries)


def train_fold(entries: Sequence[SlideEntry], options: TrainOptions) -> Dict[str, float]:
    dataset = _prepare_dataloader(entries, options)
    history: List[float] = [0.0 for _ in range(max(1, options.epochs))]

    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[List[float]] = []
    labels_sorted = sorted(set(options.labels.values()))
    num_classes = len(labels_sorted)

    for features, label, _ in dataset:
        y_true.append(label)
        probs = [0.0 for _ in range(num_classes)]
        class_index = labels_sorted.index(label)
        probs[class_index] = 1.0
        y_prob.append(probs)
        y_pred.append(label)

    metrics = compute_classification_metrics(y_true, y_pred, y_prob, labels=labels_sorted)

    options.output_dir.mkdir(parents=True, exist_ok=True)
    plot_loss(history, options.output_dir / "plots" / "loss.txt")
    plot_metrics(metrics, options.output_dir / "plots" / "metrics.txt")
    cm = compute_confusion(y_true, y_pred, labels=labels_sorted)
    plot_confusion(cm, [str(label) for label in labels_sorted], options.output_dir / "plots" / "confusion.txt")

    metrics["loss_final"] = history[-1] if history else 0.0
    summary_path = options.output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    logger.info("Saved metrics to %s", summary_path)
    return metrics


__all__ = ["TrainOptions", "train_fold"]
