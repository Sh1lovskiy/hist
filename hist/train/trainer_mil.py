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
    labels_all = list(range(5))

    for features, label, _ in dataset:
        y_true.append(label)
        probs = [0.0 for _ in labels_all]
        if 0 <= label < len(labels_all):
            probs[label] = 1.0
            y_pred.append(label)
        else:
            logger.warning("Encountered label %s outside supported range [0-4]; mapped to BG", label)
            y_pred.append(0)
            probs[0] = 1.0
        y_prob.append(probs)

    missing: List[int] = []
    if y_true:
        present = sorted(set(label for label in y_true if 0 <= label < len(labels_all)))
        missing = [label for label in labels_all if label not in present]
    if missing:
        logger.warning(
            "Fold missing classes: %s. Metrics will treat them as zero-support and log-loss will be nan.",
            ", ".join(str(label) for label in missing),
        )
    metrics = compute_classification_metrics(y_true, y_pred, y_prob, labels=labels_all)
    if missing:
        metrics["log_loss"] = float("nan")

    options.output_dir.mkdir(parents=True, exist_ok=True)
    plot_loss(history, options.output_dir / "plots" / "loss.txt")
    plot_metrics(metrics, options.output_dir / "plots" / "metrics.txt")
    cm = compute_confusion(y_true, y_pred, labels=labels_all)
    plot_confusion(cm, [str(label) for label in labels_all], options.output_dir / "plots" / "confusion.txt")

    metrics["loss_final"] = history[-1] if history else 0.0
    summary_path = options.output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    logger.info("Saved metrics to %s", summary_path)
    return metrics


__all__ = ["TrainOptions", "train_fold"]
