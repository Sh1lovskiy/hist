"""Cross-validation orchestration."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from hist.data.datasets import load_slide_entries
from hist.logging import logger
from hist.train.trainer_mil import TrainOptions, train_fold


def _chunk_indices(n: int, k: int) -> List[List[int]]:
    k = max(1, min(k, n))
    folds: List[List[int]] = [[] for _ in range(k)]
    for idx in range(n):
        folds[idx % k].append(idx)
    return folds


@dataclass
class CVOptions:
    features_dir: Path
    labels: Dict[str, int]
    output_dir: Path
    model_name: str = "mean"
    epochs: int = 1
    k_folds: int = 2
    device: str = "cpu"
    oversample: bool = False


def run_cv(options: CVOptions) -> Path:
    entries = load_slide_entries(options.features_dir, options.labels)
    folds = _chunk_indices(len(entries), options.k_folds)
    summary_rows: List[Dict[str, float]] = []
    for fold_idx, fold in enumerate(folds):
        fold_entries = [entries[i] for i in fold]
        fold_dir = options.output_dir / f"fold_{fold_idx}"
        train_opts = TrainOptions(
            features_dir=options.features_dir,
            labels=options.labels,
            output_dir=fold_dir,
            model_name=options.model_name,
            epochs=options.epochs,
            device=options.device,
        )
        metrics = train_fold(fold_entries, train_opts)
        metrics["fold"] = fold_idx
        summary_rows.append(metrics)

    output_summary = options.output_dir / "summary.csv"
    if summary_rows:
        fieldnames = sorted({key for row in summary_rows for key in row.keys()})
        options.output_dir.mkdir(parents=True, exist_ok=True)
        with output_summary.open("w", newline="", encoding="utf8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
    logger.info("Cross-validation summary saved to %s", output_summary)
    return output_summary


__all__ = ["CVOptions", "run_cv"]
