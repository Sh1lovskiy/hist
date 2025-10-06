"""Attention export utilities (simplified)."""

from __future__ import annotations

import csv
from pathlib import Path

from hist.logging import logger


def export_dummy_attention(features_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for feature_path in features_dir.glob("*.pt"):
        slide_id = feature_path.stem
        csv_path = output_dir / f"{slide_id}_attention.csv"
        with csv_path.open("w", newline="", encoding="utf8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["index", "attention"])
            for idx in range(10):
                writer.writerow([idx, idx / 9 if 9 else 0])
        logger.info("Wrote attention csv %s", csv_path)
    return output_dir


__all__ = ["export_dummy_attention"]
