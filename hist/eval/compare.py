"""Aggregate experiment results."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

from hist.logging import logger


def compare_runs(run_dirs: Iterable[Path], output: Path, metrics: Iterable[str]) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, str]] = []
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.csv"
        if not summary_path.exists():
            logger.warning("Missing summary.csv in %s", run_dir)
            continue
        with summary_path.open("r", encoding="utf8") as handle:
            reader = csv.DictReader(handle)
            values = list(reader)
        if not values:
            continue
        # assume final row is overall metrics
        final = values[-1]
        row = {"run": run_dir.name}
        for metric in metrics:
            if metric in final:
                row[metric] = final[metric]
        rows.append(row)

    output_path = output / "comparison.csv"
    if rows:
        with output_path.open("w", newline="", encoding="utf8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row.keys()}))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Saved comparison to %s", output_path)
    else:
        logger.warning("No runs to compare")
    return output_path


__all__ = ["compare_runs"]
