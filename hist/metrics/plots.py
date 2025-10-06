"""Text-based plotting stubs for testing environments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence


def plot_loss(history: Sequence[float], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(str(value) for value in history))


def plot_confusion(cm, labels: Sequence[str], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(labels)]
    for row in cm:
        lines.append(",".join(str(item) for item in row))
    output.write_text("\n".join(lines))


def plot_metrics(metrics: Dict[str, float], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key},{value}" for key, value in metrics.items()]
    output.write_text("\n".join(lines))


__all__ = ["plot_loss", "plot_confusion", "plot_metrics"]
