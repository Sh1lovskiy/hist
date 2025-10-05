from __future__ import annotations
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt


def save_curve(values: List[float], title: str, out_path: Path) -> None:
    """
    Save a simple curve (e.g., loss per epoch) as PNG, dpi=300,
    no padding, dashed semi-transparent grid.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(values) + 1), values, marker="o")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout(pad=0.0)
    plt.savefig(out_path.as_posix(), dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()
