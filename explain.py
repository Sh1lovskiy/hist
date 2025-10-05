# explain.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import csv


def save_attention_csv(
    slide_id: str,
    level: str,
    coords: List[Tuple[int, int]],
    weights: np.ndarray,
    out_csv: Path,
) -> None:
    """
    Save per-patch attention weights for a slide-level heatmap.
    CSV: slide_id,level,x,y,weight
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["slide_id", "level", "x", "y", "weight"])
        for (x, y), w in zip(coords, weights.tolist()):
            wr.writerow([slide_id, level, int(x), int(y), float(w)])
