# masking.py
from __future__ import annotations
from typing import Tuple, List
from PIL import Image, ImageDraw
import numpy as np
from loguru import logger
from annotations import Region
from config import ClassMap


def render_mask(
    size_xy: Tuple[int, int],
    regions: List[Region],
) -> np.ndarray:
    """
    Rasterize polygons into a class index mask at given size (W,H).
    Later BG is inferred where mask == 255.
    """
    w, h = size_xy
    mask = Image.new("L", (w, h), color=255)  # 255 marks BG placeholder
    draw = ImageDraw.Draw(mask)
    for r in regions:
        if r.klass not in ClassMap.name_to_idx:
            continue
        cls_idx = ClassMap.name_to_idx[r.klass]
        poly = [(float(x), float(y)) for x, y in r.vertices]
        draw.polygon(poly, outline=cls_idx, fill=cls_idx)
    arr = np.array(mask, dtype=np.int16)
    return arr


def patch_label_from_mask(
    mask: np.ndarray,
    label_min_ratio: float,
) -> int:
    """
    Choose the majority class for a patch.
    If no polygon covers the area, return BG.
    """
    flat = mask.reshape(-1)
    fg = flat[flat != 255]
    if fg.size == 0:
        return ClassMap.name_to_idx["BG"]
    vals, counts = np.unique(fg, return_counts=True)
    best = int(vals[np.argmax(counts)])
    ratio = counts.max() / flat.size
    if ratio < label_min_ratio:
        return ClassMap.name_to_idx["BG"]
    return best
