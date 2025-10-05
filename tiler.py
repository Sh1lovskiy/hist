# tiler.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Tuple
import numpy as np


@dataclass
class Grid:
    w: int
    h: int
    patch: int
    stride: int

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        for y in range(0, self.h - self.patch + 1, self.stride):
            for x in range(0, self.w - self.patch + 1, self.stride):
                yield x, y


def is_enough_foreground(img_rgb, min_fg_ratio: float) -> bool:
    import cv2
    import numpy as np

    rgb = np.array(img_rgb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1] / 255.0
    ratio = float((sat > 0.08).mean())
    return ratio >= min_fg_ratio
