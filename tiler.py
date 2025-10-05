# tiler.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from loguru import logger
from matplotlib.path import Path as MplPath
from openslide import OpenSlide

from annotations import PolygonAnn
from wsi_reader import WSIReader


@dataclass
class PatchRecord:
    """Patch metadata row used to write CSV."""

    slide_id: str
    slide_path: Path
    magnification: int
    level: int
    x: int
    y: int
    label: str


def _build_polygon_paths(polys: List[PolygonAnn]) -> List[Tuple[str, MplPath]]:
    """Convert polygons to (label, MplPath)."""
    paths: List[Tuple[str, MplPath]] = []
    for p in polys:
        pts = np.asarray(p.points, dtype=float)
        if pts.shape[0] >= 3:
            paths.append((p.label, MplPath(pts)))
    return paths


def _label_by_center(x: int, y: int, ps: int, paths: List[Tuple[str, MplPath]]) -> str:
    """Assign label if patch center lies inside any polygon, else 'background'."""
    cx, cy = x + ps * 0.5, y + ps * 0.5
    for lab, path in paths:
        if path.contains_points([[cx, cy]])[0]:
            return lab
    return "background"


class Tiler:
    """Grid tiler at level-0; labels by center-in-polygon rule."""

    def __init__(self, reader: WSIReader, patch_size: int, magnifications):
        self.reader = reader
        self.patch_size = int(patch_size)
        self.magnifications = list(magnifications)
        self.level = 0
        self.default_mag = max(self.magnifications) if self.magnifications else 40

    def _level0_dims(self, slide_path: Path) -> tuple[int, int]:
        with OpenSlide(str(slide_path)) as s:
            w, h = s.level_dimensions[self.level]
        return int(w), int(h)

    def tile_slide(
        self, slide_path: Path, polys: List[PolygonAnn]
    ) -> List[PatchRecord]:
        """Tile slide into patches and assign labels."""
        slide_id = slide_path.stem
        w, h = self._level0_dims(slide_path)
        ps = self.patch_size
        stride = ps
        paths = _build_polygon_paths(polys)

        rows: List[PatchRecord] = []
        for y in range(0, max(0, h - ps + 1), stride):
            for x in range(0, max(0, w - ps + 1), stride):
                lab = _label_by_center(x, y, ps, paths)
                rows.append(
                    PatchRecord(
                        slide_id=slide_id,
                        slide_path=slide_path,
                        magnification=self.default_mag,
                        level=self.level,
                        x=x,
                        y=y,
                        label=lab,
                    )
                )
        logger.info(
            "Tiled {}: {} patches (ps={}, stride={})",
            slide_id,
            len(rows),
            ps,
            stride,
        )
        return rows
