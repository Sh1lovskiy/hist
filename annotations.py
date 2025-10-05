"""Utilities to read polygon annotations exported from digital pathology tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List
import json

import numpy as np


@dataclass
class PolygonAnnotation:
    """Simple polygon with a label and list of vertices."""

    label: str
    vertices: np.ndarray


@dataclass
class SlideAnnotation:
    """Collection of polygons per slide."""

    slide_id: str
    polygons: List[PolygonAnnotation]


from typing import Dict, List, Sequence, Tuple, Any

from loguru import logger

Point = Tuple[int, int]


@dataclass
class PolygonAnn:
    """Single polygon annotation."""

    label: str
    points: List[Point]


class AnnotationParser:
    """
    Robust parser for polygon JSON annotations.

    Supported roots:
      - dict with 'objects' OR 'shapes' OR 'annotations'
      - list of objects

    Supported fields per object:
      - label: 'label' | 'class' | 'name' | 'category'
      - points: 'points' | 'vertices' | 'polygon'
        * each point can be [x, y] or {'x': x, 'y': y}
    """

    def __init__(self) -> None:
        # filled as labels are encountered
        self.class_map: Dict[str, int] = {}

    # ---------- public API ----------

    def parse_dir(self, ann_dir: Path) -> Dict[str, List[PolygonAnn]]:
        """Return mapping: slide_stem -> list[PolygonAnn]."""
        out: Dict[str, List[PolygonAnn]] = {}
        for p in sorted(ann_dir.glob("*.json")):
            try:
                anns = self.parse_file(p)
                out[p.stem] = anns
            except Exception as e:
                logger.error("Failed to parse {}: {}", p, e)
        logger.info("Loaded annotations for {} slides", len(out))
        return out

    def parse_file(self, path: Path) -> List[PolygonAnn]:
        """Parse a single JSON file into a list of PolygonAnn."""
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # normalize root -> list of objects
        objs: Sequence[Any]
        if isinstance(data, list):
            objs = data
        elif isinstance(data, dict):
            objs = (
                data.get("objects")
                or data.get("shapes")
                or data.get("annotations")
                or []
            )
        else:
            logger.warning("Unsupported JSON root in {}: {}", path, type(data))
            objs = []

        if not objs:
            logger.warning("No objects found in {}", path)

        polygons: List[PolygonAnn] = []
        for obj in objs:
            poly = self._parse_obj(obj)
            if poly is None or len(poly.points) < 3:
                continue
            polygons.append(poly)
            self._ensure_class(poly.label)

        logger.debug("Parsed {} polygons from {}", len(polygons), path.name)
        return polygons

    # ---------- internals ----------

    def _parse_obj(self, obj: Any) -> PolygonAnn | None:
        if not isinstance(obj, dict):
            # some tools may store raw list of points
            pts = self._as_points(obj)
            if pts:
                return PolygonAnn(label="foreground", points=pts)
            return None

        label = (
            obj.get("label")
            or obj.get("class")
            or obj.get("name")
            or obj.get("category")
            or "foreground"
        )

        raw_pts = (
            obj.get("points")
            or obj.get("vertices")
            or obj.get("polygon")
            or obj.get("geometry")
        )
        pts = self._as_points(raw_pts)
        if not pts or len(pts) < 3:
            return None
        return PolygonAnn(label=label, points=pts)

    def _as_points(self, raw: Any) -> List[Point]:
        """Normalize any supported points representation -> list[(x,y)]."""
        pts: List[Point] = []
        if raw is None:
            return pts

        # common cases:
        # 1) [[x,y], [x,y], ...]
        # 2) [{'x': x, 'y': y}, ...]
        # 3) {'points': [...]} already handled before, but keep guard
        if isinstance(raw, dict):
            cand = raw.get("points") or raw.get("vertices") or raw.get("polygon")
            if cand is None:
                return pts
            raw = cand

        if isinstance(raw, list):
            for p in raw:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    x, y = int(round(p[0])), int(round(p[1]))
                    pts.append((x, y))
                elif isinstance(p, dict):
                    if "x" in p and "y" in p:
                        x, y = int(round(p["x"])), int(round(p["y"]))
                        pts.append((x, y))
                    elif "X" in p and "Y" in p:
                        x, y = int(round(p["X"])), int(round(p["Y"]))
                        pts.append((x, y))
        return pts

    def _ensure_class(self, label: str) -> None:
        if label not in self.class_map:
            self.class_map[label] = len(self.class_map)


def rasterize_polygons(
    polygons: Iterable[PolygonAnnotation], level_downsample: float
) -> List[np.ndarray]:
    """Convert polygons into integer coordinates at a given level."""
    coords: List[np.ndarray] = []
    for poly in polygons:
        scaled = (poly.vertices / level_downsample).astype(int)
        coords.append(scaled)
    return coords
