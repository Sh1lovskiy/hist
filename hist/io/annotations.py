"""Annotation parsing utilities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import json

from hist.logging import logger

BACKGROUND_ALIASES = {"bg", "background"}


@dataclass
class PolygonAnn:
    label: str
    canonical_label: str
    points: Sequence[Sequence[float]]


@dataclass
class SlideAnnotations:
    slide_id: str
    polygons: List[PolygonAnn]
    total_polygons: int
    empty_polygons: int
    class_counts: Counter[str]


def _normalise_label(label: str) -> tuple[str, str]:
    if not label:
        return "background", "BG"
    label_lower = label.lower()
    if label_lower in BACKGROUND_ALIASES:
        return "background", "BG"
    return label, label.upper()


def _extract_polygon_list(payload) -> List[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("objects", "annotations", "polygons", "regions"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("Annotation JSON must contain a polygon list under a known key")


def parse_annotation_file(path: Path) -> SlideAnnotations:
    """Parse JSON annotation file into SlideAnnotations."""

    with path.open("r", encoding="utf8") as handle:
        payload = json.load(handle)

    polygons_payload = _extract_polygon_list(payload)

    polygons: List[PolygonAnn] = []
    class_counts: Counter[str] = Counter()
    empty_polygons = 0
    for entry in polygons_payload:
        label_raw = entry.get("class") or entry.get("label") or entry.get("type")
        raw_label, canonical = _normalise_label(str(label_raw) if label_raw is not None else "")
        points = entry.get("vertices") or entry.get("points") or entry.get("coords")
        if not points or len(points) < 3:
            empty_polygons += 1
            continue
        polygons.append(PolygonAnn(label=raw_label, canonical_label=canonical, points=points))
        class_counts[canonical] += 1

    total_polygons = len(polygons_payload)
    used_polygons = len(polygons)
    if empty_polygons:
        logger.warning("%s: skipped %d empty polygons", path.name, empty_polygons)
    classes_repr = ", ".join(f"{label}:{count}" for label, count in sorted(class_counts.items()))
    logger.info(
        "%s: polys=%d, empty=%d, used=%d, classes={%s}",
        path.name,
        total_polygons,
        empty_polygons,
        used_polygons,
        classes_repr,
    )
    return SlideAnnotations(
        slide_id=path.stem,
        polygons=polygons,
        total_polygons=total_polygons,
        empty_polygons=empty_polygons,
        class_counts=class_counts,
    )


def load_annotations(directory: Path) -> Dict[str, SlideAnnotations]:
    annotations: Dict[str, SlideAnnotations] = {}
    for path in sorted(directory.glob("*.json")):
        ann = parse_annotation_file(path)
        annotations[ann.slide_id] = ann
    return annotations


__all__ = [
    "PolygonAnn",
    "SlideAnnotations",
    "parse_annotation_file",
    "load_annotations",
    "BACKGROUND_ALIASES",
]
