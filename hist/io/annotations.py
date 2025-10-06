"""Annotation parsing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import json

from hist.logging import logger

BACKGROUND_ALIASES = {"bg", "background", "Background", "BG"}


@dataclass
class PolygonAnn:
    label: str
    points: Sequence[Sequence[float]]


@dataclass
class SlideAnnotations:
    slide_id: str
    polygons: List[PolygonAnn]
    class_map: Dict[str, int]


def _normalise_label(label: str) -> str:
    if label.lower() in {alias.lower() for alias in BACKGROUND_ALIASES}:
        return "background"
    return label


def parse_annotation_file(path: Path) -> SlideAnnotations:
    """Parse JSON annotation file into SlideAnnotations."""

    with path.open("r", encoding="utf8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "polygons" in payload:
        polygons_payload = payload["polygons"]
    elif isinstance(payload, list):
        polygons_payload = payload
    else:
        raise ValueError("Annotation JSON must contain a polygon list")

    polygons: List[PolygonAnn] = []
    label_set = set()
    for entry in polygons_payload:
        label = _normalise_label(entry.get("label", "background"))
        points = entry.get("points")
        if not points:
            logger.warning("Polygon without points skipped in %s", path)
            continue
        polygons.append(PolygonAnn(label=label, points=points))
        label_set.add(label)

    class_map = {label: idx for idx, label in enumerate(sorted(label_set))}
    return SlideAnnotations(slide_id=path.stem, polygons=polygons, class_map=class_map)


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
