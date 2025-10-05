# annotations.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import json
from pathlib import Path
from loguru import logger


@dataclass
class Region:
    klass: str
    vertices: List[Tuple[float, float]]


def load_annotation(json_path: Path) -> List[Region]:
    """
    Load polygonal regions from JSON.
    Each item has keys: "class" and "vertices".
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    regions: List[Region] = []
    for item in data:
        k = item["class"]
        verts = [tuple(map(float, xy)) for xy in item["vertices"]]
        regions.append(Region(klass=k, vertices=verts))
    logger.info(f"Loaded {len(regions)} regions from {json_path.name}")
    return regions


def regions_to_level(
    regions: List[Region],
    level_downsample: float,
) -> List[Region]:
    """
    Scale level-0 vertices to target level by downsample factor.
    """
    scaled: List[Region] = []
    for r in regions:
        vs = [(x / level_downsample, y / level_downsample) for x, y in r.vertices]
        scaled.append(Region(klass=r.klass, vertices=vs))
    return scaled
