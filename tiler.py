"""Slide tiling utilities producing patch metadata across magnifications."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from loguru import logger
from matplotlib.path import Path as MplPath

from annotations import SlideAnnotation
from wsi_reader import WSIReader


@dataclass
class PatchRecord:
    """Description of a patch in a slide."""

    slide_id: str
    slide_path: Path
    magnification: int
    level: int
    x: int
    y: int
    label: str


class Tiler:
    """Tile whole-slide images into patches."""

    def __init__(
        self,
        reader: WSIReader,
        patch_size: int = 256,
        stride: int | None = None,
        magnifications: Iterable[int] = (40, 10),
    ) -> None:
        self.reader = reader
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.magnifications = list(magnifications)

    def tile_slide(self, slide_path: Path, annotation: SlideAnnotation | None) -> List[PatchRecord]:
        """Tile a slide and return patch metadata."""
        info = self.reader.info(slide_path)
        if info.objective_power is None:
            logger.warning(f"Missing magnification metadata for {slide_path}")
        base_mag = info.objective_power or self.magnifications[0]
        patches: List[PatchRecord] = []
        poly_paths = _build_polygon_paths(annotation)
        for mag in self.magnifications:
            level = _closest_level(info.level_downsamples, base_mag, mag)
            patches.extend(self._tile_level(slide_path, annotation, poly_paths, level, mag))
        return patches

    def _tile_level(
        self,
        slide_path: Path,
        annotation: SlideAnnotation | None,
        poly_paths: Dict[str, List[MplPath]],
        level: int,
        magnification: int,
    ) -> List[PatchRecord]:
        """Tile a single pyramid level."""
        info = self.reader.info(slide_path)
        downsample = info.level_downsamples[level]
        width, height = np.array(info.dimensions) / downsample
        width = int(width)
        height = int(height)
        patches: List[PatchRecord] = []
        for y in range(0, height - self.patch_size + 1, self.stride):
            for x in range(0, width - self.patch_size + 1, self.stride):
                label = self._patch_label(
                    annotation,
                    poly_paths,
                    (x + self.patch_size // 2, y + self.patch_size // 2),
                    downsample,
                )
                patches.append(
                    PatchRecord(
                        slide_id=slide_path.stem,
                        slide_path=slide_path,
                        magnification=magnification,
                        level=level,
                        x=int(x * downsample),
                        y=int(y * downsample),
                        label=label,
                    )
                )
        logger.debug(
            f"Generated {len(patches)} patches for {slide_path.stem} at {magnification}x"
        )
        return patches

    def _patch_label(
        self,
        annotation: SlideAnnotation | None,
        poly_paths: Dict[str, List[MplPath]],
        center_lvl: Tuple[int, int],
        downsample: float,
    ) -> str:
        """Assign label by checking if the patch center is within any polygon."""
        if annotation is None:
            return "background"
        center_base = (center_lvl[0] * downsample, center_lvl[1] * downsample)
        for label, paths in poly_paths.items():
            for path in paths:
                if path.contains_point(center_base):
                    return label
        return "background"


def _build_polygon_paths(annotation: SlideAnnotation | None) -> Dict[str, List[MplPath]]:
    if annotation is None:
        return {}
    mapping: Dict[str, List[MplPath]] = {}
    for poly in annotation.polygons:
        mapping.setdefault(poly.label, []).append(MplPath(poly.vertices))
    return mapping


def _closest_level(downsamples: Tuple[float, ...], base_mag: float, target_mag: int) -> int:
    """Select the pyramid level that matches the target magnification."""
    mags = base_mag / np.array(downsamples)
    idx = int(np.abs(mags - target_mag).argmin())
    return idx
