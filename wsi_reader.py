"""WSI reader built on top of OpenSlide with metadata helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from loguru import logger

try:
    import openslide
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("OpenSlide must be installed for WSI reading") from exc


@dataclass
class SlideInfo:
    """Metadata describing a whole-slide image."""

    path: Path
    dimensions: Tuple[int, int]
    levels: int
    level_downsamples: Tuple[float, ...]
    objective_power: float | None


class WSIReader:
    """Thin wrapper over OpenSlide supporting caching and magnification queries."""

    def __init__(self) -> None:
        self._cache: Dict[Path, openslide.OpenSlide] = {}

    def open(self, path: Path) -> openslide.OpenSlide:
        """Open a slide and cache the handle."""
        if path not in self._cache:
            logger.debug(f"Opening slide {path}")
            self._cache[path] = openslide.OpenSlide(str(path))
        return self._cache[path]

    def info(self, path: Path) -> SlideInfo:
        """Return metadata for a slide."""
        slide = self.open(path)
        dims = slide.dimensions
        downsamples = tuple(float(v) for v in slide.level_downsamples)
        power = _objective_power(slide)
        return SlideInfo(
            path=path,
            dimensions=dims,
            levels=len(downsamples),
            level_downsamples=downsamples,
            objective_power=power,
        )

    def read_region(
        self,
        path: Path,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
    ) -> np.ndarray:
        """Read a region as a numpy array."""
        slide = self.open(path)
        img = slide.read_region(location, level, size)
        arr = np.asarray(img.convert("RGB"))
        return arr

    def close(self) -> None:
        """Close all cached slides."""
        for slide in self._cache.values():
            slide.close()
        self._cache.clear()


def _objective_power(slide: openslide.OpenSlide) -> float | None:
    """Extract objective power if metadata is present."""
    keys = [
        "aperio.AppMag",
        "openslide.mpp-x",
        "tiff.XResolution",
    ]
    for key in keys:
        if key in slide.properties:
            try:
                return float(slide.properties[key])
            except ValueError:  # pragma: no cover - metadata issues
                continue
    return None
