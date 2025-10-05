# wsi_reader.py
from __future__ import annotations
from typing import Tuple
from loguru import logger
import openslide


def open_wsi(path: str) -> openslide.OpenSlide:
    """Open a WSI via OpenSlide."""
    slide = openslide.OpenSlide(path)
    return slide


def get_level_dims(slide: openslide.OpenSlide) -> Tuple[Tuple[int, int], ...]:
    """Return level dimensions for all pyramid levels."""
    return slide.level_dimensions


def get_level_downsamples(slide: openslide.OpenSlide) -> Tuple[float, ...]:
    """Return level downsample factors (relative to level 0)."""
    return slide.level_downsamples


def get_mpp_mag(slide: openslide.OpenSlide) -> Tuple[float, float, float]:
    """
    Return (mpp_x, mpp_y, nominal_mag) when available.
    """
    props = slide.properties
    mpp_x = float(props.get("openslide.mpp-x", 0) or 0)
    mpp_y = float(props.get("openslide.mpp-y", 0) or 0)
    mag = float(props.get("openslide.objective-power", 0) or 0)
    logger.info(f"MPP x={mpp_x}, y={mpp_y}, nominal mag={mag}")
    return mpp_x, mpp_y, mag


def read_region_rgb(
    slide: openslide.OpenSlide,
    level: int,
    x: int,
    y: int,
    w: int,
    h: int,
):
    """
    Read an RGB patch at a pyramid level.
    Coordinates (x,y) are at that level's coordinate system.
    """
    ds = slide.level_downsamples[level]
    loc = (int(x * ds), int(y * ds))
    img = slide.read_region(loc, level, (w, h)).convert("RGB")
    return img
