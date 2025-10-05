"""Logging utilities using loguru.

This module centralizes logging configuration so that every CLI entry point can
share the same structured logger. Log files are rotated per run and written to
an experiment directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from loguru import logger


def configure_logging(output_dir: Path, verbose: Literal["info", "debug"] = "info") -> None:
    """Configure loguru handlers and formatting."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    level = "DEBUG" if verbose.lower() == "debug" else "INFO"
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(lambda msg: print(msg, end=""), level=level, format=fmt)
    logfile = output_dir / "train.log"
    logger.add(logfile, level=level, format=fmt, enqueue=True)
