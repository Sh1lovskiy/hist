"""Logging helpers for the hist pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"


def configure_logging(log_dir: Optional[Path] = None, level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=LOG_FORMAT)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_dir / "hist.log")
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logging.getLogger().addHandler(handler)


def set_verbosity(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)


logger = logging.getLogger("hist")

__all__ = ["configure_logging", "set_verbosity", "LOG_FORMAT", "logger"]
