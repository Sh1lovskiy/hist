"""Utility helpers for dealing with file-system paths."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional


def ensure_exists(path: Path, kind: str = "file") -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Expected {kind} at '{path}'")
    return path


def create_timestamped_dir(base: Path, name: Optional[str] = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if name:
        run_dir = base / f"{name}_{timestamp}"
    else:
        run_dir = base / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def symlink_latest(run_dir: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(run_dir)


__all__ = ["ensure_exists", "create_timestamped_dir", "symlink_latest"]
