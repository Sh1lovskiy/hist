"""Dataset helpers for lightweight training stubs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

from hist.logging import logger
from hist.utils.paths import ensure_exists


@dataclass
class SlideEntry:
    """Represents a single slide with features on disk."""

    slide_id: str
    features_path: Path
    label: int


def load_slide_entries(features_dir: Path, labels: Dict[str, int]) -> List[SlideEntry]:
    """Pair feature files with labels.

    Missing feature files are reported and skipped. Raises if nothing matches
    so downstream stages fail fast with a clear message.
    """

    features_dir = ensure_exists(features_dir, kind="directory")
    entries: List[SlideEntry] = []
    missing: List[str] = []

    for slide_id, label in sorted(labels.items()):
        feature_path = features_dir / f"{slide_id}.pt"
        if not feature_path.exists():
            missing.append(slide_id)
            continue
        entries.append(SlideEntry(slide_id=slide_id, features_path=feature_path, label=label))

    if missing:
        logger.warning("Missing features for slides: %s", ", ".join(missing))
    if not entries:
        raise FileNotFoundError("No feature files matched provided labels")

    return entries


class BagDataset:
    """Tiny iterable returning (features, label, meta) tuples."""

    def __init__(self, entries: Sequence[SlideEntry]):
        self._entries = list(entries)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._entries)

    def __iter__(self) -> Iterator[Tuple[List[List[float]], int, Dict[str, str]]]:
        for entry in self._entries:
            features = _load_features(entry.features_path)
            yield features, entry.label, {"slide_id": entry.slide_id}


def _load_features(path: Path) -> List[List[float]]:
    payload = json.loads(path.read_text())
    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError(f"Expected 'features' list in {path}")
    return [list(map(float, vector)) for vector in features]


__all__ = ["SlideEntry", "BagDataset", "load_slide_entries"]
