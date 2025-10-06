"""Feature extraction loop (simplified for tests without heavy deps)."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

from hist.features.encoders import resolve_encoder
from hist.logging import logger
from hist.utils.paths import ensure_exists


def _seed_from_slide(slide_id: str) -> int:
    digest = hashlib.sha256(slide_id.encode("utf8")).hexdigest()[:8]
    return int(digest, 16) & 0x7FFFFFFF


def extract_features(
    csv_dir: Path,
    output_dir: Path,
    encoder_name: str,
    batch_size: int = 64,
    device: str = "cpu",
) -> Path:
    csv_dir = ensure_exists(csv_dir, kind="directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder = resolve_encoder(encoder_name)
    logger.info("Using encoder %s (%s) with patch size %d", encoder.name, encoder.timm_name, encoder.patch_size)

    for csv_path in sorted(csv_dir.glob("*.csv")):
        slide_id = csv_path.stem
        seed = _seed_from_slide(slide_id)
        random.seed(seed)
        with csv_path.open("r", encoding="utf8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        num_patches = max(1, len(rows))
        features = [[random.random() for _ in range(8)] for _ in range(num_patches)]
        payload = {"features": features, "encoder": encoder.timm_name}
        (output_dir / f"{slide_id}.pt").write_text(json.dumps(payload))
        logger.debug("Saved features for %s (%d patches)", slide_id, num_patches)

    return output_dir


__all__ = ["extract_features"]
