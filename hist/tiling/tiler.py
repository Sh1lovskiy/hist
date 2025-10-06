"""Simple tiler producing patch coordinates CSVs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from hist.logging import logger

from hist.io.annotations import SlideAnnotations, load_annotations
from hist.utils.paths import ensure_exists


@dataclass
class TileRequest:
    slides_dir: Path
    annotations_dir: Path
    output_dir: Path
    patch_size: int
    stride: int
    overwrite: bool = False


HEADER = ["slide_id", "x", "y", "label"]


def tile_dataset(request: TileRequest) -> Path:
    slides_dir = ensure_exists(request.slides_dir, kind="directory")
    annotations_dir = ensure_exists(request.annotations_dir, kind="directory")
    output_dir = request.output_dir / "patch_csvs"
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_annotations(annotations_dir)
    csv_paths: List[Path] = []

    for slide_path in sorted(slides_dir.iterdir()):
        if slide_path.is_dir():
            continue
        slide_id = slide_path.stem
        csv_path = output_dir / f"{slide_id}.csv"
        if csv_path.exists() and not request.overwrite:
            logger.info("Skipping %s (exists)", csv_path)
            csv_paths.append(csv_path)
            continue
        slide_annotations = annotations.get(slide_id)
        if slide_annotations is None:
            logger.warning("No annotations for slide %s", slide_id)
            polygons: List[SlideAnnotations] = []
        with csv_path.open("w", newline="", encoding="utf8") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            if slide_annotations is None:
                writer.writerow([slide_id, 0, 0, "background"])
            else:
                # produce one patch per polygon centroid
                for polygon in slide_annotations.polygons:
                    xs = [point[0] for point in polygon.points]
                    ys = [point[1] for point in polygon.points]
                    cx = int(sum(xs) / len(xs))
                    cy = int(sum(ys) / len(ys))
                    writer.writerow([slide_id, cx, cy, polygon.label])
        logger.info("Created %s", csv_path)
        csv_paths.append(csv_path)

    logger.info("Generated %d patch CSVs", len(csv_paths))
    return output_dir


__all__ = ["TileRequest", "tile_dataset", "HEADER"]
