"""Dataset wrappers for patches and MIL bags."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
import csv

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from loguru import logger

from wsi_reader import WSIReader


@dataclass
class Bag:
    """Container for a slide represented as multi-magnification bags."""

    slide_id: str
    label: int
    features: Dict[int, torch.Tensor]
    coords: Dict[int, torch.Tensor]


class PatchDataset(Dataset):
    """Dataset that reads image patches from raw slides on-the-fly."""

    def __init__(
        self,
        csv_files: Sequence[Path],
        reader: WSIReader,
        transform,
        class_map: Dict[str, int],
        slides_dir: Path,
    ) -> None:
        self.reader = reader
        self.transform = transform
        self.class_map = class_map
        self.slides_dir = slides_dir
        self.records: List[Dict] = []
        self._missing_paths: set[str] = set()
        for csv_path in csv_files:
            with csv_path.open("r", newline="") as f:
                reader_obj = csv.DictReader(f)
                for row in reader_obj:
                    row["csv_path"] = csv_path
                    if "slide_path" not in row or not row["slide_path"].strip():
                        inferred = self._infer_slide_path(row)
                        if inferred is not None:
                            row["slide_path"] = str(inferred)
                    self.records.append(row)
        if not self.records:
            logger.warning(f"No patch records found in {csv_files}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        x = self._parse_int(record.get("x"), 0)
        y = self._parse_int(record.get("y"), 0)
        level = self._parse_int(record.get("level"), 0)
        patch_size = self._parse_int(record.get("patch_size"), 256)
        pil = self._load_image(record, x, y, level, patch_size)
        tensor = self.transform(pil)
        label = self._resolve_label(record)
        slide_id = self._resolve_slide_id(record)
        magnification = self._parse_int(record.get("magnification"), 40)
        coords = torch.tensor([x, y], dtype=torch.int32)
        return tensor, label, slide_id, magnification, coords

    def _load_image(
        self,
        record: Dict,
        x: int,
        y: int,
        level: int,
        patch_size: int,
    ) -> Image.Image:
        slide_path = record.get("slide_path")
        if slide_path:
            resolved = self._resolve_slide_path(record)
            array = self.reader.read_region(resolved, (x, y), level, (patch_size, patch_size))
            return transforms.functional.to_pil_image(array)
        rel_path = record.get("rel_path")
        if rel_path:
            csv_path = Path(record["csv_path"])
            img_path = (csv_path.parent / rel_path).resolve()
            if not img_path.exists():
                raise FileNotFoundError(f"Patch image not found at {img_path}")
            return Image.open(img_path).convert("RGB")
        raise KeyError("CSV must contain slide_path or rel_path")

    def _resolve_label(self, record: Dict) -> int:
        label_name = record.get("label")
        if label_name:
            return int(self.class_map.get(label_name, 0))
        label_idx = record.get("label_idx")
        return self._parse_int(label_idx, 0)

    def _resolve_slide_id(self, record: Dict) -> str:
        slide_id = record.get("slide_id")
        if slide_id:
            return slide_id
        csv_path = Path(record.get("csv_path", "unknown"))
        return csv_path.stem or "unknown"

    def _resolve_slide_path(self, record: Dict) -> Path:
        raw = record.get("slide_path", "").strip()
        if raw:
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = (record["csv_path"].parent / candidate).resolve()
            return candidate
        inferred = self._infer_slide_path(record)
        if inferred is None:
            slide_id = record.get("slide_id", "unknown")
            if slide_id not in self._missing_paths:
                self._missing_paths.add(slide_id)
                logger.error(
                    f"Could not resolve slide path for {slide_id}. "
                    f"Ensure CSVs include 'slide_path' or place WSIs in the slides directory."
                )
            raise KeyError("slide_path")
        record["slide_path"] = str(inferred)
        return inferred

    def _infer_slide_path(self, record: Dict) -> Path | None:
        slide_id = record.get("slide_id")
        if not slide_id:
            return None
        candidates = []
        suffixes = (
            ".svs",
            ".ndpi",
            ".tiff",
            ".svslide",
            ".mrxs",
            ".isyntax",
            ".bif",
            ".scn",
        )
        for suffix in suffixes:
            candidate = self.slides_dir / f"{slide_id}{suffix}"
            if candidate.exists():
                return candidate
            candidates.append(candidate)
        if slide_id not in self._missing_paths:
            self._missing_paths.add(slide_id)
            joined = ", ".join(str(path) for path in candidates[:3])
            logger.warning(
                f"slide_path missing for {slide_id}. Tried candidates like {joined}"
            )
        return None

    def _parse_int(self, value, default: int) -> int:
        if value is None or value == "":
            return default
        if isinstance(value, (int, np.integer)):
            return int(value)
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(float(value))


def build_bags(
    slide_ids: Iterable[str],
    features: Iterable[torch.Tensor],
    magnifications: Iterable[int],
    labels: Iterable[int],
    coords: Iterable[torch.Tensor],
) -> Dict[str, Bag]:
    """Group patch features into bags per slide and magnification."""
    bags: Dict[str, Bag] = {}
    for sid, feat, mag, lab, coord in zip(slide_ids, features, magnifications, labels, coords):
        bag = bags.get(sid)
        if bag is None:
            bag = Bag(slide_id=sid, label=int(lab), features={}, coords={})
            bags[sid] = bag
        feat_list = bag.features.setdefault(mag, [])
        coord_list = bag.coords.setdefault(mag, [])
        feat_list.append(feat)
        coord_list.append(coord)
    for bag in bags.values():
        for mag in list(bag.features.keys()):
            bag.features[mag] = torch.stack(bag.features[mag])
            bag.coords[mag] = torch.stack(bag.coords[mag])
    return bags


class BagDataset(Dataset):
    """Dataset returning bag dictionaries for MIL models."""

    def __init__(self, bags: Sequence[Bag]) -> None:
        self.bags = list(bags)

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int):
        bag = self.bags[idx]
        return bag.features, bag.label, bag.coords, bag.slide_id


def oversample_bags(bags: Sequence[Bag]) -> List[Bag]:
    """Oversample minority classes by duplication."""
    if not bags:
        return []
    class_to_bags: Dict[int, List[Bag]] = {}
    for bag in bags:
        class_to_bags.setdefault(bag.label, []).append(bag)
    max_count = max(len(v) for v in class_to_bags.values())
    balanced: List[Bag] = []
    for label, group in class_to_bags.items():
        reps = int(np.ceil(max_count / len(group)))
        duplicated = (group * reps)[:max_count]
        balanced.extend(duplicated)
    return balanced


def compute_class_weights(bags: Sequence[Bag]) -> torch.Tensor:
    """Compute normalized inverse-frequency class weights."""
    labels = torch.tensor([bag.label for bag in bags], dtype=torch.long)
    classes, counts = labels.unique(return_counts=True)
    weights = counts.float().reciprocal()
    weights = weights / weights.sum() * len(classes)
    full = torch.ones(int(classes.max()) + 1, dtype=torch.float32)
    full[classes] = weights
    return full
