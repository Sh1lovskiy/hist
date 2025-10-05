# exporter.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
from PIL import Image
import csv


@dataclass
class PatchWriter:
    root: Path
    img_ext: str
    quality: int

    def save_img(self, img: Image.Image, rel_path: Path) -> Path:
        out_path = self.root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if self.img_ext.lower() == ".jpg":
            img.save(out_path, "JPEG", quality=self.quality, optimize=True)
        else:
            img.save(out_path)
        return out_path


class CSVMeta:
    def __init__(self, csv_path: Path, fieldnames: list[str]):
        self.csv_path = csv_path
        self.fieldnames = fieldnames
        self._fp = open(csv_path, "w", newline="", encoding="utf-8")
        self._wr = csv.DictWriter(self._fp, fieldnames=fieldnames)
        self._wr.writeheader()

    def write(self, row: Dict[str, Any]) -> None:
        self._wr.writerow(row)

    def close(self) -> None:
        self._fp.close()
