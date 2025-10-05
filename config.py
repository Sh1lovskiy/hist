# config.py
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class Paths:
    root: Path = Path("data/wss1_v2")
    wsis: Path = root / "wsis"
    ann: Path = root / "anno"
    out: Path = root / "out"


@dataclass(frozen=True)
class ClassMap:
    name_to_idx = {"AT": 0, "BG": 1, "LP": 2, "MM": 3, "DYS": 4}
    idx_to_name = {v: k for k, v in name_to_idx.items()}


@dataclass(frozen=True)
class Tiling:
    levels: tuple = (0, 1, 2)
    patch: int = 256
    # можно задать stride для каждого уровня; если короче — fallback на stride
    stride_by_level: dict = None
    stride: int = 256
    min_fg_ratio: float = 0.10
    label_min_ratio: float = 0.30
    max_patches_per_level: int = 20000  # кап, чтобы не зависнуть

    def get_stride(self, level: int) -> int:
        if self.stride_by_level and level in self.stride_by_level:
            return int(self.stride_by_level[level])
        return self.stride


@dataclass(frozen=True)
class Export:
    img_ext: str = ".jpg"
    quality: int = 90


@dataclass(frozen=True)
class RunConf:
    seed: int = 17
    n_workers: int = 0
