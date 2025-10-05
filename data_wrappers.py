# data_wrappers.py (drop-in replacement for your file)
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    """
    Flat dataset of patches from multiple CSVs.
    Returns (tensor, label_idx, slide_id, level_str, (x,y)).
    """

    def __init__(self, csvs: List[Path], root: Path, tfm):
        self.root = root
        self.tfm = tfm
        df = [pd.read_csv(p) for p in csvs]
        self.df = pd.concat(df, ignore_index=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        rel = Path(r["rel_path"])
        img = Image.open(self.root / rel).convert("RGB")
        x = self.tfm(img)
        y = int(r["label_idx"])
        slide_id = rel.parts[0]  # train_01, etc.
        level = rel.parts[1]  # L0/L1/L2
        xy = (int(r["x"]), int(r["y"]))
        return x, y, slide_id, level, xy


def group_bags_by_slide_level(
    E: torch.Tensor,
    y: torch.Tensor,
    slides: List[str],
    levels: List[str],
    coords: List[Tuple[int, int]],
):
    """
    Build dicts:
      bags: slide_id -> level -> [Ni, d] embeddings
      targets: slide_id -> majority non-BG label else BG
      locs: slide_id -> level -> [(x,y)...] matching instance order
    """
    by_slide: Dict[str, Dict[str, list]] = {}
    locs: Dict[str, Dict[str, list]] = {}
    lab_slide: Dict[str, list] = {}

    for i, sid in enumerate(slides):
        lvl = levels[i]
        by_slide.setdefault(sid, {}).setdefault(lvl, []).append(E[i].unsqueeze(0))
        locs.setdefault(sid, {}).setdefault(lvl, []).append(coords[i])
        lab_slide.setdefault(sid, []).append(int(y[i]))

    bags, targets, locs_out = {}, {}, {}
    for sid, lv in by_slide.items():
        bags[sid] = {}
        locs_out[sid] = {}
        for L, lst in lv.items():
            bags[sid][L] = torch.cat(lst, dim=0)
            locs_out[sid][L] = locs[sid][L]
        ys = torch.as_tensor(lab_slide[sid])
        non_bg = ys[ys != 1]  # 1 = BG in your mapping
        val = int(non_bg.mode().values.item()) if non_bg.numel() > 0 else 1
        targets[sid] = val
    return bags, targets, locs_out
