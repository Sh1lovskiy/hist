# runner.py
from __future__ import annotations
from pathlib import Path
from typing import List
from loguru import logger
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from log_setup import setup_logger
from config import Paths, Tiling, Export, ClassMap
from wsi_reader import (
    open_wsi,
    get_level_dims,
    get_level_downsamples,
    get_mpp_mag,
    read_region_rgb,
)
from annotations import load_annotation, regions_to_level
from masking import render_mask, patch_label_from_mask
from tiler import Grid, is_enough_foreground
from exporter import PatchWriter, CSVMeta


def match_anno(wsi_path: Path, ann_dir: Path) -> Path:
    stem = wsi_path.stem
    cand = ann_dir / f"{stem}.json"
    if cand.exists():
        return cand
    for p in ann_dir.glob("*.json"):
        if p.stem.endswith(stem) or stem.endswith(p.stem):
            return p
    raise FileNotFoundError(f"No JSON matched for {wsi_path.name}")


def _candidate_coords(mask: np.ndarray, patch: int, stride: int) -> List[tuple]:
    """
    Быстрый пред-отсев: берем только позиции, где есть хоть немного
    размеченной области (маска != 255) в грубом сэмплинге сетки.
    Это резко снижает число обращений к WSI.
    """
    H, W = mask.shape
    xs = range(0, W - patch + 1, stride)
    ys = range(0, H - patch + 1, stride)
    coords = []
    for y in ys:
        row = mask[y : y + patch : stride, :]  # тонкий приём ускорить проверку
        for x in xs:
            m = mask[y : y + patch, x : x + patch]
            if (m != 255).any():
                coords.append((x, y))
    return coords


def process_one(
    wsi_path: Path,
    out_dir: Path,
    til_cfg: Tiling,
    exp_cfg: Export,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    slide = open_wsi(str(wsi_path))
    dims = get_level_dims(slide)
    downs = get_level_downsamples(slide)
    get_mpp_mag(slide)

    ann_path = match_anno(wsi_path, Paths.ann)
    regs_l0 = load_annotation(ann_path)

    writer = PatchWriter(root=out_dir, img_ext=exp_cfg.img_ext, quality=exp_cfg.quality)
    meta = CSVMeta(
        out_dir / f"{wsi_path.stem}.csv",
        ["rel_path", "level", "x", "y", "w", "h", "label_idx", "label_name"],
    )

    try:
        for lvl in til_cfg.levels:
            lw, lh = dims[lvl]
            ds = downs[lvl]
            stride = til_cfg.get_stride(lvl)
            regs = regions_to_level(regs_l0, ds)
            mask = render_mask((lw, lh), regs)

            logger.info(f"{wsi_path.name}: level={lvl}, size={lw}x{lh}, ds={ds}")

            # Предварительный список координат по маске
            coords = _candidate_coords(mask, til_cfg.patch, stride)
            total = len(coords)
            if total == 0:
                logger.info("No candidate patches at this level")
                continue

            # Ограничение на максимум патчей
            if total > til_cfg.max_patches_per_level:
                step = max(1, total // til_cfg.max_patches_per_level)
                coords = coords[::step]
                logger.info(
                    f"Capped patches: {total} -> {len(coords)} " f"(step={step})"
                )

            pbar = tqdm(
                coords,
                desc=f"{wsi_path.stem} L{lvl}",
                unit="patch",
                dynamic_ncols=True,
                leave=False,
            )
            for x, y in pbar:
                img = read_region_rgb(slide, lvl, x, y, til_cfg.patch, til_cfg.patch)
                if not is_enough_foreground(img, til_cfg.min_fg_ratio):
                    continue
                m = mask[y : y + til_cfg.patch, x : x + til_cfg.patch]
                lab = patch_label_from_mask(m, til_cfg.label_min_ratio)
                lab_name = ClassMap.idx_to_name[lab]

                rel = (
                    Path(wsi_path.stem)
                    / f"L{lvl}"
                    / f"{x}_{y}_{til_cfg.patch}{exp_cfg.img_ext}"
                )
                writer.save_img(img, rel)
                meta.write(
                    {
                        "rel_path": str(rel.as_posix()),
                        "level": lvl,
                        "x": x,
                        "y": y,
                        "w": til_cfg.patch,
                        "h": til_cfg.patch,
                        "label_idx": lab,
                        "label_name": lab_name,
                    }
                )
    except KeyboardInterrupt:
        logger.warning(f"Interrupted on {wsi_path.name}, flushing CSV.")
    finally:
        meta.close()
        logger.info(f"Done {wsi_path.name}")


def main() -> None:
    setup_logger()
    Paths.out.mkdir(parents=True, exist_ok=True)

    train_wsis = sorted(
        [
            Paths.wsis / "train_01.svs",
            Paths.wsis / "train_02.svs",
            Paths.wsis / "train_03.svs",
            Paths.wsis / "train_04.svs",
            Paths.wsis / "train_05.svs",
        ]
    )
    test_wsis = sorted(
        [
            Paths.wsis / "test_01.svs",
            Paths.wsis / "test_02.svs",
            Paths.wsis / "test_03.svs",
            Paths.wsis / "test_04.svs",
            Paths.wsis / "test_05.svs",
        ]
    )

    til_cfg = Tiling(
        levels=(0, 1, 2),
        patch=256,
        stride=256,
        stride_by_level={0: 256, 1: 384, 2: 512},
        min_fg_ratio=0.10,
        label_min_ratio=0.30,
        max_patches_per_level=20000,
    )
    exp_cfg = Export()

    for p in train_wsis:
        process_one(p, Paths.out / "train", til_cfg, exp_cfg)
    for p in test_wsis:
        process_one(p, Paths.out / "test", til_cfg, exp_cfg)


if __name__ == "__main__":
    main()
