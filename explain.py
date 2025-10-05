"""Explainability utilities for MIL and GNN-based models."""
from __future__ import annotations

import argparse
import glob
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import openslide
import pandas as pd
from loguru import logger

from plotting import overlay_attention_heatmap
from data_wrappers import Bag


def save_attention_weights(attn: Dict[int, "torch.Tensor"], bag: Bag, path: Path) -> None:
    """Persist attention weights for each patch to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        f.write("magnification,index,x,y,weight\n")
        for mag, weights in attn.items():
            coords = bag.coords[mag].cpu().numpy()
            for idx, weight in enumerate(weights.cpu().numpy()):
                x, y = coords[idx]
                f.write(f"{mag},{idx},{int(x)},{int(y)},{float(weight):.6f}\n")
    logger.debug("Saved attention weights to {}", path)

try:  # pragma: no cover - optional dependency
    from torch_geometric.data import Data
    from torch_geometric.nn import GNNExplainer
    import torch
except Exception:  # pragma: no cover
    Data = None  # type: ignore
    GNNExplainer = None  # type: ignore
    torch = None  # type: ignore


def _load_attention_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"x", "y", "weight"}
    if not required.issubset(df.columns):
        raise ValueError(f"Attention CSV {path} must contain columns {required}")
    return df


def _slide_id_from_path(attn_path: Path) -> str:
    stem = attn_path.stem
    for prefix in ("attention_", "attn_", "weights_"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def _find_slide(slide_root: Path, slide_id: str) -> Path:
    for ext in (".svs", ".ndpi", ".tif", ".tiff", ".mrxs"):
        candidate = slide_root / f"{slide_id}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No slide found for {slide_id} in {slide_root}")


def _read_thumbnail(slide_path: Path, max_size: int = 2048) -> tuple[np.ndarray, float, float]:
    slide = openslide.OpenSlide(str(slide_path))
    level = slide.get_best_level_for_downsample(max(slide.dimensions) / max_size)
    region = slide.read_region((0, 0), level, slide.level_dimensions[level]).convert("RGB")
    thumbnail = np.array(region)
    sx = thumbnail.shape[1] / slide.dimensions[0]
    sy = thumbnail.shape[0] / slide.dimensions[1]
    slide.close()
    return thumbnail, sx, sy


def _accumulate_heatmap(
    df: pd.DataFrame,
    width: int,
    height: int,
    sx: float,
    sy: float,
    patch_size: int,
    sigma: float,
) -> np.ndarray:
    heatmap = np.zeros((height, width), dtype=np.float32)
    radius_x = max(int(math.ceil((patch_size * sx) / 2)), 1)
    radius_y = max(int(math.ceil((patch_size * sy) / 2)), 1)
    for _, row in df.iterrows():
        x = int(row["x"] * sx)
        y = int(row["y"] * sy)
        weight = float(row["weight"])
        x_min = max(x - radius_x, 0)
        x_max = min(x + radius_x, width - 1)
        y_min = max(y - radius_y, 0)
        y_max = min(y + radius_y, height - 1)
        heatmap[y_min : y_max + 1, x_min : x_max + 1] += weight
    if sigma > 0:
        from scipy.ndimage import gaussian_filter

        heatmap = gaussian_filter(heatmap, sigma=sigma)
    return heatmap


def compute_attention_entropy(weights: Sequence[float]) -> float:
    """Compute the Shannon entropy ``-∑ p log p`` of attention weights."""

    weights = np.asarray(weights, dtype=np.float64)
    if weights.size == 0:
        return float("nan")
    norm = weights / np.clip(weights.sum(), a_min=1e-9, a_max=None)
    entropy = -np.sum(norm * np.log(norm + 1e-12))
    return float(entropy)


def aggregate_class_heatmaps(
    class_to_heatmaps: Dict[str, List[np.ndarray]]
) -> Dict[str, np.ndarray]:
    aggregated: Dict[str, np.ndarray] = {}
    for cls, maps in class_to_heatmaps.items():
        if maps:
            stacked = np.stack(maps, axis=0)
            aggregated[cls] = stacked.mean(axis=0)
    return aggregated


def run_gnn_explainer(model_path: Path, data_path: Path, out_path: Path, epochs: int = 200) -> None:
    """Run GNNExplainer on a saved torch geometric graph."""

    if GNNExplainer is None or torch is None or Data is None:  # pragma: no cover
        logger.warning("torch-geometric not available; skipping GNNExplainer")
        return
    logger.info("Loading GNN model from {}", model_path)
    model = torch.load(model_path, map_location="cpu")
    data: Data = torch.load(data_path, map_location="cpu")
    model.eval()
    explainer = GNNExplainer(model, epochs=epochs)
    node_mask, edge_mask = explainer.explain_graph(data.x, data.edge_index)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"node_mask": node_mask, "edge_mask": edge_mask}, out_path)
    logger.success("Saved GNN explanations to {}", out_path)


def parse_args() -> argparse.Namespace:
    description = "Visualise MIL attention maps and optional GNN explanations."
    epilog = """Examples:\n  python -m explain \\n    --attn runs/hiermil_vit224/fold_0/attn \\\n    --slides data/wss1_v2/out/train/slides \\\n    --out runs/hiermil_vit224/attn_vis\n"""
    parser = argparse.ArgumentParser(
        "explain",
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--attn", type=Path, required=True, help="Directory or glob of attention CSVs")
    parser.add_argument("--slides", type=Path, required=True, help="Directory containing WSI files")
    parser.add_argument("--out", type=Path, required=True, help="Destination directory for outputs")
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--sigma", type=float, default=3.0, help="Gaussian smoothing sigma in pixels")
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional CSV mapping slide_id to true_label and predicted_label",
    )
    parser.add_argument(
        "--gnn-model",
        type=Path,
        help="Optional torch saved model for running GNNExplainer",
    )
    parser.add_argument(
        "--graph-data",
        type=Path,
        help="Graph data (torch file) for GNNExplainer",
    )
    return parser.parse_args()


def _resolve_attention_files(path: Path) -> List[Path]:
    if path.is_dir():
        return sorted(path.glob("*.csv"))
    raw = str(path)
    if any(ch in raw for ch in "*?[]"):
        return sorted(Path(p) for p in glob.glob(raw))
    if path.suffix == ".csv":
        return [path]
    raise FileNotFoundError(f"No attention CSVs found for {path}")


def main() -> None:
    args = parse_args()
    attention_files = _resolve_attention_files(args.attn)
    if not attention_files:
        raise RuntimeError("No attention files found")
    args.out.mkdir(parents=True, exist_ok=True)

    metadata = None
    if args.metadata and args.metadata.exists():
        metadata = pd.read_csv(args.metadata)
        if "slide_id" not in metadata.columns:
            raise ValueError("Metadata CSV must contain a 'slide_id' column")
    class_heatmaps: Dict[str, List[np.ndarray]] = defaultdict(list)
    entropy_rows = []

    for attn_file in attention_files:
        df = _load_attention_file(attn_file)
        slide_id = df.get("slide_id", [_slide_id_from_path(attn_file)])[0]
        try:
            slide_path = _find_slide(args.slides, slide_id)
        except FileNotFoundError as exc:
            logger.warning(str(exc))
            continue
        thumbnail, sx, sy = _read_thumbnail(slide_path)
        heatmap = _accumulate_heatmap(
            df,
            width=thumbnail.shape[1],
            height=thumbnail.shape[0],
            sx=sx,
            sy=sy,
            patch_size=args.patch_size,
            sigma=args.sigma,
        )
        overlay_attention_heatmap(thumbnail, heatmap, args.out / f"{slide_id}_overlay.png")
        np.save(args.out / f"{slide_id}_heatmap.npy", heatmap)
        weights = df["weight"].to_numpy()
        entropy = compute_attention_entropy(weights)
        entropy_rows.append({"slide_id": slide_id, "attention_entropy": entropy})
        if metadata is not None:
            row = metadata.loc[metadata["slide_id"] == slide_id]
            if not row.empty:
                cls = row.iloc[0].get("true_label", "unknown")
                class_heatmaps[str(cls)].append(heatmap)
    if entropy_rows:
        pd.DataFrame(entropy_rows).to_csv(args.out / "attention_entropy.csv", index=False)
        logger.info("Saved attention entropy statistics")
    aggregated = aggregate_class_heatmaps(class_heatmaps)
    for cls, heatmap in aggregated.items():
        background = np.full((heatmap.shape[0], heatmap.shape[1], 3), 255, dtype=np.float32)
        overlay_attention_heatmap(
            background,
            heatmap,
            args.out / f"class_{cls}_mean_attention.png",
        )
    if args.gnn_model and args.graph_data:
        run_gnn_explainer(args.gnn_model, args.graph_data, args.out / "gnn_explainer.pt")


if __name__ == "__main__":
    main()
