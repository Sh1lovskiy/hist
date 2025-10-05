"""Explainability utilities for MIL and GNN models."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from loguru import logger

try:  # pragma: no cover - optional dependency
    from torch_geometric.nn import GNNExplainer
except ImportError:  # pragma: no cover
    GNNExplainer = None

from data_wrappers import Bag


def save_attention_weights(attn: Dict[int, torch.Tensor], bag: Bag, path: Path) -> None:
    """Save MIL attention weights per magnification to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        f.write("magnification,index,x,y,weight\n")
        for mag, weights in attn.items():
            coords = bag.coords[mag].cpu().numpy()
            for idx, weight in enumerate(weights.cpu().numpy()):
                x, y = coords[idx]
                f.write(f"{mag},{idx},{int(x)},{int(y)},{float(weight):.6f}\n")
    logger.debug(f"Saved attention weights to {path}")


def explain_graph(model, data, path: Path) -> None:
    """Run GNNExplainer if available and save heatmap."""
    if GNNExplainer is None:
        logger.warning("GNNExplainer not available; skipping explanation")
        return
    explainer = GNNExplainer(model, epochs=200)
    node_feat_mask, edge_mask = explainer.explain_graph(data.x, data.edge_index)
    np.savez_compressed(path, node_feat_mask=node_feat_mask.cpu().numpy(), edge_mask=edge_mask.cpu().numpy())
    logger.debug(f"Saved GNN explanation to {path}")
