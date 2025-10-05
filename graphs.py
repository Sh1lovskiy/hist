# graphs.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Literal
import numpy as np

try:
    import torch
    from torch_geometric.data import Data
except Exception:
    torch, Data = None, None

try:
    from scipy.spatial import Delaunay
except Exception:
    Delaunay = None

from loguru import logger


@dataclass(frozen=True)
class GraphCfg:
    kind: Literal["knn", "delaunay"] = "knn"
    k: int = 8


def _knn_edges(XY: np.ndarray, k: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(XY))).fit(XY)
    idx = nbrs.kneighbors(return_distance=False)[:, 1:]
    src = np.repeat(np.arange(len(XY)), idx.shape[1])
    dst = idx.reshape(-1)
    return np.stack([src, dst], axis=0)


def _delaunay_edges(XY: np.ndarray) -> np.ndarray:
    if Delaunay is None:
        logger.warning("Scipy missing, falling back to kNN(k=8).")
        return _knn_edges(XY, 8)
    tri = Delaunay(XY)
    edges = set()
    for simplex in tri.simplices:
        a, b, c = simplex
        edges.update(
            {tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((a, c)))}
        )
    e = np.array(list(edges)).T
    return np.concatenate([e, e[::-1]], axis=1)  # undirected both ways


def build_graph(feats: np.ndarray, xy: np.ndarray, cfg: GraphCfg) -> "Data | dict":
    """
    Build a graph from patch embeddings and 2D coords.
    Returns torch_geometric Data when PyG is installed, else a dict.
    """
    if cfg.kind == "knn":
        edge_index = _knn_edges(xy, cfg.k)
    else:
        edge_index = _delaunay_edges(xy)

    if torch is None or Data is None:
        return {"x": feats, "edge_index": edge_index, "pos": xy}

    x = torch.from_numpy(feats).float()
    ei = torch.from_numpy(edge_index).long()
    pos = torch.from_numpy(xy).float()
    return Data(x=x, edge_index=ei, pos=pos)
