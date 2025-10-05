"""Graph construction utilities for patch-level features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

try:  # pragma: no cover - optional dependency
    from sklearn.neighbors import NearestNeighbors
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("scikit-learn is required for graph construction") from exc

try:  # pragma: no cover
    from scipy.spatial import Delaunay
except ImportError:  # pragma: no cover
    Delaunay = None

from torch_geometric.data import Data

from data_wrappers import Bag


@dataclass
class GraphConfig:
    mode: Literal["knn", "delaunay"] = "knn"
    k: int = 8


class GraphBuilder:
    """Construct patch-level graphs from coordinates."""

    def __init__(self, cfg: GraphConfig) -> None:
        self.cfg = cfg

    def build(self, bag: Bag, magnification: int) -> Data:
        feats = bag.features[magnification]
        coords = bag.coords[magnification].float().cpu().numpy()
        edges = self._edges(coords)
        data = Data(x=feats, edge_index=edges, y=torch.tensor([bag.label]))
        data.slide_id = bag.slide_id
        data.magnification = magnification
        return data

    def _edges(self, coords: np.ndarray) -> torch.Tensor:
        if self.cfg.mode == "knn":
            return self._edges_knn(coords)
        return self._edges_delaunay(coords)

    def _edges_knn(self, coords: np.ndarray) -> torch.Tensor:
        nbrs = NearestNeighbors(n_neighbors=min(self.cfg.k + 1, len(coords)))
        nbrs.fit(coords)
        indices = nbrs.kneighbors(return_distance=False)
        src = np.repeat(np.arange(len(coords)), indices.shape[1] - 1)
        dst = indices[:, 1:].reshape(-1)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        return edge_index

    def _edges_delaunay(self, coords: np.ndarray) -> torch.Tensor:
        if Delaunay is None:
            raise RuntimeError("scipy is required for Delaunay graphs")
        tri = Delaunay(coords)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                edges.add(tuple(sorted((a, b))))
        src, dst = zip(*edges) if edges else ([], [])
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        return edge_index
