# models_gnn.py
from __future__ import annotations
import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
except Exception:
    GCNConv = GATConv = global_mean_pool = None


class GCNHead(nn.Module):
    """Two-layer GCN + global mean pool → logits."""

    def __init__(self, d: int, n_cls: int, h: int = 256):
        super().__init__()
        if GCNConv is None:
            raise RuntimeError("torch_geometric is required for GCNHead.")
        self.g1 = GCNConv(d, h)
        self.g2 = GCNConv(h, h)
        self.cls = nn.Linear(h, n_cls)

    def forward(self, data):
        x = self.g1(data.x, data.edge_index).relu()
        x = self.g2(x, data.edge_index).relu()
        # single graph per slide: batch fake as zeros
        b = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        z = global_mean_pool(x, b).unsqueeze(0)
        return self.cls(z)


class GATHead(nn.Module):
    """GAT(8 heads) + global pool → logits."""

    def __init__(self, d: int, n_cls: int, h: int = 256, heads: int = 4):
        super().__init__()
        if GATConv is None:
            raise RuntimeError("torch_geometric is required for GATHead.")
        self.g1 = GATConv(d, h, heads=heads, concat=True)
        self.g2 = GATConv(h * heads, h, heads=1, concat=True)
        self.cls = nn.Linear(h, n_cls)

    def forward(self, data):
        x = self.g1(data.x, data.edge_index).relu()
        x = self.g2(x, data.edge_index).relu()
        b = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        z = global_mean_pool(x, b).unsqueeze(0)
        return self.cls(z)
