"""Graph neural network heads for patch-level graphs."""
from __future__ import annotations

import torch
from torch import nn

try:  # pragma: no cover - optional dependency
    from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("torch_geometric is required for graph models") from exc


class GCNHead(nn.Module):
    """GCN for slide-level prediction."""

    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, n_classes)

    def forward(self, data) -> torch.Tensor:  # type: ignore[override]
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        pooled = global_mean_pool(x, batch)
        return self.lin(pooled)


class GATHead(nn.Module):
    """GAT for slide-level prediction."""

    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int, heads: int = 4) -> None:
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1)
        self.lin = nn.Linear(hidden_dim, n_classes)

    def forward(self, data) -> torch.Tensor:  # type: ignore[override]
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        pooled = global_mean_pool(x, batch)
        return self.lin(pooled)


class HybridMILGraph(nn.Module):
    """Combine MIL embedding with graph embedding."""

    def __init__(self, mil_dim: int, gnn_dim: int, n_classes: int) -> None:
        super().__init__()
        self.proj = nn.Linear(mil_dim + gnn_dim, n_classes)

    def forward(self, mil_emb: torch.Tensor, gnn_emb: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([mil_emb, gnn_emb], dim=-1)
        return self.proj(concat)
