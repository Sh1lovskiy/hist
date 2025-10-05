from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class ABMIL(nn.Module):
    """Attention-MIL head."""

    def __init__(self, d: int, n_cls: int):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1))
        self.cls = nn.Linear(d, n_cls)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.att(H).squeeze(-1)
        w = torch.softmax(a, dim=0).unsqueeze(-1)
        z = torch.sum(w * H, dim=0, keepdim=True)
        logits = self.cls(z)
        return logits, w.squeeze(-1)


class LevelGate(nn.Module):
    """Gating for level-wise fusion."""

    def __init__(self, d: int):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        # Z: [L, d]
        a = torch.softmax(self.g(Z).squeeze(-1), dim=0)  # [L]
        z = torch.sum(a.unsqueeze(-1) * Z, dim=0, keepdim=True)
        return z  # [1, d]


class HierMIL(nn.Module):
    """
    Hierarchical MIL: attention pooling inside each level,
    gated fusion across levels, then classifier.
    """

    def __init__(self, d: int, n_cls: int):
        super().__init__()
        self.att = nn.ModuleDict(
            {
                "L0": nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1)),
                "L1": nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1)),
                "L2": nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1)),
            }
        )
        self.gate = LevelGate(d)
        self.cls = nn.Linear(d, n_cls)

    def _pool_level(self, H: torch.Tensor, level: str) -> torch.Tensor:
        a = self.att[level](H).squeeze(-1)
        w = torch.softmax(a, dim=0).unsqueeze(-1)
        z = torch.sum(w * H, dim=0, keepdim=True)
        return z  # [1, d]

    def forward(self, bags_by_level: Dict[str, torch.Tensor]) -> torch.Tensor:
        Z = []
        for L in ("L0", "L1", "L2"):
            if L in bags_by_level and bags_by_level[L].size(0) > 0:
                Z.append(self._pool_level(bags_by_level[L], L))
        if not Z:
            raise ValueError("No levels provided to HierMIL.")
        Z = torch.cat(Z, dim=0)  # [L, d]
        zf = self.gate(Z)  # [1, d]
        logits = self.cls(zf)  # [1, n_cls]
        return logits
