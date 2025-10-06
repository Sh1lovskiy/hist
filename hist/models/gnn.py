"""Placeholder GNN components for testing purposes."""

from __future__ import annotations

import torch
from torch import nn


class DummyGNN(nn.Module):
    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = features.mean(dim=0, keepdim=True)
        return self.classifier(pooled)


__all__ = ["DummyGNN"]
