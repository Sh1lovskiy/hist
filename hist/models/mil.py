"""Minimal MIL models used for smoke testing."""

from __future__ import annotations

import torch
from torch import nn


class MeanMIL(nn.Module):
    """Average pool features and classify."""

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, bag: torch.Tensor) -> torch.Tensor:
        pooled = bag.mean(dim=0, keepdim=True)
        return self.classifier(pooled)


def build_model(name: str, in_features: int, num_classes: int) -> nn.Module:
    if name.lower() not in {"mean", "abmil", "hiermil", "transmil"}:
        raise ValueError(f"Unsupported model '{name}' in smoke implementation")
    return MeanMIL(in_features=in_features, num_classes=num_classes)


__all__ = ["MeanMIL", "build_model"]
