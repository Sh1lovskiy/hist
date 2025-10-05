"""MIL models: ABMIL, TransMIL, Hierarchical MIL."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn


class ABMIL(nn.Module):
    """Attention-based MIL model."""

    def __init__(self, in_dim: int, n_classes: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.feature(x)
        att = self.attention(feats)
        weights = torch.softmax(att.transpose(0, 1), dim=1)
        emb = torch.mm(weights, feats)
        logits = self.classifier(emb)
        return logits, weights.squeeze(0)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional embedding."""

    def __init__(self, dim: int, max_len: int = 5000) -> None:
        super().__init__()
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim)
        )
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransMIL(nn.Module):
    """Transformer-based MIL model."""

    def __init__(self, in_dim: int, n_classes: int, depth: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, in_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim, nhead=heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.pos_encoding = PositionalEncoding(in_dim)
        self.att_linear = nn.Linear(in_dim, 1)
        self.classifier = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        proj = self.input_proj(x).unsqueeze(0)
        h = self.pos_encoding(proj)
        h = self.transformer(h)
        att = torch.softmax(self.att_linear(h).squeeze(0).squeeze(-1), dim=0)
        pooled = torch.matmul(att.unsqueeze(0), h.squeeze(0)).squeeze(0)
        logits = self.classifier(pooled)
        return logits.unsqueeze(0), att


class HierMIL(nn.Module):
    """Hierarchical MIL combining multiple magnifications."""

    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        magnifications: Tuple[int, ...] = (40, 10),
    ) -> None:
        super().__init__()
        self.magnifications = magnifications
        self.level_attention = nn.ModuleDict(
            {str(mag): nn.Sequential(nn.Linear(in_dim, in_dim), nn.Tanh(), nn.Linear(in_dim, 1)) for mag in magnifications}
        )
        self.level_proj = nn.ModuleDict(
            {str(mag): nn.Linear(in_dim, in_dim) for mag in magnifications}
        )
        self.global_att = nn.Sequential(nn.Linear(in_dim, in_dim), nn.Tanh(), nn.Linear(in_dim, 1))
        self.classifier = nn.Linear(in_dim, n_classes)

    def forward(self, bags: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        embeddings = []
        att_weights: Dict[int, torch.Tensor] = {}
        for mag in self.magnifications:
            if mag not in bags:
                continue
            feats = bags[mag]
            att = self.level_attention[str(mag)](feats)
            weights = torch.softmax(att.transpose(0, 1), dim=1)
            pooled = torch.mm(weights, self.level_proj[str(mag)](feats))
            embeddings.append(pooled)
            att_weights[mag] = weights.squeeze(0)
        if not embeddings:
            raise ValueError("No magnification features available for HierMIL")
        stacked = torch.cat(embeddings, dim=0)
        global_att = torch.softmax(self.global_att(stacked).transpose(0, 1), dim=1)
        slide_emb = torch.mm(global_att, stacked)
        logits = self.classifier(slide_emb)
        return logits, att_weights
