"""Backbone factory for feature extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from loguru import logger

try:  # pragma: no cover - optional dependency
    import timm
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("timm is required for encoder factory") from exc

try:  # pragma: no cover
    import open_clip
except ImportError:  # pragma: no cover
    open_clip = None


@dataclass
class EncoderConfig:
    name: str = "resnet18"
    pretrained: bool = True
    fine_tune: bool = False
    drop_rate: float = 0.0
    img_size: int = 224


class Encoder(torch.nn.Module):
    """Wrapper returning pooled features."""

    def __init__(self, backbone: torch.nn.Module, feat_dim: int, fine_tune: bool) -> None:
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.fine_tune = fine_tune

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        with torch.set_grad_enabled(self.fine_tune):
            feats = self.backbone(x)
        return feats


def build_encoder(cfg: EncoderConfig) -> Tuple[Encoder, int]:
    """Instantiate encoder by name."""
    aliases = {
        "vit_b16": "vit_base_patch16_224",
        "vit_b32": "vit_base_patch32_224",
        "convnext_t": "convnext_tiny",
        "convnext_base": "convnext_base",
        "resnet18": "resnet18",
        "resnet50": "resnet50",
        "clip_vitb16": "ViT-B-16",
    }
    key = cfg.name.lower()
    name = aliases.get(key, key)
    if key.startswith("clip"):
        return _build_clip(cfg)
    if name == "hipt":
        return _build_hipt(cfg)
    extra = {}
    if "vit" in name.lower():
        extra["img_size"] = cfg.img_size
    model = timm.create_model(
        name,
        pretrained=cfg.pretrained,
        num_classes=0,
        global_pool="avg",
        drop_rate=cfg.drop_rate,
        **extra,
    )
    feat_dim = model.num_features
    encoder = Encoder(model, feat_dim, cfg.fine_tune)
    if not cfg.fine_tune:
        for param in encoder.parameters():
            param.requires_grad = False
    logger.info(f"Encoder {name} created (dim={feat_dim})")
    return encoder, feat_dim


def _build_clip(cfg: EncoderConfig) -> Tuple[Encoder, int]:
    if open_clip is None:
        raise RuntimeError("open_clip is required for CLIP encoders")
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai"
    )
    encoder = Encoder(model.visual, model.visual.output_dim, cfg.fine_tune)
    if not cfg.fine_tune:
        for param in encoder.parameters():
            param.requires_grad = False
    return encoder, model.visual.output_dim


def _build_hipt(cfg: EncoderConfig) -> Tuple[Encoder, int]:
    """Placeholder HIPT encoder using ViT hierarchical token."""
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=cfg.pretrained,
        num_classes=0,
        global_pool="avg",
    )
    feat_dim = model.num_features
    encoder = Encoder(model, feat_dim, cfg.fine_tune)
    if not cfg.fine_tune:
        for param in encoder.parameters():
            param.requires_grad = False
    return encoder, feat_dim
