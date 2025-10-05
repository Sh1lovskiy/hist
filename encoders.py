# encoders.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple
import torch
import torch.nn as nn

try:
    import timm
except Exception:
    timm = None

try:
    import open_clip
except Exception:
    open_clip = None


@dataclass(frozen=True)
class EncCfg:
    name: Literal["resnet18", "vit_b16", "convnext_t", "clip_vit_b32", "hipt"] = (
        "resnet18"
    )
    img_size: int = 224
    freeze: bool = True


def _freeze(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False


class Encoder(nn.Module):
    """Feature extractor returning (feats, out_dim)."""

    def __init__(self, cfg: EncCfg):
        super().__init__()
        self.cfg = cfg
        self.backbone, self.out_dim = self._build(cfg)
        if cfg.freeze:
            _freeze(self.backbone)

    def _build(self, cfg: EncCfg) -> Tuple[nn.Module, int]:
        if cfg.name in {"resnet18", "vit_b16", "convnext_t"}:
            if timm is None:
                raise RuntimeError("timm is required for this encoder.")
            model = timm.create_model(
                {
                    "resnet18": "resnet18",
                    "vit_b16": "vit_base_patch16_224",
                    "convnext_t": "convnext_tiny",
                }[cfg.name],
                pretrained=True,
                num_classes=0,
            )
            out_dim = model.num_features
            return model, int(out_dim)

        if cfg.name == "clip_vit_b32":
            if open_clip is None:
                raise RuntimeError("open_clip is required for CLIP encoder.")
            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            # Take visual trunk until pre-logit (global pooled)
            visual = model.visual
            out_dim = int(getattr(visual, "output_dim", 512))

            class CLIPFeat(nn.Module):
                def __init__(self, vis):
                    super().__init__()
                    self.vis = vis

                def forward(self, x):
                    return self.vis(x)

            return CLIPFeat(visual), out_dim

        if cfg.name == "hipt":
            # Minimal stub: expect an external HIPT extractor in future.
            # For now, fallback to timm ViT-B/16 features to keep pipeline running.
            if timm is None:
                raise RuntimeError("timm is required for HIPT fallback.")
            model = timm.create_model(
                "vit_base_patch16_224", pretrained=True, num_classes=0
            )
            return model, int(model.num_features)

        raise ValueError(f"Unknown encoder: {cfg.name}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
