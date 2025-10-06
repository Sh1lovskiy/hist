"""Model encoder helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

ENCODER_ALIASES = {
    "vit_b16": "vit_base_patch16_224",
    "vit_b_16": "vit_base_patch16_224",
    "resnet18": "resnet18",
}

EXPECTED_PATCH = {
    "vit_base_patch16_224": 224,
    "resnet18": 224,
}


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    timm_name: str
    patch_size: int


def resolve_encoder(name: str) -> EncoderSpec:
    timm_name = ENCODER_ALIASES.get(name, name)
    patch_size = EXPECTED_PATCH.get(timm_name, 224)
    return EncoderSpec(name=name, timm_name=timm_name, patch_size=patch_size)


__all__ = ["EncoderSpec", "resolve_encoder", "ENCODER_ALIASES", "EXPECTED_PATCH"]
