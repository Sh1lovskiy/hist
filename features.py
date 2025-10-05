"""Feature extraction utilities for WSI patches."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch.amp import autocast
from tqdm.auto import tqdm

from data_wrappers import build_bags


@dataclass
class FeatureExtractionResult:
    slide_ids: List[str]
    features: List[torch.Tensor]
    magnifications: List[int]
    labels: List[int]
    coords: List[torch.Tensor]


class FeatureExtractor:
    """Run forward passes over patch dataset to produce features."""

    def __init__(self, encoder: torch.nn.Module, device: torch.device, use_amp: bool) -> None:
        self.encoder = encoder.to(device)
        self.device = device
        self.use_amp = use_amp

    def extract(self, dataloader) -> FeatureExtractionResult:
        self.encoder.eval()
        feats: List[torch.Tensor] = []
        labels: List[int] = []
        slide_ids: List[str] = []
        magnifications: List[int] = []
        coords: List[torch.Tensor] = []
        progress = tqdm(dataloader, desc="feature", leave=False)
        with torch.no_grad():
            for images, targets, sids, mags, xy in progress:
                images = images.to(self.device, non_blocking=True)
                with autocast("cuda", enabled=self.use_amp):
                    emb = self.encoder(images)
                feats.extend(emb.detach().cpu())
                labels.extend(targets.tolist())
                slide_ids.extend(list(sids))
                if isinstance(mags, torch.Tensor):
                    magnifications.extend(mags.tolist())
                else:
                    magnifications.extend(list(mags))
                coords.extend([xy_i.cpu() for xy_i in xy])
        return FeatureExtractionResult(
            slide_ids=slide_ids,
            features=feats,
            magnifications=magnifications,
            labels=labels,
            coords=coords,
        )

    def to_bags(self, result: FeatureExtractionResult):
        return build_bags(
            result.slide_ids,
            result.features,
            result.magnifications,
            result.labels,
            result.coords,
        )
