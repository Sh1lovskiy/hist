"""Augmentation and stain normalization utilities."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from torchvision import transforms


def macenko_normalize(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Approximate Macenko stain normalization."""
    img = image.astype(np.float32) + eps
    log_rgb = np.log(img)
    centered = log_rgb - log_rgb.mean(axis=(0, 1), keepdims=True)
    u, _, vh = np.linalg.svd(centered.reshape(-1, 3), full_matrices=False)
    stain_matrix = vh[:2, :]
    concentrations = centered.reshape(-1, 3) @ stain_matrix.T
    concentrations = (concentrations - concentrations.min()) / (
        concentrations.max() - concentrations.min() + eps
    )
    norm = concentrations @ stain_matrix
    norm = np.exp(norm).reshape(image.shape)
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    return norm


def build_transforms(
    mode: str, image_size: int, stain_normalization: bool
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Factory of augmentation pipelines."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    apply_stain = stain_normalization or mode == "strong"

    def _stain(img):
        if not apply_stain:
            return img
        arr = np.array(img)
        arr = macenko_normalize(arr)
        return transforms.functional.to_pil_image(arr)

    if mode == "none":
        train_ops = [transforms.Resize((image_size, image_size))]
    else:
        train_ops = [transforms.Resize((image_size, image_size))]
        if mode in {"basic", "strong"}:
            train_ops.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(90),
                ]
            )
        if mode == "strong":
            train_ops.extend(
                [
                    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
                    transforms.RandomApply(
                        [transforms.GaussianBlur(kernel_size=3)], p=0.3
                    ),
                ]
            )
    train_ops.extend([transforms.ToTensor(), normalize])
    eval_ops = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ]
    train_tfm = transforms.Compose([transforms.Lambda(_stain), *train_ops])
    eval_tfm = transforms.Compose([transforms.Lambda(_stain), *eval_ops])
    return train_tfm, eval_tfm
