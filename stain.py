# stain.py
from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2


def _optical_density(img_rgb: np.ndarray) -> np.ndarray:
    img = np.clip(img_rgb.astype(np.float32), 1.0, 255.0)
    return -np.log(img / 255.0 + 1e-6)


def _top_eigvecs(A: np.ndarray, k: int = 2) -> np.ndarray:
    cov = np.cov(A, rowvar=False)
    w, v = np.linalg.eigh(cov)
    idx = np.argsort(w)[::-1][:k]
    return v[:, idx]


def macenko_normalize(
    img_rgb: np.ndarray,
    ref_he: Tuple[np.ndarray, np.ndarray] | None = None,
    Io: float = 255.0,
    alpha: float = 0.1,
) -> np.ndarray:
    """
    Macenko stain normalization. Returns uint8 RGB.
    Keep <55 lines, minimal numpy/OPENCV implementation.
    """
    OD = _optical_density(img_rgb)
    OD = OD.reshape(-1, 3)
    OD = OD[~np.any(OD < 0.15, axis=1)]
    if OD.size == 0:
        return img_rgb

    V = _top_eigvecs(OD, k=2)
    proj = OD @ V
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(phi, [alpha * 100, (1 - alpha) * 100])
    He = (
        np.array(
            [
                np.cos([lo, hi]),
                np.sin([lo, hi]),
            ]
        ).T
        @ V.T
    )
    He = He / np.linalg.norm(He, axis=1, keepdims=True)

    C = np.linalg.lstsq(He.T, _optical_density(img_rgb).reshape(-1, 3).T, rcond=None)[0]
    if ref_he is None:
        ref_means = np.percentile(C, 99, axis=1)
    else:
        ref_means = np.asarray(ref_he).reshape(-1)
    Cn = C * (ref_means[:, None] / (np.percentile(C, 99, axis=1) + 1e-6)[:, None])
    In = (Io * np.exp(-(He.T @ Cn).T)).reshape(img_rgb.shape).clip(0, 255)
    return In.astype(np.uint8)
