"""
Radar tensor -> radar image conversion  ("Convert to Radar Image" block).

The paper produces the range-azimuth image by taking the MAX over the Doppler
dimension of the 3D tensor. Other reductions (sum, mean) are provided for
flexibility, plus a dB conversion for visualisation.
"""
from __future__ import annotations

import numpy as np


def tensor_to_image(tensor: np.ndarray, reduction: str = "max") -> np.ndarray:
    """Collapse (R, A, D) -> (R, A) image. Paper default: max over Doppler."""
    if reduction == "max":
        return tensor.max(axis=2)
    if reduction == "sum":
        return tensor.sum(axis=2)
    if reduction == "mean":
        return tensor.mean(axis=2)
    raise ValueError(f"unknown reduction '{reduction}'")


def to_db(image: np.ndarray, ref: float = None, floor_db: float = -80.0) -> np.ndarray:
    """Linear magnitude -> dB, normalised to peak (ref) with a floor."""
    ref = image.max() if ref is None else ref
    if ref <= 0:
        return np.full_like(image, floor_db)
    db = 20.0 * np.log10(np.maximum(image, 1e-12) / ref)
    return np.maximum(db, floor_db)
