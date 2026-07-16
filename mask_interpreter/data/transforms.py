"""Numpy/torch transforms replacing the TF ``ImageUtils`` + ``resize_image`` helpers.

- ``normalize_std`` / ``normalize_minmax`` mirror ``ImageUtils.normalize_std`` /
  ``ImageUtils.normalize``.
- ``rot90`` reproduces the augmentation ``np.rot90(..., axes=(2,3))`` — a rotation of
  the **last two spatial axes** (Y, X). Callers pass channel-first arrays, so the
  spatial axes are the trailing ones (CHW -> (1,2); CDHW -> (2,3)).
- ``pad_z_to`` reproduces ``preprocess_image``'s ``np.pad(..., 'edge')`` with
  ``ceil((target - Z) / 2)`` on **both** sides (this over-shoots odd deficits — kept
  for patch-count parity with the TF pipeline).
- ``resize_nearest`` replaces ``tf.image.resize(..., NEAREST)`` with ``F.interpolate``.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def normalize_std(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    mean = image.mean()
    std = image.std()
    if std == 0:
        return image - mean
    return (image - mean) / std


def normalize_minmax(image: np.ndarray, max_value: float = 1.0) -> np.ndarray:
    image = image.astype(np.float32)
    mn = image.min()
    mx = image.max()
    if mx == mn:
        return np.zeros_like(image)
    return (image - mn) / (mx - mn) * max_value


def normalize(image: np.ndarray, norm_type: str = "std", max_value: float = 1.0) -> np.ndarray:
    if norm_type == "std":
        return normalize_std(image)
    if norm_type == "minmax":
        return normalize_minmax(image, max_value=max_value)
    raise ValueError(f"unknown norm_type {norm_type!r} (use 'std' or 'minmax')")


def rot90(image: np.ndarray, k: int, axes: tuple[int, int] | None = None) -> np.ndarray:
    """Rotate the last two axes by ``k*90`` degrees (faithful to ``axes=(2,3)`` on CDHW)."""
    if axes is None:
        axes = (image.ndim - 2, image.ndim - 1)
    return np.rot90(image, k=k, axes=axes)


def pad_z_to(image: np.ndarray, target: int = 32, axis: int = 0) -> np.ndarray:
    """Edge-pad ``axis`` up to ``target`` with ``ceil((target-Z)/2)`` on both sides."""
    z = image.shape[axis]
    if z >= target:
        return image
    pad_each = math.ceil((target - z) / 2)
    pad_width = [(0, 0)] * image.ndim
    pad_width[axis] = (pad_each, pad_each)
    return np.pad(image, pad_width, mode="edge")


def resize_nearest(image: torch.Tensor, size: tuple[int, ...]) -> torch.Tensor:
    """Nearest-neighbour resize of a channel-first tensor's spatial dims (NCHW/NCDHW)."""
    return F.interpolate(image, size=size, mode="nearest")


def sample_patch_indices(
    shape: tuple[int, ...], patch_size: tuple[int, ...], rng: np.random.Generator
) -> list[tuple[int, int]]:
    """Random start/stop per spatial axis for one patch (matches DataGen sampling)."""
    idx: list[tuple[int, int]] = []
    for dim, p in zip(shape, patch_size):
        if p > dim:
            raise ValueError(f"patch dim {p} exceeds image dim {dim}")
        start = int(rng.integers(0, dim - p + 1))
        idx.append((start, start + p))
    return idx
