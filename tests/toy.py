"""Toy frozen predictors with a known-informative region, for overfit tests."""

from __future__ import annotations

import torch
from torch import nn


def _center_region(spatial: tuple[int, ...]) -> torch.Tensor:
    region = torch.zeros(1, 1, *spatial)
    slices = [slice(None), slice(None)]
    for s in spatial:
        lo, hi = s // 4, s - s // 4
        slices.append(slice(lo, hi))
    region[tuple(slices)] = 1.0
    return region


class RegionImagePredictor(nn.Module):
    """Image->image predictor that only reads a central spatial block.

    ``forward(x) = x * region`` — the weight is a real ``Parameter`` (so ``freeze``
    disables its grad and we can assert it is unchanged after training)."""

    def __init__(self, spatial: tuple[int, ...]):
        super().__init__()
        self.weight = nn.Parameter(_center_region(spatial))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class RegionScorePredictor(nn.Module):
    """Vector-output predictor (clf/reg) sensitive only to a central block.

    Returns ``out_dim`` linear read-outs of the masked region sum."""

    def __init__(self, spatial: tuple[int, int], in_channels: int, out_dim: int):
        super().__init__()
        self.register_buffer("region", _center_region(spatial))
        self.head = nn.Linear(in_channels, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked = x * self.region  # (N, C, H, W)
        pooled = masked.sum(dim=(2, 3))  # (N, C)
        return self.head(pooled)  # (N, out_dim)


def region_masks(spatial: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    region = _center_region(spatial)[0, 0].bool()
    return region, ~region
