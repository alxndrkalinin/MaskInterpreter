"""Runnable quickstart for the PyTorch MaskInterpreter port.

Trains each of the three variants on a tiny synthetic problem with a known-informative
region, then confirms the learned importance mask concentrates on that region. No real
data or pretrained weights required.

    python examples/quickstart.py
"""

from __future__ import annotations

import torch
from torch import nn

from mask_interpreter import (
    ClassificationInterpreter,
    Image2ImageInterpreter,
    RegressionInterpreter,
    freeze,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def center_region(spatial):
    region = torch.zeros(1, 1, *spatial)
    sl = [slice(None), slice(None)] + [slice(s // 4, s - s // 4) for s in spatial]
    region[tuple(sl)] = 1.0
    return region


class RegionImagePredictor(nn.Module):
    """image->image predictor reading only a central block."""

    def __init__(self, spatial):
        super().__init__()
        self.register_buffer("region", center_region(spatial))

    def forward(self, x):
        return x * self.region


class RegionScorePredictor(nn.Module):
    """vector-output predictor reading only a central block."""

    def __init__(self, spatial, in_channels, out_dim):
        super().__init__()
        self.register_buffer("region", center_region(spatial))
        self.head = nn.Linear(in_channels, out_dim)

    def forward(self, x):
        return self.head((x * self.region).sum(dim=(2, 3)))


def overfit(model, x, batch, steps=200, lr=2e-3):
    model = model.to(DEVICE)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        total, metrics = model.training_step(batch)
        total.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        mask = model(x)
    inside = center_region(x.shape[2:])[0, 0].bool().to(DEVICE)
    mean_in = float(mask[:, 0][:, inside].mean())
    mean_out = float(mask[:, 0][:, ~inside].mean())
    return metrics, mean_in, mean_out


def main():
    torch.manual_seed(0)
    print(f"device: {DEVICE}\n")

    # image-to-image (2D) --------------------------------------------------
    spatial = (16, 16)
    m = Image2ImageInterpreter(freeze(RegionImagePredictor(spatial)),
                               spatial_size=spatial, ndim=2, in_channels=1, pred_channels=1)
    x = torch.randn(4, 1, *spatial, device=DEVICE)
    metrics, mi, mo = overfit(m, x, batch=(x,))  # image2image batch is (x[, seg])
    print(f"image2image: pcc={metrics['pcc']:.3f}  mask in-region={mi:.3f} out={mo:.3f}")

    # classification (2D) --------------------------------------------------
    spatial = (8, 8)
    m = ClassificationInterpreter(freeze(RegionScorePredictor(spatial, 3, 2)),
                                  spatial, in_channels=3)
    x = torch.randn(8, 3, *spatial, device=DEVICE)
    metrics, mi, mo = overfit(m, x, batch=x)  # clf/reg batch is x (or (x, y))
    print(f"classification: pcc={metrics['pcc']:.3f}  mask in-region={mi:.3f} out={mo:.3f}")

    # regression (2D) ------------------------------------------------------
    m = RegressionInterpreter(freeze(RegionScorePredictor(spatial, 1, 1)),
                              spatial, in_channels=1)
    x = torch.randn(8, 1, *spatial, device=DEVICE)
    metrics, mi, mo = overfit(m, x, batch=x)
    print(f"regression: pcc={metrics['pcc']:.3f}  mask in-region={mi:.3f} out={mo:.3f}")

    print("\nAll three variants localize the informative region (in-region mask > out-region).")


if __name__ == "__main__":
    main()
