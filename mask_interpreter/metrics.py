"""Pearson correlation, consistent with ``cubic.metrics.pcc``.

The numpy/eval path **delegates to** ``cubic.metrics.pcc`` (the canonical, tested,
device-agnostic implementation shared across projects). The torch/training path
**mirrors cubic's math** in a differentiable form, because ``cubic.pcc`` returns a
Python float and detaches torch inputs to numpy/cupy — it cannot carry the gradient the
MaskInterpreter loss needs.

cubic's conventions (matched in both paths):

- standard Pearson *r*: ``Σ(x-x̄)(y-ȳ) / √(Σ(x-x̄)² · Σ(y-ȳ)²)`` over the flattened
  (optionally masked) tensor — a single scalar, not per-sample.
- zero-variance guard: ``denom < 1e-12`` → ``NaN``.
- result clipped to ``[-1, 1]``.
- ``weights`` are interpreted as a **boolean mask** (``weights > 0``, so both 0/1 and
  0/255 seg encodings work); if the mask selects nothing we fall back to the full arrays
  (a MaskInterpreter safeguard kept on top of cubic, so an absent/empty segmentation
  does not poison a threshold sweep with NaN).
"""

from __future__ import annotations

import numpy as np
import torch
from cubic.metrics import pcc as _cubic_pcc

_ZERO_VAR_EPS = 1e-12


def pearson_corr(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable Pearson correlation mirroring ``cubic.metrics.pcc``."""
    x = y_true.reshape(-1)
    y = y_pred.reshape(-1)
    if weights is not None:
        mask = weights.reshape(-1) > 0
        if bool(mask.any()):
            x = x[mask]
            y = y[mask]
        # else: fall back to full arrays (empty-selection safeguard).
    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = torch.sqrt((x_c * x_c).sum() * (y_c * y_c).sum())
    if float(denom.detach()) < _ZERO_VAR_EPS:
        return torch.full((), float("nan"), dtype=x.dtype, device=x.device)
    return ((x_c * y_c).sum() / denom).clamp(-1.0, 1.0)


def pearson_corr_np(
    a: np.ndarray,
    b: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Numpy/eval Pearson correlation — delegates to ``cubic.metrics.pcc``."""
    mask = None
    if weights is not None:
        mask = np.asarray(weights) > 0
        if not mask.any():
            mask = None  # empty selection -> full arrays
    return float(_cubic_pcc(a, b, mask=mask))
