"""Model-agnostic frozen-predictor interface.

MaskInterpreter wraps *any* frozen ``nn.Module`` mapping ``x -> y``. Freezing means
``.eval()`` (deterministic BatchNorm/dropout) **and** ``requires_grad_(False)`` on the
parameters. Crucially, gradients must still flow **through** the predictor from its
output back to the adapted image and the mask — only the predictor *parameters* are
frozen, not the computation graph. So we do NOT wrap forward passes in ``no_grad``
during the loss computation; the reference prediction ``pred(x)`` is instead computed
under ``no_grad`` / detached (it is a constant, matching the TF code which computes it
outside the ``GradientTape``).
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn


class Predictor(Protocol):
    """Anything callable as ``pred(x) -> Tensor`` (an ``nn.Module`` satisfies this)."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


def freeze(module: nn.Module) -> nn.Module:
    """Put ``module`` in eval mode and disable gradients on its parameters.

    Returns the same module for convenience. Note: ``.eval()`` will not reproduce a TF
    predictor that trained with dropout active under ``fit`` — document per predictor.
    """
    module.eval()
    module.requires_grad_(False)
    return module
