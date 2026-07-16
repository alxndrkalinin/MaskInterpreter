"""Regression MaskInterpreter (port of ``models/MaskInterpreterRegression.py``).

2D. Input is augmented with a gradient-of-**sum**-output channel and fed **directly**
to the adaptor — there is **no** learned conv front-end (unlike classification).
Two-sided PCC, L1 size.
"""

from __future__ import annotations

import torch
from torch import nn

from mask_interpreter.config import REGRESSION_LOSS, LossConfig, UNetConfig
from mask_interpreter.interpreter.base import VectorOutputInterpreter
from mask_interpreter.unet import UNet


class RegressionInterpreter(VectorOutputInterpreter):
    grad_reduce = "sum"

    def __init__(
        self,
        predictor: nn.Module,
        spatial_size: tuple[int, int],
        in_channels: int = 1,
        loss: LossConfig = REGRESSION_LOSS,
        unet: UNetConfig | None = None,
    ) -> None:
        super().__init__(predictor, loss)
        if len(spatial_size) != 2:
            raise ValueError(f"regression is 2D; spatial_size must be (H, W), got {spatial_size}")
        unet = unet or UNetConfig()
        # NO conv preproc: adaptor consumes the (in_channels + 1) augmented input directly.
        self.adaptor = UNet(
            (*spatial_size, in_channels + 1),
            base_filters=unet.base_filters,
            batch_norm=unet.batch_norm,
            final_activation="sigmoid",
            out_channels=1,
        )

    def generate_mask(self, augmented: torch.Tensor) -> torch.Tensor:
        return self.adaptor(augmented)
