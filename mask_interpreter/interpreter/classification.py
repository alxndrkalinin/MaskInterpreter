"""Classification MaskInterpreter (port of ``models/MaskInterpreterCLF.py``).

2D. Input is augmented with a gradient-of-**max**-prediction channel, then a single
learned ``Conv2d(32, relu)`` front-end precedes the adaptor. Two-sided PCC, L1 size.
"""

from __future__ import annotations

import torch
from torch import nn

from mask_interpreter.config import CLASSIFICATION_LOSS, LossConfig, UNetConfig
from mask_interpreter.interpreter.base import VectorOutputInterpreter
from mask_interpreter.unet import UNet


class ClassificationInterpreter(VectorOutputInterpreter):
    grad_reduce = "max"

    def __init__(
        self,
        predictor: nn.Module,
        spatial_size: tuple[int, int],
        in_channels: int = 3,
        preproc_filters: int = 32,
        loss: LossConfig = CLASSIFICATION_LOSS,
        unet: UNetConfig | None = None,
    ) -> None:
        super().__init__(predictor, loss)
        if len(spatial_size) != 2:
            raise ValueError(f"classification is 2D; spatial_size must be (H, W), got {spatial_size}")
        unet = unet or UNetConfig()
        # augmented input has in_channels + 1 (gradient channel); Conv2d front-end -> preproc_filters.
        self.preproc = nn.Sequential(
            nn.Conv2d(in_channels + 1, preproc_filters, 3, padding=1), nn.ReLU()
        )
        self.adaptor = UNet(
            (*spatial_size, preproc_filters),
            base_filters=unet.base_filters,
            batch_norm=unet.batch_norm,
            final_activation="sigmoid",
            out_channels=1,
        )

    def generate_mask(self, augmented: torch.Tensor) -> torch.Tensor:
        return self.adaptor(self.preproc(augmented))
