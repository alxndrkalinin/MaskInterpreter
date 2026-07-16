"""Image-to-image MaskInterpreter (port of ``models/MaskInterpreter.py``).

Front-end: two **learned** convs (32 filters, relu) applied to the image and to the
predictor's output, concatenated (64 channels) then fed to the adaptor U-Net.
PCC term is **one-sided** (``|target - min(pcc, target)|`` — no penalty past target).
Size term is ``MSE(mask, 0)``; the mask-size metric is ``mean(mask)`` and the ``stop``
metric is ``pcc_loss + MSE(mask, 0)``.

The TF ``MaskInterpreter`` is 3D-only; a 2D image2image path (``ndim=2``) has no TF
reference and is built here by analogy (Conv2d front-end, 2D adaptor).
"""

from __future__ import annotations

import torch
from torch import nn

from mask_interpreter.config import IMAGE2IMAGE_LOSS, LossConfig, UNetConfig
from mask_interpreter.interpreter.base import MaskInterpreterBase
from mask_interpreter.unet import UNet


class Image2ImageInterpreter(MaskInterpreterBase):
    def __init__(
        self,
        predictor: nn.Module,
        spatial_size: tuple[int, ...],
        in_channels: int = 1,
        pred_channels: int = 1,
        ndim: int = 3,
        preproc_filters: int = 32,
        loss: LossConfig = IMAGE2IMAGE_LOSS,
        unet: UNetConfig | None = None,
        weighted_pcc: bool = False,
    ) -> None:
        super().__init__(predictor, loss)
        if ndim not in (2, 3):
            raise ValueError(f"ndim must be 2 or 3, got {ndim}")
        if len(spatial_size) != ndim:
            raise ValueError(f"spatial_size {spatial_size} does not match ndim={ndim}")
        self.ndim = ndim
        self.weighted_pcc = weighted_pcc
        unet = unet or UNetConfig()

        conv = nn.Conv3d if ndim == 3 else nn.Conv2d
        self.image_conv = nn.Sequential(conv(in_channels, preproc_filters, 3, padding=1), nn.ReLU())
        self.pred_conv = nn.Sequential(conv(pred_channels, preproc_filters, 3, padding=1), nn.ReLU())

        adaptor_in = preproc_filters * 2
        input_size = (*spatial_size, adaptor_in)
        self.adaptor = UNet(
            input_size,
            base_filters=unet.base_filters,
            batch_norm=unet.batch_norm,
            final_activation="sigmoid",
            out_channels=1,
        )

    def generate_mask(self, x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        processed = torch.cat([self.image_conv(x), self.pred_conv(reference)], dim=1)
        return self.adaptor(processed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reference = self.reference_prediction(x)
        return self.generate_mask(x, reference)

    def training_step(
        self, batch, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        x = batch[0]
        seg = batch[1] if (self.weighted_pcc and len(batch) > 1) else None

        reference = self.reference_prediction(x)
        mask = self.generate_mask(x, reference)
        noise = self.draw_noise(mask, generator)
        adapted = self.adapt(x, mask, noise)
        adapted_pred = self.predictor(adapted)

        sim = self.similarity_mse(reference, adapted_pred)
        size = torch.mean(mask ** 2)  # MSE(mask, 0)
        pcc = self.pcc(reference, adapted_pred, seg)
        target = self.loss_cfg.pcc_target
        pcc_loss = torch.abs(target - torch.minimum(pcc, torch.as_tensor(target, dtype=pcc.dtype, device=pcc.device)))

        total = self.total_loss(sim, size, pcc_loss)
        metrics = {
            "similarity_loss": float(sim.detach()),
            "mask_loss": float(size.detach()),
            "pcc": float(pcc.detach()),
            "pcc_loss": float(pcc_loss.detach()),
            "importance_mask_size": float(mask.mean().detach()),  # mean(mask)
            "stop": float((pcc_loss + size).detach()),            # pcc_loss + MSE(mask, 0)
            "total_loss": float(total.detach()),
        }
        return total, metrics
