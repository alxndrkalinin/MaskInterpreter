"""Configuration dataclasses (replace mutating ``global_vars`` at import time).

Per-variant defaults are seeded from the original TF sources — there is no single
"faithful default" (review D#3):

- image2image  : ``MaskInterpreter.py`` + ``train.py``
- classification: ``MaskInterpreterCLF.py`` ``__main__``
- regression    : ``MaskInterpreterRegression.py`` ``__main__``
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LossConfig:
    """Weights and target for the faithful three-term loss.

    ``total = w_sim*sim + w_mask*size + w_pcc*pcc``
    """

    similarity_loss_weight: float = 1.0
    mask_loss_weight: float = 1.0
    target_loss_weight: float = 10.0
    noise_scale: float = 1.5
    pcc_target: float = 0.9


# Per-variant seed configs (see module docstring for provenance).
IMAGE2IMAGE_LOSS = LossConfig(
    similarity_loss_weight=1.0,
    mask_loss_weight=1.0,
    target_loss_weight=10.0,
    noise_scale=1.5,
    pcc_target=0.9,
)

CLASSIFICATION_LOSS = LossConfig(
    similarity_loss_weight=1.0,
    mask_loss_weight=1.0,
    target_loss_weight=1.75,
    noise_scale=0.5,
    pcc_target=0.95,
)

REGRESSION_LOSS = LossConfig(
    similarity_loss_weight=1.0,
    mask_loss_weight=1.0,
    target_loss_weight=2.5,
    noise_scale=0.5,
    pcc_target=0.95,
)


@dataclass
class UNetConfig:
    """Adaptor U-Net architecture knobs (see ``UNETO.get_unet``)."""

    base_filters: int = 16
    batch_norm: bool = True
    final_activation: str = "sigmoid"  # mask generator uses sigmoid; predictor uses linear


@dataclass
class TrainConfig:
    learning_rate: float = 1e-4
    epochs: int = 100
    batch_size: int = 4
    early_stop_patience: int = 5
    checkpoint_monitor: str = "val_stop"
    amp: bool = False
    device: str = "cuda"
    dtype: str = "float32"


@dataclass
class Config:
    loss: LossConfig = field(default_factory=LossConfig)
    unet: UNetConfig = field(default_factory=UNetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
