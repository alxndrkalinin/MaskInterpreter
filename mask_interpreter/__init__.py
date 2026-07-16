"""PyTorch port of MaskInterpreter.

Self-supervised interpretability: a frozen predictor + a trained mask-generator
that produces an importance mask in [0, 1]. Non-important regions are replaced
with noise; the predictor's output on the adapted image should stay close to its
output on the original.
"""

from mask_interpreter.metrics import pearson_corr, pearson_corr_np
from mask_interpreter.unet import UNet
from mask_interpreter.predictors import freeze
from mask_interpreter.interpreter.image2image import Image2ImageInterpreter
from mask_interpreter.interpreter.classification import ClassificationInterpreter
from mask_interpreter.interpreter.regression import RegressionInterpreter

__all__ = [
    "pearson_corr",
    "pearson_corr_np",
    "UNet",
    "freeze",
    "Image2ImageInterpreter",
    "ClassificationInterpreter",
    "RegressionInterpreter",
]
