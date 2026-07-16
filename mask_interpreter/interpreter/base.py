"""Shared MaskInterpreter machinery.

The three TF variants are **not** unifiable — they differ in generator front-end,
PCC clamp (one- vs two-sided), size term (MSE vs L1) and the ``stop`` / mask-size
metrics (§2 table of the plan). Base holds only the pieces that are genuinely common
(noise draw, adapt-with-noise, similarity MSE, total-loss assembly, grad-augmentation)
and exposes them as small, auditable helpers; each variant's ``training_step`` spells
out its own differences.

Gradient flow: the reference prediction ``pred(x)`` is computed under ``no_grad`` (it is
a constant, as in the TF code which computes it outside the ``GradientTape``); the
adapted-image prediction keeps its graph so gradients reach the mask/generator through
the (frozen-parameter) predictor.
"""

from __future__ import annotations

import torch
from torch import nn

from mask_interpreter.config import LossConfig
from mask_interpreter.metrics import pearson_corr
from mask_interpreter.predictors import freeze


class MaskInterpreterBase(nn.Module):
    def __init__(self, predictor: nn.Module, loss: LossConfig) -> None:
        super().__init__()
        self.predictor = freeze(predictor)
        self.loss_cfg = loss

    def train(self, mode: bool = True):  # noqa: D401
        """Keep the frozen predictor in eval mode even when the interpreter trains."""
        super().train(mode)
        self.predictor.eval()
        return self

    # --- shared helpers -------------------------------------------------

    @torch.no_grad()
    def reference_prediction(self, x: torch.Tensor) -> torch.Tensor:
        """Detached ``pred(x)`` — constant target for similarity/PCC."""
        return self.predictor(x)

    def draw_noise(
        self, mask: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Single-channel noise ``~ N(0, noise_scale)`` shaped like the mask.

        TF draws ``tf.random.normal(tf.shape(importance_mask))`` — one channel that
        broadcasts across the image channels in the adapt step.
        """
        noise = torch.randn(mask.shape, dtype=mask.dtype, device=mask.device, generator=generator)
        return noise * self.loss_cfg.noise_scale

    @staticmethod
    def adapt(x: torch.Tensor, mask: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """``mask*x + noise*(1-mask)`` (mask/noise broadcast over image channels)."""
        return mask * x + noise * (1.0 - mask)

    @staticmethod
    def similarity_mse(reference: torch.Tensor, adapted_pred: torch.Tensor) -> torch.Tensor:
        return torch.mean((reference - adapted_pred) ** 2)

    def total_loss(
        self, sim: torch.Tensor, size: torch.Tensor, pcc_loss: torch.Tensor
    ) -> torch.Tensor:
        c = self.loss_cfg
        return (
            sim * c.similarity_loss_weight
            + size * c.mask_loss_weight
            + pcc_loss * c.target_loss_weight
        )

    def grad_augment(self, x: torch.Tensor, reduce: str) -> torch.Tensor:
        """Append a gradient-magnitude channel (clf: max(pred); reg: sum(pred)).

        Uses the **raw** predictor output (no added softmax), L2 over channels,
        per-sample min-max normalise, then ``.detach()`` and concat — matching
        ``MaskInterpreter{CLF,Regression}._augment_input_with_gradients``.
        """
        # The grad channel needs autograd locally even when forward() is called under
        # torch.no_grad() (inference). Force it on, then detach the result.
        with torch.enable_grad():
            x_req = x.detach().clone().requires_grad_(True)
            preds = self.predictor(x_req)
            if reduce == "max":
                scalar = preds.amax(dim=1)
            elif reduce == "sum":
                scalar = preds.sum(dim=1)
            else:  # pragma: no cover - guarded by callers
                raise ValueError(f"reduce must be 'max' or 'sum', got {reduce!r}")
            grad = torch.autograd.grad(scalar.sum(), x_req, create_graph=False)[0]
        grad_norm = grad.pow(2).sum(dim=1, keepdim=True).sqrt()
        dims = tuple(range(1, grad_norm.ndim))
        mn = grad_norm.amin(dim=dims, keepdim=True)
        mx = grad_norm.amax(dim=dims, keepdim=True)
        grad_norm = (grad_norm - mn) / (mx - mn + 1e-8)
        return torch.cat([x, grad_norm.detach()], dim=1)

    def pcc(
        self, reference: torch.Tensor, adapted_pred: torch.Tensor, weights: torch.Tensor | None
    ) -> torch.Tensor:
        return pearson_corr(reference, adapted_pred, weights)


class VectorOutputInterpreter(MaskInterpreterBase):
    """Shared step for the classification/regression variants (vector predictor output).

    Both are 2D, use grad-augmentation, a **two-sided** PCC clamp, an ``L1`` size term,
    a ``1 - mean(mask)`` size metric and ``stop = pcc_loss + mean(mask)``. The only
    per-variant differences are the grad reduction (``max`` vs ``sum``) and whether a
    learned Conv2d front-end precedes the adaptor — both supplied by the subclass via
    ``grad_reduce`` and ``generate_mask``.
    """

    grad_reduce: str = "max"

    def generate_mask(self, augmented: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        augmented = self.grad_augment(x, self.grad_reduce)
        return self.generate_mask(augmented)

    def training_step(
        self, batch, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch

        reference = self.reference_prediction(x)
        augmented = self.grad_augment(x, self.grad_reduce)
        mask = self.generate_mask(augmented)
        noise = self.draw_noise(mask, generator)
        adapted = self.adapt(x, mask, noise)
        adapted_pred = self.predictor(adapted)

        sim = self.similarity_mse(reference, adapted_pred)
        size = torch.mean(torch.abs(mask))  # L1
        pcc = self.pcc(reference, adapted_pred, None)
        target = self.loss_cfg.pcc_target
        pcc_loss = torch.abs(target - pcc)  # two-sided
        mean_mask = torch.mean(mask)

        total = self.total_loss(sim, size, pcc_loss)
        metrics = {
            "similarity_loss": float(sim.detach()),
            "mask_loss": float(size.detach()),
            "pcc": float(pcc.detach()),
            "pcc_loss": float(pcc_loss.detach()),
            "importance_mask_size": float((1.0 - mean_mask).detach()),  # 1 - mean(mask)
            "stop": float((pcc_loss + mean_mask).detach()),            # pcc_loss + mean(mask)
            "total_loss": float(total.detach()),
        }
        return total, metrics
