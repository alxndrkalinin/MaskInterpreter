"""Plain-PyTorch training loop (no Lightning).

Ports the behaviour of the TF ``SaveModelCallback`` + ``EarlyStopping`` used to train
the mask generators:

- **Checkpoint** the best (lowest) ``monitor`` value, gated by an optional ``term``
  guard (save only when ``logs[term] > term_value`` — TF used ``val_pcc > 0.03``).
  The TF callback's silent ``except: save_weights`` fallback is intentionally dropped;
  a failed save raises.
- **Early stop** on ``early_stop_monitor`` with ``patience``; optionally restore best
  weights.

The validation pass is **not** wrapped in ``no_grad`` — the clf/reg variants need
autograd for their gradient-augmentation channel. We simply never call ``backward``
there and free the graph each step.
"""

from __future__ import annotations

import copy
from collections import defaultdict

import torch
from torch import nn


def _to_device(batch, device):
    if isinstance(batch, (tuple, list)):
        return type(batch)(_to_device(b, device) for b in batch)
    if torch.is_tensor(batch):
        return batch.to(device)
    return batch


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        lr: float = 1e-4,
        device: str = "cuda",
        amp: bool = False,
        checkpoint_path: str | None = None,
        monitor: str = "val_stop",
        mode: str = "min",
        term: str | None = "val_pcc",
        term_value: float | None = 0.03,
        early_stop_monitor: str | None = "val_stop",
        patience: int = 5,
        restore_best_weights: bool = True,
    ) -> None:
        if not torch.cuda.is_available() and device.startswith("cuda"):
            device = "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer or torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad), lr=lr
        )
        self.amp = amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.checkpoint_path = checkpoint_path
        self.monitor = monitor
        self.mode = mode
        self.term = term
        self.term_value = term_value
        self.early_stop_monitor = early_stop_monitor
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.history: list[dict[str, float]] = []

    def _better(self, current: float, best: float) -> bool:
        return current < best if self.mode == "min" else current > best

    def _run_epoch(self, loader, train: bool) -> dict[str, float]:
        self.model.train(train)
        sums: dict[str, float] = defaultdict(float)
        n = 0
        for batch in loader:
            batch = _to_device(batch, self.device)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, enabled=self.amp):
                    total, metrics = self.model.training_step(batch)
                self.scaler.scale(total).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total, metrics = self.model.training_step(batch)
                del total
            for k, v in metrics.items():
                sums[k] += v
            n += 1
        return {k: v / max(n, 1) for k, v in sums.items()}

    def fit(self, train_loader, val_loader=None, epochs: int = 100) -> list[dict[str, float]]:
        best_monitor = float("inf") if self.mode == "min" else float("-inf")
        best_ckpt = float("inf") if self.mode == "min" else float("-inf")
        best_state = None
        wait = 0

        for epoch in range(epochs):
            train_logs = self._run_epoch(train_loader, train=True)
            logs = dict(train_logs)
            if val_loader is not None:
                val_logs = self._run_epoch(val_loader, train=False)
                logs.update({f"val_{k}": v for k, v in val_logs.items()})
            self.history.append({"epoch": epoch, **logs})

            # checkpoint (best monitor, gated by term guard)
            if self.checkpoint_path and self.monitor in logs:
                cur = logs[self.monitor]
                term_ok = (
                    self.term is None
                    or self.term_value is None
                    or logs.get(self.term, float("-inf")) > self.term_value
                )
                if self._better(cur, best_ckpt) and term_ok:
                    best_ckpt = cur
                    torch.save(self.model.state_dict(), self.checkpoint_path)

            # early stopping
            if self.early_stop_monitor and self.early_stop_monitor in logs:
                cur = logs[self.early_stop_monitor]
                if self._better(cur, best_monitor):
                    best_monitor = cur
                    wait = 0
                    if self.restore_best_weights:
                        best_state = copy.deepcopy(self.model.state_dict())
                else:
                    wait += 1
                    if wait >= self.patience:
                        if self.restore_best_weights and best_state is not None:
                            self.model.load_state_dict(best_state)
                        break

        if self.restore_best_weights and best_state is not None and wait < self.patience:
            self.model.load_state_dict(best_state)
        return self.history
