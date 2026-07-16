"""Single-cell dataset (tiff triplets), companion-style but reimplemented.

Each CSV row references separate single-channel tiffs (signal / target [/ seg]). Volumes
are normalized, optionally edge-padded in Z to a minimum depth, optionally rot90-
augmented, and returned channel-first: ``(input, target)`` with shapes ``(1, Z, Y, X)``
(or ``(1, Y, X)`` for 2D). ``target_col`` selects whichever channel serves as the second
batch element — the target organelle for regular PCC, or the seg for weighted PCC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import Dataset

from mask_interpreter.data import transforms as T


class SingleCellDataset(Dataset):
    def __init__(
        self,
        csv: str,
        signal_col: str = "signal_path",
        target_col: str = "target_path",
        norm: bool = True,
        norm_type: str = "std",
        pad_z: int | None = None,
        augment: bool = False,
        seed: int | None = None,
    ) -> None:
        self.df = pd.read_csv(csv)
        self.signal_col = signal_col
        self.target_col = target_col
        self.norm = norm
        self.norm_type = norm_type
        self.pad_z = pad_z
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.df)

    def _rng(self, index: int) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self.seed * 10_000_019 + index)

    def _prep(self, image: np.ndarray, k: int, do_norm: bool) -> np.ndarray:
        image = image.astype(np.float32)
        if do_norm and self.norm:
            image = T.normalize(image, self.norm_type)
        if self.pad_z is not None and image.ndim == 3:
            image = T.pad_z_to(image, self.pad_z, axis=0)
        if k:
            image = T.rot90(image, k=k, axes=(image.ndim - 2, image.ndim - 1))
        return image

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        rng = self._rng(index)
        k = int(rng.integers(0, 4)) if self.augment else 0

        signal = self._prep(np.asarray(tifffile.imread(row[self.signal_col])), k, do_norm=True)
        target = self._prep(np.asarray(tifffile.imread(row[self.target_col])), k, do_norm=True)

        input_t = torch.from_numpy(np.ascontiguousarray(signal[None]))
        target_t = torch.from_numpy(np.ascontiguousarray(target[None]))
        return input_t, target_t
