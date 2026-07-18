"""FOV / patch dataset — a torch-native port of the in-scope ``DataGen`` numpy path.

Ports the ``patches_from_image > 1`` sampling branch of ``dataset.DataGen``: pick a
random image within a ``[min_percentage, max_percentage)`` window, optionally rot90-
augment, extract the input/target channels, normalize, optionally concat an in-silico
prediction channel, and sample a random patch. The Keras ``Sequence`` base, the eager
buffer, the threading and the SSD cache are dropped in favour of a plain map-style
``Dataset`` (use a ``DataLoader`` with ``num_workers`` for parallelism).

**In scope** (kept): ``dilate`` (weighted-PCC training uses ``target="structure_seg"``,
``dilate=True``) and ``predictors`` (an in-silico channel concatenated to the input,
mirrored in ``test.py``).

**Out of scope** (deliberately not ported — belong to other model types): ``pairs`` /
``masking_pair`` / ``for_clf`` / ``crop_edge`` / ``mask`` / ``input_as_y`` /
``output_as_x``. Passing them raises rather than silently doing nothing.

I/O uses ``tifffile`` + ``pandas`` (no ``cell_imaging_utils``). FOV tiffs are assumed
channel-first ``(C, Z, Y, X)`` by default; set ``channel_axis`` otherwise.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import tifffile
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from mask_interpreter.data import transforms as T

_UNSUPPORTED = ("pairs", "masking_pair", "for_clf", "crop_edge", "mask", "input_as_y", "output_as_x")


def read_tiff(path: str, channel_axis: int = 0) -> np.ndarray:
    """Read a multi-channel tiff as ``(C, Z, Y, X)`` (moves ``channel_axis`` to front)."""
    arr = np.asarray(tifffile.imread(path))
    if channel_axis != 0:
        arr = np.moveaxis(arr, channel_axis, 0)
    return arr


def get_channel(arr: np.ndarray, index: int) -> np.ndarray:
    """Return channel ``index`` of a ``(C, Z, Y, X)`` array as ``(Z, Y, X)``."""
    return arr[int(index)]


def dilate_zyx(image: np.ndarray, kernel: int = 25) -> np.ndarray:
    """Per-Z-slice grey dilation with a ``kernel x kernel`` window (cv2.dilate parity)."""
    return ndimage.grey_dilation(image, size=(1, kernel, kernel))


class FOVPatchDataset(Dataset):
    def __init__(
        self,
        image_list_csv: str | None = None,
        input_col: str | int = "input",
        target_col: str | int = "target",
        patch_size: tuple[int, ...] = (32, 128, 128),
        image_path_col: str = "path_tiff",
        min_percentage: float = 0.0,
        max_percentage: float = 1.0,
        patches_per_image: int = 32,
        length: int | None = None,
        norm: bool = True,
        norm_type: str = "std",
        augment: bool = False,
        dilate: bool = False,
        dilate_kernel: int = 25,
        predictors=None,
        channel_axis: int = 0,
        seed: int | None = None,
        images=None,
        **unsupported,
    ) -> None:
        """FOV patches from a CSV/tiff list (``image_list_csv``) **or** in-memory FOVs (``images``).

        Provide exactly one source. ``images`` is a sequence of FOVs, each a channel-first
        ``(C, Z, Y, X)`` array/tensor (``input_col``/``target_col`` are integer channel indices)
        or a role→volume ``dict`` (``input_col``/``target_col`` are its keys). Channels are
        ``(Z, Y, X)`` (or ``(Y, X)`` when ``patch_size[0]==1``).
        """
        for key in unsupported:
            if key in _UNSUPPORTED:
                raise NotImplementedError(
                    f"DataGen option {key!r} is out of scope for the port (other model type)"
                )
            raise TypeError(f"unexpected argument {key!r}")

        if (image_list_csv is None) == (images is None):
            raise ValueError("provide exactly one of `image_list_csv` or `images` (in-memory FOVs)")
        self._images = list(images) if images is not None else None
        self.df = pd.read_csv(image_list_csv) if image_list_csv is not None else None
        self.n = len(self._images) if self._images is not None else len(self.df)
        self.input_col = input_col
        self.target_col = target_col
        self.image_path_col = image_path_col
        self.patch_size = tuple(int(p) for p in patch_size)
        self.is_2d = self.patch_size[0] == 1
        self.min_idx = int(self.n * min_percentage)
        self.max_idx = int(self.n * max_percentage)
        self.patches_per_image = patches_per_image
        self.norm = norm
        self.norm_type = norm_type
        self.augment = augment
        self.dilate = dilate
        self.dilate_kernel = dilate_kernel
        self.predictors = predictors
        self.channel_axis = channel_axis
        self.seed = seed
        self._length = length if length is not None else (self.max_idx - self.min_idx) * patches_per_image

    def __len__(self) -> int:
        return self._length

    def _rng(self, index: int) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self.seed * 10_000_019 + index)

    def _coerce_volume(self, a) -> np.ndarray:
        """Coerce an in-memory channel to a float32 ``(Z, Y, X)`` volume (2D -> Z=1)."""
        if torch.is_tensor(a):
            a = a.detach().cpu().numpy()
        a = np.asarray(a).astype(np.float32)
        if self.is_2d and a.ndim == 2:
            a = a[None]  # (Y, X) -> (1, Y, X)
        if a.ndim != 3:
            want = "(Z, Y, X) or (Y, X)" if self.is_2d else "(Z, Y, X)"
            raise ValueError(f"FOV channel must be {want}; got shape {a.shape}")
        return a

    def _load_channels(self, image_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the raw ``(Z, Y, X)`` input and target channels for one FOV."""
        if self._images is not None:
            fov = self._images[image_index]
            if torch.is_tensor(fov):
                fov = fov.detach().cpu().numpy()
            if isinstance(fov, dict):
                inp, tgt = fov[self.input_col], fov[self.target_col]
            else:
                arr = np.asarray(fov)
                inp, tgt = arr[int(self.input_col)], arr[int(self.target_col)]
            return self._coerce_volume(inp), self._coerce_volume(tgt)
        path = self.df.iloc[image_index][self.image_path_col]
        arr = read_tiff(path, self.channel_axis).astype(np.float32)
        return (get_channel(arr, int(self.df.iloc[image_index][self.input_col])),
                get_channel(arr, int(self.df.iloc[image_index][self.target_col])))

    def __getitem__(self, index: int):
        rng = self._rng(index)
        image_index = int(rng.integers(self.min_idx, self.max_idx))
        input_image, target_image = self._load_channels(image_index)  # (Z, Y, X) each

        k = int(rng.integers(0, 4)) if self.augment else 0
        if k:  # rotate the last two (Y, X) axes — per-channel, identical to rotating the FOV
            input_image = T.rot90(input_image, k=k, axes=(input_image.ndim - 2, input_image.ndim - 1))
            target_image = T.rot90(target_image, k=k, axes=(target_image.ndim - 2, target_image.ndim - 1))

        if self.norm:
            input_image = T.normalize(input_image, self.norm_type)

        if self.dilate:
            target_image = dilate_zyx(target_image, self.dilate_kernel)
        elif self.norm:
            target_image = T.normalize(target_image, self.norm_type)

        # channel-first stacks: (C, Z, Y, X)
        input_stack = input_image[None]
        if self.predictors is not None:
            pred = self._in_silico(input_image)
            input_stack = np.concatenate([input_stack, pred[None]], axis=0)
        target_stack = target_image[None]

        # sample one random patch over the spatial dims
        spatial = input_stack.shape[1:]
        idx = T.sample_patch_indices(spatial, self.patch_size, rng)
        sl = (slice(None),) + tuple(slice(a, b) for a, b in idx)
        input_patch = input_stack[sl]
        target_patch = target_stack[sl]

        if self.is_2d:  # drop the singleton Z -> (C, Y, X)
            input_patch = input_patch[:, 0]
            target_patch = target_patch[:, 0]

        return torch.from_numpy(np.ascontiguousarray(input_patch)), torch.from_numpy(
            np.ascontiguousarray(target_patch)
        )

    def _in_silico(self, input_image: np.ndarray) -> np.ndarray:
        """Run the predictor to produce an extra input channel (mirrors DataGen)."""
        fn = self.predictors
        with torch.no_grad():
            x = torch.from_numpy(input_image[None, None]).float()
            pred = fn(x)
        # Reshape (not squeeze) back to the input's spatial shape: squeeze() would also
        # collapse a legitimate size-1 spatial axis (e.g. a Z==1 FOV), corrupting the concat.
        pred = pred.detach().cpu().numpy().astype(np.float32).reshape(input_image.shape)
        if self.norm:
            pred = T.normalize(pred, self.norm_type)
        return pred
