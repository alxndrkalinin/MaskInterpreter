"""Pure-numpy patch collect/assemble helpers (framework-agnostic; ported verbatim
from ``utils/utils.py``). Triangular overlap weighting favours patch centres."""

from __future__ import annotations

import numpy as np
import scipy.signal


def get_weights(shape: tuple[int, ...]) -> np.ndarray:
    shape_in = shape
    shape = shape[1:]
    weights = 1
    for idx_d in range(len(shape)):
        slicey = [np.newaxis] * len(shape)
        slicey[idx_d] = slice(None)
        size = shape[idx_d]
        values = scipy.signal.windows.triang(size)
        weights = weights * values[tuple(slicey)]
    return np.broadcast_to(weights, shape_in).astype(np.float32)


def slice_image(image_ndarray: np.ndarray, indexes: list) -> np.ndarray:
    n_dim = len(image_ndarray.shape)
    slices = [slice(None)] * n_dim
    for i in range(len(indexes)):
        if indexes[i] is None:
            slices[i] = slice(None)
        else:
            slices[i] = slice(indexes[i][0], indexes[i][1])
    return image_ndarray[tuple(slices)]


def collect_patchs(px_start, py_start, pz_start, px_end, py_end, pz_end, image, patch_size, xy_step, z_step):
    pz, px, py = pz_start, px_start, py_start
    patchs = []
    while pz <= pz_end - patch_size[0]:
        while px <= px_end - patch_size[1]:
            while py <= py_end - patch_size[2]:
                px_start_patch = px - px_start
                py_start_patch = py - py_start
                s = [
                    (pz, pz + patch_size[0]),
                    (px_start_patch, px_start_patch + patch_size[1]),
                    (py_start_patch, py_start_patch + patch_size[2]),
                ]
                patchs.append(slice_image(image, s))
                py += min(xy_step, max(1, py_end - patch_size[2] - py))
            py = py_start
            px += min(xy_step, max(1, px_end - patch_size[1] - px))
        px = px_start
        pz += min(z_step, max(1, pz_end - patch_size[0] - pz))
    return np.array(patchs)


def assemble_image(
    px_start, py_start, pz_start, px_end, py_end, pz_end,
    patchs, weights, assembled_image_shape, patch_size, xy_step, z_step,
):
    patchs = np.array(patchs)
    assembled_images = np.zeros((patchs.shape[0], *assembled_image_shape))
    pz, px, py = pz_start, px_start, py_start
    i = 0
    while pz <= pz_end - patch_size[0]:
        while px <= px_end - patch_size[1]:
            while py <= py_end - patch_size[2]:
                px_start_patch = px - px_start
                py_start_patch = py - py_start
                patch_slice = (
                    slice(pz, pz + patch_size[0]),
                    slice(px_start_patch, px_start_patch + patch_size[1]),
                    slice(py_start_patch, py_start_patch + patch_size[2]),
                )
                for j in range(patchs.shape[0]):
                    assembled_images[j][patch_slice] += patchs[j][i] * weights
                py += min(xy_step, max(1, py_end - patch_size[2] - py))
                i += 1
            py = py_start
            px += min(xy_step, max(1, px_end - patch_size[1] - px))
        px = px_start
        pz += min(z_step, max(1, pz_end - patch_size[0] - pz))
    return assembled_images
