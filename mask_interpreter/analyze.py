"""Evaluation pipeline — PyTorch port of ``mg_analyzer.py``.

Ports ``analyze_th`` (modes ``agg`` / ``loo`` / ``mask`` / ``regular``),
``calc_unet_pcc`` and ``find_noise_scale`` for the 3D image-to-image path. Fidelity /
fix notes (§Phase-4 of the plan):

- **Noise is drawn ONCE per image and reused across all thresholds** (TF draws it
  outside the ``for th`` loop). A per-threshold redraw would shift every CSV value.
- ``predict`` runs batched inference on device under ``no_grad``; non-divisible patch
  counts use a trailing partial batch; on CUDA OOM it halves the batch and retries,
  then raises a real error (the TF ``bare except: raise("str")`` is not preserved).
- ``preprocess_image`` is rewritten on ``tifffile`` + ``pandas`` (no
  ``cell_imaging_utils``), reproducing the 6-channel FOV contract, the per-channel
  normalize flags and the ``Z<32`` edge-pad.
- The broad ``except -> -1`` sentinel is removed; the zero-organelle-intersection
  context division is detected and raised explicitly.
- Two models per image: the frozen predictor (``interpreter.predictor``) and the
  interpreter (``interpreter(x) -> mask``).
- Output names are cleaned: ``pcc_results.csv`` / ``mask_size_results.csv`` /
  ``context_results.csv`` and ``input_/target_/unet_prediction_/mask_/noisy_*`` tiffs.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import tifffile
import torch

from mask_interpreter.data.fov import dilate_zyx, read_tiff
from mask_interpreter.data.patches import assemble_image, collect_patchs, get_weights, slice_image
from mask_interpreter.data.transforms import normalize_std, pad_z_to
from mask_interpreter.metrics import pearson_corr_np

FOV_COLUMNS = ("input", "target", "structure_seg", "channel_dna", "channel_membrane", "membrane_seg")
FOV_NORMALIZE = (True, True, False, True, True, False)


def _create_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def preprocess_image(
    df: pd.DataFrame,
    image_index: int,
    columns: list[str],
    normalize,
    path_col: str = "path_tiff",
    min_z: int = 32,
    slice_by=None,
    channel_axis: int = 0,
):
    """Load selected channels of a FOV tiff → list of ``(Z, Y, X, 1)`` arrays (or None).

    Reproduces ``utils.preprocess_image``: per-channel std-normalize, optional spatial
    crop (``slice_by``), and the ``Z<min_z`` edge-pad with ``ceil((min_z-Z)/2)`` both
    sides. A missing/NaN channel column yields ``None``.
    """
    row = df.iloc[image_index]
    arr = read_tiff(row[path_col], channel_axis).astype(np.float32)
    images: list[np.ndarray | None] = []
    for i, col in enumerate(columns):
        ch = row[col] if col in df.columns else None
        if ch is None or (isinstance(ch, float) and math.isnan(ch)):
            images.append(None)
            continue
        image = arr[int(ch)]  # (Z, Y, X)
        image = np.expand_dims(image, axis=-1)  # (Z, Y, X, 1)
        if normalize is True or normalize[i]:
            image = normalize_std(image)
        if slice_by is not None:
            image = slice_image(image, slice_by)
        if image.shape[0] < min_z:
            image = pad_z_to(image, min_z, axis=0)
        images.append(image)
    return images


def _as_zyx1(a) -> np.ndarray:
    """Coerce a channel array/tensor to float32 ``(Z, Y, X, 1)``."""
    if torch.is_tensor(a):
        a = a.detach().cpu().numpy()
    a = np.asarray(a).astype(np.float32)
    if a.ndim == 4 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim != 3:
        raise ValueError(f"FOV channel must be (Z, Y, X) or (Z, Y, X, 1); got shape {a.shape}")
    return a[..., None]


def preprocess_fov(fov, columns, normalize, min_z: int = 32, slice_by=None):
    """In-memory analogue of ``preprocess_image`` → list of ``(Z, Y, X, 1)`` arrays (or None).

    Applies the *same* per-channel std-normalize / crop / Z-pad as the tiff path, but reads
    channels from a FOV already in memory. ``fov`` is either:

    - a ``dict`` mapping role names (the ``columns`` — e.g. ``"input"``, ``"target"``,
      ``"structure_seg"``) to ``(Z, Y, X)`` / ``(Z, Y, X, 1)`` numpy arrays or torch tensors;
      absent roles yield ``None``; or
    - a channel-first ``(C, Z, Y, X)`` array/tensor whose channels are taken in ``columns``
      order (``[input, target, structure_seg, channel_dna, channel_membrane, membrane_seg]``);
      channels beyond ``C`` yield ``None``.
    """
    if torch.is_tensor(fov):
        fov = fov.detach().cpu().numpy()
    is_dict = isinstance(fov, dict)
    arr = None if is_dict else np.asarray(fov)
    images: list[np.ndarray | None] = []
    for i, col in enumerate(columns):
        ch = fov.get(col) if is_dict else (arr[i] if i < arr.shape[0] else None)
        if ch is None:
            images.append(None)
            continue
        image = _as_zyx1(ch)
        if normalize is True or normalize[i]:
            image = normalize_std(image)
        if slice_by is not None:
            image = slice_image(image, slice_by)
        if image.shape[0] < min_z:
            image = pad_z_to(image, min_z, axis=0)
        images.append(image)
    return images


def _context_ratio(mask_binary: np.ndarray, seg: np.ndarray) -> tuple[float, float]:
    """Return ``(mask_size_fraction, context_ratio)``.

    An **empty mask** (all-zero at this threshold) is a legitimate threshold-sweep tail
    outcome: report ``(0.0, NaN)`` rather than crashing. A **non-empty mask with zero
    organelle intersection** means the organelle segmentation is absent/misaligned — a
    genuine data problem — so raise (replaces the TF broad-except -> ``-1`` sentinel).
    """
    mask_size_raw = np.sum(mask_binary, dtype=np.float64)
    if mask_size_raw == 0:
        return 0.0, float("nan")
    intersection = np.sum(mask_binary * seg, dtype=np.float64)
    if intersection == 0:
        raise ValueError("zero mask-organelle intersection: cannot compute context ratio")
    mask_organelle_intersection = intersection / mask_size_raw
    mask_size = mask_size_raw / np.prod(mask_binary.shape)
    context = 1.0 / (mask_organelle_intersection / mask_size)
    return float(mask_size), float(context)


class Analyzer:
    def __init__(
        self,
        interpreter: torch.nn.Module,
        data=None,
        input_col: str = "input",
        target_col: str = "target",
        patch_size: tuple[int, ...] = (32, 128, 128, 1),
        xy_step: int = 64,
        z_step: int = 16,
        batch_size: int = 4,
        device: str = "cuda",
        path_col: str = "path_tiff",
        channel_axis: int = 0,
        seg_col: str = "structure_seg",
        nuc_col: str = "channel_dna",
        mem_col: str = "channel_membrane",
        mem_seg_col: str = "membrane_seg",
        images=None,
    ) -> None:
        """Analyze FOVs from a CSV/tiff list (``data``) **or** from in-memory FOVs (``images``).

        Provide exactly one source. ``data`` is a CSV path / DataFrame mapping ``*_col`` names
        to channel indices inside per-row ``path_col`` tiffs (the original path). ``images`` is a
        sequence of in-memory FOVs — each a role→array ``dict`` or a channel-first
        ``(C, Z, Y, X)`` array/tensor (see ``preprocess_fov``); the per-method ``images=range(N)``
        argument then selects indices into this sequence.
        """
        if (data is None) == (images is None):
            raise ValueError("provide exactly one of `data` (CSV/DataFrame) or `images` (in-memory FOVs)")
        if not torch.cuda.is_available() and str(device).startswith("cuda"):
            device = "cpu"
        self.device = torch.device(device)
        self.interpreter = interpreter.to(self.device).eval()
        self.df = pd.read_csv(data) if isinstance(data, str) else data
        self._images = list(images) if images is not None else None
        self.input_col = input_col
        self.target_col = target_col
        self.patch_size = tuple(int(p) for p in patch_size)
        self.xy_step = xy_step
        self.z_step = z_step
        self.batch_size = batch_size
        self.path_col = path_col
        self.channel_axis = channel_axis
        self.columns = [input_col, target_col, seg_col, nuc_col, mem_col, mem_seg_col]

    # --- shared helpers -------------------------------------------------

    def _resolve_images(self, images):
        """Drop out-of-range FOV indices (the default ``range(10)`` overshoots small sets).

        The TF original clamped ``analyze_th`` to ``range(len(df))``; here we filter to
        in-bounds indices for all analyses so the default never runs ``df.iloc`` past the
        end (which raised ``IndexError``).
        """
        n = len(self._images) if self._images is not None else len(self.df)
        return [int(i) for i in images if 0 <= int(i) < n]

    def _preprocess(self, image_index: int, slice_by):
        if self._images is not None:
            return preprocess_fov(
                self._images[image_index], self.columns, list(FOV_NORMALIZE),
                min_z=self.patch_size[0], slice_by=slice_by,
            )
        return preprocess_image(
            self.df, image_index, self.columns, list(FOV_NORMALIZE),
            path_col=self.path_col, min_z=self.patch_size[0], slice_by=slice_by,
            channel_axis=self.channel_axis,
        )

    def predict(self, model: torch.nn.Module, data: np.ndarray) -> np.ndarray:
        """Batched inference on ``(N, *spatial, C)`` numpy -> ``(N, *spatial, Cout)`` numpy."""
        n = data.shape[0]
        t = torch.from_numpy(np.ascontiguousarray(data)).float()
        perm = [0, t.ndim - 1] + list(range(1, t.ndim - 1))
        t = t.permute(*perm).contiguous()  # (N, C, *spatial)
        bs = self.batch_size
        while True:
            try:
                outs = []
                for i in range(0, n, bs):
                    batch = t[i : i + bs].to(self.device)
                    with torch.no_grad():
                        outs.append(model(batch).detach().cpu())
                out = torch.cat(outs, dim=0)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs <= 1:
                    raise RuntimeError("predict failed: CUDA OOM even at batch_size=1")
                bs = max(1, bs // 2)
        inv = [0] + list(range(2, out.ndim)) + [1]
        return out.permute(*inv).contiguous().numpy()

    def _seg_dilated(self, target_seg, weighted_pcc):
        if not weighted_pcc or target_seg is None:
            return None
        seg = target_seg[..., 0]  # (Z, Y, X)
        return dilate_zyx(seg)[..., None]

    def _pcc(self, a, b, weights):
        return pearson_corr_np(a, b, weights)

    # --- analyses -------------------------------------------------------

    def calc_unet_pcc(self, out_dir, images=range(10), weighted_pcc=False):
        """PCC of the frozen predictor's assembled prediction against the target."""
        _create_dir(out_dir)
        images = self._resolve_images(images)
        rows = []
        for image_index in images:
            imgs = self._preprocess(int(image_index), slice_by=None)
            input_image, target_image, target_seg = imgs[0], imgs[1], imgs[2]
            target_seg = np.zeros_like(target_image) if target_seg is None else target_seg / 255.0
            pz_end = input_image.shape[0]
            px_end, py_end = input_image.shape[1], input_image.shape[2]
            patchs = collect_patchs(0, 0, 0, px_end, py_end, pz_end, input_image,
                                    self.patch_size, self.xy_step, self.z_step)
            seg_dil = self._seg_dilated(target_seg, weighted_pcc)
            weights = get_weights(patchs[0].shape)
            pred = self.predict(self.interpreter.predictor, patchs)
            assembled = assemble_image(0, 0, 0, px_end, py_end, pz_end,
                                       [pred, np.ones_like(patchs)], weights,
                                       input_image.shape, self.patch_size, self.xy_step, self.z_step)
            unet_p, d = assembled[0], assembled[1]
            rows.append({"PCC": self._pcc(target_image, unet_p / d, seg_dil)})
        result = pd.DataFrame(rows)
        result.to_csv(os.path.join(out_dir, "pcc_results.csv"), index=False)
        return result

    def analyze_th(
        self,
        out_dir,
        mode="regular",
        manual_th="full",
        images=range(10),
        noise_scale=1.5,
        weighted_pcc=False,
        save_image=False,
        mask_image=None,
        center_xy=None,
        margin=None,
        seed=None,
    ):
        ths_step = 0.1
        if mode == "agg":
            ths = np.arange(0.0, 1.0, ths_step)
        elif mode == "loo":
            ths = np.arange(0.0, 1.0, ths_step)
        elif mode == "mask":
            ths = [0.5]
        elif mode == "regular":
            ths = [manual_th]
        else:
            raise ValueError(f"unknown mode {mode!r}")

        _create_dir(out_dir)
        images = self._resolve_images(images)
        rng = np.random.default_rng(seed)
        pcc_rows, mask_rows, context_rows = [], [], []

        for image_index in images:
            image_index = int(image_index)
            if center_xy is not None and margin is not None:
                px_start = center_xy[0] - margin[0] - self.xy_step
                py_start = center_xy[1] - margin[1] - self.xy_step
                px_end = center_xy[0] + margin[0] + self.xy_step
                py_end = center_xy[1] + margin[1] + self.xy_step
                slice_by = [None, (px_start, px_end), (py_start, py_end)]
            else:
                slice_by = None

            imgs = self._preprocess(image_index, slice_by)
            input_image, target_image, target_seg = imgs[0], imgs[1], imgs[2]
            nuc_image, mem_image, mem_seg = imgs[3], imgs[4], imgs[5]
            target_seg = np.zeros_like(target_image) if target_seg is None else target_seg / 255.0

            pz_end = input_image.shape[0]
            px_end, py_end = input_image.shape[1], input_image.shape[2]

            patchs = collect_patchs(0, 0, 0, px_end, py_end, pz_end, input_image,
                                    self.patch_size, self.xy_step, self.z_step)
            seg_dil = self._seg_dilated(target_seg, weighted_pcc)
            weights = get_weights(patchs[0].shape)

            unet_patchs = self.predict(self.interpreter.predictor, patchs)
            mask_patchs = self.predict(self.interpreter, patchs)

            # NOISE DRAWN ONCE PER IMAGE, reused across thresholds (blocking fidelity point).
            noise = (rng.standard_normal(mask_patchs.shape) * noise_scale).astype(np.float32)

            assembled = assemble_image(0, 0, 0, px_end, py_end, pz_end,
                                       [unet_patchs, mask_patchs, np.ones_like(mask_patchs)],
                                       weights, input_image.shape, self.patch_size,
                                       self.xy_step, self.z_step)
            unet_p, mask_p_full, d = assembled[0], assembled[1], assembled[2]

            pccs, mask_sizes, contexts = [], [], []
            for th in ths:
                mask_p, mask_p_binary = self._threshold(mode, th, ths_step, mask_p_full, d, mask_image)
                mask_size, context = _context_ratio(mask_p_binary, target_seg)

                mask_patchs_term = collect_patchs(0, 0, 0, px_end, py_end, pz_end, mask_p,
                                                  self.patch_size, self.xy_step, self.z_step)
                input_patchs_p = mask_patchs_term * patchs + noise * (1 - mask_patchs_term)
                unet_noise_patchs = self.predict(self.interpreter.predictor, input_patchs_p)

                noise_assembled = assemble_image(0, 0, 0, px_end, py_end, pz_end,
                                                 [input_patchs_p, unet_noise_patchs], weights,
                                                 input_image.shape, self.patch_size,
                                                 self.xy_step, self.z_step)
                input_p, unet_noise_p = noise_assembled[0], noise_assembled[1]

                pccs.append(self._pcc(unet_p / d, unet_noise_p / d, seg_dil))
                mask_sizes.append(mask_size)
                contexts.append(context)

                if self._should_save(save_image, image_index):
                    base = os.path.join(out_dir, str(image_index), f"{th:.2f}" if not isinstance(th, str) else th)
                    _create_dir(base)
                    tifffile.imwrite(os.path.join(base, f"mask_{image_index}.tiff"), mask_p.astype(np.float16))
                    tifffile.imwrite(os.path.join(base, f"noisy_input_{image_index}.tiff"), (input_p / d).astype(np.float16))
                    tifffile.imwrite(os.path.join(base, f"noisy_unet_prediction_{image_index}.tiff"), (unet_noise_p / d).astype(np.float16))

            if self._should_save(save_image, image_index):
                self._save_globals(out_dir, image_index, input_image, target_image,
                                   nuc_image, mem_image, unet_p / d, target_seg)

            pcc_rows.append(pccs)
            mask_rows.append(mask_sizes)
            context_rows.append(contexts)

        cols = [f"{t:.2f}" if not isinstance(t, str) else t for t in ths]
        pcc_df = pd.DataFrame(pcc_rows, columns=cols)
        mask_df = pd.DataFrame(mask_rows, columns=cols)
        context_df = pd.DataFrame(context_rows, columns=cols)
        pcc_df.to_csv(os.path.join(out_dir, "pcc_results.csv"), index=False)
        mask_df.to_csv(os.path.join(out_dir, "mask_size_results.csv"), index=False)
        context_df.to_csv(os.path.join(out_dir, "context_results.csv"), index=False)
        return pcc_df, mask_df, context_df

    def find_noise_scale(self, out_dir, images=range(10), weighted_pcc=False,
                         noise_start=0.0, noise_stop=4.5, noise_step=0.5):
        _create_dir(out_dir)
        images = self._resolve_images(images)
        noises = np.arange(noise_start, noise_stop, noise_step)
        rows = []
        for image_index in images:
            imgs = self._preprocess(int(image_index), slice_by=None)
            input_image, target_image, target_seg = imgs[0], imgs[1], imgs[2]
            target_seg = np.zeros_like(target_image) if target_seg is None else target_seg / 255.0
            pz_end = input_image.shape[0]
            px_end, py_end = input_image.shape[1], input_image.shape[2]
            patchs = collect_patchs(0, 0, 0, px_end, py_end, pz_end, input_image,
                                    self.patch_size, self.xy_step, self.z_step)
            seg_dil = self._seg_dilated(target_seg, weighted_pcc)
            weights = get_weights(patchs[0].shape)
            unet_patchs = self.predict(self.interpreter.predictor, patchs)
            assembled = assemble_image(0, 0, 0, px_end, py_end, pz_end,
                                       [unet_patchs, np.ones_like(patchs)], weights,
                                       input_image.shape, self.patch_size, self.xy_step, self.z_step)
            unet_p, d = assembled[0], assembled[1]

            pccs = []
            for noise_scale in noises:
                noise = (np.random.default_rng(int(noise_scale * 1000)).standard_normal(patchs.shape) * noise_scale).astype(np.float32)
                noisy_patchs = patchs + noise
                unet_noise_patchs = self.predict(self.interpreter.predictor, noisy_patchs)
                noise_assembled = assemble_image(0, 0, 0, px_end, py_end, pz_end,
                                                 [noisy_patchs, unet_noise_patchs], weights,
                                                 input_image.shape, self.patch_size, self.xy_step, self.z_step)
                _, unet_noise_p = noise_assembled[0], noise_assembled[1]
                pccs.append(self._pcc(unet_p / d, unet_noise_p / d, seg_dil))
            rows.append(pccs)
        result = pd.DataFrame(rows, columns=[f"{n:.2f}" for n in noises])
        result.to_csv(os.path.join(out_dir, "noise_pcc_results.csv"), index=False)
        return result

    # --- threshold + saving --------------------------------------------

    @staticmethod
    def _threshold(mode, th, ths_step, mask_p_full, d, mask_image):
        if mode == "mask" and mask_image is not None:
            m = (mask_image / 255.0 if not isinstance(mask_image, str)
                 else read_tiff(mask_image) / 255.0)
            binary = np.where(m >= th, 1.0, 0.0)
            return binary, binary
        ratio = mask_p_full / d
        if mode == "agg":
            binary = np.where(ratio > th, 1.0, 0.0)
            return binary, binary
        if mode == "loo":
            binary = np.where((ratio > (th - ths_step)) & (ratio <= th), 1.0, 0.0)
            return binary, binary
        # regular
        if th != "full":
            binary = np.where(ratio > th, 1.0, 0.0)
            return binary, binary
        return ratio, np.ones_like(mask_p_full)

    @staticmethod
    def _should_save(save_image, image_index) -> bool:
        return save_image is True or (not isinstance(save_image, bool) and image_index < save_image)

    @staticmethod
    def _save_globals(out_dir, image_index, input_image, target_image, nuc, mem, unet_pred, seg):
        base = os.path.join(out_dir, str(image_index))
        _create_dir(base)
        tifffile.imwrite(os.path.join(base, f"input_{image_index}.tiff"), input_image.astype(np.float16))
        tifffile.imwrite(os.path.join(base, f"target_{image_index}.tiff"), target_image.astype(np.float16))
        if nuc is not None:
            tifffile.imwrite(os.path.join(base, f"nuc_{image_index}.tiff"), nuc.astype(np.float16))
        if mem is not None:
            tifffile.imwrite(os.path.join(base, f"mem_{image_index}.tiff"), mem.astype(np.float16))
        tifffile.imwrite(os.path.join(base, f"unet_prediction_{image_index}.tiff"), unet_pred.astype(np.float16))
        tifffile.imwrite(os.path.join(base, f"seg_target_{image_index}.tiff"), seg.astype(np.float16))
