import numpy as np
import pandas as pd
import pytest
import tifffile
import torch

from mask_interpreter.data import transforms as T
from mask_interpreter.data.fov import FOVPatchDataset
from mask_interpreter.data.single_cell import SingleCellDataset


def test_normalize_std():
    x = np.arange(100, dtype=np.float32).reshape(10, 10)
    y = T.normalize_std(x)
    assert abs(float(y.mean())) < 1e-5
    assert abs(float(y.std()) - 1.0) < 1e-5


def test_normalize_minmax():
    x = np.array([[1.0, 3.0], [5.0, 9.0]], dtype=np.float32)
    y = T.normalize_minmax(x, max_value=1.0)
    assert float(y.min()) == 0.0 and float(y.max()) == 1.0


def test_rot90_last_two_axes():
    x = np.random.default_rng(0).standard_normal((2, 3, 8, 8))  # (C, Z, Y, X)
    r = T.rot90(x, k=1)
    assert r.shape == x.shape
    # k=4 is identity
    assert np.allclose(T.rot90(x, k=4), x)
    # rotates Y,X (axes 2,3), leaves C,Z alone
    assert np.allclose(T.rot90(x, k=1), np.rot90(x, k=1, axes=(2, 3)))


def test_pad_z_to_overshoots_odd_deficit():
    assert T.pad_z_to(np.zeros((30, 4, 4)), 32).shape[0] == 32   # ceil(2/2)=1 each side
    assert T.pad_z_to(np.zeros((31, 4, 4)), 32).shape[0] == 33   # ceil(1/2)=1 each side -> overshoot
    assert T.pad_z_to(np.zeros((40, 4, 4)), 32).shape[0] == 40   # no pad when >= target


def test_sample_patch_indices_bounds():
    rng = np.random.default_rng(0)
    for _ in range(50):
        idx = T.sample_patch_indices((8, 16, 16), (4, 8, 8), rng)
        for (lo, hi), dim, p in zip(idx, (8, 16, 16), (4, 8, 8)):
            assert 0 <= lo and hi <= dim and hi - lo == p
    with pytest.raises(ValueError):
        T.sample_patch_indices((4, 4, 4), (8, 8, 8), rng)


def _write_fov(tmp_path, C=4, Z=8, Y=16, X=16):
    arr = np.random.default_rng(1).standard_normal((C, Z, Y, X)).astype(np.float32)
    path = str(tmp_path / "fov.tif")
    tifffile.imwrite(path, arr)
    csv = str(tmp_path / "list.csv")
    pd.DataFrame([{"path_tiff": path, "ch_in": 0, "ch_tgt": 1}]).to_csv(csv, index=False)
    return csv


def test_fov_patch_shapes_3d(tmp_path):
    csv = _write_fov(tmp_path)
    ds = FOVPatchDataset(csv, "ch_in", "ch_tgt", patch_size=(4, 8, 8),
                         patches_per_image=3, norm=True, seed=0)
    x, y = ds[0]
    assert x.shape == (1, 4, 8, 8) and y.shape == (1, 4, 8, 8)
    assert x.dtype == torch.float32


def test_fov_patch_2d(tmp_path):
    csv = _write_fov(tmp_path)
    ds = FOVPatchDataset(csv, "ch_in", "ch_tgt", patch_size=(1, 8, 8), seed=0)
    x, y = ds[0]
    assert x.shape == (1, 8, 8) and y.shape == (1, 8, 8)


def test_fov_predictors_concat(tmp_path):
    csv = _write_fov(tmp_path)
    predictor = torch.nn.Identity()
    ds = FOVPatchDataset(csv, "ch_in", "ch_tgt", patch_size=(4, 8, 8),
                         predictors=predictor, seed=0)
    x, _ = ds[0]
    assert x.shape == (2, 4, 8, 8)  # input + in-silico channel


def test_fov_predictors_concat_z1(tmp_path):
    # Z==1 FOV + predictor: reshape (not squeeze) must keep the singleton Z so the concat
    # stays rank-consistent.
    csv = _write_fov(tmp_path, Z=1)
    ds = FOVPatchDataset(csv, "ch_in", "ch_tgt", patch_size=(1, 8, 8),
                         predictors=torch.nn.Identity(), seed=0)
    x, _ = ds[0]
    assert x.shape == (2, 8, 8)  # (input + in-silico), Z dropped for 2D patch


def test_fov_dilate_runs(tmp_path):
    csv = _write_fov(tmp_path)
    ds = FOVPatchDataset(csv, "ch_in", "ch_tgt", patch_size=(4, 8, 8), dilate=True, seed=0)
    x, y = ds[0]
    assert y.shape == (1, 4, 8, 8)


def test_fov_unsupported_kwarg_raises(tmp_path):
    csv = _write_fov(tmp_path)
    with pytest.raises(NotImplementedError):
        FOVPatchDataset(csv, "ch_in", "ch_tgt", pairs=True)


def test_fov_seed_determinism(tmp_path):
    csv = _write_fov(tmp_path)
    ds = FOVPatchDataset(csv, "ch_in", "ch_tgt", patch_size=(4, 8, 8), seed=7)
    x1, _ = ds[0]
    x2, _ = ds[0]
    assert torch.equal(x1, x2)


def _write_single_cell(tmp_path, Z=20, Y=16, X=16):
    sig = np.random.default_rng(2).standard_normal((Z, Y, X)).astype(np.float32)
    tgt = np.random.default_rng(3).standard_normal((Z, Y, X)).astype(np.float32)
    sp, tp = str(tmp_path / "sig.tif"), str(tmp_path / "tgt.tif")
    tifffile.imwrite(sp, sig)
    tifffile.imwrite(tp, tgt)
    csv = str(tmp_path / "sc.csv")
    pd.DataFrame([{"signal_path": sp, "target_path": tp}]).to_csv(csv, index=False)
    return csv


def test_single_cell_shapes_and_padz(tmp_path):
    csv = _write_single_cell(tmp_path, Z=20)
    ds = SingleCellDataset(csv, pad_z=32, norm=True, seed=0)
    x, y = ds[0]
    assert x.shape == (1, 32, 16, 16)  # padded from 20 -> 32
    assert y.shape == (1, 32, 16, 16)
    assert abs(float(x.mean())) < 0.5  # normalized-ish (edge pad shifts slightly)
