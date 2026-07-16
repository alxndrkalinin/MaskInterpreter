import math
import os

import numpy as np
import pandas as pd
import pytest
import tifffile
import torch

from mask_interpreter.analyze import Analyzer, preprocess_image
from mask_interpreter.config import LossConfig
from mask_interpreter.interpreter.image2image import Image2ImageInterpreter
from tests.toy import RegionImagePredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH = (4, 8, 8, 1)
Z, Y, X = 8, 16, 16


def _write_fov(tmp_path, seg_full=True):
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((6, Z, Y, X)).astype(np.float32)
    seg = np.zeros((Z, Y, X), np.float32)
    if seg_full:
        seg[:, 4:12, 4:12] = 255.0  # large central organelle block
    arr[2] = seg  # structure_seg channel
    arr[5] = 255.0 * (rng.standard_normal((Z, Y, X)) > 0)  # membrane_seg
    path = str(tmp_path / "fov.tif")
    tifffile.imwrite(path, arr)
    csv = str(tmp_path / "list.csv")
    pd.DataFrame([{
        "path_tiff": path, "input": 0, "target": 1, "structure_seg": 2,
        "channel_dna": 3, "channel_membrane": 4, "membrane_seg": 5,
    }]).to_csv(csv, index=False)
    return csv


def _analyzer(csv):
    torch.manual_seed(0)
    predictor = RegionImagePredictor((4, 8, 8))
    interp = Image2ImageInterpreter(
        predictor, spatial_size=(4, 8, 8), ndim=3, in_channels=1, pred_channels=1,
        loss=LossConfig(pcc_target=0.9),
    )
    return Analyzer(interp, csv, "input", "target", patch_size=PATCH,
                    xy_step=8, z_step=4, batch_size=2, device=DEVICE)


def test_preprocess_six_channel_contract(tmp_path):
    csv = _write_fov(tmp_path)
    df = pd.read_csv(csv)
    imgs = preprocess_image(df, 0, ["input", "target", "structure_seg", "channel_dna",
                                    "channel_membrane", "membrane_seg"],
                            [True, True, False, True, True, False], min_z=4)
    assert len(imgs) == 6
    for im in imgs:
        assert im.shape == (Z, Y, X, 1)
    # seg channel (index 2) NOT normalized -> retains 0/255 values
    assert imgs[2].max() == 255.0


def test_analyze_th_regular_outputs(tmp_path):
    csv = _write_fov(tmp_path)
    az = _analyzer(csv)
    pcc_df, mask_df, ctx_df = az.analyze_th(str(tmp_path / "out"), mode="regular",
                                            images=[0], save_image=True)
    assert os.path.exists(tmp_path / "out" / "pcc_results.csv")
    assert os.path.exists(tmp_path / "out" / "mask_size_results.csv")
    assert os.path.exists(tmp_path / "out" / "context_results.csv")
    # per-image tiffs written
    assert os.path.exists(tmp_path / "out" / "0" / "input_0.tiff")
    assert os.path.exists(tmp_path / "out" / "0" / "unet_prediction_0.tiff")
    vals = pcc_df.to_numpy().ravel()
    vals = vals[~np.isnan(vals)]
    assert np.all(vals >= -1.0001) and np.all(vals <= 1.0001)


def test_analyze_th_agg_and_pcc_range(tmp_path):
    csv = _write_fov(tmp_path)
    az = _analyzer(csv)
    pcc_df, mask_df, ctx_df = az.analyze_th(str(tmp_path / "out"), mode="agg",
                                            images=[0], seed=0)
    assert pcc_df.shape[1] == 10  # 10 thresholds
    vals = pcc_df.to_numpy().ravel()
    vals = vals[~np.isnan(vals)]
    assert np.all(vals >= -1.0001) and np.all(vals <= 1.0001)


def test_analyze_th_seed_determinism(tmp_path):
    csv = _write_fov(tmp_path)
    az = _analyzer(csv)
    a, _, _ = az.analyze_th(str(tmp_path / "o1"), mode="agg", images=[0], seed=42)
    b, _, _ = az.analyze_th(str(tmp_path / "o2"), mode="agg", images=[0], seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_zero_intersection_raises(tmp_path):
    csv = _write_fov(tmp_path, seg_full=False)  # empty structure_seg
    az = _analyzer(csv)
    with pytest.raises(ValueError, match="intersection"):
        az.analyze_th(str(tmp_path / "out"), mode="regular", images=[0])


def test_calc_unet_pcc(tmp_path):
    csv = _write_fov(tmp_path)
    az = _analyzer(csv)
    result = az.calc_unet_pcc(str(tmp_path / "out"), images=[0])
    assert os.path.exists(tmp_path / "out" / "pcc_results.csv")
    assert "PCC" in result.columns
    assert math.isfinite(float(result["PCC"].iloc[0]))


def test_find_noise_scale(tmp_path):
    csv = _write_fov(tmp_path)
    az = _analyzer(csv)
    result = az.find_noise_scale(str(tmp_path / "out"), images=[0])
    assert os.path.exists(tmp_path / "out" / "noise_pcc_results.csv")
    assert result.shape[1] == len(np.arange(0.0, 4.5, 0.5))
