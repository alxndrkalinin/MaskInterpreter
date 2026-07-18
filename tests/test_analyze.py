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
    csv = _write_fov(tmp_path, seg_full=False)  # empty structure_seg (present but zero)
    az = _analyzer(csv)
    with pytest.raises(ValueError, match="intersection"):
        az.analyze_th(str(tmp_path / "out"), mode="regular", images=[0])


def test_analyze_th_without_seg_gives_nan_context(tmp_path):
    # No structure_seg channel at all: context is undefined (NaN), and analyze_th must NOT
    # raise (unlike a provided-but-empty seg, which still raises above).
    rng = np.random.default_rng(0)
    fov = {"input": rng.standard_normal((Z, Y, X)).astype(np.float32),
           "target": rng.standard_normal((Z, Y, X)).astype(np.float32)}
    torch.manual_seed(0)
    interp = Image2ImageInterpreter(RegionImagePredictor((4, 8, 8)), spatial_size=(4, 8, 8),
                                    ndim=3, in_channels=1, pred_channels=1,
                                    loss=LossConfig(pcc_target=0.9))
    az = _in_memory_analyzer(interp, [fov])
    pcc_df, mask_df, ctx_df = az.analyze_th(str(tmp_path / "out"), mode="agg", images=[0], seed=0)
    assert np.isnan(ctx_df.to_numpy()).all()          # context undefined without seg
    assert not np.isnan(mask_df.to_numpy()).all()     # mask size still computed
    pv = pcc_df.to_numpy().ravel()
    pv = pv[~np.isnan(pv)]
    assert np.all(pv >= -1.0001) and np.all(pv <= 1.0001)


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


def _in_memory_analyzer(interp, fovs):
    return Analyzer(interp, images=fovs, input_col="input", target_col="target",
                    patch_size=PATCH, xy_step=8, z_step=4, batch_size=2, device=DEVICE)


def test_analyze_in_memory_matches_csv(tmp_path):
    # In-memory FOVs (array and dict form) must reproduce the CSV/tiff path exactly.
    csv = _write_fov(tmp_path)
    arr = np.asarray(tifffile.imread(pd.read_csv(csv)["path_tiff"].iloc[0])).astype(np.float32)
    fov_array = arr                                   # (C, Z, Y, X), canonical channel order
    fov_dict = {
        "input": arr[0], "target": arr[1], "structure_seg": arr[2],
        "channel_dna": arr[3], "channel_membrane": arr[4], "membrane_seg": arr[5],
    }

    ref = _analyzer(csv).analyze_th(str(tmp_path / "csv"), mode="agg", images=[0], seed=0)[0]

    torch.manual_seed(0)
    interp = Image2ImageInterpreter(RegionImagePredictor((4, 8, 8)), spatial_size=(4, 8, 8),
                                    ndim=3, in_channels=1, pred_channels=1,
                                    loss=LossConfig(pcc_target=0.9))
    got_arr = _in_memory_analyzer(interp, [fov_array]).analyze_th(
        str(tmp_path / "arr"), mode="agg", images=[0], seed=0)[0]
    got_dict = _in_memory_analyzer(interp, [fov_dict]).analyze_th(
        str(tmp_path / "dct"), mode="agg", images=[0], seed=0)[0]

    pd.testing.assert_frame_equal(ref, got_arr)
    pd.testing.assert_frame_equal(ref, got_dict)


def test_in_memory_missing_optional_channel(tmp_path):
    # Only input/target present; seg absent -> structure_seg None -> zeros (like the tiff path).
    rng = np.random.default_rng(0)
    fov = {"input": rng.standard_normal((Z, Y, X)).astype(np.float32),
           "target": rng.standard_normal((Z, Y, X)).astype(np.float32)}
    torch.manual_seed(0)
    interp = Image2ImageInterpreter(RegionImagePredictor((4, 8, 8)), spatial_size=(4, 8, 8),
                                    ndim=3, in_channels=1, pred_channels=1,
                                    loss=LossConfig(pcc_target=0.9))
    az = _in_memory_analyzer(interp, [fov])
    r = az.calc_unet_pcc(str(tmp_path / "o"), images=[0])
    assert math.isfinite(float(r["PCC"].iloc[0]))


def test_analyze_2d_fov(tmp_path):
    # 2D FOV (patch_size[0]==1) with a 2D interpreter: dict/array/CSV paths agree, tiffs are
    # saved as degenerate (1, Y, X, 1) volumes, and results are finite.
    torch.manual_seed(0)
    interp = Image2ImageInterpreter(RegionImagePredictor((Y, X)), spatial_size=(Y, X),
                                    ndim=2, in_channels=1, pred_channels=1,
                                    loss=LossConfig(pcc_target=0.9))
    rng = np.random.default_rng(0)
    sig = rng.standard_normal((Y, X)).astype(np.float32)
    reg = np.zeros((Y, X), np.float32)
    reg[Y // 4:Y - Y // 4, X // 4:X - X // 4] = 1.0
    chans = [sig, sig * reg, reg * 255.0, rng.standard_normal((Y, X)).astype(np.float32),
             rng.standard_normal((Y, X)).astype(np.float32),
             255.0 * (rng.standard_normal((Y, X)) > 0)]
    fov_dict = dict(zip(["input", "target", "structure_seg", "channel_dna",
                         "channel_membrane", "membrane_seg"], chans))
    fov_arr = np.stack(chans).astype(np.float32)   # (6, Y, X)

    def az(source):
        return Analyzer(interp, patch_size=(1, Y, X, 1), xy_step=8, z_step=1,
                        batch_size=2, device=DEVICE, **source)

    p_dict = az({"images": [fov_dict]}).analyze_th(str(tmp_path / "d"), mode="agg",
                                                   images=[0], seed=0)[0]
    p_arr = az({"images": [fov_arr]}).analyze_th(str(tmp_path / "a"), mode="agg",
                                                 images=[0], seed=0)[0]
    np.testing.assert_allclose(p_dict.to_numpy(), p_arr.to_numpy(), equal_nan=True)

    vals = p_dict.to_numpy().ravel()
    vals = vals[~np.isnan(vals)]
    assert np.all(vals >= -1.0001) and np.all(vals <= 1.0001)
    assert math.isfinite(float(az({"images": [fov_dict]}).calc_unet_pcc(
        str(tmp_path / "u"), images=[0])["PCC"].iloc[0]))

    az({"images": [fov_dict]}).analyze_th(str(tmp_path / "s"), mode="regular",
                                          images=[0], save_image=True)
    assert np.asarray(tifffile.imread(tmp_path / "s" / "0" / "full" / "mask_0.tiff")).shape == (1, Y, X, 1)


def test_analyzer_requires_exactly_one_source():
    with pytest.raises(ValueError, match="exactly one"):
        Analyzer(None)  # neither data nor images
    with pytest.raises(ValueError, match="exactly one"):
        Analyzer(None, data="x.csv", images=[{}])  # both


def test_default_images_clamped_to_dataset_size(tmp_path):
    # Default images=range(10) must not run df.iloc past the end of a small (1-FOV) set.
    csv = _write_fov(tmp_path)
    az = _analyzer(csv)
    assert len(az.calc_unet_pcc(str(tmp_path / "u"))) == 1                 # default range(10)
    assert az.find_noise_scale(str(tmp_path / "n")).shape[0] == 1
    assert az.analyze_th(str(tmp_path / "a"), mode="regular")[0].shape[0] == 1
