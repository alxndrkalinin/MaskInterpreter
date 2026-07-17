import pytest
import torch

from mask_interpreter.unet import UNet


def _out_spatial(unet, batch_shape):
    x = torch.randn(*batch_shape)
    with torch.no_grad():
        y = unet(x)
    return tuple(y.shape)


def test_unet_2d_shape():
    unet = UNet((128, 128, 4), batch_norm=True, final_activation="sigmoid")
    assert _out_spatial(unet, (2, 4, 128, 128)) == (2, 1, 128, 128)
    # depth driven by first spatial axis: 128 -> 64 -> 32 -> 16 -> 8 -> 4 = 5 levels
    assert len(unet.down_pre) == 5


def test_unet_3d_anisotropic_shape():
    # (32,128,128): first-axis-driven depth => 3 levels (32->16->8->4), NOT 5.
    unet = UNet((32, 128, 128, 1), batch_norm=True, final_activation="linear")
    assert len(unet.down_pre) == 3
    assert _out_spatial(unet, (1, 1, 32, 128, 128)) == (1, 1, 32, 128, 128)


def test_unet_zero_depth_when_first_axis_small():
    # first axis == 4 is not > 4, so no down/up sampling at all.
    unet = UNet((4, 16, 16, 2), batch_norm=False)
    assert len(unet.down_pre) == 0
    assert len(unet.up_sample) == 0
    assert _out_spatial(unet, (1, 2, 4, 16, 16)) == (1, 1, 4, 16, 16)


def test_unet_2d_leading_one_marks_2d():
    # (1,H,W,C) convention marks a 2D image.
    unet = UNet((1, 32, 32, 3))
    assert unet.ndim == 2
    assert _out_spatial(unet, (2, 3, 32, 32)) == (2, 1, 32, 32)


def test_unet_sigmoid_range():
    unet = UNet((32, 32, 3), final_activation="sigmoid")
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y = unet(x)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_unet_out_channels():
    unet = UNet((32, 32, 3), out_channels=5, final_activation="linear")
    assert _out_spatial(unet, (2, 3, 32, 32)) == (2, 5, 32, 32)


@pytest.mark.parametrize("s", [(20, 20), (40, 40), (7, 7), (31, 33), (30, 120, 120)])
def test_unet_non_power_of_two_shapes(s):
    # Floor-halving on the down path is not exactly reversed by x2 upsampling for these
    # sizes; the up path must still emit one stage per skip and reconstruct input size.
    unet = UNet((*s, 3), batch_norm=False, final_activation="sigmoid")
    assert len(unet.up_sample) == len(unet.down_pre)
    assert _out_spatial(unet, (1, 3, *s)) == (1, 1, *s)
