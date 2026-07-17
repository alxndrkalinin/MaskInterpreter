import numpy as np

from mask_interpreter.data.patches import assemble_image, collect_patchs, get_weights


def test_get_weights_feathers_every_spatial_axis():
    # Triangular overlap weighting must vary along ALL spatial axes (Z, Y, X) and be
    # uniform across the trailing channel axis. (The old shape[1:] dropped the Z axis.)
    w = get_weights((4, 8, 8, 1))
    assert w.shape == (4, 8, 8, 1)
    assert np.ptp(w, axis=0).max() > 0  # Z feathered
    assert np.ptp(w, axis=1).max() > 0  # Y feathered
    assert np.ptp(w, axis=2).max() > 0  # X feathered
    assert np.ptp(w, axis=3).max() == 0  # channels uniform
    assert (w > 0).all()  # strictly positive -> no zero-denominator in assembly


def test_get_weights_multichannel_uniform_over_channels():
    w = get_weights((4, 8, 8, 3))
    assert w.shape == (4, 8, 8, 3)
    assert np.array_equal(w[..., 0], w[..., 1]) and np.array_equal(w[..., 1], w[..., 2])


def test_assembled_denominator_covers_full_volume():
    # With feathered weights the overlap-blend denominator (sum of weights) must stay
    # strictly positive everywhere the tiling covers, so unet_p / d never divides by zero.
    shape = (8, 16, 16, 1)
    image = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    patchs = collect_patchs(0, 0, 0, 16, 16, 8, image, (4, 8, 8, 1), xy_step=8, z_step=4)
    w = get_weights(patchs[0].shape)
    d = assemble_image(0, 0, 0, 16, 16, 8, [np.ones_like(patchs)], w, shape,
                       (4, 8, 8, 1), xy_step=8, z_step=4)[0]
    assert (d > 0).all()
