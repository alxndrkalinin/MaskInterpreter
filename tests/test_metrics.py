import numpy as np
import scipy.stats
import torch

from mask_interpreter.metrics import pearson_corr, pearson_corr_np


def test_pearson_matches_scipy_population():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(500).astype(np.float64)
    b = 0.7 * a + 0.3 * rng.standard_normal(500)
    expected = scipy.stats.pearsonr(a, b)[0]
    got = float(pearson_corr(torch.tensor(a), torch.tensor(b)))
    assert abs(got - expected) < 1e-6
    assert abs(pearson_corr_np(a, b) - expected) < 1e-6


def test_pearson_scale_invariant_to_bessel():
    # A naive torch.std (unbiased=True) would give (N-1)/N * r; population must match scipy.
    rng = np.random.default_rng(1)
    a = torch.tensor(rng.standard_normal(37))  # small N exaggerates Bessel error
    b = torch.tensor(rng.standard_normal(37))
    expected = scipy.stats.pearsonr(a.numpy(), b.numpy())[0]
    got = float(pearson_corr(a, b))
    assert abs(got - expected) < 1e-6


def test_pearson_multidim_flattened():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((4, 1, 8, 8))
    b = 0.5 * a + rng.standard_normal((4, 1, 8, 8))
    expected = scipy.stats.pearsonr(a.ravel(), b.ravel())[0]
    got = float(pearson_corr(torch.tensor(a), torch.tensor(b)))
    assert abs(got - expected) < 1e-6


def test_weighted_selects_seg_pixels():
    rng = np.random.default_rng(3)
    a = rng.standard_normal(200)
    b = 0.6 * a + 0.4 * rng.standard_normal(200)
    weights = np.zeros(200)
    weights[:50] = 1.0
    weights[50:80] = 255.0  # both encodings count as "in"
    sel = weights > 0
    expected = scipy.stats.pearsonr(a[sel], b[sel])[0]
    got = float(pearson_corr(torch.tensor(a), torch.tensor(b), torch.tensor(weights)))
    assert abs(got - expected) < 1e-6


def test_weighted_empty_falls_back_to_full():
    rng = np.random.default_rng(4)
    a = rng.standard_normal(100)
    b = rng.standard_normal(100)
    weights = np.zeros(100)  # no in-pixels -> fall back to full arrays
    expected = scipy.stats.pearsonr(a, b)[0]
    got = float(pearson_corr(torch.tensor(a), torch.tensor(b), torch.tensor(weights)))
    assert abs(got - expected) < 1e-6


def test_constant_input_is_nan_like_cubic():
    a = torch.ones(10)
    b = torch.arange(10, dtype=torch.float64)
    assert np.isnan(float(pearson_corr(a, b)))          # torch path
    assert np.isnan(pearson_corr_np(a.numpy(), b.numpy()))  # numpy/cubic path


def test_torch_matches_cubic_numpy():
    from cubic.metrics import pcc as cubic_pcc

    rng = np.random.default_rng(5)
    a = rng.standard_normal((3, 1, 8, 8))
    b = 0.4 * a + rng.standard_normal((3, 1, 8, 8))
    torch_val = float(pearson_corr(torch.tensor(a), torch.tensor(b)))
    assert abs(torch_val - float(cubic_pcc(a, b))) < 1e-6


def test_torch_pcc_is_differentiable():
    x = torch.randn(64, requires_grad=True)
    y = torch.randn(64)
    r = pearson_corr(x, y)
    r.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_result_clipped_to_unit_range():
    a = torch.randn(50)
    r = float(pearson_corr(a, a))  # perfectly correlated
    assert -1.0 <= r <= 1.0 and abs(r - 1.0) < 1e-5
