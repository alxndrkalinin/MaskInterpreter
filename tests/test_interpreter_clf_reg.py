import torch

from mask_interpreter.config import LossConfig
from mask_interpreter.interpreter.classification import ClassificationInterpreter
from mask_interpreter.interpreter.regression import RegressionInterpreter
from tests.toy import RegionScorePredictor, region_masks

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPATIAL = (8, 8)


def _clf():
    torch.manual_seed(0)
    predictor = RegionScorePredictor(SPATIAL, in_channels=3, out_dim=2)
    loss = LossConfig(target_loss_weight=1.75, noise_scale=0.5, pcc_target=0.95)
    return ClassificationInterpreter(predictor, SPATIAL, in_channels=3, loss=loss).to(DEVICE)


def _reg():
    torch.manual_seed(0)
    predictor = RegionScorePredictor(SPATIAL, in_channels=1, out_dim=1)
    loss = LossConfig(target_loss_weight=2.5, noise_scale=0.5, pcc_target=0.95)
    return RegressionInterpreter(predictor, SPATIAL, in_channels=1, loss=loss).to(DEVICE)


def test_clf_has_conv_preproc_regression_does_not():
    assert hasattr(_clf(), "preproc")
    assert not hasattr(_reg(), "preproc")


def test_grad_reduce_differs():
    assert _clf().grad_reduce == "max"
    assert _reg().grad_reduce == "sum"


def test_grad_aug_channel_in_range_and_detached():
    model = _clf()
    x = torch.randn(4, 3, *SPATIAL, device=DEVICE)
    aug = model.grad_augment(x, model.grad_reduce)
    assert aug.shape == (4, 4, *SPATIAL)  # C+1 channels
    grad_ch = aug[:, -1]
    assert float(grad_ch.min()) >= 0.0 and float(grad_ch.max()) <= 1.0
    assert not aug.requires_grad  # augmented input is detached from predictor graph


def test_clf_overfit_localizes():
    model = _clf()
    x = torch.randn(8, 3, *SPATIAL, device=DEVICE)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=2e-3)
    torch.manual_seed(1)
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        total, _ = model.training_step(x)
        total.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        mask = model(x)
    assert float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0
    inside, outside = region_masks(SPATIAL)
    inside, outside = inside.to(DEVICE), outside.to(DEVICE)
    assert float(mask[:, 0][:, inside].mean()) > float(mask[:, 0][:, outside].mean()) + 0.05


def test_reg_overfit_localizes():
    model = _reg()
    x = torch.randn(8, 1, *SPATIAL, device=DEVICE)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=2e-3)
    torch.manual_seed(1)
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        total, _ = model.training_step(x)
        total.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        mask = model(x)
    assert float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0
    inside, outside = region_masks(SPATIAL)
    inside, outside = inside.to(DEVICE), outside.to(DEVICE)
    assert float(mask[:, 0][:, inside].mean()) > float(mask[:, 0][:, outside].mean()) + 0.05
