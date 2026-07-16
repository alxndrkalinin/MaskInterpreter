import copy

import torch

from mask_interpreter.config import LossConfig
from mask_interpreter.interpreter.image2image import Image2ImageInterpreter
from tests.toy import RegionImagePredictor, region_masks

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make(spatial=(16, 16)):
    torch.manual_seed(0)
    predictor = RegionImagePredictor(spatial)
    loss = LossConfig(
        similarity_loss_weight=1.0, mask_loss_weight=1.0,
        target_loss_weight=10.0, noise_scale=1.5, pcc_target=0.9,
    )
    model = Image2ImageInterpreter(
        predictor, spatial_size=spatial, in_channels=1, pred_channels=1, ndim=2, loss=loss,
    ).to(DEVICE)
    return model


def test_predictor_is_frozen():
    model = _make()
    assert all(not p.requires_grad for p in model.predictor.parameters())
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) > 0  # generator/adaptor are trainable


def test_gradients_flow_to_generator_not_predictor():
    model = _make()
    x = torch.randn(2, 1, 16, 16, device=DEVICE)
    total, _ = model.training_step((x,))
    total.backward()
    assert all(p.grad is None for p in model.predictor.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.adaptor.parameters())


def test_mask_in_unit_range_and_localizes():
    model = _make()
    x = torch.randn(4, 1, 16, 16, device=DEVICE)
    predictor_before = copy.deepcopy(model.predictor.weight.detach().clone())
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=2e-3)

    torch.manual_seed(1)
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        total, _ = model.training_step((x,))
        total.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        mask = model(x)
    assert float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0

    inside, outside = region_masks((16, 16))
    inside, outside = inside.to(DEVICE), outside.to(DEVICE)
    mean_in = float(mask[:, 0][:, inside].mean())
    mean_out = float(mask[:, 0][:, outside].mean())
    assert mean_in > mean_out + 0.1, (mean_in, mean_out)

    # frozen predictor weights must be unchanged.
    assert torch.equal(model.predictor.weight.detach().cpu(), predictor_before.cpu())
