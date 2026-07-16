import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from mask_interpreter.config import LossConfig
from mask_interpreter.interpreter.image2image import Image2ImageInterpreter
from mask_interpreter.train import Trainer
from tests.toy import RegionImagePredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_trainer_runs_and_checkpoints(tmp_path):
    spatial = (16, 16)
    torch.manual_seed(0)
    predictor = RegionImagePredictor(spatial)
    model = Image2ImageInterpreter(
        predictor, spatial_size=spatial, in_channels=1, pred_channels=1, ndim=2,
        loss=LossConfig(pcc_target=0.9),
    )
    x = torch.randn(16, 1, *spatial)
    train_loader = DataLoader(TensorDataset(x), batch_size=4)
    val_loader = DataLoader(TensorDataset(x[:8]), batch_size=4)

    ckpt = os.path.join(tmp_path, "mg.pt")
    trainer = Trainer(
        model, lr=1e-3, device=DEVICE, checkpoint_path=ckpt,
        monitor="val_stop", term="val_pcc", term_value=-1.0,  # ensure term guard passes
        early_stop_monitor="val_stop", patience=10,
    )
    history = trainer.fit(train_loader, val_loader, epochs=3)
    assert len(history) >= 1
    assert "val_stop" in history[-1]
    assert os.path.exists(ckpt)

    # checkpoint reloads into a fresh model
    fresh = Image2ImageInterpreter(
        RegionImagePredictor(spatial), spatial_size=spatial, in_channels=1,
        pred_channels=1, ndim=2, loss=LossConfig(pcc_target=0.9),
    ).to(trainer.device)
    fresh.load_state_dict(torch.load(ckpt, map_location=trainer.device))


def test_trainer_keeps_predictor_frozen_after_train_mode():
    spatial = (16, 16)
    predictor = RegionImagePredictor(spatial)
    model = Image2ImageInterpreter(
        predictor, spatial_size=spatial, in_channels=1, pred_channels=1, ndim=2,
    )
    model.train()  # would normally put all submodules in train mode
    assert not model.predictor.training  # predictor stays in eval
