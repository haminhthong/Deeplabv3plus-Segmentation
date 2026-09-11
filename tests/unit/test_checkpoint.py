"""Unit tests for checkpoint lifecycle and RNG reproducibility."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from vocseg.models.deeplabv3plus import validate_checkpoint_metadata
from vocseg.training.checkpoint import (
    load_checkpoint,
    save_final_model,
    save_resume_checkpoint,
)
from vocseg.training.reproducibility import restore_rng_state


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 21, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def test_resume_checkpoint_rng_restoration(tmp_path: Path):
    ckpt_path = tmp_path / "last.ckpt"
    model = DummyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Đặt trạng thái ngẫu nhiên
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)

    save_resume_checkpoint(
        path=ckpt_path,
        epoch=5,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        best_metric=0.72,
        config_dict={"test": "config"},
    )

    # Thay đổi trạng thái ngẫu nhiên
    _ = random.random()
    _ = np.random.rand()
    _ = torch.rand(5)

    # Tải lại checkpoint và phục hồi
    loaded = load_checkpoint(ckpt_path, torch.device("cpu"))
    assert loaded["epoch"] == 5
    assert loaded["best_metric"] == 0.72
    assert "rng_state" in loaded

    restore_rng_state(loaded["rng_state"])

    # Tạo số ngẫu nhiên sau restore
    val_py = random.random()
    val_np = np.random.rand()
    val_torch = torch.rand(1).item()

    # Kiểm tra tính lặp lại từ trạng thái ban đầu
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    assert val_py == pytest.approx(random.random())
    assert val_np == pytest.approx(np.random.rand())
    assert val_torch == pytest.approx(torch.rand(1).item())


def test_save_final_model_structure(tmp_path: Path):
    path = tmp_path / "final_model.pth"
    model = DummyModel()
    save_final_model(
        path=path,
        model=model,
        config_dict={"data": {"image_size": 320}},
        trained_epochs=31,
    )

    loaded = load_checkpoint(path, torch.device("cpu"))
    assert loaded["artifact_type"] == "final_production_model"
    assert loaded["architecture"] == "deeplabv3plus"
    assert loaded["encoder"] == "resnet50"
    assert loaded["num_classes"] == 21
    assert "model_state_dict" in loaded
    # Không lưu optimizer/scheduler overhead
    assert "optimizer_state_dict" not in loaded
    assert "scheduler_state_dict" not in loaded


def test_validate_checkpoint_metadata():
    valid = {"architecture": "deeplabv3plus", "encoder": "resnet50", "num_classes": 21}
    validate_checkpoint_metadata(valid)

    with pytest.raises(ValueError, match="architecture không tương thích"):
        validate_checkpoint_metadata({"architecture": "unet", "encoder": "resnet50", "num_classes": 21})

    with pytest.raises(ValueError, match="encoder không tương thích"):
        validate_checkpoint_metadata({"architecture": "deeplabv3plus", "encoder": "vgg16", "num_classes": 21})

    with pytest.raises(ValueError, match="num_classes không khớp"):
        validate_checkpoint_metadata({"architecture": "deeplabv3plus", "encoder": "resnet50", "num_classes": 19})
