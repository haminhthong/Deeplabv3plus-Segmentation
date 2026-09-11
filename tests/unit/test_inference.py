"""Unit tests for canonical Predictor geometry, uncertainty map and parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from vocseg.inference.predictor import Predictor


class DummySegmentationModel(nn.Module):
    def __init__(self, num_classes: int = 21):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        # Trả về logits cố định
        logits = torch.zeros((b, self.num_classes, h, w), dtype=torch.float32)
        logits[:, 1, :, :] = 2.0  # Lớp 1 có logit cao nhất
        return logits


@pytest.fixture
def dummy_predictor(tmp_path: Path) -> Predictor:
    ckpt_path = tmp_path / "dummy_final.pth"
    model = DummySegmentationModel(21)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "deeplabv3plus",
            "encoder": "resnet50",
            "num_classes": 21,
            "model_version": "1.0.0",
        },
        ckpt_path,
    )
    predictor = Predictor.__new__(Predictor)
    predictor.device = torch.device("cpu")
    predictor.image_size = 320
    predictor.checkpoint_meta = {"model_version": "1.0.0"}
    predictor.model = model
    return predictor


@pytest.mark.parametrize(
    "width, height",
    [
        (640, 480),  # Ảnh ngang
        (480, 640),  # Ảnh dọc
        (500, 500),  # Ảnh vuông
        (120, 80),  # Ảnh nhỏ
    ],
)
def test_predictor_output_dimensions_exact_parity(dummy_predictor: Predictor, width: int, height: int):
    image = Image.new("RGB", (width, height), color="green")
    res = dummy_predictor.predict(image)

    # Output mask PHẢI khớp chính xác (height, width) của ảnh gốc
    assert res.hard_mask.shape == (height, width)
    assert res.max_prob_map.shape == (height, width)
    assert res.entropy_map.shape == (height, width)
    assert res.original_size == (width, height)
    assert res.hard_mask.dtype == np.int64


def test_predictor_uncertainty_and_reliability_maps(dummy_predictor: Predictor):
    image = Image.new("RGB", (300, 200), color="blue")
    res = dummy_predictor.predict(image)

    assert np.all(res.max_prob_map >= 0.0) and np.all(res.max_prob_map <= 1.0)
    assert np.all(res.entropy_map >= 0.0) and np.all(res.entropy_map <= 1.0)
    assert res.latency_ms > 0


def test_predictor_deterministic_output(dummy_predictor: Predictor):
    image = Image.new("RGB", (200, 200), color="yellow")
    res1 = dummy_predictor.predict(image)
    res2 = dummy_predictor.predict(image)

    np.testing.assert_array_equal(res1.hard_mask, res2.hard_mask)
    np.testing.assert_array_almost_equal(res1.entropy_map, res2.entropy_map)


def test_predictor_oversized_image_rejected(dummy_predictor: Predictor):
    # Tạo ảnh vượt quá giới hạn 25 Megapixels
    huge_image = Image.new("RGB", (6000, 5000))
    with pytest.raises(ValueError, match="Kích thước ảnh quá lớn"):
        dummy_predictor.predict(huge_image)
