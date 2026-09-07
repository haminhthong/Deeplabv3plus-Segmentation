"""Unit tests for transforms and augmentations."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from vocseg.constants import IGNORE_INDEX
from vocseg.data.transforms import (
    LetterboxTransform,
    TrainJointTransform,
    calculate_letterbox_geometry,
    resize_and_pad,
)


def test_letterbox_geometry_calculation():
    # Ảnh ngang 640x480 về 320x320
    scale, new_w, new_h, pad_left, pad_top, pad_right, pad_bottom = calculate_letterbox_geometry(640, 480, 320, 320)
    assert scale == pytest.approx(320 / 640)
    assert new_w == 320
    assert new_h == 240
    assert pad_left == 0
    assert pad_right == 0
    assert pad_top == 40
    assert pad_bottom == 40


def test_letterbox_geometry_vertical():
    # Ảnh dọc 480x640 về 320x320
    scale, new_w, new_h, pad_left, pad_top, pad_right, pad_bottom = calculate_letterbox_geometry(480, 640, 320, 320)
    assert scale == pytest.approx(320 / 640)
    assert new_w == 240
    assert new_h == 320
    assert pad_left == 40
    assert pad_right == 40
    assert pad_top == 0
    assert pad_bottom == 0


def test_letterbox_geometry_invalid():
    with pytest.raises(ValueError):
        calculate_letterbox_geometry(0, 480, 320, 320)
    with pytest.raises(ValueError):
        calculate_letterbox_geometry(640, -10, 320, 320)


def test_resize_and_pad_mismatched_size():
    img = Image.new("RGB", (100, 100))
    mask = Image.new("L", (100, 200))
    with pytest.raises(ValueError, match="cùng kích thước"):
        resize_and_pad(img, mask, 320, 320)


def test_letterbox_transform_output():
    img = Image.new("RGB", (400, 200), color="blue")
    mask = Image.new("L", (400, 200), color=5)

    transform = LetterboxTransform(h=320, w=320)
    img_t, mask_t = transform(img, mask)

    assert isinstance(img_t, torch.Tensor)
    assert isinstance(mask_t, torch.Tensor)
    assert img_t.shape == (3, 320, 320)
    assert mask_t.shape == (320, 320)
    assert mask_t.dtype == torch.int64

    # Kiểm tra phần padding chứa giá trị IGNORE_INDEX
    # Với ảnh 400x200 resize về 320x160 -> pad_top = 80, pad_bottom = 80
    assert mask_t[0, 160].item() == IGNORE_INDEX
    assert mask_t[319, 160].item() == IGNORE_INDEX
    # Phần nội dung ảnh ở giữa chứa class 5
    assert mask_t[160, 160].item() == 5


def test_train_joint_transform_dimensions():
    img = Image.new("RGB", (250, 350), color="red")
    mask = Image.new("L", (250, 350), color=1)

    transform = TrainJointTransform(h=320, w=320)
    img_t, mask_t = transform(img, mask)

    assert img_t.shape == (3, 320, 320)
    assert mask_t.shape == (320, 320)
    assert mask_t.dtype == torch.int64
    # Mask chỉ chứa giá trị 1 hoặc IGNORE_INDEX (255)
    unique_vals = set(torch.unique(mask_t).tolist())
    assert unique_vals.issubset({1, IGNORE_INDEX})
