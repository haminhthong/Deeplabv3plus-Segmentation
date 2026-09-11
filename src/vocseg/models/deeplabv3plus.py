"""Khởi tạo và kiểm định mô hình DeepLabV3+ ResNet50 chuẩn hóa."""

from __future__ import annotations

from typing import Any

import segmentation_models_pytorch as smp
from torch import nn

from vocseg.constants import NUM_CLASSES


def build_deeplabv3plus(
    encoder: str = "resnet50",
    encoder_weights: str | None = "imagenet",
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """Khởi tạo mô hình DeepLabV3+ ResNet50 theo chuẩn sản phẩm.

    Đầu ra là raw logits [B, num_classes, H, W] (không Softmax trước loss).
    """
    if num_classes <= 0:
        raise ValueError("num_classes phải lớn hơn 0")
    if encoder.lower() != "resnet50":
        raise ValueError(f"Core baseline chỉ hỗ trợ encoder 'resnet50', nhận được: '{encoder}'")

    model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=num_classes,
        activation=None,
    )
    return model


def validate_checkpoint_metadata(
    checkpoint: dict[str, Any],
    expected_num_classes: int = NUM_CLASSES,
    expected_encoder: str = "resnet50",
    expected_arch: str = "deeplabv3plus",
) -> None:
    """Xác thực tính tương thích của checkpoint artifact trước khi tải weights.

    Fail-fast nếu phát hiện sai lệch siêu tham số hoặc kiến trúc.
    """
    if not isinstance(checkpoint, dict):
        return  # Raw state dict, bỏ qua kiểm tra siêu dữ liệu

    arch = checkpoint.get("architecture")
    if arch and arch.lower() != expected_arch.lower():
        raise ValueError(f"Checkpoint architecture không tương thích: '{arch}' != kỳ vọng '{expected_arch}'")

    enc = checkpoint.get("encoder")
    if enc and enc.lower() != expected_encoder.lower():
        raise ValueError(f"Checkpoint encoder không tương thích: '{enc}' != kỳ vọng '{expected_encoder}'")

    n_cls = checkpoint.get("num_classes")
    if n_cls is not None and int(n_cls) != expected_num_classes:
        raise ValueError(f"Checkpoint num_classes không khớp: {n_cls} != kỳ vọng {expected_num_classes}")
