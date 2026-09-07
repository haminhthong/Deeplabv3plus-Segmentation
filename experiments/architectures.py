"""Experimental models archive (U-Net, FPN).

LƯU Ý: Các kiến trúc này được lưu trữ phục vụ nghiên cứu thực nghiệm / ablation.
Hệ thống chính (Production Core Pipeline) chỉ tập trung độc quyền vào DeepLabV3+ ResNet50.
"""

from __future__ import annotations

from typing import Optional

import segmentation_models_pytorch as smp
import torch.nn as nn

from vocseg.constants import NUM_CLASSES


def build_experimental_model(
    architecture: str,
    encoder: str = "resnet50",
    encoder_weights: Optional[str] = "imagenet",
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    arch = architecture.lower()
    if arch == "unet":
        return smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=None,
        )
    elif arch == "fpn":
        return smp.FPN(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=None,
        )
    else:
        raise ValueError(f"Kiến trúc thực nghiệm không hợp lệ: {architecture}. Lựa chọn: unet, fpn")
