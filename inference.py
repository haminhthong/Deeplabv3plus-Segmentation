"""Tương thích ngược (Backward Compatibility) cho inference.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

from experiments.architectures import build_experimental_model
from vocseg.constants import DEFAULT_IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, NUM_CLASSES
from vocseg.data.transforms import calculate_letterbox_geometry
from vocseg.inference.predictor import Predictor
from vocseg.inference.visualization import overlay_mask
from vocseg.models.deeplabv3plus import build_deeplabv3plus
from vocseg.training.checkpoint import load_checkpoint


def build_model(
    encoder: str = "resnet50",
    encoder_weights: Optional[str] = "imagenet",
    num_classes: int = NUM_CLASSES,
    architecture: str = "deeplabv3plus",
):
    arch = architecture.lower()
    if arch in ("deeplabv3plus", "deeplabv3+"):
        return build_deeplabv3plus(encoder=encoder, encoder_weights=encoder_weights, num_classes=num_classes)
    elif arch in ("unet", "fpn"):
        return build_experimental_model(architecture=arch, encoder=encoder, encoder_weights=encoder_weights, num_classes=num_classes)
    else:
        raise ValueError(f"Kiến trúc không được hỗ trợ: {architecture}. Lựa chọn hợp lệ: deeplabv3plus, unet, fpn")


def load_checkpoint_model(path: str | Path, device: torch.device):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")
    checkpoint = load_checkpoint(path, device)
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    state_dict = metadata.get("model_state_dict", checkpoint)
    encoder = metadata.get("encoder", "resnet50")
    architecture = metadata.get("architecture", "deeplabv3plus")
    num_classes = int(metadata.get("num_classes", NUM_CLASSES))

    model = build_model(encoder=encoder, encoder_weights=None, num_classes=num_classes, architecture=architecture)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, metadata


def prepare_image(image: Image.Image, image_size: int):
    if image_size <= 0:
        raise ValueError("Kích thước ảnh đầu vào phải lớn hơn 0")
    _, resized_w, resized_h, pad_left, pad_top, pad_right, pad_bottom = calculate_letterbox_geometry(
        image.width, image.height, image_size, image_size
    )
    resized = TF.resize(image, (resized_h, resized_w), interpolation=transforms.InterpolationMode.BILINEAR)
    padded = TF.pad(resized, [pad_left, pad_top, pad_right, pad_bottom], fill=0)
    tensor = TF.normalize(TF.to_tensor(padded), IMAGE_MEAN, IMAGE_STD)
    return tensor, (pad_left, pad_top, resized_w, resized_h), image.size


@torch.inference_mode()
def predict_with_uncertainty(
    model: torch.nn.Module,
    image: Image.Image,
    image_size: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    was_training = model.training
    model.eval()
    try:
        tensor, (left, top, resized_w, resized_h), (original_w, original_h) = prepare_image(image, image_size)
        logits = model(tensor.unsqueeze(0).to(device))
        logits = logits[:, :, top : top + resized_h, left : left + resized_w]
        logits = F.interpolate(logits, size=(original_h, original_w), mode="bilinear", align_corners=False)

        probs = F.softmax(logits, dim=1).squeeze(0)
        hard_mask = probs.argmax(dim=0).cpu().numpy().astype(np.int64)
        max_prob = probs.max(dim=0).values.cpu().numpy().astype(np.float32)

        eps = 1e-7
        entropy = -(probs * torch.log(probs + eps)).sum(dim=0)
        norm_factor = float(np.log(max(probs.shape[0], 2)))
        normalized_entropy = (entropy / norm_factor).clamp(0.0, 1.0).cpu().numpy().astype(np.float32)

        return {
            "hard_mask": hard_mask,
            "max_prob_map": max_prob,
            "entropy_map": normalized_entropy,
            "softmax_probs": probs.cpu().numpy().astype(np.float32),
        }
    finally:
        if was_training:
            model.train()


@torch.inference_mode()
def predict_original_size(model, image: Image.Image, image_size: int, device: torch.device) -> np.ndarray:
    was_training = model.training
    model.eval()
    try:
        tensor, (left, top, resized_w, resized_h), (original_w, original_h) = prepare_image(image, image_size)
        logits = model(tensor.unsqueeze(0).to(device))
        logits = logits[:, :, top : top + resized_h, left : left + resized_w]
        logits = F.interpolate(logits, size=(original_h, original_w), mode="bilinear", align_corners=False)
        return logits.argmax(1).squeeze(0).cpu().numpy().astype(np.int64)
    finally:
        if was_training:
            model.train()


__all__ = [
    "build_model",
    "load_checkpoint_model",
    "prepare_image",
    "overlay_mask",
    "predict_with_uncertainty",
    "predict_original_size",
]
