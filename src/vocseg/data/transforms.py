"""Quy chuẩn tiền xử lý và tăng cường dữ liệu (Data Transforms & Augmentations).

QUY ƯỚC TIỀN XỬ LÝ (TRANSFORM CONTRACT):
- Training pipeline: Random Scale [0.75, 1.5] -> Unbiased Padding (nếu kích thước nhỏ hơn target)
  -> Random Crop (target_h, target_w) -> Random Horizontal Flip (p=0.5) -> Mild Color Jitter (RGB only)
  -> ImageNet Normalize. Mask luôn dùng NEAREST và padding IGNORE_INDEX=255.
- Validation & Serving pipeline: Deterministic Letterbox (giữ nguyên tỷ lệ khung hình, đệm đều vào giữa,
  không crop mất thông tin).
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms

from vocseg.constants import DEFAULT_IMAGE_SIZE, IGNORE_INDEX, IMAGE_MEAN, IMAGE_STD


def calculate_letterbox_geometry(
    width: int,
    height: int,
    target_width: int,
    target_height: int,
) -> tuple[float, int, int, int, int, int, int]:
    """Tính toán hình học letterbox duy nhất (scale, new_w, new_h, pad_left, pad_top, pad_right, pad_bottom)."""
    if width <= 0 or height <= 0 or target_width <= 0 or target_height <= 0:
        raise ValueError("Chiều cao và chiều rộng phải lớn hơn 0")

    scale = min(target_width / width, target_height / height)
    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))

    pad_left = (target_width - new_w) // 2
    pad_top = (target_height - new_h) // 2
    pad_right = target_width - new_w - pad_left
    pad_bottom = target_height - new_h - pad_top

    return scale, new_w, new_h, pad_left, pad_top, pad_right, pad_bottom


def resize_and_pad(
    image: Image.Image,
    mask: Image.Image,
    target_h: int,
    target_w: int,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[Image.Image, Image.Image]:
    """Letterbox ảnh và mặt nạ về kích thước cố định mà không làm méo tỷ lệ."""
    if image.size != mask.size:
        raise ValueError(f"Ảnh và mặt nạ phải cùng kích thước: {image.size} != {mask.size}")

    _, new_w, new_h, pad_left, pad_top, pad_right, pad_bottom = calculate_letterbox_geometry(
        image.width, image.height, target_w, target_h
    )

    image = TF.resize(image, (new_h, new_w), interpolation=transforms.InterpolationMode.BILINEAR)
    mask = TF.resize(mask, (new_h, new_w), interpolation=transforms.InterpolationMode.NEAREST)

    padding = [pad_left, pad_top, pad_right, pad_bottom]
    image_padded = TF.pad(image, padding, fill=0)
    mask_padded = TF.pad(mask, padding, fill=ignore_index)
    return image_padded, mask_padded


class LetterboxTransform:
    """Deterministic letterbox transform cho validation và serving."""

    def __init__(
        self,
        h: int = DEFAULT_IMAGE_SIZE,
        w: int = DEFAULT_IMAGE_SIZE,
        mean: tuple[float, float, float] = IMAGE_MEAN,
        std: tuple[float, float, float] = IMAGE_STD,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        if h <= 0 or w <= 0:
            raise ValueError("Kích thước transform phải lớn hơn 0")
        self.h = h
        self.w = w
        self.normalize = transforms.Normalize(mean, std)
        self.to_tensor = transforms.ToTensor()
        self.ignore_index = ignore_index

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image_padded, mask_padded = resize_and_pad(image, mask, self.h, self.w, ignore_index=self.ignore_index)
        image_t = self.normalize(self.to_tensor(image_padded))
        mask_t = torch.from_numpy(np.array(mask_padded, dtype=np.int64))
        return image_t, mask_t


class TrainJointTransform:
    """Joint augmentations cho training baseline:
    - Random scale [0.75, 1.5]
    - Unbiased random padding nếu kích thước sau scale < target
    - Random crop (h, w)
    - Random horizontal flip (p=0.5)
    - Mild color jitter (chỉ trên RGB image)
    - ImageNet Normalize
    - Mask luôn nội suy NEAREST và điền IGNORE_INDEX=255
    """

    def __init__(
        self,
        h: int = DEFAULT_IMAGE_SIZE,
        w: int = DEFAULT_IMAGE_SIZE,
        mean: tuple[float, float, float] = IMAGE_MEAN,
        std: tuple[float, float, float] = IMAGE_STD,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        if h <= 0 or w <= 0:
            raise ValueError("Kích thước transform phải lớn hơn 0")
        self.h = h
        self.w = w
        self.ignore_index = ignore_index
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        self.normalize = transforms.Normalize(mean, std)

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if image.size != mask.size:
            raise ValueError(f"Ảnh và mặt nạ phải cùng kích thước: {image.size} != {mask.size}")

        scale = random.uniform(0.75, 1.5)
        scaled_h = max(1, round(image.height * scale))
        scaled_w = max(1, round(image.width * scale))

        image = TF.resize(image, (scaled_h, scaled_w), interpolation=transforms.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (scaled_h, scaled_w), interpolation=transforms.InterpolationMode.NEAREST)

        pad_w = max(0, self.w - scaled_w)
        pad_h = max(0, self.h - scaled_h)
        if pad_w > 0 or pad_h > 0:
            pad_left = random.randint(0, pad_w) if pad_w > 0 else 0
            pad_right = pad_w - pad_left
            pad_top = random.randint(0, pad_h) if pad_h > 0 else 0
            pad_bottom = pad_h - pad_top
            image = TF.pad(image, [pad_left, pad_top, pad_right, pad_bottom], fill=0)
            mask = TF.pad(mask, [pad_left, pad_top, pad_right, pad_bottom], fill=self.ignore_index)

        top, left, _, _ = transforms.RandomCrop.get_params(image, (self.h, self.w))
        image = TF.crop(image, top, left, self.h, self.w)
        mask = TF.crop(mask, top, left, self.h, self.w)

        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() > 0.5:
            image = self.color_jitter(image)

        image_t = self.normalize(TF.to_tensor(image))
        mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image_t, mask_t
