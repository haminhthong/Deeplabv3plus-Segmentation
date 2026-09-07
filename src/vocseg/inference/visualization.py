"""Tiện ích trực quan hóa (Mask colormap, overlay, bản đồ bất định)."""

from __future__ import annotations

import io
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from vocseg.constants import IGNORE_INDEX, VOC_COLORMAP, mask_to_color_rgb


def overlay_mask(
    image: np.ndarray,
    color_mask: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Phủ mặt nạ màu lên ảnh RGB với độ trong suốt alpha."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = image.astype(np.float32) * (1.0 - alpha) + color_mask.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    """Chuyển mảng mặt nạ 2D (int64) thành PNG bytes sử dụng colormap."""
    rgb = mask_to_color_rgb(mask, ignore_index=IGNORE_INDEX)
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
