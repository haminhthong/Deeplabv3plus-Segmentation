"""Cấu hình tương thích ngược cho hệ thống phân đoạn ảnh DeepLabV3+ ResNet50."""

from __future__ import annotations

from pathlib import Path

from vocseg.config import configure_console
from vocseg.constants import (
    DEFAULT_IMAGE_SIZE,
    IGNORE_INDEX,
    IMAGE_MEAN,
    IMAGE_STD,
    NUM_CLASSES,
)

IMAGE_SIZE = DEFAULT_IMAGE_SIZE
VOC_ROOT = Path("data") / "VOC2012_train_val" / "VOC2012_train_val"
OUTPUT_DIR = Path("outputs")
CHECKPOINT_DIR = Path("checkpoints")

# Thống nhất artifact path: ưu tiên final_model.pth, tiếp theo best.ckpt
FINAL_MODEL_PATH = CHECKPOINT_DIR / "final_model.pth"
BEST_CKPT_PATH = CHECKPOINT_DIR / "best.ckpt"
CHECKPOINT_PATH = FINAL_MODEL_PATH if FINAL_MODEL_PATH.is_file() else BEST_CKPT_PATH

__all__ = [
    "NUM_CLASSES",
    "IGNORE_INDEX",
    "IMAGE_SIZE",
    "IMAGE_MEAN",
    "IMAGE_STD",
    "VOC_ROOT",
    "OUTPUT_DIR",
    "CHECKPOINT_DIR",
    "CHECKPOINT_PATH",
    "FINAL_MODEL_PATH",
    "BEST_CKPT_PATH",
    "configure_console",
]
