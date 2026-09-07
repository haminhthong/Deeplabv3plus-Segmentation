"""Module cấu hình typed configuration cho VOC Segmentation System."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from vocseg.constants import DEFAULT_IMAGE_SIZE, IGNORE_INDEX, IMAGE_MEAN, IMAGE_STD, NUM_CLASSES


@dataclass
class ModelConfig:
    architecture: str = "deeplabv3plus"
    encoder: str = "resnet50"
    encoder_weights: str | None = "imagenet"
    num_classes: int = NUM_CLASSES


@dataclass
class DataConfig:
    dataset_name: str = "Pascal VOC 2012"
    image_size: int = DEFAULT_IMAGE_SIZE
    ignore_index: int = IGNORE_INDEX
    mean: tuple[float, float, float] = IMAGE_MEAN
    std: tuple[float, float, float] = IMAGE_STD


@dataclass
class TrainingConfig:
    batch_size: int = 8
    epochs: int = 50
    optimizer: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    eta_min: float = 1e-6
    amp: bool = True
    seed: int = 42
    num_workers: int = 2
    patience: int = 0
    deterministic: bool = False


@dataclass
class LossConfig:
    cross_entropy: float = 1.0
    dice: float = 0.5


@dataclass
class SelectionConfig:
    metric: str = "val_miou_all"


@dataclass
class PathConfig:
    data_root: Path = Path("data/VOC2012_train_val/VOC2012_train_val")
    output_dir: Path = Path("outputs")
    checkpoint_dir: Path = Path("checkpoints")
    manifest_path: Path = Path("artifacts/data/dataset_manifest.json")
    splits_dir: Path = Path("artifacts/data/splits")


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> AppConfig:
        yaml_path = Path(yaml_path)
        if not yaml_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file cấu hình YAML: {yaml_path}")

        with yaml_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        config = cls()
        if "model" in raw:
            for k, v in raw["model"].items():
                if hasattr(config.model, k):
                    setattr(config.model, k, v)
        if "data" in raw:
            for k, v in raw["data"].items():
                if hasattr(config.data, k):
                    if k in ("mean", "std") and isinstance(v, list):
                        v = tuple(v)
                    setattr(config.data, k, v)
        if "training" in raw:
            for k, v in raw["training"].items():
                if hasattr(config.training, k):
                    setattr(config.training, k, v)
        if "loss" in raw:
            for k, v in raw["loss"].items():
                if hasattr(config.loss, k):
                    setattr(config.loss, k, v)
        if "selection" in raw:
            for k, v in raw["selection"].items():
                if hasattr(config.selection, k):
                    setattr(config.selection, k, v)
        if "paths" in raw:
            for k, v in raw["paths"].items():
                if hasattr(config.paths, k):
                    setattr(config.paths, k, Path(v))
        return config


def configure_console() -> None:
    """Cho phép terminal Windows hiển thị thông báo tiếng Việt."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
