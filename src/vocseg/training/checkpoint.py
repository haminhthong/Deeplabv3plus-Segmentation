"""Quản lý và lưu trữ 3 tầng Artifact Checkpoint chuẩn mực: last.ckpt, best.ckpt, final_model.pth."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from vocseg.constants import DEFAULT_IMAGE_SIZE, IGNORE_INDEX, IMAGE_MEAN, IMAGE_STD, NUM_CLASSES, VOC_CLASSES
from vocseg.training.reproducibility import capture_rng_state


def get_git_commit() -> Optional[str]:
    """Lấy mã Git SHA của commit hiện tại."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def calculate_dict_sha256(data: Dict[str, Any]) -> str:
    """Tính SHA-256 từ nội dung chuỗi của từ điển cấu hình."""
    content = str(sorted(data.items())).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def save_resume_checkpoint(
    path: Path | str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    best_metric: float,
    config_dict: Dict[str, Any],
    manifest_sha256: str = "",
    best_epoch: int = -1,
) -> None:
    """Lưu last.ckpt chứa đầy đủ trạng thái để resume chính xác 100%."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "artifact_type": "resume_checkpoint",
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "rng_state": capture_rng_state(),
        "config": config_dict,
        "split_manifest_sha256": manifest_sha256,
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
    }
    torch.save(checkpoint, path)


def save_best_checkpoint(
    path: Path | str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    best_miou: float,
    config_dict: Dict[str, Any],
    val_metrics: Dict[str, Any],
    manifest_sha256: str = "",
) -> None:
    """Lưu best.ckpt: Mô hình phát triển tốt nhất theo validation mIoU."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "artifact_type": "best_development_checkpoint",
        "best_epoch": epoch,
        "best_val_miou": best_miou,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "config": config_dict,
        "val_metrics": val_metrics,
        "split_manifest_sha256": manifest_sha256,
        "git_commit": get_git_commit(),
    }
    torch.save(checkpoint, path)


def save_final_model(
    path: Path | str,
    model: nn.Module,
    config_dict: Dict[str, Any],
    trained_epochs: int,
    manifest_sha256: str = "",
    model_version: str = "1.0.0",
) -> None:
    """Lưu final_model.pth: Artifact phát hành triển khai tinh gọn, không chứa optimizer overhead."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "artifact_type": "final_production_model",
        "model_version": model_version,
        "architecture": "deeplabv3plus",
        "encoder": "resnet50",
        "num_classes": NUM_CLASSES,
        "image_size": config_dict.get("data", {}).get("image_size", DEFAULT_IMAGE_SIZE),
        "normalization": {
            "mean": list(IMAGE_MEAN),
            "std": list(IMAGE_STD),
        },
        "ignore_index": IGNORE_INDEX,
        "class_names": VOC_CLASSES,
        "dataset": "Pascal VOC 2012",
        "split_manifest_sha256": manifest_sha256,
        "config_sha256": calculate_dict_sha256(config_dict),
        "git_commit": get_git_commit(),
        "training_protocol": {
            "loss": "CE (ignore 255) + 0.5 * Dice",
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "epochs_fitted": trained_epochs,
            "seed": config_dict.get("training", {}).get("seed", 42),
        },
        "model_state_dict": model.state_dict(),
    }
    torch.save(artifact, path)


def load_checkpoint(
    path: Path | str,
    device: torch.device,
) -> Dict[str, Any]:
    """Tải và trả về từ điển checkpoint kèm siêu dữ liệu."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Định dạng checkpoint không hợp lệ: {path}")
    return checkpoint
