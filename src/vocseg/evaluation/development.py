"""Đánh giá phát triển (Development Evaluation): Nhanh, dùng trong mỗi epoch trên dev_val."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from vocseg.constants import IGNORE_INDEX, NUM_CLASSES
from vocseg.evaluation.metrics import SegmentationMetrics


@torch.inference_mode()
def evaluate_development(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[float, dict[str, Any]]:
    """Đánh giá nhanh trên không gian model letterboxed (320x320) sau mỗi epoch."""
    was_training = model.training
    model.eval()

    total_loss = 0.0
    metrics = SegmentationMetrics(num_classes)

    try:
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            metrics.update(preds, masks, ignore_index=ignore_index, compute_boundary=False)

        avg_loss = total_loss / max(len(val_loader), 1)
        res = metrics.compute(ignore_index=ignore_index)
        return avg_loss, res
    finally:
        if was_training:
            model.train()
