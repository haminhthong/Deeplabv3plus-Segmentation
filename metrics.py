"""Tương thích ngược (Backward Compatibility) cho metrics.py."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from vocseg.evaluation.metrics import (
    SegmentationMetrics,
    compute_adaptive_tolerance_radius,
    compute_boundary_f1_score,
    dilate_boundary,
    extract_boundary,
    extract_confusion_analysis,
    save_metrics,
)


def calculate_region_size_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 21,
    ignore_index: int = 255,
) -> Dict[str, Any]:
    """[DEPRECATED] Hàm tính tổng pixel cũ. Đã loại khỏi Core Benchmark v1 theo khuyến nghị."""
    valid = targets != ignore_index
    small_ious: list[float] = []
    medium_ious: list[float] = []
    large_ious: list[float] = []

    for c in range(1, num_classes):
        gt_c = (targets == c) & valid
        pred_c = (predictions == c) & valid

        area = int(gt_c.sum())
        if area == 0:
            continue

        intersection = int((gt_c & pred_c).sum())
        union = int((gt_c | pred_c).sum())
        iou = float(intersection / union) if union > 0 else 0.0

        if area < 1024:
            small_ious.append(iou)
        elif area < 9216:
            medium_ious.append(iou)
        else:
            large_ious.append(iou)

    return {
        "small_region_miou": float(np.mean(small_ious)) if small_ious else None,
        "medium_region_miou": float(np.mean(medium_ious)) if medium_ious else None,
        "large_region_miou": float(np.mean(large_ious)) if large_ious else None,
        "small_regions_count": len(small_ious),
        "medium_regions_count": len(medium_ious),
        "large_regions_count": len(large_ious),
    }


__all__ = [
    "extract_boundary",
    "dilate_boundary",
    "compute_boundary_f1_score",
    "compute_adaptive_tolerance_radius",
    "extract_confusion_analysis",
    "SegmentationMetrics",
    "save_metrics",
    "calculate_region_size_metrics",
]
