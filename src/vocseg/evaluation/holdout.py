"""Đánh giá Locked Holdout (Official VOC val) tại độ phân giải gốc của ảnh thông qua Predictor Canonical."""

from __future__ import annotations

import logging
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from vocseg.constants import IGNORE_INDEX, NUM_CLASSES
from vocseg.data.splits import read_split_file
from vocseg.evaluation.metrics import (
    SegmentationMetrics,
    compute_adaptive_tolerance_radius,
    compute_boundary_f1_score,
)
from vocseg.inference.predictor import Predictor

logger = logging.getLogger(__name__)


def evaluate_holdout_dataset(
    predictor: Predictor,
    data_root: Path | str,
    holdout_split_file: Path | str,
) -> dict[str, Any]:
    """Đánh giá tập Holdout bị khóa (LOCKED) tại độ phân giải gốc của ảnh.

    NGUYÊN TẮC: Sử dụng đúng 100% logic của Predictor Canonical để đảm bảo tính nhất quán (Parity).
    """
    data_root = Path(data_root)
    jpeg_dir = data_root / "JPEGImages"
    mask_dir = data_root / "SegmentationClass"

    sample_ids = read_split_file(Path(holdout_split_file))
    metrics = SegmentationMetrics(NUM_CLASSES)

    latencies_ms: list[float] = []
    t_start = time.perf_counter()

    for i, sid in enumerate(sample_ids):
        img_p = jpeg_dir / f"{sid}.jpg"
        mask_p = mask_dir / f"{sid}.png"

        if not img_p.is_file() or not mask_p.is_file():
            raise FileNotFoundError(f"Thiếu file mẫu {sid} trong tập dữ liệu: {img_p} hoặc {mask_p}")

        with Image.open(img_p) as img_src:
            raw_img = img_src.convert("RGB")
        with Image.open(mask_p) as mask_src:
            gt_mask = np.asarray(mask_src, dtype=np.int64)

        # 1. Dự đoán qua Predictor Canonical tại kích thước gốc
        pred_res = predictor.predict(raw_img)
        latencies_ms.append(pred_res.latency_ms)

        pred_mask = pred_res.hard_mask

        # Kiểm tra kích thước tuyệt đối khớp với Ground Truth gốc
        if pred_mask.shape != gt_mask.shape:
            raise ValueError(
                f"Lỗi Parity: Dự đoán {pred_mask.shape} không khớp kích thước GT gốc {gt_mask.shape} cho mẫu {sid}"
            )

        # 2. Cập nhật metric confusion matrix
        metrics.update(pred_mask, gt_mask, ignore_index=IGNORE_INDEX, compute_boundary=False)

        # 3. Tính Boundary F1 với dung sai thích ứng theo độ phân giải gốc
        adaptive_radius = compute_adaptive_tolerance_radius(gt_mask.shape[0], gt_mask.shape[1])
        b_scores = compute_boundary_f1_score(
            pred_mask, gt_mask, NUM_CLASSES, radius=adaptive_radius, ignore_index=IGNORE_INDEX
        )
        for c, score in b_scores.items():
            metrics.boundary_scores[c].append(score)

        if (i + 1) % 100 == 0 or (i + 1) == len(sample_ids):
            logger.info("Đã đánh giá: %d/%d mẫu holdout...", i + 1, len(sample_ids))

    total_time = time.perf_counter() - t_start
    result = metrics.compute(ignore_index=IGNORE_INDEX)

    # Profiling & Latency Breakdown
    p50_lat = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
    p95_lat = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0
    mean_lat = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    fps = len(sample_ids) / max(total_time, 1e-6)

    device_name = torch.cuda.get_device_name(0) if predictor.device.type == "cuda" else platform.processor() or "CPU"

    result["profiling"] = {
        "images_evaluated": len(sample_ids),
        "total_time_seconds": round(total_time, 2),
        "fps": round(fps, 2),
        "latency_ms_per_image": {
            "mean": round(mean_lat, 2),
            "p50": round(p50_lat, 2),
            "p95": round(p95_lat, 2),
        },
        "hardware": {
            "device_type": predictor.device.type,
            "device_name": device_name,
            "pytorch_version": torch.__version__,
        },
    }
    result["model_version"] = predictor.checkpoint_meta.get("model_version", "1.0.0")
    result["benchmark_protocol"] = "Original-Resolution Inference with Adaptive Boundary Tolerance"
    return result
