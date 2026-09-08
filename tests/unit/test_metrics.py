"""Unit tests for metrics computation and reporting."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from vocseg.evaluation.metrics import (
    SegmentationMetrics,
    compute_adaptive_tolerance_radius,
    compute_boundary_f1_score,
    save_metrics,
)


def test_perfect_prediction():
    pred = torch.tensor([[0, 1], [1, 0]])
    target = pred.clone()

    metrics = SegmentationMetrics(num_classes=2)
    metrics.update(pred, target)
    res = metrics.compute()

    assert res["mean_iou_all"] == 1.0
    assert res["mean_dice_all"] == 1.0
    assert res["pixel_accuracy"] == 1.0


def test_ignore_index_handling():
    pred = torch.tensor([[0, 1], [1, 0]])
    target = torch.tensor([[0, 1], [1, 255]])  # 255 là ignore

    metrics = SegmentationMetrics(num_classes=2)
    metrics.update(pred, target, ignore_index=255)
    res = metrics.compute(ignore_index=255)

    assert res["mean_iou_all"] == 1.0
    assert res["pixel_accuracy"] == 1.0


def test_absent_class_handling():
    pred = torch.tensor([[0, 0], [0, 0]])
    target = torch.tensor([[0, 0], [0, 0]])

    metrics = SegmentationMetrics(num_classes=3)
    metrics.update(pred, target)
    res = metrics.compute()

    assert res["mean_iou_all"] == 1.0
    assert res["present_classes_count"] == 1
    assert np.isnan(res["per_class_iou"][1])


def test_shape_mismatch_raises():
    metrics = SegmentationMetrics(num_classes=2)
    with pytest.raises(ValueError, match="Shape mismatch"):
        metrics.update(np.zeros((2, 2)), np.zeros((3, 3)))


def test_adaptive_boundary_radius():
    r1 = compute_adaptive_tolerance_radius(320, 320)
    # đường chéo = sqrt(320^2 + 320^2) = 452.55 -> * 0.005 = 2.26 -> round = 2
    assert r1 == 2

    r2 = compute_adaptive_tolerance_radius(1080, 1920)
    # đường chéo = 2202.9 -> * 0.005 = 11.01 -> round = 11
    assert r2 == 11


def test_boundary_f1_calculation():
    gt = np.zeros((30, 30), dtype=np.int64)
    gt[10:20, 10:20] = 1
    pred_perfect = gt.copy()

    scores = compute_boundary_f1_score(pred_perfect, gt, num_classes=2, radius=2)
    assert scores[1] == 1.0


def test_save_metrics_json_and_csv(tmp_path: Path):
    metrics = SegmentationMetrics(num_classes=2)
    metrics.update(torch.tensor([[0, 1]]), torch.tensor([[0, 1]]))
    res = metrics.compute()

    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "per_class.csv"
    save_metrics(res, json_path, csv_path)

    assert json_path.is_file()
    assert csv_path.is_file()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["mean_iou_all"] == 1.0
    assert "classes" in data
