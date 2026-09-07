"""Schemas và Contracts dữ liệu chuẩn hóa cho hệ thống phân đoạn."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class ClassPresence(BaseModel):
    id: int
    name: str
    pixels: int
    coverage: float = Field(description="Tỷ lệ diện tích (%) trên toàn ảnh")


class PredictionMetadata(BaseModel):
    model_version: str = "1.0.0"
    architecture: str = "deeplabv3plus"
    encoder: str = "resnet50"
    width: int
    height: int
    classes_present: list[ClassPresence]
    mean_entropy: float
    mean_max_prob: float
    latency_ms: float


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str = "deeplabv3plus-resnet50-v1"
    version: str = "1.0.0"
    device: str


class AuditReport(BaseModel):
    dataset_name: str = "Pascal VOC 2012"
    source_train_images: int = 0
    source_val_images: int = 0
    dev_train_images: int = 0
    dev_val_images: int = 0
    holdout_images: int = 0
    missing_images: int = 0
    missing_masks: int = 0
    dimension_mismatches: int = 0
    invalid_mask_values: int = 0
    duplicate_ids: int = 0
    exact_sha256_duplicates: int = 0
    audit_status: str = "PENDING"
    details: dict[str, Any] = Field(default_factory=dict)


class DatasetManifest(BaseModel):
    dataset: str = "Pascal VOC 2012"
    num_classes: int = 21
    ignore_index: int = 255
    seed: int = 42
    stratification_method: str = "multilabel"

    source_train_sha256: str = ""
    source_val_sha256: str = ""

    dev_train_sha256: str = ""
    dev_val_sha256: str = ""
    holdout_sha256: str = ""

    splits: dict[str, dict[str, Any]] = Field(default_factory=dict)
    audit: AuditReport
