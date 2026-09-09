"""Thẩm định tính toàn vẹn của dataset VOC và tạo Data Manifest có kiểm chứng SHA-256."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from vocseg.constants import IGNORE_INDEX, NUM_CLASSES, VOC_CLASSES
from vocseg.data.splits import calculate_file_sha256
from vocseg.schemas import AuditReport, DatasetManifest


def audit_voc_dataset(
    data_root: Path,
    split_ids_map: Dict[str, List[str]],
) -> AuditReport:
    """Audit toàn diện tính toàn vẹn dữ liệu, kích thước, phân bố lớp và kiểm tra rò rỉ SHA-256."""
    data_root = Path(data_root)
    jpeg_dir = data_root / "JPEGImages"
    mask_dir = data_root / "SegmentationClass"

    missing_images = 0
    missing_masks = 0
    dimension_mismatches = 0
    invalid_mask_values = 0
    duplicate_ids_count = 0
    exact_sha256_duplicates = 0

    seen_ids: set[str] = set()
    hash_to_id: Dict[str, tuple[str, str]] = {}
    details: Dict[str, Any] = {
        "class_distribution": {},
        "mismatch_samples": [],
        "duplicate_hash_samples": [],
    }

    total_images_checked = 0

    for split_name, ids in split_ids_map.items():
        split_pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        split_image_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        total_valid_pixels = 0

        for img_id in ids:
            total_images_checked += 1
            if img_id in seen_ids:
                duplicate_ids_count += 1
            seen_ids.add(img_id)

            img_path = jpeg_dir / f"{img_id}.jpg"
            mask_path = mask_dir / f"{img_id}.png"

            if not img_path.is_file():
                missing_images += 1
                continue
            if not mask_path.is_file():
                missing_masks += 1
                continue

            # SHA-256 exact duplicate check
            img_hash = calculate_file_sha256(img_path)
            if img_hash in hash_to_id:
                prev_split, prev_id = hash_to_id[img_hash]
                exact_sha256_duplicates += 1
                details["duplicate_hash_samples"].append(
                    {
                        "hash": img_hash,
                        "first": f"{prev_split}/{prev_id}",
                        "second": f"{split_name}/{img_id}",
                    }
                )
            else:
                hash_to_id[img_hash] = (split_name, img_id)

            # Dimension & Mask Validation
            try:
                with Image.open(img_path) as img, Image.open(mask_path) as mask:
                    if img.size != mask.size:
                        dimension_mismatches += 1
                        details["mismatch_samples"].append(
                            {
                                "id": img_id,
                                "image_size": list(img.size),
                                "mask_size": list(mask.size),
                            }
                        )

                    mask_arr = np.array(mask)
                    invalid = np.setdiff1d(mask_arr, list(range(NUM_CLASSES)) + [IGNORE_INDEX])
                    if len(invalid) > 0:
                        invalid_mask_values += 1

                    valid_pixels = mask_arr[mask_arr != IGNORE_INDEX]
                    if len(valid_pixels) > 0:
                        counts = np.bincount(valid_pixels, minlength=NUM_CLASSES)
                        split_pixel_counts[: len(counts)] += counts[:NUM_CLASSES]
                        total_valid_pixels += int(counts[:NUM_CLASSES].sum())

                        for c in np.unique(valid_pixels):
                            if c < NUM_CLASSES:
                                split_image_counts[c] += 1
            except Exception as ex:
                missing_images += 1
                details["mismatch_samples"].append({"id": img_id, "error": str(ex)})

        bg_px = int(split_pixel_counts[0])
        fg_px = int(split_pixel_counts[1:].sum())
        fg_bg_ratio = float(fg_px / max(bg_px, 1))

        details["class_distribution"][split_name] = {
            "total_images": len(ids),
            "total_pixels": total_valid_pixels,
            "background_pixels": bg_px,
            "foreground_pixels": fg_px,
            "foreground_to_background_ratio": fg_bg_ratio,
            "present_classes_count": int(np.sum(split_pixel_counts > 0)),
            "per_class_pixels": {VOC_CLASSES[c]: int(split_pixel_counts[c]) for c in range(NUM_CLASSES)},
            "per_class_image_occurrences": {VOC_CLASSES[c]: int(split_image_counts[c]) for c in range(NUM_CLASSES)},
        }

    status = "PASSED"
    if (
        missing_images > 0
        or missing_masks > 0
        or dimension_mismatches > 0
        or invalid_mask_values > 0
        or duplicate_ids_count > 0
        or exact_sha256_duplicates > 0
    ):
        status = "FAILED"

    return AuditReport(
        dataset_name="Pascal VOC 2012",
        source_train_images=len(split_ids_map.get("dev_train", [])) + len(split_ids_map.get("dev_val", [])),
        source_val_images=len(split_ids_map.get("holdout", [])),
        dev_train_images=len(split_ids_map.get("dev_train", [])),
        dev_val_images=len(split_ids_map.get("dev_val", [])),
        holdout_images=len(split_ids_map.get("holdout", [])),
        missing_images=missing_images,
        missing_masks=missing_masks,
        dimension_mismatches=dimension_mismatches,
        invalid_mask_values=invalid_mask_values,
        duplicate_ids=duplicate_ids_count,
        exact_sha256_duplicates=exact_sha256_duplicates,
        audit_status=status,
        details=details,
    )


def generate_dataset_manifest(
    data_root: Path,
    splits_info: Dict[str, Any],
    audit_report: AuditReport,
    seed: int = 42,
) -> DatasetManifest:
    """Tạo đối tượng DatasetManifest chuẩn hóa có đầy đủ checksum và kết quả audit."""
    return DatasetManifest(
        dataset="Pascal VOC 2012",
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX,
        seed=seed,
        stratification_method="multilabel",
        source_train_sha256=splits_info["source_train"]["sha256"],
        source_val_sha256=splits_info["source_val"]["sha256"],
        dev_train_sha256=splits_info["dev_train"]["sha256"],
        dev_val_sha256=splits_info["dev_val"]["sha256"],
        holdout_sha256=splits_info["holdout"]["sha256"],
        splits={
            "dev_train": {
                "count": splits_info["dev_train"]["count"],
                "file": "dev_train.txt",
                "sha256": splits_info["dev_train"]["sha256"],
            },
            "dev_val": {
                "count": splits_info["dev_val"]["count"],
                "file": "dev_val.txt",
                "sha256": splits_info["dev_val"]["sha256"],
            },
            "holdout": {
                "count": splits_info["holdout"]["count"],
                "file": "holdout.txt",
                "sha256": splits_info["holdout"]["sha256"],
                "source": "official_voc_val",
                "locked": True,
            },
        },
        audit=audit_report,
    )
