"""Quản lý phân tách dữ liệu (Split Contract) và Phân tầng đa nhãn (Multilabel Stratification).

Quy ước phân chia:
- VOC official train (khoảng 1,464 ảnh) -> Development Dataset:
    ├── DEV TRAIN (~85%)
    └── DEV VAL (~15%) - dùng chọn model checkpoint & hyperparameter
- VOC official val (khoảng 1,449 ảnh) -> LOCKED HOLDOUT:
    └── holdout (100% official val) - chỉ mở đúng 1 lần cho báo cáo cuối cùng
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vocseg.constants import IGNORE_INDEX, NUM_CLASSES


def calculate_file_sha256(path: Path) -> str:
    """Tính mã băm SHA-256 của một tệp tin."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_split_file(file_path: Path) -> list[str]:
    """Đọc danh sách mã ảnh từ tệp tin split. Báo lỗi nếu tệp không tồn tại hoặc rỗng."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp split: {file_path}")
    ids = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"Tệp split không chứa bất kỳ ID nào: {file_path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Tệp split chứa ID trùng lặp: {file_path}")
    return ids


def extract_class_presence_vector(mask_path: Path, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Trích xuất vector nhị phân đại diện cho các lớp tiền cảnh xuất hiện trong mặt nạ (1..20)."""
    with Image.open(mask_path) as mask_img:
        arr = np.array(mask_img)
    valid = (arr >= 1) & (arr < num_classes) & (arr != IGNORE_INDEX)
    present_classes = np.unique(arr[valid])
    vector = np.zeros(num_classes - 1, dtype=np.int64)
    for c in present_classes:
        vector[c - 1] = 1
    return vector


def multilabel_stratified_split(
    sample_ids: list[str],
    labels_matrix: np.ndarray,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Phân tầng đa nhãn (Multilabel Stratification) theo phương pháp tham lam (Greedy / Iterative).

    Đảm bảo phân bố của 20 lớp tiền cảnh giữa train và val đạt tỷ lệ đồng đều nhất có thể.
    """
    n_samples = len(sample_ids)
    if n_samples < 2:
        raise ValueError("Cần ít nhất 2 mẫu để tạo train/validation split")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio phải nằm trong khoảng (0, 1)")
    labels_matrix = np.asarray(labels_matrix)
    if labels_matrix.ndim != 2 or labels_matrix.shape[0] != n_samples:
        raise ValueError("labels_matrix phải có dạng [số_mẫu, số_lớp] và khớp với sample_ids")

    rng = np.random.default_rng(seed)
    n_val = max(1, round(n_samples * val_ratio))
    n_train = n_samples - n_val

    # Tính mục tiêu số lượng mẫu dương cho mỗi fold
    target_proportions = np.array([n_train / n_samples, n_val / n_samples])

    # Khởi tạo danh sách kết quả
    folds: list[list[int]] = [[], []]
    fold_counts = np.zeros(2, dtype=np.int64)
    fold_label_counts = np.zeros((2, labels_matrix.shape[1]), dtype=np.float64)

    # Thứ tự xét: ưu tiên các mẫu chứa lớp hiếm trước
    label_frequencies = labels_matrix.sum(axis=0)
    sample_rarity_scores = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        present = np.where(labels_matrix[i] > 0)[0]
        if len(present) > 0:
            sample_rarity_scores[i] = np.mean(1.0 / (label_frequencies[present] + 1e-5))
        else:
            sample_rarity_scores[i] = 0.0

    # Trộn ngẫu nhiên có seed rồi sắp xếp theo độ hiếm giảm dần
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    sorted_indices = indices[np.argsort(-sample_rarity_scores[indices], kind="stable")]

    for idx in sorted_indices:
        sample_labels = labels_matrix[idx]

        # Nếu mẫu không có nhãn tiền cảnh nào (chỉ background)
        if sample_labels.sum() == 0:
            assigned_fold = 0 if (fold_counts[0] / n_train) <= (fold_counts[1] / n_val) else 1
        else:
            # Tìm fold cần mẫu này nhất dựa trên sai số so với tỷ lệ mục tiêu
            errors = []
            for f in range(2):
                if (f == 0 and fold_counts[0] >= n_train) or (f == 1 and fold_counts[1] >= n_val):
                    errors.append(float("inf"))
                    continue
                # Giả định thêm mẫu vào fold f
                temp_label_counts = fold_label_counts[f] + sample_labels
                # Sai số chuẩn hóa
                target_f = labels_matrix.sum(axis=0) * target_proportions[f]
                diff = np.abs(temp_label_counts - target_f)
                errors.append(float(diff.sum()))

            assigned_fold = int(np.argmin(errors))
            if errors[assigned_fold] == float("inf"):
                assigned_fold = 0 if fold_counts[0] < n_train else 1

        folds[assigned_fold].append(idx)
        fold_counts[assigned_fold] += 1
        fold_label_counts[assigned_fold] += sample_labels

    train_ids = sorted([sample_ids[i] for i in folds[0]])
    val_ids = sorted([sample_ids[i] for i in folds[1]])
    return train_ids, val_ids


def create_development_and_holdout_splits(
    data_root: Path,
    output_dir: Path,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, Any]:
    """Tạo split dev_train, dev_val và holdout từ official VOC2012 ImageSets.

    TUYỆT ĐỐI KHÔNG DÙNG FALLBACK 100 ID. Nếu thiếu VOC -> Báo lỗi ngay lập tức.
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seg_dir = data_root / "ImageSets" / "Segmentation"
    official_train_txt = seg_dir / "train.txt"
    official_val_txt = seg_dir / "val.txt"

    if not official_train_txt.is_file() or not official_val_txt.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy tập split chính thức của Pascal VOC tại: {seg_dir}\n"
            "Vui lòng đảm bảo thư mục data chứa 'ImageSets/Segmentation/train.txt' và 'val.txt'."
        )

    official_train_ids = read_split_file(official_train_txt)
    official_val_ids = read_split_file(official_val_txt)

    # 1. Kiểm tra sự tồn tại của ảnh và mask
    mask_dir = data_root / "SegmentationClass"
    jpeg_dir = data_root / "JPEGImages"

    for img_id in official_train_ids + official_val_ids:
        img_p = jpeg_dir / f"{img_id}.jpg"
        mask_p = mask_dir / f"{img_id}.png"
        if not img_p.is_file():
            raise FileNotFoundError(f"Thiếu file ảnh: {img_p}")
        if not mask_p.is_file():
            raise FileNotFoundError(f"Thiếu file mask: {mask_p}")

    # 2. Xây dựng ma trận nhãn đa phân lớp cho official train
    train_labels = []
    for img_id in official_train_ids:
        v = extract_class_presence_vector(mask_dir / f"{img_id}.png")
        train_labels.append(v)
    train_labels_mat = np.array(train_labels)

    # 3. Phân tầng dev_train và dev_val
    dev_train_ids, dev_val_ids = multilabel_stratified_split(
        official_train_ids,
        train_labels_mat,
        val_ratio=val_ratio,
        seed=seed,
    )
    holdout_ids = sorted(official_val_ids)

    # 4. Anti-leakage checks
    if set(official_train_ids).intersection(official_val_ids):
        raise ValueError("Official train và official val bị trùng ID")
    if set(dev_train_ids).intersection(dev_val_ids):
        raise ValueError("Rò rỉ ID giữa dev_train và dev_val")
    if set(dev_train_ids).intersection(holdout_ids):
        raise ValueError("Rò rỉ ID giữa dev_train và holdout")
    if set(dev_val_ids).intersection(holdout_ids):
        raise ValueError("Rò rỉ ID giữa dev_val và holdout")

    # 5. Lưu ra các tệp .txt
    dev_train_path = output_dir / "dev_train.txt"
    dev_val_path = output_dir / "dev_val.txt"
    holdout_path = output_dir / "holdout.txt"

    dev_train_path.write_text("\n".join(dev_train_ids) + "\n", encoding="utf-8")
    dev_val_path.write_text("\n".join(dev_val_ids) + "\n", encoding="utf-8")
    holdout_path.write_text("\n".join(holdout_ids) + "\n", encoding="utf-8")

    return {
        "source_train": {
            "path": str(official_train_txt),
            "sha256": calculate_file_sha256(official_train_txt),
            "count": len(official_train_ids),
        },
        "source_val": {
            "path": str(official_val_txt),
            "sha256": calculate_file_sha256(official_val_txt),
            "count": len(official_val_ids),
        },
        "dev_train": {
            "path": str(dev_train_path),
            "sha256": calculate_file_sha256(dev_train_path),
            "count": len(dev_train_ids),
            "ids": dev_train_ids,
        },
        "dev_val": {
            "path": str(dev_val_path),
            "sha256": calculate_file_sha256(dev_val_path),
            "count": len(dev_val_ids),
            "ids": dev_val_ids,
        },
        "holdout": {
            "path": str(holdout_path),
            "sha256": calculate_file_sha256(holdout_path),
            "count": len(holdout_ids),
            "ids": holdout_ids,
        },
    }
