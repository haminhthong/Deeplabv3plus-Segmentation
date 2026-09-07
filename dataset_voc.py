"""Tương thích ngược (Backward Compatibility) cho dataset_voc."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from vocseg.constants import DEFAULT_IMAGE_SIZE, IGNORE_INDEX, IMAGE_MEAN, IMAGE_STD
from vocseg.data.dataset import VOCSegmentationDataset
from vocseg.data.splits import read_split_file
from vocseg.data.transforms import (
    LetterboxTransform,
    TrainJointTransform,
    calculate_letterbox_geometry,
    resize_and_pad,
)

# Alias tương thích ngược
JointTransform = LetterboxTransform


def get_val_transforms(h: int = DEFAULT_IMAGE_SIZE, w: int = DEFAULT_IMAGE_SIZE):
    return LetterboxTransform(h=h, w=w)


def get_train_transforms(h: int = DEFAULT_IMAGE_SIZE, w: int = DEFAULT_IMAGE_SIZE):
    return TrainJointTransform(h=h, w=w)


def read_split_ids(
    root: Path | str,
    split: str,
    split_dir: Optional[Path | str] = None,
    split_type: str = "benchmark",
) -> List[str]:
    """Đọc split IDs tường minh không fallback ngầm mâu thuẫn."""
    root = Path(root)
    candidates = []
    if split_dir is not None:
        candidates.append(Path(split_dir) / f"{split}.txt")
    candidates.extend([
        Path("artifacts/data/splits") / f"{split}.txt",
        Path("splits") / split_type / f"{split}.txt",
        Path("splits") / f"{split}.txt",
        root / "splits" / split_type / f"{split}.txt",
        root / "ImageSets" / "Segmentation" / f"{split}.txt",
    ])

    for cand in candidates:
        if cand.is_file():
            return read_split_file(cand)

    raise FileNotFoundError(
        f"Không tìm thấy tệp chia dữ liệu cho split='{split}' tại {candidates[:3]}"
    )


def validate_voc_dataset(
    root: Path | str,
    split_dir: Optional[Path | str] = None,
    split_type: str = "benchmark",
) -> None:
    """Kiểm tra sự tồn tại của dữ liệu và rò rỉ split."""
    root = Path(root)
    splits_to_check = []
    for s in ["train", "val", "test", "dev_train", "dev_val", "holdout"]:
        try:
            ids = read_split_ids(root, s, split_dir=split_dir, split_type=split_type)
            splits_to_check.append((s, ids))
        except FileNotFoundError:
            pass

    if not splits_to_check:
        raise FileNotFoundError("Không tìm thấy tệp chia dữ liệu nào để kiểm tra")

    # Kiểm tra rò rỉ ID
    for i in range(len(splits_to_check)):
        for j in range(i + 1, len(splits_to_check)):
            s1_name, s1_ids = splits_to_check[i]
            s2_name, s2_ids = splits_to_check[j]
            # Bỏ qua cặp so sánh dev_train vs train nếu dev_train là tập con của train
            if "dev_" in s1_name and s2_name in ("train", "val"):
                continue
            if "dev_" in s2_name and s1_name in ("train", "val"):
                continue
            overlap = set(s1_ids).intersection(s2_ids)
            if overlap:
                raise ValueError(f"Split {s1_name} và {s2_name} bị trùng {len(overlap)} ảnh")

    # Kiểm tra tồn tại file ảnh và mask
    missing = []
    all_ids = set()
    for _, ids in splits_to_check:
        all_ids.update(ids)

    for image_id in all_ids:
        for p in (root / "JPEGImages" / f"{image_id}.jpg", root / "SegmentationClass" / f"{image_id}.png"):
            if not p.is_file():
                missing.append(p)
    if missing:
        preview = "\n".join(str(p) for p in missing[:10])
        raise FileNotFoundError(f"Thiếu {len(missing)} file dữ liệu, ví dụ:\n{preview}")


__all__ = [
    "calculate_letterbox_geometry",
    "resize_and_pad",
    "LetterboxTransform",
    "JointTransform",
    "TrainJointTransform",
    "VOCSegmentationDataset",
    "get_val_transforms",
    "get_train_transforms",
    "read_split_ids",
    "validate_voc_dataset",
]
