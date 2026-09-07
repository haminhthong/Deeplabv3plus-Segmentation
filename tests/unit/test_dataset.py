"""Unit tests for dataset loading and split contract validation."""

from __future__ import annotations

from pathlib import Path
import pytest
from PIL import Image

from vocseg.data.dataset import VOCSegmentationDataset
from vocseg.data.splits import read_split_file


def test_voc_dataset_missing_folders(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Thư mục VOC không hợp lệ"):
        VOCSegmentationDataset(root=tmp_path, split="dev_val")


def test_voc_dataset_missing_split(tmp_path: Path):
    (tmp_path / "JPEGImages").mkdir()
    (tmp_path / "SegmentationClass").mkdir()
    with pytest.raises(FileNotFoundError, match="Không tìm thấy split file"):
        VOCSegmentationDataset(root=tmp_path, split="non_existent")


def test_voc_dataset_load_sample(tmp_path: Path):
    jpeg_dir = tmp_path / "JPEGImages"
    mask_dir = tmp_path / "SegmentationClass"
    jpeg_dir.mkdir()
    mask_dir.mkdir()

    img = Image.new("RGB", (64, 64), color="blue")
    img.save(jpeg_dir / "sample_01.jpg")

    mask = Image.new("L", (64, 64), color=2)
    mask.save(mask_dir / "sample_01.png")

    split_file = tmp_path / "test_split.txt"
    split_file.write_text("sample_01\n", encoding="utf-8")

    ds = VOCSegmentationDataset(root=tmp_path, split_file=split_file)
    assert len(ds) == 1

    img_t, mask_t = ds[0]
    assert img_t.shape == (3, 64, 64)
    assert mask_t.shape == (64, 64)
    assert mask_t[0, 0].item() == 2
