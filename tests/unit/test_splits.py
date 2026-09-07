"""Unit tests for split contract and multilabel stratification."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from vocseg.data.splits import (
    calculate_file_sha256,
    create_development_and_holdout_splits,
    multilabel_stratified_split,
    read_split_file,
)


def test_multilabel_stratified_split():
    # 100 mẫu với 20 nhãn ngẫu nhiên
    rng = np.random.default_rng(42)
    sample_ids = [f"sample_{i:03d}" for i in range(100)]
    labels_matrix = rng.binomial(1, 0.2, size=(100, 20))

    train_ids, val_ids = multilabel_stratified_split(
        sample_ids, labels_matrix, val_ratio=0.15, seed=42
    )

    assert len(train_ids) + len(val_ids) == 100
    assert len(val_ids) == 15
    assert len(train_ids) == 85
    # Kiểm tra rò rỉ (Zero ID overlap)
    assert len(set(train_ids).intersection(set(val_ids))) == 0


def test_missing_voc_fails_fast(tmp_path: Path):
    # Thư mục rỗng không có official ImageSets
    with pytest.raises(FileNotFoundError, match="Không tìm thấy tập split chính thức"):
        create_development_and_holdout_splits(
            data_root=tmp_path / "non_existent_voc",
            output_dir=tmp_path / "output",
        )


def test_read_split_file_empty(tmp_path: Path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="không chứa bất kỳ ID nào"):
        read_split_file(empty_file)


def test_read_split_file_duplicates(tmp_path: Path):
    dup_file = tmp_path / "dup.txt"
    dup_file.write_text("id_1\nid_2\nid_1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="trùng lặp"):
        read_split_file(dup_file)


def test_calculate_file_sha256(tmp_path: Path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello voc", encoding="utf-8")
    h = calculate_file_sha256(test_file)
    assert isinstance(h, str)
    assert len(h) == 64
