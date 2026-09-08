"""Integration test: Kiểm thử toàn bộ vòng đời ML end-to-end trên mini synthetic VOC."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from vocseg.data.audit import audit_voc_dataset, generate_dataset_manifest
from vocseg.data.splits import create_development_and_holdout_splits
from vocseg.evaluation.holdout import evaluate_holdout_dataset
from vocseg.inference.predictor import Predictor
from vocseg.training.checkpoint import save_final_model
from vocseg.training.trainer import Trainer


@pytest.fixture
def synthetic_voc(tmp_path: Path) -> Path:
    voc_root = tmp_path / "VOC2012"
    jpeg_dir = voc_root / "JPEGImages"
    mask_dir = voc_root / "SegmentationClass"
    seg_dir = voc_root / "ImageSets" / "Segmentation"

    jpeg_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    seg_dir.mkdir(parents=True)

    # Tạo 8 mẫu ảnh: 6 cho official train, 2 cho official val
    sample_ids = [f"2007_{i:06d}" for i in range(1, 9)]
    for i, sid in enumerate(sample_ids):
        img = Image.new("RGB", (120, 100), color=(i * 25, 100, 150))
        img.save(jpeg_dir / f"{sid}.jpg")

        mask_arr = np.zeros((100, 120), dtype=np.uint8)
        mask_arr[20:60, 30:80] = (i % 3) + 1  # Lớp 1, 2 hoặc 3
        mask_img = Image.fromarray(mask_arr)
        mask_img.save(mask_dir / f"{sid}.png")

    train_ids = sample_ids[:6]
    val_ids = sample_ids[6:]

    (seg_dir / "train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (seg_dir / "val.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")

    return voc_root


def test_full_ml_lifecycle_synthetic_voc(synthetic_voc: Path, tmp_path: Path):
    out_dir = tmp_path / "artifacts"
    splits_dir = out_dir / "splits"
    ckpt_dir = tmp_path / "checkpoints"

    # Bước 1: Data Ingestion, Stratification & Audit
    splits_info = create_development_and_holdout_splits(
        data_root=synthetic_voc,
        output_dir=splits_dir,
        val_ratio=0.33,  # 2 dev_val, 4 dev_train
        seed=42,
    )

    split_ids_map = {
        "dev_train": splits_info["dev_train"]["ids"],
        "dev_val": splits_info["dev_val"]["ids"],
        "holdout": splits_info["holdout"]["ids"],
    }
    audit_rep = audit_voc_dataset(synthetic_voc, split_ids_map)
    assert audit_rep.audit_status == "PASSED"

    manifest = generate_dataset_manifest(synthetic_voc, splits_info, audit_rep)
    assert manifest.dev_train_sha256 != ""
    assert manifest.holdout_sha256 != ""

    # Bước 2: Development Training (1 epoch)
    cfg = {
        "model": {"architecture": "deeplabv3plus", "encoder": "resnet50", "num_classes": 21},
        "data": {"image_size": 128, "ignore_index": 255},
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "scheduler": "cosine",
            "eta_min": 1e-6,
            "amp": False,
            "seed": 42,
            "num_workers": 0,
        },
        "loss": {"cross_entropy": 1.0, "dice": 0.5},
    }

    trainer = Trainer(
        config=cfg,
        data_root=synthetic_voc,
        output_dir=tmp_path / "outputs",
        dev_train_split=splits_dir / "dev_train.txt",
        dev_val_split=splits_dir / "dev_val.txt",
        checkpoint_dir=ckpt_dir,
    )
    trainer.fit()

    assert (ckpt_dir / "best.ckpt").is_file()
    assert (ckpt_dir / "last.ckpt").is_file()

    # Bước 3: Lưu Final Model
    final_model_path = ckpt_dir / "final_model.pth"
    save_final_model(
        path=final_model_path,
        model=trainer.model,
        config_dict=cfg,
        trained_epochs=1,
    )
    assert final_model_path.is_file()

    # Bước 4: Locked Holdout Evaluation (đánh giá trên official val)
    predictor = Predictor(checkpoint_path=final_model_path, image_size=128)
    holdout_res = evaluate_holdout_dataset(
        predictor=predictor,
        data_root=synthetic_voc,
        holdout_split_file=splits_dir / "holdout.txt",
    )

    assert "mean_iou_all" in holdout_res
    assert holdout_res["profiling"]["images_evaluated"] == len(splits_info["holdout"]["ids"])

    # Bước 5: Online Predictor Parity
    test_img = Image.new("RGB", (150, 80), color="blue")
    pred = predictor.predict(test_img)
    assert pred.hard_mask.shape == (80, 150)
    assert pred.latency_ms > 0
