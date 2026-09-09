"""CLI script huấn luyện giai đoạn phát triển (Development Training)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vocseg.config import AppConfig, configure_console
from vocseg.data.splits import calculate_file_sha256
from vocseg.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(
        description="Huấn luyện DeepLabV3+ ResNet50 trên Pascal VOC 2012 (Development Phase)"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/deeplabv3plus_resnet50_320.yaml"), help="File cấu hình YAML"
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Ghi đè đường dẫn data root")
    parser.add_argument("--epochs", type=int, default=None, help="Ghi đè số epoch")
    parser.add_argument("--batch-size", type=int, default=None, help="Ghi đè kích thước batch")
    parser.add_argument("--lr", type=float, default=None, help="Ghi đè learning rate")
    parser.add_argument("--output-dir", type=Path, default=None, help="Thư mục ghi log và kết quả")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="Thư mục lưu checkpoints")
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("artifacts/data/splits"),
        help="Thư mục chứa dev_train.txt và dev_val.txt",
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/data/dataset_manifest.json"), help="Dataset manifest file"
    )
    parser.add_argument("--resume", type=Path, default=None, help="Tiếp tục từ checkpoint (last.ckpt)")
    args = parser.parse_args()

    # 1. Tải cấu hình
    app_cfg = AppConfig.from_yaml(args.config)
    data_root = args.data_root or app_cfg.paths.data_root
    output_dir = args.output_dir or app_cfg.paths.output_dir
    checkpoint_dir = args.checkpoint_dir or app_cfg.paths.checkpoint_dir

    dev_train_split = args.splits_dir / "dev_train.txt"
    dev_val_split = args.splits_dir / "dev_val.txt"

    if not dev_train_split.is_file() or not dev_val_split.is_file():
        logger.error(
            "Không tìm thấy tệp split dev_train hoặc dev_val tại: %s.\n"
            "Vui lòng chạy: python scripts/prepare_data.py trước.",
            args.splits_dir,
        )
        sys.exit(1)

    # Đọc hash của manifest nếu có
    manifest_sha256 = ""
    if args.manifest.is_file():
        manifest_sha256 = calculate_file_sha256(args.manifest)

    # Chuyển đổi thành từ điển cấu hình và áp dụng CLI overrides
    cfg_dict = {
        "model": {
            "architecture": app_cfg.model.architecture,
            "encoder": app_cfg.model.encoder,
            "encoder_weights": app_cfg.model.encoder_weights,
            "num_classes": app_cfg.model.num_classes,
        },
        "data": {
            "image_size": app_cfg.data.image_size,
            "ignore_index": app_cfg.data.ignore_index,
        },
        "training": {
            "epochs": args.epochs if args.epochs is not None else app_cfg.training.epochs,
            "batch_size": args.batch_size if args.batch_size is not None else app_cfg.training.batch_size,
            "lr": args.lr if args.lr is not None else app_cfg.training.lr,
            "weight_decay": app_cfg.training.weight_decay,
            "scheduler": app_cfg.training.scheduler,
            "eta_min": app_cfg.training.eta_min,
            "amp": app_cfg.training.amp,
            "seed": app_cfg.training.seed,
            "num_workers": app_cfg.training.num_workers,
            "patience": app_cfg.training.patience,
            "deterministic": app_cfg.training.deterministic,
        },
        "loss": {
            "cross_entropy": app_cfg.loss.cross_entropy,
            "dice": app_cfg.loss.dice,
        },
    }

    trainer = Trainer(
        config=cfg_dict,
        data_root=data_root,
        output_dir=output_dir,
        dev_train_split=dev_train_split,
        dev_val_split=dev_val_split,
        checkpoint_dir=checkpoint_dir,
        manifest_sha256=manifest_sha256,
        resume_checkpoint_path=args.resume,
    )

    res = trainer.fit()
    logger.info("Kết quả huấn luyện phát triển: %s", res)


if __name__ == "__main__":
    main()
