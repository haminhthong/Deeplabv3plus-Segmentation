"""Final Fit: Huấn luyện mô hình cuối cùng trên toàn bộ tập Official Train (Dev Train + Dev Val).

QUY TRÌNH:
1. Đọc số epoch tối ưu (best_epoch) và siêu tham số đã chốt từ best.ckpt.
2. Khởi tạo mô hình mới từ trọng số ImageNet (ResNet50).
3. Huấn luyện đúng best_epoch trên toàn bộ tập dữ liệu official train.
4. Xuất artifact triển khai tinh gọn: checkpoints/final_model.pth (không lưu optimizer/scheduler).
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vocseg.config import AppConfig, configure_console
from vocseg.constants import IGNORE_INDEX, NUM_CLASSES
from vocseg.data.dataset import VOCSegmentationDataset
from vocseg.data.splits import calculate_file_sha256, read_split_file
from vocseg.data.transforms import TrainJointTransform
from vocseg.models.deeplabv3plus import build_deeplabv3plus
from vocseg.training.checkpoint import load_checkpoint, save_final_model
from vocseg.training.losses import CombinedLoss
from vocseg.training.reproducibility import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description="Final Fit trên toàn bộ dữ liệu official train của Pascal VOC 2012")
    parser.add_argument("--config", type=Path, default=Path("configs/deeplabv3plus_resnet50_320.yaml"))
    parser.add_argument(
        "--best-checkpoint", type=Path, default=Path("checkpoints/best.ckpt"), help="best.ckpt từ development phase"
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--splits-dir", type=Path, default=Path("artifacts/data/splits"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/data/dataset_manifest.json"))
    parser.add_argument("--output-path", type=Path, default=Path("checkpoints/final_model.pth"))
    parser.add_argument("--epochs", type=int, default=None, help="Ghi đè số epoch fit (nếu không lấy từ best.ckpt)")
    args = parser.parse_args()

    app_cfg = AppConfig.from_yaml(args.config)
    data_root = args.data_root or app_cfg.paths.data_root
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Đọc cấu hình và best_epoch
    if not args.best_checkpoint.is_file():
        logger.error("Không tìm thấy best checkpoint tại %s. Hãy chạy scripts/train.py trước.", args.best_checkpoint)
        sys.exit(1)
    ckpt = load_checkpoint(args.best_checkpoint, device=torch.device("cpu"))
    fit_epochs = args.epochs or int(ckpt.get("best_epoch", 0))
    if fit_epochs <= 0:
        logger.error("best.ckpt không chứa best_epoch hợp lệ. Không thể thực hiện final fit.")
        sys.exit(1)
    logger.info("Đã tìm thấy best.ckpt. Chốt số epoch huấn luyện: %d", fit_epochs)

    # 2. Gom toàn bộ danh sách official train: dev_train + dev_val (hoặc official train.txt)
    dev_train_split = args.splits_dir / "dev_train.txt"
    dev_val_split = args.splits_dir / "dev_val.txt"
    if not dev_train_split.is_file() or not dev_val_split.is_file():
        logger.error(
            "Không tìm thấy dev_train.txt hoặc dev_val.txt tại %s. Hãy chạy scripts/prepare_data.py trước.",
            args.splits_dir,
        )
        sys.exit(1)
    full_train_ids = sorted(set(read_split_file(dev_train_split) + read_split_file(dev_val_split)))

    logger.info("=== BẮT ĐẦU FINAL FIT TRÊN FULL OFFICIAL TRAIN (%d ẢNH) ===", len(full_train_ids))

    # Tạo tệp split tạm thời
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        tmp.write("\n".join(full_train_ids) + "\n")
        temp_split_path = Path(tmp.name)

    try:
        train_ds = VOCSegmentationDataset(
            root=data_root,
            split_file=temp_split_path,
            joint_transform=TrainJointTransform(app_cfg.data.image_size, app_cfg.data.image_size),
        )
        generator = set_seed(app_cfg.training.seed, app_cfg.training.deterministic)
        loader = DataLoader(
            train_ds,
            batch_size=app_cfg.training.batch_size,
            shuffle=True,
            num_workers=app_cfg.training.num_workers,
            pin_memory=(device.type == "cuda"),
            generator=generator,
        )

        # Khởi tạo mô hình mới tinh từ ImageNet
        model = build_deeplabv3plus(
            encoder="resnet50",
            encoder_weights="imagenet",
            num_classes=NUM_CLASSES,
        ).to(device)

        criterion = CombinedLoss(
            ce_weight=app_cfg.loss.cross_entropy,
            dice_weight=app_cfg.loss.dice,
            ignore_index=IGNORE_INDEX,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=app_cfg.training.lr,
            weight_decay=app_cfg.training.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=fit_epochs,
            eta_min=app_cfg.training.eta_min,
        )
        amp = app_cfg.training.amp and (device.type == "cuda")
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=amp)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=amp)

        model.train()
        for epoch in range(1, fit_epochs + 1):
            epoch_loss = 0.0
            for images, masks in loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, enabled=amp):
                    logits = model(images)
                    loss = criterion(logits, masks)

                optimizer.zero_grad(set_to_none=True)
                if amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()

            scheduler.step()
            avg_loss = epoch_loss / max(len(loader), 1)
            logger.info(
                "Final Fit Epoch %02d/%d | Loss=%.4f | LR=%.2e", epoch, fit_epochs, avg_loss, scheduler.get_last_lr()[0]
            )

        manifest_sha256 = ""
        if args.manifest.is_file():
            manifest_sha256 = calculate_file_sha256(args.manifest)

        cfg_dict = {
            "model": {"architecture": "deeplabv3plus", "encoder": "resnet50", "num_classes": NUM_CLASSES},
            "data": {"image_size": app_cfg.data.image_size, "ignore_index": IGNORE_INDEX},
            "training": {"seed": app_cfg.training.seed, "epochs": fit_epochs, "lr": app_cfg.training.lr},
        }

        # Lưu final_model.pth tinh gọn
        save_final_model(
            path=args.output_path,
            model=model,
            config_dict=cfg_dict,
            trained_epochs=fit_epochs,
            manifest_sha256=manifest_sha256,
        )
        logger.info("=== HOÀN TẤT FINAL FIT! ARTIFACT SẴN SÀNG: %s ===", args.output_path)

    finally:
        temp_split_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
