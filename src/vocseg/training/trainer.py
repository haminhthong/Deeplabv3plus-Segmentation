"""Trainer cho Development Phase (huấn luyện trên dev_train, kiểm định trên dev_val)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from vocseg.constants import IGNORE_INDEX, NUM_CLASSES
from vocseg.data.dataset import VOCSegmentationDataset
from vocseg.data.transforms import LetterboxTransform, TrainJointTransform
from vocseg.evaluation.development import evaluate_development
from vocseg.evaluation.metrics import save_metrics
from vocseg.models.deeplabv3plus import build_deeplabv3plus
from vocseg.training.checkpoint import (
    load_checkpoint,
    save_best_checkpoint,
    save_resume_checkpoint,
)
from vocseg.training.losses import CombinedLoss
from vocseg.training.reproducibility import restore_rng_state, set_seed

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer quản lý vòng đời huấn luyện phát triển (Development Training Lifecycle)."""

    def __init__(
        self,
        config: Dict[str, Any],
        data_root: Path | str,
        output_dir: Path | str,
        dev_train_split: Path | str,
        dev_val_split: Path | str,
        checkpoint_dir: Path | str = Path("checkpoints"),
        manifest_sha256: str = "",
        resume_checkpoint_path: Optional[Path | str] = None,
    ) -> None:
        self.config = config
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.dev_train_split = Path(dev_train_split)
        self.dev_val_split = Path(dev_val_split)
        self.manifest_sha256 = manifest_sha256

        train_cfg = config.get("training", {})
        self.seed = train_cfg.get("seed", 42)
        self.deterministic = train_cfg.get("deterministic", False)
        self.generator = set_seed(self.seed, self.deterministic)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = int(train_cfg.get("epochs", 50))
        self.batch_size = int(train_cfg.get("batch_size", 8))
        self.lr = float(train_cfg.get("lr", 1e-4))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        self.num_workers = int(train_cfg.get("num_workers", 2))
        self.patience = int(train_cfg.get("patience", 0))
        self.amp = bool(train_cfg.get("amp", True)) and (self.device.type == "cuda")

        data_cfg = config.get("data", {})
        self.image_size = int(data_cfg.get("image_size", 320))

        # Khởi tạo mô hình
        encoder_weights = config.get("model", {}).get("encoder_weights")
        self.model = build_deeplabv3plus(
            encoder="resnet50",
            encoder_weights=encoder_weights if resume_checkpoint_path is None else None,
            num_classes=NUM_CLASSES,
        ).to(self.device)

        # Loss & Optimizer
        loss_cfg = config.get("loss", {})
        self.criterion = CombinedLoss(
            ce_weight=float(loss_cfg.get("cross_entropy", 1.0)),
            dice_weight=float(loss_cfg.get("dice", 0.5)),
            ignore_index=IGNORE_INDEX,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=float(train_cfg.get("eta_min", 1e-6)),
        )
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        self.start_epoch = 1
        self.best_miou = -1.0
        self.best_epoch = -1
        self.stale_epochs = 0
        self.history_path = self.output_dir / "train_log.csv"
        header = "epoch,train_loss,val_loss,val_miou_all,val_miou_fg,val_dice,pixel_acc"
        if resume_checkpoint_path is not None and self.history_path.is_file():
            self.history_lines = self.history_path.read_text(encoding="utf-8").splitlines() or [header]
        else:
            self.history_lines = [header]

        # Xử lý resume
        if resume_checkpoint_path is not None:
            self._resume_from_checkpoint(resume_checkpoint_path)

        # Dataloaders
        self.train_loader, self.val_loader = self._create_dataloaders()

    def _create_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        train_ds = VOCSegmentationDataset(
            root=self.data_root,
            split_file=self.dev_train_split,
            joint_transform=TrainJointTransform(self.image_size, self.image_size),
        )
        val_ds = VOCSegmentationDataset(
            root=self.data_root,
            split_file=self.dev_val_split,
            joint_transform=LetterboxTransform(self.image_size, self.image_size),
        )
        # Tránh lỗi BatchNorm2d khi batch cuối cùng chỉ có đúng 1 sample trong training mode
        drop_last = (len(train_ds) > self.batch_size) and (len(train_ds) % self.batch_size == 1)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            generator=self.generator,
            drop_last=drop_last,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )
        return train_loader, val_loader

    def _resume_from_checkpoint(self, path: Path | str) -> None:
        ckpt = load_checkpoint(path, self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"]:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"]:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if self.amp and ckpt.get("scaler_state_dict"):
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "rng_state" in ckpt:
            restore_rng_state(ckpt["rng_state"])

        self.start_epoch = int(ckpt.get("epoch", 0)) + 1
        self.best_miou = float(ckpt.get("best_metric", -1.0))
        self.best_epoch = int(ckpt.get("best_epoch", -1))
        if self.start_epoch > self.epochs:
            raise ValueError(f"Checkpoint đã ở epoch {self.start_epoch - 1}, không thể resume với epochs={self.epochs}")
        logger.info(
            "Đã phục hồi hoàn toàn trạng thái huấn luyện từ %s (Tiếp tục từ epoch %d, best mIoU=%.4f)",
            path,
            self.start_epoch,
            self.best_miou,
        )

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for images, masks in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            self.optimizer.zero_grad(set_to_none=True)
            if self.amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(len(self.train_loader), 1)

    def fit(self) -> Dict[str, Any]:
        logger.info(
            "Bắt đầu huấn luyện DeepLabV3+ ResNet50 (%d epochs, batch_size=%d, lr=%.2e) trên %s",
            self.epochs,
            self.batch_size,
            self.lr,
            self.device,
        )

        for epoch in range(self.start_epoch, self.epochs + 1):
            train_loss = self.train_epoch()
            val_loss, val_metrics = evaluate_development(
                self.model,
                self.val_loader,
                self.criterion,
                self.device,
                NUM_CLASSES,
                IGNORE_INDEX,
            )
            self.scheduler.step()

            val_miou_all = val_metrics["mean_iou_all"]
            val_miou_fg = val_metrics["mean_iou_no_background"]
            val_dice = val_metrics["mean_dice_all"]
            pixel_acc = val_metrics["pixel_accuracy"]

            logger.info(
                "Epoch %02d/%d | LR=%.2e | train_loss=%.4f | val_loss=%.4f | val_mIoU=%.4f (fg=%.4f, Dice=%.4f)",
                epoch,
                self.epochs,
                self.scheduler.get_last_lr()[0],
                train_loss,
                val_loss,
                val_miou_all,
                val_miou_fg,
                val_dice,
            )

            self.history_lines.append(
                f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_miou_all:.6f},{val_miou_fg:.6f},{val_dice:.6f},{pixel_acc:.6f}"
            )
            # Kiểm tra lưu best.ckpt theo primary metric (val_miou_all).
            if val_miou_all > self.best_miou:
                self.best_miou = val_miou_all
                self.best_epoch = epoch
                self.stale_epochs = 0
                save_best_checkpoint(
                    path=self.checkpoint_dir / "best.ckpt",
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    best_miou=val_miou_all,
                    config_dict=self.config,
                    val_metrics=val_metrics,
                    manifest_sha256=self.manifest_sha256,
                )
                save_metrics(
                    val_metrics,
                    self.output_dir / "best_dev_metrics.json",
                    self.output_dir / "best_dev_per_class.csv",
                )
                logger.info(
                    "--> Đạt kỷ lục mới! Đã cập nhật %s (mIoU=%.4f)", self.checkpoint_dir / "best.ckpt", val_miou_all
                )
            else:
                self.stale_epochs += 1

            self.history_path.write_text("\n".join(self.history_lines) + "\n", encoding="utf-8")

            # Lưu last.ckpt sau khi cập nhật best metric để resume đúng trạng thái.
            save_resume_checkpoint(
                path=self.checkpoint_dir / "last.ckpt",
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                best_metric=self.best_miou,
                config_dict=self.config,
                manifest_sha256=self.manifest_sha256,
                best_epoch=self.best_epoch,
            )

            if self.patience > 0 and self.stale_epochs >= self.patience:
                logger.info("Dừng sớm (early stopping) sau %d epoch không cải thiện.", self.stale_epochs)
                break

        logger.info("Huấn luyện phát triển hoàn tất. Best epoch: %d (val mIoU: %.4f)", self.best_epoch, self.best_miou)
        return {
            "best_epoch": self.best_epoch,
            "best_val_miou": self.best_miou,
            "best_checkpoint": str(self.checkpoint_dir / "best.ckpt"),
            "last_checkpoint": str(self.checkpoint_dir / "last.ckpt"),
        }
