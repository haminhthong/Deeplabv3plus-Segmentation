"""CLI script đánh giá tập Holdout bị khóa (Official VOC val) - Chạy DUY NHẤT 1 LẦN cho Final Report."""

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
from vocseg.evaluation.holdout import evaluate_holdout_dataset
from vocseg.evaluation.metrics import save_metrics
from vocseg.inference.predictor import Predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description="Đánh giá mô hình cuối cùng trên Locked Holdout (Official VOC Val)")
    parser.add_argument("--config", type=Path, default=Path("configs/deeplabv3plus_resnet50_320.yaml"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/final_model.pth"),
        help="Đường dẫn final_model.pth hoặc best.ckpt",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--holdout-split", type=Path, default=Path("artifacts/data/splits/holdout.txt"))
    parser.add_argument("--output-report", type=Path, default=Path("outputs/final_holdout_report.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/final_holdout_per_class.csv"))
    args = parser.parse_args()

    app_cfg = AppConfig.from_yaml(args.config)
    data_root = args.data_root or app_cfg.paths.data_root

    if not args.checkpoint.is_file():
        logger.error("Không tìm thấy checkpoint tại: %s. Hãy chắc chắn đã chạy train hoặc final_fit.", args.checkpoint)
        sys.exit(1)

    if not args.holdout_split.is_file():
        logger.error(
            "Không tìm thấy locked holdout split tại: %s. Hãy chạy scripts/prepare_data.py trước.",
            args.holdout_split,
        )
        sys.exit(1)

    logger.info("=== BẮT ĐẦU ĐÁNH GIÁ LOCKED HOLDOUT (OFFICIAL VOC VAL) ===")
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Holdout split: %s", args.holdout_split)
    logger.info("Data Root: %s", data_root)

    predictor = Predictor(checkpoint_path=args.checkpoint)
    result = evaluate_holdout_dataset(
        predictor=predictor,
        data_root=data_root,
        holdout_split_file=args.holdout_split,
    )

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    save_metrics(result, args.output_report, args.output_csv)

    logger.info("=== BÁO CÁO ĐÁNH GIÁ CHÍNH THỨC (LOCKED HOLDOUT) ===")
    logger.info("mIoU (Tất cả 21 lớp):      %.4f", result["mean_iou_all"])
    logger.info("mIoU (Tiền cảnh / no-bg):  %.4f", result["mean_iou_no_background"])
    logger.info("Mean Dice:                 %.4f", result["mean_dice_all"])
    logger.info("Pixel Accuracy:            %.4f", result["pixel_accuracy"])
    if result.get("boundary_f1_all") is not None:
        logger.info("Adaptive Boundary F1:      %.4f", result["boundary_f1_all"])

    prof = result["profiling"]
    lat = prof["latency_ms_per_image"]
    logger.info(
        "Độ trễ suy luận ảnh gốc: Mean=%.2f ms, p50=%.2f ms, p95=%.2f ms (%.1f FPS)",
        lat["mean"],
        lat["p50"],
        lat["p95"],
        prof["fps"],
    )

    if result.get("best_classes"):
        best_str = ", ".join(f"{c['class_name']} ({c['iou']:.2f})" for c in result["best_classes"][:3])
        logger.info("Top 3 lớp tốt nhất: %s", best_str)
    if result.get("worst_classes"):
        worst_str = ", ".join(f"{c['class_name']} ({c['iou']:.2f})" for c in result["worst_classes"][:3])
        logger.info("Top 3 lớp kém nhất: %s", worst_str)

    logger.info("Đã lưu báo cáo JSON: %s", args.output_report)
    logger.info("Đã lưu CSV chi tiết: %s", args.output_csv)


if __name__ == "__main__":
    main()
