"""Script chuẩn bị dữ liệu: Audit toàn vẹn, phân tầng đa nhãn và xuất Manifest có kiểm chứng SHA-256."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Đảm bảo đường dẫn import
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vocseg.config import configure_console
from vocseg.data.audit import audit_voc_dataset, generate_dataset_manifest
from vocseg.data.splits import create_development_and_holdout_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description="Audit toàn vẹn và phân tầng dataset Pascal VOC 2012")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/VOC2012_train_val/VOC2012_train_val"),
        help="Đường dẫn thư mục chứa dataset VOC2012",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/data"),
        help="Thư mục xuất manifest và các tệp split",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Tỷ lệ development validation (mặc định 15%)")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên cho phân tầng đa nhãn")
    args = parser.parse_args()

    data_root = args.data_root
    output_dir = args.output_dir
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== BẮT ĐẦU CHUẨN BỊ DỮ LIỆU & AUDIT CHỐNG RÒ RỈ ===")
    logger.info("Data Root: %s", data_root.resolve())
    logger.info("Output Dir: %s", output_dir.resolve())

    if not data_root.is_dir():
        logger.error("Không tìm thấy thư mục dataset tại: %s", data_root)
        logger.error("Pipeline thất bại (Fail-fast). Dự án này TUYỆT ĐỐI không sử dụng dummy/fake split.")
        sys.exit(1)

    try:
        # 1. Phân chia stratified dev_train / dev_val và holdout
        splits_info = create_development_and_holdout_splits(
            data_root=data_root,
            output_dir=splits_dir,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

        split_ids_map = {
            "dev_train": splits_info["dev_train"]["ids"],
            "dev_val": splits_info["dev_val"]["ids"],
            "holdout": splits_info["holdout"]["ids"],
        }

        # 2. Audit tính toàn vẹn và phân bố lớp
        audit_report = audit_voc_dataset(data_root, split_ids_map)
        audit_path = output_dir / "audit.json"
        audit_path.write_text(audit_report.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Đã lưu báo cáo audit tại: %s", audit_path)

        # 3. Tạo Manifest
        manifest = generate_dataset_manifest(data_root, splits_info, audit_report, seed=args.seed)
        manifest_path = output_dir / "dataset_manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Đã lưu dataset manifest tại: %s", manifest_path)

        logger.info("=== TỔNG KẾT DATA PIPELINE ===")
        logger.info("Dev Train: %d ảnh (SHA-256: %s...)", splits_info["dev_train"]["count"], splits_info["dev_train"]["sha256"][:12])
        logger.info("Dev Val:   %d ảnh (SHA-256: %s...)", splits_info["dev_val"]["count"], splits_info["dev_val"]["sha256"][:12])
        logger.info("Holdout:   %d ảnh (SHA-256: %s... - LOCKED)", splits_info["holdout"]["count"], splits_info["holdout"]["sha256"][:12])
        logger.info("Trạng thái Audit: %s", audit_report.audit_status)

        if audit_report.audit_status != "PASSED":
            logger.warning("Audit phát hiện bất thường trong dữ liệu! Vui lòng kiểm tra %s", audit_path)
            sys.exit(1)

    except Exception as ex:
        logger.error("Lỗi trong quá trình chuẩn bị dữ liệu: %s", ex, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
