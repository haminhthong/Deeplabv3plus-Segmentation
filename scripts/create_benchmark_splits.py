"""Tạo và thẩm định (audit) benchmark splits có chủ đích cho Pascal VOC 2012.

Quy ước phân chia:
- Development Train (~85% official train)
- Validation (~15% official train) - dùng chọn model & hyperparameter
- Locked Holdout (100% official val) - chỉ mở 1 lần duy nhất cho kết quả cuối
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Tạo và thẩm định split benchmark chuẩn cho VOC 2012")
    parser.add_argument("--data-root", type=Path, default=Path("data/VOC2012_train_val/VOC2012_train_val"))
    parser.add_argument("--output-dir", type=Path, default=Path("splits/benchmark"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    if not args.data_root.is_dir():
        logger.error("Không tìm thấy dữ liệu VOC tại: %s. Pipeline thất bại (Fail-fast).", args.data_root)
        logger.error("Dự án này TUYỆT ĐỐI không dùng fallback 100 ID mẫu để tạo kết quả giả.")
        sys.exit(1)

    splits_info = create_development_and_holdout_splits(
        data_root=args.data_root,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    split_ids_map = {
        "dev_train": splits_info["dev_train"]["ids"],
        "dev_val": splits_info["dev_val"]["ids"],
        "holdout": splits_info["holdout"]["ids"],
    }
    audit_rep = audit_voc_dataset(args.data_root, split_ids_map)
    manifest = generate_dataset_manifest(args.data_root, splits_info, audit_rep, seed=args.seed)

    manifest_path = args.output_dir / "split_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    logger.info("Đã tạo benchmark split thành công tại: %s", args.output_dir)
    logger.info(
        "Dev Train: %d ảnh | Dev Val: %d ảnh | Holdout (LOCKED): %d ảnh",
        splits_info["dev_train"]["count"],
        splits_info["dev_val"]["count"],
        splits_info["holdout"]["count"],
    )
    logger.info("Audit status: %s", audit_rep.audit_status)


if __name__ == "__main__":
    main()
