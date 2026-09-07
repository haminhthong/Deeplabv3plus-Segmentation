"""CLI script suy luận (Inference) trên ảnh bất kỳ sử dụng Predictor Canonical."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vocseg.config import configure_console
from vocseg.constants import IGNORE_INDEX, mask_to_color_rgb
from vocseg.inference.predictor import Predictor
from vocseg.inference.visualization import overlay_mask

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description="Chạy dự đoán phân đoạn ảnh trên ảnh đơn lẻ")
    parser.add_argument("--image", type=Path, required=True, help="Đường dẫn ảnh đầu vào (JPG/PNG)")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/final_model.pth"), help="Đường dẫn checkpoint mô hình")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/predictions"), help="Thư mục lưu kết quả")
    parser.add_argument("--alpha", type=float, default=0.5, help="Độ trong suốt overlay (0.0 - 1.0)")
    args = parser.parse_args()

    if not args.image.is_file():
        logger.error("Không tìm thấy ảnh tại: %s", args.image)
        sys.exit(1)
    if not args.checkpoint.is_file():
        # Thử fallback sang best.ckpt nếu final_model.pth chưa có
        if Path("checkpoints/best.ckpt").is_file():
            args.checkpoint = Path("checkpoints/best.ckpt")
        else:
            logger.error("Không tìm thấy checkpoint tại: %s", args.checkpoint)
            sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictor = Predictor(checkpoint_path=args.checkpoint)

    with Image.open(args.image) as src:
        raw_image = src.convert("RGB")

    logger.info("Đang xử lý ảnh: %s (%dx%d)", args.image.name, raw_image.width, raw_image.height)
    res = predictor.predict(raw_image)

    stem = args.image.stem
    # 1. Mask màu
    mask_rgb = mask_to_color_rgb(res.hard_mask, ignore_index=IGNORE_INDEX)
    Image.fromarray(mask_rgb).save(args.output_dir / f"{stem}_mask.png")

    # 2. Overlay
    img_rgb = np.asarray(raw_image)
    overlay = overlay_mask(img_rgb, mask_rgb, alpha=args.alpha)
    Image.fromarray(overlay).save(args.output_dir / f"{stem}_overlay.png")

    # 3. Bản đồ bất định (Normalized Entropy)
    plt.figure(figsize=(6, 6))
    plt.imshow(res.entropy_map, cmap="inferno", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Normalized Entropy Map")
    plt.savefig(args.output_dir / f"{stem}_uncertainty.png", bbox_inches="tight", dpi=120)
    plt.close()

    # 4. Metadata JSON
    meta = {
        "image": str(args.image),
        "width": res.original_size[0],
        "height": res.original_size[1],
        "classes_present": res.classes_present,
        "mean_entropy": res.mean_entropy,
        "mean_max_prob": res.mean_max_prob,
        "latency_ms": res.latency_ms,
        "model_version": res.model_version,
    }
    (args.output_dir / f"{stem}_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Hoàn tất! Thời gian: %.2f ms. Kết quả đã lưu tại: %s", res.latency_ms, args.output_dir)
    for c in res.classes_present:
        logger.info("  - %s: %.2f%% diện tích", c["name"], c["coverage"])


if __name__ == "__main__":
    main()
