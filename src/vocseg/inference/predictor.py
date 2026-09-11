"""Predictor Canonical: Đơn vị suy luận chuẩn mực đảm bảo tính nhất quán (Parity) 100% giữa Eval và Serving."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from torchvision import transforms

from vocseg.constants import (
    DEFAULT_IMAGE_SIZE,
    IMAGE_MEAN,
    IMAGE_STD,
    NUM_CLASSES,
    VOC_CLASSES,
)
from vocseg.data.transforms import calculate_letterbox_geometry
from vocseg.models.deeplabv3plus import build_deeplabv3plus, validate_checkpoint_metadata
from vocseg.training.checkpoint import load_checkpoint

MAX_IMAGE_PIXELS = 25_000_000  # Giới hạn 25 MP tránh OOM


@dataclass
class PredictionResult:
    hard_mask: np.ndarray  # [H_orig, W_orig] int64
    max_prob_map: np.ndarray  # [H_orig, W_orig] float32
    entropy_map: np.ndarray  # [H_orig, W_orig] float32 (Normalized Entropy)
    original_size: tuple[int, int]  # (W, H)
    classes_present: list[dict[str, Any]]
    mean_entropy: float
    mean_max_prob: float
    latency_ms: float
    model_version: str = "1.0.0"


class Predictor:
    """Canonical DeepLabV3+ Predictor tại độ phân giải gốc của ảnh."""

    def __init__(
        self,
        checkpoint_path: Path | str,
        device: torch.device | None = None,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.checkpoint_meta: dict[str, Any] = {}
        self.model = self._load_model_from_checkpoint(checkpoint_path)

    def _load_model_from_checkpoint(self, checkpoint_path: Path | str) -> torch.nn.Module:
        ckpt = load_checkpoint(checkpoint_path, self.device)
        self.checkpoint_meta = ckpt

        # Kiểm tra contract siêu dữ liệu
        validate_checkpoint_metadata(ckpt)

        state_dict = ckpt.get("model_state_dict", ckpt)
        model = build_deeplabv3plus(
            encoder="resnet50",
            encoder_weights=None,
            num_classes=NUM_CLASSES,
        )
        model.load_state_dict(state_dict)
        model.to(self.device).eval()
        return model

    def validate_image(self, image: Image.Image) -> Image.Image:
        """Xác thực định dạng ảnh, xử lý EXIF xoay chiều và kiểm soát kích thước tối đa."""
        if not isinstance(image, Image.Image):
            raise TypeError(f"Kỳ vọng PIL Image, nhận được: {type(image)}")

        # Xoay theo EXIF tags
        image = ImageOps.exif_transpose(image) or image
        image = image.convert("RGB")

        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError(
                f"Kích thước ảnh quá lớn ({image.width}x{image.height} = {image.width * image.height} pixels). "
                f"Giới hạn tối đa là {MAX_IMAGE_PIXELS} pixels."
            )
        return image

    def preprocess(self, image: Image.Image) -> tuple[torch.Tensor, tuple[int, int, int, int], tuple[int, int]]:
        """Letterbox và chuẩn hóa ảnh về Tensor kích thước model (320x320)."""
        w_orig, h_orig = image.size
        _, resized_w, resized_h, pad_left, pad_top, pad_right, pad_bottom = calculate_letterbox_geometry(
            w_orig, h_orig, self.image_size, self.image_size
        )

        resized = TF.resize(image, (resized_h, resized_w), interpolation=transforms.InterpolationMode.BILINEAR)
        padded = TF.pad(resized, [pad_left, pad_top, pad_right, pad_bottom], fill=0)
        tensor = TF.normalize(TF.to_tensor(padded), IMAGE_MEAN, IMAGE_STD)
        pad_coords = (pad_left, pad_top, resized_w, resized_h)
        return tensor, pad_coords, (w_orig, h_orig)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass model lấy raw logits [1, 21, 320, 320]."""
        with torch.inference_mode():
            logits = self.model(tensor.unsqueeze(0).to(self.device))
        return logits

    def postprocess(
        self,
        logits: torch.Tensor,
        pad_coords: tuple[int, int, int, int],
        original_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cắt bỏ letterbox padding, nội suy logits về H_orig x W_orig và tính Softmax/Argmax."""
        pad_left, pad_top, resized_w, resized_h = pad_coords
        w_orig, h_orig = original_size

        # 1. Cắt bỏ padding của letterbox khỏi logits
        cropped_logits = logits[:, :, pad_top : pad_top + resized_h, pad_left : pad_left + resized_w]

        # 2. Resize song tuyến logits về kích thước ban đầu (h_orig, w_orig)
        rescaled_logits = F.interpolate(cropped_logits, size=(h_orig, w_orig), mode="bilinear", align_corners=False)

        # 3. Softmax xác suất
        probs = F.softmax(rescaled_logits, dim=1).squeeze(0)  # [C, H_orig, W_orig]

        # 4. Hard mask
        hard_mask = probs.argmax(dim=0).cpu().numpy().astype(np.int64)

        # 5. Reliability / Uncertainty heuristics
        max_prob_map = probs.max(dim=0).values.cpu().numpy().astype(np.float32)

        eps = 1e-7
        entropy = -(probs * torch.log(probs + eps)).sum(dim=0)
        norm_factor = float(np.log(max(probs.shape[0], 2)))
        normalized_entropy = (entropy / norm_factor).clamp(0.0, 1.0).cpu().numpy().astype(np.float32)

        return hard_mask, max_prob_map, normalized_entropy

    def predict(self, image: Image.Image) -> PredictionResult:
        """Thực hiện toàn bộ quy trình suy luận canonical cho 1 ảnh."""
        t_start = time.perf_counter()

        image = self.validate_image(image)
        tensor, pad_coords, orig_size = self.preprocess(image)
        logits = self.forward(tensor)
        hard_mask, max_prob, entropy = self.postprocess(logits, pad_coords, orig_size)

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Thống kê phân bố lớp xuất hiện
        flat_mask = hard_mask.reshape(-1)
        total_pixels = int(flat_mask.size)
        counts = np.bincount(flat_mask, minlength=NUM_CLASSES)

        classes_present: list[dict[str, Any]] = []
        for c_id in range(1, NUM_CLASSES):  # Bỏ background khỏi bảng thống kê đối tượng
            px = int(counts[c_id])
            if px > 0:
                cov = float((px / total_pixels) * 100.0)
                classes_present.append(
                    {
                        "id": c_id,
                        "name": VOC_CLASSES[c_id],
                        "pixels": px,
                        "coverage": round(cov, 2),
                    }
                )
        classes_present.sort(key=lambda x: x["pixels"], reverse=True)

        return PredictionResult(
            hard_mask=hard_mask,
            max_prob_map=max_prob,
            entropy_map=entropy,
            original_size=orig_size,
            classes_present=classes_present,
            mean_entropy=float(np.mean(entropy)),
            mean_max_prob=float(np.mean(max_prob)),
            latency_ms=round(t_elapsed_ms, 2),
            model_version=self.checkpoint_meta.get("model_version", "1.0.0"),
        )
