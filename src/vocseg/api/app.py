"""FastAPI Serving: Production inference server với Predictor singleton."""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from vocseg.inference.predictor import MAX_IMAGE_PIXELS, Predictor
from vocseg.inference.visualization import mask_to_png_bytes

app = FastAPI(
    title="DeepLabV3+ Semantic Segmentation API",
    description="REST API cho hệ thống phân đoạn ngữ nghĩa Pascal VOC 2012",
    version="1.0.0",
)

_PREDICTOR_INSTANCE: Optional[Predictor] = None


def get_checkpoint_path() -> Path:
    env_path = os.getenv("CHECKPOINT_PATH")
    if env_path:
        return Path(env_path)
    if Path("checkpoints/final_model.pth").is_file():
        return Path("checkpoints/final_model.pth")
    if Path("checkpoints/best.ckpt").is_file():
        return Path("checkpoints/best.ckpt")
    return Path("checkpoints/final_model.pth")


def get_predictor() -> Predictor:
    """Singleton accessor cho Predictor Canonical."""
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        ckpt_path = get_checkpoint_path()
        if not ckpt_path.is_file():
            raise RuntimeError(f"Chưa tìm thấy mô hình tại: {ckpt_path}. Hãy huấn luyện hoặc cung cấp checkpoint.")
        _PREDICTOR_INSTANCE = Predictor(checkpoint_path=ckpt_path)
    return _PREDICTOR_INSTANCE


def set_predictor(predictor: Predictor) -> None:
    """Hỗ trợ tiêm (inject) predictor trong kiểm thử."""
    global _PREDICTOR_INSTANCE
    _PREDICTOR_INSTANCE = predictor


@app.get("/health")
def health_check():
    """Kiểm tra trạng thái sẵn sàng của dịch vụ."""
    try:
        pred = get_predictor()
        device_str = pred.device.type
        model_str = f"deeplabv3plus-resnet50-v{pred.checkpoint_meta.get('model_version', '1')}"
        status = "ok"
    except Exception:
        device_str = "uninitialized"
        model_str = "deeplabv3plus-resnet50-v1"
        status = "unavailable"

    return {
        "status": status,
        "model": model_str,
        "device": device_str,
    }


@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    """Phân đoạn ảnh đầu vào và trả về siêu dữ liệu kèm mặt nạ PNG base64."""
    filename = file.filename or ""
    if not file.content_type or not (
        file.content_type.startswith("image/")
        or filename.lower().endswith((".jpg", ".jpeg", ".png"))
    ):
        raise HTTPException(status_code=400, detail="Tệp tải lên phải là ảnh (JPG hoặc PNG).")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Tệp ảnh rỗng.")

    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except Exception as ex:
        raise HTTPException(status_code=400, detail=f"Không thể đọc file ảnh: {ex}")

    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=f"Kích thước ảnh vượt quá giới hạn cho phép ({image.width}x{image.height} > {MAX_IMAGE_PIXELS} pixels)",
        )

    try:
        predictor = get_predictor()
        res = predictor.predict(image)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Lỗi trong quá trình suy luận: {ex}")

    # Encode mask thành base64 PNG
    png_bytes = mask_to_png_bytes(res.hard_mask)
    b64_mask = base64.b64encode(png_bytes).decode("utf-8")

    return {
        "model_version": res.model_version,
        "width": res.original_size[0],
        "height": res.original_size[1],
        "classes_present": res.classes_present,
        "mean_entropy": round(res.mean_entropy, 4),
        "mean_max_prob": round(res.mean_max_prob, 4),
        "latency_ms": res.latency_ms,
        "mask_png_base64": f"data:image/png;base64,{b64_mask}",
    }
