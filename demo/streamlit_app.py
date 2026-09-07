"""Streamlit Interactive Demonstration UI: Thuần túy tải ảnh và trình diễn kết quả phân đoạn."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from vocseg.constants import IGNORE_INDEX, NUM_CLASSES, VOC_CLASSES, mask_to_color_rgb
from vocseg.inference.predictor import MAX_IMAGE_PIXELS, Predictor
from vocseg.inference.visualization import overlay_mask


@st.cache_resource(show_spinner=False)
def load_predictor_cached(checkpoint_path_str: str) -> Predictor:
    """Tải và lưu trữ Predictor singleton trong bộ nhớ đệm của Streamlit."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return Predictor(checkpoint_path=checkpoint_path_str, device=device)


def main():
    st.set_page_config(
        page_title="DeepLabV3+ ResNet50 Semantic Segmentation Demo",
        layout="wide",
        page_icon="🔍",
    )
    st.title("DeepLabV3+ ResNet50 Semantic Segmentation System")
    st.caption("Pascal VOC 2012 — Original-Resolution Inference & Reliability Analysis")

    # Mặc định tìm model
    default_ckpts = [
        "checkpoints/final_model.pth",
        "checkpoints/best.ckpt",
        "outputs/deeplabv3plus_resnet50_voc_best.pth",
    ]
    selected_ckpt = None
    for c in default_ckpts:
        if Path(c).is_file():
            selected_ckpt = c
            break

    if selected_ckpt is None:
        st.warning(
            "Chưa phát hiện checkpoint nào trong thư mục `checkpoints/`. "
            "Vui lòng chạy `python scripts/train.py` hoặc tải checkpoint trước khi sử dụng giao diện."
        )
        st.info("Ví dụ: `python scripts/final_fit.py` hoặc `python scripts/train.py`")

    with st.sidebar:
        st.header("Thông tin hệ thống")
        device_str = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
        st.metric(label="Thiết bị tính toán", value=device_str)
        if selected_ckpt:
            st.caption(f"Model: `{selected_ckpt}`")

        st.divider()
        alpha = st.slider("Độ trong suốt của Overlay", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        st.info("Chế độ: Interactive Image Demo (Tập Holdout được bảo vệ nghiêm ngặt và không thể duyệt qua UI)")

    uploaded = st.file_uploader(
        "Tải ảnh của bạn lên để phân đoạn (Hỗ trợ JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded is None:
        st.info("👉 Hãy tải lên một bức ảnh để bắt đầu phân tích và phân đoạn.")
        return

    try:
        with Image.open(uploaded) as src:
            image = src.convert("RGB")
    except Exception:
        st.error("Không thể đọc tệp ảnh được tải lên. Định dạng không hợp lệ.")
        return

    if image.width * image.height > MAX_IMAGE_PIXELS:
        st.error(f"Ảnh quá lớn ({image.width}x{image.height}). Vui lòng tải ảnh dưới {MAX_IMAGE_PIXELS // 1_000_000} MP.")
        return

    if selected_ckpt is None:
        st.error("Không tìm thấy checkpoint để thực thi.")
        return

    try:
        with st.spinner("Đang thực hiện suy luận tại độ phân giải gốc..."):
            predictor = load_predictor_cached(selected_ckpt)
            res = predictor.predict(image)
    except Exception as ex:
        st.error(f"Lỗi suy luận: {ex}")
        return

    raw_rgb = np.asarray(image)
    mask_rgb = mask_to_color_rgb(res.hard_mask, ignore_index=IGNORE_INDEX)
    overlay_rgb = overlay_mask(raw_rgb, mask_rgb, alpha=alpha)

    st.markdown("### 1. Kết quả Phân đoạn Ngữ nghĩa")
    col1, col2, col3 = st.columns(3)
    col1.image(raw_rgb, caption=f"Ảnh gốc ({res.original_size[0]}x{res.original_size[1]})", use_container_width=True)
    col2.image(mask_rgb, caption="Mặt nạ phân đoạn (Class Mask)", use_container_width=True)
    col3.image(overlay_rgb, caption="Phủ màu (Overlay)", use_container_width=True)

    st.markdown("### 2. Phân tích Độ Bất định & Độ Tin cậy (Uncertainty Heuristics)")
    st.caption(
        "Normalized Entropy Map thể hiện mức độ phân vân của phân bố xác suất Softmax tại từng pixel. "
        "Vùng càng sáng thể hiện ranh giới hoặc đối tượng mà mô hình ít chắc chắn nhất (Reliability heuristic)."
    )
    u1, u2 = st.columns(2)
    u1.image(
        res.entropy_map,
        caption=f"Normalized Entropy (Trung bình: {res.mean_entropy:.3f})",
        clamp=True,
        use_container_width=True,
    )
    u2.image(
        res.max_prob_map,
        caption=f"Max Softmax Probability (Trung bình: {res.mean_max_prob:.3f})",
        clamp=True,
        use_container_width=True,
    )

    st.markdown("### 3. Thống kê các lớp đối tượng xuất hiện")
    if not res.classes_present:
        st.warning("Không phát hiện lớp đối tượng tiền cảnh nào (toàn bộ ảnh là Background).")
    else:
        st.dataframe(
            res.classes_present,
            use_container_width=True,
            column_config={
                "id": st.column_config.NumberColumn("ID"),
                "name": st.column_config.TextColumn("Lớp ngữ nghĩa"),
                "pixels": st.column_config.NumberColumn("Số lượng pixel", format="%d"),
                "coverage": st.column_config.NumberColumn("Tỷ lệ diện tích (%)", format="%.2f%%"),
            },
            hide_index=True,
        )

    st.success(f"Hoàn tất trong {res.latency_ms:.1f} ms | Model version: {res.model_version}")


if __name__ == "__main__":
    main()
