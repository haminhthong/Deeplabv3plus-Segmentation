"""Hằng số chuẩn của Pascal VOC 2012 và quy chuẩn tiền xử lý."""

from __future__ import annotations

import numpy as np

NUM_CLASSES = 21
IGNORE_INDEX = 255
DEFAULT_IMAGE_SIZE = 320

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

VOC_CLASSES = [
    "Background (Nền)",
    "Aeroplane (Máy bay)",
    "Bicycle (Xe đạp)",
    "Bird (Chim)",
    "Boat (Thuyền)",
    "Bottle (Chai lọ)",
    "Bus (Xe buýt)",
    "Car (Ô tô)",
    "Cat (Mèo)",
    "Chair (Ghế)",
    "Cow (Bò)",
    "Diningtable (Bàn ăn)",
    "Dog (Chó)",
    "Horse (Ngựa)",
    "Motorbike (Xe máy)",
    "Person (Người)",
    "Pottedplant (Cây chậu)",
    "Sheep (Cừu)",
    "Sofa (Sofa)",
    "Train (Tàu hỏa)",
    "Tvmonitor (Tivi/Màn hình)",
]

VOC_CLASS_NAMES_SHORT = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

VOC_COLORMAP = np.array(
    [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ],
    dtype=np.uint8,
)


def mask_to_color_rgb(mask: np.ndarray, ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    """Chuyển mặt nạ nhãn hai chiều thành ảnh màu RGB."""
    if mask.ndim != 2:
        raise ValueError(f"Mặt nạ phải có 2 chiều, nhận được hình dạng {mask.shape}")

    valid = (mask >= 0) & (mask < len(VOC_COLORMAP))
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[valid] = VOC_COLORMAP[mask[valid]]
    if ignore_index is not None:
        rgb[mask == ignore_index] = [255, 255, 255]
    return rgb
