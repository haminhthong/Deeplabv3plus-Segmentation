"""Dataset PyTorch cho Pascal VOC Semantic Segmentation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from vocseg.data.splits import read_split_file


class VOCSegmentationDataset(Dataset):
    """Dataset đọc ảnh và nhãn phân đoạn từ Pascal VOC theo Split Contract tường minh."""

    def __init__(
        self,
        root: Path | str,
        split: str = "dev_val",
        split_file: Optional[Path | str] = None,
        manifest_path: Optional[Path | str] = None,
        split_dir: Optional[Path | str] = None,
        split_type: str = "benchmark",
        joint_transform: Optional[Callable[[Image.Image, Image.Image], Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.joint_transform = joint_transform

        self.jpeg_dir = self.root / "JPEGImages"
        self.mask_dir = self.root / "SegmentationClass"

        if not self.jpeg_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(
                f"Thư mục VOC không hợp lệ tại {self.root}. Yêu cầu JPEGImages/ và SegmentationClass/."
            )

        # Xác định danh sách ID theo quy tắc tường minh
        if split_file is not None:
            self.ids = read_split_file(Path(split_file))
        elif split_dir is not None:
            split_path = Path(split_dir) / f"{split}.txt"
            self.ids = read_split_file(split_path)
        elif manifest_path is not None:
            manifest_file = Path(manifest_path)
            if not manifest_file.is_file():
                raise FileNotFoundError(f"Không tìm thấy manifest tại: {manifest_file}")
            json.loads(manifest_file.read_text(encoding="utf-8"))
            # Tìm trong manifest.splits hoặc file tham chiếu
            splits_dir = manifest_file.parent / "splits"
            target_txt = splits_dir / f"{split}.txt"
            if target_txt.is_file():
                self.ids = read_split_file(target_txt)
            else:
                raise FileNotFoundError(f"Không tìm thấy split '{split}' trong manifest splits tại: {target_txt}")
        else:
            # Tìm trực tiếp tại root/splits/ hoặc artifacts/data/splits/
            default_paths = [
                Path("artifacts/data/splits") / f"{split}.txt",
                Path("splits") / split_type / f"{split}.txt",
                Path("splits") / f"{split}.txt",
                self.root / "ImageSets" / "Segmentation" / f"{split}.txt",
            ]
            found = None
            for p in default_paths:
                if p.is_file():
                    found = p
                    break
            if found is None:
                raise FileNotFoundError(
                    f"Không tìm thấy split file cho split='{split}'. Vui lòng chỉ định 'split_file' hoặc 'manifest_path'."
                )
            self.ids = read_split_file(found)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_id = self.ids[idx]
        img_path = self.jpeg_dir / f"{img_id}.jpg"
        mask_path = self.mask_dir / f"{img_id}.png"

        with Image.open(img_path) as src:
            image = src.convert("RGB")
        with Image.open(mask_path) as src:
            mask = src.copy()

        if self.joint_transform is not None:
            image_t, mask_t = self.joint_transform(image, mask)
        else:
            image_t = transforms.ToTensor()(image)
            mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))

        return image_t, mask_t
