"""Data module for VOC Segmentation."""

from vocseg.data.dataset import VOCSegmentationDataset
from vocseg.data.transforms import LetterboxTransform, TrainJointTransform, calculate_letterbox_geometry

__all__ = [
    "VOCSegmentationDataset",
    "LetterboxTransform",
    "TrainJointTransform",
    "calculate_letterbox_geometry",
]
