"""Loss functions cho Semantic Segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss

from vocseg.constants import IGNORE_INDEX


class CombinedLoss(nn.Module):
    """Hàm mất mát kết hợp chuẩn hóa: Cross-Entropy (1.0) + Dice Loss (0.5)."""

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 0.5,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(mode="multiclass", ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, masks)
        loss_dice = self.dice(logits, masks)
        return self.ce_weight * loss_ce + self.dice_weight * loss_dice
