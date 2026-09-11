"""Quản lý trạng thái tái lập và Random Number Generator (RNG) State Capture."""

from __future__ import annotations

import contextlib
import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> torch.Generator:
    """Khởi tạo seed thống nhất cho Python, NumPy và PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    g = torch.Generator()
    g.manual_seed(seed)
    return g


def capture_rng_state() -> dict[str, Any]:
    """Chụp lại toàn bộ trạng thái RNG để phục vụ resume chính xác."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Khôi phục lại toàn bộ trạng thái RNG từ checkpoint."""
    if not isinstance(state, dict):
        return

    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        with contextlib.suppress(RuntimeError, TypeError, ValueError):
            torch.cuda.set_rng_state_all(state["cuda"])
