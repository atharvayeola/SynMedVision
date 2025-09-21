"""Utility helpers for optional mask based conditioning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.common.fs import ensure_dir


def prepare_mask(mask_path: str | Path, image_size: int = 256) -> np.ndarray:
    """Load a binary mask and resize to match the generation size."""

    mask = Image.open(mask_path).convert("L").resize((image_size, image_size))
    return (np.asarray(mask, dtype=np.float32) / 255.0) > 0.5


def apply_mask_to_image(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    overlay = image.copy()
    for channel, value in enumerate(color):
        overlay[..., channel] = np.where(mask, value / 255.0, overlay[..., channel])
    return overlay


def save_conditioning(mask: np.ndarray, out_dir: str | Path, name: str) -> Path:
    ensure_dir(out_dir)
    image = Image.fromarray((mask.astype(np.float32) * 255).astype(np.uint8))
    path = Path(out_dir) / f"{name}.png"
    image.save(path)
    return path


__all__ = ["prepare_mask", "apply_mask_to_image", "save_conditioning"]
