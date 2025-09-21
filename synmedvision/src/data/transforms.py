"""Basic image transformation utilities used for preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image


@dataclass
class TransformConfig:
    image_size: int = 256
    normalize: bool = True


class ResizeNormalize:
    """Resize an image and optionally normalize pixel values."""

    def __init__(self, config: TransformConfig):
        self.config = config

    def __call__(self, image: Image.Image) -> np.ndarray:
        resized = image.resize((self.config.image_size, self.config.image_size), Image.BILINEAR)
        array = np.asarray(resized, dtype=np.float32)
        if self.config.normalize:
            array = array / 255.0
        return array


def compose(*funcs: Callable[[Image.Image], np.ndarray]) -> Callable[[Image.Image], np.ndarray]:
    def _inner(image: Image.Image) -> np.ndarray:
        result = image
        for func in funcs:
            result = func(result)
        return result

    return _inner


__all__ = ["TransformConfig", "ResizeNormalize", "compose"]
