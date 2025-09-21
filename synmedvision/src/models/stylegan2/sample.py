"""Sampling utilities for the dummy StyleGAN2 model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from src.common.fs import ensure_dir
from src.common.seed import set_global_seed
from src.models.stylegan2.train import DummyStyleGAN2, _load_checkpoint


def load_generator(ckpt_path: str | Path) -> DummyStyleGAN2:
    return _load_checkpoint(Path(ckpt_path))


def _upsample(feature: np.ndarray) -> np.ndarray:
    image = Image.fromarray(feature.astype(np.uint8))
    return np.asarray(image.resize((256, 256), Image.BILINEAR), dtype=np.uint8)


def sample_images(
    ckpt_path: str | Path,
    n: int,
    class_label: str,
    out_dir: str | Path,
    seed: int = 123,
) -> Iterable[Path]:
    set_global_seed(seed)
    generator = load_generator(ckpt_path)

    out_path = ensure_dir(Path(out_dir) / class_label)
    rng = np.random.default_rng(seed)
    latents = rng.standard_normal((n, generator.latent_dim))
    outputs = generator.forward(latents)
    outputs = outputs.reshape(n, 3, 16, 16)

    saved_paths = []
    for idx, tensor in enumerate(outputs):
        channels = []
        for channel in tensor:
            channel = channel - channel.min()
            channel = 255 * channel / (channel.max() + 1e-8)
            channels.append(_upsample(channel))
        array = np.stack(channels, axis=-1)
        image = Image.fromarray(array.astype(np.uint8))
        file_path = out_path / f"{class_label}_{idx:05d}.png"
        image.save(file_path)
        saved_paths.append(file_path)
    return saved_paths


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample images from a dummy StyleGAN2 checkpoint")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--class", dest="class_label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=123)
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    parser = build_argparser()
    parsed = parser.parse_args(args=args)
    sample_images(parsed.ckpt, parsed.n, parsed.class_label, parsed.out, parsed.seed)


if __name__ == "__main__":
    main()
