"""Sampling utilities for the dummy diffusion pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image

from src.common.fs import ensure_dir
from src.common.seed import set_global_seed
from src.models.diffusion.finetune_lora import DummyDiffusion, _load_checkpoint


def load_model(base: str | Path) -> DummyDiffusion:
    ckpt_path = Path(base) / "lora.pt"
    return _load_checkpoint(ckpt_path)


def _apply_mask_overlay(image: np.ndarray, mask_path: Optional[str]) -> np.ndarray:
    if not mask_path:
        return image
    mask_img = Image.open(mask_path).convert("L").resize(image.shape[:2][::-1])
    mask = np.asarray(mask_img, dtype=np.float32) / 255.0
    overlay = image.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] * (0.5 + 0.5 * mask), 0, 255)
    return overlay


def sample_images(
    base: str | Path,
    n: int,
    class_label: str,
    steps: int = 30,
    guidance: float = 7.5,
    seed: int = 123,
    out: str | Path = "out/samples",
    mask_path: Optional[str] = None,
) -> Iterable[Path]:
    set_global_seed(seed)
    model = load_model(base)

    out_dir = ensure_dir(Path(out) / class_label)
    rng = np.random.default_rng(seed)
    latents = rng.standard_normal((n, model.latent_dim))
    features = model.forward(latents)
    features = np.tanh(features)
    features = features.reshape(n, 8, 8, -1).mean(axis=-1)

    saved_paths = []
    for idx, feat in enumerate(features):
        array = feat - feat.min()
        array = 255 * array / (array.max() + 1e-8)
        array = np.asarray(Image.fromarray(array.astype(np.uint8)).resize((256, 256)), dtype=np.float32)
        rgb = np.stack([array, np.flipud(array), np.roll(array, 5, axis=0)], axis=-1)
        rgb = _apply_mask_overlay(rgb, mask_path)
        image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
        path = out_dir / f"{class_label}_{idx:05d}.png"
        image.save(path)
        saved_paths.append(path)
    return saved_paths


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample images from the dummy diffusion pipeline")
    parser.add_argument("--base", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--class", dest="class_label", required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mask_path")
    return parser


def main(args: Optional[list[str]] = None) -> None:
    parser = build_argparser()
    parsed = parser.parse_args(args=args)
    sample_images(
        base=parsed.base,
        n=parsed.n,
        class_label=parsed.class_label,
        steps=parsed.steps,
        guidance=parsed.guidance,
        seed=parsed.seed,
        out=parsed.out,
        mask_path=parsed.mask_path,
    )


if __name__ == "__main__":
    main()
