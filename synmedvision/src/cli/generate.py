"""CLI orchestrator for sampling synthetic images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable

from PIL import Image

from src.common.fs import ensure_dir
from src.models.diffusion.sample import sample_images as diffusion_sample
from src.models.stylegan2.sample import sample_images as stylegan_sample


def _parse_mix(spec: str) -> Dict[str, float]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    mix: Dict[str, float] = {}
    for part in parts:
        label, prob = part.split(":")
        mix[label.strip()] = float(prob)
    total = sum(mix.values())
    if not total:
        raise ValueError("Class mix probabilities sum to zero")
    for key in mix:
        mix[key] /= total
    return mix


def _compute_counts(total: int, mix: Dict[str, float]) -> Dict[str, int]:
    counts = {label: int(round(total * prob)) for label, prob in mix.items()}
    diff = total - sum(counts.values())
    labels = list(mix.keys())
    idx = 0
    while diff != 0 and labels:
        counts[labels[idx % len(labels)]] += 1 if diff > 0 else -1
        diff = total - sum(counts.values())
        idx += 1
    return counts


def _validate_png(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.load()
            return img.size == (256, 256) and img.mode == "RGB"
    except Exception:
        return False


def generate_samples(args: argparse.Namespace) -> Iterable[Path]:
    mix = _parse_mix(args.class_mix)
    counts = _compute_counts(args.n, mix)
    out_dir = ensure_dir(args.out)

    manifest_rows = []
    generated_paths = []

    for idx, (label, count) in enumerate(counts.items()):
        if count <= 0:
            continue
        seed = args.seed + idx
        if args.model == "stylegan2":
            ckpt = args.ckpt or "out/stylegan2/ckpt.pt"
            paths = stylegan_sample(ckpt_path=ckpt, n=count, class_label=label, out_dir=out_dir, seed=seed)
        else:
            base = args.base or "out/diffusion_lora"
            paths = diffusion_sample(base=base, n=count, class_label=label, steps=args.steps, guidance=args.guidance, seed=seed, out=args.out, mask_path=args.mask_path)
        for path in paths:
            manifest_rows.append({
                "path": str(path),
                "class": label,
                "seed": seed,
                "model": args.model,
                "params": json.dumps({"steps": args.steps, "guidance": args.guidance}),
            })
        generated_paths.extend(paths)

    manifest_path = Path(args.out) / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "class", "seed", "model", "params"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    invalid = [path for path in generated_paths if not _validate_png(path)]
    if invalid:
        raise RuntimeError(f"Invalid images generated: {invalid[:3]}")

    return generated_paths


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic images")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--class_mix", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--model", choices=["stylegan2", "diffusion_lora"], default="diffusion_lora")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--mask_path")
    parser.add_argument("--ckpt")
    parser.add_argument("--base")
    return parser


def main(args: list[str] | None = None) -> None:
    parser = build_argparser()
    parsed = parser.parse_args(args=args)
    paths = generate_samples(parsed)
    print(f"Generated {len(paths)} images to {parsed.out}")


if __name__ == "__main__":
    main()
