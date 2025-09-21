"""Training entrypoints for all supported models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from src.models.diffusion.finetune_lora import finetune_lora
from src.models.stylegan2.train import train_stylegan2


def _load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train synthetic image generators")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=["stylegan2", "diffusion_lora"], required=True)
    return parser


def main(args: list[str] | None = None) -> None:
    parser = build_argparser()
    parsed = parser.parse_args(args=args)
    config = _load_config(parsed.config)

    if parsed.model == "stylegan2":
        ckpt = train_stylegan2(config)
        print(f"Saved StyleGAN2 checkpoint to {ckpt}")
    elif parsed.model == "diffusion_lora":
        ckpt = finetune_lora(config)
        print(f"Saved diffusion LoRA checkpoint to {ckpt}")
    else:
        raise ValueError(f"Unsupported model: {parsed.model}")


if __name__ == "__main__":
    main()
