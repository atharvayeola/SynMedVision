"""Command line interface for dataset preparation."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.prepare_pcam import prepare_pcam


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the PCam dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    return parser


def main(args: list[str] | None = None) -> None:
    parser = build_argparser()
    parsed = parser.parse_args(args=args)
    result = prepare_pcam(parsed.config)
    print(f"Prepared dataset with {len(result['manifest'])} entries")


if __name__ == "__main__":
    main()
