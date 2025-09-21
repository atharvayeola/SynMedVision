"""CLI for producing the HTML evaluation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common.fs import read_json
from src.eval.report import generate_report


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate HTML evaluation report")
    parser.add_argument("--metrics", default="out/eval_metrics.json")
    parser.add_argument("--out", default="reports/eval_report.html")
    return parser


def main(args: list[str] | None = None) -> None:
    parser = build_argparser()
    parsed = parser.parse_args(args=args)
    metrics = read_json(parsed.metrics) if Path(parsed.metrics).exists() else {}
    path = generate_report(metrics, parsed.out)
    print(f"Saved report to {path}")


if __name__ == "__main__":
    main()
