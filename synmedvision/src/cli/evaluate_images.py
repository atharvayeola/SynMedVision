"""CLI for evaluating synthetic image datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import yaml

from src.common.fs import ensure_dir, write_json
from src.eval.privacy import privacy_report
from src.eval.realism import compute_realism_metrics
from src.eval.utility import compute_utility_metrics


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate generated images")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="out/eval_metrics.json")
    return parser


def _load_config(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def evaluate(config_path: str | Path) -> Dict[str, Dict]:
    config = _load_config(config_path)
    real_dir = config.get("real_dir")
    synth_dir = config.get("synth_dir")
    image_size = int(config.get("image_size", 256))

    metrics_cfg = config.get("metrics", {})
    results: Dict[str, Dict] = {}

    if metrics_cfg.get("realism", {}).get("kid"):
        results["realism"] = compute_realism_metrics(real_dir, synth_dir, image_size)

    utility_cfg = metrics_cfg.get("utility", {})
    if utility_cfg.get("tstr") or utility_cfg.get("trts"):
        data_root = Path(config.get("labels_csv", "")).parent
        results["utility"] = compute_utility_metrics(data_root, synth_dir, image_size)

    privacy_cfg = metrics_cfg.get("privacy", {})
    if privacy_cfg:
        tau = float(privacy_cfg.get("nn_tau", 0.3))
        phash_threshold = int(privacy_cfg.get("phash_dup_threshold", 6))
        results["privacy"] = privacy_report(real_dir, synth_dir, tau, phash_threshold)

    return results


def main(args: list[str] | None = None) -> None:
    parser = build_argparser()
    parsed = parser.parse_args(args=args)
    metrics = evaluate(parsed.config)
    ensure_dir(Path(parsed.out).parent)
    write_json(metrics, parsed.out)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
