"""Data preparation utilities for the PatchCamelyon dataset."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
from PIL import Image

from src.common.fs import ensure_dir, write_json
from src.common.seed import set_global_seed


@dataclass
class DataConfig:
    dataset: str
    root: str
    img_size: int
    val_split: float
    test_split: float
    split_by_patient: bool
    make_masks: bool
    normalize: Dict[str, str]
    output_csv: str


CLASSES = ["normal", "tumor"]


def load_config(path: str | Path) -> DataConfig:
    with Path(path).open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return DataConfig(**cfg)


def _create_synthetic_raw(root: Path, n_per_class: int = 30, img_size: int = 256) -> None:
    """Create a minimal synthetic dataset for local testing.

    The generated images contain simple geometric patterns so unit tests have
    deterministic content to reason about.
    """

    rng = np.random.default_rng(123)
    for label in CLASSES:
        class_dir = ensure_dir(root / "raw" / label)
        for idx in range(n_per_class):
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            color = 255 if label == "tumor" else 100
            rr, cc = np.ogrid[:img_size, :img_size]
            center = img_size // 2
            mask = (rr - center) ** 2 + (cc - center) ** 2 <= (img_size // 4) ** 2
            img[..., :] = rng.integers(0, 40)
            img[mask] = color
            image = Image.fromarray(img, mode="RGB")
            image.save(class_dir / f"synthetic_{idx:03d}.png")


def _load_raw_samples(root: Path) -> List[Tuple[Path, str]]:
    samples: List[Tuple[Path, str]] = []
    for label in CLASSES:
        class_dir = root / "raw" / label
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.glob("*")):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                continue
            samples.append((path, label))
    return samples


def _stratified_split(samples: List[Tuple[Path, str]], val: float, test: float) -> Dict[str, List[Tuple[Path, str]]]:
    rng = random.Random(123)
    per_class: Dict[str, List[Tuple[Path, str]]] = {label: [] for label in CLASSES}
    for path, label in samples:
        per_class[label].append((path, label))
    for entries in per_class.values():
        rng.shuffle(entries)

    splits: Dict[str, List[Tuple[Path, str]]] = {"train": [], "val": [], "test": []}
    for label, entries in per_class.items():
        n_total = len(entries)
        n_val = int(round(n_total * val))
        n_test = int(round(n_total * test))
        n_train = max(n_total - n_val - n_test, 0)
        splits["val"].extend(entries[:n_val])
        splits["test"].extend(entries[n_val:n_val + n_test])
        splits["train"].extend(entries[n_val + n_test:n_val + n_test + n_train])
    return splits


def _process_and_save(samples: List[Tuple[Path, str]], out_dir: Path, img_size: int) -> List[Dict[str, str]]:
    manifest_rows: List[Dict[str, str]] = []
    ensure_dir(out_dir)
    for src_path, label in samples:
        image = Image.open(src_path).convert("RGB")
        image = image.resize((img_size, img_size), Image.BILINEAR)
        split_dir = ensure_dir(out_dir / label)
        dst_path = split_dir / src_path.name.replace(src_path.suffix, ".png")
        image.save(dst_path, format="PNG")
        manifest_rows.append({
            "path": str(dst_path),
            "label": label,
        })
    return manifest_rows


def prepare_pcam(config_path: str | Path) -> Dict[str, any]:
    config = load_config(config_path)
    set_global_seed(123)

    root = ensure_dir(config.root)
    raw_samples = _load_raw_samples(Path(config.root))

    if not raw_samples:
        _create_synthetic_raw(Path(config.root), img_size=config.img_size)
        raw_samples = _load_raw_samples(Path(config.root))

    if not raw_samples:
        raise RuntimeError("No raw samples found or generated.")

    splits = _stratified_split(raw_samples, config.val_split, config.test_split)

    split_dirs = {
        "train": ensure_dir(Path(config.root) / "train_images"),
        "val": ensure_dir(Path(config.root) / "val_images"),
        "test": ensure_dir(Path(config.root) / "test_images"),
    }

    manifest: List[Dict[str, str]] = []
    stats: Dict[str, Dict[str, int]] = {}

    for split_name, split_samples in splits.items():
        processed = _process_and_save(split_samples, split_dirs[split_name], config.img_size)
        csv_path = Path(config.root) / f"{split_name}_labels.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["path", "label"])
            writer.writeheader()
            writer.writerows(processed)
        manifest.extend({
            "path": row["path"],
            "label": row["label"],
            "split": split_name,
            "patient_id": "synthetic",
            "mask_path": "",
        } for row in processed)

        label_counts: Dict[str, int] = {label: 0 for label in CLASSES}
        for row in processed:
            label_counts[row["label"]] += 1
        stats[split_name] = label_counts

    manifest_path = Path(config.output_csv)
    ensure_dir(manifest_path.parent)
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "label", "split", "patient_id", "mask_path"])
        writer.writeheader()
        writer.writerows(manifest)

    write_json({"class_balance": stats}, Path(config.root) / "stats.json")

    return {"manifest": manifest, "stats": stats}


__all__ = ["DataConfig", "prepare_pcam", "load_config"]
