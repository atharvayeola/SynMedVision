from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.eval.realism import compute_realism_metrics
from src.eval.utility import compute_utility_metrics


def _create_image(directory: Path, name: str, intensity: float) -> None:
    array = np.full((256, 256, 3), fill_value=intensity, dtype=np.uint8)
    Image.fromarray(array).save(directory / f"{name}.png")


def test_realism_metrics_return_finite_values(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    synth_dir = tmp_path / "synth"
    real_dir.mkdir()
    synth_dir.mkdir()

    for idx in range(10):
        _create_image(real_dir, f"real_{idx}", intensity=50 + idx)
        _create_image(synth_dir, f"synth_{idx}", intensity=100 + idx)

    metrics = compute_realism_metrics(real_dir, synth_dir)
    assert np.isfinite(metrics["kid"])
    assert np.isfinite(metrics["fid"])
    assert np.isfinite(metrics["lpips"])


def test_utility_metrics_pipeline(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    synth_dir = tmp_path / "synth"
    (data_root / "train_images" / "normal").mkdir(parents=True, exist_ok=True)
    (data_root / "train_images" / "tumor").mkdir(parents=True, exist_ok=True)
    (data_root / "val_images" / "normal").mkdir(parents=True, exist_ok=True)
    (data_root / "val_images" / "tumor").mkdir(parents=True, exist_ok=True)
    (synth_dir / "normal").mkdir(parents=True, exist_ok=True)
    (synth_dir / "tumor").mkdir(parents=True, exist_ok=True)

    def write_split(split: str, base_intensity: int) -> None:
        rows = ["path,label\n"]
        for idx in range(10):
            label = "normal" if idx % 2 == 0 else "tumor"
            path = data_root / f"{split}_images" / label / f"img_{idx}.png"
            _create_image(path.parent, f"img_{idx}", intensity=base_intensity + idx * 5)
            rows.append(f"{path},{label}\n")
        (data_root / f"{split}_labels.csv").write_text("".join(rows), encoding="utf-8")

    write_split("train", 30)
    write_split("val", 60)

    for idx in range(10):
        label = "normal" if idx % 2 == 0 else "tumor"
        _create_image(synth_dir / label, f"synth_{idx}", intensity=80 + idx * 3)

    metrics = compute_utility_metrics(data_root, synth_dir)
    assert "tstr" in metrics and "trts" in metrics
    for section in metrics.values():
        for value in section.values():
            assert np.isfinite(value)
