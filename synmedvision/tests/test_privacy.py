from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.eval.privacy import membership_inference, nearest_neighbor_distances, phash_duplicates


def _create_image(path: Path, value: int) -> None:
    array = np.full((256, 256, 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path)


def test_nearest_neighbor_distance_range(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    synth_dir = tmp_path / "synth"
    real_dir.mkdir()
    synth_dir.mkdir()

    for idx in range(4):
        _create_image(real_dir / f"real_{idx}.png", 30 + idx)
        _create_image(synth_dir / f"synth_{idx}.png", 60 + idx)

    distances, indices = nearest_neighbor_distances(real_dir, synth_dir)
    assert distances.size == 4
    assert np.all(distances >= 0) and np.all(distances <= 2)


def test_phash_duplicates_detects_match(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    synth_dir = tmp_path / "synth"
    real_dir.mkdir()
    synth_dir.mkdir()

    _create_image(real_dir / "real.png", 100)
    _create_image(synth_dir / "synth.png", 100)

    duplicates = phash_duplicates(real_dir, synth_dir, threshold=1)
    assert duplicates, "Expected duplicate pair to be identified"


def test_membership_inference_auc_returns_value(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    for idx in range(10):
        _create_image(real_dir / f"img_{idx}.png", 40 + idx)

    auc = membership_inference(real_dir)
    assert 0.0 <= auc <= 1.0
