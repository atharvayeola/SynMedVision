"""Privacy related diagnostics for synthetic datasets."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
from imagehash import phash
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_distances


def _load_image_features(paths: Iterable[Path], image_size: int = 256) -> np.ndarray:
    features: List[np.ndarray] = []
    for path in paths:
        image = Image.open(path).convert("RGB").resize((image_size, image_size))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        features.append(arr.mean(axis=2).flatten())
    if not features:
        return np.zeros((0, image_size * image_size))
    return np.stack(features)


def nearest_neighbor_distances(real_dir: str | Path, synth_dir: str | Path, image_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    real_paths = list(Path(real_dir).rglob("*.png"))
    synth_paths = list(Path(synth_dir).rglob("*.png"))

    real_features = _load_image_features(real_paths, image_size)
    synth_features = _load_image_features(synth_paths, image_size)

    if real_features.size == 0 or synth_features.size == 0:
        return np.array([]), np.array([])

    distances = cosine_distances(synth_features, real_features)
    nearest = distances.min(axis=1)
    indices = distances.argmin(axis=1)
    return nearest, indices


def find_near_duplicates(real_dir: str | Path, synth_dir: str | Path, tau: float) -> List[Tuple[Path, Path, float]]:
    nearest, indices = nearest_neighbor_distances(real_dir, synth_dir)
    if nearest.size == 0:
        return []
    real_paths = list(Path(real_dir).rglob("*.png"))
    synth_paths = list(Path(synth_dir).rglob("*.png"))
    risky = []
    for dist, idx_synth, idx_real in zip(nearest, range(len(synth_paths)), indices):
        if dist <= tau:
            risky.append((synth_paths[idx_synth], real_paths[idx_real], float(dist)))
    return risky


def membership_inference(real_dir: str | Path, image_size: int = 256) -> float:
    real_paths = list(Path(real_dir).rglob("*.png"))
    if len(real_paths) < 4:
        return float("nan")
    split = len(real_paths) // 2
    member_paths = real_paths[:split]
    non_member_paths = real_paths[split:2 * split]

    member_features = _load_image_features(member_paths, image_size)
    non_member_features = _load_image_features(non_member_paths, image_size)
    features = np.concatenate([member_features, non_member_features], axis=0)
    labels = np.concatenate([np.ones(len(member_features)), np.zeros(len(non_member_features))])

    clf = LogisticRegression(max_iter=200)
    clf.fit(features, labels)
    probs = clf.predict_proba(features)[:, 1]
    return float(roc_auc_score(labels, probs))


def phash_duplicates(real_dir: str | Path, synth_dir: str | Path, threshold: int = 6) -> List[Tuple[Path, Path, int]]:
    def hash_directory(directory: Path) -> List[Tuple[Path, any]]:
        hashes = []
        for path in directory.rglob("*.png"):
            hashes.append((path, phash(Image.open(path))))
        return hashes

    real_hashes = hash_directory(Path(real_dir))
    synth_hashes = hash_directory(Path(synth_dir))

    duplicates: List[Tuple[Path, Path, int]] = []
    for (path_s, hash_s), (path_r, hash_r) in itertools.product(synth_hashes, real_hashes):
        dist = hash_s - hash_r
        if dist <= threshold:
            duplicates.append((path_s, path_r, int(dist)))
    return duplicates


def privacy_report(real_dir: str | Path, synth_dir: str | Path, tau: float, phash_threshold: int = 6) -> Dict[str, any]:
    nearest, _ = nearest_neighbor_distances(real_dir, synth_dir)
    duplicates = phash_duplicates(real_dir, synth_dir, phash_threshold)
    membership_auc = membership_inference(real_dir)
    risky = np.mean(nearest <= tau) * 100 if nearest.size else float("nan")
    return {
        "nn_min": float(nearest.min()) if nearest.size else float("nan"),
        "nn_mean": float(nearest.mean()) if nearest.size else float("nan"),
        "near_duplicate_pct": float(risky),
        "membership_auc": membership_auc,
        "phash_duplicates": duplicates,
    }


__all__ = [
    "nearest_neighbor_distances",
    "find_near_duplicates",
    "membership_inference",
    "phash_duplicates",
    "privacy_report",
]
