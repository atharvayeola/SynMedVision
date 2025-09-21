"""Lightweight image realism and diversity metrics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from PIL import Image
from scipy import linalg

from src.common.fs import ensure_dir


def _list_image_paths(directory: str | Path) -> Iterable[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    for path in directory.rglob("*.png"):
        yield path


def _load_image(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((image_size, image_size))
    return np.asarray(image, dtype=np.float32) / 255.0


def _extract_features(images: Iterable[np.ndarray]) -> np.ndarray:
    feats = []
    for img in images:
        pooled = img.mean(axis=2)
        downsampled = pooled[::8, ::8]
        feats.append(downsampled.flatten())
    if not feats:
        return np.zeros((0, 1), dtype=np.float32)
    return np.stack(feats)


def polynomial_mmd(x: np.ndarray, y: np.ndarray, degree: int = 3, gamma: float = 1.0, coef0: float = 1.0) -> float:
    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (gamma * a @ b.T + coef0) ** degree

    k_xx = kernel(x, x)
    k_yy = kernel(y, y)
    k_xy = kernel(x, y)
    return float(k_xx.mean() + k_yy.mean() - 2 * k_xy.mean())


def compute_kid(real: np.ndarray, synth: np.ndarray) -> float:
    if len(real) == 0 or len(synth) == 0:
        return float("nan")
    return polynomial_mmd(real, synth)


def compute_fid(real: np.ndarray, synth: np.ndarray) -> float:
    if len(real) == 0 or len(synth) == 0:
        return float("nan")
    mu_r, mu_s = real.mean(axis=0), synth.mean(axis=0)
    cov_r = np.cov(real, rowvar=False)
    cov_s = np.cov(synth, rowvar=False)
    covmean = linalg.sqrtm(cov_r @ cov_s)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_r - mu_s
    fid = diff @ diff + np.trace(cov_r + cov_s - 2 * covmean)
    return float(np.real(fid))


def compute_lpips(real_images: np.ndarray, synth_images: np.ndarray) -> float:
    if len(real_images) == 0 or len(synth_images) == 0:
        return float("nan")
    n = min(len(real_images), len(synth_images))
    diffs = []
    for idx in range(n):
        diffs.append(np.mean(np.abs(real_images[idx] - synth_images[idx])))
    return float(np.mean(diffs))


def covariance_rank_ratio(real: np.ndarray, synth: np.ndarray) -> float:
    if real.size == 0 or synth.size == 0:
        return float("nan")
    cov_real = np.cov(real, rowvar=False)
    cov_synth = np.cov(synth, rowvar=False)
    rank_real = np.linalg.matrix_rank(cov_real)
    rank_synth = np.linalg.matrix_rank(cov_synth)
    if rank_real == 0:
        return float("nan")
    return float(rank_synth / rank_real)


def feature_entropy(features: np.ndarray, bins: int = 32) -> float:
    if features.size == 0:
        return float("nan")
    hist, _ = np.histogram(features, bins=bins, density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log(hist + 1e-12)))


def compute_realism_metrics(real_dir: str | Path, synth_dir: str | Path, image_size: int = 256) -> Dict[str, float]:
    real_paths = list(_list_image_paths(real_dir))
    synth_paths = list(_list_image_paths(synth_dir))

    real_images = np.array([_load_image(p, image_size) for p in real_paths]) if real_paths else np.zeros((0, image_size, image_size, 3))
    synth_images = np.array([_load_image(p, image_size) for p in synth_paths]) if synth_paths else np.zeros((0, image_size, image_size, 3))

    real_features = _extract_features(real_images)
    synth_features = _extract_features(synth_images)

    metrics = {
        "kid": compute_kid(real_features, synth_features),
        "fid": compute_fid(real_features, synth_features),
        "lpips": compute_lpips(real_images, synth_images),
        "feature_entropy_real": feature_entropy(real_features.flatten()),
        "feature_entropy_synth": feature_entropy(synth_features.flatten()),
        "cov_rank_ratio": covariance_rank_ratio(real_features, synth_features),
    }
    return metrics


__all__ = [
    "compute_realism_metrics",
    "compute_kid",
    "compute_fid",
    "compute_lpips",
    "feature_entropy",
    "covariance_rank_ratio",
]
