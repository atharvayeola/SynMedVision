"""Utility metrics comparing synthetic and real datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


@dataclass
class Dataset:
    features: np.ndarray
    labels: np.ndarray


CLASS_TO_INT = {"normal": 0, "tumor": 1}


def _load_image(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((image_size, image_size))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr.mean(axis=2).flatten()


def _load_from_csv(csv_path: Path, image_size: int) -> Dataset:
    if not csv_path.exists():
        return Dataset(features=np.zeros((0, image_size * image_size)), labels=np.zeros((0,)))
    paths: list[str] = []
    labels: list[int] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        for line in fh:
            row = dict(zip(header, line.strip().split(",")))
            paths.append(row["path"])
            labels.append(CLASS_TO_INT[row["label"]])
    features = np.stack([_load_image(Path(p), image_size) for p in paths]) if paths else np.zeros((0, image_size * image_size))
    return Dataset(features=features, labels=np.array(labels, dtype=np.int32))


def _load_from_directory(directory: Path, image_size: int) -> Dataset:
    features: list[np.ndarray] = []
    labels: list[int] = []
    for label_name, label_idx in CLASS_TO_INT.items():
        label_dir = directory / label_name
        if not label_dir.exists():
            continue
        for path in sorted(label_dir.glob("*.png")):
            features.append(_load_image(path, image_size))
            labels.append(label_idx)
    if not features:
        return Dataset(features=np.zeros((0, image_size * image_size)), labels=np.zeros((0,)))
    return Dataset(features=np.stack(features), labels=np.array(labels, dtype=np.int32))


def _train_classifier(dataset: Dataset) -> LogisticRegression:
    if dataset.features.shape[0] == 0:
        raise ValueError("Empty dataset provided to classifier training")
    clf = LogisticRegression(max_iter=200)
    clf.fit(dataset.features, dataset.labels)
    return clf


def _evaluate(clf: LogisticRegression, dataset: Dataset) -> Dict[str, float]:
    if dataset.features.shape[0] == 0:
        return {"accuracy": float("nan"), "roc_auc": float("nan"), "pr_auc": float("nan")}
    probs = clf.predict_proba(dataset.features)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(dataset.labels, preds)),
        "roc_auc": float(roc_auc_score(dataset.labels, probs)),
        "pr_auc": float(average_precision_score(dataset.labels, probs)),
    }


def compute_utility_metrics(data_root: str | Path, synth_dir: str | Path, image_size: int = 256) -> Dict[str, Dict[str, float]]:
    data_root = Path(data_root)
    synth_dir = Path(synth_dir)

    train_dataset = _load_from_csv(data_root / "train_labels.csv", image_size)
    val_dataset = _load_from_csv(data_root / "val_labels.csv", image_size)
    synth_dataset = _load_from_directory(synth_dir, image_size)

    metrics: Dict[str, Dict[str, float]] = {}

    if synth_dataset.features.size and val_dataset.features.size:
        clf_synth = _train_classifier(synth_dataset)
        metrics["tstr"] = _evaluate(clf_synth, val_dataset)
    else:
        metrics["tstr"] = {"accuracy": float("nan"), "roc_auc": float("nan"), "pr_auc": float("nan")}

    if train_dataset.features.size and synth_dataset.features.size:
        clf_real = _train_classifier(train_dataset)
        metrics["trts"] = _evaluate(clf_real, synth_dataset)
    else:
        metrics["trts"] = {"accuracy": float("nan"), "roc_auc": float("nan"), "pr_auc": float("nan")}

    return metrics


__all__ = ["compute_utility_metrics"]
