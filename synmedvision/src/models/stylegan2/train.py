"""Placeholder StyleGAN2 training loop used for local testing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json
import numpy as np

from src.common.fs import ensure_dir, write_json
from src.common.seed import set_global_seed


class DummyStyleGAN2:
    """A minimal generator implemented with NumPy."""

    def __init__(self, latent_dim: int = 32):
        self.latent_dim = latent_dim
        rng = np.random.default_rng(123)
        self.weights = rng.standard_normal((latent_dim, 3 * 16 * 16))
        self.bias = rng.standard_normal(3 * 16 * 16)

    def forward(self, z: np.ndarray) -> np.ndarray:
        return z @ self.weights + self.bias

    def to_dict(self) -> Dict[str, Any]:
        return {"latent_dim": self.latent_dim, "weights": self.weights.tolist(), "bias": self.bias.tolist()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DummyStyleGAN2":
        obj = cls(latent_dim=int(data["latent_dim"]))
        obj.weights = np.asarray(data["weights"], dtype=np.float32)
        obj.bias = np.asarray(data["bias"], dtype=np.float32)
        return obj


def _save_checkpoint(model: DummyStyleGAN2, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".json")
    tmp_path.write_text(json.dumps(model.to_dict()))
    tmp_path.replace(path)


def _load_checkpoint(path: Path) -> DummyStyleGAN2:
    data = json.loads(Path(path).read_text())
    return DummyStyleGAN2.from_dict(data)


def train_stylegan2(config: Dict[str, Any]) -> Path:
    """Create a deterministic mock checkpoint without relying on torch."""

    set_global_seed(int(config.get("seed", 123)))
    out_dir = ensure_dir(config.get("out_dir", "out/stylegan2"))

    model = DummyStyleGAN2()
    rng = np.random.default_rng(123)
    latents = rng.standard_normal((8, model.latent_dim))
    target = np.zeros((8, 3 * 16 * 16), dtype=np.float32)
    outputs = model.forward(latents)
    loss = float(np.mean((outputs - target) ** 2))

    # Single gradient-like update to modify weights
    grad = (outputs - target)
    model.weights -= 0.001 * latents.T @ grad / len(latents)
    model.bias -= 0.001 * grad.mean(axis=0)

    ckpt_path = out_dir / "ckpt.pt"
    _save_checkpoint(model, ckpt_path)
    write_json({"loss": loss}, out_dir / "train_log.json")
    return ckpt_path


__all__ = ["train_stylegan2", "DummyStyleGAN2", "_load_checkpoint"]
