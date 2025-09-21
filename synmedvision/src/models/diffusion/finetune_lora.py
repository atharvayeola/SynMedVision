"""Mock diffusion LoRA fine-tuning implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json
import numpy as np

from src.common.fs import ensure_dir, write_json
from src.common.seed import set_global_seed


class DummyDiffusion:
    def __init__(self, latent_dim: int = 64):
        self.latent_dim = latent_dim
        rng = np.random.default_rng(321)
        self.weights = rng.standard_normal((latent_dim, latent_dim))

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights

    def to_dict(self) -> Dict[str, Any]:
        return {"latent_dim": self.latent_dim, "weights": self.weights.tolist()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DummyDiffusion":
        obj = cls(latent_dim=int(data["latent_dim"]))
        obj.weights = np.asarray(data["weights"], dtype=np.float32)
        return obj


def _save_checkpoint(model: DummyDiffusion, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".json")
    tmp_path.write_text(json.dumps(model.to_dict()))
    tmp_path.replace(path)


def _load_checkpoint(path: Path) -> DummyDiffusion:
    data = json.loads(Path(path).read_text())
    return DummyDiffusion.from_dict(data)


def finetune_lora(config: Dict[str, Any]) -> Path:
    set_global_seed(int(config.get("seed", 123)))
    out_dir = ensure_dir(config.get("out_dir", "out/diffusion_lora"))

    model = DummyDiffusion()
    rng = np.random.default_rng(321)
    inputs = rng.standard_normal((16, model.latent_dim))
    targets = np.zeros_like(inputs)
    outputs = model.forward(inputs)
    loss = float(np.mean((outputs - targets) ** 2))

    grad = outputs - targets
    model.weights -= 0.001 * inputs.T @ grad / len(inputs)

    ckpt_path = out_dir / "lora.pt"
    _save_checkpoint(model, ckpt_path)
    write_json({"loss": loss, "epsilon": float(config.get("dp", {}).get("target_epsilon", 10))}, out_dir / "train_log.json")
    return ckpt_path


__all__ = ["finetune_lora", "DummyDiffusion", "_load_checkpoint"]
