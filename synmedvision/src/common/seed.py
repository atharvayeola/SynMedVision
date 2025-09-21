"""Utilities for reproducible experiments."""

from __future__ import annotations


def set_global_seed(seed: int) -> None:
    """Set all relevant random seeds.

    Parameters
    ----------
    seed:
        The seed value to broadcast across Python, NumPy, and Torch.
    """

    import os
    import random

    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # type: ignore
    except ImportError:  # pragma: no cover
        torch = None

    if torch is not None:
        torch.manual_seed(seed)
        if hasattr(torch, "cuda"):
            torch.cuda.manual_seed_all(seed)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int, base_seed: int = 123) -> None:
    """Helper to seed dataloader workers deterministically."""

    set_global_seed(base_seed + worker_id)
