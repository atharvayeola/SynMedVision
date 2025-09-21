"""Filesystem helpers used across the pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    """Create a directory (and parents) if it does not exist."""

    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def write_json(data: Dict[str, Any], path: os.PathLike[str] | str) -> None:
    """Write a JSON file with pretty formatting."""

    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


def read_json(path: os.PathLike[str] | str) -> Dict[str, Any]:
    """Read a JSON file returning a dictionary."""

    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class RunMetadata:
    """Container describing a pipeline execution."""

    seed: int
    git_commit: str
    configs: Dict[str, str]
    metrics: Dict[str, float]
    environment: Dict[str, Any]

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


def write_run_metadata(metadata: RunMetadata, path: os.PathLike[str] | str) -> None:
    write_json(metadata.to_json(), path)
