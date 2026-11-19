from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_observations(dataset_dir: str | Path, mmap: bool = True) -> np.ndarray:
    """Load observations.npy from *dataset_dir*."""
    path = Path(dataset_dir) / "observations.npy"
    mode = "r" if mmap else None
    return np.load(path, mmap_mode=mode)


def load_ground_truth(dataset_dir: str | Path) -> list[dict[str, Any]]:
    """Load ground_truth.jsonl from *dataset_dir*."""
    path = Path(dataset_dir) / "ground_truth.jsonl"
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_manifest(dataset_dir: str | Path) -> dict[str, Any]:
    """Load manifest.json from *dataset_dir*."""
    path = Path(dataset_dir) / "manifest.json"
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
