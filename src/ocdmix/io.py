"""I/O utilities for loading observations and ground truth."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_observations(dataset_dir: str | Path, mmap: bool = True) -> np.ndarray:
    """Load observations.npy from *dataset_dir*.

    Parameters
    ----------
    dataset_dir : path-like
        Directory containing observations.npy.
    mmap : bool
        If True, use memory-mapped mode (``mmap_mode='r'``).

    Returns
    -------
    np.ndarray
        Shape ``[n_streams, T, p]``, dtype as stored (typically float32).
    """
    path = Path(dataset_dir) / "observations.npy"
    mode = "r" if mmap else None
    return np.load(path, mmap_mode=mode)


def load_ground_truth(dataset_dir: str | Path) -> list[dict[str, Any]]:
    """Load ground_truth.jsonl from *dataset_dir*.

    Returns a list of dicts, one per stream, with at least the fields:
    ``changepoint_index_zero_based``, ``affected_coordinates_zero_based``,
    ``mean_shift_vector``, ``change_type``, ``phi``.
    """
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
