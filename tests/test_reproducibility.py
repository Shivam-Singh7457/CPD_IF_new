"""Tests for reproducibility (T9).

T9 Acceptance Criterion:
  Running run_eval.py twice gives byte-identical CSVs.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from generate_synthetic_streams_fast import (
    _stationary_covariance,
    generate_one_stream,
    make_innovation_covariance,
    make_var_matrix,
)


def _setup_dataset(tmp_path: Path) -> Path:
    data_dir = tmp_path / "test_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    p = 10
    T = 1500
    m0 = 1000
    n_streams = 5
    phi = 0.3

    rng_model = np.random.default_rng(np.random.SeedSequence([42, 0, 9173]))
    A = make_var_matrix(p, phi, "isotropic", rng_model)
    sigma_pre = make_innovation_covariance(p, 0.25)
    gamma_pre = _stationary_covariance(A, sigma_pre, phi)

    obs = np.empty((n_streams, T, p), dtype=np.float32)
    gt_records = []

    for i in range(n_streams):
        rng = np.random.default_rng(np.random.SeedSequence([42, 0, i, 2718]))
        X, gt = generate_one_stream(
            T=T, p=p, A=A, phi=phi,
            sigma_pre=sigma_pre, gamma_pre=gamma_pre,
            rng=rng, change_type="mean",
            cp_fraction=0.5, cp_index=1200,
            sparsity=0.2, effect=1.5,
            covariance_multiplier=1.5,
        )
        obs[i] = X.astype(np.float32)
        gt_records.append(gt)

    np.save(data_dir / "observations.npy", obs)
    with open(data_dir / "ground_truth.jsonl", "w", encoding="utf-8") as f:
        for r in gt_records:
            f.write(json.dumps(r) + "\n")

    return data_dir


def test_reproducibility_t9(tmp_path: Path):
    """T9: Running run_eval.py twice gives byte-identical CSVs."""
    data_dir = _setup_dataset(tmp_path)

    # Create dummy calibration JSON
    calib_json = tmp_path / "calib.json"
    calib_data = {
        "c_A": 1.25,
        "c_Q": 3.0,
        "base_A": 3.5,
        "tau_A": 4.375,
        "tau_Q": 3.0,
        "alpha0": 0.05,
        "m0": 1000,
        "method": "proposed",
        "lrv_method": "prewhiten",
        "window_grid": [8, 16, 32, 64, 128, 256, 512],
        "kmax": 512,
        "p": 10,
        "T": 1500,
        "n_streams": 5,
        "achieved_far": 0.05,
    }
    with open(calib_json, "w", encoding="utf-8") as f:
        json.dump(calib_data, f, indent=2)

    csv1 = tmp_path / "eval1.csv"
    json1 = tmp_path / "eval1.json"
    csv2 = tmp_path / "eval2.csv"
    json2 = tmp_path / "eval2.json"

    run_eval_script = Path(__file__).resolve().parent.parent / "scripts" / "run_eval.py"

    cmd1 = [
        sys.executable, str(run_eval_script),
        "--dataset-dir", str(data_dir),
        "--calib-json", str(calib_json),
        "--out-csv", str(csv1),
        "--out-json", str(json1),
    ]
    proc1 = subprocess.run(cmd1, capture_output=True, text=True)
    assert proc1.returncode == 0, f"run_eval run 1 failed:\n{proc1.stderr}"

    cmd2 = [
        sys.executable, str(run_eval_script),
        "--dataset-dir", str(data_dir),
        "--calib-json", str(calib_json),
        "--out-csv", str(csv2),
        "--out-json", str(json2),
    ]
    proc2 = subprocess.run(cmd2, capture_output=True, text=True)
    assert proc2.returncode == 0, f"run_eval run 2 failed:\n{proc2.stderr}"

    bytes1 = csv1.read_bytes()
    bytes2 = csv2.read_bytes()

    assert bytes1 == bytes2, "CSVs produced by two runs of run_eval.py are NOT byte-identical!"
