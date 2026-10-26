"""Tests for detector blindness (T8).

T8 Acceptance Criterion:
  Asserts none of the detector/calibration modules opens ground_truth.jsonl or manifest.json
  (by inspecting source for those strings and by running with those files removed).
"""
from __future__ import annotations

import inspect
from pathlib import Path
import numpy as np
import pytest

import ocdmix.calibrate
import ocdmix.fit
import ocdmix.localize
import ocdmix.online
import ocdmix.scan
from ocdmix.calibrate import calibrate_thresholds
from ocdmix.fit import fit_training
from ocdmix.io import load_observations
from ocdmix.localize import localize_at_alarm
from ocdmix.online import OnlineDetector
from ocdmix.scan import compute_base_A, compute_statistics_vectorised, detect_alarm


def test_blindness_source_inspection():
    """Verify detector and calibration modules contain no references to ground truth or manifest."""
    detector_modules = [
        ocdmix.fit,
        ocdmix.scan,
        ocdmix.online,
        ocdmix.calibrate,
        ocdmix.localize,
    ]

    forbidden_strings = [
        "ground_truth.jsonl",
        "manifest.json",
        "load_ground_truth",
        "load_manifest",
        "affected_coordinates",
        "changepoint_index",
    ]

    for mod in detector_modules:
        source = inspect.getsource(mod)
        for forbidden in forbidden_strings:
            assert forbidden not in source, (
                f"Module {mod.__name__} violates blindness! Found forbidden string '{forbidden}' in source."
            )


def test_blindness_runtime_without_ground_truth(tmp_path: Path):
    """Verify detector and calibration run completely when only observations.npy exists."""
    p = 10
    T = 1500
    m0 = 1000
    n_streams = 5

    # Create dummy observations.npy in an isolated directory
    obs_data = np.random.default_rng(42).standard_normal((n_streams, T, p)).astype(np.float32)
    obs_path = tmp_path / "observations.npy"
    np.save(obs_path, obs_data)

    # Ensure NO ground_truth.jsonl or manifest.json exists in tmp_path
    assert not (tmp_path / "ground_truth.jsonl").exists()
    assert not (tmp_path / "manifest.json").exists()

    # 1. Load observations
    obs = load_observations(tmp_path, mmap=True)
    assert obs.shape == (n_streams, T, p)

    # 2. Calibrate thresholds
    calib = calibrate_thresholds(obs, m0=m0, alpha0=0.05, method="proposed")
    assert calib.c_A > 0
    assert calib.c_Q != 0

    # 3. Batch scan
    fit = fit_training(obs[0, :m0], m0=m0, method="proposed")
    A, Q, _ = compute_statistics_vectorised(obs[0], fit)
    alarm = detect_alarm(A, Q, calib.c_A, calib.c_Q, fit)

    # 4. Localize (if alarm)
    if alarm is not None:
        base_A = compute_base_A(p, len(fit.window_grid))
        nu_hat, CI, support = localize_at_alarm(
            N=alarm.N,
            t_N=alarm.t_N,
            X=obs[0],
            fit=fit,
            tau_A=calib.c_A * base_A,
            tau_Q=calib.c_Q,
            c_A=calib.c_A,
            c_Q=calib.c_Q,
            ci_a=1.0,
        )
        assert isinstance(nu_hat, int)

    # 5. OnlineDetector
    detector = OnlineDetector(fit, c_A=calib.c_A, c_Q=calib.c_Q, ci_a=1.0)
    for t in range(T):
        detector.update(obs[0, t])

    # Succeeded with zero access to ground truth or manifest
