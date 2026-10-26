"""Tests for calibration module (Task 4 / T6).

T6 Reduction to independence: with phi=0, c_A of "proposed" and of "vanilla"
    differ by <10% on the same null set.
"""
from __future__ import annotations

import numpy as np
import pytest

from generate_synthetic_streams_fast import (
    _stationary_covariance,
    generate_one_stream,
    make_innovation_covariance,
    make_var_matrix,
)
from ocdmix.calibrate import calibrate_thresholds


def _generate_null_dataset(n_streams: int, T: int, p: int, phi: float = 0.0, seed: int = 20261014) -> np.ndarray:
    rng_model = np.random.default_rng(np.random.SeedSequence([seed, 0, 9173]))
    A = make_var_matrix(p, phi, "isotropic", rng_model)
    sigma_pre = make_innovation_covariance(p, 0.25)
    gamma_pre = _stationary_covariance(A, sigma_pre, phi)

    obs = np.empty((n_streams, T, p), dtype=np.float64)
    for i in range(n_streams):
        stream_seed = np.random.SeedSequence([seed, 0, i, 2718])
        rng = np.random.default_rng(stream_seed)
        X, _ = generate_one_stream(
            T=T, p=p, A=A, phi=phi,
            sigma_pre=sigma_pre, gamma_pre=gamma_pre,
            rng=rng, change_type="none",
            cp_fraction=0.5, cp_index=None,
            sparsity=0.1, effect=0.5,
            covariance_multiplier=1.5,
        )
        obs[i] = X
    return obs


def test_reduction_to_independence_t6():
    """T6: with phi=0, c_A of 'proposed' and 'vanilla' differ by <10% on the same null set."""
    n_streams = 40
    T = 5000
    p = 50
    m0 = 1000

    obs = _generate_null_dataset(n_streams=n_streams, T=T, p=p, phi=0.0, seed=20261014)

    calib_prop = calibrate_thresholds(
        obs, m0=m0, alpha0=0.05, method="proposed", lrv_method="prewhiten"
    )
    calib_van = calibrate_thresholds(
        obs, m0=m0, alpha0=0.05, method="vanilla"
    )

    c_A_prop = calib_prop.c_A
    c_A_van = calib_van.c_A
    rel_diff = abs(c_A_prop - c_A_van) / c_A_van

    print(f"\n[phi=0.0] Proposed c_A = {c_A_prop:.4f}")
    print(f"[phi=0.0] Vanilla  c_A = {c_A_van:.4f}")
    print(f"Relative difference = {rel_diff * 100:.2f}%")

    assert rel_diff < 0.10, (
        f"c_A differs by {rel_diff * 100:.2f}% (exceeds 10% limit): proposed={c_A_prop}, vanilla={c_A_van}"
    )
