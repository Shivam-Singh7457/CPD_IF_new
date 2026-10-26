"""Tests for power sanity (Task 6 / T7).

T7 Power sanity: with effect 3.0, phi=0.6, sparsity 0.1:
    detection rate >=99% and median delay <60.
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
from ocdmix.fit import fit_training
from ocdmix.scan import compute_statistics_vectorised, detect_alarm


def test_power_sanity_t7():
    """T7: effect 3.0, phi=0.6, sparsity 0.1 -> detection >= 99% and median delay < 60."""
    p = 50
    T = 3000
    m0 = 1000
    phi = 0.6
    n_calib = 150
    n_test = 100
    b = 1100

    # Shared model
    rng_model = np.random.default_rng(np.random.SeedSequence([20261014, 0, 9173]))
    A = make_var_matrix(p, phi, "isotropic", rng_model)
    sigma_pre = make_innovation_covariance(p, 0.25)
    gamma_pre = _stationary_covariance(A, sigma_pre, phi)

    # 1. Null streams for calibration
    null_obs = np.empty((n_calib, T, p), dtype=np.float64)
    for i in range(n_calib):
        rng = np.random.default_rng(np.random.SeedSequence([20261014, 0, i, 2718]))
        X, _ = generate_one_stream(
            T=T, p=p, A=A, phi=phi,
            sigma_pre=sigma_pre, gamma_pre=gamma_pre,
            rng=rng, change_type="none",
            cp_fraction=0.5, cp_index=None,
            sparsity=0.1, effect=0.5,
            covariance_multiplier=1.5,
        )
        null_obs[i] = X

    calib = calibrate_thresholds(
        null_obs, m0=m0, alpha0=0.05, method="proposed", lrv_method="prewhiten"
    )

    # 2. Test streams with effect 3.0
    detected_count = 0
    delays = []

    for i in range(n_test):
        rng = np.random.default_rng(np.random.SeedSequence([20261014, 777, i, 2718]))
        X, gt = generate_one_stream(
            T=T, p=p, A=A, phi=phi,
            sigma_pre=sigma_pre, gamma_pre=gamma_pre,
            rng=rng, change_type="mean",
            cp_fraction=0.5, cp_index=b,
            sparsity=0.1, effect=3.0,
            covariance_multiplier=1.5,
            sign_mode="random",
        )
        fit = fit_training(X[:m0], m0=m0, method="proposed", lrv_method="prewhiten")
        A_stat, Q_stat, _ = compute_statistics_vectorised(X, fit)
        alarm = detect_alarm(A_stat, Q_stat, calib.c_A, calib.c_Q, fit)

        if alarm is not None and alarm.N >= b:
            detected_count += 1
            delays.append(alarm.N - b)

    det_rate = detected_count / n_test
    med_delay = float(np.median(delays)) if delays else float("inf")

    print(f"\n[Effect 3.0, phi=0.6]")
    print(f"Detection rate: {det_rate * 100:.1f}% ({detected_count}/{n_test})")
    print(f"Median delay  : {med_delay:.1f} steps")

    assert det_rate >= 0.99, f"Detection rate {det_rate * 100:.1f}% is below 99%"
    assert med_delay < 60, f"Median delay {med_delay} is not < 60"
