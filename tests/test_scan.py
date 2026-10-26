"""Tests for scan module (T2 and T3).

T2 Standardisation: on null streams, the pooled mean of W^2 for k in {8,32,128,512}, t>=m0,
    lies in [0.93,1.10] for phi in {0,0.3,0.6,0.8} (prototype: 1.00 to 1.04).
T3 Window sums: C[t]-C[t-k] equals a brute-force sum for 20 random (t,k).
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
from ocdmix.fit import fit_training
from ocdmix.scan import compute_statistics_vectorised


def _generate_stream(phi: float, T: int = 5000, p: int = 50, seed: int = 42) -> np.ndarray:
    rng_model = np.random.default_rng(np.random.SeedSequence([seed, 0, 9173]))
    A = make_var_matrix(p, phi, "isotropic", rng_model)
    sigma_pre = make_innovation_covariance(p, 0.25)
    gamma_pre = _stationary_covariance(A, sigma_pre, phi)

    rng = np.random.default_rng(np.random.SeedSequence([seed, 0, 0, 2718]))
    X, _ = generate_one_stream(
        T=T, p=p, A=A, phi=phi,
        sigma_pre=sigma_pre, gamma_pre=gamma_pre,
        rng=rng, change_type="none",
        cp_fraction=0.5, cp_index=None,
        sparsity=0.1, effect=0.5,
        covariance_multiplier=1.5,
    )
    return X


def test_window_sums_t3():
    """T3: C[t] - C[t-k] equals brute-force sum for 20 random (t, k)."""
    p = 50
    T = 5000
    m0 = 1000
    X = _generate_stream(phi=0.6, T=T, p=p, seed=20261014)
    fit = fit_training(X[:m0], m0=m0, method="proposed")

    Xc = X - fit.mu
    monitoring = Xc[m0:]
    C_raw = np.zeros((monitoring.shape[0] + 1, p), dtype=np.float64)
    np.cumsum(monitoring, axis=0, out=C_raw[1:])

    rng = np.random.default_rng(999)
    window_grid = fit.window_grid

    for _ in range(20):
        k = int(rng.choice(window_grid))
        t = int(rng.integers(m0 + k - 1, T))

        # From cumsum:
        ci_t = t - m0 + 1
        ci_tk = t - k - m0 + 1
        window_sum_cumsum = C_raw[ci_t] - C_raw[ci_tk]

        # Brute-force sum: Xc[t-k+1 : t+1]
        window_sum_brute = np.sum(Xc[t - k + 1 : t + 1], axis=0)

        np.testing.assert_allclose(
            window_sum_cumsum,
            window_sum_brute,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Mismatch at t={t}, k={k}",
        )


@pytest.mark.parametrize("phi", [0.0, 0.3, 0.6, 0.8])
def test_standardisation_t2(phi: float):
    """T2: Pooled mean of W^2 for k in {8, 32, 128, 512}, t>=m0, lies in [0.93, 1.10]."""
    n_streams = 20
    T = 5000
    m0 = 1000
    p = 50
    test_ks = [8, 32, 128, 512]

    # Generate streams
    rng_model = np.random.default_rng(np.random.SeedSequence([2026, int(phi * 10), 9173]))
    A = make_var_matrix(p, phi, "isotropic", rng_model)
    sigma_pre = make_innovation_covariance(p, 0.25)
    gamma_pre = _stationary_covariance(A, sigma_pre, phi)

    W2_accum: dict[int, list[float]] = {k: [] for k in test_ks}

    for i in range(n_streams):
        rng = np.random.default_rng(np.random.SeedSequence([2026, int(phi * 10), i, 2718]))
        X, _ = generate_one_stream(
            T=T, p=p, A=A, phi=phi,
            sigma_pre=sigma_pre, gamma_pre=gamma_pre,
            rng=rng, change_type="none",
            cp_fraction=0.5, cp_index=None,
            sparsity=0.1, effect=0.5,
            covariance_multiplier=1.5,
        )
        fit = fit_training(X[:m0], m0=m0, method="proposed", lrv_method="prewhiten")
        _, _, W_all = compute_statistics_vectorised(X, fit)

        for k in test_ks:
            ki = fit.window_grid.index(k)
            W_k = W_all[m0 + k - 1 :, ki, :]  # shape: (T - m0 - k + 1, p)
            valid_W = W_k[~np.isnan(W_k)]
            W2_accum[k].append(float(np.mean(valid_W ** 2)))

    for k in test_ks:
        pooled_w2 = float(np.mean(W2_accum[k]))
        print(f"[phi={phi}, k={k:3d}] Pooled mean W^2 = {pooled_w2:.4f}")
        assert 0.93 <= pooled_w2 <= 1.10, (
            f"Pooled mean W^2 for phi={phi}, k={k} is {pooled_w2:.4f}, not in [0.93, 1.10]"
        )
