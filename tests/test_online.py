"""Tests for OnlineDetector (T4 and T5).

T4 Batch vs online: for 5 streams the OnlineDetector alarm time, tail length and nu_hat
    equal the batch scan results, and the standardised statistics agree to relative tolerance 1e-9.
T5 Constant memory: tracemalloc peak of OnlineDetector after 20,000 and after 200,000 updates
    differs by <5%.
"""
from __future__ import annotations

import gc
import tracemalloc
import numpy as np
import pytest

from generate_synthetic_streams_fast import (
    _stationary_covariance,
    generate_one_stream,
    make_innovation_covariance,
    make_var_matrix,
)
from ocdmix.fit import fit_training
from ocdmix.localize import localize_at_alarm
from ocdmix.online import OnlineDetector
from ocdmix.scan import compute_base_A, compute_statistics_vectorised, detect_alarm


def _create_stream(phi: float, T: int, p: int, change: bool, seed: int = 123) -> np.ndarray:
    rng_model = np.random.default_rng(np.random.SeedSequence([seed, 0, 9173]))
    A = make_var_matrix(p, phi, "isotropic", rng_model)
    sigma_pre = make_innovation_covariance(p, 0.25)
    gamma_pre = _stationary_covariance(A, sigma_pre, phi)

    rng = np.random.default_rng(np.random.SeedSequence([seed, 0, 1, 2718]))
    X, _ = generate_one_stream(
        T=T, p=p, A=A, phi=phi,
        sigma_pre=sigma_pre, gamma_pre=gamma_pre,
        rng=rng,
        change_type="mean" if change else "none",
        cp_fraction=0.5,
        cp_index=2500 if change else None,
        sparsity=0.1,
        effect=1.5 if change else 0.5,
        covariance_multiplier=1.5,
    )
    return X


def test_batch_vs_online_t4():
    """T4: OnlineDetector alarm time, tail length (t_N), and nu_hat equal batch scan results,
    and standardised statistics agree to relative tolerance 1e-9 for 5 streams."""
    p = 50
    T = 4000
    m0 = 1000
    n_streams = 5
    c_A = 1.05
    c_Q = 3.5

    for s in range(n_streams):
        # Mix of null and changed streams
        change = (s % 2 == 1)
        X = _create_stream(phi=0.3, T=T, p=p, change=change, seed=1000 + s)
        fit = fit_training(X[:m0], m0=m0, method="proposed", lrv_method="prewhiten")

        # 1. Batch scan
        A_batch, Q_batch, W_batch = compute_statistics_vectorised(X, fit)
        alarm_batch = detect_alarm(A_batch, Q_batch, c_A, c_Q, fit)

        if alarm_batch is not None:
            base_A = compute_base_A(p, len(fit.window_grid))
            nu_hat_batch, CI_batch, supp_batch = localize_at_alarm(
                N=alarm_batch.N,
                t_N=alarm_batch.t_N,
                X=X,
                fit=fit,
                tau_A=c_A * base_A,
                tau_Q=c_Q,
                c_A=c_A,
                c_Q=c_Q,
                ci_a=1.0,
            )
        else:
            nu_hat_batch = None

        # 2. Online detector: feed observation by observation
        detector = OnlineDetector(fit, c_A=c_A, c_Q=c_Q, ci_a=1.0)
        alarm_online = None

        # Also check statistics agreement at each step up to alarm or T
        for t in range(T):
            # Check statistics at time t before update if t >= m0
            res = detector.update(X[t])
            if t >= m0:
                for ki, k in enumerate(fit.window_grid):
                    if t >= m0 + k - 1:
                        A_k_on, Q_k_on = detector.get_statistic_at(k)
                        A_k_ba = A_batch[t, ki]
                        Q_k_ba = Q_batch[t, ki]

                        np.testing.assert_allclose(
                            A_k_on, A_k_ba, rtol=1e-9, atol=1e-12,
                            err_msg=f"A mismatch at stream {s}, t={t}, k={k}",
                        )
                        np.testing.assert_allclose(
                            Q_k_on, Q_k_ba, rtol=1e-9, atol=1e-12,
                            err_msg=f"Q mismatch at stream {s}, t={t}, k={k}",
                        )

            if res is not None:
                alarm_online = res
                break

        # Compare alarm results
        if alarm_batch is None:
            assert alarm_online is None, f"Stream {s}: batch had no alarm, online alarmed at {alarm_online.N}"
        else:
            assert alarm_online is not None, f"Stream {s}: batch alarmed at {alarm_batch.N}, online had no alarm"
            assert alarm_online.N == alarm_batch.N, (
                f"Stream {s}: Alarm time N mismatch: online {alarm_online.N} vs batch {alarm_batch.N}"
            )
            assert alarm_online.t_N == alarm_batch.t_N, (
                f"Stream {s}: Tail length t_N mismatch: online {alarm_online.t_N} vs batch {alarm_batch.t_N}"
            )
            assert alarm_online.nu_hat == nu_hat_batch, (
                f"Stream {s}: nu_hat mismatch: online {alarm_online.nu_hat} vs batch {nu_hat_batch}"
            )


def test_constant_memory_t5():
    """T5: tracemalloc peak of OnlineDetector after 20,000 and after 200,000 updates differs by <5%."""
    p = 50
    m0 = 1000
    # Create a synthetic fit object
    X_train = np.random.default_rng(42).standard_normal((m0, p))
    fit = fit_training(X_train, m0=m0, method="proposed")

    # High thresholds so detector never alarms
    detector = OnlineDetector(fit, c_A=100.0, c_Q=100.0)

    # Use a fixed random generator for data
    rng = np.random.default_rng(2026)

    # Run 20,000 updates
    gc.collect()
    tracemalloc.start()
    for _ in range(20000):
        x = rng.standard_normal(p)
        detector.update(x)
    gc.collect()
    _, peak_20k = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Continue running up to 200,000 updates (180,000 more)
    gc.collect()
    tracemalloc.start()
    for _ in range(180000):
        x = rng.standard_normal(p)
        detector.update(x)
    gc.collect()
    _, peak_200k = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # The memory used during the second 180k chunk should not exceed the first chunk by >= 5%
    rel_diff = abs(peak_200k - peak_20k) / peak_20k
    print(f"\nPeak memory after 20k steps : {peak_20k / 1024:.2f} KiB")
    print(f"Peak memory after 200k steps: {peak_200k / 1024:.2f} KiB")
    print(f"Relative difference: {rel_diff * 100:.2f}%")

    assert rel_diff < 0.05, f"Peak memory grew by {rel_diff * 100:.2f}% (exceeds 5%)"
