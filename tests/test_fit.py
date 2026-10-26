"""Tests for training fit (Task 1 / T1).

T1 Acceptance Criterion:
  isotropic VAR, phi in {0.3, 0.6, 0.8}, 40 null streams from the fast generator, m0=1000:
  the mean over coordinates and streams of sigma2_j is within 5% of 1/(1-phi)^2 for lrv_method="prewhiten".
  (Bartlett option printed, not asserted).
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


def _generate_null_training_data(
    n_streams: int, m0: int, p: int, phi: float, seed: int = 20261014
) -> np.ndarray:
    """Generate n_streams x m0 x p null data."""
    rng_model = np.random.default_rng(np.random.SeedSequence([seed, 0, 9173]))
    A = make_var_matrix(p, phi, "isotropic", rng_model)
    sigma_pre = make_innovation_covariance(p, 0.25)
    gamma_pre = _stationary_covariance(A, sigma_pre, phi)

    data = np.empty((n_streams, m0, p), dtype=np.float64)
    for i in range(n_streams):
        stream_seed = np.random.SeedSequence([seed, 0, i, 2718])
        rng = np.random.default_rng(stream_seed)
        X, _ = generate_one_stream(
            T=m0,
            p=p,
            A=A,
            phi=phi,
            sigma_pre=sigma_pre,
            gamma_pre=gamma_pre,
            rng=rng,
            change_type="none",
            cp_fraction=0.5,
            cp_index=None,
            sparsity=0.1,
            effect=0.5,
            covariance_multiplier=1.5,
        )
        data[i] = X
    return data


@pytest.mark.parametrize("phi", [0.3, 0.6, 0.8])
def test_lrv_accuracy_t1(phi: float):
    """T1: LRV accuracy on isotropic VAR with m0=1000, 40 null streams."""
    n_streams = 40
    m0 = 1000
    p = 50
    expected_lrv = 1.0 / ((1.0 - phi) ** 2)

    data = _generate_null_training_data(n_streams=n_streams, m0=m0, p=p, phi=phi, seed=20261014)

    sigma2_prewhiten = []
    sigma2_bartlett = []

    for i in range(n_streams):
        fit_pw = fit_training(data[i], m0=m0, method="proposed", lrv_method="prewhiten")
        sigma2_prewhiten.append(fit_pw.sigma2)

        fit_bt = fit_training(data[i], m0=m0, method="proposed", lrv_method="bartlett")
        sigma2_bartlett.append(fit_bt.sigma2)

    mean_sigma2_pw = float(np.mean(sigma2_prewhiten))
    bias_pw = (mean_sigma2_pw - expected_lrv) / expected_lrv

    mean_sigma2_bt = float(np.mean(sigma2_bartlett))
    bias_bt = (mean_sigma2_bt - expected_lrv) / expected_lrv

    print(f"\n[phi={phi}] Target LRV = {expected_lrv:.4f}")
    print(f"  Prewhiten: mean={mean_sigma2_pw:.4f}, bias={bias_pw*100:+.2f}%")
    print(f"  Bartlett : mean={mean_sigma2_bt:.4f}, bias={bias_bt*100:+.2f}%")

    # Assert prewhiten within 5%
    assert abs(bias_pw) < 0.05, f"Prewhiten LRV bias {bias_pw*100:.2f}% exceeds 5% for phi={phi}"
