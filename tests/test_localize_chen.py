"""Unit tests for Chen-Wang-Samworth (2024) post-alarm inference (Eq. 16-17).

F6:
1. Symmetry test: synthetic shift with b > 0 and b < 0 must give the same expected t_j.
2. Mathematical verification of scale grid B, support S, b_tilde_j, t_j, and CI.
3. Prewhitened residual variant comparison.
"""
from __future__ import annotations

import numpy as np
import pytest

from ocdmix.fit import FitResult, fit_training
from ocdmix.localize import localize_chen, make_scale_grid


def test_make_scale_grid():
    """Verify scale grid construction follows Eq. 16."""
    b_min = 0.1
    p = 50
    B = make_scale_grid(b_min, p)
    assert len(B) > 1
    assert B[0] == pytest.approx(b_min)
    assert np.all(np.diff(B) > 0)  # strictly increasing
    # Verify dyadic geometric factor 2^{1/2}
    assert B[1] / B[0] == pytest.approx(np.sqrt(2.0))


def test_symmetry_positive_negative_shift():
    """F6: synthetic shift with b > 0 and b < 0 must give the same expected t_j.

    By symmetry of the signed CUSUM objective:
        argmax_h { b_tilde_j * sum(zeta) - h * b_tilde_j^2 / 2 }
    A shift of +b produces positive E_j and positive b_tilde_j, while
    a shift of -b produces negative E_j and negative b_tilde_j.
    For antithetic noise realizations (-Z vs +Z), the objective values are
    identically equal path-by-path. Across independent streams, the expected
    values E[t_j | +b] and E[t_j | -b] are statistically equal.
    """
    p = 10
    T = 2000
    m0 = 500
    cp = 1000
    b_mag = 1.5
    coord = 2
    n_reps = 100

    # 1. Exact antithetic symmetry path-by-path
    rng = np.random.default_rng(20261021)
    for _ in range(10):
        Z = rng.standard_normal((T, p))
        # Stream with +b shift
        X_pos = Z.copy()
        X_pos[cp:, coord] += b_mag

        # Stream with -b shift and mirrored noise -Z
        X_neg = -Z.copy()
        X_neg[cp:, coord] -= b_mag

        # Fit on training segment
        fit_pos = fit_training(X_pos[:m0], m0=m0, method="proposed")
        fit_neg = fit_training(X_neg[:m0], m0=m0, method="proposed")

        alarm_N = 1050
        m_star = 50

        res_pos = localize_chen(alarm_N, m_star, X_pos, fit_pos, extra_obs=20, c_support=0.5, b_min=0.1)
        res_neg = localize_chen(alarm_N, m_star, X_neg, fit_neg, extra_obs=20, c_support=0.5, b_min=0.1)

        # Coordinate must be selected in support for both
        assert coord in res_pos["support"], "Positive shift coordinate must be in support"
        assert coord in res_neg["support"], "Negative shift coordinate must be in support"

        # b_tilde must be opposite signs with identical magnitude
        assert res_pos["b_tilde"][coord] == pytest.approx(-res_neg["b_tilde"][coord], rel=1e-5)

        # t_j must be exactly identical
        assert res_pos["t_j"][coord] == res_neg["t_j"][coord], (
            f"Path-by-path antithetic symmetry failed: {res_pos['t_j'][coord]} != {res_neg['t_j'][coord]}"
        )

    # 2. Monte Carlo expected value equality across independent streams
    t_pos_list = []
    t_neg_list = []
    rng_mc = np.random.default_rng(9999)

    alarm_N = 1050
    m_star = 50

    for _ in range(n_reps):
        Z_pos = rng_mc.standard_normal((T, p))
        X_pos = Z_pos
        X_pos[cp:, coord] += b_mag
        fit_p = fit_training(X_pos[:m0], m0=m0, method="proposed")
        r_p = localize_chen(alarm_N, m_star, X_pos, fit_p, extra_obs=20, c_support=0.5, b_min=0.1)
        if coord in r_p["support"]:
            t_pos_list.append(r_p["t_j"][coord])

        Z_neg = rng_mc.standard_normal((T, p))
        X_neg = Z_neg
        X_neg[cp:, coord] -= b_mag
        fit_n = fit_training(X_neg[:m0], m0=m0, method="proposed")
        r_n = localize_chen(alarm_N, m_star, X_neg, fit_n, extra_obs=20, c_support=0.5, b_min=0.1)
        if coord in r_n["support"]:
            t_neg_list.append(r_n["t_j"][coord])

    mean_pos = float(np.mean(t_pos_list))
    mean_neg = float(np.mean(t_neg_list))
    # Two-sample t-test / standard error tolerance
    pooled_se = np.sqrt(np.var(t_pos_list)/len(t_pos_list) + np.var(t_neg_list)/len(t_neg_list))
    diff = abs(mean_pos - mean_neg)
    assert diff <= 3.0 * pooled_se, (
        f"Expected t_j differs significantly: E[t_j|+b]={mean_pos:.2f}, E[t_j|-b]={mean_neg:.2f}, diff={diff:.2f}, 3*SE={3*pooled_se:.2f}"
    )


def test_support_and_confidence_interval():
    """Verify that CI covers true change point and support identifies changed coordinates."""
    p = 20
    T = 2500
    m0 = 500
    cp = 1500
    rng = np.random.default_rng(42)

    X = rng.standard_normal((T, p))
    # Shift coordinates 1, 4, 7 by +1.5
    changed_coords = [1, 4, 7]
    for c in changed_coords:
        X[cp:, c] += 1.5

    fit = fit_training(X[:m0], m0=m0, method="proposed")
    alarm_N = 1530  # 30 steps after change
    m_star = 30

    res = localize_chen(alarm_N, m_star, X, fit, extra_obs=20, c_support=0.5, b_min=0.1)
    # Check that changed coordinates are recovered
    for c in changed_coords:
        assert c in res["support"]

    # CI must cover true change point 1500
    ci_lo, ci_hi = res["CI"]
    assert ci_lo <= cp <= ci_hi, f"CI [{ci_lo}, {ci_hi}] does not cover true change point {cp}"
    assert ci_hi == alarm_N


def test_prewhitened_residual_variant():
    """Verify prewhitened residual variant executes and produces valid results."""
    p = 10
    T = 2000
    m0 = 500
    cp = 1200
    rng = np.random.default_rng(123)

    X = rng.standard_normal((T, p))
    X[cp:, 0] += 2.0

    fit = fit_training(X[:m0], m0=m0, method="proposed")
    alarm_N = 1225
    m_star = 25

    res_zeta = localize_chen(alarm_N, m_star, X, fit, extra_obs=20, c_support=0.5, use_residuals=False)
    res_resid = localize_chen(alarm_N, m_star, X, fit, extra_obs=20, c_support=0.5, use_residuals=True)

    assert 0 in res_zeta["support"]
    assert 0 in res_resid["support"]
    assert res_zeta["CI"][0] <= cp <= res_zeta["CI"][1]
    assert res_resid["CI"][0] <= cp <= res_resid["CI"][1]


def test_decoupled_support_ci_and_report():
    """Verify decoupling of support: S_CI (high recall) and S_report (high precision)."""
    p = 20
    T = 2500
    m0 = 500
    cp = 1500
    rng = np.random.default_rng(2026)

    X = rng.standard_normal((T, p))
    changed_coords = [2, 5, 8]
    for c in changed_coords:
        X[cp:, c] += 1.5

    fit = fit_training(X[:m0], m0=m0, method="proposed")
    alarm_N = 1530
    m_star = 30

    res = localize_chen(
        alarm_N, m_star, X, fit, extra_obs=20,
        c_support=0.35, c_report=0.80, b_min=0.1
    )

    assert "support_CI" in res
    assert "support_report" in res

    # S_CI has lower threshold => S_report must be a subset of S_CI
    assert set(res["support_report"]).issubset(set(res["support_CI"]))

    # Changed coordinates should be captured in both
    for c in changed_coords:
        assert c in res["support_CI"]
        assert c in res["support_report"]
