from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.stats import norm

from .fit import FitResult


def localize_at_alarm(
    N: int,
    t_N: int,
    X: np.ndarray,
    fit: FitResult,
    tau_A: float,
    tau_Q: float,
    c_A: float,
    c_Q: float,
    ci_a: float = 0.0,
    statistic: str = "both",
    m_min: Optional[int] = None,
    z: Optional[float] = None,
) -> tuple[int, tuple[int, int], list[int]]:
    """Localise change-point from full stream at alarm time."""
    X = np.asarray(X, dtype=np.float64)
    p = fit.p
    m0 = fit.m0
    Xc = X - fit.mu

    m_max = min(2 * t_N, N - m0 + 1)
    if m_min is None:
        m_min = min(fit.window_grid) if (hasattr(fit, "window_grid") and len(fit.window_grid) > 0) else 8

    if m_max < m_min:
        m_star = t_N
        nu_hat = N - m_star + 1
    else:
        best_R = -np.inf
        m_star = m_min

        for m in range(m_min, m_max + 1):
            window_sum = np.sum(Xc[N - m + 1: N + 1], axis=0)

            if m < fit.s2_table.shape[0]:
                s2_m = fit.s2_table[m]
            else:
                s2_m = fit.s2_table[-1]

            sqrt_s2_m = np.sqrt(np.maximum(s2_m, 1e-16))
            W = window_sum / sqrt_s2_m

            A_m = np.max(np.abs(W))
            Q_m = (np.sum(W ** 2) - p) / np.sqrt(2.0 * p)

            if statistic in ("linf_only", "l_infinity"):
                R_m = A_m / tau_A if (tau_A is not None and not np.isinf(tau_A) and tau_A > 0) else A_m
            elif statistic == "q_only":
                R_m = Q_m / tau_Q if (tau_Q is not None and not np.isinf(tau_Q) and tau_Q > 0) else Q_m
            else:
                val_A = A_m / tau_A if (tau_A is not None and not np.isinf(tau_A) and tau_A > 0) else -np.inf
                val_Q = Q_m / tau_Q if (tau_Q is not None and not np.isinf(tau_Q) and tau_Q > 0) else -np.inf
                R_m = max(val_A, val_Q)

            if R_m > best_R:
                best_R = R_m
                m_star = m

        nu_hat = N - m_star + 1

    if ci_a > 0:
        half_width = math.ceil(ci_a * m_star)
        ci_lo = max(m0, nu_hat - half_width)
        ci_hi = min(N, nu_hat + half_width)
        CI = (ci_lo, ci_hi)
    else:
        CI = (nu_hat, nu_hat)

    window_sum = np.sum(Xc[N - m_star + 1: N + 1], axis=0)
    if m_star < fit.s2_table.shape[0]:
        s2_m = fit.s2_table[m_star]
    else:
        s2_m = fit.s2_table[-1]
    sqrt_s2_m = np.sqrt(np.maximum(s2_m, 1e-16))
    W_star = window_sum / sqrt_s2_m
    z_thresh = z if z is not None else norm.ppf(1.0 - 0.05 / (2.0 * p))
    support = sorted(int(j) for j in range(p) if abs(W_star[j]) > z_thresh)

    return nu_hat, CI, support


def localize_from_buffer(
    N: int,
    t_N: int,
    m0: int,
    obs_buf: np.ndarray,
    obs_head: int,
    obs_buf_size: int,
    fit: FitResult,
    tau_A: float,
    tau_Q: float,
    c_A: float,
    c_Q: float,
    ci_a: float = 0.0,
    statistic: str = "both",
    m_min: Optional[int] = None,
    z: Optional[float] = None,
) -> tuple[Optional[int], Optional[tuple[int, int]], Optional[list[int]]]:
    """Localise from the observation ring buffer (for OnlineDetector).

    The ring buffer must cover length at least 2*kmax because the post-alarm
    local refinement window m ranges up to min(2*t_N, N - m0 + 1), where
    t_N can be as large as kmax. Thus up to 2*kmax past observations are required.
    """
    p = fit.p

    m_max = min(2 * t_N, N - m0 + 1)
    if m_min is None:
        m_min = min(fit.window_grid) if (hasattr(fit, "window_grid") and len(fit.window_grid) > 0) else 8

    if m_max < m_min or m_max > obs_buf_size:
        m_star = t_N
        nu_hat = N - m_star + 1
    else:
        indices = (obs_head - m_max + np.arange(m_max)) % obs_buf_size
        recent_obs = obs_buf[indices]

        # rev_cumsum[m-1] holds the sum of the last m observations
        rev_cumsum = np.cumsum(recent_obs[::-1], axis=0)

        best_R = -np.inf
        m_star = m_min

        for m in range(m_min, m_max + 1):
            window_sum = rev_cumsum[m - 1]
            s2_m = fit.s2_table[m] if m < fit.s2_table.shape[0] else fit.s2_table[-1]
            sqrt_s2_m = np.sqrt(np.maximum(s2_m, 1e-16))
            W = window_sum / sqrt_s2_m

            A_m = np.max(np.abs(W))
            Q_m = (np.sum(W ** 2) - p) / np.sqrt(2.0 * p)

            if statistic in ("linf_only", "l_infinity"):
                R_m = A_m / tau_A if (tau_A is not None and not np.isinf(tau_A) and tau_A > 0) else A_m
            elif statistic == "q_only":
                R_m = Q_m / tau_Q if (tau_Q is not None and not np.isinf(tau_Q) and tau_Q > 0) else Q_m
            else:
                val_A = A_m / tau_A if (tau_A is not None and not np.isinf(tau_A) and tau_A > 0) else -np.inf
                val_Q = Q_m / tau_Q if (tau_Q is not None and not np.isinf(tau_Q) and tau_Q > 0) else -np.inf
                R_m = max(val_A, val_Q)

            if R_m > best_R:
                best_R = R_m
                m_star = m

        nu_hat = N - m_star + 1

    if ci_a > 0:
        half_width = math.ceil(ci_a * m_star)
        ci_lo = max(m0, nu_hat - half_width)
        ci_hi = min(N, nu_hat + half_width)
        CI = (ci_lo, ci_hi)
    else:
        CI = (nu_hat, nu_hat)

    if m_star <= obs_buf_size:
        indices = (obs_head - m_star + np.arange(m_star)) % obs_buf_size
        window_sum = np.sum(obs_buf[indices], axis=0)

        s2_m = fit.s2_table[m_star] if m_star < fit.s2_table.shape[0] else fit.s2_table[-1]
        sqrt_s2_m = np.sqrt(np.maximum(s2_m, 1e-16))
        W_star = window_sum / sqrt_s2_m
        z_thresh = z if z is not None else norm.ppf(1.0 - 0.05 / (2.0 * p))
        support = sorted(int(j) for j in range(p) if abs(W_star[j]) > z_thresh)
    else:
        support = []

    return nu_hat, CI, support


def make_scale_grid(b_min: float, p: int) -> np.ndarray:
    """Construct positive scale grid B ∪ B0 (Eq. 16).

    B0 = {b_min}, B = {2^{m/2} * b_min : m = 1, ..., M}
    where M = ceil(2 * log2(1 / b_min)).
    Returns sorted array of positive scales.
    """
    M = max(1, int(np.ceil(2 * np.log2(max(1.0 / b_min, 2.0)))))
    scales = [b_min]
    for m in range(1, M + 1):
        scales.append(2 ** (m / 2.0) * b_min)
    return np.unique(np.array(scales))


def _a_r_bar(r_bar: float) -> float:
    """Heuristic a(r̄) multiplier: a = 0.1222 + 1.1604·r̄.

    Used in legacy inference mode for empirical CI half-width.
    """
    return 0.1222 + 1.1604 * float(r_bar)


def localize_chen(
    N: int,
    m_star: int,
    X: np.ndarray,
    fit: FitResult,
    extra_obs: int = 20,
    alpha: float = 0.05,
    c_support: float = 0.35,
    c_report: Optional[float] = 0.80,
    b_min: float = 0.1,
    kmax: Optional[int] = None,
    use_residuals: bool = False,
    inference_mode: str = "legacy",
    c_CI: Optional[float] = None,
    c_rep: Optional[float] = None,
) -> dict:
    """Chen-Wang-Samworth post-alarm inference (Eq. 16-17).

    inference_mode=\"legacy\": original behaviour with a(r̄) CI half-width.
    inference_mode=\"new\": CWS Algorithm 1 compliant with anchor coordinate.
    """
    if inference_mode == "new":
        return _localize_chen_new(
            N=N, m_star=m_star, X=X, fit=fit,
            extra_obs=extra_obs, alpha=alpha,
            c_CI=(c_CI if c_CI is not None else c_support),
            c_rep=(c_rep if c_rep is not None else (c_report if c_report is not None else c_support)),
            b_min=b_min, kmax=kmax,
        )
    return _localize_chen_legacy(
        N=N, m_star=m_star, X=X, fit=fit,
        extra_obs=extra_obs, alpha=alpha,
        c_support=c_support, c_report=c_report,
        b_min=b_min, kmax=kmax, use_residuals=use_residuals,
    )


def _localize_chen_legacy(
    N: int,
    m_star: int,
    X: np.ndarray,
    fit: FitResult,
    extra_obs: int = 20,
    alpha: float = 0.05,
    c_support: float = 0.35,
    c_report: Optional[float] = 0.80,
    b_min: float = 0.1,
    kmax: Optional[int] = None,
    use_residuals: bool = False,
) -> dict:
    """Legacy localize_chen implementation.

    Maintained for comparison with earlier heuristic benchmarks.
    Computes standardised statistics, support recovery, tail lengths, and CI.
    """
    X = np.asarray(X, dtype=np.float64)
    p = fit.p
    m0 = fit.m0
    T = X.shape[0]
    if kmax is None:
        kmax = fit.kmax
    h = m_star

    N_end = min(N + extra_obs, T - 1)
    actual_ell = N_end - N

    if use_residuals:
        Xc = X - fit.mu
        e = Xc[1:] - fit.r[np.newaxis, :] * Xc[:-1]
        omega_std = np.sqrt(np.maximum(fit.omega, 1e-16))
        zeta = np.zeros((T, p), dtype=np.float64)
        zeta[1:] = e / omega_std[np.newaxis, :]
    else:
        sigma_hat = np.sqrt(np.maximum(fit.sigma2, 1e-16))
        zeta = (X - fit.mu) / sigma_hat[np.newaxis, :]

    # E_j = Σ_{i=N-h+1}^{N+ℓ} ζ_{i,j} / sqrt(h + ℓ)
    start_E = max(N - h + 1, 0)
    end_E = N + actual_ell + 1
    E = np.sum(zeta[start_E:end_E], axis=0) / np.sqrt(h + actual_ell)

    d1 = c_support * np.sqrt(np.log(max(p / alpha, 1.0)))
    sqrt_ml = np.sqrt(m_star + actual_ell)
    support_mask_ci = np.abs(E) - b_min * sqrt_ml >= d1
    support_ci = sorted(int(j) for j in range(p) if support_mask_ci[j])

    if c_report is not None:
        d1_report = c_report * np.sqrt(np.log(max(p / alpha, 1.0)))
        support_mask_report = np.abs(E) - b_min * sqrt_ml >= d1_report
        support_report = sorted(int(j) for j in range(p) if support_mask_report[j])
    else:
        support_report = support_ci

    support = support_ci

    B = make_scale_grid(b_min, p)
    b_tilde = np.zeros(p, dtype=np.float64)
    for j in support:
        feasible = B[np.abs(E[j]) - B * sqrt_ml >= d1]
        if len(feasible) > 0:
            b_tilde[j] = np.sign(E[j]) * feasible[-1]
        else:
            b_tilde[j] = np.sign(E[j]) * b_min

    t_j = np.zeros(p, dtype=int)
    max_h = min(kmax, N - m0)
    for j in support:
        bj = b_tilde[j]
        best_val = 0.0
        best_h = 0
        cumsum = 0.0
        for hh in range(1, max_h + 1):
            idx = N - hh + 1
            if idx < 0:
                break
            cumsum += zeta[idx, j]
            val = bj * cumsum - hh * bj * bj / 2.0
            if val > best_val:
                best_val = val
                best_h = hh
        t_j[j] = best_h

    if len(support) > 0:
        ci_offsets = []
        for j in support:
            bj2 = b_tilde[j] ** 2
            offset = float(t_j[j]) + (4.0 * d1 ** 2 / bj2 if bj2 > 1e-16 else 0.0)
            ci_offsets.append(offset)
        ci_lo = max(int(np.ceil(N - min(ci_offsets))), m0)
        CI = (ci_lo, N)
    else:
        CI = (N, N)

    nu_hat = N - m_star + 1

    return {
        "support": support,
        "support_CI": support_ci,
        "support_report": support_report,
        "b_tilde": b_tilde,
        "t_j": t_j,
        "CI": CI,
        "E": E,
        "d1": d1,
        "nu_hat": nu_hat,
        "m_star": m_star,
        "extra_obs_actual": actual_ell,
        "b_min": b_min,
        "c_support": c_support,
        "c_report": c_report,
        "scale_grid": B,
        "use_residuals": use_residuals,
        "inference_mode": "legacy",
    }


def _localize_chen_new(
    N: int,
    m_star: int,
    X: np.ndarray,
    fit: FitResult,
    extra_obs: int = 0,
    alpha: float = 0.05,
    c_CI: float = 0.5,
    c_rep: float = 0.8,
    b_min: float = 0.1,
    kmax: Optional[int] = None,
) -> dict:
    """Chen, Wang & Samworth (2024) Algorithm 1 compliant post-alarm inference.

    Procedure overview:
    1. E_j = Σ_{i=N-m*+1}^{N+ℓ} (X_{ij} − µ̂_j) / ŝ_{k,j}
       where k = m* + ℓ and ŝ_{k,j} = sqrt(s2_table[k, j]).
    2. Anchor coordinate ĵ = argmax_j |W_{N,m*,j}|.
    3. Ŝ = {j ≠ ĵ : |E_j| − b_min·√k ≥ d1_CI}, where d1_CI = c_CI·√log(p/α).
    4. Reported support = Ŝ ∪ {ĵ}.
    5. Multi-scale effect magnitude: b̃_j = sgn(E_j) · max{b ∈ B∪B0 : |E_j| − b·√k ≥ d1_CI}.
    6. Tail lengths t_j searched over h in [0 .. N-m0], truncated to kmax.
    7. CI = [max(N − min_{j∈Ŝ}(t_j + 4·d1_CI²/b̃_j²), m0), N].
    """
    X = np.asarray(X, dtype=np.float64)
    p = fit.p
    m0 = fit.m0
    T = X.shape[0]
    if kmax is None:
        kmax = fit.kmax

    N_end = min(N + extra_obs, T - 1)
    actual_ell = N_end - N
    k = m_star + actual_ell

    Xc = X - fit.mu
    start_mstar = max(N - m_star + 1, 0)
    window_sum_mstar = np.sum(Xc[start_mstar: N + 1], axis=0)
    if m_star < fit.s2_table.shape[0]:
        s2_mstar = fit.s2_table[m_star]
    else:
        s2_mstar = fit.s2_table[-1]
    sqrt_s2_mstar = np.sqrt(np.maximum(s2_mstar, 1e-16))
    W_mstar = window_sum_mstar / sqrt_s2_mstar
    j_hat = int(np.argmax(np.abs(W_mstar)))

    if k < fit.s2_table.shape[0]:
        s2_k = fit.s2_table[k]
    else:
        s2_k = fit.s2_table[-1]
    s_hat_k = np.sqrt(np.maximum(s2_k, 1e-16))

    start_E = max(N - m_star + 1, 0)
    end_E = N + actual_ell + 1
    window_sum_k = np.sum(Xc[start_E: end_E], axis=0)
    E = window_sum_k / s_hat_k

    d1_CI = c_CI * np.sqrt(np.log(max(p / alpha, 1.0)))
    sqrt_k = np.sqrt(float(k))
    support_ci_mask = np.zeros(p, dtype=bool)
    for j in range(p):
        if j != j_hat:
            support_ci_mask[j] = (np.abs(E[j]) - b_min * sqrt_k >= d1_CI)
    support_ci = sorted(int(j) for j in range(p) if support_ci_mask[j])

    d1_rep = c_rep * np.sqrt(np.log(max(p / alpha, 1.0)))
    support_rep_mask = np.zeros(p, dtype=bool)
    for j in range(p):
        if j != j_hat:
            support_rep_mask[j] = (np.abs(E[j]) - b_min * sqrt_k >= d1_rep)
    support_rep_base = sorted(int(j) for j in range(p) if support_rep_mask[j])
    support_report = sorted(set(support_rep_base) | {j_hat})

    support = support_ci

    B = make_scale_grid(b_min, p)
    b_tilde = np.zeros(p, dtype=np.float64)
    for j in support:
        feasible = B[np.abs(E[j]) - B * sqrt_k >= d1_CI]
        if len(feasible) > 0:
            b_tilde[j] = np.sign(E[j]) * feasible[-1]
        else:
            b_tilde[j] = np.sign(E[j]) * b_min

    sigma_hat = np.sqrt(np.maximum(fit.sigma2, 1e-16))
    zeta = (X - fit.mu) / sigma_hat[np.newaxis, :]

    full_max_h = N - m0
    tail_cap_flag = full_max_h > kmax
    max_h = min(kmax, full_max_h) if kmax is not None else full_max_h

    t_j = np.zeros(p, dtype=int)
    for j in support:
        bj = b_tilde[j]
        best_val = 0.0
        best_h = 0
        cumsum = 0.0
        for hh in range(1, max_h + 1):
            idx = N - hh + 1
            if idx < 0:
                break
            cumsum += zeta[idx, j]
            val = bj * cumsum - hh * bj * bj / 2.0
            if val > best_val:
                best_val = val
                best_h = hh
        t_j[j] = best_h

    if len(support) > 0:
        ci_offsets = []
        for j in support:
            bj2 = b_tilde[j] ** 2
            offset = float(t_j[j]) + (4.0 * d1_CI ** 2 / bj2 if bj2 > 1e-16 else 0.0)
            ci_offsets.append(offset)
        ci_lo = max(int(np.ceil(N - min(ci_offsets))), m0)
        CI = (ci_lo, N)
    else:
        # Ŝ empty: fall back to point estimate using anchor only
        CI = (N, N)

    nu_hat = N - m_star + 1

    return {
        "support": support,
        "support_CI": support_ci,
        "support_report": support_report,
        "j_hat": j_hat,
        "b_tilde": b_tilde,
        "t_j": t_j,
        "CI": CI,
        "E": E,
        "d1": d1_CI,
        "nu_hat": nu_hat,
        "m_star": m_star,
        "k": k,
        "extra_obs_actual": actual_ell,
        "b_min": b_min,
        "c_support": c_CI,
        "c_report": c_rep,
        "c_CI": c_CI,
        "c_rep": c_rep,
        "scale_grid": B,
        "tail_cap_flag": tail_cap_flag,
        "use_residuals": False,
        "inference_mode": "new",
    }
