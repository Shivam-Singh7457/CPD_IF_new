"""Vectorised batch scan of one stream (E7-E9)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .fit import FitResult


@dataclass
class AlarmResult:
    """Result of a batch scan alarm detection."""
    N: int                  # alarm time (zero-based index)
    t_N: int                # largest violating window length at alarm time
    nu_dyadic: int          # N - t_N + 1


def compute_statistics(
    X: np.ndarray,
    fit: FitResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute monitoring statistics A[t,k] and Q[t,k] for one stream.

    Parameters
    ----------
    X : (T, p) full stream
    fit : FitResult from training

    Returns
    -------
    A : (T, len(K)) – max statistic; NaN where window invalid
    Q : (T, len(K)) – quadratic statistic; NaN where window invalid
    W_all : (T, len(K), p) – standardised window sums; NaN where invalid
    """
    X = np.asarray(X, dtype=np.float64)
    T, p = X.shape
    m0 = fit.m0
    K = fit.window_grid

    # Centre data using training mean
    Xc = X - fit.mu  # (T, p)

    # E7: cumulative sums C[t] = sum_{u=m0..t} Xc[u]
    # We build C starting from index m0, so C_raw[i] = sum_{u=m0..m0+i-1} Xc[u]
    # C_raw[0] = 0 (before any monitoring obs)
    # C_raw[t - m0 + 1] = sum_{u=m0..t} Xc[u]  for t >= m0
    monitoring = Xc[m0:]  # (T - m0, p)
    n_mon = monitoring.shape[0]
    C_raw = np.zeros((n_mon + 1, p), dtype=np.float64)
    np.cumsum(monitoring, axis=0, out=C_raw[1:])

    n_k = len(K)
    A = np.full((T, n_k), np.nan, dtype=np.float64)
    Q = np.full((T, n_k), np.nan, dtype=np.float64)
    W_all = np.full((T, n_k, p), np.nan, dtype=np.float64)

    for ki, k in enumerate(K):
        s2k = fit.s2_grid[k]  # (p,)
        sqrt_s2k = np.sqrt(s2k)  # (p,)

        # Valid range: t >= m0, t - k + 1 >= m0  =>  t >= m0 + k - 1
        t_start = m0 + k - 1
        if t_start >= T:
            continue

        # For t in [t_start, T-1]:
        # index into C_raw: C[t] = C_raw[t - m0 + 1], C[t-k] = C_raw[t - k - m0 + 1]
        for t in range(t_start, T):
            ci_t = t - m0 + 1
            ci_tk = t - k - m0 + 1
            window_sum = C_raw[ci_t] - C_raw[ci_tk]  # (p,)
            W = window_sum / sqrt_s2k  # (p,)
            W_all[t, ki] = W
            A[t, ki] = np.max(np.abs(W))
            Q[t, ki] = (np.sum(W ** 2) - p) / np.sqrt(2.0 * p)

    return A, Q, W_all


def compute_statistics_vectorised(
    X: np.ndarray,
    fit: FitResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fully vectorised version of compute_statistics (much faster).

    Returns same shapes as compute_statistics.
    """
    X = np.asarray(X, dtype=np.float64)
    T, p = X.shape
    m0 = fit.m0
    K = fit.window_grid

    Xc = X - fit.mu
    monitoring = Xc[m0:]
    n_mon = monitoring.shape[0]
    C_raw = np.zeros((n_mon + 1, p), dtype=np.float64)
    np.cumsum(monitoring, axis=0, out=C_raw[1:])

    n_k = len(K)
    A = np.full((T, n_k), np.nan, dtype=np.float64)
    Q = np.full((T, n_k), np.nan, dtype=np.float64)
    W_all = np.full((T, n_k, p), np.nan, dtype=np.float64)

    for ki, k in enumerate(K):
        s2k = fit.s2_grid[k]
        sqrt_s2k = np.sqrt(s2k)

        t_start = m0 + k - 1
        if t_start >= T:
            continue

        # Vectorised: all valid t at once
        # t ranges from t_start to T-1
        # C[t] = C_raw[t - m0 + 1], C[t-k] = C_raw[t - k - m0 + 1]
        t_indices = np.arange(t_start, T)
        ci_t = t_indices - m0 + 1     # indices into C_raw for C[t]
        ci_tk = t_indices - k - m0 + 1  # indices into C_raw for C[t-k]

        window_sums = C_raw[ci_t] - C_raw[ci_tk]  # (n_valid, p)
        W = window_sums / sqrt_s2k[np.newaxis, :]  # (n_valid, p)

        W_all[t_indices, ki, :] = W
        A[t_indices, ki] = np.max(np.abs(W), axis=1)
        Q[t_indices, ki] = (np.sum(W ** 2, axis=1) - p) / np.sqrt(2.0 * p)

    return A, Q, W_all


def detect_alarm(
    A: np.ndarray,
    Q: np.ndarray,
    c_A: float,
    c_Q: float,
    fit: FitResult,
    statistic: str = "both",
) -> Optional[AlarmResult]:
    """Find the first alarm time given thresholds.

    Parameters
    ----------
    A : (T, len(K)) max statistics
    Q : (T, len(K)) quadratic statistics
    c_A : threshold multiplier for A
    c_Q : threshold for Q
    fit : FitResult
    statistic : {"both", "linf_only", "l_infinity", "q_only"}

    Returns
    -------
    AlarmResult or None
    """
    T, n_k = A.shape
    K = fit.window_grid
    p = fit.p

    base_A = np.sqrt(2.0 * np.log(2.0 * p * len(K)))
    tau_A = c_A * base_A if (c_A is not None and not np.isinf(c_A)) else np.inf
    tau_Q = c_Q if (c_Q is not None and not np.isinf(c_Q)) else np.inf

    use_A = statistic in ("both", "linf_only", "l_infinity") and not np.isinf(tau_A)
    use_Q = statistic in ("both", "q_only") and not np.isinf(tau_Q)

    # Find first t where any k violates
    for t in range(T):
        violating_ks = []
        for ki, k in enumerate(K):
            if np.isnan(A[t, ki]):
                continue
            violates = False
            if use_A and A[t, ki] > tau_A:
                violates = True
            if use_Q and Q[t, ki] > tau_Q:
                violates = True
            if violates:
                violating_ks.append(k)

        if violating_ks:
            t_N = max(violating_ks)  # E11: largest violating window
            nu_dyadic = t - t_N + 1
            return AlarmResult(N=t, t_N=t_N, nu_dyadic=nu_dyadic)

    return None


def compute_base_A(p: int, n_K: int) -> float:
    """Compute base_A = sqrt(2 * ln(2 * p * |K|))."""
    return np.sqrt(2.0 * np.log(2.0 * p * n_K))


def scan_stream(
    X: np.ndarray,
    fit: FitResult,
    c_A: float,
    c_Q: float,
    statistic: str = "both",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[AlarmResult]]:
    """Full pipeline: compute statistics and detect alarm.

    Returns
    -------
    A, Q, W_all, alarm_result
    """
    A, Q, W_all = compute_statistics_vectorised(X, fit)
    alarm = detect_alarm(A, Q, c_A, c_Q, fit, statistic=statistic)
    return A, Q, W_all, alarm
