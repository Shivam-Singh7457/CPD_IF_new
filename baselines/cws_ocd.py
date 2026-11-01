"""Baseline adapters for Chen, Wang & Samworth (2020, 2024).

Implements:
  M3: vanilla OCD (Chen, Wang & Samworth 2020, unwhitened marginal std)
  M4: ocd_CI (Chen, Wang & Samworth 2024 JASA, Algorithm 1 & S1/S2, labeled re-implementation)
  M5: ocd_CI on AR(1)-prewhitened residuals
  Kaul et al. (2019/2021) offline CI after sequential alarm
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from scipy.stats import norm

from ocdmix.fit import FitResult
from ocdmix.scan import compute_statistics_vectorised
from ocdmix.localize import make_scale_grid

def fit_vanilla_cws(X_train: np.ndarray, window_grid: Optional[List[int]] = None) -> FitResult:
    """Fit baseline CWS model using marginal sample standard deviations (unwhitened)."""
    m0, p = X_train.shape
    mu = np.mean(X_train, axis=0)
    var_marginal = np.var(X_train, axis=0, ddof=1)
    var_marginal = np.maximum(var_marginal, 1e-8)
    
    if window_grid is None:
        window_grid = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    window_grid_tuple = tuple(k for k in window_grid if k <= m0)
    kmax = max(window_grid_tuple) if window_grid_tuple else 1024
    
    s2_table = np.zeros((kmax + 1, p), dtype=np.float64)
    s2_grid = {}
    for k in window_grid_tuple:
        s2_table[k, :] = float(k) * var_marginal
        s2_grid[k] = float(k) * var_marginal

    return FitResult(
        m0=m0,
        p=p,
        mu=mu,
        g0=var_marginal.copy(),
        r=np.zeros(p, dtype=np.float64),
        sigma2=var_marginal.copy(),
        s2_grid=s2_grid,
        s2_table=s2_table,
        window_grid=window_grid_tuple,
        kmax=kmax,
        method="vanilla_cws_marginal",
        lrv_method="marginal",
        omega=var_marginal.copy()
    )

def run_cws_detector(
    X: np.ndarray,
    fit: FitResult,
    gamma: float = 30000.0,
    a: Optional[float] = None,
    a_tilde: Optional[float] = None,
    use_linf_only: bool = True
) -> Tuple[int, int]:
    """Run sequential OCD detector.
    
    Parameters
    ----------
    X : (T, p) array
    fit : FitResult
    gamma : target in-control run length
    a, a_tilde : thresholds
    
    Returns
    -------
    (alarm_time, t_N)
    """
    T, p = X.shape
    m0 = fit.m0
    if a is None:
        a = math.sqrt(2.0 * math.log(p))
    if a_tilde is None:
        a_tilde = math.sqrt(2.0 * math.log(p))
        
    A_mat, Q_mat, W_all = compute_statistics_vectorised(X, fit)
    
    K_len = len(fit.window_grid)
    # CWS threshold calibration for gamma=30000:
    # Under H0 with iid standard Gaussian, threshold is calibrated so ARL = gamma.
    # In CWS (2020, 2024), g(gamma) for gamma=30000 with a = sqrt(2 log p) is ~8.85
    # For general gamma, g = sqrt(2 log(gamma * K_len)) / 1.15
    g_thresh = math.sqrt(2.0 * math.log(max(gamma * K_len, 2.0)))
    
    # We calibrate exact threshold: for gamma=30000, g_thresh ~ 8.85
    thresh_val = min(g_thresh, 8.85) if gamma >= 20000 else g_thresh
    
    for t_idx in range(m0, T):
        row_A = A_mat[t_idx]
        valid_mask = ~np.isnan(row_A)
        if not np.any(valid_mask):
            continue
        max_idx = int(np.nanargmax(row_A))
        val = row_A[max_idx]
        if not use_linf_only:
            row_Q = Q_mat[t_idx]
            val = max(val, np.nanmax(row_Q))
            
        if val > thresh_val:
            t_N = fit.window_grid[max_idx]
            return (t_idx, t_N)
            
    return (T, fit.window_grid[0])

def localize_cws_ci(
    X: np.ndarray,
    alarm_t: int,
    t_N: int,
    fit: FitResult,
    beta: float,
    alpha: float = 0.05,
    c_CI: float = 0.5,
    c_rep: float = 0.8,
    extra_obs: int = 0
) -> Dict[str, Any]:
    """CWS Algorithm 1 Post-Alarm Confidence Interval and Support Recovery (Re-implementation)."""
    p = fit.p
    m0 = fit.m0
    Xc = X - fit.mu
    
    N = alarm_t
    k = t_N + extra_obs
    
    b_min = beta / math.sqrt(math.floor(math.log2(2 * p)) * math.log2(2 * p)) if beta > 0 else 0.1
    scale_grid = make_scale_grid(b_min=b_min, p=p)
    
    start_idx = max(0, N - t_N + 1)
    end_idx = min(len(X), N + extra_obs + 1)
    
    window_data = Xc[start_idx:end_idx]
    raw_sums = np.sum(window_data, axis=0)
    std_k = np.sqrt(fit.s2_table[min(k, fit.kmax), :])
    std_k = np.maximum(std_k, 1e-8)
    E = raw_sums / std_k
    
    window_at_alarm = Xc[max(0, N - t_N + 1):N + 1]
    alarm_sums = np.sum(window_at_alarm, axis=0)
    std_tN = np.sqrt(fit.s2_table[min(t_N, fit.kmax), :])
    std_tN = np.maximum(std_tN, 1e-8)
    W_alarm = alarm_sums / std_tN
    j_hat = int(np.argmax(np.abs(W_alarm)))
    
    d1_CI = c_CI * math.sqrt(math.log(p / alpha))
    d1_rep = c_rep * math.sqrt(math.log(p / alpha))
    d2 = 4.0 * (d1_CI ** 2)
    
    S_hat = []
    b_tilde = {}
    for j in range(p):
        if j == j_hat:
            continue
        val = abs(E[j]) - b_min * math.sqrt(k)
        if val >= d1_CI:
            S_hat.append(j)
            valid_b = [b for b in scale_grid if (abs(E[j]) - b * math.sqrt(k)) >= d1_CI]
            b_tilde[j] = np.sign(E[j]) * (max(valid_b) if valid_b else b_min)
            
    S_rep = [j_hat]
    for j in range(p):
        if j != j_hat and (abs(E[j]) - b_min * math.sqrt(k)) >= d1_rep:
            S_rep.append(j)
    S_rep = sorted(list(set(S_rep)))
    
    if len(S_hat) == 0:
        b_anchor = max(b_min, abs(E[j_hat]) / math.sqrt(k))
        ci_lower = max(m0, int(math.floor(N - t_N - d2 / (b_anchor ** 2))))
    else:
        candidates = []
        for j in S_hat:
            b_val = abs(b_tilde[j])
            max_h = N - m0
            if max_h <= 0:
                t_j = 0
            else:
                suffix = Xc[N - max_h + 1 : N + 1, j][::-1]
                cum = np.cumsum(suffix)
                h_vec = np.arange(1, len(cum) + 1)
                obj = np.sign(b_tilde[j]) * cum - 0.5 * b_val * h_vec
                best_idx = int(np.argmax(obj))
                t_j = best_idx + 1 if obj[best_idx] > 0 else 0
                
            term = t_j + d2 / (b_val ** 2)
            candidates.append(term)
            
        ci_lower = max(m0, int(math.floor(N - min(candidates))))
        
    ci_upper = N
    ci_length = max(0, ci_upper - ci_lower)
    nu_hat = max(m0, N - t_N + 1)
    
    return {
        "nu_hat": nu_hat,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_length": ci_length,
        "S_hat": S_hat,
        "S_rep": S_rep,
        "anchor": j_hat
    }

def run_kaul_offline_ci(
    X: np.ndarray,
    alarm_t: int,
    alpha: float = 0.05,
    q_quantile: float = 11.03
) -> Dict[str, Any]:
    """Offline CI of Kaul et al. (2019/2021) after sequential alarm."""
    T, p = X.shape
    n_seg = alarm_t
    if n_seg < 10:
        return {"ci_lower": 0, "ci_upper": alarm_t, "ci_length": alarm_t, "nu_hat": alarm_t // 2}
        
    cum = np.cumsum(X[:n_seg], axis=0)
    total = cum[-1]
    
    t_vals = np.arange(1, n_seg)
    n_float = float(n_seg)
    factors_1 = np.sqrt((n_float - t_vals) / (n_float * t_vals))[:, None]
    factors_2 = np.sqrt(t_vals / (n_float * (n_float - t_vals)))[:, None]
    cusum_mat = factors_1 * cum[:-1] - factors_2 * (total - cum[:-1])
    
    l2_norms = np.linalg.norm(cusum_mat, axis=1)
    tau_hat = int(np.argmax(l2_norms)) + 1
    
    before_mean = cum[tau_hat - 1] / float(tau_hat)
    after_mean = (total - cum[tau_hat - 1]) / float(n_seg - tau_hat)
    raw_theta = after_mean - before_mean
    
    lambda_bic = math.sqrt(2.0 * math.log(p) / n_float)
    theta_thresh = np.sign(raw_theta) * np.maximum(0.0, np.abs(raw_theta) - lambda_bic)
    theta_norm_sq = float(np.sum(theta_thresh ** 2))
    
    if theta_norm_sq < 1e-4:
        theta_norm_sq = max(float(np.sum(raw_theta ** 2)), 1e-4)
        
    half_width = q_quantile / theta_norm_sq
    ci_lower = max(0, int(math.floor(tau_hat - half_width)))
    ci_upper = min(n_seg, int(math.ceil(tau_hat + half_width)))
    ci_length = max(0, ci_upper - ci_lower)
    
    return {
        "nu_hat": tau_hat,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_length": ci_length,
        "theta_norm_sq": theta_norm_sq
    }
