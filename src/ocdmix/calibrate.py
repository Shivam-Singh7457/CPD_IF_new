"""Calibration of detection thresholds on null streams (E10)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .fit import DEFAULT_ALPHA0, DEFAULT_KMAX, DEFAULT_LE, DEFAULT_M0, DEFAULT_WINDOW_GRID, fit_training
from .scan import compute_base_A, compute_statistics_vectorised


@dataclass
class CalibrationResult:
    """Threshold calibration results."""
    c_A: float
    c_Q: float
    base_A: float
    tau_A: float
    tau_Q: float
    alpha0: float
    m0: int
    method: str
    lrv_method: str
    window_grid: list[int]
    kmax: int
    p: int
    T: int
    n_streams: int
    achieved_far: float
    statistic: str = "both"
    joint: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> CalibrationResult:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "statistic" not in data:
            data["statistic"] = "both"
        if "joint" not in data:
            data["joint"] = False
        return cls(**data)


def calibrate_thresholds(
    observations: np.ndarray,
    *,
    m0: int = DEFAULT_M0,
    alpha0: float = DEFAULT_ALPHA0,
    window_grid: tuple[int, ...] = DEFAULT_WINDOW_GRID,
    kmax: int = DEFAULT_KMAX,
    Le: int = DEFAULT_LE,
    method: Literal["proposed", "vanilla"] = "proposed",
    lrv_method: Literal["prewhiten", "bartlett"] = "prewhiten",
    statistic: Literal["both", "linf_only", "l_infinity", "q_only"] = "both",
    joint: bool = False,
) -> CalibrationResult:
    """E10: Calibrate thresholds c_A, c_Q from null streams.

    Parameters
    ----------
    observations : np.ndarray, shape (n_streams, T, p)
        Null stream observations.
    m0 : int
        Training length.
    alpha0 : float
        Target false-alarm probability (default 0.05).
    window_grid : tuple of int
        Candidate window lengths.
    kmax : int
        Maximum window length.
    Le : int
        Bandwidth for innovation LRV estimation.
    method : {"proposed", "vanilla"}
    lrv_method : {"prewhiten", "bartlett"}
    statistic : {"both", "linf_only", "l_infinity", "q_only"}

    Returns
    -------
    CalibrationResult
    """
    n_streams, T, p = observations.shape
    base_A = compute_base_A(p, len(window_grid))

    M_A = np.empty(n_streams, dtype=np.float64)
    M_Q = np.empty(n_streams, dtype=np.float64)

    for i in range(n_streams):
        X = np.asarray(observations[i], dtype=np.float64)
        fit = fit_training(
            X[:m0],
            m0=m0,
            window_grid=window_grid,
            kmax=kmax,
            Le=Le,
            method=method,
            lrv_method=lrv_method,
        )
        A, Q, _ = compute_statistics_vectorised(X, fit)
        # Max over monitoring period t >= m0 and windows
        M_A[i] = np.nanmax(A) / base_A
        M_Q[i] = np.nanmax(Q)

    if statistic in ("linf_only", "l_infinity"):
        q_level = 1.0 - alpha0
        c_A = float(np.quantile(M_A, q_level))
        c_Q = float("inf")
        tau_A = c_A * base_A
        tau_Q = float("inf")
        alarmed = M_A > c_A
    elif statistic == "q_only":
        q_level = 1.0 - alpha0
        c_A = float("inf")
        c_Q = float(np.quantile(M_Q, q_level))
        tau_A = float("inf")
        tau_Q = c_Q
        alarmed = M_Q > c_Q
    elif joint:
        # G2: Joint maximum calibration at level alpha0
        # Standardise each statistic by its marginal (1 - alpha0) quantile
        c_A_marg = float(np.quantile(M_A, 1.0 - alpha0))
        c_Q_marg = float(np.quantile(M_Q, 1.0 - alpha0))
        # Joint maximum of normalised test statistics
        R = np.maximum(M_A / c_A_marg, M_Q / c_Q_marg)
        kappa = float(np.quantile(R, 1.0 - alpha0))
        c_A = float(c_A_marg * kappa)
        c_Q = float(c_Q_marg * kappa)
        tau_A = c_A * base_A
        tau_Q = c_Q
        alarmed = (M_A > c_A) | (M_Q > c_Q)
    else:
        # Empirical (1 - alpha0/2) quantiles (Bonferroni alpha0/2 split)
        q_level = 1.0 - alpha0 / 2.0
        c_A = float(np.quantile(M_A, q_level))
        c_Q = float(np.quantile(M_Q, q_level))
        tau_A = c_A * base_A
        tau_Q = c_Q
        alarmed = (M_A > c_A) | (M_Q > c_Q)

    achieved_far = float(np.mean(alarmed))

    return CalibrationResult(
        c_A=c_A,
        c_Q=c_Q,
        base_A=base_A,
        tau_A=tau_A,
        tau_Q=tau_Q,
        alpha0=alpha0,
        m0=m0,
        method=method,
        lrv_method=lrv_method,
        window_grid=list(window_grid),
        kmax=kmax,
        p=p,
        T=T,
        n_streams=n_streams,
        achieved_far=achieved_far,
        statistic=statistic,
        joint=joint,
    )


def self_calibrate(
    X_train: np.ndarray,
    *,
    T: int = 5000,
    m0: int = DEFAULT_M0,
    alpha0: float = DEFAULT_ALPHA0,
    window_grid: tuple[int, ...] = DEFAULT_WINDOW_GRID,
    kmax: int = DEFAULT_KMAX,
    Le: int = DEFAULT_LE,
    method: Literal["proposed", "vanilla"] = "proposed",
    lrv_method: Literal["prewhiten", "bartlett"] = "prewhiten",
    n_sim_streams: int = 300,
    seed: int = 20261014,
    joint: bool = False,
) -> CalibrationResult:
    """Task 9: Self-calibration from training segment X_train (m0, p).

    Fits per-coordinate AR(1) coefficients and innovation covariance from X_train,
    simulates n_sim_streams null streams of length T from the fitted model,
    and computes c_A, c_Q by E10.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    p = X_train.shape[1]

    # 1. Fit per-coordinate AR(1) and innovation covariance
    mu = X_train.mean(axis=0)
    Xc = X_train - mu

    num = np.sum(Xc[:-1] * Xc[1:], axis=0)
    den = np.sum(Xc[:-1] ** 2, axis=0)
    r = np.clip(np.where(den > 0, num / den, 0.0), -0.95, 0.95)

    # Innovations
    e = Xc[1:] - r[np.newaxis, :] * Xc[:-1]
    e_c = e - e.mean(axis=0)
    sigma_eps = (e_c.T @ e_c) / e_c.shape[0]
    # Symmetrize and regularize
    sigma_eps = 0.5 * (sigma_eps + sigma_eps.T)
    min_eig = float(np.linalg.eigvalsh(sigma_eps).min())
    if min_eig <= 1e-8:
        sigma_eps += (1e-6 - min_eig) * np.eye(p)

    # Stationary covariance for coordinate-wise AR(1):
    # Gamma_ij = Sigma_eps_ij / (1 - r_i * r_j)
    denom = 1.0 - np.outer(r, r)
    gamma_stat = sigma_eps / np.maximum(denom, 1e-4)
    gamma_stat = 0.5 * (gamma_stat + gamma_stat.T)
    min_eig_g = float(np.linalg.eigvalsh(gamma_stat).min())
    if min_eig_g <= 1e-8:
        gamma_stat += (1e-6 - min_eig_g) * np.eye(p)

    # Cholesky factors
    chol_gamma = np.linalg.cholesky(gamma_stat)
    chol_eps = np.linalg.cholesky(sigma_eps)

    # 2. Simulate n_sim_streams of length T
    rng = np.random.default_rng(seed)
    sim_obs = np.empty((n_sim_streams, T, p), dtype=np.float64)

    for i in range(n_sim_streams):
        # Initial state
        z = np.empty((T, p), dtype=np.float64)
        z[0] = chol_gamma @ rng.standard_normal(p)

        # Innovations
        eps = rng.standard_normal((T, p)) @ chol_eps.T

        # AR(1) recursion
        for t in range(1, T):
            z[t] = r * z[t - 1] + eps[t]

        sim_obs[i] = mu + z

    # 3. Calibrate on the simulated null streams
    return calibrate_thresholds(
        sim_obs,
        m0=m0,
        alpha0=alpha0,
        window_grid=window_grid,
        kmax=kmax,
        Le=Le,
        method=method,
        lrv_method=lrv_method,
        joint=joint,
    )

