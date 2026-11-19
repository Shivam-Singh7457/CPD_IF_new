from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# Default parameters
DEFAULT_M0 = 1000
DEFAULT_WINDOW_GRID = (8, 16, 32, 64, 128, 256, 512)
DEFAULT_KMAX = 512
DEFAULT_LE = 5
DEFAULT_ALPHA0 = 0.05


@dataclass
class FitResult:
    """All quantities estimated from the training window X[0:m0]."""
    m0: int
    p: int
    mu: np.ndarray
    g0: np.ndarray
    r: np.ndarray
    sigma2: np.ndarray
    s2_grid: dict[int, np.ndarray]
    s2_table: np.ndarray
    window_grid: tuple[int, ...]
    kmax: int
    method: str
    lrv_method: str

    # Additional diagnostic quantities
    omega: np.ndarray = field(default_factory=lambda: np.empty(0))


def _bartlett_weight(h: int, L: int) -> float:
    return 1.0 - h / (L + 1)


def _autocovariance(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocovariances gamma(0..max_lag) for each column of x."""
    n, p = x.shape
    gam = np.empty((max_lag + 1, p), dtype=np.float64)
    for h in range(max_lag + 1):
        gam[h] = np.sum(x[: n - h] * x[h:], axis=0) / n
    return gam


def fit_training(
    X_train: np.ndarray,
    *,
    m0: int = DEFAULT_M0,
    window_grid: tuple[int, ...] = DEFAULT_WINDOW_GRID,
    kmax: int = DEFAULT_KMAX,
    Le: int = DEFAULT_LE,
    method: Literal["proposed", "vanilla"] = "proposed",
    lrv_method: Literal["prewhiten", "bartlett"] = "prewhiten",
) -> FitResult:
    """Estimate all training-stage quantities from X[0:m0]."""
    X_train = np.asarray(X_train, dtype=np.float64)
    if X_train.shape[0] != m0:
        raise ValueError(f"X_train has {X_train.shape[0]} rows but m0={m0}")
    p = X_train.shape[1]

    mu = X_train.mean(axis=0)
    Xc = X_train - mu

    g0 = np.mean(Xc ** 2, axis=0)
    num = np.sum(Xc[:-1] * Xc[1:], axis=0)
    den = np.sum(Xc[:-1] ** 2, axis=0)
    r_raw = np.where(den > 0, num / den, 0.0)
    r = np.clip(r_raw, -0.95, 0.95)

    if method == "vanilla":
        sigma2 = g0.copy()
        omega = g0.copy()
        s2_grid, s2_table = _vanilla_normalisation(g0, window_grid, kmax)
        return FitResult(
            m0=m0, p=p, mu=mu, g0=g0, r=r, sigma2=sigma2,
            s2_grid=s2_grid, s2_table=s2_table,
            window_grid=window_grid, kmax=kmax,
            method=method, lrv_method=lrv_method, omega=omega,
        )

    # Proposed method
    if lrv_method == "prewhiten":
        sigma2, omega = _prewhiten_lrv(Xc, r, g0, m0, Le)
    else:
        sigma2, omega = _bartlett_lrv(Xc, g0, m0, L=20)

    s2_grid, s2_table = _proposed_normalisation(
        sigma2, g0, r, m0, window_grid, kmax, lrv_method, Xc,
    )

    return FitResult(
        m0=m0, p=p, mu=mu, g0=g0, r=r, sigma2=sigma2,
        s2_grid=s2_grid, s2_table=s2_table,
        window_grid=window_grid, kmax=kmax,
        method=method, lrv_method=lrv_method, omega=omega,
    )


def _prewhiten_lrv(
    Xc: np.ndarray, r: np.ndarray, g0: np.ndarray, m0: int, Le: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Prewhiten-then-HAC estimate of the long-run variance."""
    e = Xc[1:] - r[np.newaxis, :] * Xc[:-1]
    e = e - e.mean(axis=0)

    gam_e = _autocovariance(e, Le)

    omega = gam_e[0].copy()
    for h in range(1, Le + 1):
        omega += 2.0 * _bartlett_weight(h, Le) * gam_e[h]
    omega = np.maximum(omega, 1e-8)

    sigma2 = omega / (1.0 - r) ** 2

    return sigma2, omega


def _bartlett_lrv(
    Xc: np.ndarray, g0: np.ndarray, m0: int, L: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Bartlett-kernel LRV estimate directly from centred data."""
    gam = _autocovariance(Xc, L)

    sigma2 = gam[0].copy()
    for h in range(1, L + 1):
        sigma2 += 2.0 * _bartlett_weight(h, L) * gam[h]
    sigma2 = np.maximum(sigma2, 1e-8)

    return sigma2, sigma2.copy()


def _proposed_normalisation(
    sigma2: np.ndarray,
    g0: np.ndarray,
    r: np.ndarray,
    m0: int,
    window_grid: tuple[int, ...],
    kmax: int,
    lrv_method: str,
    Xc: np.ndarray,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Normalisation constants for the proposed method."""
    p = sigma2.shape[0]
    max_m = 2 * kmax
    s2_table = np.empty((max_m + 1, p), dtype=np.float64)
    s2_table[0] = 0.0

    if lrv_method == "prewhiten":
        for m in range(1, max_m + 1):
            rk = r ** m
            v = sigma2 - 2.0 * g0 * r * (1.0 - rk) / (m * (1.0 - r) ** 2)
            v = np.maximum(v, 0.2 * g0)
            s2_table[m] = m * v + (m ** 2) * sigma2 / m0
    else:
        # Bartlett ablation
        L = 20
        gam = _autocovariance(Xc, L)
        for m in range(1, max_m + 1):
            v = gam[0].copy()
            H = min(L, m - 1)
            for h in range(1, H + 1):
                v += 2.0 * _bartlett_weight(h, L) * (1.0 - h / m) * gam[h]
            v = np.maximum(v, 0.2 * g0)
            s2_table[m] = m * v + (m ** 2) * sigma2 / m0

    s2_grid: dict[int, np.ndarray] = {}
    for k in window_grid:
        s2_grid[k] = s2_table[k].copy()

    return s2_grid, s2_table


def _vanilla_normalisation(
    g0: np.ndarray,
    window_grid: tuple[int, ...],
    kmax: int,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Vanilla baseline: s2[k,j] = k * g0_j."""
    p = g0.shape[0]
    max_m = 2 * kmax
    s2_table = np.empty((max_m + 1, p), dtype=np.float64)
    for m in range(max_m + 1):
        s2_table[m] = m * g0

    s2_grid: dict[int, np.ndarray] = {}
    for k in window_grid:
        s2_grid[k] = s2_table[k].copy()

    return s2_grid, s2_table
