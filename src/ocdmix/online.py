from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .fit import FitResult
from .localize import localize_from_buffer


@dataclass
class Alarm:
    N: int
    t_N: int
    nu_dyadic: int
    nu_hat: Optional[int]
    CI: Optional[tuple[int, int]]
    support: Optional[list[int]]


class OnlineDetector:
    """Streaming change-point detector with O(p * |K|) per-update cost.

    Memory usage is O(p * kmax), independent of stream length.
    """

    def __init__(
        self,
        fit: FitResult,
        c_A: float,
        c_Q: float,
        ci_a: float = 0.0,
    ) -> None:
        self.fit = fit
        self.p = fit.p
        self.m0 = fit.m0
        self.kmax = fit.kmax
        self.K = list(fit.window_grid)
        self.n_K = len(self.K)

        self.base_A = np.sqrt(2.0 * np.log(2.0 * self.p * self.n_K))
        self.tau_A = c_A * self.base_A
        self.tau_Q = c_Q
        self.c_A = c_A
        self.c_Q = c_Q
        self.ci_a = ci_a

        self._sqrt_s2_grid: dict[int, np.ndarray] = {}
        for k in self.K:
            self._sqrt_s2_grid[k] = np.sqrt(fit.s2_grid[k])

        # Ring buffer for cumulative sums: shape [kmax+1, p].
        # Stores the last kmax+1 cumulative sums so we can compute
        # C[t] - C[t-k] for any k <= kmax during online monitoring.
        self._cumsum_buf = np.zeros((self.kmax + 1, self.p), dtype=np.float64)
        self._cumsum_head = 1
        self._cumsum_current = np.zeros(self.p, dtype=np.float64)

        # Ring buffer for raw centred observations: shape [2*kmax, p].
        # Post-alarm localisation searches over m in [kmin, min(2*t_N, N - m0 + 1)].
        # Because t_N can reach kmax, the buffer must cover 2*kmax past observations.
        self._obs_buf = np.zeros((2 * self.kmax, self.p), dtype=np.float64)
        self._obs_head = 0

        self._t = -1

        # Re-anchor every kmax steps to prevent floating-point drift
        self._anchor_cumsum = np.zeros(self.p, dtype=np.float64)
        self._steps_since_anchor = 0

        self._alarm: Optional[Alarm] = None

    @property
    def alarmed(self) -> bool:
        """Whether the detector has already alarmed."""
        return self._alarm is not None

    def update(self, x: np.ndarray) -> Optional[Alarm]:
        """Process one new observation."""
        if self._alarm is not None:
            return self._alarm

        self._t += 1
        t = self._t

        xc = np.asarray(x, dtype=np.float64) - self.fit.mu

        self._obs_buf[self._obs_head % (2 * self.kmax)] = xc
        self._obs_head += 1

        if t < self.m0:
            return None

        self._cumsum_current += xc
        self._cumsum_buf[self._cumsum_head % (self.kmax + 1)] = (
            self._cumsum_current - self._anchor_cumsum
        )
        self._cumsum_head += 1
        self._steps_since_anchor += 1

        if self._steps_since_anchor >= self.kmax:
            self._reanchor()

        violating_ks: list[int] = []
        p = self.p

        for k in self.K:
            if t < self.m0 + k - 1:
                continue

            n_steps = t - self.m0 + 1
            if n_steps < k:
                continue

            idx_t = (self._cumsum_head - 1) % (self.kmax + 1)
            idx_tk = (self._cumsum_head - 1 - k) % (self.kmax + 1)

            window_sum = self._cumsum_buf[idx_t] - self._cumsum_buf[idx_tk]
            sqrt_s2 = self._sqrt_s2_grid[k]
            W = window_sum / sqrt_s2

            A_k = np.max(np.abs(W))
            Q_k = (np.sum(W ** 2) - p) / np.sqrt(2.0 * p)

            if A_k > self.tau_A or Q_k > self.tau_Q:
                violating_ks.append(k)

        if violating_ks:
            N = t
            t_N = max(violating_ks)
            nu_dyadic = N - t_N + 1

            nu_hat, CI, support = self._localize(N, t_N)

            self._alarm = Alarm(
                N=N, t_N=t_N, nu_dyadic=nu_dyadic,
                nu_hat=nu_hat, CI=CI, support=support,
            )
            return self._alarm

        return None

    def _reanchor(self) -> None:
        """Re-anchor cumulative sums to prevent floating-point drift."""
        oldest_idx = self._cumsum_head % (self.kmax + 1)
        oldest_val = self._cumsum_buf[oldest_idx].copy()

        for i in range(self.kmax + 1):
            self._cumsum_buf[i] -= oldest_val

        self._anchor_cumsum += oldest_val
        self._steps_since_anchor = 0

    def _localize(
        self, N: int, t_N: int,
    ) -> tuple[Optional[int], Optional[tuple[int, int]], Optional[list[int]]]:
        """Localisation at alarm time using the observation ring buffer."""
        return localize_from_buffer(
            N=N,
            t_N=t_N,
            m0=self.m0,
            obs_buf=self._obs_buf,
            obs_head=self._obs_head,
            obs_buf_size=2 * self.kmax,
            fit=self.fit,
            tau_A=self.tau_A,
            tau_Q=self.tau_Q,
            c_A=self.c_A,
            c_Q=self.c_Q,
            ci_a=self.ci_a,
        )

    def get_statistic_at(self, k: int) -> tuple[float, float]:
        """Get current A[t,k] and Q[t,k] for a given window length.

        Useful for testing / diagnostics. Returns (nan, nan) if invalid.
        """
        t = self._t
        if t < self.m0 + k - 1:
            return (np.nan, np.nan)

        idx_t = (self._cumsum_head - 1) % (self.kmax + 1)
        idx_tk = (self._cumsum_head - 1 - k) % (self.kmax + 1)

        window_sum = self._cumsum_buf[idx_t] - self._cumsum_buf[idx_tk]
        sqrt_s2 = np.sqrt(self.fit.s2_grid[k])
        W = window_sum / sqrt_s2
        A_k = float(np.max(np.abs(W)))
        Q_k = float((np.sum(W ** 2) - self.p) / np.sqrt(2.0 * self.p))
        return (A_k, Q_k)
