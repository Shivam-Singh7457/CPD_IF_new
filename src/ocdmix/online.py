"""Streaming OnlineDetector: constant memory, constant time per observation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .fit import FitResult
from .localize import localize_from_buffer


@dataclass
class Alarm:
    """Alarm object returned by OnlineDetector.update()."""
    N: int                  # alarm time (zero-based index of triggering observation)
    t_N: int                # largest violating window length at alarm time
    nu_dyadic: int          # N - t_N + 1
    nu_hat: Optional[int]   # refined change-point estimate (E12)
    CI: Optional[tuple[int, int]]   # confidence interval for change location (E13)
    support: Optional[list[int]]    # estimated affected coordinates (E14)


class OnlineDetector:
    """Streaming change-point detector with O(p * |K|) per-update cost.

    Memory usage is O(p * kmax), independent of stream length.

    Parameters
    ----------
    fit : FitResult
        Training-stage estimates.
    c_A : float
        Threshold multiplier for the max statistic.
    c_Q : float
        Threshold for the quadratic statistic.
    ci_a : float, optional
        CI half-width parameter (from calibrate_ci). If None, CI is not computed.
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

        # Thresholds (E9)
        self.base_A = np.sqrt(2.0 * np.log(2.0 * self.p * self.n_K))
        self.tau_A = c_A * self.base_A
        self.tau_Q = c_Q
        self.c_A = c_A
        self.c_Q = c_Q
        self.ci_a = ci_a

        # Precompute sqrt(s2) for each grid window
        self._sqrt_s2_grid: dict[int, np.ndarray] = {}
        for k in self.K:
            self._sqrt_s2_grid[k] = np.sqrt(fit.s2_grid[k])

        # Ring buffer for cumulative sums: shape [kmax+1, p]
        # We store the last kmax+1 cumulative sums so we can compute
        # C[t] - C[t-k] for any k <= kmax during online monitoring.
        # Position 0 stores C[m0-1] = 0. Next write is at position 1.
        self._cumsum_buf = np.zeros((self.kmax + 1, self.p), dtype=np.float64)
        self._cumsum_head = 1  # next write position in ring buffer
        # Current cumulative sum value (running total)
        self._cumsum_current = np.zeros(self.p, dtype=np.float64)

        # Ring buffer for raw centred observations: shape [2*kmax, p]
        # NOTE (Footnote G1 correction): While online monitoring only requires
        # kmax+1 cumulative sums to evaluate dyadic windows k in K, post-alarm
        # localization (Eq. 13) searches over m in [kmin, min(2*t_N, N - m0 + 1)].
        # Because t_N can reach kmax, candidate windows can extend up to 2*kmax.
        # Hence the ring buffer for observations must cover 2*kmax past observations.
        self._obs_buf = np.zeros((2 * self.kmax, self.p), dtype=np.float64)
        self._obs_head = 0

        # Step counter (zero-based: first observation is t=0)
        self._t = -1

        # Re-anchoring: every kmax steps, subtract the oldest cumsum to
        # prevent floating-point drift
        self._anchor_cumsum = np.zeros(self.p, dtype=np.float64)
        self._steps_since_anchor = 0

        # Store the alarm if detected (detector stops after first alarm)
        self._alarm: Optional[Alarm] = None

    @property
    def alarmed(self) -> bool:
        """Whether the detector has already alarmed."""
        return self._alarm is not None

    def update(self, x: np.ndarray) -> Optional[Alarm]:
        """Process one new observation.

        Parameters
        ----------
        x : (p,) observation vector

        Returns
        -------
        Alarm or None
        """
        if self._alarm is not None:
            return self._alarm  # already alarmed

        self._t += 1
        t = self._t

        # Centre observation
        xc = np.asarray(x, dtype=np.float64) - self.fit.mu  # (p,)

        # Store in observation ring buffer
        self._obs_buf[self._obs_head % (2 * self.kmax)] = xc
        self._obs_head += 1

        if t < self.m0:
            # Still in training window — just accumulate
            # We don't monitor during training but we do update cumsum
            # Actually, cumsum starts from m0. So skip.
            return None

        # Update cumulative sum: C[t] = C[t-1] + Xc[t]
        self._cumsum_current += xc
        # Store in ring buffer (relative to anchor)
        self._cumsum_buf[self._cumsum_head % (self.kmax + 1)] = (
            self._cumsum_current - self._anchor_cumsum
        )
        self._cumsum_head += 1
        self._steps_since_anchor += 1

        # Re-anchor every kmax steps
        if self._steps_since_anchor >= self.kmax:
            self._reanchor()

        # Check each window length
        violating_ks: list[int] = []
        p = self.p

        for k in self.K:
            # Valid only when t - k + 1 >= m0, i.e. t >= m0 + k - 1
            if t < self.m0 + k - 1:
                continue

            # Number of monitoring steps so far (including this one)
            n_steps = t - self.m0 + 1

            if n_steps < k:
                continue

            # C[t] is at position (cumsum_head - 1) in ring buffer
            # C[t-k] is at position (cumsum_head - 1 - k) in ring buffer
            idx_t = (self._cumsum_head - 1) % (self.kmax + 1)
            idx_tk = (self._cumsum_head - 1 - k) % (self.kmax + 1)

            window_sum = self._cumsum_buf[idx_t] - self._cumsum_buf[idx_tk]  # (p,)
            sqrt_s2 = self._sqrt_s2_grid[k]
            W = window_sum / sqrt_s2  # (p,)

            A_k = np.max(np.abs(W))
            Q_k = (np.sum(W ** 2) - p) / np.sqrt(2.0 * p)

            if A_k > self.tau_A or Q_k > self.tau_Q:
                violating_ks.append(k)

        if violating_ks:
            N = t
            t_N = max(violating_ks)
            nu_dyadic = N - t_N + 1

            # Localisation using observation ring buffer
            nu_hat, CI, support = self._localize(N, t_N)

            self._alarm = Alarm(
                N=N, t_N=t_N, nu_dyadic=nu_dyadic,
                nu_hat=nu_hat, CI=CI, support=support,
            )
            return self._alarm

        return None

    def _reanchor(self) -> None:
        """Re-anchor cumulative sums to prevent floating-point drift."""
        # The oldest entry in the ring buffer
        oldest_idx = self._cumsum_head % (self.kmax + 1)
        oldest_val = self._cumsum_buf[oldest_idx].copy()

        # Subtract oldest from all entries
        for i in range(self.kmax + 1):
            self._cumsum_buf[i] -= oldest_val

        # Update anchor
        self._anchor_cumsum += oldest_val
        self._steps_since_anchor = 0

    def _localize(
        self, N: int, t_N: int,
    ) -> tuple[Optional[int], Optional[tuple[int, int]], Optional[list[int]]]:
        """E12-E14: localisation at alarm time.

        Uses the observation ring buffer to compute window sums for
        arbitrary integer window lengths.
        """
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
