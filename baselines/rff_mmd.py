"""Online Kernel and Random Fourier Feature (RFF) Change Detectors.

Includes:
  M6: StreamingRFFMMD (Kalinke & Gavioli-Akilagun, NeurIPS 2025)
  M7: ScanBStatistic, OKCUSUM (Song Wei et al., 2022 / Kalinke & Gavioli-Akilagun 2025)
      NEWMA (Keriven et al., 2020 standalone implementation)
"""
from __future__ import annotations

import math
from typing import Optional, List, Tuple, Callable
import numpy as np
from scipy.special import comb as nchoosek
from sklearn.metrics.pairwise import rbf_kernel

class Gauss:
    def __init__(self, gamma: float):
        self.gamma = gamma

    def eval(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return rbf_kernel(X, Y, gamma=self.gamma)
    
    def rff_sampler(self, d: int) -> Callable[[int], np.ndarray]:
        sigma = 1.0 / np.sqrt(2.0 * self.gamma)
        return lambda n: np.random.randn(n, d) / sigma
    
    @staticmethod
    def _est_nu(X_: np.ndarray, max_len: int = 500) -> float:
        n = min(len(X_), max_len)
        rng = np.random.default_rng(1234)
        idx = rng.choice(len(X_), size=n, replace=False) if len(X_) > n else np.arange(len(X_))
        X = X_[idx]
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(np.sum((X[i] - X[j]) ** 2))
        nu = float(np.median(dists)) if dists else 1.0
        return np.sqrt(nu * 0.5)

    @staticmethod
    def _nu2gamma(nu: float) -> float:
        return 1.0 / (2.0 * (nu ** 2)) if nu > 0 else 1.0

    @staticmethod
    def est_gamma(X: np.ndarray, max_len: int = 500) -> float:
        return Gauss._nu2gamma(Gauss._est_nu(X, max_len=max_len))


class StreamingRFFMMD:
    """M6: Streaming RFF MMD Change Detector (Kalinke & Gavioli-Akilagun 2025)."""
    
    class Window:
        def __init__(self, z: np.ndarray, c: int):
            self.z = z
            self.c = c
            
    def __init__(self, kernel: Gauss, d: int, num_omegas: int = 100, seed: Optional[int] = None):
        self.windows: List[StreamingRFFMMD.Window] = []
        if seed is not None:
            np.random.seed(seed)
        self.omegas = (kernel.rff_sampler(d))(num_omegas)
        self.num_omegas = num_omegas
        self.d = d
        self.recent_statistic = 0.0
        self.num_samples = 0

    def insert(self, element: np.ndarray):
        element = np.asarray(element, dtype=np.float64).reshape(1, -1)
        tmp = element @ self.omegas.T
        z = (1.0 / np.sqrt(self.num_omegas)) * np.concatenate((np.cos(tmp), np.sin(tmp)), axis=1)
        W = StreamingRFFMMD.Window(z=z, c=1)
        self.windows.append(W)
        self.num_samples += 1
        
        while len(self.windows) >= 2:
            W1 = self.windows[-1]
            W2 = self.windows[-2]
            if W1.c == W2.c:
                W_merged = StreamingRFFMMD.Window(z=W1.z + W2.z, c=W1.c + W2.c)
                self.windows = self.windows[:-2] + [W_merged]
            else:
                break
                
        vals = self.mmd_values()
        self.recent_statistic = float(np.max(vals)) if len(vals) > 0 else 0.0

    def mmd_values(self) -> np.ndarray:
        if len(self.windows) < 2:
            return np.zeros(0)
        zs = [w.z for w in self.windows]
        cs = [w.c for w in self.windows]
        lz = np.zeros_like(zs[0])
        lc = 0
        rz = np.sum(zs, axis=0)
        rc = sum(cs)
        vals = np.zeros(len(zs) - 1)
        for i in range(len(zs) - 1):
            lz = lz + zs[i]
            rz = rz - zs[i]
            lc += cs[i]
            rc -= cs[i]
            vals[i] = np.linalg.norm((1.0 / lc) * lz - (1.0 / rc) * rz)
        return vals

    def statistic(self) -> float:
        return self.recent_statistic


class ScanBStatistic:
    """M7: Scan B-Statistic Change Detector (Li et al., 2019 / Kalinke & Gavioli-Akilagun 2025)."""
    
    def __init__(self, reference_sample: np.ndarray, B0: int = 50, N: int = 5, gamma: Optional[float] = None, seed: Optional[int] = 1234):
        self.rng = np.random.default_rng(seed)
        self.B0 = B0
        self.N = N
        if gamma is None:
            self.gamma = Gauss.est_gamma(reference_sample)
        else:
            self.gamma = gamma

        total_ref = N * B0
        if len(reference_sample) < total_ref:
            # Tile or sample with replacement if reference sample is smaller
            idx = self.rng.choice(len(reference_sample), size=total_ref, replace=True)
        else:
            idx = self.rng.choice(len(reference_sample), size=total_ref, replace=False)
            
        self.ref_blocks = reference_sample[idx]
        self.ref_grams_XX = []
        for i in range(self.N):
            x = self.ref_blocks[i * self.B0 : (i + 1) * self.B0]
            self.ref_grams_XX.append(rbf_kernel(x, gamma=self.gamma))

        self.post_block = reference_sample[-B0:].copy()
        
        # Estimate variance
        var_est = (1.0 / max(1.0, nchoosek(B0, 2))) * (
            (1.0 / N) * self.expectation_h_squared(reference_sample) +
            ((N - 1.0) / N) * self.covariance_h(reference_sample)
        )
        self.var = max(var_est, 1e-8)
        self.recent_stat = 0.0

    def mmd_u(self, sample_x: np.ndarray, gram_XX: np.ndarray, sample_y: np.ndarray, gram_YY: np.ndarray) -> float:
        n_x = len(sample_x)
        n_y = len(sample_y)
        XX = gram_XX - np.eye(n_x)
        YY = gram_YY - np.eye(n_y)
        XY = rbf_kernel(sample_x, sample_y, gamma=self.gamma)
        return float(np.sum(XX) / (n_x * (n_x - 1)) + np.sum(YY) / (n_y * (n_y - 1)) - 2.0 * np.mean(XY))

    def expectation_h_squared(self, sample: np.ndarray) -> float:
        n = len(sample) // 4
        if n < 1:
            return 1.0
        x = sample[0:n]
        x_p = sample[n:2*n]
        y = sample[2*n:3*n]
        y_p = sample[3*n:4*n]
        K_xx = rbf_kernel(x, x_p, gamma=self.gamma)
        K_yy = rbf_kernel(y, y_p, gamma=self.gamma)
        K_xy1 = rbf_kernel(x, y_p, gamma=self.gamma)
        K_xy2 = rbf_kernel(x_p, y, gamma=self.gamma)
        return float(np.mean((K_xx + K_yy - K_xy1 - K_xy2) ** 2))

    def covariance_h(self, sample: np.ndarray) -> float:
        n = len(sample) // 6
        if n < 1:
            return 0.1
        x = sample[0:n]
        x_p = sample[n:2*n]
        x_pp = sample[2*n:3*n]
        x_ppp = sample[3*n:4*n]
        y = sample[4*n:5*n]
        y_p = sample[5*n:6*n]
        h1 = rbf_kernel(x, x_p, gamma=self.gamma) + rbf_kernel(y, y_p, gamma=self.gamma) - rbf_kernel(x, y_p, gamma=self.gamma) - rbf_kernel(x_p, y, gamma=self.gamma)
        h2 = rbf_kernel(x, x_pp, gamma=self.gamma) + rbf_kernel(x_ppp, y_p, gamma=self.gamma) - rbf_kernel(x, y_p, gamma=self.gamma) - rbf_kernel(x_pp, x_ppp, gamma=self.gamma)
        return float(np.mean(h1 * h2))

    def insert(self, input_value: np.ndarray):
        input_value = np.asarray(input_value, dtype=np.float64).reshape(1, -1)
        self.post_block = np.concatenate((self.post_block[1:], input_value))
        acc = 0.0
        gram_YY = rbf_kernel(self.post_block, gamma=self.gamma)
        for i in range(self.N):
            acc += self.mmd_u(self.ref_blocks[i * self.B0 : (i + 1) * self.B0], self.ref_grams_XX[i], self.post_block, gram_YY)
        self.recent_stat = float(acc / (self.N * np.sqrt(self.var)))

    def statistic(self) -> float:
        return self.recent_stat


class OKCUSUM:
    """M7: Online Kernel CUSUM (Song Wei et al., 2022 / Kalinke & Gavioli-Akilagun 2025)."""
    
    def __init__(self, reference_sample: np.ndarray, B_max: int = 60, N: int = 5, gamma: Optional[float] = None, B_min: int = 2, seed: Optional[int] = 1234):
        self.rng = np.random.default_rng(seed)
        if gamma is None:
            self.gamma = Gauss.est_gamma(reference_sample)
        else:
            self.gamma = gamma
        self.B_max = B_max
        self.B_min = B_min
        self.N = N

        total_ref = N * B_max
        if len(reference_sample) < total_ref:
            idx = self.rng.choice(len(reference_sample), size=total_ref, replace=True)
        else:
            idx = self.rng.choice(len(reference_sample), size=total_ref, replace=False)
            
        self.ref_blocks = reference_sample[idx]
        self.post_block = reference_sample[-B_max:].copy()
        self.recent_stat = 0.0

    def insert(self, input_value: np.ndarray):
        input_value = np.asarray(input_value, dtype=np.float64).reshape(1, -1)
        self.post_block = np.concatenate((self.post_block[1:], input_value))
        
        # Max over window sizes B in [B_min, B_max]
        max_stat = 0.0
        for B in range(self.B_min, self.B_max + 1, 4):
            sub_post = self.post_block[-B:]
            sub_ref = self.ref_blocks[:B]
            XY = rbf_kernel(sub_post, sub_ref, gamma=self.gamma)
            YY = rbf_kernel(sub_post, sub_post, gamma=self.gamma)
            stat_B = float(np.mean(YY) - np.mean(XY))
            if stat_B > max_stat:
                max_stat = stat_B
        self.recent_stat = max_stat

    def statistic(self) -> float:
        return self.recent_stat


class NEWMA:
    """M7: NEWMA Change Detector (Keriven et al., 2020 standalone implementation)."""
    
    def __init__(self, reference_sample: np.ndarray, d: int, B: int = 50, m_features: int = 100, seed: Optional[int] = 1234):
        self.B = B
        self.Lambda = 2.0 / B
        self.small_lambda = 0.5 / B
        self.m = m_features
        self.d = d
        
        rng = np.random.default_rng(seed)
        # Choose median sigma
        gamma = Gauss.est_gamma(reference_sample[:min(500, len(reference_sample))])
        sigma = 1.0 / np.sqrt(2.0 * max(gamma, 1e-4))
        self.W = rng.normal(scale=1.0 / sigma, size=(self.m, self.d))
        self.b = rng.uniform(0, 2 * np.pi, size=(self.m,))
        
        self.s = np.zeros(self.m, dtype=np.float64)
        self.l = np.zeros(self.m, dtype=np.float64)
        self.recent_stat = 0.0
        
        # Initialize on reference sample
        for x in reference_sample:
            self.insert(x)
            
    def insert(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64).flatten()
        z = np.sqrt(2.0 / self.m) * np.cos(self.W @ x + self.b)
        self.s = (1.0 - self.Lambda) * self.s + self.Lambda * z
        self.l = (1.0 - self.small_lambda) * self.l + self.small_lambda * z
        self.recent_stat = float(np.sum((self.s - self.l) ** 2))
        
    def statistic(self) -> float:
        return self.recent_stat

