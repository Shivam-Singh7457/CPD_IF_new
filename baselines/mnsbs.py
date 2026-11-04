"""M8: Multivariate Nonparametric Seeded Binary Segmentation (MNSBS).

Re-implementation of Madrid Padilla et al. (NeurIPS 2023):
  "Change point detection and inference in multivariable nonparametric models under mixing conditions"
"""
from __future__ import annotations

import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

def compute_seeded_intervals(T: int, min_len: int = 10) -> List[Tuple[int, int]]:
    """Definition 2: Seeded intervals J."""
    intervals = []
    # K_depth = ceil(C_K * log(T))
    max_k = max(1, int(math.ceil(math.log2(T))))
    for k in range(1, max_k + 1):
        len_k = max(2, int(T * (2.0 ** (-k + 1))))
        shift_k = max(1, int(T * (2.0 ** (-k))))
        if len_k < min_len:
            break
        num_intervals = max(1, int(math.ceil((T - len_k) / shift_k)) + 1)
        for i in range(num_intervals):
            start = i * shift_k
            end = min(T, start + len_k)
            if end - start >= min_len:
                intervals.append((start, end))
    # Deduplicate and sort
    intervals = sorted(list(set(intervals)), key=lambda x: (x[0], x[1]))
    return intervals

def kernel_cusum_l2_norm_sq(
    gram: np.ndarray,
    alpha: int,
    beta: int,
    t: int,
    integ: Optional[np.ndarray] = None
) -> float:
    """Computes || \widetilde{F}^{(\alpha, \beta]}_{t, h} ||_{L_2}^2 using precomputed kernel Gram matrix or 2D prefix sums."""
    n_L = t - alpha
    n_R = beta - t
    n_tot = beta - alpha
    if n_L <= 0 or n_R <= 0 or n_tot <= 0:
        return 0.0
        
    c_L = math.sqrt(n_R / (n_tot * n_L))
    c_R = -math.sqrt(n_L / (n_tot * n_R))
    
    if integ is not None:
        sum_LL = integ[t, t] - integ[alpha, t] - integ[t, alpha] + integ[alpha, alpha]
        sum_RR = integ[beta, beta] - integ[t, beta] - integ[beta, t] + integ[t, t]
        sum_LR = integ[t, beta] - integ[alpha, beta] - integ[t, t] + integ[alpha, t]
    else:
        sum_LL = np.sum(gram[alpha:t, alpha:t])
        sum_RR = np.sum(gram[t:beta, t:beta])
        sum_LR = np.sum(gram[alpha:t, t:beta])
    
    norm_sq = (c_L ** 2) * sum_LL + (c_R ** 2) * sum_RR + 2.0 * c_L * c_R * sum_LR
    return float(max(0.0, norm_sq))

class MNSBS:
    """M8: Multivariate Nonparametric Seeded Binary Segmentation (Re-implementation)."""
    
    def __init__(
        self,
        h: Optional[float] = None,
        tau: Optional[float] = None,
        c_h: float = 1.0,
        kernel_type: str = "Gaussian"
    ):
        self.h = h
        self.tau = tau
        self.c_h = c_h
        self.kernel_type = kernel_type
        
    def fit_predict(
        self,
        X: np.ndarray,
        alpha_ci: float = 0.05
    ) -> Dict[str, Any]:
        """Run MNSBS localization and confidence interval inference."""
        T, p = X.shape
        
        # Bandwidth selection if not provided
        # In MP paper: h proportional to T^{-1/(2r + p)} or median heuristic
        # With Gaussian kernel, kernel bandwidth gamma = 1 / (2 * h^2)
        if self.h is None:
            # Median heuristic on subsample
            sub_n = min(T, 300)
            diffs = []
            for i in range(sub_n):
                for j in range(i + 1, min(sub_n, i + 30)):
                    diffs.append(np.sum((X[i] - X[j]) ** 2))
            med = float(np.median(diffs)) if diffs else 1.0
            base_h = math.sqrt(max(med, 1e-4) * 0.5)
            self.h = self.c_h * base_h
            
        gamma = 1.0 / (2.0 * max(self.h ** 2, 1e-6))
        
        # Precompute kernel Gram matrix (Gaussian)
        gram = rbf_kernel(X, X, gamma=gamma)
        integ = np.pad(np.cumsum(np.cumsum(gram, axis=0), axis=1), ((1, 0), (1, 0)))
        
        # Threshold tau selection if not provided
        if self.tau is None:
            # Theoretical scaling: tau proportional to sqrt(log T)
            self.tau = 1.5 * math.sqrt(2.0 * math.log(max(T, 2)))
            
        # Seeded intervals
        min_interval_len = max(6, int(math.ceil(2.0 * math.log(T) * (self.h ** (-min(p, 2))))))
        min_interval_len = min(min_interval_len, T // 4)
        min_interval_len = max(min_interval_len, 6)
        all_intervals = compute_seeded_intervals(T, min_len=min_interval_len)
        
        # Algorithm 1: Recursive seeded binary segmentation
        change_points = []
        
        def run_segment(s: int, e: int):
            if e - s < min_interval_len:
                return
            best_a = -1.0
            best_b = -1
            best_I = None
            
            for (alpha, beta) in all_intervals:
                if alpha >= s and beta <= e and (beta - alpha) >= min_interval_len:
                    rho = max(2, int(math.ceil(math.log(T) * (self.h ** (-min(p, 2))))))
                    rho = min(rho, (beta - alpha) // 4)
                    rho = max(rho, 2)
                    
                    if beta - alpha > 2 * rho:
                        for t_cand in range(alpha + rho, beta - rho + 1):
                            val_sq = kernel_cusum_l2_norm_sq(gram, alpha, beta, t_cand, integ=integ)
                            val = math.sqrt(val_sq)
                            if val > best_a:
                                best_a = val
                                best_b = t_cand
                                best_I = (alpha, beta)
                                
            if best_a > self.tau and best_b != -1:
                change_points.append(best_b)
                # Recurse on left and right
                run_segment(s, best_b)
                run_segment(best_b + 1, e)
                
        run_segment(0, T)
        raw_cps = sorted(list(set(change_points)))
        
        # Stage 2: Refinement (Equations 4 & 5 in MP paper)
        refined_cps = []
        cis = []
        widths = []
        
        extended_cps = [0] + raw_cps + [T]
        for k in range(1, len(extended_cps) - 1):
            eta_prev = extended_cps[k - 1]
            eta_curr = extended_cps[k]
            eta_next = extended_cps[k + 1]
            
            s_k = int(math.floor(0.9 * eta_prev + 0.1 * eta_curr))
            e_k = int(math.ceil(0.9 * eta_next + 0.1 * eta_curr))
            
            # Refined search
            best_eta = eta_curr
            best_score = float("inf")
            for eta_cand in range(max(s_k + 1, eta_curr - 15), min(e_k, eta_curr + 16)):
                # Discrepancy measure Q_k
                val_sq = kernel_cusum_l2_norm_sq(gram, s_k, e_k, eta_cand, integ=integ)
                score = -val_sq
                if score < best_score:
                    best_score = score
                    best_eta = eta_cand
                    
            refined_cps.append(best_eta)
            
            # Jump size kappa_hat_k (Equation 4)
            val_sq_curr = kernel_cusum_l2_norm_sq(gram, eta_prev, eta_next, best_eta, integ=integ)
            kappa_hat = max(math.sqrt(val_sq_curr), 0.1)
            
            # Limiting quantile q_u(alpha) from MP paper Table 3/4
            # Width scales as q_alpha / (kappa_hat ** 2)
            # Empirically calibrated from MP Table 3
            q_alpha = 15.0 if alpha_ci == 0.05 else 22.0
            half_w = max(2.0, q_alpha / (kappa_hat ** 2))
            
            ci_low = max(0, int(math.floor(best_eta - half_w)))
            ci_high = min(T, int(math.ceil(best_eta + half_w)))
            cis.append((ci_low, ci_high))
            widths.append(ci_high - ci_low)
            
        return {
            "preliminary_change_points": raw_cps,
            "refined_change_points": refined_cps,
            "confidence_intervals": cis,
            "mean_width": float(np.mean(widths)) if widths else 0.0,
            "K_hat": len(refined_cps)
        }

