#!/usr/bin/env python3
"""TASK 1: Diagnostic and Recalibration for Null Calibration (False-Alarm Control).

Runs calibration for set sizes n in [400, 1000, 2000, 4000] and evaluates on 2000 fresh null streams.
Tabulates and plots c_A, c_Q, and out-of-sample FAR against n.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import math
import os
from pathlib import Path
import time

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate_synthetic_streams_fast import (
    _stationary_covariance,
    generate_one_stream,
    make_innovation_covariance,
    make_var_matrix,
)
from ocdmix.calibrate import CalibrationResult
from ocdmix.fit import DEFAULT_KMAX, DEFAULT_M0, DEFAULT_WINDOW_GRID, fit_training
from ocdmix.scan import compute_base_A


def _compute_stream_max_stats(args: tuple) -> tuple[float, float]:
    """Worker function to generate one stream and compute (M_A, M_Q)."""
    (
        stream_idx,
        master_seed,
        scenario_id,
        T,
        p,
        phi,
        A,
        sig_pre,
        gam_pre,
        m0,
        window_grid,
        kmax,
        base_A,
    ) = args

    # Generate stream with unique disjoint seed
    stream_seed = np.random.SeedSequence([master_seed, scenario_id, stream_idx, 2718])
    rng = np.random.default_rng(stream_seed)
    X, _ = generate_one_stream(
        T=T,
        p=p,
        A=A,
        phi=phi,
        sigma_pre=sig_pre,
        gamma_pre=gam_pre,
        rng=rng,
        change_type="none",
        cp_fraction=0.5,
        cp_index=None,
        sparsity=0.1,
        effect=0.5,
        covariance_multiplier=1.5,
    )

    # Training fit
    fit = fit_training(
        X[:m0],
        m0=m0,
        window_grid=window_grid,
        kmax=kmax,
        method="proposed",
        lrv_method="prewhiten",
    )

    # Scan monitoring window
    Xc = X - fit.mu
    monitoring = Xc[m0:]
    n_mon = monitoring.shape[0]
    C_raw = np.zeros((n_mon + 1, p), dtype=np.float64)
    np.cumsum(monitoring, axis=0, out=C_raw[1:])

    max_A = -np.inf
    max_Q = -np.inf
    for k in window_grid:
        sqrt_s2k = np.sqrt(fit.s2_grid[k])
        t_start = m0 + k - 1
        if t_start >= T:
            continue
        t_indices = np.arange(t_start, T)
        ci_t = t_indices - m0 + 1
        ci_tk = t_indices - k - m0 + 1
        W = (C_raw[ci_t] - C_raw[ci_tk]) / sqrt_s2k[np.newaxis, :]
        max_A = max(max_A, float(np.max(np.abs(W))))
        max_Q = max(max_Q, float(np.max((np.sum(W ** 2, axis=1) - p) / np.sqrt(2.0 * p))))

    return max_A / base_A, max_Q


def generate_stats_dataset(
    phi: float,
    scenario_id: int,
    n_streams: int,
    T: int = 5000,
    p: int = 50,
    m0: int = 1000,
    master_seed: int = 20261014,
    n_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate n_streams null streams and return arrays of M_A and M_Q."""
    # Shared model
    model_seed = np.random.SeedSequence([master_seed, scenario_id, 9173])
    rng_model = np.random.default_rng(model_seed)
    A = make_var_matrix(p, phi, "isotropic", rng_model)
    sig_pre = make_innovation_covariance(p, 0.25)
    gam_pre = _stationary_covariance(A, sig_pre, phi)

    window_grid = DEFAULT_WINDOW_GRID
    kmax = DEFAULT_KMAX
    base_A = compute_base_A(p, len(window_grid))

    task_args = [
        (
            idx,
            master_seed,
            scenario_id,
            T,
            p,
            phi,
            A,
            sig_pre,
            gam_pre,
            m0,
            window_grid,
            kmax,
            base_A,
        )
        for idx in range(n_streams)
    ]

    M_A = np.empty(n_streams, dtype=np.float64)
    M_Q = np.empty(n_streams, dtype=np.float64)

    print(f"Generating and scanning {n_streams} streams for phi={phi} (scenario={scenario_id})...")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for idx, (mA, mQ) in enumerate(executor.map(_compute_stream_max_stats, task_args, chunksize=25)):
            M_A[idx] = mA
            M_Q[idx] = mQ
            if (idx + 1) % 1000 == 0 or (idx + 1) == n_streams:
                elapsed = time.perf_counter() - t0
                print(f"  Processed {idx + 1}/{n_streams} in {elapsed:.1f}s ({elapsed / (idx + 1) * 1000:.1f} ms/stream)")

    return M_A, M_Q


def calibrate_thresholds_from_stats(
    M_A: np.ndarray,
    M_Q: np.ndarray,
    alpha0: float = 0.05,
    method: str = "split",
    ucb_level: float = 0.0,
) -> tuple[float, float]:
    """Compute c_A and c_Q from stream statistics arrays.
    
    method: 'split', 'joint', or 'split_ucb' / 'joint_ucb'
    """
    n = len(M_A)
    if method == "split":
        q = 1.0 - alpha0 / 2.0
        if ucb_level > 0:
            # Conservative upper confidence bound for quantile
            # Binomial SE for quantile level q
            z = 1.645  # 95% one-sided UCB
            q_adj = min(1.0 - 1.0 / n, q + z * math.sqrt(q * (1.0 - q) / n))
            cA = float(np.quantile(M_A, q_adj))
            cQ = float(np.quantile(M_Q, q_adj))
        else:
            cA = float(np.quantile(M_A, q))
            cQ = float(np.quantile(M_Q, q))
    elif method == "joint":
        q = 1.0 - alpha0
        if ucb_level > 0:
            z = 1.645
            q_adj = min(1.0 - 1.0 / n, q + z * math.sqrt(q * (1.0 - q) / n))
            cA_marg = float(np.quantile(M_A, q_adj))
            cQ_marg = float(np.quantile(M_Q, q_adj))
            R = np.maximum(M_A / cA_marg, M_Q / cQ_marg)
            kappa = float(np.quantile(R, q_adj))
        else:
            cA_marg = float(np.quantile(M_A, q))
            cQ_marg = float(np.quantile(M_Q, q))
            R = np.maximum(M_A / cA_marg, M_Q / cQ_marg)
            kappa = float(np.quantile(R, q))
        cA = float(cA_marg * kappa)
        cQ = float(cQ_marg * kappa)
    else:
        raise ValueError(f"Unknown method {method}")
    return cA, cQ


def evaluate_far(
    M_A_eval: np.ndarray,
    M_Q_eval: np.ndarray,
    cA: float,
    cQ: float,
) -> tuple[float, float, tuple[float, float]]:
    """Evaluate false alarm rate and 95% Wilson confidence interval."""
    n = len(M_A_eval)
    alarmed = (M_A_eval > cA) | (M_Q_eval > cQ)
    far = float(np.mean(alarmed))
    se = float(math.sqrt(far * (1.0 - far) / n))
    
    # 95% Wilson score interval
    z = 1.95996
    denom = 1.0 + z * z / n
    centre = (far + z * z / (2.0 * n)) / denom
    margin = (z * math.sqrt(far * (1.0 - far) / n + z * z / (4.0 * n * n))) / denom
    ci_lower = max(0.0, centre - margin)
    ci_upper = min(1.0, centre + margin)
    return far, se, (ci_lower, ci_upper)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=4, help="Number of worker processes.")
    args = parser.parse_args()

    out_dir = Path("results/fix")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path("results/fix/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    phis = [0.0, 0.3, 0.6, 0.8]
    phi_tags = ["00", "03", "06", "08"]
    n_calibs = [400, 1000, 2000, 4000]
    n_eval = 2000

    diagnostic_rows = []
    
    # Cache and compute max stats for each phi
    for i, (phi, tag) in enumerate(zip(phis, phi_tags)):
        print(f"\n==================== PHI = {phi} ====================")
        calib_cache = cache_dir / f"stats_calib4000_phi{tag}.npz"
        eval_cache = cache_dir / f"stats_eval2000_phi{tag}.npz"

        # Calibration stats (4000 streams, scenario 5000 + i)
        if calib_cache.exists():
            print(f"Loading cached calibration stats from {calib_cache}...")
            data = np.load(calib_cache)
            M_A_calib, M_Q_calib = data["M_A"], data["M_Q"]
        else:
            M_A_calib, M_Q_calib = generate_stats_dataset(
                phi=phi,
                scenario_id=5000 + i,
                n_streams=4000,
                n_workers=args.n_workers,
            )
            np.savez_compressed(calib_cache, M_A=M_A_calib, M_Q=M_Q_calib)

        # Evaluation stats (2000 fresh streams, scenario 6000 + i)
        if eval_cache.exists():
            print(f"Loading cached evaluation stats from {eval_cache}...")
            data = np.load(eval_cache)
            M_A_eval, M_Q_eval = data["M_A"], data["M_Q"]
        else:
            M_A_eval, M_Q_eval = generate_stats_dataset(
                phi=phi,
                scenario_id=6000 + i,
                n_streams=n_eval,
                n_workers=args.n_workers,
            )
            np.savez_compressed(eval_cache, M_A=M_A_eval, M_Q=M_Q_eval)

        # Recalibrate for each n in [400, 1000, 2000, 4000]
        # and test split, joint, and UCB variants
        for n_cal in n_calibs:
            M_A_sub = M_A_calib[:n_cal]
            M_Q_sub = M_Q_calib[:n_cal]

            for method in ["split", "joint"]:
                for ucb in [0.0, 0.95]:
                    method_label = f"{method}_ucb" if ucb > 0 else method
                    cA, cQ = calibrate_thresholds_from_stats(
                        M_A_sub, M_Q_sub, alpha0=0.05, method=method, ucb_level=ucb
                    )
                    far, se, (ci_lo, ci_hi) = evaluate_far(M_A_eval, M_Q_eval, cA, cQ)
                    ci_contains_005 = (ci_lo <= 0.05 <= ci_hi) or (ci_hi <= 0.05)

                    diagnostic_rows.append({
                        "phi": phi,
                        "n_calib": n_cal,
                        "n_eval": n_eval,
                        "method": method_label,
                        "c_A": cA,
                        "c_Q": cQ,
                        "eval_far": far,
                        "eval_se": se,
                        "ci_95_lower": ci_lo,
                        "ci_95_upper": ci_hi,
                        "ci_contains_target": ci_contains_005,
                    })
                    print(
                        f"phi={phi} | n={n_cal:4d} | {method_label:9s} | cA={cA:.4f}, cQ={cQ:.4f} | "
                        f"FAR={far:.4f} +/- {se:.4f} [95% CI: {ci_lo:.4f}, {ci_hi:.4f}] | target met: {ci_contains_005}"
                    )

    df_diag = pd.DataFrame(diagnostic_rows)
    csv_path = out_dir / "task1_null_calibration_diagnostic.csv"
    df_diag.to_csv(csv_path, index=False)
    print(f"\nSaved Task 1 diagnostic results to {csv_path}")

    # Plot diagnostic figures
    # Plot c_A, c_Q and out-of-sample FAR against n for split and joint
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), sharex="col")
    fig.suptitle("Task 1 Diagnostic: Null Calibration Thresholds and Out-of-Sample FAR vs n", fontsize=16)

    colors = {"split": "tab:blue", "joint": "tab:green", "split_ucb": "tab:red", "joint_ucb": "tab:purple"}

    for col_idx, phi in enumerate(phis):
        sub_df = df_diag[df_diag["phi"] == phi]
        
        # Row 0: c_A vs n
        ax_cA = axes[0, col_idx]
        for m in ["split", "joint"]:
            m_sub = sub_df[sub_df["method"] == m]
            ax_cA.plot(m_sub["n_calib"], m_sub["c_A"], marker="o", label=m, color=colors[m])
        ax_cA.set_title(f"$\phi = {phi}$")
        ax_cA.set_ylabel("$c_A$")
        ax_cA.grid(True, alpha=0.3)
        if col_idx == 0:
            ax_cA.legend()

        # Row 1: c_Q vs n
        ax_cQ = axes[1, col_idx]
        for m in ["split", "joint"]:
            m_sub = sub_df[sub_df["method"] == m]
            ax_cQ.plot(m_sub["n_calib"], m_sub["c_Q"], marker="s", label=m, color=colors[m])
        ax_cQ.set_ylabel("$c_Q$")
        ax_cQ.grid(True, alpha=0.3)

        # Row 2: FAR vs n
        ax_far = axes[2, col_idx]
        ax_far.axhline(0.05, color="black", linestyle="--", label="Target $\\alpha_0=0.05$")
        for m in ["split", "joint"]:
            m_sub = sub_df[sub_df["method"] == m]
            ax_far.errorbar(
                m_sub["n_calib"],
                m_sub["eval_far"],
                yerr=1.96 * m_sub["eval_se"],
                marker="o",
                capsize=4,
                label=f"{m} (95% CI)",
                color=colors[m],
            )
        ax_far.set_ylabel("Out-of-sample FAR")
        ax_far.set_xlabel("Calibration Set Size $n$")
        ax_far.grid(True, alpha=0.3)
        ax_far.set_ylim(0.0, 0.12)
        if col_idx == 0:
            ax_far.legend()

    plt.tight_layout()
    fig_path = out_dir / "task1_diagnostic_plot.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved diagnostic plot to {fig_path}")


if __name__ == "__main__":
    main()
