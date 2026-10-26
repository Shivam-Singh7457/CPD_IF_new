#!/usr/bin/env python3
"""Task 2: Evaluation of 1000 extra null streams per phi with oracle and self-calibration."""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ocdmix.calibrate import CalibrationResult
from ocdmix.fit import fit_training
from ocdmix.io import load_observations
from ocdmix.scan import compute_base_A, compute_statistics_vectorised


def eval_single_stream(args: tuple) -> dict:
    stream_idx, X_stream, m0, window_grid, kmax, base_A, oracle_cA, oracle_cQ, n_sim, seed = args
    p = X_stream.shape[1]
    T = X_stream.shape[0]
    T_mon = T - m0

    # 1. Fit training model on stream's own training segment
    X_train = np.asarray(X_stream[:m0], dtype=np.float64)
    fit = fit_training(X_train, m0=m0, window_grid=window_grid, kmax=kmax)

    # 2. Compute stream's own monitoring statistics
    A, Q, _ = compute_statistics_vectorised(X_stream, fit)
    M_A = float(np.nanmax(A) / base_A)
    M_Q = float(np.nanmax(Q))

    # (a) Oracle alarm check
    oracle_alarm = bool((M_A > oracle_cA) or (M_Q > oracle_cQ))

    # (b) Self-calibration: simulate n_sim paths from fitted model
    # Model parameters
    Xc = X_train - fit.mu
    r = fit.r.astype(np.float32)
    e = (Xc[1:] - fit.r[np.newaxis, :] * Xc[:-1]).astype(np.float32)
    e_c = e - e.mean(axis=0)
    sigma_eps = (e_c.T @ e_c) / e_c.shape[0]
    sigma_eps = 0.5 * (sigma_eps + sigma_eps.T)
    # Ensure positive definiteness
    min_eig = float(np.linalg.eigvalsh(sigma_eps).min())
    if min_eig <= 1e-6:
        sigma_eps += (1e-5 - min_eig) * np.eye(p, dtype=np.float32)
    chol_eps = np.linalg.cholesky(sigma_eps)

    # Stationary covariance for initial state
    denom = 1.0 - np.outer(r, r)
    gamma_stat = sigma_eps / np.maximum(denom, 1e-4)
    gamma_stat = 0.5 * (gamma_stat + gamma_stat.T)
    min_eig_g = float(np.linalg.eigvalsh(gamma_stat).min())
    if min_eig_g <= 1e-6:
        gamma_stat += (1e-5 - min_eig_g) * np.eye(p, dtype=np.float32)
    chol_gamma = np.linalg.cholesky(gamma_stat)

    rng = np.random.default_rng(seed)
    z0 = rng.standard_normal((n_sim, p), dtype=np.float32) @ chol_gamma.T
    eps = rng.standard_normal((n_sim, T_mon, p), dtype=np.float32) @ chol_eps.T

    Z = np.empty((n_sim, T_mon, p), dtype=np.float32)
    Z[:, 0, :] = z0
    for t in range(1, T_mon):
        Z[:, t, :] = Z[:, t - 1, :] * r + eps[:, t, :]

    C = np.zeros((n_sim, T_mon + 1, p), dtype=np.float32)
    np.cumsum(Z, axis=1, out=C[:, 1:, :])

    sim_M_A = np.zeros(n_sim, dtype=np.float32)
    sim_M_Q = np.zeros(n_sim, dtype=np.float32)
    sqrt_2p = np.float32(np.sqrt(2.0 * p))

    for k in window_grid:
        inv_s = (1.0 / np.sqrt(fit.s2_grid[k])).astype(np.float32)
        W = (C[:, k:, :] - C[:, :-k, :]) * inv_s
        A_k = np.max(np.abs(W), axis=(1, 2))
        Q_k = np.max((np.sum(W * W, axis=2) - p) / sqrt_2p, axis=1)
        sim_M_A = np.maximum(sim_M_A, A_k / base_A)
        sim_M_Q = np.maximum(sim_M_Q, Q_k)

    self_cA = float(np.quantile(sim_M_A, 0.975))
    self_cQ = float(np.quantile(sim_M_Q, 0.975))

    self_alarm = bool((M_A > self_cA) or (M_Q > self_cQ))

    return {
        "stream_idx": stream_idx,
        "M_A": M_A,
        "M_Q": M_Q,
        "oracle_cA": oracle_cA,
        "oracle_cQ": oracle_cQ,
        "oracle_alarm": oracle_alarm,
        "self_cA": self_cA,
        "self_cQ": self_cQ,
        "self_alarm": self_alarm,
    }


def run_evaluation(
    phis: list[float] = [0.0, 0.3, 0.6, 0.8],
    tags: list[str] = ["00", "03", "06", "08"],
    n_streams: int = 1000,
    n_sim: int = 100,
    max_workers: int = 4,
    out_csv: str = "results/tables/task2_null1000_evaluation.csv",
) -> pd.DataFrame:
    summary_rows = []

    for phi, tag in zip(phis, tags):
        dataset_dir = f"data/null1000_phi{tag}"
        cal_path = f"results/calib/proposed_phi{tag}.json"
        cal = CalibrationResult.load_json(cal_path)

        obs = load_observations(dataset_dir, mmap=True)
        N = min(n_streams, obs.shape[0])

        print(f"\nEvaluating phi={phi} ({N} streams) using oracle ({cal_path}) & self-calibration (n_sim={n_sim})...")
        t0 = time.perf_counter()

        tasks = []
        for i in range(N):
            X_stream = np.asarray(obs[i], dtype=np.float64)
            seed = 20261014 + int(phi * 10000) + i
            tasks.append((
                i,
                X_stream,
                cal.m0,
                tuple(cal.window_grid),
                cal.kmax,
                cal.base_A,
                cal.c_A,
                cal.c_Q,
                n_sim,
                seed,
            ))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(eval_single_stream, tasks))

        t1 = time.perf_counter()
        print(f"Completed phi={phi} in {t1 - t0:.2f}s ({(t1 - t0) / N * 1000:.1f}ms/stream)")

        df_res = pd.DataFrame(results)
        oracle_far = df_res["oracle_alarm"].mean()
        oracle_se = np.sqrt(oracle_far * (1.0 - oracle_far) / N)

        self_far = df_res["self_alarm"].mean()
        self_se = np.sqrt(self_far * (1.0 - self_far) / N)

        row = {
            "phi": phi,
            "n_streams": N,
            "oracle_far": float(oracle_far),
            "oracle_se": float(oracle_se),
            "self_far": float(self_far),
            "self_se": float(self_se),
            "mean_self_cA": float(df_res["self_cA"].mean()),
            "mean_self_cQ": float(df_res["self_cQ"].mean()),
            "oracle_cA": float(cal.c_A),
            "oracle_cQ": float(cal.c_Q),
        }
        summary_rows.append(row)
        print(
            f"  Oracle FAR: {oracle_far:.4f} +/- {oracle_se:.4f} | "
            f"Self FAR: {self_far:.4f} +/- {self_se:.4f}"
        )

    summary_df = pd.DataFrame(summary_rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSummary saved to: {out_csv}")
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 1000 null streams with oracle & self-calibration.")
    parser.add_argument("--n-streams", type=int, default=1000, help="Number of streams to evaluate per phi.")
    parser.add_argument("--n-sim", type=int, default=100, help="Number of simulation paths for self-calibration.")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes.")
    parser.add_argument("--out-csv", type=str, default="results/tables/task2_null1000_evaluation.csv")
    args = parser.parse_args()

    run_evaluation(
        n_streams=args.n_streams,
        n_sim=args.n_sim,
        max_workers=args.workers,
        out_csv=args.out_csv,
    )
