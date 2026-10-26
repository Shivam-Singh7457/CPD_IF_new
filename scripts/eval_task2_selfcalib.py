#!/usr/bin/env python3
"""Run Task 2 full evaluation: Oracle and Self-Calibration across 1000 null streams per phi."""
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


def fast_vectorized_self_cal(
    X_train: np.ndarray,
    p: int,
    T: int,
    m0: int,
    alpha0: float,
    window_grid: tuple[int, ...],
    base_A: float,
    n_sim_streams: int,
    seed: int,
) -> tuple[float, float]:
    mu = X_train.mean(axis=0)
    Xc = X_train - mu

    num = np.sum(Xc[:-1] * Xc[1:], axis=0)
    den = np.sum(Xc[:-1] ** 2, axis=0)
    r = np.clip(np.where(den > 0, num / den, 0.0), -0.95, 0.95).astype(np.float32)

    e = (Xc[1:] - r[np.newaxis, :] * Xc[:-1]).astype(np.float32)
    e_c = e - e.mean(axis=0)
    sigma_eps = (e_c.T @ e_c) / e_c.shape[0]
    sigma_eps = 0.5 * (sigma_eps + sigma_eps.T)
    min_eig = float(np.linalg.eigvalsh(sigma_eps).min())
    if min_eig <= 1e-6:
        sigma_eps += (1e-5 - min_eig) * np.eye(p, dtype=np.float32)

    denom = 1.0 - np.outer(r, r)
    gamma_stat = sigma_eps / np.maximum(denom, 1e-4)
    gamma_stat = 0.5 * (gamma_stat + gamma_stat.T)
    min_eig_g = float(np.linalg.eigvalsh(gamma_stat).min())
    if min_eig_g <= 1e-6:
        gamma_stat += (1e-5 - min_eig_g) * np.eye(p, dtype=np.float32)

    chol_gamma = np.linalg.cholesky(gamma_stat)
    chol_eps = np.linalg.cholesky(sigma_eps)

    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((n_sim_streams, T, p), dtype=np.float32) @ chol_eps.T
    z = np.empty((n_sim_streams, T, p), dtype=np.float32)
    z[:, 0, :] = rng.standard_normal((n_sim_streams, p), dtype=np.float32) @ chol_gamma.T
    for t in range(1, T):
        z[:, t, :] = r * z[:, t - 1, :] + eps[:, t, :]
    sim_obs = mu.astype(np.float32) + z

    sim_train = sim_obs[:, :m0, :]
    mu_sim = sim_train.mean(axis=1)  # (n_sim, p)
    Xc_sim = sim_obs - mu_sim[:, np.newaxis, :]
    train_c = Xc_sim[:, :m0, :]
    g0_sim = np.mean(train_c ** 2, axis=1)  # (n_sim, p)

    num_s = np.sum(train_c[:, :-1, :] * train_c[:, 1:, :], axis=1)
    den_s = np.sum(train_c[:, :-1, :] ** 2, axis=1)
    r_sim = np.clip(np.where(den_s > 0, num_s / den_s, 0.0), -0.95, 0.95)

    e_s = train_c[:, 1:, :] - r_sim[:, np.newaxis, :] * train_c[:, :-1, :]
    omega_sim = np.mean(e_s ** 2, axis=1)
    sigma2_sim = omega_sim / ((1.0 - r_sim) ** 2)

    inv_sqrt_s2 = {}
    for k in window_grid:
        rk = r_sim ** k
        v = sigma2_sim - 2.0 * g0_sim * r_sim * (1.0 - rk) / (k * (1.0 - r_sim) ** 2)
        v = np.maximum(v, 0.2 * g0_sim)
        s2 = k * v + (k ** 2) * sigma2_sim / m0
        inv_sqrt_s2[k] = (1.0 / np.sqrt(s2))[:, np.newaxis, :]  # (n_sim, 1, p)

    mon_c = Xc_sim[:, m0:, :]
    T_mon = mon_c.shape[1]
    C_raw = np.zeros((n_sim_streams, T_mon + 1, p), dtype=np.float32)
    np.cumsum(mon_c, axis=1, out=C_raw[:, 1:, :])

    M_A = np.zeros(n_sim_streams, dtype=np.float32)
    M_Q = np.zeros(n_sim_streams, dtype=np.float32)
    sqrt_2p = np.float32(np.sqrt(2.0 * p))

    for k in window_grid:
        W = (C_raw[:, k:, :] - C_raw[:, :-k, :]) * inv_sqrt_s2[k]
        ak = np.max(np.abs(W), axis=(1, 2))
        qk = np.max((np.sum(W ** 2, axis=2) - p) / sqrt_2p, axis=1)
        M_A = np.maximum(M_A, ak / base_A)
        M_Q = np.maximum(M_Q, qk)

    q_level = 1.0 - alpha0 / 2.0
    c_A = float(np.quantile(M_A, q_level))
    c_Q = float(np.quantile(M_Q, q_level))
    return c_A, c_Q


def process_stream(args: tuple) -> dict:
    idx, X, m0, window_grid, kmax, base_A, oracle_cA, oracle_cQ, n_sim, seed = args
    p = X.shape[1]
    T = X.shape[0]

    # 1. Fit training model on stream's own training segment
    X_train = np.asarray(X[:m0], dtype=np.float64)
    fit = fit_training(X_train, m0=m0, window_grid=window_grid, kmax=kmax)

    # 2. Compute stream's own monitoring statistics
    A, Q, _ = compute_statistics_vectorised(X, fit)
    M_A = float(np.nanmax(A) / base_A)
    M_Q = float(np.nanmax(Q))
    oracle_alarm = bool((M_A > oracle_cA) or (M_Q > oracle_cQ))

    # 3. Self-calibration from stream's own training segment
    self_cA, self_cQ = fast_vectorized_self_cal(
        X_train=X_train,
        p=p,
        T=T,
        m0=m0,
        alpha0=0.05,
        window_grid=window_grid,
        base_A=base_A,
        n_sim_streams=n_sim,
        seed=seed,
    )
    self_alarm = bool((M_A > self_cA) or (M_Q > self_cQ))

    return {
        "stream_idx": idx,
        "M_A": M_A,
        "M_Q": M_Q,
        "oracle_alarm": oracle_alarm,
        "self_cA": self_cA,
        "self_cQ": self_cQ,
        "self_alarm": self_alarm,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-streams", type=int, default=1000)
    parser.add_argument("--n-sim", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-csv", type=str, default="results/tables/task2_full_1000.csv")
    args = parser.parse_args()

    phis = [0.0, 0.3, 0.6, 0.8]
    tags = ["00", "03", "06", "08"]

    summary_rows = []

    for phi, tag in zip(phis, tags):
        dataset_dir = f"data/null1000_phi{tag}"
        cal_path = f"results/calib/proposed_phi{tag}.json"
        cal = CalibrationResult.load_json(cal_path)

        obs = load_observations(dataset_dir, mmap=True)
        N = min(args.n_streams, obs.shape[0])
        print(f"\n==================================================")
        print(f"Running Task 2 for phi={phi} ({N} streams, n_sim={args.n_sim}, workers={args.workers})...")
        t0 = time.perf_counter()

        tasks = []
        for i in range(N):
            seed = 20261014 + int(phi * 10000) + i
            tasks.append((
                i,
                np.asarray(obs[i], dtype=np.float64),
                cal.m0,
                tuple(cal.window_grid),
                cal.kmax,
                cal.base_A,
                cal.c_A,
                cal.c_Q,
                args.n_sim,
                seed,
            ))

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(process_stream, tasks))

        t1 = time.perf_counter()
        df = pd.DataFrame(results)

        oracle_far = float(df["oracle_alarm"].mean())
        oracle_se = float(np.sqrt(oracle_far * (1.0 - oracle_far) / N))

        self_far = float(df["self_alarm"].mean())
        self_se = float(np.sqrt(self_far * (1.0 - self_far) / N))

        print(f"phi={phi} finished in {t1-t0:.1f}s ({(t1-t0)/N*1000:.1f}ms/stream):")
        print(f"  Oracle FAR: {oracle_far:.4f} +/- {oracle_se:.4f} ({int(df['oracle_alarm'].sum())}/{N})")
        print(f"  Self FAR:   {self_far:.4f} +/- {self_se:.4f} ({int(df['self_alarm'].sum())}/{N})")
        print(f"  Mean self_cA: {df['self_cA'].mean():.4f} (oracle: {cal.c_A:.4f})")
        print(f"  Mean self_cQ: {df['self_cQ'].mean():.4f} (oracle: {cal.c_Q:.4f})")

        # Save per-stream results
        per_stream_path = Path(f"results/tables/task2_per_stream_phi{tag}.csv")
        per_stream_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(per_stream_path, index=False)

        summary_rows.append({
            "phi": phi,
            "n_streams": N,
            "oracle_far": oracle_far,
            "oracle_se": oracle_se,
            "self_far": self_far,
            "self_se": self_se,
            "oracle_cA": cal.c_A,
            "oracle_cQ": cal.c_Q,
            "mean_self_cA": float(df["self_cA"].mean()),
            "mean_self_cQ": float(df["self_cQ"].mean()),
            "elapsed_sec": t1 - t0,
        })

    summary_df = pd.DataFrame(summary_rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.out_csv, index=False)
    print(f"\nTask 2 summary written to {args.out_csv}")


if __name__ == "__main__":
    main()
