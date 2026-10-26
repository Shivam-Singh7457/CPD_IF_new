#!/usr/bin/env python3
"""Calibrate the confidence interval half-width multiplier 'a' on cicalib (Task 5)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ocdmix.calibrate import CalibrationResult
from ocdmix.fit import fit_training
from ocdmix.io import load_ground_truth, load_observations
from ocdmix.localize import localize_at_alarm
from ocdmix.scan import compute_statistics_vectorised, detect_alarm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate CI half-width parameter 'a' on cicalib.")
    parser.add_argument("--dataset-dir", type=str, default="data/cicalib", help="Path to cicalib dataset directory.")
    parser.add_argument("--calib-json", type=str, required=True, help="Path to calibrated thresholds JSON.")
    parser.add_argument("--target-coverage", type=float, default=0.95, help="Target empirical coverage (default: 0.95).")
    parser.add_argument("--out", type=str, required=True, help="Path to save ci_a JSON.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    obs = load_observations(dataset_dir, mmap=True)
    gt = load_ground_truth(dataset_dir)
    n_streams = obs.shape[0]

    calib = CalibrationResult.load_json(args.calib_json)
    m0 = calib.m0

    print(f"Loaded cicalib dataset from {dataset_dir}: {n_streams} streams")
    print(f"Using thresholds from {args.calib_json}: c_A={calib.c_A:.4f}, c_Q={calib.c_Q:.4f}")

    # Collect (m_hat, nu_hat, b, N) for all detected streams
    detected_data: list[tuple[int, int, int, int]] = []

    for i in range(n_streams):
        b = gt[i]["changepoint_index_zero_based"]
        if b is None:
            continue

        X = np.asarray(obs[i], dtype=np.float64)
        fit = fit_training(
            X[:m0],
            m0=m0,
            window_grid=tuple(calib.window_grid),
            kmax=calib.kmax,
            method=calib.method,
            lrv_method=calib.lrv_method,
        )

        A, Q, _ = compute_statistics_vectorised(X, fit)
        alarm = detect_alarm(A, Q, calib.c_A, calib.c_Q, fit)

        if alarm is None or alarm.N < b:
            continue  # undetected or false alarm before change

        # E12: localize to get nu_hat
        nu_hat, _, _ = localize_at_alarm(
            N=alarm.N,
            t_N=alarm.t_N,
            X=X,
            fit=fit,
            tau_A=calib.tau_A,
            tau_Q=calib.tau_Q,
            c_A=calib.c_A,
            c_Q=calib.c_Q,
            ci_a=0.0,
        )
        m_hat = alarm.N - nu_hat + 1
        detected_data.append((m_hat, nu_hat, b, alarm.N))

    n_det = len(detected_data)
    print(f"Number of detected streams for CI calibration: {n_det}")
    if n_det == 0:
        raise RuntimeError("No streams were detected in cicalib; cannot calibrate CI.")

    # Find smallest 'a' giving empirical coverage >= target_coverage
    # Evaluate over fine grid
    a_grid = np.linspace(0.0, 10.0, 10001)  # step 0.001
    best_a = float(a_grid[-1])
    achieved_cov = 1.0

    for a_val in a_grid:
        covered = 0
        for m_hat, nu_hat, b, N in detected_data:
            hw = math.ceil(a_val * m_hat)
            ci_lo = max(m0, nu_hat - hw)
            ci_hi = min(N, nu_hat + hw)
            if ci_lo <= b <= ci_hi:
                covered += 1
        cov = covered / n_det
        if cov >= args.target_coverage:
            best_a = float(a_val)
            achieved_cov = float(cov)
            break

    result = {
        "ci_a": best_a,
        "target_coverage": args.target_coverage,
        "achieved_coverage": achieved_cov,
        "n_detected_streams": n_det,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nCalibrated CI Parameter:")
    print(f"  ci_a              : {best_a:.4f}")
    print(f"  Achieved Coverage : {achieved_cov * 100:.2f}% (target: {args.target_coverage * 100:.1f}%)")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
