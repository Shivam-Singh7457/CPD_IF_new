#!/usr/bin/env python3
"""Calibrate c for Chen-Wang-Samworth (Eq. 16-17) on separate calibration change streams.

F2: Support S = { j : |E_j| - b_min*sqrt(m*+l) >= d1 }, d1 = c*sqrt(log(p/alpha)),
    c calibrated on separate simulated change streams (start at c = 0.5, grid 0.3..1.0).
    Compare with the current Bonferroni rule in terms of precision, recall, Jaccard.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from ocdmix.calibrate import CalibrationResult
from ocdmix.fit import fit_training
from ocdmix.io import load_ground_truth, load_observations
from ocdmix.localize import localize_at_alarm, localize_chen
from ocdmix.scan import compute_statistics_vectorised, detect_alarm

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def evaluate_stream(
    X: np.ndarray,
    gt: dict,
    fit: any,
    calib: CalibrationResult,
    c_grid: list[float],
    ell_list: list[int],
    b_min: float = 0.1,
    alpha: float = 0.05,
) -> list[dict] | None:
    b_true = gt["changepoint_index_zero_based"]
    s_true = set(gt.get("affected_coordinates_zero_based") or [])
    if b_true is None:
        return None

    A, Q, _ = compute_statistics_vectorised(X, fit)
    alarm = detect_alarm(A, Q, c_A=calib.c_A, c_Q=calib.c_Q, fit=fit)
    if alarm is None:
        return None

    N = alarm.N
    t_N = alarm.t_N
    # Check if alarm occurred before the change point (false alarm)
    if N < b_true:
        return None

    delay = N - b_true

    # Baseline Bonferroni and heuristic CI
    nu_hat, ci_old, s_old = localize_at_alarm(
        N, t_N, X, fit,
        tau_A=calib.tau_A, tau_Q=calib.tau_Q,
        c_A=calib.c_A, c_Q=calib.c_Q,
        ci_a=0.5,
    )
    m_star = N - nu_hat + 1

    s_old_set = set(s_old)
    tp_old = len(s_old_set & s_true)
    prec_old = tp_old / len(s_old_set) if len(s_old_set) > 0 else 0.0
    rec_old = tp_old / len(s_true) if len(s_true) > 0 else 0.0
    jacc_old = tp_old / len(s_old_set | s_true) if len(s_old_set | s_true) > 0 else 0.0
    cov_old = int(ci_old[0] <= b_true <= ci_old[1])
    width_old = ci_old[1] - ci_old[0]

    rows = []
    for ell in ell_list:
        for c_val in c_grid:
            # 1. Chen with zeta
            res_zeta = localize_chen(
                N=N, m_star=m_star, X=X, fit=fit,
                extra_obs=ell, alpha=alpha, c_support=c_val, b_min=b_min,
                use_residuals=False,
            )
            s_zeta = set(res_zeta["support"])
            tp_z = len(s_zeta & s_true)
            prec_z = tp_z / len(s_zeta) if len(s_zeta) > 0 else 0.0
            rec_z = tp_z / len(s_true) if len(s_true) > 0 else 0.0
            jacc_z = tp_z / len(s_zeta | s_true) if len(s_zeta | s_true) > 0 else 0.0
            ci_z = res_zeta["CI"]
            cov_z = int(ci_z[0] <= b_true <= ci_z[1])
            width_z = ci_z[1] - ci_z[0]

            # 2. Chen with residuals
            res_res = localize_chen(
                N=N, m_star=m_star, X=X, fit=fit,
                extra_obs=ell, alpha=alpha, c_support=c_val, b_min=b_min,
                use_residuals=True,
            )
            s_res = set(res_res["support"])
            tp_r = len(s_res & s_true)
            prec_r = tp_r / len(s_res) if len(s_res) > 0 else 0.0
            rec_r = tp_r / len(s_true) if len(s_true) > 0 else 0.0
            jacc_r = tp_r / len(s_res | s_true) if len(s_res | s_true) > 0 else 0.0
            ci_r = res_res["CI"]
            cov_r = int(ci_r[0] <= b_true <= ci_r[1])
            width_r = ci_r[1] - ci_r[0]

            rows.append({
                "ell": ell,
                "c": c_val,
                "delay": delay,
                # Old
                "cov_old": cov_old,
                "width_old": width_old,
                "prec_old": prec_old,
                "rec_old": rec_old,
                "jacc_old": jacc_old,
                # Zeta
                "cov_zeta": cov_z,
                "width_zeta": width_z,
                "prec_zeta": prec_z,
                "rec_zeta": rec_z,
                "jacc_zeta": jacc_z,
                # Resid
                "cov_resid": cov_r,
                "width_resid": width_r,
                "prec_resid": prec_r,
                "rec_resid": rec_r,
                "jacc_resid": jacc_r,
            })
    return rows


def calibrate_on_dataset(dataset_dir: Path, calib_json: Path, c_grid: list[float], ell_list: list[int]):
    obs = load_observations(dataset_dir, mmap=True)
    gt_records = load_ground_truth(dataset_dir)
    n_streams, T, p = obs.shape
    calib = CalibrationResult.load_json(calib_json)
    m0 = calib.m0

    all_rows = []
    for i in range(n_streams):
        X = np.asarray(obs[i], dtype=np.float64)
        fit = fit_training(
            X[:m0], m0=m0, window_grid=tuple(calib.window_grid),
            kmax=calib.kmax, method=calib.method, lrv_method=calib.lrv_method,
        )
        res = evaluate_stream(X, gt_records[i], fit, calib, c_grid, ell_list)
        if res:
            all_rows.extend(res)

    df = pd.DataFrame(all_rows)
    summary = df.groupby(["ell", "c"]).agg({
        "cov_old": "mean",
        "width_old": "mean",
        "prec_old": "mean",
        "rec_old": "mean",
        "jacc_old": "mean",
        "cov_zeta": "mean",
        "width_zeta": "mean",
        "prec_zeta": "mean",
        "rec_zeta": "mean",
        "jacc_zeta": "mean",
        "cov_resid": "mean",
        "width_resid": "mean",
        "prec_resid": "mean",
        "rec_resid": "mean",
        "jacc_resid": "mean",
        "delay": "mean",
    }).reset_index()
    return summary


def main():
    c_grid = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ell_list = [0, 20, 50]

    scenarios = [
        ("cicalib_phi00", PROJECT_ROOT / "data/cicalib_phi00", PROJECT_ROOT / "results/calib/proposed_phi00.json"),
        ("cicalib_phi03", PROJECT_ROOT / "data/cicalib_phi03", PROJECT_ROOT / "results/calib/proposed_phi03.json"),
        ("cicalib_phi06", PROJECT_ROOT / "data/cicalib", PROJECT_ROOT / "results/calib/proposed_phi06.json"),
        ("cicalib_phi08", PROJECT_ROOT / "data/cicalib_phi08", PROJECT_ROOT / "results/calib/proposed_phi08.json"),
    ]

    out_dir = PROJECT_ROOT / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, ds_dir, calib_file in scenarios:
        if not ds_dir.exists() or not calib_file.exists():
            print(f"Skipping {name}: path not found")
            continue
        print(f"--- Calibrating on {name} ---")
        summary = calibrate_on_dataset(ds_dir, calib_file, c_grid, ell_list)
        out_csv = out_dir / f"chen_calibration_{name}.csv"
        summary.to_csv(out_csv, index=False)
        print(f"Saved to {out_csv}")
        # Print l=20 results
        s20 = summary[summary["ell"] == 20]
        print(s20[["c", "cov_zeta", "width_zeta", "jacc_zeta", "prec_zeta", "rec_zeta", "cov_resid", "width_resid", "jacc_resid"]])


if __name__ == "__main__":
    main()
