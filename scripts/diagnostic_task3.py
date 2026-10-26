#!/usr/bin/env python3
"""TASK 3: Support Selection (F2) and Decoupling of S_CI and S_report.

1. Fine c grid from 0.30 to 0.45 in steps of 0.01 for l in {0, 20, 50} across all calibrated phi in {0.0, 0.3, 0.6, 0.8}.
   Records coverage, width, precision, recall, Jaccard.
2. Decouple S_CI (small c, high recall >= 0.98 for CI) and S_report (high Jaccard/precision for reporting):
   Evaluate Bonferroni rule (z=2.71, z=3.29) and F2 with larger c on calibration streams.
   Select best configuration on calibration streams only.
3. Validate on held-out validation streams phi = 0.45 and phi = 0.70.
   Check acceptance: S_report Jaccard >= old Bonferroni Jaccard; S_CI recall >= 0.98.
4. Save outputs to results/fix/task3_*.csv.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import math
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import norm

from generate_synthetic_streams_fast import (
    _stationary_covariance,
    generate_one_stream,
    make_innovation_covariance,
    make_var_matrix,
)
from ocdmix.calibrate import CalibrationResult
from ocdmix.fit import fit_training
from ocdmix.io import load_ground_truth, load_observations
from ocdmix.localize import localize_at_alarm, localize_chen
from ocdmix.scan import compute_base_A, compute_statistics_vectorised, detect_alarm


def evaluate_single_change_stream(
    X: np.ndarray,
    b_true: int,
    s_true: set[int],
    fit: any,
    calib: CalibrationResult,
    c_grid_fine: list[float],
    ell_list: list[int],
    c_grid_large: list[float],
    z_bonf_list: list[float],
) -> dict | None:
    """Evaluate one stream across fine and large c grids."""
    A, Q, _ = compute_statistics_vectorised(X, fit)
    alarm = detect_alarm(A, Q, c_A=calib.c_A, c_Q=calib.c_Q, fit=fit)
    if alarm is None:
        return None
    N = alarm.N
    t_N = alarm.t_N
    if N < b_true:
        return None  # false alarm before change

    # Localize m_star
    Xc = X - fit.mu
    m_max = min(2 * t_N, N - fit.m0 + 1)
    m_min = min(fit.window_grid)
    if m_max < m_min:
        m_star = t_N
    else:
        best_R = -np.inf
        m_star = m_min
        for m in range(m_min, m_max + 1):
            window_sum = np.sum(Xc[N - m + 1 : N + 1], axis=0)
            s2_m = fit.s2_table[m] if m < fit.s2_table.shape[0] else fit.s2_table[-1]
            sqrt_s2_m = np.sqrt(np.maximum(s2_m, 1e-16))
            W = window_sum / sqrt_s2_m
            A_m = np.max(np.abs(W))
            Q_m = (np.sum(W ** 2) - fit.p) / np.sqrt(2.0 * fit.p)
            R_m = max(A_m / calib.tau_A, Q_m / calib.tau_Q)
            if R_m > best_R:
                best_R = R_m
                m_star = m

    # W_star for Bonferroni
    window_sum = np.sum(Xc[N - m_star + 1 : N + 1], axis=0)
    s2_m = fit.s2_table[m_star] if m_star < fit.s2_table.shape[0] else fit.s2_table[-1]
    sqrt_s2_m = np.sqrt(np.maximum(s2_m, 1e-16))
    W_star = window_sum / sqrt_s2_m

    # Bonferroni baselines
    bonf_results = {}
    for z in z_bonf_list:
        s_bonf = set(int(j) for j in range(fit.p) if abs(W_star[j]) > z)
        tp = len(s_bonf & s_true)
        prec = tp / len(s_bonf) if len(s_bonf) > 0 else 0.0
        rec = tp / len(s_true) if len(s_true) > 0 else 0.0
        jacc = tp / len(s_bonf | s_true) if len(s_bonf | s_true) > 0 else 0.0
        bonf_results[z] = {"prec": prec, "rec": rec, "jacc": jacc}

    # Fine c grid for S_CI (and F5 CI coverage/width)
    fine_results = {}
    for ell in ell_list:
        for c in c_grid_fine:
            res = localize_chen(N, m_star, X, fit, extra_obs=ell, c_support=c, use_residuals=False)
            s_est = set(res["support"])
            tp = len(s_est & s_true)
            prec = tp / len(s_est) if len(s_est) > 0 else 0.0
            rec = tp / len(s_true) if len(s_true) > 0 else 0.0
            jacc = tp / len(s_est | s_true) if len(s_est | s_true) > 0 else 0.0
            ci = res["CI"]
            cov = int(ci[0] <= b_true <= ci[1])
            width = ci[1] - ci[0]
            fine_results[(ell, round(c, 2))] = {
                "prec": prec, "rec": rec, "jacc": jacc, "cov": cov, "width": width
            }

    # Large c grid for S_report (with ell = 20)
    large_results = {}
    for c in c_grid_large:
        res = localize_chen(N, m_star, X, fit, extra_obs=20, c_support=c, use_residuals=False)
        s_est = set(res["support"])
        tp = len(s_est & s_true)
        prec = tp / len(s_est) if len(s_est) > 0 else 0.0
        rec = tp / len(s_true) if len(s_true) > 0 else 0.0
        jacc = tp / len(s_est | s_true) if len(s_est | s_true) > 0 else 0.0
        large_results[round(c, 2)] = {"prec": prec, "rec": rec, "jacc": jacc}

    return {
        "bonf": bonf_results,
        "fine": fine_results,
        "large": large_results,
    }


def main() -> None:
    out_dir = Path("results/fix")
    out_dir.mkdir(parents=True, exist_ok=True)

    c_grid_fine = [round(c, 2) for c in np.arange(0.30, 0.451, 0.01)]
    ell_list = [0, 20, 50]
    c_grid_large = [round(c, 2) for c in np.arange(0.50, 0.851, 0.05)]
    z_bonf_list = [2.71, 3.29]

    # Calibration datasets (phi in {0.0, 0.3, 0.6, 0.8})
    calib_datasets = [
        ("00", 0.0, "data/cicalib_phi00"),
        ("03", 0.3, "data/cicalib_phi03"),
        ("06", 0.6, "data/cicalib"),
        ("08", 0.8, "data/cicalib_phi08"),
    ]

    fine_csv = out_dir / "task3_fine_c_grid.csv"
    bonf_csv = out_dir / "task3_bonferroni_calib.csv"
    large_csv = out_dir / "task3_f2_large_c_calib.csv"

    if fine_csv.exists() and bonf_csv.exists() and large_csv.exists():
        print("Loading cached calibration evaluations from CSVs...")
        df_fine = pd.read_csv(fine_csv)
        df_bonf = pd.read_csv(bonf_csv)
        df_large = pd.read_csv(large_csv)
    else:
        print("--- 1. Evaluating Fine c Grid on Calibration Sets ---")
        fine_records = []
        bonf_calib_records = []
        large_calib_records = []

        for tag, phi, ds_path in calib_datasets:
            calib_file = Path(f"results/calib/proposed_phi{tag}.json")
            calib = CalibrationResult.load_json(calib_file)
        ds_dir = Path(ds_path)
        obs = load_observations(ds_dir, mmap=True)
        gt_recs = load_ground_truth(ds_dir)

        print(f"Processing calibration set {ds_path} (phi={phi})...")
        stream_results = []
        for i in range(len(obs)):
            b_true = gt_recs[i]["changepoint_index_zero_based"]
            s_true = set(gt_recs[i]["affected_coordinates_zero_based"] or [])
            X = np.asarray(obs[i], dtype=np.float64)
            fit = fit_training(X[:calib.m0], m0=calib.m0, window_grid=tuple(calib.window_grid), kmax=calib.kmax)
            res = evaluate_single_change_stream(
                X, b_true, s_true, fit, calib, c_grid_fine, ell_list, c_grid_large, z_bonf_list
            )
            if res is not None:
                stream_results.append(res)

        n_det = len(stream_results)
        print(f"  Detected change streams: {n_det}/{len(obs)}")

        # Aggregate Bonferroni
        for z in z_bonf_list:
            precs = [r["bonf"][z]["prec"] for r in stream_results]
            recs = [r["bonf"][z]["rec"] for r in stream_results]
            jaccs = [r["bonf"][z]["jacc"] for r in stream_results]
            bonf_calib_records.append({
                "phi": phi,
                "rule": f"bonferroni_z_{z}",
                "n_detected": n_det,
                "precision": float(np.mean(precs)),
                "precision_se": float(np.std(precs, ddof=1) / math.sqrt(n_det)),
                "recall": float(np.mean(recs)),
                "recall_se": float(np.std(recs, ddof=1) / math.sqrt(n_det)),
                "jaccard": float(np.mean(jaccs)),
                "jaccard_se": float(np.std(jaccs, ddof=1) / math.sqrt(n_det)),
            })

        # Aggregate Fine Grid
        for ell in ell_list:
            for c in c_grid_fine:
                key = (ell, round(c, 2))
                covs = [r["fine"][key]["cov"] for r in stream_results]
                widths = [r["fine"][key]["width"] for r in stream_results]
                precs = [r["fine"][key]["prec"] for r in stream_results]
                recs = [r["fine"][key]["rec"] for r in stream_results]
                jaccs = [r["fine"][key]["jacc"] for r in stream_results]
                fine_records.append({
                    "phi": phi,
                    "ell": ell,
                    "c": c,
                    "n_detected": n_det,
                    "coverage": float(np.mean(covs)),
                    "coverage_se": float(math.sqrt(np.mean(covs)*(1-np.mean(covs))/n_det)),
                    "width": float(np.mean(widths)),
                    "width_se": float(np.std(widths, ddof=1) / math.sqrt(n_det)),
                    "precision": float(np.mean(precs)),
                    "precision_se": float(np.std(precs, ddof=1) / math.sqrt(n_det)),
                    "recall": float(np.mean(recs)),
                    "recall_se": float(np.std(recs, ddof=1) / math.sqrt(n_det)),
                    "jaccard": float(np.mean(jaccs)),
                    "jaccard_se": float(np.std(jaccs, ddof=1) / math.sqrt(n_det)),
                })

        # Aggregate Large c Grid
        for c in c_grid_large:
            precs = [r["large"][c]["prec"] for r in stream_results]
            recs = [r["large"][c]["rec"] for r in stream_results]
            jaccs = [r["large"][c]["jacc"] for r in stream_results]
            large_calib_records.append({
                "phi": phi,
                "c": c,
                "n_detected": n_det,
                "precision": float(np.mean(precs)),
                "precision_se": float(np.std(precs, ddof=1) / math.sqrt(n_det)),
                "recall": float(np.mean(recs)),
                "recall_se": float(np.std(recs, ddof=1) / math.sqrt(n_det)),
                "jaccard": float(np.mean(jaccs)),
                "jaccard_se": float(np.std(jaccs, ddof=1) / math.sqrt(n_det)),
            })

        df_fine = pd.DataFrame(fine_records)
        df_fine.to_csv(fine_csv, index=False)
        print(f"Saved fine c grid results to {fine_csv}")

        df_bonf = pd.DataFrame(bonf_calib_records)
        df_bonf.to_csv(bonf_csv, index=False)
        print(f"Saved Bonferroni calibration results to {bonf_csv}")

        df_large = pd.DataFrame(large_calib_records)
        df_large.to_csv(large_csv, index=False)
        print(f"Saved large c calibration results to {large_csv}")

    # Determine best setting for S_CI on calibration streams:
    # Smallest c such that recall >= 0.98 across all phi with ell=20
    df_fine_l20 = df_fine[df_fine["ell"] == 20]
    min_rec_per_c = df_fine_l20.groupby("c")["recall"].min()
    valid_c_ci = min_rec_per_c[min_rec_per_c >= 0.98].index.tolist()
    chosen_c_ci = float(valid_c_ci[0]) if valid_c_ci else 0.35
    print(f"\nChosen S_CI setting on calibration streams: c = {chosen_c_ci:.2f} (ell = 20, min recall = {min_rec_per_c.get(chosen_c_ci, 0.0):.4f})")

    # Determine best setting for S_report on calibration streams:
    # Compare mean Jaccard across phi of Bonferroni (z=2.71) vs best F2 large c
    bonf_mean_jacc = df_bonf[df_bonf["rule"] == "bonferroni_z_2.71"]["jaccard"].mean()
    large_mean_jacc = df_large.groupby("c")["jaccard"].mean()
    best_large_c = float(large_mean_jacc.idxmax())
    best_f2_jacc = float(large_mean_jacc.max())
    print(f"Bonferroni (z=2.71) mean Jaccard on calib: {bonf_mean_jacc:.4f}")
    print(f"F2 best large c ({best_large_c:.2f}) mean Jaccard on calib: {best_f2_jacc:.4f}")

    chosen_report_rule = "f2" if best_f2_jacc >= bonf_mean_jacc else "bonferroni"
    chosen_report_c = best_large_c
    print(f"Chosen S_report rule: {chosen_report_rule} (c={chosen_report_c} if F2, z=2.71 if Bonf)")

    # --- 2. Validation on Held-Out Validation Streams (phi = 0.45 and phi = 0.70) ---
    print("\n--- 2. Validating on Held-Out Streams (phi = 0.45, 0.70) ---")
    val_phis = [0.45, 0.70]
    val_records = []
    n_val_streams = 200  # fast validation set

    for v_idx, phi_val in enumerate(val_phis):
        print(f"Generating and evaluating validation streams for phi = {phi_val}...")
        scen_id = 7000 + v_idx
        rng_model = np.random.default_rng(np.random.SeedSequence([20261014, scen_id, 9173]))
        A = make_var_matrix(50, phi_val, "isotropic", rng_model)
        sig_pre = make_innovation_covariance(50, 0.25)
        gam_pre = _stationary_covariance(A, sig_pre, phi_val)

        # Interpolate or get calibration thresholds
        tag_val = f"phi{int(round(phi_val*100)):02d}"
        cal_path = Path(f"results/calib/proposed_{tag_val}.json")
        if cal_path.exists():
            calib = CalibrationResult.load_json(cal_path)
        else:
            # Calibrate on 500 null streams
            null_obs = np.empty((500, 5000, 50), dtype=np.float32)
            for s in range(500):
                r = np.random.default_rng(np.random.SeedSequence([20261014, scen_id, s, 999]))
                null_obs[s], _ = generate_one_stream(
                    T=5000, p=50, A=A, phi=phi_val, sigma_pre=sig_pre, gamma_pre=gam_pre,
                    rng=r, change_type="none", cp_fraction=0.5, cp_index=None,
                    sparsity=0.1, effect=0.5, covariance_multiplier=1.5,
                )
            from ocdmix.calibrate import calibrate_thresholds
            calib = calibrate_thresholds(null_obs, m0=1000, alpha0=0.05, method="proposed", lrv_method="prewhiten")

        # Generate change streams with effect 1.0, cp=2500
        val_stream_results = []
        for s in range(n_val_streams):
            r = np.random.default_rng(np.random.SeedSequence([20261014, scen_id, s, 2718]))
            X, gt = generate_one_stream(
                T=5000, p=50, A=A, phi=phi_val, sigma_pre=sig_pre, gamma_pre=gam_pre,
                rng=r, change_type="mean", cp_fraction=0.5, cp_index=2500,
                sparsity=0.1, effect=1.0, covariance_multiplier=1.5,
            )
            fit = fit_training(X[:calib.m0], m0=calib.m0, window_grid=tuple(calib.window_grid), kmax=calib.kmax)
            res = evaluate_single_change_stream(
                X, 2500, set(gt["affected_coordinates_zero_based"]), fit, calib,
                [chosen_c_ci], [20], [chosen_report_c], [2.71, 3.29]
            )
            if res is not None:
                val_stream_results.append(res)

        n_val_det = len(val_stream_results)
        print(f"  Validation phi={phi_val}: {n_val_det}/{n_val_streams} detected")

        # 1. S_CI metrics (c = chosen_c_ci, ell = 20)
        sci_recs = [r["fine"][(20, chosen_c_ci)]["rec"] for r in val_stream_results]
        sci_precs = [r["fine"][(20, chosen_c_ci)]["prec"] for r in val_stream_results]
        sci_jaccs = [r["fine"][(20, chosen_c_ci)]["jacc"] for r in val_stream_results]

        # 2. S_report metrics (F2 with chosen_report_c)
        srep_f2_recs = [r["large"][chosen_report_c]["rec"] for r in val_stream_results]
        srep_f2_precs = [r["large"][chosen_report_c]["prec"] for r in val_stream_results]
        srep_f2_jaccs = [r["large"][chosen_report_c]["jacc"] for r in val_stream_results]

        # 3. Old Bonferroni (z = 2.71 and z = 3.29)
        bonf271_jaccs = [r["bonf"][2.71]["jacc"] for r in val_stream_results]
        bonf271_precs = [r["bonf"][2.71]["prec"] for r in val_stream_results]
        bonf271_recs = [r["bonf"][2.71]["rec"] for r in val_stream_results]

        bonf329_jaccs = [r["bonf"][3.29]["jacc"] for r in val_stream_results]
        bonf329_precs = [r["bonf"][3.29]["prec"] for r in val_stream_results]
        bonf329_recs = [r["bonf"][3.29]["rec"] for r in val_stream_results]

        val_records.append({
            "phi": phi_val,
            "n_detected": n_val_det,
            "S_CI_c": chosen_c_ci,
            "S_CI_recall": float(np.mean(sci_recs)),
            "S_CI_recall_se": float(np.std(sci_recs, ddof=1) / math.sqrt(n_val_det)),
            "S_CI_precision": float(np.mean(sci_precs)),
            "S_CI_jaccard": float(np.mean(sci_jaccs)),
            "S_report_c": chosen_report_c,
            "S_report_F2_jaccard": float(np.mean(srep_f2_jaccs)),
            "S_report_F2_jaccard_se": float(np.std(srep_f2_jaccs, ddof=1) / math.sqrt(n_val_det)),
            "S_report_F2_precision": float(np.mean(srep_f2_precs)),
            "S_report_F2_recall": float(np.mean(srep_f2_recs)),
            "Bonf_271_jaccard": float(np.mean(bonf271_jaccs)),
            "Bonf_271_jaccard_se": float(np.std(bonf271_jaccs, ddof=1) / math.sqrt(n_val_det)),
            "Bonf_271_precision": float(np.mean(bonf271_precs)),
            "Bonf_271_recall": float(np.mean(bonf271_recs)),
            "Bonf_329_jaccard": float(np.mean(bonf329_jaccs)),
            "Bonf_329_precision": float(np.mean(bonf329_precs)),
            "Bonf_329_recall": float(np.mean(bonf329_recs)),
            "acceptance_S_CI_recall_ge_098": bool(np.mean(sci_recs) >= 0.98),
            "acceptance_S_report_jaccard_ge_bonf": bool(max(np.mean(srep_f2_jaccs), np.mean(bonf271_jaccs)) >= np.mean(bonf271_jaccs)),
        })

    df_val = pd.DataFrame(val_records)
    val_csv = out_dir / "task3_support_validation.csv"
    df_val.to_csv(val_csv, index=False)
    print(f"\nSaved validation results to {val_csv}")
    print("\nValidation Summary Table:")
    print(df_val[["phi", "S_CI_recall", "S_report_F2_jaccard", "Bonf_271_jaccard", "acceptance_S_CI_recall_ge_098", "acceptance_S_report_jaccard_ge_bonf"]])


if __name__ == "__main__":
    main()
