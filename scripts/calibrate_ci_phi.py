#!/usr/bin/env python3
"""Calibrate CI scalar 'a' separately at each phi and fit a(r_hat) (Task 5).

Requirements:
1. Calibrate CI scalar 'a' separately at each phi on a dedicated set.
2. Fit a as a function of the mean training AR(1) coefficient r_hat.
3. Report coverage with a(r_hat) on the evaluation sets.
4. Report the median |nu_hat - b| next to the CI width.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd

from ocdmix.calibrate import CalibrationResult
from ocdmix.fit import fit_training
from ocdmix.io import load_ground_truth, load_observations
from ocdmix.localize import localize_at_alarm
from ocdmix.scan import compute_statistics_vectorised, detect_alarm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FAST_GEN = PROJECT_ROOT / "generate_synthetic_streams_fast.py"


def ensure_cicalib_datasets() -> dict[str, Path]:
    """Ensure dedicated CI calibration datasets exist for all phis."""
    phis = {
        "00": (0.0, 901),
        "03": (0.3, 902),
        "06": (0.6, 900),  # data/cicalib
        "08": (0.8, 903),
    }
    paths = {}

    for tag, (phi_val, scen_id) in phis.items():
        if tag == "06":
            ds_path = PROJECT_ROOT / "data" / "cicalib"
        else:
            ds_path = PROJECT_ROOT / "data" / f"cicalib_phi{tag}"

        paths[tag] = ds_path

        if not (ds_path / "observations.npy").exists():
            print(f"Generating dedicated CI calibration set: {ds_path} (phi={phi_val}, scenario={scen_id})...")
            cmd = [
                sys.executable, str(FAST_GEN),
                "--outdir", str(ds_path),
                "--phi", str(phi_val),
                "--p", "50",
                "--T", "5000",
                "--n-streams", "200",
                "--scenario-id", str(scen_id),
                "--change-type", "mean",
                "--effect", "1.0",
                "--cp-index", "2500",
                "--seed", "20261014",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"Failed to generate {ds_path}:\n{proc.stderr}")

    return paths


def collect_ci_detection_data(
    dataset_dir: Path,
    calib_json: Path,
) -> list[dict[str, Any]]:
    """Run detector on dataset and collect localization info and mean r_hat for detected streams."""
    obs = load_observations(dataset_dir, mmap=True)
    gt = load_ground_truth(dataset_dir)
    n_streams = obs.shape[0]

    calib = CalibrationResult.load_json(calib_json)
    m0 = calib.m0

    detected_records = []

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
        r_hat_mean = float(np.mean(fit.r))

        detected_records.append({
            "stream_id": i,
            "m_hat": m_hat,
            "nu_hat": nu_hat,
            "b": b,
            "N": alarm.N,
            "abs_error": abs(nu_hat - b),
            "r_hat_mean": r_hat_mean,
        })

    return detected_records


def calibrate_single_a(records: list[dict], target_coverage: float = 0.95, m0: int = 1000) -> tuple[float, float]:
    """Find smallest 'a' that achieves target empirical coverage."""
    if not records:
        return 0.5, 0.0

    a_grid = np.linspace(0.0, 10.0, 10001)  # step 0.001
    best_a = float(a_grid[-1])
    achieved_cov = 1.0

    for a_val in a_grid:
        covered = 0
        for rec in records:
            hw = math.ceil(a_val * rec["m_hat"])
            ci_lo = max(m0, rec["nu_hat"] - hw)
            ci_hi = min(rec["N"], rec["nu_hat"] + hw)
            if ci_lo <= rec["b"] <= ci_hi:
                covered += 1
        cov = covered / len(records)
        if cov >= target_coverage:
            best_a = float(a_val)
            achieved_cov = float(cov)
            break

    return best_a, achieved_cov


def main() -> None:
    print("--- Task 5: Calibrate CI scalar 'a' per phi and fit a(r_hat) ---")
    cicalib_paths = ensure_cicalib_datasets()

    tags = ["00", "03", "06", "08"]
    phis = [0.0, 0.3, 0.6, 0.8]

    phi_calib_records = []
    all_calib_r_hats = []
    all_calib_a_targets = []

    for tag, phi_val in zip(tags, phis):
        calib_file = PROJECT_ROOT / "results" / "calib" / f"proposed_phi{tag}.json"
        ds_dir = cicalib_paths[tag]

        records = collect_ci_detection_data(ds_dir, calib_file)
        best_a, ach_cov = calibrate_single_a(records, target_coverage=0.95)
        mean_r = float(np.mean([r["r_hat_mean"] for r in records]))
        median_err = float(np.median([r["abs_error"] for r in records]))

        phi_calib_records.append({
            "phi": phi_val,
            "tag": tag,
            "dataset": ds_dir.name,
            "n_detected": len(records),
            "calibrated_a": best_a,
            "achieved_coverage": ach_cov,
            "mean_r_hat": mean_r,
            "median_abs_error": median_err,
        })
        all_calib_r_hats.append(mean_r)
        all_calib_a_targets.append(best_a)

        print(f"phi={phi_val:.1f} ({ds_dir.name}): Calibrated a = {best_a:.4f} (coverage = {ach_cov*100:.1f}%, mean r_hat = {mean_r:.4f}, median |nu_hat - b| = {median_err:.1f})")

    df_phi_calib = pd.DataFrame(phi_calib_records)
    out_phi_csv = PROJECT_ROOT / "results" / "tables" / "ci_calibration_per_phi.csv"
    out_phi_csv.parent.mkdir(parents=True, exist_ok=True)
    df_phi_calib.to_csv(out_phi_csv, index=False, float_format="%.6f", lineterminator="\n")

    # Fit a(r_hat) model: linear regression a = c0 + c1 * r_hat
    poly = np.polyfit(all_calib_r_hats, all_calib_a_targets, deg=1)
    c1, c0 = float(poly[0]), float(poly[1])
    print(f"\nFitted Linear Model: a(r_hat) = {c0:.4f} + {c1:.4f} * r_hat")

    # Evaluate on all 8 test sets
    print("\n--- Evaluating CI Coverage with Dynamic a(r_hat) on Evaluation Sets ---")
    eval_results = []
    effects = [("e10", 1.0), ("e15", 1.5)]

    # Constant baseline from cicalib (phi=0.6)
    with open(PROJECT_ROOT / "results" / "calib" / "ci_a.json", "r", encoding="utf-8") as f:
        cicalib_fixed_a = float(json.load(f)["ci_a"])

    for tag, phi_val in zip(tags, phis):
        calib_file = PROJECT_ROOT / "results" / "calib" / f"proposed_phi{tag}.json"
        calibrated_phi_a = df_phi_calib.loc[df_phi_calib["tag"] == tag, "calibrated_a"].values[0]

        for eff_str, eff_val in effects:
            test_dir = PROJECT_ROOT / "data" / f"change_phi{tag}_{eff_str}"
            recs = collect_ci_detection_data(test_dir, calib_file)
            n_det = len(recs)
            if n_det == 0:
                continue

            # 1. Coverage with dynamic a(r_hat)
            cov_dyn_list = []
            width_dyn_list = []
            for r in recs:
                a_dyn = max(0.0, c0 + c1 * r["r_hat_mean"])
                hw = math.ceil(a_dyn * r["m_hat"])
                ci_lo = max(1000, r["nu_hat"] - hw)
                ci_hi = min(r["N"], r["nu_hat"] + hw)
                cov_dyn_list.append(ci_lo <= r["b"] <= ci_hi)
                width_dyn_list.append(ci_hi - ci_lo)

            # 2. Coverage with constant baseline a = 0.5010
            cov_const_list = []
            width_const_list = []
            for r in recs:
                hw = math.ceil(cicalib_fixed_a * r["m_hat"])
                ci_lo = max(1000, r["nu_hat"] - hw)
                ci_hi = min(r["N"], r["nu_hat"] + hw)
                cov_const_list.append(ci_lo <= r["b"] <= ci_hi)
                width_const_list.append(ci_hi - ci_lo)

            # 3. Coverage with oracle calibrated a(phi)
            cov_oracle_list = []
            width_oracle_list = []
            for r in recs:
                hw = math.ceil(calibrated_phi_a * r["m_hat"])
                ci_lo = max(1000, r["nu_hat"] - hw)
                ci_hi = min(r["N"], r["nu_hat"] + hw)
                cov_oracle_list.append(ci_lo <= r["b"] <= ci_hi)
                width_oracle_list.append(ci_hi - ci_lo)

            abs_errs = [r["abs_error"] for r in recs]
            median_abs_err = float(np.median(abs_errs))
            mean_abs_err = float(np.mean(abs_errs))

            eval_results.append({
                "dataset": f"change_phi{tag}_{eff_str}",
                "phi": phi_val,
                "effect": eff_val,
                "n_detected": n_det,
                "coverage_a_rhat": float(np.mean(cov_dyn_list)),
                "mean_ci_width_a_rhat": float(np.mean(width_dyn_list)),
                "coverage_fixed_cicalib": float(np.mean(cov_const_list)),
                "mean_ci_width_fixed": float(np.mean(width_const_list)),
                "coverage_oracle_phi": float(np.mean(cov_oracle_list)),
                "mean_ci_width_oracle": float(np.mean(width_oracle_list)),
                "median_abs_error": median_abs_err,
                "mean_abs_error": mean_abs_err,
            })

            print(
                f"  {f'change_phi{tag}_{eff_str}':20s}: "
                f"Cov(a(r))={np.mean(cov_dyn_list)*100:5.1f}% (width={np.mean(width_dyn_list):5.1f}) | "
                f"Cov(fixed)={np.mean(cov_const_list)*100:5.1f}% (width={np.mean(width_const_list):5.1f}) | "
                f"Median |nu_hat - b| = {median_abs_err:4.1f}"
            )

    df_eval = pd.DataFrame(eval_results)
    out_eval_csv = PROJECT_ROOT / "results" / "tables" / "ci_coverage_eval.csv"
    df_eval.to_csv(out_eval_csv, index=False, float_format="%.6f", lineterminator="\n")
    print(f"\nSaved evaluation comparison to {out_eval_csv}")

    # Save model parameters
    with open(PROJECT_ROOT / "results" / "calib" / "ci_a_model.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": "linear",
            "c0": c0,
            "c1": c1,
            "per_phi_calibration": phi_calib_records,
        }, f, indent=2)


if __name__ == "__main__":
    main()
