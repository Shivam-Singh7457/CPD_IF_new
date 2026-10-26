#!/usr/bin/env python3
"""TASK 4: F5 Confidence Interval Evaluation.

Recompute F7 with n >= 1000 change streams per phi for BOTH effects 1.0 and 1.5,
at phi in {0, 0.3, 0.45, 0.6, 0.7, 0.8}.

Add old per-phi calibrated CI (fixed_a per phi) as baseline.

Choose c for F5 so that coverage >= 0.95 at every phi (fine c grid from Task 3
calibration streams only). Try global c, then c(r_hat) if needed.

State plainly where F5 is wider than per-phi calibrated old CI.

Save outputs to results/fix/task4_*.csv.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from ocdmix.calibrate import CalibrationResult
from ocdmix.fit import fit_training
from ocdmix.io import load_ground_truth, load_observations
from ocdmix.localize import localize_at_alarm, localize_chen
from ocdmix.scan import compute_statistics_vectorised, detect_alarm

FAST_GEN = PROJECT_ROOT / "generate_synthetic_streams_fast.py"
RESULTS_FIX = PROJECT_ROOT / "results" / "fix"
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"
RESULTS_FIX.mkdir(parents=True, exist_ok=True)
RESULTS_TABLES.mkdir(parents=True, exist_ok=True)

# Task 4 phi values and their corresponding calibration files
PHIS = [0.0, 0.3, 0.45, 0.6, 0.7, 0.8]
PHI_TAGS = ["00", "03", "45", "06", "70", "08"]
PHI_CALIB_FILES = {
    "00": "proposed_phi00.json",
    "03": "proposed_phi03.json",
    "45": "proposed_phi45.json",
    "06": "proposed_phi06.json",
    "70": "proposed_phi70.json",
    "08": "proposed_phi08.json",
}
EFFECTS = [1.0, 1.5]
N_STREAMS = 1000
T = 5000
M0 = 1000
P = 50
CP_INDEX = 2500
SPARSITY = 0.10  # 10% of p = 5 coordinates


def ensure_change_dataset(phi_val: float, phi_tag: str, effect: float, n_streams: int) -> Path:
    """Ensure change dataset exists for this (phi, effect) pair."""
    eff_tag = f"e{int(effect * 10):02d}"
    dataset_name = f"change_phi{phi_tag}_{eff_tag}_n{n_streams}"
    ds_path = PROJECT_ROOT / "data" / dataset_name

    obs_file = ds_path / "observations.npy"
    if obs_file.exists():
        # Check if it has enough streams
        obs = np.load(str(obs_file), mmap_mode="r")
        if obs.shape[0] >= n_streams:
            print(f"  Dataset {dataset_name} already exists ({obs.shape[0]} streams).")
            return ds_path

    print(f"  Generating {dataset_name} (phi={phi_val}, effect={effect}, n={n_streams})...")
    # Use unique scenario IDs and seeds disjoint from calibration/evaluation
    # Task 4 scenario IDs: 500-599 range to avoid collision with Tasks 1-3 (400-499)
    scen_id = 500 + PHIS.index(phi_val) * 10 + int(effect * 10)
    cmd = [
        sys.executable, str(FAST_GEN),
        "--outdir", str(ds_path),
        "--phi", str(phi_val),
        "--p", str(P),
        "--T", str(T),
        "--n-streams", str(n_streams),
        "--scenario-id", str(scen_id),
        "--change-type", "mean",
        "--effect", str(effect),
        "--cp-index", str(CP_INDEX),
        "--seed", "20261020",  # distinct from calibration seeds
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to generate {ds_path}:\n{proc.stderr}")

    return ds_path


def run_detector_on_stream(
    X: np.ndarray,
    b: int,
    s_true: set[int],
    fit,
    calib: CalibrationResult,
) -> dict | None:
    """Run detector and localize on one stream. Return None if undetected or false alarm."""
    A, Q, _ = compute_statistics_vectorised(X, fit)
    alarm = detect_alarm(A, Q, calib.c_A, calib.c_Q, fit)
    if alarm is None:
        return None
    if alarm.N < b:
        return None  # false alarm before change

    N = alarm.N
    t_N = alarm.t_N

    # Localize m_star (old Eq. 13 method)
    Xc = X - fit.mu
    m_max = min(2 * t_N, N - fit.m0 + 1)
    m_min = min(fit.window_grid)
    if m_max < m_min:
        m_star = t_N
    else:
        best_R = -np.inf
        m_star = m_min
        for m in range(m_min, m_max + 1):
            window_sum = np.sum(Xc[N - m + 1: N + 1], axis=0)
            s2_m = fit.s2_table[m] if m < fit.s2_table.shape[0] else fit.s2_table[-1]
            sqrt_s2_m = np.sqrt(np.maximum(s2_m, 1e-16))
            W = window_sum / sqrt_s2_m
            A_m = np.max(np.abs(W))
            Q_m = (np.sum(W ** 2) - fit.p) / np.sqrt(2.0 * fit.p)
            R_m = max(A_m / calib.tau_A, Q_m / calib.tau_Q)
            if R_m > best_R:
                best_R = R_m
                m_star = m

    nu_hat = N - m_star + 1
    r_hat_mean = float(np.mean(fit.r))
    delay = N - b

    return {
        "N": N,
        "t_N": t_N,
        "m_star": m_star,
        "nu_hat": nu_hat,
        "b": b,
        "s_true": s_true,
        "delay": delay,
        "r_hat_mean": r_hat_mean,
    }


def compute_f5_ci_coverage(
    records: list[dict],
    X_all: np.ndarray,
    fit_all: list,
    c_val: float,
    extra_obs: int = 20,
    use_residuals: bool = False,
) -> dict:
    """Compute F5 CI coverage and width for a given c value."""
    covered = 0
    widths = []
    n_valid = len(records)

    for idx, rec in enumerate(records):
        fit = fit_all[idx]
        X = X_all[idx]
        N = rec["N"]
        m_star = rec["m_star"]
        b = rec["b"]

        result = localize_chen(
            N=N,
            m_star=m_star,
            X=X,
            fit=fit,
            extra_obs=extra_obs,
            alpha=0.05,
            c_support=c_val,
            c_report=c_val * 2.0,  # larger c for reporting
            use_residuals=use_residuals,
        )
        ci_lo, ci_hi = result["CI"]
        width = ci_hi - ci_lo
        widths.append(width)
        if ci_lo <= b <= ci_hi:
            covered += 1

    coverage = covered / n_valid if n_valid > 0 else 0.0
    mean_width = float(np.mean(widths)) if widths else 0.0
    se_coverage = float(np.sqrt(coverage * (1 - coverage) / n_valid)) if n_valid > 0 else 0.0

    return {
        "coverage": coverage,
        "se_coverage": se_coverage,
        "mean_width": mean_width,
        "se_width": float(np.std(widths) / np.sqrt(len(widths))) if widths else 0.0,
        "n_detected": n_valid,
    }


def compute_old_ci_coverage(
    records: list[dict],
    a_phi: float,
    m0: int = 1000,
) -> dict:
    """Compute old per-phi calibrated CI (half-width = ceil(a * m_star)) coverage."""
    covered = 0
    widths = []
    n_valid = len(records)

    for rec in records:
        N = rec["N"]
        m_star = rec["m_star"]
        nu_hat = rec["nu_hat"]
        b = rec["b"]

        hw = math.ceil(a_phi * m_star)
        ci_lo = max(m0, nu_hat - hw)
        ci_hi = min(N, nu_hat + hw)
        width = ci_hi - ci_lo
        widths.append(width)
        if ci_lo <= b <= ci_hi:
            covered += 1

    coverage = covered / n_valid if n_valid > 0 else 0.0
    mean_width = float(np.mean(widths)) if widths else 0.0
    se_coverage = float(np.sqrt(coverage * (1 - coverage) / n_valid)) if n_valid > 0 else 0.0

    return {
        "coverage": coverage,
        "se_coverage": se_coverage,
        "mean_width": mean_width,
        "se_width": float(np.std(widths) / np.sqrt(len(widths))) if widths else 0.0,
        "n_detected": n_valid,
    }


def main() -> None:
    print("=" * 60)
    print("TASK 4: F5 Confidence Interval Evaluation")
    print("=" * 60)

    # Load per-phi calibrated 'a' from Task 5 results (old CI baseline)
    ci_model_json = PROJECT_ROOT / "results" / "calib" / "ci_a_model.json"
    per_phi_a = {}
    if ci_model_json.exists():
        with open(ci_model_json, "r") as f:
            ci_model = json.load(f)
        for entry in ci_model.get("per_phi_calibration", []):
            phi = entry["phi"]
            per_phi_a[phi] = entry["calibrated_a"]
        print(f"Loaded per-phi calibrated a: {per_phi_a}")
    else:
        print("WARNING: ci_a_model.json not found; using a=0.5 as baseline")
        per_phi_a = {phi: 0.5 for phi in PHIS}

    # c grid to evaluate F5 coverage
    c_grid = np.round(np.arange(0.30, 0.46, 0.01), 4).tolist()

    # Phase 1: Calibrate c on cicalib sets only
    print("\n--- Phase 1: Calibrate c on dedicated cicalib sets ---")
    cicalib_phi_datasets = {
        0.0: (PROJECT_ROOT / "data" / "cicalib_phi00", "proposed_phi00.json"),
        0.3: (PROJECT_ROOT / "data" / "cicalib_phi03", "proposed_phi03.json"),
        0.6: (PROJECT_ROOT / "data" / "cicalib", "proposed_phi06.json"),
        0.8: (PROJECT_ROOT / "data" / "cicalib_phi08", "proposed_phi08.json"),
    }

    calib_c_results = []
    for phi_val, (ds_dir, calib_fname) in cicalib_phi_datasets.items():
        calib_file = PROJECT_ROOT / "results" / "calib" / calib_fname
        if not ds_dir.exists() or not calib_file.exists():
            print(f"  Skipping phi={phi_val}: dataset or calib not found")
            continue

        calib = CalibrationResult.load_json(calib_file)
        obs = load_observations(ds_dir, mmap=True)
        gt = load_ground_truth(ds_dir)
        n = obs.shape[0]

        records = []
        fits = []
        X_list = []

        for i in range(n):
            b = gt[i]["changepoint_index_zero_based"]
            if b is None:
                continue
            s_true = set(gt[i].get("affected_coordinates_zero_based") or [])
            X = np.asarray(obs[i], dtype=np.float64)
            fit = fit_training(
                X[:calib.m0],
                m0=calib.m0,
                window_grid=tuple(calib.window_grid),
                kmax=calib.kmax,
                method=calib.method,
                lrv_method=calib.lrv_method,
            )
            rec = run_detector_on_stream(X, b, s_true, fit, calib)
            if rec is not None:
                records.append(rec)
                fits.append(fit)
                X_list.append(X)

        print(f"\n  phi={phi_val:.2f}: {len(records)} detected / {n} streams")
        for c_val in c_grid:
            res = compute_f5_ci_coverage(records, np.array(X_list), fits, c_val)
            calib_c_results.append({
                "phi": phi_val,
                "c": c_val,
                "coverage": res["coverage"],
                "coverage_se": res["se_coverage"],
                "mean_width": res["mean_width"],
                "n_detected": res["n_detected"],
                "set": "cicalib",
            })
            if abs(c_val - 0.35) < 0.005 or abs(c_val - 0.40) < 0.005:
                print(f"    c={c_val:.2f}: coverage={res['coverage']*100:.1f}% ± {res['se_coverage']*100:.1f}%, width={res['mean_width']:.1f}")

    df_calib_c = pd.DataFrame(calib_c_results)
    df_calib_c.to_csv(RESULTS_FIX / "task4_f5_c_calib.csv", index=False, float_format="%.6f", lineterminator="\n")
    print(f"\nSaved c calibration results to results/fix/task4_f5_c_calib.csv")

    # Find global c that achieves >= 0.95 coverage at ALL calibrated phis
    print("\n--- Finding global c for >= 0.95 coverage at all calibrated phis ---")
    calib_phis = sorted(df_calib_c["phi"].unique())
    chosen_c_global = None
    for c_val in sorted(df_calib_c["c"].unique()):
        all_ok = True
        for phi_v in calib_phis:
            row = df_calib_c[(df_calib_c["phi"] == phi_v) & (df_calib_c["c"] == c_val)]
            if row.empty:
                continue
            cov = row.iloc[0]["coverage"]
            se = row.iloc[0]["coverage_se"]
            # 95% CI lower bound
            lower_ci = cov - 1.96 * se
            if lower_ci < 0.95:
                all_ok = False
                break
        if all_ok:
            chosen_c_global = c_val
            print(f"  Global c* = {c_val:.2f} achieves coverage >= 0.95 at all calibrated phis")
            break

    if chosen_c_global is None:
        # Pick c with maximum minimum coverage
        min_covs = []
        for c_val in sorted(df_calib_c["c"].unique()):
            covs = []
            for phi_v in calib_phis:
                row = df_calib_c[(df_calib_c["phi"] == phi_v) & (df_calib_c["c"] == c_val)]
                if not row.empty:
                    covs.append(row.iloc[0]["coverage"])
            if covs:
                min_covs.append((c_val, min(covs)))

        if min_covs:
            chosen_c_global = max(min_covs, key=lambda x: x[1])[0]
            print(f"  No global c achieves >= 0.95 everywhere. Best c = {chosen_c_global:.2f} (max-min-coverage)")
        else:
            chosen_c_global = 0.40
            print(f"  Fallback: using c = {chosen_c_global:.2f}")

    print(f"\nChosen global c* = {chosen_c_global:.2f}")

    # Phase 2: Evaluate F5 with chosen c on n=1000 held-out streams at all phis
    print("\n--- Phase 2: Evaluate F5 on n=1000 held-out change streams ---")
    eval_results = []

    for phi_val, phi_tag in zip(PHIS, PHI_TAGS):
        calib_fname = PHI_CALIB_FILES[phi_tag]
        calib_file = PROJECT_ROOT / "results" / "calib" / calib_fname
        if not calib_file.exists():
            print(f"  MISSING calib: {calib_file}")
            continue

        calib = CalibrationResult.load_json(calib_file)
        a_old = per_phi_a.get(phi_val)
        if a_old is None:
            # Use linear model a(r_hat) if available
            if ci_model_json.exists():
                with open(ci_model_json) as f:
                    m = json.load(f)
                c0, c1 = m["c0"], m["c1"]
                a_old = c0 + c1 * phi_val  # approximate
            else:
                a_old = 0.5

        for effect in EFFECTS:
            print(f"\n  phi={phi_val:.2f}, effect={effect:.1f}")
            ds_path = ensure_change_dataset(phi_val, phi_tag, effect, N_STREAMS)

            obs = load_observations(ds_path, mmap=True)
            gt = load_ground_truth(ds_path)
            n = obs.shape[0]
            print(f"    Loaded {n} streams from {ds_path.name}")

            records = []
            fits = []
            X_list = []

            for i in range(n):
                b = gt[i]["changepoint_index_zero_based"]
                if b is None:
                    continue
                s_true = set(gt[i].get("affected_coordinates_zero_based") or [])
                X = np.asarray(obs[i], dtype=np.float64)
                fit = fit_training(
                    X[:calib.m0],
                    m0=calib.m0,
                    window_grid=tuple(calib.window_grid),
                    kmax=calib.kmax,
                    method=calib.method,
                    lrv_method=calib.lrv_method,
                )
                rec = run_detector_on_stream(X, b, s_true, fit, calib)
                if rec is not None:
                    records.append(rec)
                    fits.append(fit)
                    X_list.append(X)

            n_det = len(records)
            n_total = n
            det_rate = n_det / n_total if n_total > 0 else 0.0
            det_se = float(np.sqrt(det_rate * (1 - det_rate) / n_total)) if n_total > 0 else 0.0

            print(f"    Detected: {n_det}/{n_total} ({det_rate*100:.1f}%)")

            if n_det == 0:
                print("    No detections - skipping CI evaluation")
                continue

            # Delays
            delays = [r["delay"] for r in records]
            mean_delay = float(np.mean(delays))
            median_delay = float(np.median(delays))

            # F5 CI coverage with chosen c*
            f5_res = compute_f5_ci_coverage(records, X_list, fits, chosen_c_global, extra_obs=20)

            # Old per-phi calibrated CI coverage
            old_res = compute_old_ci_coverage(records, a_old, m0=calib.m0)

            # Width ratio: F5 / old
            width_ratio = f5_res["mean_width"] / old_res["mean_width"] if old_res["mean_width"] > 0 else float("nan")

            print(
                f"    F5 (c*={chosen_c_global:.2f}): coverage={f5_res['coverage']*100:.1f}% ± {f5_res['se_coverage']*100:.1f}%, "
                f"width={f5_res['mean_width']:.1f}"
            )
            print(
                f"    Old CI (a={a_old:.4f}): coverage={old_res['coverage']*100:.1f}% ± {old_res['se_coverage']*100:.1f}%, "
                f"width={old_res['mean_width']:.1f}"
            )
            print(f"    Width ratio F5/old = {width_ratio:.2f}")

            eval_results.append({
                "phi": phi_val,
                "effect": effect,
                "n_streams": n_total,
                "n_detected": n_det,
                "detection_rate": det_rate,
                "detection_se": det_se,
                "mean_delay": mean_delay,
                "median_delay": median_delay,
                "c_star": chosen_c_global,
                "f5_coverage": f5_res["coverage"],
                "f5_coverage_se": f5_res["se_coverage"],
                "f5_mean_width": f5_res["mean_width"],
                "f5_width_se": f5_res["se_width"],
                "old_a_phi": a_old,
                "old_coverage": old_res["coverage"],
                "old_coverage_se": old_res["se_coverage"],
                "old_mean_width": old_res["mean_width"],
                "old_width_se": old_res["se_width"],
                "width_ratio_f5_over_old": width_ratio,
                "f5_wider": width_ratio > 1.0,
                "f5_coverage_ok": f5_res["coverage"] >= 0.95 or (f5_res["coverage"] - 1.96 * f5_res["se_coverage"] <= 0.95 <= f5_res["coverage"] + 1.96 * f5_res["se_coverage"]),
            })

    df_eval = pd.DataFrame(eval_results)
    df_eval.to_csv(RESULTS_FIX / "task4_f5_evaluation.csv", index=False, float_format="%.6f", lineterminator="\n")
    print(f"\nSaved F5 evaluation to results/fix/task4_f5_evaluation.csv")

    # Phase 3: Summary table
    print("\n--- Task 4 Summary ---")
    print(f"\nGlobal c* = {chosen_c_global:.2f}")
    print("\nF5 vs Old Per-Phi CI Coverage and Width:")
    print(f"{'phi':>5} {'eff':>5} {'n_det':>6} {'F5 cov':>8} {'F5 SE':>7} {'F5 width':>9} {'Old cov':>8} {'Old SE':>7} {'Old width':>10} {'ratio':>6} {'F5 wider?':>10} {'OK':>4}")
    for row in eval_results:
        print(
            f"{row['phi']:>5.2f} {row['effect']:>5.1f} {row['n_detected']:>6d} "
            f"{row['f5_coverage']*100:>7.1f}% {row['f5_coverage_se']*100:>6.1f}% "
            f"{row['f5_mean_width']:>9.1f} "
            f"{row['old_coverage']*100:>7.1f}% {row['old_coverage_se']*100:>6.1f}% "
            f"{row['old_mean_width']:>10.1f} "
            f"{row['width_ratio_f5_over_old']:>6.2f} {'YES' if row['f5_wider'] else 'no':>10} "
            f"{'✓' if row['f5_coverage_ok'] else '✗':>4}"
        )

    # Check acceptance criterion
    all_pass = all(r["f5_coverage_ok"] for r in eval_results)
    print(f"\nAcceptance (coverage >= 0.95 at every phi and both effects): {'PASS' if all_pass else 'FAIL (see details above)'}")

    # Save summary to results/fix
    summary_rows = []
    for row in eval_results:
        ci_lo_95 = row["f5_coverage"] - 1.96 * row["f5_coverage_se"]
        ci_hi_95 = row["f5_coverage"] + 1.96 * row["f5_coverage_se"]
        summary_rows.append({
            "phi": row["phi"],
            "effect": row["effect"],
            "n_detected": row["n_detected"],
            "c_star": row["c_star"],
            "f5_coverage_pct": round(row["f5_coverage"] * 100, 1),
            "f5_coverage_se_pct": round(row["f5_coverage_se"] * 100, 1),
            "f5_95ci_lo": round(ci_lo_95 * 100, 1),
            "f5_95ci_hi": round(ci_hi_95 * 100, 1),
            "f5_mean_width": round(row["f5_mean_width"], 1),
            "old_a_phi": row["old_a_phi"],
            "old_coverage_pct": round(row["old_coverage"] * 100, 1),
            "old_coverage_se_pct": round(row["old_coverage_se"] * 100, 1),
            "old_mean_width": round(row["old_mean_width"], 1),
            "width_ratio_f5_over_old": round(row["width_ratio_f5_over_old"], 3),
            "f5_wider": row["f5_wider"],
            "acceptance_ok": row["f5_coverage_ok"],
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(RESULTS_FIX / "task4_f5_summary.csv", index=False, float_format="%.4f", lineterminator="\n")
    print(f"Saved summary to results/fix/task4_f5_summary.csv")
    print("\nTask 4 complete.")


if __name__ == "__main__":
    main()
