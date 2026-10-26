#!/usr/bin/env python3
"""TASK 5: Delay and Proposition 2 - Fit delay scaling law.

Problem: "delay and width scale as (1-phi)^-2" is wrong.
Observed delay ratios: 1, 1.6, 3.6, 8.1 for phi=0, 0.3, 0.6, 0.8.
(1-phi)^-2 predicts 1, 2.0, 6.3, 25. (1+phi)/(1-phi) predicts 1, 1.9, 4, 9.

Do:
a. Fit log(delay) against log(1-phi), log((1+phi)/(1-phi)) and log(1/(1-phi^2))
   using phi in {0, 0.3, 0.45, 0.6, 0.7, 0.8}, effects 1.0 and 1.5.
   Report fitted exponents and R^2.
b. Rewrite Proposition 2 to match best-fitting form, or state empirical approximation.
c. For phi=0.8, effect=1.0 (delay ~125-190), compare against known-phi oracle CUSUM.
   Diagnosis only, no change to the algorithm.

Save outputs to results/fix/task5_*.csv.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import linregress

from ocdmix.calibrate import CalibrationResult
from ocdmix.fit import fit_training
from ocdmix.io import load_ground_truth, load_observations
from ocdmix.localize import localize_at_alarm
from ocdmix.scan import compute_statistics_vectorised, detect_alarm

RESULTS_FIX = PROJECT_ROOT / "results" / "fix"
RESULTS_FIX.mkdir(parents=True, exist_ok=True)

FAST_GEN = PROJECT_ROOT / "generate_synthetic_streams_fast.py"

# Phi values including the new ones from Task 4
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


def ensure_change_dataset(phi_val: float, phi_tag: str, effect: float, n_streams: int) -> Path:
    """Ensure change dataset exists."""
    eff_tag = f"e{int(effect * 10):02d}"
    dataset_name = f"change_phi{phi_tag}_{eff_tag}_n{n_streams}"
    ds_path = PROJECT_ROOT / "data" / dataset_name

    obs_file = ds_path / "observations.npy"
    if obs_file.exists():
        obs = np.load(str(obs_file), mmap_mode="r")
        if obs.shape[0] >= n_streams:
            return ds_path

    print(f"  Generating {dataset_name} (phi={phi_val}, effect={effect}, n={n_streams})...")
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
        "--seed", "20261020",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to generate {ds_path}:\n{proc.stderr}")
    return ds_path


def compute_oracle_cusum_delay(
    X_all: list[np.ndarray],
    phi_val: float,
    effect: float,
    b: int,
    m0: int,
    n_coordinates: int,
    p: int,
) -> list[int]:
    """Oracle CUSUM: known phi, known signal coordinates, known effect magnitude.

    CUSUM statistic: C_t = max(0, C_{t-1} + delta * x_t - delta^2/2)
    where delta = effect / sigma_lrv, sigma_lrv = 1/(1-phi).
    Alarm when C_t > threshold.
    """
    # Oracle knows the signal size and LRV
    sigma_lrv = 1.0 / (1.0 - phi_val) if phi_val < 1.0 else 10.0
    delta = effect / sigma_lrv  # standardised effect

    # Oracle threshold calibrated for FAR = 0.05 over 4000 steps
    # For simplicity use ARL0 ~ 1/0.05 = 20 rule: threshold = -log(0.05) / delta ~ log(20)/delta
    # Classic CUSUM threshold h for 5% FAR over horizon T-m0=4000:
    # Conservative: h = log(1/alpha0) / delta  (approximately)
    h_thresh = -np.log(0.05) / max(delta, 0.01)

    delays = []
    for X in X_all:
        # Standardise: we know mean (0 pre-change), LRV
        Xc = X - np.mean(X[:m0], axis=0)
        # Only look at the first affected coordinate as oracle
        j = 0  # oracle knows coordinate 0 is affected
        C = 0.0
        alarm_t = None
        for t in range(m0, len(X)):
            x_t = Xc[t, j] / sigma_lrv  # standardised
            C = max(0.0, C + delta * x_t - delta**2 / 2.0)
            if C > h_thresh:
                alarm_t = t
                break
        if alarm_t is not None and alarm_t >= b:
            delays.append(alarm_t - b)
        # If alarm before b: false alarm - skip
    return delays


def get_mean_delays(phi_val: float, phi_tag: str) -> dict[float, float]:
    """Get mean delays from Task 4 evaluation if available, else compute fresh."""
    task4_csv = RESULTS_FIX / "task4_f5_evaluation.csv"
    if task4_csv.exists():
        df = pd.read_csv(task4_csv)
        result = {}
        for eff in EFFECTS:
            row = df[(df["phi"] == phi_val) & (df["effect"] == eff)]
            if not row.empty:
                result[eff] = float(row.iloc[0]["mean_delay"])
        if result:
            return result

    # Compute fresh from datasets
    result = {}
    calib_fname = PHI_CALIB_FILES[phi_tag]
    calib_file = PROJECT_ROOT / "results" / "calib" / calib_fname
    if not calib_file.exists():
        return result

    calib = CalibrationResult.load_json(calib_file)

    for effect in EFFECTS:
        eff_tag = f"e{int(effect * 10):02d}"
        dataset_name = f"change_phi{phi_tag}_{eff_tag}_n{N_STREAMS}"
        ds_path = PROJECT_ROOT / "data" / dataset_name

        # Try original 200-stream datasets
        if not ds_path.exists():
            orig_name = f"change_phi{phi_tag}_{eff_tag}"
            orig_path = PROJECT_ROOT / "data" / orig_name
            if orig_path.exists():
                ds_path = orig_path
            else:
                continue

        obs = load_observations(ds_path, mmap=True)
        gt = load_ground_truth(ds_path)
        n = obs.shape[0]

        delays = []
        for i in range(n):
            b = gt[i]["changepoint_index_zero_based"]
            if b is None:
                continue
            X = np.asarray(obs[i], dtype=np.float64)
            fit = fit_training(
                X[:calib.m0],
                m0=calib.m0,
                window_grid=tuple(calib.window_grid),
                kmax=calib.kmax,
                method=calib.method,
                lrv_method=calib.lrv_method,
            )
            A, Q, _ = compute_statistics_vectorised(X, fit)
            alarm = detect_alarm(A, Q, calib.c_A, calib.c_Q, fit)
            if alarm is not None and alarm.N >= b:
                delays.append(alarm.N - b)

        if delays:
            result[effect] = float(np.mean(delays))

    return result


def main() -> None:
    print("=" * 60)
    print("TASK 5: Delay Scaling Law Analysis")
    print("=" * 60)

    # Gather delay data for all phis and effects
    delay_data = []
    oracle_data = []

    for phi_val, phi_tag in zip(PHIS, PHI_TAGS):
        print(f"\nCollecting delays for phi={phi_val:.2f}...")
        calib_fname = PHI_CALIB_FILES[phi_tag]
        calib_file = PROJECT_ROOT / "results" / "calib" / calib_fname
        if not calib_file.exists():
            print(f"  Missing calibration file: {calib_file}")
            continue

        calib = CalibrationResult.load_json(calib_file)

        for effect in EFFECTS:
            eff_tag = f"e{int(effect * 10):02d}"

            # Try Task 4 n=1000 dataset first, then original
            ds_path = PROJECT_ROOT / "data" / f"change_phi{phi_tag}_{eff_tag}_n{N_STREAMS}"
            if not ds_path.exists():
                ds_path = PROJECT_ROOT / "data" / f"change_phi{phi_tag}_{eff_tag}"
            if not ds_path.exists():
                print(f"  Dataset not found for phi={phi_val}, effect={effect}")
                continue

            obs = load_observations(ds_path, mmap=True)
            gt = load_ground_truth(ds_path)
            n = obs.shape[0]
            print(f"  phi={phi_val:.2f}, effect={effect:.1f}: {n} streams from {ds_path.name}")

            delays = []
            X_detected = []
            b_val = CP_INDEX  # we know it's 2500

            for i in range(min(n, N_STREAMS)):
                b = gt[i]["changepoint_index_zero_based"]
                if b is None:
                    continue
                X = np.asarray(obs[i], dtype=np.float64)
                fit = fit_training(
                    X[:calib.m0],
                    m0=calib.m0,
                    window_grid=tuple(calib.window_grid),
                    kmax=calib.kmax,
                    method=calib.method,
                    lrv_method=calib.lrv_method,
                )
                A, Q, _ = compute_statistics_vectorised(X, fit)
                alarm = detect_alarm(A, Q, calib.c_A, calib.c_Q, fit)
                if alarm is not None and alarm.N >= b:
                    delays.append(alarm.N - b)
                    X_detected.append(X)

            if not delays:
                print(f"  No detections for phi={phi_val}, effect={effect}")
                continue

            mean_delay = float(np.mean(delays))
            median_delay = float(np.median(delays))
            p90_delay = float(np.percentile(delays, 90))
            se_delay = float(np.std(delays) / np.sqrt(len(delays)))
            n_det = len(delays)

            print(f"    mean delay = {mean_delay:.1f} (SE={se_delay:.1f}, n={n_det})")
            delay_data.append({
                "phi": phi_val,
                "effect": effect,
                "n_streams": n,
                "n_detected": n_det,
                "mean_delay": mean_delay,
                "median_delay": median_delay,
                "p90_delay": p90_delay,
                "se_delay": se_delay,
            })

            # Oracle CUSUM for phi=0.8, effect=1.0 (diagnosis only)
            if abs(phi_val - 0.8) < 0.01 and abs(effect - 1.0) < 0.01:
                print(f"  Running oracle CUSUM diagnosis for phi=0.8, effect=1.0...")
                oracle_delays = compute_oracle_cusum_delay(
                    X_all=X_detected[:200],
                    phi_val=phi_val,
                    effect=effect,
                    b=b_val,
                    m0=calib.m0,
                    n_coordinates=max(1, int(P * 0.10)),
                    p=P,
                )
                if oracle_delays:
                    oracle_mean = float(np.mean(oracle_delays))
                    oracle_se = float(np.std(oracle_delays) / np.sqrt(len(oracle_delays)))
                    print(f"    Oracle CUSUM mean delay = {oracle_mean:.1f} (SE={oracle_se:.1f}, n={len(oracle_delays)})")
                    oracle_data.append({
                        "phi": phi_val,
                        "effect": effect,
                        "ocdmix_mean_delay": mean_delay,
                        "ocdmix_se_delay": se_delay,
                        "oracle_cusum_mean_delay": oracle_mean,
                        "oracle_cusum_se": oracle_se,
                        "oracle_cusum_n": len(oracle_delays),
                        "inherent_fraction": oracle_mean / mean_delay if mean_delay > 0 else float("nan"),
                    })

    df_delays = pd.DataFrame(delay_data)
    df_delays.to_csv(RESULTS_FIX / "task5_delays.csv", index=False, float_format="%.4f", lineterminator="\n")
    print(f"\nSaved delay data to results/fix/task5_delays.csv")

    # Part (a): Fit scaling laws
    print("\n--- Part (a): Fit delay scaling laws ---")
    scaling_results = []

    for effect in EFFECTS:
        subset = df_delays[df_delays["effect"] == effect].copy()
        if len(subset) < 3:
            print(f"  effect={effect}: insufficient data for fit")
            continue

        phis_arr = subset["phi"].values
        delays_arr = subset["mean_delay"].values

        # Normalize: delay at phi=0 as reference
        delay_phi0 = subset[subset["phi"] == 0.0]["mean_delay"].values
        if len(delay_phi0) == 0:
            print(f"  effect={effect}: no phi=0 data, skipping normalization")
            continue
        d0 = delay_phi0[0]
        delay_ratios = delays_arr / d0

        print(f"\n  effect={effect:.1f}: reference delay at phi=0 = {d0:.1f}")
        print(f"  phi: {phis_arr}")
        print(f"  delay ratios: {[f'{r:.2f}' for r in delay_ratios]}")

        # Candidate features (excluding phi=0 to avoid log(0))
        nonzero = phis_arr > 0
        phis_nz = phis_arr[nonzero]
        ratios_nz = delay_ratios[nonzero]
        log_ratios = np.log(ratios_nz)

        features = {
            "log(1/(1-phi))": np.log(1.0 / (1.0 - phis_nz)),
            "log(1/(1-phi)^2)": 2.0 * np.log(1.0 / (1.0 - phis_nz)),
            "log((1+phi)/(1-phi))": np.log((1.0 + phis_nz) / (1.0 - phis_nz)),
            "log(1/(1-phi^2))": np.log(1.0 / (1.0 - phis_nz**2)),
        }

        for feat_name, X_feat in features.items():
            # OLS: log(delay_ratio) = beta * feat (forced through 0 since both are 0 at phi=0)
            if len(X_feat) < 2:
                continue
            # Force through origin: slope = sum(x*y) / sum(x^2)
            beta = float(np.sum(X_feat * log_ratios) / np.sum(X_feat**2))
            y_pred = beta * X_feat
            ss_res = float(np.sum((log_ratios - y_pred)**2))
            ss_tot = float(np.sum((log_ratios - np.mean(log_ratios))**2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

            # Predicted ratios vs actual
            pred_ratios = np.exp(y_pred)
            actual_ratios_str = " | ".join([f"{r:.2f}" for r in ratios_nz])
            pred_ratios_str = " | ".join([f"{r:.2f}" for r in pred_ratios])

            print(f"  [{effect:.1f}] {feat_name}: exponent={beta:.3f}, R²={r2:.4f}")
            print(f"    actual ratios: {actual_ratios_str}")
            print(f"    predicted:     {pred_ratios_str}")

            scaling_results.append({
                "effect": effect,
                "feature": feat_name,
                "exponent": beta,
                "r_squared": r2,
                "n_points": len(phis_nz),
            })

    df_scaling = pd.DataFrame(scaling_results)
    df_scaling.to_csv(RESULTS_FIX / "task5_scaling_fit.csv", index=False, float_format="%.6f", lineterminator="\n")
    print(f"\nSaved scaling fit to results/fix/task5_scaling_fit.csv")

    # Best fit per effect
    print("\n--- Best scaling law per effect ---")
    for effect in EFFECTS:
        sub = df_scaling[df_scaling["effect"] == effect]
        if not sub.empty:
            best = sub.loc[sub["r_squared"].idxmax()]
            print(f"  effect={effect:.1f}: best feature = '{best['feature']}', exponent={best['exponent']:.3f}, R²={best['r_squared']:.4f}")

    # Oracle diagnosis
    if oracle_data:
        df_oracle = pd.DataFrame(oracle_data)
        df_oracle.to_csv(RESULTS_FIX / "task5_oracle_cusum.csv", index=False, float_format="%.4f", lineterminator="\n")
        print(f"\nSaved oracle CUSUM comparison to results/fix/task5_oracle_cusum.csv")
        for row in oracle_data:
            print(f"\nOracle CUSUM diagnosis (phi=0.8, effect=1.0):")
            print(f"  OCD-Mix mean delay: {row['ocdmix_mean_delay']:.1f} (SE={row['ocdmix_se_delay']:.1f})")
            print(f"  Oracle CUSUM mean delay: {row['oracle_cusum_mean_delay']:.1f} (SE={row['oracle_cusum_se']:.1f})")
            print(f"  Inherent delay fraction: {row['inherent_fraction']*100:.0f}%")
    else:
        print("\nNo oracle diagnosis data collected (phi=0.8 dataset not found)")

    print("\nTask 5 complete.")


if __name__ == "__main__":
    main()
