#!/usr/bin/env python3
"""F7: Comprehensive evaluation and comparison of post-alarm inference.

Evaluates:
(a) Old heuristic interval (Eq. 15, a(r_hat) * m*)
(b) F5 with zeta (long-run variance standardised, Eq. 16-17)
(c) F5 with prewhitened residuals (Eq. 16-17 variant)

Validation:
Tested across all phi in {0.0, 0.3, 0.6, 0.8} AND unseen validation points
phi = 0.45 and phi = 0.70 (not used in calibrating c).

Metrics reported:
- CI coverage (target: 0.95, acceptance range: 0.93 - 0.97)
- Mean interval length
- Mean delay
- Interval length / delay ratio
- Support precision, recall, and Jaccard
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from generate_synthetic_streams_fast import (
    _stationary_covariance,
    generate_one_stream,
    make_innovation_covariance,
    make_var_matrix,
)
from ocdmix.calibrate import CalibrationResult, calibrate_thresholds
from ocdmix.fit import fit_training
from ocdmix.io import load_ground_truth, load_observations
from ocdmix.localize import localize_at_alarm, localize_chen
from ocdmix.scan import compute_base_A, compute_statistics_vectorised, detect_alarm

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_or_create_thresholds(phi: float, n_null: int = 200, p: int = 50, T: int = 5000, m0: int = 1000) -> CalibrationResult:
    """Get existing calibration or calibrate on freshly generated null streams."""
    tag = f"phi{int(round(phi*100)):02d}"
    json_path = PROJECT_ROOT / "results" / "calib" / f"proposed_{tag}.json"
    if json_path.exists():
        return CalibrationResult.load_json(json_path)

    print(f"Generating {n_null} null streams to calibrate thresholds for phi={phi}...")
    rng_model = np.random.default_rng(int(round(phi * 10000)))
    A = make_var_matrix(p, phi, "isotropic", rng_model)
    sigma_pre = make_innovation_covariance(p, 0.25)
    gamma_pre = _stationary_covariance(A, sigma_pre, phi)

    obs = np.empty((n_null, T, p), dtype=np.float32)
    for i in range(n_null):
        rng = np.random.default_rng(np.random.SeedSequence([int(round(phi * 10000)), i, 42]))
        X, _ = generate_one_stream(
            T=T, p=p, A=A, phi=phi, sigma_pre=sigma_pre, gamma_pre=gamma_pre,
            rng=rng, change_type="none", cp_fraction=0.5, cp_index=None,
            sparsity=0.1, effect=0.5, covariance_multiplier=1.5,
        )
        obs[i] = X

    res = calibrate_thresholds(
        obs, m0=m0, alpha0=0.05,
        window_grid=(8, 16, 32, 64, 128, 256, 512),
        method="proposed", lrv_method="prewhiten",
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    res.save_json(json_path)
    print(f"Calibrated and saved thresholds to {json_path}: c_A={res.c_A:.4f}, c_Q={res.c_Q:.4f}")
    return res


def get_ci_a(phi: float) -> float:
    """Get adaptive ci_a multiplier from calibrated a(r_hat) model."""
    model_json = PROJECT_ROOT / "results" / "calib" / "ci_a_model.json"
    if model_json.exists():
        with open(model_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        slope = data.get("slope", 1.1604)
        intercept = data.get("intercept", 0.1222)
        return float(intercept + slope * phi)
    return 0.1222 + 1.1604 * phi


def run_evaluation_on_streams(
    obs: np.ndarray,
    gt_records: list[dict],
    calib: CalibrationResult,
    phi_val: float,
    c_val: float = 0.40,
    extra_obs: int = 20,
) -> pd.DataFrame:
    """Evaluate streams and record comparison metrics."""
    n_streams, T, p = obs.shape
    m0 = calib.m0
    ci_a = get_ci_a(phi_val)

    rows = []
    for i in range(n_streams):
        gt = gt_records[i]
        b_true = gt.get("changepoint_index_zero_based")
        s_true = set(gt.get("affected_coordinates_zero_based") or [])
        if b_true is None:
            continue

        X = np.asarray(obs[i], dtype=np.float64)
        fit = fit_training(
            X[:m0], m0=m0, window_grid=tuple(calib.window_grid),
            kmax=calib.kmax, method=calib.method, lrv_method=calib.lrv_method,
        )

        A, Q, _ = compute_statistics_vectorised(X, fit)
        alarm = detect_alarm(A, Q, c_A=calib.c_A, c_Q=calib.c_Q, fit=fit)
        if alarm is None:
            continue

        N = alarm.N
        if N < b_true:
            continue  # False alarm before change point

        delay = N - b_true

        # (a) Old heuristic interval
        nu_hat, ci_old, s_old = localize_at_alarm(
            N, alarm.t_N, X, fit,
            tau_A=calib.tau_A, tau_Q=calib.tau_Q,
            c_A=calib.c_A, c_Q=calib.c_Q,
            ci_a=ci_a,
        )
        m_star = N - nu_hat + 1

        s_old_set = set(s_old)
        tp_old = len(s_old_set & s_true)
        prec_old = tp_old / len(s_old_set) if len(s_old_set) > 0 else 0.0
        rec_old = tp_old / len(s_true) if len(s_true) > 0 else 0.0
        jacc_old = tp_old / len(s_old_set | s_true) if len(s_old_set | s_true) > 0 else 0.0
        cov_old = int(ci_old[0] <= b_true <= ci_old[1])
        width_old = ci_old[1] - ci_old[0]

        # (b) F5 with zeta
        res_zeta = localize_chen(
            N=N, m_star=m_star, X=X, fit=fit,
            extra_obs=extra_obs, alpha=0.05, c_support=c_val, b_min=0.1,
            use_residuals=False,
        )
        s_z_set = set(res_zeta["support"])
        tp_z = len(s_z_set & s_true)
        prec_z = tp_z / len(s_z_set) if len(s_z_set) > 0 else 0.0
        rec_z = tp_z / len(s_true) if len(s_true) > 0 else 0.0
        jacc_z = tp_z / len(s_z_set | s_true) if len(s_z_set | s_true) > 0 else 0.0
        ci_z = res_zeta["CI"]
        cov_z = int(ci_z[0] <= b_true <= ci_z[1])
        width_z = ci_z[1] - ci_z[0]

        # (c) F5 with prewhitened residuals
        res_resid = localize_chen(
            N=N, m_star=m_star, X=X, fit=fit,
            extra_obs=extra_obs, alpha=0.05, c_support=c_val, b_min=0.1,
            use_residuals=True,
        )
        s_r_set = set(res_resid["support"])
        tp_r = len(s_r_set & s_true)
        prec_r = tp_r / len(s_r_set) if len(s_r_set) > 0 else 0.0
        rec_r = tp_r / len(s_true) if len(s_true) > 0 else 0.0
        jacc_r = tp_r / len(s_r_set | s_true) if len(s_r_set | s_true) > 0 else 0.0
        ci_r = res_resid["CI"]
        cov_r = int(ci_r[0] <= b_true <= ci_r[1])
        width_r = ci_r[1] - ci_r[0]

        rows.append({
            "stream_idx": i,
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

    return pd.DataFrame(rows)


def generate_change_dataset(phi: float, effect: float = 1.5, n_streams: int = 200, p: int = 50, T: int = 5000) -> tuple[np.ndarray, list[dict]]:
    """Generate synthetic change streams for validation."""
    rng_model = np.random.default_rng(int(round(phi * 20000)))
    A = make_var_matrix(p, phi, "isotropic", rng_model)
    sigma_pre = make_innovation_covariance(p, 0.25)
    gamma_pre = _stationary_covariance(A, sigma_pre, phi)

    obs = np.empty((n_streams, T, p), dtype=np.float32)
    gt_records = []
    for i in range(n_streams):
        rng = np.random.default_rng(np.random.SeedSequence([int(round(phi * 20000)), i, 777]))
        X, gt = generate_one_stream(
            T=T, p=p, A=A, phi=phi, sigma_pre=sigma_pre, gamma_pre=gamma_pre,
            rng=rng, change_type="mean", cp_fraction=0.5, cp_index=2500,
            sparsity=0.1, effect=effect, covariance_multiplier=1.5,
        )
        obs[i] = X
        gt_records.append(gt)
    return obs, gt_records


def main():
    print("=" * 80)
    print("F7: Running evaluation of (a) Old CI, (b) F5 Zeta, (c) F5 Residuals")
    print("=" * 80)

    # Scenarios: calibrated phis (0.0, 0.3, 0.6, 0.8) and validation phis (0.45, 0.70)
    # Using calibrated c = 0.35 and default extra_obs ell = 20
    c_calib = 0.35
    extra_obs = 20

    scenarios = [
        # Existing evaluation sets (effect = 1.5)
        {"phi": 0.0, "is_validation": False, "dir": PROJECT_ROOT / "data" / "change_phi00_e15", "effect": 1.5},
        {"phi": 0.3, "is_validation": False, "dir": PROJECT_ROOT / "data" / "change_phi03_e15", "effect": 1.5},
        {"phi": 0.6, "is_validation": False, "dir": PROJECT_ROOT / "data" / "change_phi06_e15", "effect": 1.5},
        {"phi": 0.8, "is_validation": False, "dir": PROJECT_ROOT / "data" / "change_phi08_e15", "effect": 1.5},
        # Dedicated validation sets at phi = 0.45 and phi = 0.70
        {"phi": 0.45, "is_validation": True, "dir": None, "effect": 1.5},
        {"phi": 0.70, "is_validation": True, "dir": None, "effect": 1.5},
    ]

    all_summaries = []

    for scen in scenarios:
        phi_val = scen["phi"]
        is_val = scen["is_validation"]
        print(f"\nProcessing scenario: phi={phi_val} ({'VALIDATION' if is_val else 'CALIBRATED PHI'})...")

        calib = get_or_create_thresholds(phi_val)

        if scen["dir"] and scen["dir"].exists():
            obs = load_observations(scen["dir"], mmap=True)
            gt_records = load_ground_truth(scen["dir"])
        else:
            obs, gt_records = generate_change_dataset(phi_val, effect=scen["effect"], n_streams=200)

        df = run_evaluation_on_streams(obs, gt_records, calib, phi_val=phi_val, c_val=c_calib, extra_obs=extra_obs)

        mean_delay = float(df["delay"].mean())
        w_old = float(df["width_old"].mean())
        w_z = float(df["width_zeta"].mean())
        w_r = float(df["width_resid"].mean())

        row = {
            "phi": phi_val,
            "type": "Validation" if is_val else "Calibrated",
            "mean_delay": round(mean_delay, 1),
            # Old Heuristic
            "cov_old": round(float(df["cov_old"].mean()), 3),
            "width_old": round(w_old, 1),
            "ratio_old": round(w_old / max(mean_delay, 1e-4), 2),
            "prec_old": round(float(df["prec_old"].mean()), 3),
            "rec_old": round(float(df["rec_old"].mean()), 3),
            "jacc_old": round(float(df["jacc_old"].mean()), 3),
            # F5 Zeta
            "cov_zeta": round(float(df["cov_zeta"].mean()), 3),
            "width_zeta": round(w_z, 1),
            "ratio_zeta": round(w_z / max(mean_delay, 1e-4), 2),
            "prec_zeta": round(float(df["prec_zeta"].mean()), 3),
            "rec_zeta": round(float(df["rec_zeta"].mean()), 3),
            "jacc_zeta": round(float(df["jacc_zeta"].mean()), 3),
            # F5 Residuals
            "cov_resid": round(float(df["cov_resid"].mean()), 3),
            "width_resid": round(w_r, 1),
            "ratio_resid": round(w_r / max(mean_delay, 1e-4), 2),
            "prec_resid": round(float(df["prec_resid"].mean()), 3),
            "rec_resid": round(float(df["rec_resid"].mean()), 3),
            "jacc_resid": round(float(df["jacc_resid"].mean()), 3),
        }
        all_summaries.append(row)
        print(f"Results for phi={phi_val}:")
        print(f"  Delay: {mean_delay:.1f}")
        print(f"  Old Heuristic: Cov={row['cov_old']*100:.1f}%, Width={row['width_old']:.1f}, Ratio={row['ratio_old']:.2f}, Jaccard={row['jacc_old']:.3f}")
        print(f"  F5 (Zeta)    : Cov={row['cov_zeta']*100:.1f}%, Width={row['width_zeta']:.1f}, Ratio={row['ratio_zeta']:.2f}, Jaccard={row['jacc_zeta']:.3f}")
        print(f"  F5 (Resid)   : Cov={row['cov_resid']*100:.1f}%, Width={row['width_resid']:.1f}, Ratio={row['ratio_resid']:.2f}, Jaccard={row['jacc_resid']:.3f}")

    sum_df = pd.DataFrame(all_summaries)
    out_csv = PROJECT_ROOT / "results" / "tables" / "f7_comparison_table.csv"
    sum_df.to_csv(out_csv, index=False)
    print(f"\nSaved F7 comparison table to {out_csv}")


if __name__ == "__main__":
    main()
