#!/usr/bin/env python3
"""Tune support threshold parameter 'z' on cicalib and evaluate on test sets (Task 4).

Optimization criterion:
Choose z on the cicalib set only to maximize Jaccard subject to precision >= 0.90.
Then report support recovery (precision, recall, Jaccard) on all evaluation sets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from ocdmix.calibrate import CalibrationResult
from ocdmix.fit import fit_training
from ocdmix.io import load_ground_truth, load_observations
from ocdmix.localize import localize_at_alarm
from ocdmix.scan import compute_statistics_vectorised, detect_alarm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def extract_detected_W_stars(
    dataset_dir: str | Path,
    calib_json: str | Path,
) -> list[dict[str, Any]]:
    """Run detector on dataset and return W_star and ground truth support for detected streams."""
    dataset_dir = Path(dataset_dir)
    obs = load_observations(dataset_dir, mmap=True)
    gt_records = load_ground_truth(dataset_dir)
    n_streams, T, p = obs.shape

    calib = CalibrationResult.load_json(calib_json)
    m0 = calib.m0

    detected_items = []

    for i in range(n_streams):
        b = gt_records[i]["changepoint_index_zero_based"]
        if b is None:
            continue

        s_true = set(gt_records[i]["affected_coordinates_zero_based"] or [])
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

        # Localize to get m_star
        Xc = X - fit.mu
        m_max = min(2 * alarm.t_N, alarm.N - m0 + 1)
        m_min = min(fit.window_grid)

        if m_max < m_min:
            m_star = alarm.t_N
        else:
            best_R = -np.inf
            m_star = m_min
            for m in range(m_min, m_max + 1):
                window_sum = np.sum(Xc[alarm.N - m + 1: alarm.N + 1], axis=0)
                s2_m = fit.s2_table[m] if m < fit.s2_table.shape[0] else fit.s2_table[-1]
                sqrt_s2_m = np.sqrt(np.maximum(s2_m, 1e-16))
                W = window_sum / sqrt_s2_m
                A_m = np.max(np.abs(W))
                Q_m = (np.sum(W ** 2) - p) / np.sqrt(2.0 * p)
                R_m = max(A_m / calib.tau_A, Q_m / calib.tau_Q)
                if R_m > best_R:
                    best_R = R_m
                    m_star = m

        # Compute W_star
        window_sum = np.sum(Xc[alarm.N - m_star + 1: alarm.N + 1], axis=0)
        s2_m = fit.s2_table[m_star] if m_star < fit.s2_table.shape[0] else fit.s2_table[-1]
        sqrt_s2_m = np.sqrt(np.maximum(s2_m, 1e-16))
        W_star = window_sum / sqrt_s2_m

        detected_items.append({
            "stream_id": i,
            "W_star": W_star,
            "s_true": s_true,
            "p": p,
        })

    return detected_items


def evaluate_z_on_items(items: list[dict], z: float) -> tuple[float, float, float]:
    """Compute mean precision, recall, and Jaccard for a given threshold z."""
    precisions = []
    recalls = []
    jaccards = []

    for item in items:
        W_star = item["W_star"]
        s_true = item["s_true"]
        p = item["p"]

        s_pred = set(j for j in range(p) if abs(W_star[j]) > z)
        tp = len(s_pred & s_true)

        prec = tp / len(s_pred) if len(s_pred) > 0 else 0.0
        rec = tp / len(s_true) if len(s_true) > 0 else 1.0
        union = len(s_pred | s_true)
        jacc = tp / union if union > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        jaccards.append(jacc)

    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(jaccards))


def tune_z_on_cicalib(cicalib_items: list[dict]) -> tuple[float, float, pd.DataFrame]:
    """Find z that maximizes Jaccard subject to precision >= 0.90."""
    p = cicalib_items[0]["p"]
    default_z = float(norm.ppf(1.0 - 0.05 / (2.0 * p)))

    z_grid = np.linspace(1.5, 4.5, 301)  # step 0.01
    records = []

    best_z = default_z
    best_jaccard = -1.0

    for z_val in z_grid:
        prec, rec, jacc = evaluate_z_on_items(cicalib_items, float(z_val))
        records.append({
            "z": float(z_val),
            "precision": prec,
            "recall": rec,
            "jaccard": jacc,
            "feasible": prec >= 0.90,
        })
        if prec >= 0.90 and jacc > best_jaccard:
            best_jaccard = jacc
            best_z = float(z_val)

    df_grid = pd.DataFrame(records)
    return best_z, default_z, df_grid


def main() -> None:
    print("--- Task 4: Tune Support Threshold z on cicalib ---")
    cicalib_dir = PROJECT_ROOT / "data" / "cicalib"
    calib_phi06 = PROJECT_ROOT / "results" / "calib" / "proposed_phi06.json"

    print(f"Loading cicalib from {cicalib_dir}...")
    cicalib_items = extract_detected_W_stars(cicalib_dir, calib_phi06)
    print(f"Detected streams in cicalib: {len(cicalib_items)}")

    best_z, default_z, df_grid = tune_z_on_cicalib(cicalib_items)
    def_prec, def_rec, def_jacc = evaluate_z_on_items(cicalib_items, default_z)
    best_prec, best_rec, best_jacc = evaluate_z_on_items(cicalib_items, best_z)

    print(f"\nDefault z (alpha=0.05/p) : z = {default_z:.4f}")
    print(f"  cicalib Precision      : {def_prec * 100:.2f}%")
    print(f"  cicalib Recall         : {def_rec * 100:.2f}%")
    print(f"  cicalib Jaccard        : {def_jacc:.4f}")

    print(f"\nTuned z* (prec >= 0.90) : z* = {best_z:.4f}")
    print(f"  cicalib Precision      : {best_prec * 100:.2f}%")
    print(f"  cicalib Recall         : {best_rec * 100:.2f}%")
    print(f"  cicalib Jaccard        : {best_jacc:.4f}")

    grid_csv = PROJECT_ROOT / "results" / "tables" / "support_z_grid_cicalib.csv"
    grid_csv.parent.mkdir(parents=True, exist_ok=True)
    df_grid.to_csv(grid_csv, index=False, float_format="%.6f", lineterminator="\n")

    # Evaluate on all 8 evaluation sets
    print("\n--- Evaluating Support Recovery on Test Sets (Proposed Method) ---")
    eval_records = []
    tags = ["00", "03", "06", "08"]
    effects = [("e10", 1.0), ("e15", 1.5)]

    for tag in tags:
        phi_val = float(f"0.{tag[1:]}" if tag != "00" else "0.0")
        calib_file = PROJECT_ROOT / "results" / "calib" / f"proposed_phi{tag}.json"

        for eff_str, eff_val in effects:
            test_dir = PROJECT_ROOT / "data" / f"change_phi{tag}_{eff_str}"
            items = extract_detected_W_stars(test_dir, calib_file)

            d_prec, d_rec, d_jacc = evaluate_z_on_items(items, default_z)
            t_prec, t_rec, t_jacc = evaluate_z_on_items(items, best_z)

            eval_records.append({
                "dataset": f"change_phi{tag}_{eff_str}",
                "phi": phi_val,
                "effect": eff_val,
                "n_detected": len(items),
                "default_z": default_z,
                "default_precision": d_prec,
                "default_recall": d_rec,
                "default_jaccard": d_jacc,
                "tuned_z": best_z,
                "tuned_precision": t_prec,
                "tuned_recall": t_rec,
                "tuned_jaccard": t_jacc,
                "jaccard_gain": t_jacc - d_jacc,
            })
            print(f"  {f'change_phi{tag}_{eff_str}':20s}: Default Jaccard={d_jacc:.4f} (Prec={d_prec*100:.1f}%) -> Tuned Jaccard={t_jacc:.4f} (Prec={t_prec*100:.1f}%)")

    df_eval = pd.DataFrame(eval_records)
    eval_csv = PROJECT_ROOT / "results" / "tables" / "support_recovery_eval.csv"
    df_eval.to_csv(eval_csv, index=False, float_format="%.6f", lineterminator="\n")
    print(f"\nSaved evaluation comparison to {eval_csv}")

    # Also save the chosen z to a JSON
    z_json = PROJECT_ROOT / "results" / "calib" / "support_z.json"
    with open(z_json, "w", encoding="utf-8") as f:
        json.dump({
            "default_z": default_z,
            "tuned_z": best_z,
            "cicalib_default_jaccard": def_jacc,
            "cicalib_tuned_jaccard": best_jacc,
            "cicalib_default_precision": def_prec,
            "cicalib_tuned_precision": best_prec,
        }, f, indent=2)
    print(f"Saved tuned z to {z_json}")


if __name__ == "__main__":
    main()
