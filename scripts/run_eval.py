#!/usr/bin/env python3
"""Run evaluation of change-point detector on a dataset against ground truth (Task 6)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from ocdmix.calibrate import CalibrationResult
from ocdmix.fit import fit_training
from ocdmix.io import load_ground_truth, load_observations
from ocdmix.localize import localize_at_alarm
from ocdmix.metrics import evaluate_single_stream, summarize_evaluations
from ocdmix.scan import compute_statistics_vectorised, detect_alarm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate OCD on a dataset against ground truth.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Directory containing observations.npy and ground_truth.jsonl.")
    parser.add_argument("--calib-json", type=str, required=True, help="Path to calibrated thresholds JSON.")
    parser.add_argument("--ci-json", type=str, default=None, help="Path to ci_a JSON (optional).")
    parser.add_argument("--method", choices=["proposed", "vanilla"], default=None, help="Override method from calibration JSON.")
    parser.add_argument("--statistic", choices=["both", "linf_only", "l_infinity", "q_only"], default=None, help="Override detection statistic ablation.")
    parser.add_argument("--k-min", type=int, default=None, help="Override minimum window length k_min (e.g. 4).")
    parser.add_argument("--support-z", type=float, default=None, help="Support threshold z parameter (default: norm.ppf).")
    parser.add_argument("--out-csv", type=str, required=True, help="Path to save per-stream CSV.")
    parser.add_argument("--out-json", type=str, required=True, help="Path to save summary JSON.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    obs = load_observations(dataset_dir, mmap=True)
    gt_records = load_ground_truth(dataset_dir)
    n_streams, T, p = obs.shape

    calib = CalibrationResult.load_json(args.calib_json)
    method = args.method or calib.method
    lrv_method = calib.lrv_method
    m0 = calib.m0
    statistic = args.statistic or getattr(calib, "statistic", "both")

    if args.k_min is not None:
        k_min = args.k_min
        window_grid = tuple(sorted(set([k_min] + list(calib.window_grid))))
        base_A = np.sqrt(2.0 * np.log(2.0 * p * len(window_grid)))
    else:
        window_grid = tuple(calib.window_grid)
        k_min = min(window_grid)
        base_A = calib.base_A

    kmax = calib.kmax
    c_A = calib.c_A
    c_Q = calib.c_Q
    tau_A = c_A * base_A if (c_A is not None and not np.isinf(c_A)) else np.inf
    tau_Q = c_Q if (c_Q is not None and not np.isinf(c_Q)) else np.inf

    if statistic in ("linf_only", "l_infinity"):
        c_Q = float("inf")
        tau_Q = float("inf")
    elif statistic == "q_only":
        c_A = float("inf")
        tau_A = float("inf")

    ci_a = 0.0
    if args.ci_json is not None and Path(args.ci_json).exists():
        with open(args.ci_json, "r", encoding="utf-8") as f:
            ci_data = json.load(f)
        ci_a = float(ci_data.get("ci_a", 0.0))

    # Determine if dataset is null
    first_b = gt_records[0].get("changepoint_index_zero_based")
    is_null = (first_b is None)

    eval_records = []

    for i in range(n_streams):
        X = np.asarray(obs[i], dtype=np.float64)
        gt = gt_records[i]

        fit = fit_training(
            X[:m0],
            m0=m0,
            window_grid=window_grid,
            kmax=kmax,
            method=method,
            lrv_method=lrv_method,
        )

        A, Q, _ = compute_statistics_vectorised(X, fit)
        alarm = detect_alarm(A, Q, c_A, c_Q, fit, statistic=statistic)

        if alarm is not None:
            alarm_N = alarm.N
            nu_hat, CI, support = localize_at_alarm(
                N=alarm.N,
                t_N=alarm.t_N,
                X=X,
                fit=fit,
                tau_A=tau_A,
                tau_Q=tau_Q,
                c_A=c_A,
                c_Q=c_Q,
                ci_a=ci_a,
                statistic=statistic,
                m_min=k_min,
                z=args.support_z,
            )
        else:
            alarm_N = None
            nu_hat = None
            CI = None
            support = None

        rec = evaluate_single_stream(
            stream_id=i,
            ground_truth=gt,
            alarm_N=alarm_N,
            nu_hat=nu_hat,
            CI=CI,
            support=support,
        )
        eval_records.append(rec)

    # Convert to DataFrame
    df = pd.DataFrame([r.to_dict() for r in eval_records])

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, float_format="%.6f", lineterminator="\n")

    summary = summarize_evaluations(eval_records, is_null=is_null, m0=m0, T=T)
    summary["dataset"] = dataset_dir.name
    summary["method"] = method
    summary["lrv_method"] = lrv_method
    summary["calib_json"] = str(args.calib_json)
    summary["ci_a"] = ci_a
    summary["statistic"] = statistic
    if args.k_min is not None:
        summary["k_min"] = args.k_min
    if args.support_z is not None:
        summary["support_z"] = args.support_z

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Evaluated {n_streams} streams on {dataset_dir.name} ({method}):")
    if is_null:
        far = summary.get("false_alarm_probability", 0.0)
        arl = summary.get("censored_arl", 0.0)
        print(f"  False-alarm rate: {far * 100:.2f}%, Censored ARL: {arl:.1f}")
    else:
        det = summary.get("detection_rate", 0.0)
        far = summary.get("false_alarm_rate", 0.0)
        del_m = summary.get("mean_delay")
        del_str = f"{del_m:.1f}" if del_m is not None else "N/A"
        cov = summary.get("coverage")
        cov_str = f"{cov * 100:.1f}%" if cov is not None else "N/A"
        jacc = summary.get("mean_support_jaccard")
        jacc_str = f"{jacc:.4f}" if jacc is not None else "N/A"
        print(f"  Det rate: {det * 100:.1f}%, FA rate: {far * 100:.2f}%, Mean delay: {del_str}, Coverage: {cov_str}, Jaccard: {jacc_str}")
    print(f"  Saved CSV : {out_csv}")
    print(f"  Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
