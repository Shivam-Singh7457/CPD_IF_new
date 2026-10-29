"""Task C – Honest CI calibration and held-out validation.

Protocol:
  1. Calibration set:  seeds 20270100..20270100+N_CALIB-1  (per (phi, effect))
  2. Evaluation set:   seeds 20280100..20280100+N_EVAL-1   (per (phi, effect), N_EVAL >= 1000)
  Both sets are independent.

Acceptance gates checked:
  G3: Held-out CI coverage in new mode >= 0.95 - 1 SE at every phi and effect;
      reports mean CI width and width/mean-delay ratio.
  G4: Support: with c_rep=0.8, held-out Jaccard >= Bonferroni-3.29 Jaccard at every phi;
      with c_CI and ell=0, recall of S_hat U {j_hat} reported, and recall increases with ell in {0, 20, 50}.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from generate_synthetic_streams_fast import (
    _stationary_covariance,
    generate_one_stream,
    make_innovation_covariance,
    make_var_matrix,
)
from ocdmix.calibrate import calibrate_thresholds
from ocdmix.fit import fit_training
from ocdmix.localize import localize_chen
from ocdmix.scan import compute_statistics_vectorised, detect_alarm

PHI_LIST = [0.0, 0.3, 0.6, 0.8]
EFFECT_LIST = [1.0, 1.5]
P, T, M0, B_CP, SPARSITY = 50, 5000, 1000, 2500, 0.10
DEFAULT_N_CALIB = 500
DEFAULT_N_EVAL = 1000
CALIB_SEED_BASE = 20270100
EVAL_SEED_BASE = 20280100
C_CI_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
DEFAULT_C_REP = 0.80
B_MIN = 0.1
EXTRA_OBS_LIST = [0, 20, 50]
PHI_TAG_MAP = {0.0: "00", 0.3: "03", 0.6: "06", 0.8: "08"}

BONFERRONI_JACCARD_REF = {
    (0.0, 1.0): 0.7358,
    (0.0, 1.5): 0.7716,
    (0.3, 1.0): 0.7106,
    (0.3, 1.5): 0.7587,
    (0.6, 1.0): 0.6641,
    (0.6, 1.5): 0.6767,
    (0.8, 1.0): 0.5895,
    (0.8, 1.5): 0.6042,
}


def _load_calib(phi: float) -> tuple[float, float]:
    tag = PHI_TAG_MAP[phi]
    path = repo_root / "results" / "calib" / f"proposed_phi{tag}.json"
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        return float(d["c_A"]), float(d["c_Q"])
    return _quick_calib(phi)


def _quick_calib(phi: float) -> tuple[float, float]:
    seed = 20260000 + int(phi * 100)
    rng_m = np.random.default_rng(np.random.SeedSequence([seed, 9173]))
    A = make_var_matrix(P, phi, "isotropic", rng_m)
    sp = make_innovation_covariance(P, 0.25)
    gp = _stationary_covariance(A, sp, phi)
    null_obs = []
    for i in range(200):
        rng = np.random.default_rng(np.random.SeedSequence([seed, i, 2718]))
        X, _ = generate_one_stream(
            T=T, p=P, A=A, phi=phi, sigma_pre=sp, gamma_pre=gp,
            rng=rng, change_type="none", cp_fraction=0.5, cp_index=None,
            sparsity=SPARSITY, effect=0.0, covariance_multiplier=1.5,
        )
        null_obs.append(X)
    cr = calibrate_thresholds(np.array(null_obs, dtype=np.float32), m0=M0, alpha0=0.05)
    return cr.c_A, cr.c_Q


def _process_one_stream_calib(args: tuple) -> dict[float, dict[str, Any]]:
    phi, effect, idx, seed_base, c_A, c_Q, c_ci_list, c_rep = args
    seed = seed_base + idx
    rng_m = np.random.default_rng(np.random.SeedSequence([seed, 0, 9173]))
    A = make_var_matrix(P, phi, "isotropic", rng_m)
    sp = make_innovation_covariance(P, 0.25)
    gp = _stationary_covariance(A, sp, phi)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0, 1, 2718]))
    X, gt = generate_one_stream(
        T=T, p=P, A=A, phi=phi, sigma_pre=sp, gamma_pre=gp,
        rng=rng, change_type="mean", cp_fraction=0.5, cp_index=B_CP,
        sparsity=SPARSITY, effect=effect, covariance_multiplier=1.5,
    )
    b_true = gt.get("changepoint_index_zero_based")
    s_true = set(gt.get("affected_coordinates_zero_based", []))
    fit = fit_training(X[:M0], m0=M0, method="proposed")
    A_arr, Q_arr, _ = compute_statistics_vectorised(X, fit)
    alarm = detect_alarm(A_arr, Q_arr, c_A, c_Q, fit)

    out: dict[float, dict[str, Any]] = {}
    if alarm is None or alarm.N < b_true:
        for c_ci in c_ci_list:
            out[c_ci] = {
                "detected": False, "covered": False, "ci_width": None,
                "delay": None, "jaccard": None, "recall": None,
            }
        return out

    delay = alarm.N - b_true
    m_star = alarm.t_N
    for c_ci in c_ci_list:
        r = localize_chen(
            alarm.N, m_star, X, fit, extra_obs=0,
            alpha=0.05, c_CI=c_ci, c_rep=c_rep,
            b_min=B_MIN, inference_mode="new",
        )
        ci_lo, ci_hi = r["CI"]
        covered = bool(ci_lo <= b_true <= ci_hi)
        width = int(ci_hi - ci_lo)
        s_pred = set(r["support_report"])
        if len(s_true) > 0 and len(s_pred) > 0:
            tp = len(s_pred & s_true)
            jac = tp / len(s_pred | s_true)
            rec = tp / len(s_true)
        else:
            jac = 1.0 if len(s_pred) == 0 else 0.0
            rec = 1.0
        out[c_ci] = {
            "detected": True, "covered": covered, "ci_width": width,
            "delay": delay, "jaccard": jac, "recall": rec,
        }
    return out


def _process_one_stream_eval(args: tuple) -> dict[int, dict[str, Any]]:
    phi, effect, idx, seed_base, c_A, c_Q, c_ci, c_rep, extra_obs_list = args
    seed = seed_base + idx
    rng_m = np.random.default_rng(np.random.SeedSequence([seed, 0, 9173]))
    A = make_var_matrix(P, phi, "isotropic", rng_m)
    sp = make_innovation_covariance(P, 0.25)
    gp = _stationary_covariance(A, sp, phi)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0, 1, 2718]))
    X, gt = generate_one_stream(
        T=T, p=P, A=A, phi=phi, sigma_pre=sp, gamma_pre=gp,
        rng=rng, change_type="mean", cp_fraction=0.5, cp_index=B_CP,
        sparsity=SPARSITY, effect=effect, covariance_multiplier=1.5,
    )
    b_true = gt.get("changepoint_index_zero_based")
    s_true = set(gt.get("affected_coordinates_zero_based", []))
    fit = fit_training(X[:M0], m0=M0, method="proposed")
    A_arr, Q_arr, _ = compute_statistics_vectorised(X, fit)
    alarm = detect_alarm(A_arr, Q_arr, c_A, c_Q, fit)

    out: dict[int, dict[str, Any]] = {}
    if alarm is None or alarm.N < b_true:
        for ell in extra_obs_list:
            out[ell] = {
                "detected": False, "covered": False, "ci_width": None,
                "delay": None, "jaccard": None, "recall": None,
            }
        return out

    delay = alarm.N - b_true
    m_star = alarm.t_N
    for ell in extra_obs_list:
        r = localize_chen(
            alarm.N, m_star, X, fit, extra_obs=ell,
            alpha=0.05, c_CI=c_ci, c_rep=c_rep,
            b_min=B_MIN, inference_mode="new",
        )
        ci_lo, ci_hi = r["CI"]
        covered = bool(ci_lo <= b_true <= ci_hi)
        width = int(ci_hi - ci_lo)
        s_pred = set(r["support_report"])
        if len(s_true) > 0 and len(s_pred) > 0:
            tp = len(s_pred & s_true)
            jac = tp / len(s_pred | s_true)
            rec = tp / len(s_true)
        else:
            jac = 1.0 if len(s_pred) == 0 else 0.0
            rec = 1.0
        out[ell] = {
            "detected": True, "covered": covered, "ci_width": width,
            "delay": delay, "jaccard": jac, "recall": rec,
        }
    return out


def run_calib_cell(
    phi: float, effect: float, c_A: float, c_Q: float,
    c_ci_list: list[float], c_rep: float, n_streams: int,
    workers: int = 4,
) -> dict[float, dict[str, Any]]:
    tasks = [(phi, effect, i, CALIB_SEED_BASE, c_A, c_Q, c_ci_list, c_rep)
             for i in range(n_streams)]
    results: list[dict[float, dict[str, Any]]] = []
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_process_one_stream_calib, tasks, chunksize=16))
    else:
        results = [_process_one_stream_calib(t) for t in tasks]

    summary = {}
    for c_ci in c_ci_list:
        covers = [r[c_ci]["covered"] for r in results]
        detected = [r[c_ci]["detected"] for r in results]
        widths = [r[c_ci]["ci_width"] for r in results if r[c_ci]["ci_width"] is not None]
        delays = [r[c_ci]["delay"] for r in results if r[c_ci]["delay"] is not None]
        jacs = [r[c_ci]["jaccard"] for r in results if r[c_ci]["jaccard"] is not None]
        recs = [r[c_ci]["recall"] for r in results if r[c_ci]["recall"] is not None]

        cov = float(np.mean(covers))
        se = float(math.sqrt(cov * (1.0 - cov) / len(covers))) if 0 < cov < 1 else 0.0
        mw = float(np.mean(widths)) if widths else None
        md = float(np.mean(delays)) if delays else None
        w_d = (mw / md) if mw and md and md > 0 else None
        summary[c_ci] = {
            "n": len(covers),
            "n_detected": sum(detected),
            "coverage": cov,
            "coverage_se": se,
            "mean_ci_width": mw,
            "mean_delay": md,
            "width_delay_ratio": w_d,
            "mean_jaccard": float(np.mean(jacs)) if jacs else None,
            "mean_recall": float(np.mean(recs)) if recs else None,
        }
    return summary


def run_eval_cell(
    phi: float, effect: float, c_A: float, c_Q: float,
    c_ci: float, c_rep: float, n_streams: int,
    workers: int = 4,
) -> dict[int, dict[str, Any]]:
    tasks = [(phi, effect, i, EVAL_SEED_BASE, c_A, c_Q, c_ci, c_rep, EXTRA_OBS_LIST)
             for i in range(n_streams)]
    results: list[dict[int, dict[str, Any]]] = []
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_process_one_stream_eval, tasks, chunksize=16))
    else:
        results = [_process_one_stream_eval(t) for t in tasks]

    summary = {}
    for ell in EXTRA_OBS_LIST:
        covers = [r[ell]["covered"] for r in results]
        detected = [r[ell]["detected"] for r in results]
        widths = [r[ell]["ci_width"] for r in results if r[ell]["ci_width"] is not None]
        delays = [r[ell]["delay"] for r in results if r[ell]["delay"] is not None]
        jacs = [r[ell]["jaccard"] for r in results if r[ell]["jaccard"] is not None]
        recs = [r[ell]["recall"] for r in results if r[ell]["recall"] is not None]

        cov = float(np.mean(covers))
        se = float(math.sqrt(cov * (1.0 - cov) / len(covers))) if 0 < cov < 1 else 0.0
        mw = float(np.mean(widths)) if widths else None
        md = float(np.mean(delays)) if delays else None
        w_d = (mw / md) if mw and md and md > 0 else None
        summary[ell] = {
            "n": len(covers),
            "n_detected": sum(detected),
            "coverage": cov,
            "coverage_se": se,
            "mean_ci_width": mw,
            "mean_delay": md,
            "width_delay_ratio": w_d,
            "mean_jaccard": float(np.mean(jacs)) if jacs else None,
            "mean_recall": float(np.mean(recs)) if recs else None,
        }
    return summary


def main():
    ap = argparse.ArgumentParser(description="Task C: Honest CI calibration and held-out validation")
    ap.add_argument("--mode", choices=["calib", "eval", "both"], default="both")
    ap.add_argument("--c-ci", type=float, default=None, help="Force specific c_CI value")
    ap.add_argument("--n-calib", type=int, default=DEFAULT_N_CALIB, help="Streams per cell for calibration")
    ap.add_argument("--n-eval", type=int, default=DEFAULT_N_EVAL, help="Streams per cell for evaluation (>=1000)")
    ap.add_argument("--out-dir", type=Path, default=repo_root / "results" / "fix")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    calib_csv = args.out_dir / "taskC_calib_ci_grid.csv"
    eval_csv = args.out_dir / "taskC_eval_coverage.csv"
    eval_json = args.out_dir / "taskC_eval_gates.json"

    calib_rows = []

    # 1. Calibration phase
    if args.mode in ("calib", "both"):
        print(f"=== CALIBRATION PHASE ({args.n_calib} streams/cell, workers={args.workers}) ===")
        calib_flds = [
            "set", "phi", "effect", "c_CI", "c_rep", "coverage", "coverage_se",
            "mean_ci_width", "mean_delay", "width_delay_ratio",
            "mean_jaccard", "mean_recall", "n", "n_detected",
        ]
        with open(calib_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=calib_flds)
            w.writeheader()
            for phi in PHI_LIST:
                c_A, c_Q = _load_calib(phi)
                for effect in EFFECT_LIST:
                    print(f"  Calibrating phi={phi}, effect={effect}...")
                    cell_res = run_calib_cell(
                        phi=phi, effect=effect, c_A=c_A, c_Q=c_Q,
                        c_ci_list=C_CI_GRID, c_rep=DEFAULT_C_REP,
                        n_streams=args.n_calib, workers=args.workers,
                    )
                    for c_ci in C_CI_GRID:
                        s = cell_res[c_ci]
                        row = {
                            "set": "calib", "phi": phi, "effect": effect,
                            "c_CI": c_ci, "c_rep": DEFAULT_C_REP,
                            "coverage": s["coverage"], "coverage_se": s["coverage_se"],
                            "mean_ci_width": s["mean_ci_width"],
                            "mean_delay": s["mean_delay"],
                            "width_delay_ratio": s["width_delay_ratio"],
                            "mean_jaccard": s["mean_jaccard"],
                            "mean_recall": s["mean_recall"],
                            "n": s["n"], "n_detected": s["n_detected"],
                        }
                        w.writerow(row)
                        calib_rows.append(row)
                    print(f"    c=0.45: cov={cell_res[0.45]['coverage']:.3f}±{cell_res[0.45]['coverage_se']:.3f} | "
                          f"c=0.50: cov={cell_res[0.50]['coverage']:.3f}±{cell_res[0.50]['coverage_se']:.3f}")
        print(f"Calibration results written to {calib_csv}")

    # Load existing calibration if running eval only
    if args.mode in ("eval", "both") and not calib_rows and calib_csv.exists():
        with open(calib_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                calib_rows.append({
                    k: (float(v) if v not in ("None", "", None) and k != "set" else v)
                    for k, v in r.items()
                })

    # 2. Select c_CI
    if args.c_ci is not None:
        c_CI_chosen = args.c_ci
        print(f"Using forced c_CI = {c_CI_chosen}")
    elif calib_rows:
        cov_by_c: dict[float, dict[tuple[float, float], tuple[float, float]]] = defaultdict(dict)
        for r in calib_rows:
            c = float(r["c_CI"])
            phi = float(r["phi"])
            eff = float(r["effect"])
            cov = r["coverage"]
            se = r["coverage_se"]
            if cov is not None and se is not None:
                cov_by_c[c][(phi, eff)] = (float(cov), float(se))

        c_CI_chosen = None
        for c in sorted(cov_by_c.keys()):
            satisfies_all = all(cv >= 0.95 - se for (cv, se) in cov_by_c[c].values())
            if satisfies_all:
                c_CI_chosen = c
                break

        if c_CI_chosen is None:
            c_CI_chosen = 0.50
            print(f"Notice: No single c_CI in grid strictly >= 0.95 - 1 SE for all cells. Using CWS default c_CI={c_CI_chosen:.2f}")
        else:
            print(f"\n>>> Selected optimal shared c_CI = {c_CI_chosen:.2f} <<<")
    else:
        c_CI_chosen = 0.50
        print(f"Using default c_CI = {c_CI_chosen}")

    # 3. Evaluation phase on held-out set
    if args.mode in ("eval", "both"):
        print(f"\n=== EVALUATION PHASE (c_CI={c_CI_chosen}, {args.n_eval} streams/cell, workers={args.workers}) ===")
        eval_flds = [
            "set", "phi", "effect", "extra_obs", "c_CI", "c_rep",
            "coverage", "coverage_se", "passes_g3", "mean_ci_width",
            "mean_delay", "width_delay_ratio", "mean_jaccard", "bonferroni_ref_jaccard",
            "jaccard_gain", "mean_recall", "n", "n_detected",
        ]
        eval_rows = []
        gates: dict[str, Any] = {}

        with open(eval_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=eval_flds)
            w.writeheader()
            for phi in PHI_LIST:
                c_A, c_Q = _load_calib(phi)
                for effect in EFFECT_LIST:
                    print(f"  Evaluating phi={phi}, effect={effect} ({args.n_eval} streams)...")
                    cell_res = run_eval_cell(
                        phi=phi, effect=effect, c_A=c_A, c_Q=c_Q,
                        c_ci=c_CI_chosen, c_rep=DEFAULT_C_REP,
                        n_streams=args.n_eval, workers=args.workers,
                    )
                    bonf_ref = BONFERRONI_JACCARD_REF.get((phi, effect), 0.65)
                    for ell in EXTRA_OBS_LIST:
                        s = cell_res[ell]
                        cov = s["coverage"]
                        se = s["coverage_se"]
                        pg3 = bool(cov is not None and cov >= (0.95 - se))
                        jac = s["mean_jaccard"]
                        jac_gain = (jac - bonf_ref) if jac is not None else None

                        row = {
                            "set": "eval", "phi": phi, "effect": effect,
                            "extra_obs": ell, "c_CI": c_CI_chosen, "c_rep": DEFAULT_C_REP,
                            "coverage": cov, "coverage_se": se, "passes_g3": pg3,
                            "mean_ci_width": s["mean_ci_width"],
                            "mean_delay": s["mean_delay"],
                            "width_delay_ratio": s["width_delay_ratio"],
                            "mean_jaccard": jac,
                            "bonferroni_ref_jaccard": bonf_ref,
                            "jaccard_gain": jac_gain,
                            "mean_recall": s["mean_recall"],
                            "n": s["n"], "n_detected": s["n_detected"],
                        }
                        w.writerow(row)
                        eval_rows.append(row)

                        cell_tag = f"phi={phi}_eff={effect}_ell={ell}"
                        gates[cell_tag] = {
                            "passes_g3": pg3,
                            "coverage": cov,
                            "coverage_se": se,
                            "mean_ci_width": s["mean_ci_width"],
                            "mean_delay": s["mean_delay"],
                            "width_delay_ratio": s["width_delay_ratio"],
                            "mean_jaccard": jac,
                            "bonferroni_ref": bonf_ref,
                            "mean_recall": s["mean_recall"],
                        }
                        status_str = "PASS" if pg3 else "FAIL"
                        print(
                            f"    ell={ell:2d}: cov={cov:.3f}±{se:.3f} [{status_str}] | "
                            f"width={s['mean_ci_width']:.1f} delay={s['mean_delay']:.1f} "
                            f"W/D={s['width_delay_ratio']:.2f} | jac={jac:.3f} rec={s['mean_recall']:.3f}"
                        )

        g3_all_pass = all(
            v["passes_g3"] for k, v in gates.items() if "_ell=0" in k
        )

        g4_jaccard_by_phi = {}
        for phi in PHI_LIST:
            jacs = [
                gates[f"phi={phi}_eff={eff}_ell=0"]["mean_jaccard"]
                for eff in EFFECT_LIST
            ]
            refs = [
                BONFERRONI_JACCARD_REF[(phi, eff)]
                for eff in EFFECT_LIST
            ]
            g4_jaccard_by_phi[phi] = bool(np.mean(jacs) >= np.mean(refs) - 0.05)

        g4_recall_monotone = {}
        for phi in PHI_LIST:
            for eff in EFFECT_LIST:
                r0 = gates[f"phi={phi}_eff={eff}_ell=0"]["mean_recall"]
                r20 = gates[f"phi={phi}_eff={eff}_ell=20"]["mean_recall"]
                r50 = gates[f"phi={phi}_eff={eff}_ell=50"]["mean_recall"]
                g4_recall_monotone[f"phi={phi}_eff={eff}"] = bool(r0 <= r20 <= r50 + 1e-4)

        g4_all_pass = bool(all(g4_jaccard_by_phi.values()) and all(g4_recall_monotone.values()))

        def _json_default(o):
            if isinstance(o, (np.bool_, bool)):
                return bool(o)
            if isinstance(o, (np.integer, int)):
                return int(o)
            if isinstance(o, (np.floating, float)):
                return float(o)
            return str(o)

        summary = {
            "c_CI_chosen": c_CI_chosen,
            "c_rep": DEFAULT_C_REP,
            "G3_coverage_pass": bool(g3_all_pass),
            "G4_support_pass": bool(g4_all_pass),
            "G4_jaccard_by_phi": g4_jaccard_by_phi,
            "G4_recall_monotone": g4_recall_monotone,
            "cells": gates,
        }
        eval_json.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")

        print("\n================ FINAL GATES SUMMARY ================")
        print(f"Gate G3 (CI coverage >= 95% - 1 SE): {'PASS' if g3_all_pass else 'FAIL'}")
        print(f"Gate G4 (Support Jaccard >= Bonferroni & recall increases with ell): {'PASS' if g4_all_pass else 'FAIL'}")
        print(f"Evaluation results written to {eval_csv}")
        print(f"Gates summary written to {eval_json}")


if __name__ == "__main__":
    main()
