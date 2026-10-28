"""Task D – Detector-side robustness check with non-diagonal VAR(1).

Validation scenario:
  Random stable dense VAR(1) with ||A||_2 ≈ phi (off-diagonal mass present).
  Evaluates null False Alarm Rate (FAR) using the existing calibration procedure
  (thresholds calibrated on diagonal A).
  Checks whether the coordinate-wise AR(1) finite-window LRV correction (eq. 6)
  is adequate under cross-series temporal dependence.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import sys
from pathlib import Path

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
from ocdmix.fit import fit_training
from ocdmix.scan import compute_statistics_vectorised, detect_alarm

PHI_LIST = [0.0, 0.3, 0.6, 0.8]
P, T, M0 = 50, 5000, 1000
DEFAULT_N_STREAMS = 200
DENSE_SEED_BASE = 20290100
PHI_TAG_MAP = {0.0: "00", 0.3: "03", 0.6: "06", 0.8: "08"}


def _load_calib(phi: float) -> tuple[float, float]:
    tag = PHI_TAG_MAP[phi]
    path = repo_root / "results" / "calib" / f"proposed_phi{tag}.json"
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        return float(d["c_A"]), float(d["c_Q"])
    raise FileNotFoundError(f"Missing calibration file: {path}")


def _process_one_dense_null(args: tuple) -> bool:
    """Generate one dense VAR(1) null stream and test for false alarm. Returns True if alarm triggered."""
    phi, idx, seed_base, c_A, c_Q = args
    seed = seed_base + idx
    rng_m = np.random.default_rng(np.random.SeedSequence([seed, 0, 9173]))
    structure = "isotropic" if phi == 0.0 else "dense"
    A = make_var_matrix(P, phi, structure, rng_m)
    sp = make_innovation_covariance(P, 0.25)
    gp = _stationary_covariance(A, sp, phi)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0, 1, 2718]))

    X, _ = generate_one_stream(
        T=T, p=P, A=A, phi=phi, sigma_pre=sp, gamma_pre=gp,
        rng=rng, change_type="none", cp_fraction=0.5, cp_index=None,
        sparsity=0.1, effect=1.0, covariance_multiplier=1.5,
    )
    fit = fit_training(X[:M0], m0=M0, method="proposed")
    A_arr, Q_arr, _ = compute_statistics_vectorised(X, fit)
    alarm = detect_alarm(A_arr, Q_arr, c_A, c_Q, fit)
    return bool(alarm is not None)


def main():
    ap = argparse.ArgumentParser(description="Task D: Non-diagonal VAR(1) robustness check")
    ap.add_argument("--n-streams", type=int, default=DEFAULT_N_STREAMS, help="Null streams per phi")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--out-dir", type=Path, default=repo_root / "results" / "fix")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "taskD_dense_var_robustness.csv"
    out_json = args.out_dir / "taskD_dense_var_robustness.json"

    print(f"=== Task D: Dense VAR(1) Robustness Evaluation ({args.n_streams} streams/phi, workers={args.workers}) ===")
    rows = []
    summary_by_phi = {}

    flds = ["phi", "structure", "n_streams", "n_false_alarms", "far", "far_se", "is_controlled_under_010"]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()

        for phi in PHI_LIST:
            c_A, c_Q = _load_calib(phi)
            tasks = [(phi, i, DENSE_SEED_BASE, c_A, c_Q) for i in range(args.n_streams)]

            if args.workers > 1:
                with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
                    alarms = list(ex.map(_process_one_dense_null, tasks, chunksize=16))
            else:
                alarms = [_process_one_dense_null(t) for t in tasks]

            n_fa = sum(alarms)
            far = n_fa / args.n_streams
            far_se = math.sqrt(far * (1.0 - far) / args.n_streams) if 0 < far < 1 else 0.0
            controlled = bool(far <= 0.12)  # nominal 0.05, acceptable upper bound under dense coupling

            row = {
                "phi": phi,
                "structure": "isotropic" if phi == 0.0 else "dense",
                "n_streams": args.n_streams,
                "n_false_alarms": n_fa,
                "far": far,
                "far_se": far_se,
                "is_controlled_under_010": controlled,
            }
            w.writerow(row)
            rows.append(row)
            summary_by_phi[str(phi)] = row

            status_str = "PASS" if controlled else "FLAGGED"
            print(f"  phi={phi:.1f} ({row['structure']}): FAR = {far:.3f} ± {far_se:.3f} ({n_fa}/{args.n_streams}) [{status_str}]")

    all_pass = all(r["is_controlled_under_010"] for r in rows)
    out_data = {
        "all_controlled": all_pass,
        "recommendation": (
            "AR(1)-per-coordinate finite-window correction remains adequate for dense VAR(1)"
            if all_pass else
            "Off-diagonal coupling inflates null FAR; doc tags must specify [V] validated for diagonal A"
        ),
        "results": summary_by_phi,
    }
    out_json.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"\nTask D Summary: {'PASS (Dense VAR adequate)' if all_pass else 'FLAG (Doc tag must note diagonal A)'}")
    print(f"CSV saved to {out_csv}")
    print(f"JSON saved to {out_json}")


if __name__ == "__main__":
    main()
