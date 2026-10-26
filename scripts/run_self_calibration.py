#!/usr/bin/env python3
"""Task 9: Self-calibration experiment and comparison with oracle calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

from ocdmix.calibrate import CalibrationResult, self_calibrate
from ocdmix.io import load_observations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 9 self-calibration vs oracle comparison.")
    parser.add_argument("--out-csv", type=str, default="results/tables/self_calibration_comparison.csv")
    parser.add_argument("--out-json", type=str, default="results/calib/self_calibration_comparison.json")
    args = parser.parse_args()

    phis = [0.0, 0.3, 0.6, 0.8]
    tags = ["00", "03", "06", "08"]

    results = []

    print("Running Task 9 Self-Calibration (simulating 300 null streams per phi)...")

    for phi, tag in zip(phis, tags):
        oracle_json = Path(f"results/calib/proposed_phi{tag}.json")
        oracle = CalibrationResult.load_json(oracle_json)

        # Load first training segment from calib_null_phi{tag}
        obs = load_observations(f"data/calib_null_phi{tag}", mmap=True)
        X_train = np.asarray(obs[0, :oracle.m0], dtype=np.float64)

        # Run self-calibration
        self_cal = self_calibrate(
            X_train=X_train,
            T=oracle.T,
            m0=oracle.m0,
            alpha0=oracle.alpha0,
            window_grid=tuple(oracle.window_grid),
            kmax=oracle.kmax,
            n_sim_streams=300,
            seed=20261014 + int(phi * 100),
        )

        row = {
            "phi": phi,
            "c_A_oracle": oracle.c_A,
            "c_A_self": self_cal.c_A,
            "c_A_diff_pct": (self_cal.c_A - oracle.c_A) / oracle.c_A * 100,
            "c_Q_oracle": oracle.c_Q,
            "c_Q_self": self_cal.c_Q,
            "c_Q_diff_pct": (self_cal.c_Q - oracle.c_Q) / oracle.c_Q * 100,
            "far_oracle": oracle.achieved_far,
            "far_self": self_cal.achieved_far,
        }
        results.append(row)
        print(f"  phi={phi:.1f}: c_A oracle={oracle.c_A:.4f}, self={self_cal.c_A:.4f} ({row['c_A_diff_pct']:+.1f}%); "
              f"c_Q oracle={oracle.c_Q:.4f}, self={self_cal.c_Q:.4f} ({row['c_Q_diff_pct']:+.1f}%)")

    df = pd.DataFrame(results)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved self-calibration comparison to {out_csv} and {out_json}")


if __name__ == "__main__":
    main()
