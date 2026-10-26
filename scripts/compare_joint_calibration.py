#!/usr/bin/env python3
"""G2: Compare joint calibration of (c_A, c_Q) at level alpha0 vs alpha0/2 split."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from ocdmix.calibrate import calibrate_thresholds
from ocdmix.io import load_observations


def main() -> None:
    data_dir = Path("data")
    phis = ["00", "03", "06", "08"]
    records = []

    print("=== G2: Joint Calibration vs alpha0/2 Split Comparison ===")
    for tag in phis:
        null_dir = data_dir / f"calib_null_phi{tag}"
        if not null_dir.exists():
            continue

        obs = load_observations(null_dir, mmap=True)
        # 1. Standard alpha0/2 split
        calib_split = calibrate_thresholds(obs, alpha0=0.05, method="proposed", joint=False)
        # 2. Joint calibration
        calib_joint = calibrate_thresholds(obs, alpha0=0.05, method="proposed", joint=True)

        rec = {
            "phi_tag": tag,
            "n_streams": obs.shape[0],
            "split_c_A": calib_split.c_A,
            "split_c_Q": calib_split.c_Q,
            "split_far": calib_split.achieved_far,
            "joint_c_A": calib_joint.c_A,
            "joint_c_Q": calib_joint.c_Q,
            "joint_far": calib_joint.achieved_far,
            "c_A_diff_pct": (calib_joint.c_A - calib_split.c_A) / calib_split.c_A * 100,
            "c_Q_diff_pct": (calib_joint.c_Q - calib_split.c_Q) / calib_split.c_Q * 100,
        }
        records.append(rec)
        print(
            f"phi={tag}: Split FAR = {rec['split_far']:.4f} (cA={rec['split_c_A']:.4f}, cQ={rec['split_c_Q']:.4f}) | "
            f"Joint FAR = {rec['joint_far']:.4f} (cA={rec['joint_c_A']:.4f}, cQ={rec['joint_c_Q']:.4f}) | "
            f"Diff: cA {rec['c_A_diff_pct']:+.2f}%, cQ {rec['c_Q_diff_pct']:+.2f}%"
        )

    df = pd.DataFrame(records)
    out_path = Path("results/tables/joint_vs_split_calibration.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved comparison to {out_path}")


if __name__ == "__main__":
    main()
