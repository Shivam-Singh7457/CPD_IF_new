#!/usr/bin/env python3
"""Calibrate detection thresholds on null streams (Task 4)."""
from __future__ import annotations

import argparse
from pathlib import Path

from ocdmix.calibrate import calibrate_thresholds
from ocdmix.fit import DEFAULT_ALPHA0, DEFAULT_M0
from ocdmix.io import load_observations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate OCD thresholds on null streams.")
    parser.add_argument("--null-dir", type=str, required=True, help="Directory containing null stream observations.npy.")
    parser.add_argument("--method", choices=["proposed", "vanilla"], default="proposed", help="Detection method.")
    parser.add_argument("--lrv-method", choices=["prewhiten", "bartlett"], default="prewhiten", help="LRV estimation method.")
    parser.add_argument("--alpha0", type=float, default=DEFAULT_ALPHA0, help="Target false-alarm rate (default: 0.05).")
    parser.add_argument("--m0", type=int, default=DEFAULT_M0, help="Training segment length (default: 1000).")
    parser.add_argument("--statistic", choices=["both", "linf_only", "l_infinity", "q_only"], default="both", help="Detection statistic.")
    parser.add_argument("--k-min", type=int, default=None, help="Minimum window length (e.g. 4).")
    parser.add_argument("--out", type=str, required=True, help="Path to save calibrated thresholds JSON.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    null_dir = Path(args.null_dir)
    obs = load_observations(null_dir, mmap=True)
    print(f"Loaded observations from {null_dir}: shape={obs.shape}, dtype={obs.dtype}")

    window_grid = (8, 16, 32, 64, 128, 256, 512)
    if args.k_min is not None:
        window_grid = tuple(sorted(set([args.k_min] + list(window_grid))))

    result = calibrate_thresholds(
        obs,
        m0=args.m0,
        alpha0=args.alpha0,
        window_grid=window_grid,
        method=args.method,
        lrv_method=args.lrv_method,
        statistic=args.statistic,
    )

    out_path = Path(args.out)
    result.save_json(out_path)

    print(f"\nCalibration Results ({args.method}, lrv={args.lrv_method}):")
    print(f"  c_A           : {result.c_A:.6f}")
    print(f"  c_Q           : {result.c_Q:.6f}")
    print(f"  base_A        : {result.base_A:.6f}")
    print(f"  tau_A         : {result.tau_A:.6f}")
    print(f"  tau_Q         : {result.tau_Q:.6f}")
    print(f"  Achieved FAR  : {result.achieved_far * 100:.2f}% (target: {args.alpha0 * 100:.1f}%)")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
