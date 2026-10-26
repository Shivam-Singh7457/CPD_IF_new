#!/usr/bin/env python3
"""Benchmark OnlineDetector throughput and memory scaling (Task 1 / Task 8).

Two-pass benchmark:
1. Speed pass: tracemalloc OFF, timing per-observation updates over 50,000 steps.
2. Memory pass: tracemalloc started BEFORE detector is built, recording:
   - footprint_kib_after_build
   - live_kib and peak_kib at 5k, 20k, 50k steps (reset_peak between checkpoints)
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
import tracemalloc

import numpy as np
import pandas as pd

from ocdmix.fit import fit_training
from ocdmix.online import OnlineDetector


def benchmark_single_p(
    p: int,
    speed_steps: int = 50000,
    checkpoints: tuple[int, ...] = (5000, 20000, 50000),
    m0: int = 1000,
    seed: int = 20261014,
) -> tuple[dict, dict]:
    """Two-pass benchmark for dimension p: speed pass and memory pass."""
    rng = np.random.default_rng(seed + p)

    # Pre-generate synthetic training data and cyclic test data pool
    X_train = rng.standard_normal((m0, p))
    fit = fit_training(X_train, m0=m0, method="proposed", lrv_method="prewhiten")

    pool_size = 5000
    data_pool = rng.standard_normal((pool_size, p))

    # =========================================================================
    # PASS 1: Speed Pass (tracemalloc OFF)
    # =========================================================================
    detector_speed = OnlineDetector(fit, c_A=100.0, c_Q=100.0)
    for t in range(m0):
        detector_speed.update(X_train[t])

    t0 = time.perf_counter()
    for i in range(speed_steps):
        detector_speed.update(data_pool[i % pool_size])
    t1 = time.perf_counter()

    total_speed_time = t1 - t0
    us_per_update = (total_speed_time / speed_steps) * 1e6
    updates_per_sec = speed_steps / total_speed_time

    speed_record = {
        "p": p,
        "speed_steps": speed_steps,
        "total_time_s": total_speed_time,
        "us_per_update": us_per_update,
        "updates_per_sec": updates_per_sec,
    }

    # =========================================================================
    # PASS 2: Memory Pass (tracemalloc started BEFORE detector is built)
    # =========================================================================
    del detector_speed
    gc.collect()

    tracemalloc.start()
    tracemalloc.reset_peak()

    # Build detector inside tracemalloc
    detector_mem = OnlineDetector(fit, c_A=100.0, c_Q=100.0)
    current_mem, _ = tracemalloc.get_traced_memory()
    footprint_kib_after_build = current_mem / 1024.0

    # Feed training steps
    for t in range(m0):
        detector_mem.update(X_train[t])

    tracemalloc.reset_peak()

    mem_checkpoints: dict[str, float] = {
        "p": p,
        "footprint_kib_after_build": footprint_kib_after_build,
    }

    current_step = 0
    for cp in checkpoints:
        steps_to_run = cp - current_step
        for i in range(steps_to_run):
            step_idx = (current_step + i) % pool_size
            detector_mem.update(data_pool[step_idx])
        current_step = cp

        live, peak = tracemalloc.get_traced_memory()
        k_tag = f"{cp // 1000}k"
        mem_checkpoints[f"live_kib_{k_tag}"] = live / 1024.0
        mem_checkpoints[f"peak_kib_{k_tag}"] = peak / 1024.0

        # Reset peak between checkpoints as required
        tracemalloc.reset_peak()

    tracemalloc.stop()
    del detector_mem
    gc.collect()

    return speed_record, mem_checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-pass benchmark of OnlineDetector.")
    parser.add_argument("--steps", type=int, default=50000, help="Steps for speed pass (default: 50000).")
    parser.add_argument("--out-speed", type=str, default="results/tables/benchmark_speed.csv", help="Speed CSV.")
    parser.add_argument("--out-mem", type=str, default="results/tables/benchmark_memory.csv", help="Memory CSV.")
    parser.add_argument("--out-combined", type=str, default="results/tables/benchmark_online.csv", help="Combined CSV.")
    args = parser.parse_args()

    dimensions = [50, 100, 200, 400]
    speed_results = []
    mem_results = []
    combined_results = []

    print(f"Running two-pass OnlineDetector benchmark for p in {dimensions}...")
    for p in dimensions:
        print(f"\n--- Dimension p = {p} ---")
        t_start = time.perf_counter()
        speed_rec, mem_rec = benchmark_single_p(p, speed_steps=args.steps)
        t_elapsed = time.perf_counter() - t_start

        print(f"  [Speed pass]  : {speed_rec['us_per_update']:.2f} us/update ({speed_rec['updates_per_sec']:,.0f} updates/sec)")
        print(f"  [Build mem]   : footprint = {mem_rec['footprint_kib_after_build']:.2f} KiB")
        print(f"  [5k steps]    : live = {mem_rec['live_kib_5k']:.2f} KiB, peak = {mem_rec['peak_kib_5k']:.2f} KiB")
        print(f"  [20k steps]   : live = {mem_rec['live_kib_20k']:.2f} KiB, peak = {mem_rec['peak_kib_20k']:.2f} KiB")
        print(f"  [50k steps]   : live = {mem_rec['live_kib_50k']:.2f} KiB, peak = {mem_rec['peak_kib_50k']:.2f} KiB")
        print(f"  Elapsed       : {t_elapsed:.1f}s")

        speed_results.append(speed_rec)
        mem_results.append(mem_rec)

        combined = {**speed_rec, **mem_rec}
        combined_results.append(combined)

    # Save CSVs
    Path(args.out_speed).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(speed_results).to_csv(args.out_speed, index=False)
    pd.DataFrame(mem_results).to_csv(args.out_mem, index=False)
    pd.DataFrame(combined_results).to_csv(args.out_combined, index=False)

    print(f"\nSaved CSVs:")
    print(f"  Speed CSV   : {args.out_speed}")
    print(f"  Memory CSV  : {args.out_mem}")
    print(f"  Combined CSV: {args.out_combined}")


if __name__ == "__main__":
    main()
