#!/usr/bin/env python3
"""Run all 24 evaluation experiments in parallel (Step d)."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def build_eval_tasks() -> list[dict]:
    tags = ["00", "03", "06", "08"]
    tasks = []

    for tag in tags:
        prop_calib = f"results/calib/proposed_phi{tag}.json"
        van_calib = "results/calib/vanilla_iid.json"
        ci_json = "results/calib/ci_a.json"

        # 1. Null streams
        null_dir = f"data/null_phi{tag}"
        tasks.append({
            "name": f"proposed_null_phi{tag}",
            "dataset_dir": null_dir,
            "calib_json": prop_calib,
            "ci_json": None,
            "method": "proposed",
            "out_csv": f"results/eval/proposed_null_phi{tag}.csv",
            "out_json": f"results/eval/proposed_null_phi{tag}.json",
        })
        tasks.append({
            "name": f"vanilla_null_phi{tag}",
            "dataset_dir": null_dir,
            "calib_json": van_calib,
            "ci_json": None,
            "method": "vanilla",
            "out_csv": f"results/eval/vanilla_null_phi{tag}.csv",
            "out_json": f"results/eval/vanilla_null_phi{tag}.json",
        })

        # 2. Changed streams: e10 and e15
        for eff in ["e10", "e15"]:
            change_dir = f"data/change_phi{tag}_{eff}"
            tasks.append({
                "name": f"proposed_change_phi{tag}_{eff}",
                "dataset_dir": change_dir,
                "calib_json": prop_calib,
                "ci_json": ci_json,
                "method": "proposed",
                "out_csv": f"results/eval/proposed_change_phi{tag}_{eff}.csv",
                "out_json": f"results/eval/proposed_change_phi{tag}_{eff}.json",
            })
            tasks.append({
                "name": f"vanilla_change_phi{tag}_{eff}",
                "dataset_dir": change_dir,
                "calib_json": van_calib,
                "ci_json": ci_json,
                "method": "vanilla",
                "out_csv": f"results/eval/vanilla_change_phi{tag}_{eff}.csv",
                "out_json": f"results/eval/vanilla_change_phi{tag}_{eff}.json",
            })

    return tasks


def run_single_eval(task: dict) -> tuple[str, float]:
    t0 = time.perf_counter()
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_eval.py"),
        "--dataset-dir", task["dataset_dir"],
        "--calib-json", task["calib_json"],
        "--out-csv", task["out_csv"],
        "--out-json", task["out_json"],
        "--method", task["method"],
    ]
    if task["ci_json"]:
        cmd.extend(["--ci-json", task["ci_json"]])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed evaluation {task['name']}:\n{proc.stderr}")

    elapsed = time.perf_counter() - t0
    return task["name"], elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all 24 evaluation experiments.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Worker count.")
    args = parser.parse_args()

    tasks = build_eval_tasks()
    print(f"Running {len(tasks)} evaluations using {args.workers} workers...")
    t0 = time.perf_counter()

    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_eval, t): t["name"] for t in tasks}
        for fut in as_completed(futures):
            name, elapsed = fut.result()
            completed += 1
            print(f"[{completed:2d}/{len(tasks):2d}] {name:32s} -> DONE ({elapsed:.1f}s)", flush=True)

    total_time = time.perf_counter() - t0
    print(f"\nAll {len(tasks)} evaluations completed in {total_time:.1f}s.")


if __name__ == "__main__":
    main()
