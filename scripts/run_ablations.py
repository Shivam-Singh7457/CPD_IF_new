#!/usr/bin/env python3
"""Run ablations for vanilla and proposed (Task 3).

Ablations:
1. l-infinity statistic only (linf_only)
2. Q statistic only (q_only)
3. k_min = 4 (window grid starting at 4)
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def run_cmd(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nStderr: {proc.stderr}")


def calibrate_all_ablations() -> None:
    print("--- Calibrating Ablation Thresholds ---")
    calib_script = SCRIPT_DIR / "run_calibration.py"
    tags = ["00", "03", "06", "08"]

    # 1. Vanilla ablations (calibrated on calib_null_phi00)
    van_tasks = [
        ("results/calib/vanilla_linf.json", "vanilla", "linf_only", None),
        ("results/calib/vanilla_q.json", "vanilla", "q_only", None),
        ("results/calib/vanilla_kmin4.json", "vanilla", "both", 4),
    ]
    for out_json, method, stat, k_min in van_tasks:
        if not Path(out_json).exists():
            print(f"Calibrating {out_json}...")
            cmd = [
                sys.executable, str(calib_script),
                "--null-dir", "data/calib_null_phi00",
                "--method", method,
                "--statistic", stat,
                "--out", out_json,
            ]
            if k_min is not None:
                cmd.extend(["--k-min", str(k_min)])
            run_cmd(cmd)

    # 2. Proposed ablations for each phi
    for tag in tags:
        prop_tasks = [
            (f"results/calib/proposed_linf_phi{tag}.json", "proposed", "linf_only", None),
            (f"results/calib/proposed_q_phi{tag}.json", "proposed", "q_only", None),
            (f"results/calib/proposed_kmin4_phi{tag}.json", "proposed", "both", 4),
        ]
        for out_json, method, stat, k_min in prop_tasks:
            if not Path(out_json).exists():
                print(f"Calibrating {out_json}...")
                cmd = [
                    sys.executable, str(calib_script),
                    "--null-dir", f"data/calib_null_phi{tag}",
                    "--method", method,
                    "--statistic", stat,
                    "--out", out_json,
                ]
                if k_min is not None:
                    cmd.extend(["--k-min", str(k_min)])
                run_cmd(cmd)

    print("All ablation calibrations complete.\n")


def build_ablation_eval_tasks() -> list[dict]:
    tags = ["00", "03", "06", "08"]
    ablations = [
        ("linf", "linf_only", None),
        ("q", "q_only", None),
        ("kmin4", "both", 4),
    ]
    tasks = []

    for ab_name, stat, k_min in ablations:
        for tag in tags:
            # Calib files
            van_calib = f"results/calib/vanilla_{ab_name}.json"
            prop_calib = f"results/calib/proposed_{ab_name}_phi{tag}.json"
            ci_json = "results/calib/ci_a.json"

            # 1. Null streams
            null_dir = f"data/null_phi{tag}"
            for method, calib_file in [("proposed", prop_calib), ("vanilla", van_calib)]:
                t_name = f"{method}_{ab_name}_null_phi{tag}"
                tasks.append({
                    "name": t_name,
                    "ablation": ab_name,
                    "method": method,
                    "phi": float(f"0.{tag[1:]}" if tag != "00" else "0.0"),
                    "tag": tag,
                    "dataset_dir": null_dir,
                    "dataset_name": f"null_phi{tag}",
                    "calib_json": calib_file,
                    "ci_json": None,
                    "statistic": stat,
                    "k_min": k_min,
                    "out_csv": f"results/eval/{t_name}.csv",
                    "out_json": f"results/eval/{t_name}.json",
                })

            # 2. Changed streams
            for eff in ["e10", "e15"]:
                change_dir = f"data/change_phi{tag}_{eff}"
                for method, calib_file in [("proposed", prop_calib), ("vanilla", van_calib)]:
                    t_name = f"{method}_{ab_name}_change_phi{tag}_{eff}"
                    tasks.append({
                        "name": t_name,
                        "ablation": ab_name,
                        "method": method,
                        "phi": float(f"0.{tag[1:]}" if tag != "00" else "0.0"),
                        "tag": tag,
                        "effect": 1.0 if eff == "e10" else 1.5,
                        "dataset_dir": change_dir,
                        "dataset_name": f"change_phi{tag}_{eff}",
                        "calib_json": calib_file,
                        "ci_json": ci_json,
                        "statistic": stat,
                        "k_min": k_min,
                        "out_csv": f"results/eval/{t_name}.csv",
                        "out_json": f"results/eval/{t_name}.json",
                    })

    return tasks


def run_single_eval(task: dict) -> dict:
    eval_script = SCRIPT_DIR / "run_eval.py"
    cmd = [
        sys.executable, str(eval_script),
        "--dataset-dir", task["dataset_dir"],
        "--calib-json", task["calib_json"],
        "--method", task["method"],
        "--statistic", task["statistic"],
        "--out-csv", task["out_csv"],
        "--out-json", task["out_json"],
    ]
    if task["k_min"] is not None:
        cmd.extend(["--k-min", str(task["k_min"])])
    if task["ci_json"]:
        cmd.extend(["--ci-json", task["ci_json"]])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed evaluation {task['name']}:\n{proc.stderr}")

    with open(task["out_json"], "r", encoding="utf-8") as f:
        res = json.load(f)

    res["name"] = task["name"]
    res["ablation"] = task["ablation"]
    res["phi"] = task["phi"]
    res["effect"] = task.get("effect")
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablations for OCD detector.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Worker count.")
    args = parser.parse_args()

    calibrate_all_ablations()

    tasks = build_ablation_eval_tasks()
    print(f"Running {len(tasks)} ablation evaluations using {args.workers} workers...")
    t0 = time.perf_counter()

    results = []
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_eval, t): t["name"] for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            completed += 1
            print(f"[{completed:2d}/{len(tasks):2d}] {res['name']:38s} -> DONE", flush=True)

    total_time = time.perf_counter() - t0
    print(f"\nAll {len(tasks)} ablation evaluations finished in {total_time:.1f}s.")

    # Compile summary table
    df = pd.DataFrame(results)
    out_table = PROJECT_ROOT / "results" / "tables" / "ablations_summary.csv"
    out_table.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_table, index=False, float_format="%.6f", lineterminator="\n")
    print(f"Summary saved to {out_table}")


if __name__ == "__main__":
    main()
