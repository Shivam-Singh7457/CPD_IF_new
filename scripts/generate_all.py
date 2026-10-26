#!/usr/bin/env python3
"""Generate all synthetic datasets using generate_synthetic_streams_fast.py in parallel."""
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
FAST_GEN = PROJECT_ROOT / "generate_synthetic_streams_fast.py"


def build_dataset_specs(smoke: bool) -> list[dict]:
    """Build dataset generation specifications."""
    phis = [0.0, 0.3, 0.6, 0.8]
    tags = ["00", "03", "06", "08"]

    specs = []

    # 1. calib_null_phi{tag}: 400 streams (or 10 in smoke), scenario_id = 100 + i
    for i, (phi, tag) in enumerate(zip(phis, tags)):
        n_streams = 10 if smoke else 400
        specs.append({
            "name": f"calib_null_phi{tag}",
            "phi": phi,
            "change_type": "none",
            "n_streams": n_streams,
            "scenario_id": 100 + i,
            "effect": 0.5,
            "cp_index": None,
        })

    # 2. null_phi{tag}: 200 streams (or 10 in smoke), scenario_id = 200 + i
    for i, (phi, tag) in enumerate(zip(phis, tags)):
        n_streams = 10 if smoke else 200
        specs.append({
            "name": f"null_phi{tag}",
            "phi": phi,
            "change_type": "none",
            "n_streams": n_streams,
            "scenario_id": 200 + i,
            "effect": 0.5,
            "cp_index": None,
        })

    # 3. change_phi{tag}_e10 and change_phi{tag}_e15: 200 streams (or 10 in smoke)
    # scenario_id = 300 + 10*i + j (j=0 for e10, 1 for e15)
    for i, (phi, tag) in enumerate(zip(phis, tags)):
        for j, (eff_str, eff_val) in enumerate([("10", 1.0), ("15", 1.5)]):
            n_streams = 10 if smoke else 200
            specs.append({
                "name": f"change_phi{tag}_e{eff_str}",
                "phi": phi,
                "change_type": "mean",
                "effect": eff_val,
                "n_streams": n_streams,
                "scenario_id": 300 + 10 * i + j,
                "cp_index": 2500,
            })

    # 4. cicalib: phi=0.6, effect=1.0, 200 streams (or 10 in smoke), scenario_id = 900
    n_streams = 10 if smoke else 200
    specs.append({
        "name": "cicalib",
        "phi": 0.6,
        "change_type": "mean",
        "effect": 1.0,
        "n_streams": n_streams,
        "scenario_id": 900,
        "cp_index": 2500,
    })

    return specs


def run_one_generation(spec: dict, outdir_root: Path) -> tuple[str, float, bool]:
    """Run generation for a single dataset via subprocess."""
    outdir = outdir_root / spec["name"]
    obs_file = outdir / "observations.npy"

    if obs_file.exists():
        return spec["name"], 0.0, True  # skipped

    t0 = time.perf_counter()
    cmd = [
        sys.executable,
        str(FAST_GEN),
        "--outdir", str(outdir),
        "--T", "5000",
        "--p", "50",
        "--n-streams", str(spec["n_streams"]),
        "--phi", str(spec["phi"]),
        "--var-structure", "isotropic",
        "--cross-corr", "0.25",
        "--change-type", spec["change_type"],
        "--sparsity", "0.1",
        "--effect", str(spec["effect"]),
        "--sign-mode", "random",
        "--seed", "20261014",
        "--scenario-id", str(spec["scenario_id"]),
        "--dtype", "float32",
    ]
    if spec["cp_index"] is not None:
        cmd.extend(["--cp-index", str(spec["cp_index"])])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to generate {spec['name']}:\n{proc.stderr}")

    elapsed = time.perf_counter() - t0
    return spec["name"], elapsed, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all synthetic datasets in parallel.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Number of worker processes.")
    parser.add_argument("--smoke", action="store_true", help="Generate small datasets (10 streams each) for testing.")
    parser.add_argument("--outdir", type=str, default="data", help="Root data directory.")
    args = parser.parse_args()

    outdir_root = PROJECT_ROOT / args.outdir
    outdir_root.mkdir(parents=True, exist_ok=True)

    specs = build_dataset_specs(smoke=args.smoke)
    print(f"Generating {len(specs)} datasets with {args.workers} workers (smoke={args.smoke})...")
    total_t0 = time.perf_counter()

    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one_generation, spec, outdir_root): spec["name"] for spec in specs}
        for fut in as_completed(futures):
            name, elapsed, skipped = fut.result()
            completed += 1
            status = f"SKIPPED (already exists)" if skipped else f"DONE ({elapsed:.1f}s)"
            print(f"[{completed:2d}/{len(specs):2d}] {name:25s} -> {status}", flush=True)

    total_time = time.perf_counter() - total_t0
    print(f"\nAll datasets generated successfully in {total_time:.1f}s.")


if __name__ == "__main__":
    main()
