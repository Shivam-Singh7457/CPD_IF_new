#!/usr/bin/env python3
"""Evaluate 1000 null streams under oracle calibration."""
import time
from pathlib import Path
import numpy as np
import pandas as pd

from ocdmix.io import load_observations
from ocdmix.fit import fit_training
from ocdmix.scan import compute_statistics_vectorised
from ocdmix.calibrate import CalibrationResult

phis = [0.0, 0.3, 0.6, 0.8]
tags = ["00", "03", "06", "08"]

rows = []
for phi, tag in zip(phis, tags):
    dataset_dir = f"data/null1000_phi{tag}"
    cal_path = f"results/calib/proposed_phi{tag}.json"
    cal = CalibrationResult.load_json(cal_path)
    
    obs = load_observations(dataset_dir, mmap=True)
    N = obs.shape[0]
    print(f"Evaluating phi={phi} on {N} streams with oracle calibration...")
    
    t0 = time.perf_counter()
    alarmed = np.zeros(N, dtype=bool)
    for i in range(N):
        X = np.asarray(obs[i], dtype=np.float64)
        fit = fit_training(X[:cal.m0], m0=cal.m0, window_grid=tuple(cal.window_grid), kmax=cal.kmax)
        A, Q, _ = compute_statistics_vectorised(X, fit)
        M_A = np.nanmax(A) / cal.base_A
        M_Q = np.nanmax(Q)
        if (M_A > cal.c_A) or (M_Q > cal.c_Q):
            alarmed[i] = True
            
    t1 = time.perf_counter()
    far = float(np.mean(alarmed))
    se = float(np.sqrt(far * (1.0 - far) / N))
    print(f"  phi={phi}: FAR = {far:.4f} +/- {se:.4f} (took {t1-t0:.1f}s)")
    rows.append({
        "phi": phi,
        "n_streams": N,
        "oracle_far": far,
        "oracle_se": se,
        "c_A": cal.c_A,
        "c_Q": cal.c_Q,
        "alarmed_count": int(np.sum(alarmed)),
    })

df = pd.DataFrame(rows)
Path("results/tables").mkdir(parents=True, exist_ok=True)
df.to_csv("results/tables/task2_oracle_1000.csv", index=False)
print("Saved to results/tables/task2_oracle_1000.csv")
