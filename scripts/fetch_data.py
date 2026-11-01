#!/usr/bin/env python3
"""Dataset acquisition and provenance recording for OCD-Mix benchmark.

Acquires real-world datasets:
  D1: CDC NCHS weekly deaths by jurisdiction (50 states + DC, Jan 2017 - Jun 2020)
  D2: FRED daily closes (SP500, DJIA, NASDAQCOM, 2021-01-01 to 2023-01-19)
  D3: HASC 2011 challenge person 101 walking & staying files
  D4: MNIST (784 pixels, fixed seed stream order)

Enforces:
  - Never simulate real data
  - Record URL, timestamp, sha256, n, p, date range in results/side_by_side/data_provenance.csv
  - Exits non-zero on any validation failure
"""

import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn.datasets

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
RESULTS_DIR = REPO_ROOT / "results" / "side_by_side"
PROVENANCE_CSV = RESULTS_DIR / "data_provenance.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

provenance_records = []

def record_provenance(dataset_id, title, url, filepath, n, p, date_range, stat_check, stat_status):
    sha = compute_sha256(filepath)
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record = {
        "dataset_id": dataset_id,
        "title": title,
        "source_url": url,
        "local_path": str(filepath.relative_to(REPO_ROOT)),
        "retrieval_timestamp": ts,
        "sha256": sha,
        "n": n,
        "p": p,
        "date_range": date_range,
        "summary_statistic_check": stat_check,
        "check_status": stat_status
    }
    provenance_records.append(record)
    print(f"[{dataset_id}] {title} -> n={n}, p={p}, check: {stat_status} ({stat_check})")

# ==============================================================================
# D1: CDC NCHS Weekly Deaths
# ==============================================================================
def fetch_d1():
    print("\n--- Fetching D1: CDC NCHS Weekly Deaths ---")
    url = "https://data.cdc.gov/api/views/xkkf-xrst/rows.csv?accessType=DOWNLOAD"
    dataset_title = "Excess Deaths Associated with COVID-19"
    dataset_id_cdc = "xkkf-xrst"
    raw_csv = RAW_DIR / "cdc_excess_deaths_xkkf_xrst.csv"
    
    if not raw_csv.exists():
        print(f"Downloading CDC dataset {dataset_id_cdc}...")
        subprocess.run(["curl.exe", "-s", url, "-o", str(raw_csv)], check=True)
    
    df = pd.read_csv(raw_csv)
    # Check jurisdictions: 50 states + DC (combine NY and NYC if separate)
    df["date"] = pd.to_datetime(df["Week Ending Date"])
    
    # Filter to All causes and Predicted (weighted)
    sub = df[(df["Outcome"] == "All causes") & (df["Type"] == "Predicted (weighted)")].copy()
    
    # Exclude non-states like Puerto Rico and United States
    excluded = ["Puerto Rico", "United States"]
    sub = sub[~sub["State"].isin(excluded)]
    
    # Sum New York and New York City into "New York"
    sub["Jurisdiction"] = sub["State"].replace({"New York City": "New York"})
    
    # Group by Jurisdiction and date
    pivoted = sub.groupby(["date", "Jurisdiction"])["Observed Number"].sum().unstack()
    
    # Restrict to Jan 2017 - Jun 2020
    pivoted_period = pivoted.loc["2017-01-01":"2020-06-30"]
    
    # Save processed clean CSV
    clean_path = RAW_DIR / "cdc_weekly_deaths_clean.csv"
    pivoted_period.to_csv(clean_path)
    
    n_weeks, n_jur = pivoted_period.shape
    date_range_str = f"{pivoted_period.index.min().strftime('%Y-%m-%d')} to {pivoted_period.index.max().strftime('%Y-%m-%d')}"
    
    # Verification check: p == 51 and training end 2019-06-29 present
    has_train_end = pd.Timestamp("2019-06-29") in pivoted_period.index
    pass_d1 = (n_jur == 51) and has_train_end
    check_str = f"p={n_jur} (expected 51), train_end 2019-06-29 in index: {has_train_end}"
    status_str = "PASS" if pass_d1 else "FAIL"
    
    record_provenance("D1", f"{dataset_title} ({dataset_id_cdc})", url, clean_path, n_weeks, n_jur, date_range_str, check_str, status_str)
    
    if not pass_d1:
        print(f"ERROR: D1 failed validation: {check_str}")
        sys.exit(1)

# ==============================================================================
# D2: FRED Daily Closes (SP500, DJIA, NASDAQCOM)
# ==============================================================================
def fetch_d2():
    print("\n--- Fetching D2: FRED Stock Indices ---")
    series_names = ["SP500", "DJIA", "NASDAQCOM"]
    series_dfs = []
    
    for s in series_names:
        s_csv = RAW_DIR / f"fred_{s}.csv"
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}"
        if not s_csv.exists():
            print(f"Downloading FRED {s}...")
            subprocess.run(["curl.exe", "-s", url, "-o", str(s_csv)], check=True)
        df_s = pd.read_csv(s_csv)
        df_s.columns = ["date", s]
        df_s[s] = pd.to_numeric(df_s[s], errors="coerce")
        series_dfs.append(df_s)
        
    merged = series_dfs[0]
    for df_s in series_dfs[1:]:
        merged = pd.merge(merged, df_s, on="date", how="outer")
        
    merged["date"] = pd.to_datetime(merged["date"])
    sub = merged[(merged["date"] >= "2021-01-01") & (merged["date"] <= "2023-01-19")]
    sub_clean = sub.dropna().sort_values("date").reset_index(drop=True)
    
    clean_path = RAW_DIR / "fred_indices_clean.csv"
    sub_clean.to_csv(clean_path, index=False)
    
    n_days = len(sub_clean)
    p_indices = 3
    date_range_str = f"{sub_clean['date'].min().strftime('%Y-%m-%d')} to {sub_clean['date'].max().strftime('%Y-%m-%d')}"
    
    pass_d2 = (n_days == 515) and (p_indices == 3)
    check_str = f"n={n_days} (expected 515), p={p_indices} (expected 3)"
    status_str = "PASS" if pass_d2 else "FAIL"
    
    record_provenance("D2", "FRED Daily Closes (SP500, DJIA, NASDAQCOM)", "https://fred.stlouisfed.org/graph/fredgraph.csv", clean_path, n_days, p_indices, date_range_str, check_str, status_str)
    
    if not pass_d2:
        print(f"ERROR: D2 failed validation: {check_str}")
        sys.exit(1)

# ==============================================================================
# D3: HASC 2011 Challenge (Person 101, Walk & Stay)
# ==============================================================================
def fetch_d3():
    print("\n--- Fetching D3: HASC 2011 Challenge Person 101 ---")
    url = "http://hasc.jp/hc2012/sampleData/SampleData20121101.zip"
    zip_path = RAW_DIR / "SampleData20121101.zip"
    hasc_dir = RAW_DIR / "hasc_person101"
    hasc_dir.mkdir(parents=True, exist_ok=True)
    
    if not zip_path.exists():
        print("Downloading HASC SampleData20121101.zip...")
        subprocess.run(["curl.exe", "-s", url, "-o", str(zip_path)], check=True)
        
    # Extract the 10 walk and 10 stay csv files for person101
    walk_counts = []
    stay_counts = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for fname in sorted(z.namelist()):
            if "2_walk/person101/HASC" in fname and fname.endswith(".csv"):
                out_file = hasc_dir / f"walk_{Path(fname).name}"
                if not out_file.exists():
                    with z.open(fname) as src, open(out_file, "wb") as dst:
                        dst.write(src.read())
                df = pd.read_csv(out_file, header=None)
                walk_counts.append(len(df))
            elif "1_stay/person101/HASC" in fname and fname.endswith(".csv"):
                out_file = hasc_dir / f"stay_{Path(fname).name}"
                if not out_file.exists():
                    with z.open(fname) as src, open(out_file, "wb") as dst:
                        dst.write(src.read())
                df = pd.read_csv(out_file, header=None)
                stay_counts.append(len(df))
                
    n_datasets = len(walk_counts)
    min_walk = min(walk_counts) if walk_counts else 0
    min_stay = min(stay_counts) if stay_counts else 0
    
    # Requirement: 10 datasets with at least 1596 + 100 walking and 100 staying samples per dataset
    pass_d3 = (n_datasets == 10) and (min_walk >= 1696) and (min_stay >= 100)
    check_str = f"10 pairs: min_walk={min_walk} (>=1696), min_stay={min_stay} (>=100)"
    status_str = "PASS" if pass_d3 else "FAIL"
    
    manifest = {
        "n_pairs": n_datasets,
        "walk_sample_counts": walk_counts,
        "stay_sample_counts": stay_counts,
        "p": 3
    }
    manifest_path = hasc_dir / "hasc_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    record_provenance("D3", "HASC 2011 Challenge Subject 101 Walk & Stay", url, manifest_path, f"10x({min_walk}+{min_stay})", 3, "2011 HASC Corpus", check_str, status_str)
    
    if not pass_d3:
        print(f"ERROR: D3 failed validation: {check_str}")
        sys.exit(1)

# ==============================================================================
# D4: MNIST (via OpenML mnist_784)
# ==============================================================================
def fetch_d4():
    print("\n--- Fetching D4: MNIST ---")
    d4_npz = RAW_DIR / "mnist_784.npz"
    
    if not d4_npz.exists():
        print("Fetching mnist_784 from OpenML...")
        mnist = sklearn.datasets.fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = mnist.data.astype(np.float32)
        y = mnist.target.astype(np.int64)
        np.savez_compressed(d4_npz, X=X, y=y)
    else:
        loaded = np.load(d4_npz)
        X = loaded["X"]
        y = loaded["y"]
        
    n_samples, p_pixels = X.shape
    pass_d4 = (p_pixels == 784) and (n_samples == 70000)
    check_str = f"pixels={p_pixels} (expected 784), total_samples={n_samples} (expected 70000)"
    status_str = "PASS" if pass_d4 else "FAIL"
    
    record_provenance("D4", "MNIST 784 OpenML Dataset", "https://www.openml.org/d/554", d4_npz, n_samples, p_pixels, "N/A (Image classification)", check_str, status_str)
    
    if not pass_d4:
        print(f"ERROR: D4 failed validation: {check_str}")
        sys.exit(1)

def main():
    fetch_d1()
    fetch_d2()
    fetch_d3()
    fetch_d4()
    
    # Write provenance CSV
    with open(PROVENANCE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset_id", "title", "source_url", "local_path", "retrieval_timestamp",
            "sha256", "n", "p", "date_range", "summary_statistic_check", "check_status"
        ])
        writer.writeheader()
        writer.writerows(provenance_records)
        
    print(f"\nAll datasets verified successfully! Provenance saved to {PROVENANCE_CSV}")

if __name__ == "__main__":
    main()

