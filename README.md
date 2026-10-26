# OCD-Mix: Online Change-Point Detection for Dependent Time Series

OCD-Mix is an online, multi-chart change-point detector and inference framework for high-dimensional ($p$-dimensional) time series exhibiting temporal dependence ($\alpha$-mixing VAR(1) dynamics). It extends the online change-point detection (OCD) idea of Chen, Wang, and Samworth (2020) by standardising window sums with long-run variance (LRV) estimates and calibrating detection thresholds for dependent observations.

---

## 1. Repository Layout

```
.
├── generate_synthetic_streams.py        # Original synthetic data generator (untouched)
├── generate_synthetic_streams_fast.py   # Cholesky-accelerated fast generator
├── pyproject.toml                       # Package configuration and pytest settings
├── README.md                            # Complete execution instructions
├── src/
│   └── ocdmix/
│       ├── __init__.py                  # Package init
│       ├── io.py                        # Memory-mapped I/O (blind to ground truth)
│       ├── fit.py                       # Training fit: mean, AR(1), LRV, s2 tables (E1-E6)
│       ├── scan.py                      # Vectorised batch scan and alarm detection (E7-E9)
│       ├── online.py                    # Constant-memory, constant-time OnlineDetector
│       ├── calibrate.py                 # Threshold calibration and self-calibration (E10)
│       ├── localize.py                  # Change-point estimation, CI, and support (E11-E14)
│       └── metrics.py                   # Evaluation against evaluator-only ground truth
├── scripts/
│   ├── generate_all.py                  # Parallel synthetic dataset generator
│   ├── run_calibration.py               # E10 threshold calibration CLI
│   ├── calibrate_ci.py                  # E13 CI half-width multiplier calibration CLI
│   ├── run_eval.py                      # Single dataset evaluation CLI
│   ├── run_all_evals.py                 # Full 24-experiment evaluation runner
│   ├── bench_online.py                  # Streaming throughput and memory benchmark
│   ├── run_self_calibration.py          # Task 9 self-calibration comparison
│   └── make_report.py                   # Tables and 300-dpi figures generator
├── tests/
│   ├── test_fast_generator.py           # Task 0 verification
│   ├── test_fit.py                      # T1: LRV accuracy across phi
│   ├── test_scan.py                     # T2: standardisation variance; T3: window sums
│   ├── test_online.py                   # T4: batch vs online equivalence; T5: constant memory
│   ├── test_calibrate.py                # T6: reduction to independence (phi=0)
│   ├── test_power.py                    # T7: power sanity under strong effect
│   ├── test_blindness.py                # T8: detector blindness to ground truth / manifest
│   └── test_reproducibility.py          # T9: byte-identical CSV outputs
└── results/
    ├── calib/                           # Calibrated threshold JSON files
    ├── eval/                            # Per-stream evaluation CSVs and JSON summaries
    ├── tables/                          # Summary metrics and benchmark CSV tables
    ├── figures/                         # Publication-quality 300-dpi PNG plots
    └── REPORT.md                        # Comprehensive research report
```

---

## 2. Installation and Setup

Requires Python 3.10+, NumPy, SciPy, pandas, matplotlib, pytest.

```bash
# Clone and enter directory
cd /path/to/OCD_CI_extended

# Install dependencies and package in editable mode
pip install -e .
```

---

## 3. Running Acceptance Tests

Run the complete test suite (all tests T0–T9):

```bash
pytest -v
```

To run individual test gates:

```bash
# Task 0: Fast generator marginal and long-run variance checks
pytest tests/test_fast_generator.py -v

# T1: LRV estimation accuracy (<5% bias for prewhiten)
pytest tests/test_fit.py -v

# T2 & T3: Window sums and standardisation variance in [0.93, 1.10]
pytest tests/test_scan.py -v

# T4 & T5: Batch vs online equivalence and constant memory (<5% growth)
pytest tests/test_online.py -v

# T6: Reduction to independence (proposed vs vanilla within 10% at phi=0)
pytest tests/test_calibrate.py -v

# T7: Power sanity (detection >= 99%, median delay < 60)
pytest tests/test_power.py -v

# T8: Detector blindness to ground truth / manifest files
pytest tests/test_blindness.py -v

# T9: Reproducibility (byte-identical CSV outputs)
pytest tests/test_reproducibility.py -v
```

---

## 4. End-to-End Experiment Pipeline

### Step 1: Generate All Synthetic Datasets
Generates all 17 datasets ($T=5000, p=50$) across $\phi \in \{0.0, 0.3, 0.6, 0.8\}$, null/changed conditions, and `cicalib` in parallel:

```bash
python scripts/generate_all.py --workers 6
```

*(For quick smoke-testing with 10 streams per scenario, add the `--smoke` flag).*

---

### Step 2: Calibrate Detection Thresholds

#### A. Proposed Method (per $\phi$ from `calib_null_phi{tag}`):
```bash
python scripts/run_calibration.py --null-dir data/calib_null_phi00 --method proposed --out results/calib/proposed_phi00.json
python scripts/run_calibration.py --null-dir data/calib_null_phi03 --method proposed --out results/calib/proposed_phi03.json
python scripts/run_calibration.py --null-dir data/calib_null_phi06 --method proposed --out results/calib/proposed_phi06.json
python scripts/run_calibration.py --null-dir data/calib_null_phi08 --method proposed --out results/calib/proposed_phi08.json
```

#### B. Vanilla Baseline (calibrated ONLY on i.i.d. $\phi=0.0$ data):
```bash
python scripts/run_calibration.py --null-dir data/calib_null_phi00 --method vanilla --out results/calib/vanilla_iid.json
```

---

### Step 3: Calibrate Confidence Interval Multiplier $a$
Calibrates parameter $a$ to achieve 95% empirical coverage on `cicalib`:

```bash
python scripts/calibrate_ci.py --dataset-dir data/cicalib --calib-json results/calib/proposed_phi06.json --out results/calib/ci_a.json
```

---

### Step 4: Run Evaluation Experiments
Evaluates both proposed and vanilla detectors across all null and changed streams:

```bash
# Run all 24 evaluations in parallel
python scripts/run_all_evals.py --workers 6
```

*(Alternatively, run individual evaluations via `python scripts/run_eval.py --dataset-dir data/<NAME> --calib-json results/calib/<CALIB>.json --ci-json results/calib/ci_a.json --out-csv results/eval/<OUT>.csv --out-json results/eval/<OUT>.json`).*

---

### Step 5: Streaming Throughput and Constant Memory Benchmark
Benchmarks `OnlineDetector` on dimensions $p \in \{50, 100, 200, 400\}$ over 200,000 steps:

```bash
python scripts/bench_online.py --steps 200000
```

---

### Step 6: Task 9 Self-Calibration Option
Fits per-coordinate AR(1) coefficients and innovation covariance from training data, simulates 300 null streams, and compares calibrated thresholds with oracle thresholds:

```bash
python scripts/run_self_calibration.py
```

---

### Step 7: Generate Report Figures and Tables
Generates all 5 publication-ready 300-dpi figures and summary CSV tables:

```bash
python scripts/make_report.py
```

Outputs:
- Figures (`results/figures/`):
  1. `fig1_false_alarm_vs_phi.png`
  2. `fig2_mean_delay_vs_phi.png`
  3. `fig3_ci_coverage_vs_phi.png`
  4. `fig4_support_jaccard_vs_phi.png`
  5. `fig5_memory_vs_stream_length.png`
- Tables (`results/tables/`):
  - `null_metrics.csv`
  - `change_metrics.csv`
  - `all_eval_summaries.csv`
  - `benchmark_online.csv`
  - `self_calibration_comparison.csv`
- Full text report: `results/REPORT.md`
