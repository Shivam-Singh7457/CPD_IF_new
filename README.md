# OCD-Mix: Online Change-Point Detection for Temporally Dependent Multivariate Time Series

This repository contains the reference implementation and experimental evaluation suite for **OCD-Mix**, a multi-chart online change-point detection and post-alarm inference framework for high-dimensional ($p$-variate) time series under temporal dependence ($\alpha$-mixing VAR dynamics).

OCD-Mix extends the multi-chart sequential detection framework of Chen, Wang, and Samworth (2020, 2024) by:
1. Standardising dyadic window cumulative sums with prewhitened long-run variance (LRV) estimators to prevent false alarm inflation under autocorrelation.
2. Calibrating simultaneous detection thresholds $(\tau_A, \tau_Q)$ under dependent null distributions.
3. Providing finite-sample confidence intervals for the change-point location and coordinate-level support recovery.
4. Implementing a constant-time $O(p \cdot |K|)$ per update, constant-memory $O(p \cdot k_{\max})$ streaming detector (`OnlineDetector`) with periodic re-anchoring against floating-point drift.

---

## Repository Structure

```
.
├── baselines/
│   ├── cws_ocd.py                       # Chen, Wang & Samworth (2020, 2024 JASA) multi-chart & CI
│   ├── mnsbs.py                         # Madrid Padilla et al. (NeurIPS 2023) seeded binary segmentation
│   └── rff_mmd.py                       # Streaming RFF-MMD (NeurIPS 2025), OKCUSUM, and NEWMA
├── src/
│   └── ocdmix/
│       ├── fit.py                       # Training estimators: mean, AR(1), long-run variance, finite-sample s2
│       ├── scan.py                      # Vectorised multi-scale batch scan and alarm evaluation
│       ├── online.py                    # Streaming OnlineDetector (ring buffer, constant memory)
│       ├── calibrate.py                 # Null calibration and autonomous self-calibration
│       ├── localize.py                  # Change-point estimation, confidence interval, and support recovery
│       ├── metrics.py                   # Evaluation metrics (FAR, delay, CI coverage, Jaccard)
│       └── io.py                        # Memory-mapped observation streaming and ground-truth loaders
├── scripts/
│   ├── generate_all.py                  # Parallel synthetic stream generation
│   ├── run_calibration.py               # Empirical threshold calibration CLI
│   ├── calibrate_ci.py                  # CI half-width parameter calibration CLI
│   ├── run_eval.py                      # Single dataset evaluation CLI
│   ├── run_all_evals.py                 # 24-experiment parallel benchmark runner
│   ├── bench_online.py                  # Real-time throughput and memory scaling benchmark
│   ├── run_self_calibration.py          # Stream-by-stream autonomous self-calibration
│   ├── run_cws_benchmark.py             # CWS baseline benchmark runner
│   ├── run_mp_benchmark.py              # MNSBS baseline benchmark runner
│   ├── run_rff_benchmark.py             # Kernel / RFF baseline benchmark runner
│   └── make_report.py                   # Generates publication figures and summary tables
├── tests/
│   ├── test_fast_generator.py           # Synthetic generator moments and spectral decay
│   ├── test_fit.py                      # Long-run variance estimator consistency
│   ├── test_scan.py                     # Multi-scale standardisation variance and window sum correctness
│   ├── test_online.py                   # Equivalence of batch and streaming detectors; constant memory
│   ├── test_calibrate.py                # Calibration reduction to independent limit (phi = 0)
│   ├── test_power.py                    # Detection power and delay bounds under significant shift
│   ├── test_blindness.py                # Asserts zero ground-truth leakage into detector modules
│   ├── test_reproducibility.py          # Deterministic reproducibility of outputs
│   └── test_localize_chen.py            # Localisation, scale-grid search, and confidence interval bounds
├── results/
│   ├── calib/                           # Calibrated threshold JSON files
│   ├── eval/                            # Stream-level evaluation logs and summaries
│   ├── tables/                          # Summary metrics and benchmark tables (CSV)
│   ├── figures/                         # Publication-quality figures (300 DPI)
│   ├── REPORT.md                        # Empirical results and experimental report
│   └── FINAL_SUMMARY.md                 # Verification summary and baseline comparison
├── generate_synthetic_streams_fast.py   # Cholesky-accelerated VAR(1) synthetic stream generator
└── pyproject.toml                       # Package build definition and test configuration
```

---

## Installation

### Prerequisites
- Python 3.10 or higher
- Standard scientific stack: NumPy, SciPy, pandas, scikit-learn, matplotlib, pytest

### Setup
```bash
# Clone the repository
git clone https://github.com/anonymous/OCD-Mix.git
cd OCD-Mix

# Install package in editable mode
pip install -e .
```

---

## Test Suite

Run the unit and validation tests:

```bash
pytest -v
```

To run targeted verification modules:

```bash
# Variance standardisation and LRV consistency
pytest tests/test_fit.py tests/test_scan.py -v

# Streaming detector equivalence and memory footprint
pytest tests/test_online.py -v

# Change-point localisation and CI coverage bounds
pytest tests/test_localize_chen.py -v

# Evaluator blindness and evaluation reproducibility
pytest tests/test_blindness.py tests/test_reproducibility.py -v
```

---

## Experimental Pipeline

### 1. Data Generation
Generate synthetic benchmark datasets across dependence levels $\phi \in \{0.0, 0.3, 0.6, 0.8\}$, null and changed conditions ($p = 50, T = 5000$):

```bash
python scripts/generate_all.py --workers 4
```

*(Pass `--smoke` to generate a lightweight set for quick smoke-testing).*

### 2. Threshold Calibration

#### Proposed Dependent Multi-Chart Calibration
```bash
python scripts/run_calibration.py --null-dir data/calib_null_phi00 --method proposed --out results/calib/proposed_phi00.json
python scripts/run_calibration.py --null-dir data/calib_null_phi03 --method proposed --out results/calib/proposed_phi03.json
python scripts/run_calibration.py --null-dir data/calib_null_phi06 --method proposed --out results/calib/proposed_phi06.json
python scripts/run_calibration.py --null-dir data/calib_null_phi08 --method proposed --out results/calib/proposed_phi08.json
```

#### Vanilla Baseline Calibration (Independent Null)
```bash
python scripts/run_calibration.py --null-dir data/calib_null_phi00 --method vanilla --out results/calib/vanilla_iid.json
```

### 3. Confidence Interval Calibration
Calibrate the confidence interval half-width parameter on the calibration partition to achieve nominal 95% coverage:

```bash
python scripts/calibrate_ci.py --dataset-dir data/cicalib --calib-json results/calib/proposed_phi06.json --out results/calib/ci_a.json
```

### 4. Running Evaluations

Run the full evaluation matrix (all $\phi$, null/changed regimes, vanilla vs proposed):

```bash
python scripts/run_all_evals.py --workers 4
```

To evaluate baselines:
```bash
python scripts/run_cws_benchmark.py
python scripts/run_mp_benchmark.py
python scripts/run_rff_benchmark.py
```

### 5. Autonomous Self-Calibration
Run stream-by-stream self-calibration using only each stream's in-control training window $X[0:m_0]$:

```bash
python scripts/run_self_calibration.py
```

### 6. Streaming Performance Benchmark
Measure execution throughput and verify flat memory scaling across dimensions $p \in \{50, 100, 200, 400\}$:

```bash
python scripts/bench_online.py --steps 50000
```

### 7. Figures and Tables Generation
Generate all summary CSV tables and publication figures (300 DPI):

```bash
python scripts/make_report.py
```

Generated outputs include:
- `results/figures/fig1_false_alarm_vs_phi.png`: Empirical false-alarm rate versus $\phi$
- `results/figures/fig2_mean_delay_vs_phi.png`: Detection delay across effect sizes
- `results/figures/fig3_ci_coverage_vs_phi.png`: 95% Confidence interval coverage
- `results/figures/fig4_support_jaccard_vs_phi.png`: Support recovery accuracy
- `results/figures/fig5_memory_vs_stream_length.png`: Streaming memory footprint scaling
- `results/tables/null_metrics.csv` & `results/tables/change_metrics.csv`: Tabulated evaluation metrics
