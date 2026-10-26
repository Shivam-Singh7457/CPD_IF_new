"""Tests for the fast generator (Task 0).

Checks:
  - phi=0.6, p=50, 50 null streams:
    * pooled marginal variance within 5% of 1/(1-phi^2)
    * mean per-coordinate long-run variance (Bartlett, L=60, T=5000) within 15% of 1/(1-phi)^2
"""
from __future__ import annotations

import sys
import os
import tempfile

import numpy as np
import pytest

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generate_synthetic_streams_fast import (
    generate_one_stream,
    make_var_matrix,
    make_innovation_covariance,
    _stationary_covariance,
)


def _generate_null_streams(
    n_streams: int, T: int, p: int, phi: float, seed: int = 42
) -> np.ndarray:
    """Generate n_streams null streams using the fast generator."""
    rng_model = np.random.default_rng(np.random.SeedSequence([seed, 0, 9173]))
    A = make_var_matrix(p, phi, "isotropic", rng_model)
    sigma_pre = make_innovation_covariance(p, 0.25)
    gamma_pre = _stationary_covariance(A, sigma_pre, phi)

    all_streams = np.empty((n_streams, T, p), dtype=np.float64)
    for i in range(n_streams):
        stream_seed = np.random.SeedSequence([seed, 0, i, 2718])
        rng = np.random.default_rng(stream_seed)
        X, _ = generate_one_stream(
            T=T, p=p, A=A, phi=phi,
            sigma_pre=sigma_pre, gamma_pre=gamma_pre,
            rng=rng, change_type="none",
            cp_fraction=0.5, cp_index=None,
            sparsity=0.1, effect=0.5,
            covariance_multiplier=1.5,
        )
        all_streams[i] = X
    return all_streams


def _bartlett_lrv_per_coord(X: np.ndarray, L: int = 60) -> np.ndarray:
    """Estimate long-run variance per coordinate using Bartlett kernel.

    Parameters
    ----------
    X : (T, p) centred data
    L : int, bandwidth

    Returns
    -------
    lrv : (p,) estimated long-run variance per coordinate
    """
    T, p = X.shape
    # Autocovariance
    gam = np.empty((L + 1, p), dtype=np.float64)
    for h in range(L + 1):
        gam[h] = np.sum(X[: T - h] * X[h:], axis=0) / T

    lrv = gam[0].copy()
    for h in range(1, L + 1):
        w = 1.0 - h / (L + 1)
        lrv += 2.0 * w * gam[h]
    return lrv


class TestFastGenerator:
    """Tests for Task 0 fast generator."""

    PHI = 0.6
    P = 50
    N_STREAMS = 50
    T = 5000

    @pytest.fixture(scope="class")
    def null_streams(self):
        """Generate null streams once for all tests in this class."""
        return _generate_null_streams(
            self.N_STREAMS, self.T, self.P, self.PHI, seed=42
        )

    def test_marginal_variance(self, null_streams: np.ndarray):
        """Pooled marginal variance within 5% of 1/(1-phi^2)."""
        phi = self.PHI
        expected = 1.0 / (1.0 - phi ** 2)  # 1.5625

        # Pool: mean over streams and coordinates of per-stream per-coord variance
        # Use all time points
        var_per_stream_coord = np.var(null_streams, axis=1)  # (n_streams, p)
        pooled_var = np.mean(var_per_stream_coord)

        rel_err = abs(pooled_var - expected) / expected
        print(f"\nMarginal variance: pooled={pooled_var:.4f}, expected={expected:.4f}, "
              f"rel_err={rel_err:.4f}")
        assert rel_err < 0.05, (
            f"Pooled marginal variance {pooled_var:.4f} is not within 5% of "
            f"expected {expected:.4f} (rel_err={rel_err:.4f})"
        )

    def test_long_run_variance(self, null_streams: np.ndarray):
        """Mean per-coord LRV (Bartlett, L=60) within 15% of 1/(1-phi)^2."""
        phi = self.PHI
        expected_lrv = 1.0 / (1.0 - phi) ** 2  # 6.25

        # For each stream, compute per-coordinate LRV, then average
        lrv_all = np.empty((self.N_STREAMS, self.P), dtype=np.float64)
        for i in range(self.N_STREAMS):
            Xc = null_streams[i] - null_streams[i].mean(axis=0)
            lrv_all[i] = _bartlett_lrv_per_coord(Xc, L=60)

        mean_lrv = np.mean(lrv_all)
        rel_err = abs(mean_lrv - expected_lrv) / expected_lrv
        print(f"\nLong-run variance: mean={mean_lrv:.4f}, expected={expected_lrv:.4f}, "
              f"rel_err={rel_err:.4f}")
        assert rel_err < 0.15, (
            f"Mean LRV {mean_lrv:.4f} is not within 15% of "
            f"expected {expected_lrv:.4f} (rel_err={rel_err:.4f})"
        )

    def test_cli_produces_output(self, tmp_path):
        """Smoke test: the CLI produces expected output files."""
        import subprocess
        outdir = str(tmp_path / "test_output")
        cmd = [
            sys.executable, "generate_synthetic_streams_fast.py",
            "--outdir", outdir,
            "--T", "100", "--p", "5", "--n-streams", "3",
            "--phi", "0.3", "--change-type", "none",
            "--seed", "123", "--scenario-id", "0",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=os.path.dirname(__file__) + "/..")
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert os.path.exists(os.path.join(outdir, "observations.npy"))
        assert os.path.exists(os.path.join(outdir, "ground_truth.jsonl"))
        assert os.path.exists(os.path.join(outdir, "manifest.json"))

        # Check shape
        obs = np.load(os.path.join(outdir, "observations.npy"))
        assert obs.shape == (3, 100, 5)
