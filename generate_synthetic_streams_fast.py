#!/usr/bin/env python3
"""Reproducible synthetic multivariate time-series generator for online CPD.

FAST VERSION — identical CLI, file formats and ground-truth fields.
Only generate_one_stream is changed: Cholesky-factored innovations
and bulk standard-normal draws replace per-step multivariate_normal calls.

Model
-----
    Z_t = A Z_{t-1} + eps_t,     eps_t ~ N(0, Sigma_eps,r)
    X_t = mu_t + Z_t

The regime r changes at a known-to-the-evaluator index b, but b and the
affected coordinates are written to a separate ground-truth file and are not
stored in observations.npy.

For the isotropic VAR(1) baseline A = phi * I. For the dense robustness model,
A is a random dense matrix scaled so ||A||_2 = phi < 1. Thus the state dynamics
are contractive and, under the usual nonsingular Gaussian-innovation
conditions, the process has geometric alpha-mixing. The parameter phi is the
base of a conservative geometric decay envelope (alpha(k) <= C * phi**k); it
is not claimed to equal the exact alpha-mixing coefficient.

Dependencies: numpy, scipy
    python -m pip install numpy scipy

Example
-------
python generate_synthetic_streams_fast.py \
  --outdir data/mean_phi06 --T 5000 --p 50 --n-streams 100 \
  --phi 0.6 --var-structure isotropic --change-type mean \
  --cp-fraction 0.5 --sparsity 0.1 --effect 0.5 --seed 20261014

Use --change-type none to create null streams for false-alarm / ARL evaluation.
Use --change-type covariance for an innovation-covariance change.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

SCRIPT_VERSION = "1.0.0-fast"


def _jsonable(value: Any) -> Any:
    """Convert common NumPy values to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(obj), indent=2) + "\n", encoding="utf-8")


def make_var_matrix(p: int, phi: float, structure: str, rng: np.random.Generator) -> np.ndarray:
    """Build a stable VAR(1) matrix whose spectral norm is at most phi."""
    if structure == "isotropic":
        return phi * np.eye(p, dtype=np.float64)

    if structure == "dense":
        raw = rng.normal(size=(p, p)) / math.sqrt(p)
        spectral_norm = float(np.linalg.norm(raw, ord=2))
        if spectral_norm == 0.0:  # Practically impossible, but keeps the function total.
            return np.zeros((p, p), dtype=np.float64)
        return (phi / spectral_norm) * raw

    raise ValueError(f"Unsupported VAR structure: {structure}")


def make_innovation_covariance(p: int, cross_corr: float) -> np.ndarray:
    """Positive-definite Toeplitz covariance, Sigma[i,j] = cross_corr**|i-j|."""
    idx = np.arange(p)
    return np.power(cross_corr, np.abs(idx[:, None] - idx[None, :]), dtype=np.float64)


def make_post_change_covariance(
    sigma_pre: np.ndarray,
    support: np.ndarray,
    covariance_multiplier: float,
) -> np.ndarray:
    """Scale selected innovation standard deviations; preserves positive definiteness.

    For each selected coordinate j, its innovation variance is multiplied by
    covariance_multiplier. Cross-covariances touching that coordinate are
    scaled accordingly, preserving the innovation correlation matrix.
    """
    scale = np.ones(sigma_pre.shape[0], dtype=np.float64)
    scale[support] = math.sqrt(covariance_multiplier)
    sigma_post = scale[:, None] * sigma_pre * scale[None, :]
    return (sigma_post + sigma_post.T) * 0.5


def _stationary_covariance(A: np.ndarray, sigma: np.ndarray, phi: float) -> np.ndarray:
    """Solve Gamma = A Gamma A.T + Sigma for the stationary VAR covariance."""
    if phi == 0.0:
        return sigma.copy()
    if np.array_equal(A, phi * np.eye(A.shape[0])):
        return sigma / (1.0 - phi * phi)
    gamma = solve_discrete_lyapunov(A, sigma)
    gamma = (gamma + gamma.T) * 0.5
    # Guard against a pathological numerical failure before simulation starts.
    eig_min = float(np.linalg.eigvalsh(gamma).min())
    if not np.isfinite(eig_min) or eig_min <= 0:
        raise RuntimeError(
            "Stationary covariance was not numerically positive definite; "
            "try a smaller phi or a lower-dimensional dense VAR."
        )
    return gamma


def generate_one_stream(
    *,
    T: int,
    p: int,
    A: np.ndarray,
    phi: float,
    sigma_pre: np.ndarray,
    gamma_pre: np.ndarray,
    rng: np.random.Generator,
    change_type: str,
    cp_fraction: float,
    cp_index: int | None,
    sparsity: float,
    effect: float,
    covariance_multiplier: float,
    support_mode: str = "random",
    fixed_support: list[int] | None = None,
    sign_mode: str = "random",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate one stream plus evaluator-only ground truth (FAST version).

    Uses Cholesky factorisation and bulk standard-normal draws instead of
    per-step multivariate_normal calls.

    Index convention: b is zero-based and X[b] is the first post-change
    observation.
    """
    if T < 2:
        raise ValueError("T must be >= 2")
    if p < 1:
        raise ValueError("p must be >= 1")
    if change_type not in {"none", "mean", "covariance"}:
        raise ValueError("change_type must be 'none', 'mean', or 'covariance'")
    if not 0.0 < cp_fraction < 1.0:
        raise ValueError("cp_fraction must be strictly between 0 and 1")
    if not 0.0 < sparsity <= 1.0:
        raise ValueError("sparsity must be in (0, 1]")
    if effect <= 0:
        raise ValueError("effect must be > 0")
    if change_type == "covariance" and (covariance_multiplier <= 0 or np.isclose(covariance_multiplier, 1.0)):
        raise ValueError("For covariance changes, covariance_multiplier must be positive and different from 1")

    if change_type == "none":
        b = None
    elif cp_index is not None:
        if not 1 <= cp_index < T:
            raise ValueError(f"cp_index must be in [1, {T - 1}]")
        b = int(cp_index)
    else:
        b = min(T - 1, max(1, int(round(cp_fraction * T))))

    if change_type == "none":
        support = np.empty(0, dtype=int)
    elif support_mode == "fixed":
        if not fixed_support:
            raise ValueError("support_mode='fixed' requires --support, e.g. --support 0,3,5")
        support = np.asarray(sorted(set(fixed_support)), dtype=int)
        if support.size == 0 or np.any(support < 0) or np.any(support >= p):
            raise ValueError(f"Fixed support must contain indices in [0, {p - 1}]")
    else:
        k = min(p, max(1, int(math.ceil(sparsity * p))))
        support = np.sort(rng.choice(p, size=k, replace=False))

    mu_pre = np.zeros(p, dtype=np.float64)
    mu_post = np.zeros(p, dtype=np.float64)
    sigma_post = sigma_pre.copy()
    shift = np.zeros(p, dtype=np.float64)

    if change_type == "mean":
        if sign_mode == "positive":
            signs = np.ones(support.size)
        elif sign_mode == "alternating":
            signs = np.where(np.arange(support.size) % 2 == 0, 1.0, -1.0)
        elif sign_mode == "random":
            signs = rng.choice(np.array([-1.0, 1.0]), size=support.size)
        else:
            raise ValueError("sign_mode must be positive, alternating, or random")
        stationary_sd = np.sqrt(np.diag(gamma_pre))
        shift[support] = signs * effect * stationary_sd[support]
        mu_post = mu_pre + shift

    elif change_type == "covariance":
        sigma_post = make_post_change_covariance(
            sigma_pre, support, covariance_multiplier
        )

    # --- FAST PATH: Cholesky + bulk draws ---
    # Precompute Cholesky factors
    chol_gamma_pre = np.linalg.cholesky(gamma_pre)
    chol_sigma_pre = np.linalg.cholesky(sigma_pre)

    if change_type == "covariance" and b is not None:
        chol_sigma_post = np.linalg.cholesky(sigma_post)
    else:
        chol_sigma_post = chol_sigma_pre

    # Draw initial state from stationary distribution
    z0_std = rng.standard_normal(p)

    # Draw ALL standard normals for innovations up front
    Zs = rng.standard_normal((T, p))

    # Compute innovations: eps = Zs @ chol.T
    eps = np.empty((T, p), dtype=np.float64)
    if change_type == "covariance" and b is not None:
        # Pre-change innovations
        if b > 0:
            eps[:b] = Zs[:b] @ chol_sigma_pre.T
        # Post-change innovations
        eps[b:] = Zs[b:] @ chol_sigma_post.T
    else:
        eps[:] = Zs @ chol_sigma_pre.T

    # Initial state from stationary distribution
    z = np.empty((T, p), dtype=np.float64)
    z[0] = chol_gamma_pre @ z0_std

    # VAR(1) recursion (cannot be vectorised due to sequential dependency)
    for t in range(1, T):
        z[t] = A @ z[t - 1] + eps[t]

    # Build observations X = mu_t + Z
    x = np.empty((T, p), dtype=np.float64)
    if change_type == "mean" and b is not None:
        x[:b] = mu_pre + z[:b]
        x[b:] = mu_post + z[b:]
    else:
        x[:] = mu_pre + z

    truth = {
        "change_type": change_type,
        "changepoint_index_zero_based": b,
        "changepoint_convention": "b is the first post-change observation; X[b] is post-change",
        "affected_coordinates_zero_based": support.tolist(),
        "support_definition": (
            "Coordinates with a directly shifted mean" if change_type == "mean" else
            "Coordinates whose innovation standard deviations are directly scaled" if change_type == "covariance" else
            "No change; empty support"
        ),
        "mean_shift_vector": shift.tolist() if change_type == "mean" else None,
        "covariance_multiplier_on_selected_innovation_variances": (
            covariance_multiplier if change_type == "covariance" else None
        ),
        "phi": phi,
    }
    return x, truth


def generate_dataset(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.T < 2:
        raise ValueError("--T must be >= 2")
    if args.p < 1 or args.n_streams < 1:
        raise ValueError("--p and --n-streams must be positive")
    if not (0.0 <= args.phi < 1.0):
        raise ValueError("--phi must satisfy 0 <= phi < 1")
    if not (-1.0 < args.cross_corr < 1.0):
        raise ValueError("--cross-corr must be strictly between -1 and 1")
    if args.dtype not in {"float32", "float64"}:
        raise ValueError("--dtype must be float32 or float64")

    if args.change_type != "none" and args.cp_index is not None and not 1 <= args.cp_index < args.T:
        raise ValueError(f"--cp-index must be in [1, {args.T - 1}]")
    if args.change_type != "none" and args.support_mode == "fixed" and not args.support:
        raise ValueError("--support is required when --support-mode=fixed for changed streams")

    fixed_support = None
    if args.support:
        try:
            fixed_support = [int(v.strip()) for v in args.support.split(",") if v.strip()]
        except ValueError as exc:
            raise ValueError("--support must be comma-separated integer indices, e.g. 0,3,5") from exc
        if len(set(fixed_support)) != len(fixed_support):
            raise ValueError("--support cannot contain duplicate coordinate indices")
        if any(v < 0 or v >= args.p for v in fixed_support):
            raise ValueError(f"--support coordinates must be in [0, {args.p - 1}]")

    # Separate, deterministic RNG streams for the shared dynamics and each sample.
    model_rng = np.random.default_rng(np.random.SeedSequence([args.seed, args.scenario_id, 9173]))
    A = make_var_matrix(args.p, args.phi, args.var_structure, model_rng)
    sigma_pre = make_innovation_covariance(args.p, args.cross_corr)
    gamma_pre = _stationary_covariance(A, sigma_pre, args.phi)
    sigma_post_fixed = None
    if args.change_type == "covariance" and args.support_mode == "fixed":
        if not fixed_support:
            raise ValueError("--support is required when --support-mode=fixed")
        preliminary_support = np.asarray(sorted(set(fixed_support)), dtype=int)
        if preliminary_support.size == 0 or np.any(preliminary_support < 0) or np.any(preliminary_support >= args.p):
            raise ValueError(f"Fixed support must contain indices in [0, {args.p - 1}]")
        sigma_post_fixed = make_post_change_covariance(
            sigma_pre, preliminary_support, args.covariance_multiplier
        )

    obs_path = outdir / "observations.npy"
    truth_path = outdir / "ground_truth.jsonl"
    dtype = np.dtype(args.dtype)
    X_mm = np.lib.format.open_memmap(
        obs_path, mode="w+", dtype=dtype, shape=(args.n_streams, args.T, args.p)
    )

    with truth_path.open("w", encoding="utf-8") as truth_file:
        for rep in range(args.n_streams):
            stream_seed = np.random.SeedSequence([args.seed, args.scenario_id, rep, 2718])
            rng = np.random.default_rng(stream_seed)
            X, truth = generate_one_stream(
                T=args.T,
                p=args.p,
                A=A,
                phi=args.phi,
                sigma_pre=sigma_pre,
                gamma_pre=gamma_pre,
                rng=rng,
                change_type=args.change_type,
                cp_fraction=args.cp_fraction,
                cp_index=args.cp_index,
                sparsity=args.sparsity,
                effect=args.effect,
                covariance_multiplier=args.covariance_multiplier,
                support_mode=args.support_mode,
                fixed_support=fixed_support,
                sign_mode=args.sign_mode,
            )
            X_mm[rep] = X.astype(dtype, copy=False)
            row = {
                "stream_id": rep,
                "master_seed": args.seed,
                "scenario_id": args.scenario_id,
                "replication_id": rep,
                "seed_sequence_entropy": _jsonable(stream_seed.entropy),
                "seed_sequence_spawn_key": list(stream_seed.spawn_key),
                **truth,
            }
            truth_file.write(json.dumps(_jsonable(row)) + "\n")
            if (rep + 1) % max(1, min(100, args.n_streams // 10 or 1)) == 0 or rep + 1 == args.n_streams:
                print(f"Generated {rep + 1}/{args.n_streams} streams", flush=True)

    X_mm.flush()
    del X_mm

    # Save fixed model matrices to make the exact generating system auditable.
    np.save(outdir / "var_matrix_A.npy", A)
    np.save(outdir / "innovation_covariance_pre.npy", sigma_pre)
    np.save(outdir / "stationary_covariance_pre.npy", gamma_pre)
    if sigma_post_fixed is not None:
        np.save(outdir / "innovation_covariance_post_fixed_support.npy", sigma_post_fixed)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "description": "Synthetic VAR(1) streams for online changepoint detection and inference.",
        "observations_file": "observations.npy",
        "observations_shape": [args.n_streams, args.T, args.p],
        "observations_axis_order": ["stream", "time", "coordinate"],
        "observations_dtype": args.dtype,
        "ground_truth_file": "ground_truth.jsonl",
        "warning": "Do not provide ground_truth.jsonl or design metadata to the online detector; these contain hidden evaluation truth.",
        "master_seed": args.seed,
        "scenario_id": args.scenario_id,
        "n_streams": args.n_streams,
        "T": args.T,
        "p": args.p,
        "phi": args.phi,
        "var_structure": args.var_structure,
        "spectral_norm_A": float(np.linalg.norm(A, ord=2)),
        "mixing_note": (
            "For this contractive Gaussian VAR, phi is the geometric envelope base q=||A||_2; "
            "under standard nonsingularity conditions alpha(k) <= C*q^k for a suitable constant C. "
            "This is an upper-bound rate parameter, not the exact alpha-mixing coefficient."
        ),
        "cross_corr_innovation": args.cross_corr,
        "change_type": args.change_type,
        "changepoint_location_policy": "Exact changepoint indices are stored only in ground_truth.jsonl; do not expose that file or generation arguments to the detector.",
        "sparsity_fraction": args.sparsity,
        "standardized_mean_effect": args.effect if args.change_type == "mean" else None,
        "covariance_multiplier": args.covariance_multiplier if args.change_type == "covariance" else None,
        "support_mode": args.support_mode,
        "fixed_support_zero_based": fixed_support,
        "mean_shift_sign_mode": args.sign_mode if args.change_type == "mean" else None,
        "initialization": "Z[0] is sampled from the pre-change stationary Gaussian covariance.",
        "model_equations": [
            "Z_t = A Z_{t-1} + epsilon_t",
            "X_t = mu_t + Z_t",
            "epsilon_t ~ N(0, Sigma_epsilon,regime)",
        ],
        "seed_scheme": "model: SeedSequence([master_seed, scenario_id, 9173]); stream r: SeedSequence([master_seed, scenario_id, r, 2718]).",
    }
    _write_json(outdir / "manifest.json", manifest)

    readme = f"""Synthetic VAR(1) dataset
========================

Files
- observations.npy: numeric array with shape {tuple([args.n_streams, args.T, args.p])}, ordered (stream, time, coordinate).
- ground_truth.jsonl: one evaluator-only JSON record per stream; do NOT give this to the detector.
- manifest.json: generator settings and model specification; keep separate from detector input.
- var_matrix_A.npy: the VAR(1) coefficient matrix.
- innovation_covariance_pre.npy: pre-change innovation covariance.
- stationary_covariance_pre.npy: pre-change stationary covariance.
"""
    (outdir / "README.txt").write_text(readme, encoding="utf-8")
    print(f"\nDone. Dataset written to: {outdir}")
    print(f"Observations: {obs_path}")
    print(f"Ground truth (evaluator only): {truth_path}")
    print(f"Manifest: {outdir / 'manifest.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate reproducible synthetic multivariate VAR(1) streams for changepoint research (FAST)."
    )
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for this one scenario.")
    parser.add_argument("--T", type=int, default=5000, help="Length of each stream (default: 5000).")
    parser.add_argument("--p", type=int, default=50, help="Number of coordinates (default: 50).")
    parser.add_argument("--n-streams", type=int, default=100, help="Number of independent streams (default: 100).")
    parser.add_argument("--phi", type=float, default=0.6, help="Dependence envelope base; must lie in [0,1).")
    parser.add_argument(
        "--var-structure", choices=["isotropic", "dense"], default="isotropic",
        help="isotropic: A=phi*I baseline; dense: random cross-lag VAR matrix with spectral norm phi."
    )
    parser.add_argument(
        "--change-type", choices=["none", "mean", "covariance"], default="mean",
        help="none for null streams; mean for sparse mean shift; covariance for innovation-covariance change."
    )
    parser.add_argument("--cp-fraction", type=float, default=0.5, help="Change index as fraction of T for changed streams (ignored if --cp-index is set).")
    parser.add_argument("--cp-index", type=int, default=None, help="Exact zero-based changepoint index b; X[b] is first post-change observation.")
    parser.add_argument("--sparsity", type=float, default=0.1, help="Affected-coordinate fraction (default: 0.1).")
    parser.add_argument("--effect", type=float, default=0.5, help="Mean shift in pre-change marginal standard deviations.")
    parser.add_argument(
        "--covariance-multiplier", type=float, default=1.5,
        help="Multiplier on selected innovation variances; positive and not 1 (e.g. 0.5 or 1.5)."
    )
    parser.add_argument("--cross-corr", type=float, default=0.25, help="Innovation covariance Toeplitz correlation parameter in (-1,1).")
    parser.add_argument("--support-mode", choices=["random", "fixed"], default="random", help="Random support per stream or fixed --support coordinates.")
    parser.add_argument("--support", type=str, default=None, help="Fixed zero-based coordinates, comma-separated; e.g. 0,3,5.")
    parser.add_argument("--sign-mode", choices=["positive", "alternating", "random"], default="random", help="Sign pattern for nonzero mean shifts.")
    parser.add_argument("--seed", type=int, default=20261014, help="Master seed for reproducible streams.")
    parser.add_argument("--scenario-id", type=int, default=0, help="Integer ID used to derive independent random streams across scenarios.")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32", help="Stored observations dtype; computations use float64.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        generate_dataset(args)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
