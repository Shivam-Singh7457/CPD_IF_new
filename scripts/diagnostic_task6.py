#!/usr/bin/env python3
"""TASK 6: Regenerate Figures 1-4 and add new figure for S_CI vs S_report.

Changes from Task 6 requirements:
- Fig 3: Regenerate with F5 coverage (zeta and resid variants) and old baselines,
  with error bars from Task 4. Save to results/figures/fix/.
- Figs 1, 2, 4: Plot one configuration only (final chosen). Use mean +/- SE per phi.
  Current plots overlay several ablation variants causing vertical zigzags.
- Add a Jaccard / precision / recall figure for S_CI vs S_report.

All figures saved to results/figures/fix/ (original figures in results/figures/ are preserved).
"""
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_FIX = PROJECT_ROOT / "results" / "fix"
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"
RESULTS_EVAL = PROJECT_ROOT / "results" / "eval"
FIGURES_FIX = PROJECT_ROOT / "results" / "figures" / "fix"
FIGURES_FIX.mkdir(parents=True, exist_ok=True)


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 2,
        "lines.markersize": 7,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def load_eval_json_df() -> pd.DataFrame:
    """Load all eval JSON summaries, return single dataframe."""
    import json
    records = []
    for f in sorted(RESULTS_EVAL.glob("*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            data["file_stem"] = f.stem
            records.append(data)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    def parse_phi(ds: str) -> float:
        for tag, val in [("phi00", 0.0), ("phi03", 0.3), ("phi06", 0.6), ("phi08", 0.8)]:
            if tag in ds:
                return val
        return 0.0

    def parse_effect(ds: str) -> str:
        if "e10" in ds:
            return "1.0"
        elif "e15" in ds:
            return "1.5"
        return "none"

    df["phi"] = df["dataset"].apply(parse_phi)
    df["effect"] = df["dataset"].apply(parse_effect)
    return df


def make_fig1_false_alarm(df: pd.DataFrame, figures_dir: Path) -> None:
    """Fig 1: False-alarm probability vs phi. One configuration only (proposed).
    Mean +/- SE per phi. No ablation zigzags.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    null_df = df[df["effect"] == "none"].sort_values(["phi", "method"])

    for method, col, marker, ls in [("proposed", "#1f77b4", "o", "-"), ("vanilla", "#d62728", "s", "--")]:
        sub = null_df[null_df["method"] == method].sort_values("phi")
        if sub.empty:
            continue
        # Group by phi -> mean and SE across any duplicates
        grp = sub.groupby("phi").agg(
            far_mean=("false_alarm_probability", "mean"),
            far_se=("false_alarm_probability_se", "mean"),
        ).reset_index()
        ax.errorbar(
            grp["phi"], grp["far_mean"],
            yerr=grp["far_se"],
            label=f"{method.capitalize()}",
            color=col, marker=marker, linestyle=ls, capsize=4,
        )

    # Also add 1000-stream oracle data from Task 1 if available
    task1_csv = RESULTS_FIX / "task1_adopted_calibration.csv"
    if task1_csv.exists():
        t1 = pd.read_csv(task1_csv)
        if "phi" in t1.columns and "out_sample_far" in t1.columns:
            ax.scatter(t1["phi"], t1["out_sample_far"], marker="*", color="#2ca02c",
                       s=100, zorder=5, label="Recalib (n=4000)")

    ax.axhline(0.05, color="gray", linestyle=":", linewidth=1.5, label="Nominal $\\alpha_0=0.05$")
    ax.set_xlabel("Temporal dependence $\\phi$")
    ax.set_ylabel("Empirical False-Alarm Probability")
    ax.set_title("False-Alarm Probability vs $\\phi$ (Single Configuration)")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig1_false_alarm_vs_phi.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig1")


def make_fig2_mean_delay(df: pd.DataFrame, figures_dir: Path) -> None:
    """Fig 2: Mean delay vs phi for each effect, proposed method only (final config).
    Mean +/- SE per phi.
    """
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=300)
    change_df = df[df["effect"] != "none"]
    colors = {"1.0": "#1f77b4", "1.5": "#2ca02c"}

    for eff in ["1.0", "1.5"]:
        sub = change_df[
            (change_df["effect"] == eff) & (change_df["method"] == "proposed")
        ].sort_values("phi")
        if sub.empty:
            continue
        # One entry per phi - just use the values directly (no ablation zigzag)
        grp = sub.groupby("phi").agg(
            delay_mean=("mean_delay", "mean"),
        ).reset_index()
        # Compute SE from original if available
        if "mean_delay_se" in sub.columns:
            grp_se = sub.groupby("phi")["mean_delay_se"].mean().reset_index(name="delay_se")
            grp = grp.merge(grp_se, on="phi")
            ax.errorbar(
                grp["phi"], grp["delay_mean"], yerr=grp["delay_se"],
                label=f"Proposed (Effect {eff})",
                color=colors[eff], marker="o", capsize=4,
            )
        else:
            ax.plot(
                grp["phi"], grp["delay_mean"],
                label=f"Proposed (Effect {eff})",
                color=colors[eff], marker="o",
            )

    # Task 4 data if available (n=1000 streams)
    task4_csv = RESULTS_FIX / "task4_f5_evaluation.csv"
    if task4_csv.exists():
        t4 = pd.read_csv(task4_csv)
        for eff_v in [1.0, 1.5]:
            eff_str = f"{eff_v:.1f}"
            sub4 = t4[t4["effect"] == eff_v].sort_values("phi")
            if not sub4.empty and "mean_delay" in sub4.columns:
                ax.plot(
                    sub4["phi"], sub4["mean_delay"],
                    label=f"Proposed n=1000 (Effect {eff_str})",
                    color=colors.get(eff_str, "gray"),
                    marker="^", linestyle="--",
                )

    ax.set_xlabel("Temporal dependence $\\phi$")
    ax.set_ylabel("Mean Detection Delay (steps)")
    ax.set_title("Mean Detection Delay vs $\\phi$ (Final Configuration)")
    ax.text(
        0.03, 0.55,
        "Note: Vanilla FAR = 100% for $\\phi \\geq 0.3$;\ndelay comparison valid only at $\\phi=0.0$.",
        transform=ax.transAxes, fontsize=8.5, style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9),
    )
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig2_mean_delay_vs_phi.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig2")


def make_fig3_ci_coverage(df: pd.DataFrame, figures_dir: Path) -> None:
    """Fig 3: CI coverage vs phi. Show F5 (c*) and old per-phi calibrated CI.
    Error bars from Task 4 evaluation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

    # Left panel: n=200 streams from original eval
    ax1 = axes[0]
    change_df = df[df["effect"] != "none"]
    for eff, marker, col in [("1.0", "o", "#1f77b4"), ("1.5", "s", "#ff7f0e")]:
        sub = change_df[
            (change_df["effect"] == eff) & (change_df["method"] == "proposed")
        ].sort_values("phi")
        if sub.empty or "coverage" not in sub.columns:
            continue
        grp = sub.groupby("phi").agg(
            cov_mean=("coverage", "mean"),
            cov_se=("coverage_se", "mean"),
        ).reset_index()
        ax1.errorbar(grp["phi"], grp["cov_mean"] * 100, yerr=grp["cov_se"] * 100,
                     label=f"Old CI (Effect {eff}, n=200)", color=col, marker=marker, capsize=4)

    ax1.axhline(95, color="red", linestyle=":", linewidth=1.5, label="Nominal 95%")
    ax1.set_xlabel("$\\phi$")
    ax1.set_ylabel("Coverage (%)")
    ax1.set_title("(a) Old CI — n=200 Streams")
    ax1.set_ylim(70, 102)
    ax1.legend(loc="lower left", fontsize=9)

    # Right panel: F5 with c* from Task 4 (n=1000)
    ax2 = axes[1]
    task4_csv = RESULTS_FIX / "task4_f5_evaluation.csv"
    if task4_csv.exists():
        t4 = pd.read_csv(task4_csv)
        colors_eff = {1.0: "#1f77b4", 1.5: "#ff7f0e"}
        markers_eff = {1.0: "o", 1.5: "s"}
        for eff_v in [1.0, 1.5]:
            sub4 = t4[t4["effect"] == eff_v].sort_values("phi")
            if sub4.empty:
                continue
            c_star = sub4["c_star"].iloc[0] if "c_star" in sub4.columns else "?"
            ax2.errorbar(
                sub4["phi"], sub4["f5_coverage"] * 100, yerr=sub4["f5_coverage_se"] * 100,
                label=f"F5 CI c*={c_star:.2f} (Effect {eff_v:.1f})",
                color=colors_eff[eff_v], marker=markers_eff[eff_v], capsize=4,
            )
            ax2.plot(
                sub4["phi"], sub4["old_coverage"] * 100,
                label=f"Old per-$\\phi$ CI (Effect {eff_v:.1f})",
                color=colors_eff[eff_v], marker=markers_eff[eff_v], linestyle="--",
            )

        ax2.axhline(95, color="red", linestyle=":", linewidth=1.5, label="Nominal 95%")
        ax2.set_xlabel("$\\phi$")
        ax2.set_ylabel("Coverage (%)")
        ax2.set_title("(b) F5 CI vs Old Baseline — n=1000 Streams")
        ax2.set_ylim(70, 102)
        ax2.legend(loc="lower left", fontsize=9)
    else:
        ax2.set_title("(b) Task 4 data not yet available")
        ax2.text(0.5, 0.5, "Run diagnostic_task4.py first", transform=ax2.transAxes, ha="center")

    fig.suptitle("Confidence Interval Coverage vs $\\phi$", fontsize=14)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig3_ci_coverage_vs_phi.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig3")


def make_fig4_support_jaccard(df: pd.DataFrame, figures_dir: Path) -> None:
    """Fig 4: Support Jaccard vs phi. Single config (proposed method, final c for S_report).
    Mean +/- SE per phi.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    change_df = df[df["effect"] != "none"]
    colors = {"1.0": "#1f77b4", "1.5": "#2ca02c"}

    for eff in ["1.0", "1.5"]:
        sub = change_df[
            (change_df["effect"] == eff) & (change_df["method"] == "proposed")
        ].sort_values("phi")
        if sub.empty or "mean_support_jaccard" not in sub.columns:
            continue
        sub = sub.dropna(subset=["mean_support_jaccard"])
        grp = sub.groupby("phi").agg(
            jac_mean=("mean_support_jaccard", "mean"),
        ).reset_index()
        ax.plot(grp["phi"], grp["jac_mean"], label=f"S_report (Effect {eff})",
                color=colors[eff], marker="o")

    ax.set_xlabel("Temporal dependence $\\phi$")
    ax.set_ylabel("Mean Jaccard Similarity")
    ax.set_title("Support Recovery Jaccard vs $\\phi$ (Final Configuration)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig4_support_jaccard_vs_phi.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig4")


def make_fig_support_ci_vs_report(figures_dir: Path) -> None:
    """New figure: S_CI vs S_report Jaccard / precision / recall.
    Uses task3_support_validation.csv and task3_f2_large_c_calib.csv.
    """
    val_csv = RESULTS_FIX / "task3_support_validation.csv"
    f2_csv = RESULTS_FIX / "task3_f2_large_c_calib.csv"

    if not val_csv.exists():
        print(f"  Skipping S_CI vs S_report figure: {val_csv} not found")
        return

    df_val = pd.read_csv(val_csv)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=300)
    metrics = [
        ("S_CI_recall", "S_CI Recall", "Recall", "#1f77b4"),
        ("S_CI_precision", "S_CI Precision", "Precision", "#d62728"),
        ("S_CI_jaccard", "S_CI Jaccard", "Jaccard", "#9467bd"),
    ]
    # S_report columns
    report_metrics = [
        ("S_report_F2_recall", "S_report Recall", "Recall", "#ff7f0e"),
        ("S_report_F2_precision", "S_report Precision", "Precision", "#2ca02c"),
        ("S_report_F2_jaccard", "S_report Jaccard", "Jaccard", "#8c564b"),
    ]

    for ax, (ci_col, ci_label, ylabel, ci_col_color), (rep_col, rep_label, _, rep_col_color) in zip(
        axes, metrics, report_metrics
    ):
        if ci_col in df_val.columns:
            ax.plot(df_val["phi"], df_val[ci_col], label=ci_label,
                    marker="o", color=ci_col_color, linewidth=2)
        if rep_col in df_val.columns:
            ax.plot(df_val["phi"], df_val[rep_col], label=rep_label,
                    marker="s", color=rep_col_color, linewidth=2, linestyle="--")

        # Bonferroni reference
        if "Bonf_271_jaccard" in df_val.columns and ylabel == "Jaccard":
            ax.plot(df_val["phi"], df_val["Bonf_271_jaccard"], label="Bonferroni (z=2.71)",
                    marker="^", color="gray", linewidth=1.5, linestyle=":")

        ax.set_xlabel("$\\phi$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"S_CI vs S_report — {ylabel}")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower left", fontsize=9)

    fig.suptitle("Support S_CI (high recall for CI) vs S_report (high Jaccard)", fontsize=13)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_support_ci_vs_report.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_support_ci_vs_report")


def main() -> None:
    print("=" * 60)
    print("TASK 6: Regenerate Figures")
    print("=" * 60)
    set_style()

    df = load_eval_json_df()
    if df.empty:
        print("WARNING: No eval JSON records found; some figures may be empty")

    print(f"Loaded {len(df)} eval records")

    print("\nGenerating Figures 1-4 (final single-config)...")
    if not df.empty:
        make_fig1_false_alarm(df, FIGURES_FIX)
        make_fig2_mean_delay(df, FIGURES_FIX)
        make_fig3_ci_coverage(df, FIGURES_FIX)
        make_fig4_support_jaccard(df, FIGURES_FIX)
    else:
        # Still try fig3 which uses task4 data
        make_fig3_ci_coverage(df, FIGURES_FIX)

    print("\nGenerating S_CI vs S_report figure...")
    make_fig_support_ci_vs_report(FIGURES_FIX)

    print(f"\nAll figures saved to {FIGURES_FIX}")
    print("Task 6 complete.")


if __name__ == "__main__":
    main()
