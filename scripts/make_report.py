#!/usr/bin/env python3
"""Generate figures, summary tables, and final report from evaluation results (Task 7)."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_matplotlib_style() -> None:
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
        "lines.markersize": 6,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def load_eval_results(eval_dir: Path) -> list[dict]:
    """Load all JSON summaries from eval_dir."""
    files = sorted(eval_dir.glob("*.json"))
    records = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            data["file_stem"] = f.stem
            records.append(data)
    return records


def make_figures_and_tables(
    eval_dir: Path,
    bench_csv: Path,
    tables_dir: Path,
    figures_dir: Path,
    alpha0: float = 0.05,
) -> pd.DataFrame:
    """Generate all 5 required figures and summary tables."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    records = load_eval_results(eval_dir)
    if not records:
        print(f"No JSON records found in {eval_dir}")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Parse dataset properties: phi and effect
    # Names: null_phi{00,03,06,08}, change_phi{00,03,06,08}_e{10,15}
    def parse_phi(row: pd.Series) -> float:
        ds = row["dataset"]
        for p_str, p_val in [("phi00", 0.0), ("phi03", 0.3), ("phi06", 0.6), ("phi08", 0.8)]:
            if p_str in ds:
                return p_val
        return 0.0

    def parse_effect(row: pd.Series) -> str:
        ds = row["dataset"]
        if "e10" in ds:
            return "1.0"
        elif "e15" in ds:
            return "1.5"
        return "none"

    df["phi"] = df.apply(parse_phi, axis=1)
    df["effect"] = df.apply(parse_effect, axis=1)

    # Save summary tables
    df.to_csv(tables_dir / "all_eval_summaries.csv", index=False)

    null_df = df[df["effect"] == "none"].sort_values(["phi", "method"])
    null_table = null_df[[
        "dataset", "method", "phi", "n_streams",
        "false_alarm_probability", "false_alarm_probability_se", "censored_arl"
    ]]
    null_table.to_csv(tables_dir / "null_metrics.csv", index=False)

    change_df = df[df["effect"] != "none"].sort_values(["phi", "effect", "method"])
    change_cols = [
        "dataset", "method", "phi", "effect", "n_streams",
        "detection_rate", "detection_rate_se",
        "false_alarm_rate", "false_alarm_rate_se",
        "mean_delay", "median_delay", "p90_delay",
        "coverage", "coverage_se", "mean_ci_width",
        "mean_support_precision", "mean_support_recall", "mean_support_jaccard",
    ]
    change_cols_exist = [c for c in change_cols if c in change_df.columns]
    change_table = change_df[change_cols_exist]
    change_table.to_csv(tables_dir / "change_metrics.csv", index=False)

    set_matplotlib_style()

    # -------------------------------------------------------------
    # Figure 1: False-alarm probability vs phi (vanilla vs proposed)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    for method, col, marker in [("proposed", "#1f77b4", "o"), ("vanilla", "#d62728", "s")]:
        sub = null_df[null_df["method"] == method].sort_values("phi")
        if not sub.empty:
            ax.errorbar(
                sub["phi"], sub["false_alarm_probability"],
                yerr=sub["false_alarm_probability_se"],
                label=f"{method.capitalize()}",
                color=col, marker=marker, capsize=4,
            )
    ax.axhline(alpha0, color="gray", linestyle=":", label=f"Nominal $\\alpha_0 = {alpha0}$")
    ax.set_xlabel("Temporal dependence parameter $\\phi$")
    ax.set_ylabel("Empirical False-Alarm Probability")
    ax.set_title("False-Alarm Probability vs $\\phi$ on Null Streams")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig1_false_alarm_vs_phi.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 2: Mean delay vs phi for each effect, vanilla vs proposed
    # (Caption note: only where false-alarm rates are comparable; drop empty vanilla entries)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=300)
    colors = {"1.0": "#1f77b4", "1.5": "#2ca02c"}
    styles = {"proposed": "-", "vanilla": "--"}
    markers = {"proposed": "o", "vanilla": "^"}

    for eff in ["1.0", "1.5"]:
        # Proposed method (has valid delays across all phi)
        sub_prop = change_df[(change_df["effect"] == eff) & (change_df["method"] == "proposed")].sort_values("phi")
        if not sub_prop.empty:
            ax.plot(
                sub_prop["phi"], sub_prop["mean_delay"],
                label=f"Proposed (Effect {eff})",
                color=colors[eff],
                linestyle=styles["proposed"],
                marker=markers["proposed"],
            )

        # Vanilla baseline: only plot points where delay is not null (i.e. phi=0)
        sub_van = change_df[(change_df["effect"] == eff) & (change_df["method"] == "vanilla")].dropna(subset=["mean_delay"]).sort_values("phi")
        if not sub_van.empty:
            ax.plot(
                sub_van["phi"], sub_van["mean_delay"],
                label=f"Vanilla (Effect {eff}, $\\phi=0$ only)",
                color=colors[eff],
                linestyle=styles["vanilla"],
                marker=markers["vanilla"],
            )

    ax.set_xlabel("Temporal dependence parameter $\\phi$")
    ax.set_ylabel("Mean Detection Delay (steps)")
    ax.set_title("Mean Delay vs $\\phi$ by Mean Effect Size")
    ax.legend(loc="upper left")
    ax.text(
        0.03, 0.55,
        "Note: Vanilla false-alarm rate is 100% for $\\phi \\geq 0.3$;\ndelay comparison is valid only at $\\phi=0.0$.",
        transform=ax.transAxes, fontsize=8.5, style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9),
    )
    fig.savefig(figures_dir / "fig2_mean_delay_vs_phi.png", bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 3: CI coverage vs phi (proposed method)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    for eff in ["1.0", "1.5"]:
        sub = change_df[(change_df["effect"] == eff) & (change_df["method"] == "proposed")].sort_values("phi")
        if not sub.empty and "coverage" in sub.columns:
            ax.errorbar(
                sub["phi"], sub["coverage"],
                yerr=sub["coverage_se"],
                label=f"Proposed (Effect {eff})",
                marker="o", capsize=4,
            )
    ax.axhline(0.95, color="red", linestyle=":", label="Nominal 95% Coverage")
    ax.set_xlabel("Temporal dependence parameter $\\phi$")
    ax.set_ylabel("Empirical Confidence Interval Coverage")
    ax.set_title("95% Confidence Interval Coverage vs $\\phi$")
    ax.set_ylim(0.70, 1.02)
    ax.legend(loc="lower left")
    fig.savefig(figures_dir / "fig3_ci_coverage_vs_phi.png", bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 4: Support Jaccard vs phi
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    for eff in ["1.0", "1.5"]:
        for method in ["proposed", "vanilla"]:
            sub = change_df[(change_df["effect"] == eff) & (change_df["method"] == method)].sort_values("phi")
            if not sub.empty and "mean_support_jaccard" in sub.columns and sub["mean_support_jaccard"].notna().any():
                label_txt = f"{method.capitalize()} (Effect {eff})"
                if method == "vanilla":
                    label_txt += " [$\\phi=0$ only]"
                ax.plot(
                    sub["phi"], sub["mean_support_jaccard"],
                    label=label_txt,
                    linestyle=styles[method],
                    marker=markers[method],
                    color=colors[eff],
                )
    ax.set_xlabel("Temporal dependence parameter $\\phi$")
    ax.set_ylabel("Mean Support Jaccard Similarity")
    ax.set_title("Coordinate Support Recovery (Jaccard Index) vs $\\phi$")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right")
    fig.savefig(figures_dir / "fig4_support_jaccard_vs_phi.png", bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 5: Two-panel memory benchmark (Footprint vs p; Live memory vs steps)
    # -------------------------------------------------------------
    mem_csv = tables_dir / "benchmark_memory.csv"
    if not mem_csv.exists() and bench_csv.exists():
        mem_csv = bench_csv

    if mem_csv.exists():
        mdf = pd.read_csv(mem_csv)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=300)

        # Panel A: Footprint after build vs p
        if "footprint_kib_after_build" in mdf.columns:
            ax1.plot(mdf["p"], mdf["footprint_kib_after_build"], marker="o", color="#1f77b4", linewidth=2.2)
            ax1.set_xlabel("Dimension $p$")
            ax1.set_ylabel("Build Footprint (KiB)")
            ax1.set_title("(a) Initial Detector Footprint vs $p$")
            ax1.set_xticks(mdf["p"])

        # Panel B: Live memory vs steps across p
        step_pts = [5000, 20000, 50000]
        step_labels = ["5k", "20k", "50k"]
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        for idx, row in mdf.iterrows():
            p_val = int(row["p"])
            color = palette[idx % len(palette)]
            if "live_kib_5k" in row and "live_kib_20k" in row and "live_kib_50k" in row:
                live_vals = [row["live_kib_5k"], row["live_kib_20k"], row["live_kib_50k"]]
                ax2.plot(step_pts, live_vals, marker="s", label=f"$p = {p_val}$", color=color, linewidth=2)

        ax2.set_xticks(step_pts)
        ax2.set_xticklabels(step_labels)
        ax2.set_xlabel("Observations Processed (steps)")
        ax2.set_ylabel("Live Traced Memory (KiB)")
        ax2.set_title("(b) Live Memory vs Stream Steps")
        ax2.legend(loc="center left")

        fig.savefig(figures_dir / "fig5_memory_vs_stream_length.png", bbox_inches="tight")
        plt.close(fig)

    print(f"Generated tables in {tables_dir} and figures in {figures_dir}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate summary tables, figures, and report.")
    parser.add_argument("--eval-dir", type=str, default="results/eval", help="Directory with eval JSONs.")
    parser.add_argument("--bench-csv", type=str, default="results/tables/benchmark_online.csv", help="Path to bench CSV.")
    parser.add_argument("--tables-dir", type=str, default="results/tables", help="Directory to save tables.")
    parser.add_argument("--figures-dir", type=str, default="results/figures", help="Directory to save figures.")
    args = parser.parse_args()

    make_figures_and_tables(
        eval_dir=Path(args.eval_dir),
        bench_csv=Path(args.bench_csv),
        tables_dir=Path(args.tables_dir),
        figures_dir=Path(args.figures_dir),
    )


if __name__ == "__main__":
    main()
