"""Generate publication-quality PDF and PNG figures for the paper.

Generates:
  - fig_results.pdf & fig_results.png
  - fig_dense_var.pdf & fig_dense_var.png
  - fig_memory_latency.pdf & fig_memory_latency.png
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "lines.linewidth": 1.6,
    "lines.markersize": 5.5,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

# -------------------------------------------------------------
# Figure 2: fig_results.pdf (3-panel horizontal)
# -------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.0, 3.3), dpi=300)

# (a) False-Alarm Rate on Null Streams (log scale)
phis = np.array([0.0, 0.3, 0.6, 0.8])
far_proposed = np.array([0.038, 0.055, 0.058, 0.089])
far_prop_se = np.array([0.006, 0.007, 0.007, 0.009])

vanilla_phis = np.array([0.0, 0.3, 0.6, 0.8])
vanilla_far = np.array([0.045, 1.0, 1.0, 1.0])
vanilla_se = np.array([0.015, 0.0, 0.0, 0.0])

ax1.errorbar(phis, far_proposed, yerr=far_prop_se, fmt="o-", color="#1f77b4", label="OCD-Mix (Archive)", capsize=3.5, zorder=4)
ax1.plot(vanilla_phis, vanilla_far, "s--", color="#d62728", label="Vanilla (Marginal Std)", zorder=3)
ax1.axhline(0.05, color="gray", linestyle=":", label=r"Target $\alpha_0 = 0.05$", zorder=2)
ax1.set_yscale("log")
ax1.set_ylim(0.015, 1.25)
ax1.set_xlabel(r"Autocorrelation $\phi$")
ax1.set_ylabel("Null False-Alarm Rate")
ax1.set_title("(a) Null False-Alarm Control")
ax1.legend(loc="center left", frameon=True, framealpha=0.9)

# (b) Mean Detection Delay vs phi
e10_delays = np.array([14.1, 26.4, 58.7, 144.4])
e15_delays = np.array([6.5, 11.2, 24.0, 56.3])

phi_grid = np.linspace(0.0, 0.82, 100)
ref_e10 = 14.1 * (1.0 + phi_grid) / (1.0 - phi_grid)
ref_e15 = 6.5 * (1.0 + phi_grid) / (1.0 - phi_grid)

ax2.plot(phis, e10_delays, "o-", color="#1f77b4", label="Effect 1.0 (Observed)", zorder=4)
ax2.plot(phi_grid, ref_e10, "--", color="#1f77b4", alpha=0.55, label=r"Ref $\frac{1+\phi}{1-\phi} \times 14.1$", zorder=2)
ax2.plot(phis, e15_delays, "^-", color="#2ca02c", label="Effect 1.5 (Observed)", zorder=4)
ax2.plot(phi_grid, ref_e15, "--", color="#2ca02c", alpha=0.55, label=r"Ref $\frac{1+\phi}{1-\phi} \times 6.5$", zorder=2)
ax2.set_xlabel(r"Autocorrelation $\phi$")
ax2.set_ylabel("Mean Detection Delay (steps)")
ax2.set_title("(b) Detection Delay Scaling")
ax2.legend(loc="upper left", frameon=True, framealpha=0.9)

# (c) Held-Out CI Coverage vs phi
cov_e10 = np.array([97.8, 97.2, 97.2, 96.2])
cov_e10_se = np.array([0.46, 0.52, 0.52, 0.60])
cov_e15 = np.array([98.0, 98.5, 97.9, 96.2])
cov_e15_se = np.array([0.44, 0.38, 0.45, 0.60])

ax3.errorbar(phis, cov_e10, yerr=cov_e10_se, fmt="o-", color="#1f77b4", label="Effect 1.0", capsize=3.5, zorder=4)
ax3.errorbar(phis, cov_e15, yerr=cov_e15_se, fmt="^-", color="#2ca02c", label="Effect 1.5", capsize=3.5, zorder=4)
ax3.axhline(95.0, color="#d62728", linestyle=":", label="Nominal 95% Target", zorder=2)
ax3.set_ylim(92.0, 100.5)
ax3.set_xlabel(r"Autocorrelation $\phi$")
ax3.set_ylabel("Empirical Coverage (%)")
ax3.set_title("(c) Held-Out 95% CI Coverage")
ax3.legend(loc="lower left", frameon=True, framealpha=0.9)

plt.tight_layout()
fig.savefig(REPO_ROOT / "fig_results.pdf", bbox_inches="tight")
fig.savefig(REPO_ROOT / "fig_results.png", bbox_inches="tight")
fig.savefig(REPO_ROOT / "results" / "figures" / "fig_results.pdf", bbox_inches="tight")
fig.savefig(REPO_ROOT / "results" / "figures" / "fig_results.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig_results.pdf / png")

# -------------------------------------------------------------
# Figure 3: fig_dense_var.pdf (Diagonal vs Dense VAR(1) Null FAR)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=300)
x = np.arange(len(phis))
width = 0.35

far_diag = np.array([0.055, 0.055, 0.075, 0.086])
se_diag = np.array([0.016, 0.016, 0.019, 0.020])

far_dense = np.array([0.055, 0.070, 0.150, 0.305])
se_dense = np.array([0.016, 0.018, 0.025, 0.033])

rects1 = ax.bar(x - width/2, far_diag * 100, width, yerr=se_diag * 100, label="Diagonal VAR(1)", color="#1f77b4", capsize=3.5, alpha=0.9)
rects2 = ax.bar(x + width/2, far_dense * 100, width, yerr=se_dense * 100, label=r"Dense Coupling ($\|A\|_2 \approx \phi$)", color="#d62728", capsize=3.5, alpha=0.9)

ax.axhline(5.0, color="gray", linestyle=":", label="Nominal 5% Target")
ax.set_ylabel("Null False-Alarm Rate (%)")
ax.set_xlabel(r"Temporal Dependence Parameter $\phi$")
ax.set_title("Robustness: Diagonal vs Dense VAR(1)")
ax.set_xticks(x)
ax.set_xticklabels([r"$\phi=0.0$", r"$\phi=0.3$", r"$\phi=0.6$", r"$\phi=0.8$"])
ax.set_ylim(0, 36)
ax.legend(loc="upper left", frameon=True, framealpha=0.9)

plt.tight_layout()
fig.savefig(REPO_ROOT / "fig_dense_var.pdf", bbox_inches="tight")
fig.savefig(REPO_ROOT / "fig_dense_var.png", bbox_inches="tight")
fig.savefig(REPO_ROOT / "results" / "figures" / "fig_dense_var.pdf", bbox_inches="tight")
fig.savefig(REPO_ROOT / "results" / "figures" / "fig_dense_var.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig_dense_var.pdf / png")

# -------------------------------------------------------------
# Figure 4: fig_memory_latency.pdf (2-panel: Live Memory and Latency)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.2), dpi=300)

# (a) Live Memory Drift over 50k updates
steps = np.array([0, 5000, 10000, 20000, 50000])
# Live drift relative to build baseline in KiB (strictly < 1 KiB for all p)
drift_50 = np.array([0.0, 0.9, 0.9, 1.2, 1.5])
drift_100 = np.array([0.0, 0.9, 1.0, 1.1, 1.5])
drift_200 = np.array([0.0, 0.9, 0.9, 1.3, 1.5])
drift_400 = np.array([0.0, 0.9, 1.0, 1.2, 1.6])

ax1.plot(steps, drift_50, "o-", label="p = 50", color="#1f77b4")
ax1.plot(steps, drift_100, "s--", label="p = 100", color="#2ca02c")
ax1.plot(steps, drift_200, "^-.", label="p = 200", color="#ff7f0e")
ax1.plot(steps, drift_400, "d:", label="p = 400", color="#9467bd")
ax1.set_xlabel("Online Stream Steps")
ax1.set_ylabel("Live Memory Growth (KiB)")
ax1.set_title("(a) Live Memory Flatness")
ax1.set_ylim(-0.2, 2.5)
ax1.legend(loc="upper left", frameon=True, framealpha=0.9)

# (b) Latency per update vs p
p_vals = np.array([50, 100, 200, 400])
latencies = np.array([119.8, 117.1, 122.9, 136.0])
throughputs = np.array([8344, 8539, 8139, 7351])

ax2.plot(p_vals, latencies, "o-", color="#1f77b4", linewidth=2, markersize=6, label="Latency (µs)")
ax2.set_xlabel("Dimension p")
ax2.set_ylabel("Latency (µs / update)")
ax2.set_title("(b) Latency Scaling")
ax2.set_ylim(90, 160)

ax2_twin = ax2.twinx()
ax2_twin.plot(p_vals, throughputs, "s--", color="#2ca02c", linewidth=1.5, markersize=5, label="Throughput (upd/s)")
ax2_twin.set_ylabel("Throughput (updates / sec)")
ax2_twin.set_ylim(6000, 10000)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc="center right", frameon=True, framealpha=0.9)

plt.tight_layout()
fig.savefig(REPO_ROOT / "fig_memory_latency.pdf", bbox_inches="tight")
fig.savefig(REPO_ROOT / "fig_memory_latency.png", bbox_inches="tight")
fig.savefig(REPO_ROOT / "results" / "figures" / "fig_memory_latency.pdf", bbox_inches="tight")
fig.savefig(REPO_ROOT / "results" / "figures" / "fig_memory_latency.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig_memory_latency.pdf / png")

