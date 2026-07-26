"""
Generate additional paper figures for Results subsections that currently
have only a table (session fix, fusion sweep, threshold transfer, baselines).
Reads only from artifacts/paper/tables/ (the same CSVs already cited in the
paper's tables) so every figure is guaranteed consistent with the text.

Outputs -> artifacts/paper/figures/
    session_fix_bar.png
    fusion_sweep_line.png
    threshold_transfer_bar.png
    baseline_auc_bar.png

Usage: py scripts/make_extra_paper_figures.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "artifacts" / "paper" / "tables"
OUT = ROOT / "artifacts" / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

DPI = 160

# ---------------------------------------------------------------------------
# 1) Session construction fix: CSIC 2010 AUC/F1, original vs corrected
# ---------------------------------------------------------------------------
sv = pd.read_csv(TABLES / "session_v1_vs_v2.csv")
csic = sv[sv["dataset"] == "csic_eval"].set_index("session_version")

labels = ["AUC", "F1"]
original = [csic.loc["v1_naive", "session_auc"], csic.loc["v1_naive", "session_f1"]]
corrected = [csic.loc["v2_stateful", "session_auc"], csic.loc["v2_stateful", "session_f1"]]

x = np.arange(len(labels))
w = 0.35
fig, ax = plt.subplots(figsize=(6, 4.3))
ax.bar(x - w / 2, original, w, label="Original (naive windows)")
ax.bar(x + w / 2, corrected, w, label="Corrected (per-client windows)")
for i, v in enumerate(original):
    ax.text(x[i] - w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
for i, v in enumerate(corrected):
    ax.text(x[i] + w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("Session Branch • Effect of the Construction Fix (CSIC 2010)", fontsize=11)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8, frameon=False)
fig.savefig(OUT / "session_fix_bar.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 2) Fusion weight sweep: AUC and F1 vs session weight, CSIC 2010 only
# ---------------------------------------------------------------------------
fs = pd.read_csv(TABLES / "fusion_sweep.csv")
fs_csic = fs[fs["dataset"] == "csic_eval"].sort_values("w_session")

fig, ax1 = plt.subplots(figsize=(5.5, 4))
ax1.plot(fs_csic["w_session"], fs_csic["auc"], marker="o", color="tab:blue", label="AUC")
ax1.set_xlabel(r"Session weight $w_s$ (content weight $w_c = 1 - w_s$)")
ax1.set_ylabel("AUC", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.set_ylim(0.65, 1.0)

ax2 = ax1.twinx()
ax2.plot(fs_csic["w_session"], fs_csic["f1"], marker="s", color="tab:red", label="F1")
ax2.set_ylabel("F1", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
ax2.set_ylim(0.10, 0.25)

fig.suptitle("Fusion Weight Sweep • CSIC 2010")
fig.tight_layout()
fig.savefig(OUT / "fusion_sweep_line.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 3) Threshold transfer: self-calibrated vs transferred F1, per target dataset
# ---------------------------------------------------------------------------
tt = pd.read_csv(TABLES / "threshold_transfer.csv")
datasets = tt["target_dataset"].unique().tolist()
branches = ["content", "fused"]

fig, ax = plt.subplots(figsize=(7, 4.5))
n_groups = len(datasets)
n_bars = len(branches) * 2  # self + transferred per branch
w = 0.8 / n_bars
x_base = np.arange(n_groups)

colors = {"content_self": "tab:orange", "content_transf": "moccasin",
          "fused_self": "tab:blue", "fused_transf": "lightskyblue"}
offsets = np.linspace(-0.8 / 2 + w / 2, 0.8 / 2 - w / 2, n_bars)

series = []
for br in branches:
    sub = tt[tt["branch"] == br].set_index("target_dataset")
    series.append((f"{br} (self-calibrated)", [sub.loc[d, "self_f1"] for d in datasets], colors[f"{br}_self"]))
    series.append((f"{br} (transferred)", [sub.loc[d, "transferred_f1"] for d in datasets], colors[f"{br}_transf"]))

for off, (label, vals, color) in zip(offsets, series):
    ax.bar(x_base + off, vals, width=w, label=label, color=color)

ax.set_xticks(x_base)
ax.set_xticklabels([d.replace("_", " ") for d in datasets], rotation=10)
ax.set_ylabel("F1")
ax.set_title("Self-Calibrated vs. Transferred Threshold • F1 by Target Dataset")
ax.legend(fontsize=7, ncol=2, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "threshold_transfer_bar.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 4) Baseline comparison: AUC by method, per dataset (excludes the
#    all-normal session-training-only dataset, where AUC is undefined)
# ---------------------------------------------------------------------------
bc = pd.read_csv(TABLES / "baseline_comparison.csv")
bc = bc[bc["dataset"] != "access_super_long_session_2000"].copy()
bc = bc.sort_values("dataset")

methods = [
    ("content_auc", "Content (ours)"),
    ("fused_auc", "Fused (ours)"),
    ("rf_auc", "Random Forest"),
    ("iforest_auc", "Isolation Forest"),
    ("ocsvm_auc", "One-Class SVM"),
]
x = np.arange(len(bc))
w = 0.8 / len(methods)
fig, ax = plt.subplots(figsize=(9, 4.5))
for i, (col, label) in enumerate(methods):
    ax.bar(x + (i - len(methods) / 2 + 0.5) * w, bc[col], width=w, label=label)
ax.set_xticks(x)
ax.set_xticklabels([d.replace("_", " ") for d in bc["dataset"]], rotation=15, ha="right")
ax.set_ylabel("AUC")
ax.set_title("AUC by Method and Dataset")
ax.legend(fontsize=8, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
fig.savefig(OUT / "baseline_auc_bar.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

print("Wrote:")
for f in ["session_fix_bar.png", "fusion_sweep_line.png", "threshold_transfer_bar.png", "baseline_auc_bar.png"]:
    print(" -", OUT / f)
