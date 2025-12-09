"""
Aggregate and plot evaluation results (AUC, PR-AUC, F1, etc.) from artifacts/eval/.
Also (optionally) overlays ROC/PR curves across datasets if labels are found.

Usage examples (from project root):
  python scripts/plot_evals.py
  python scripts/plot_evals.py --labels-root "D:\\path\\to\\logs_and_labels"

Outputs:
  artifacts/eval/plots/
    - auc_bar.png
    - prauc_bar.png
    - f1_bar.png
    - precision_bar.png
    - recall_bar.png
    - fpr_bar.png
    - alert_rate_bar.png     (present if available in metrics)
    - roc_overlay_fused.png  (if labels found)
    - roc_overlay_content.png (if labels found)
    - roc_overlay_session.png (if labels found)
    - pr_overlay_fused.png   (if labels found)
    - pr_overlay_content.png (if labels found)
    - pr_overlay_session.png (if labels found)
  artifacts/eval/summary_metrics.csv
"""

import os
import re
import json
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

EVAL_DIR = Path("artifacts/eval")
PLOTS_DIR = EVAL_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def stem_from_metrics_path(p: Path) -> str:
    # metrics_<stem>.json
    name = p.stem
    if name.startswith("metrics_"):
        return name[len("metrics_"):]
    return name

def load_all_metrics() -> pd.DataFrame:
    rows = []
    for mp in sorted(EVAL_DIR.glob("metrics_*.json")):
        try:
            data = json.load(open(mp, "r", encoding="utf-8"))
        except Exception:
            continue
        stem = stem_from_metrics_path(mp)
        n_lines = data.get("n_lines")
        labels_used = data.get("labels_used", False)
        thrs = data.get("thresholds", {})
        M = data.get("metrics", {}) or {}

        for branch in ["fused","content","session"]:
            m = M.get(branch)
            if not m:
                rows.append({
                    "dataset": stem, "branch": branch, "labels_used": labels_used, "n_lines": n_lines,
                    "auc": np.nan, "prauc": np.nan, "f1": np.nan, "precision": np.nan, "recall": np.nan,
                    "fpr": np.nan, "tn": np.nan, "fp": np.nan, "fn": np.nan, "tp": np.nan,
                    "threshold": thrs.get(branch), "alert_rate": np.nan
                })
                continue
            rows.append({
                "dataset": stem, "branch": branch, "labels_used": labels_used, "n_lines": n_lines,
                "auc": m.get("auc"), "prauc": m.get("prauc"), "f1": m.get("f1"),
                "precision": m.get("precision"), "recall": m.get("recall"), "fpr": m.get("fpr"),
                "tn": m.get("tn"), "fp": m.get("fp"), "fn": m.get("fn"), "tp": m.get("tp"),
                "threshold": m.get("threshold", thrs.get(branch)),
                "alert_rate": m.get("alert_rate", np.nan)
            })
    df = pd.DataFrame(rows)
    return df

def grouped_bar(df: pd.DataFrame, metric: str, fname: str):
    """Make a grouped bar chart: x = dataset, bars = branches."""
    if df.empty:
        return
    pivot = df.pivot_table(index="dataset", columns="branch", values=metric, aggfunc="first")
    pivot = pivot[["fused","content","session"]].reindex(columns=["fused","content","session"])
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_xlabel("Dataset")
    ax.set_ylabel(metric.upper())
    ax.set_title(metric.upper())
    ax.legend(title="Branch")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname, dpi=160)
    plt.close()

def find_predictions(stem: str) -> Optional[Path]:
    p = EVAL_DIR / f"predictions_{stem}.csv"
    return p if p.exists() else None

def find_labels_for_stem(stem: str, labels_root: Optional[Path]) -> Optional[Path]:
    """
    Try to find <stem>.labels.txt under labels_root (recursive) if provided,
    otherwise look in common locations (same dir as project, eval dir, etc.).
    """
    candidates: List[Path] = []
    # 1) artifacts/eval/<stem>.labels.txt (rare)
    candidates.append(EVAL_DIR / f"{stem}.labels.txt")
    # 2) if labels_root given, search recursively for file named "<stem>.labels.txt"
    if labels_root and labels_root.exists():
        for p in labels_root.rglob(f"{stem}.labels.txt"):
            candidates.append(p)
    # 3) common: alongside a known dummy bundle directory name in project tree
    # (best effort – user can pass --labels-root to be explicit)
    for c in candidates:
        if c.exists():
            return c
    return None

def load_scores_and_labels(pred_csv: Path, labels_path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Returns dict branch -> (y_true, y_score) for fused/content/session
    """
    y_true = []
    with open(labels_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            try:
                lab = int(s.split(",")[-1])
                y_true.append(lab)
            except Exception:
                pass
    y_true = np.array(y_true, dtype=int)

    sc_c, sc_s, sc_f = [], [], []
    with open(pred_csv, "r", encoding="utf-8", errors="ignore") as f:
        header = next(f, None)
        for ln in f:
            parts = ln.rstrip("\n").split(",", 3)
            if len(parts) < 4:
                continue
            sc, ss, sf = parts[0].strip(), parts[1].strip(), parts[2].strip()
            sc_c.append(float(sc) if sc else np.nan)
            sc_s.append(float(ss) if ss else np.nan)
            sc_f.append(float(sf) if sf else np.nan)
    sc_c = np.array(sc_c, dtype=float)
    sc_s = np.array(sc_s, dtype=float)
    sc_f = np.array(sc_f, dtype=float)

    n = min(len(y_true), len(sc_f))
    y_true = y_true[:n]
    sc_c = sc_c[:n]
    sc_s = sc_s[:n]
    sc_f = sc_f[:n]

    out = {
        "fused": (y_true[~np.isnan(sc_f)], sc_f[~np.isnan(sc_f)]),
        "content": (y_true[~np.isnan(sc_c)], sc_c[~np.isnan(sc_c)]),
        "session": (y_true[~np.isnan(sc_s)], sc_s[~np.isnan(sc_s)]),
    }
    return out

def overlay_curve(curve: str, pairs: List[Tuple[str, np.ndarray, np.ndarray]], title: str, fname: str):
    """
    curve: 'roc' or 'pr'
    pairs: list of (label_for_legend, y_true, y_score)
    """
    if not pairs:
        return
    plt.figure(figsize=(8, 6))
    for label, y_true, y_score in pairs:
        # Guard against single-class labels (ROC undefined)
        classes = np.unique(y_true)
        if len(classes) < 2:
            continue
        if curve == "roc":
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")
        else:
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            pr_auc = auc(rec, prec)
            plt.plot(rec, prec, label=f"{label} (PR-AUC={pr_auc:.3f})")
    if curve == "roc":
        plt.plot([0,1],[0,1],'--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
    else:
        plt.xlabel("Recall")
        plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", type=str, default=None,
                    help="Root folder to search for <stem>.labels.txt (recursive). Optional but required for curves if labels aren't next to evals.")
    args = ap.parse_args()
    labels_root = Path(args.labels_root) if args.labels_root else None

    df = load_all_metrics()
    if df.empty:
        print("[ERR] No metrics_*.json found in artifacts/eval/")
        return

    # Save summary CSV
    df.sort_values(["dataset","branch"]).to_csv(EVAL_DIR / "summary_metrics.csv", index=False)
    print("[OK] Wrote", EVAL_DIR / "summary_metrics.csv")

    # Bar charts per metric
    for metric, fname in [
        ("auc", "auc_bar.png"),
        ("prauc", "prauc_bar.png"),
        ("f1", "f1_bar.png"),
        ("precision", "precision_bar.png"),
        ("recall", "recall_bar.png"),
        ("fpr", "fpr_bar.png"),
        ("alert_rate", "alert_rate_bar.png"),
    ]:
        grouped_bar(df, metric, fname)
        print("[OK] Plot", fname)

    # Overlays: ROC/PR per branch across datasets (requires labels + predictions)
    stems = sorted(df["dataset"].unique())
    for branch in ["fused","content","session"]:
        roc_pairs = []
        pr_pairs = []
        for stem in stems:
            pred_csv = find_predictions(stem)
            if not pred_csv:
                continue
            labels_path = find_labels_for_stem(stem, labels_root)
            if not labels_path:
                continue
            d = load_scores_and_labels(pred_csv, labels_path)
            y_true, y_score = d[branch]
            if y_true.size == 0:
                continue
            if len(np.unique(y_true)) < 2:
                # Skip single-class datasets for ROC/PR curves
                continue
            roc_pairs.append((stem, y_true, y_score))
            pr_pairs.append((stem, y_true, y_score))
        # Make figures (one figure per branch per curve type)
        overlay_curve("roc", roc_pairs, f"ROC curves • {branch}", f"roc_overlay_{branch}.png")
        overlay_curve("pr", pr_pairs, f"PR curves • {branch}", f"pr_overlay_{branch}.png")
        print(f"[OK] Overlays for {branch} (if any)")

if __name__ == "__main__":
    main()
