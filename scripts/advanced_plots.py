# FILE: scripts/advanced_plots.py
"""
Make advanced visualizations from previously generated artifacts:
  1) Coherence of score time-series (content/session/fused) + time traces
  2) Scatter plot with legend (content vs session, colored by labels)
  3) Annotated heatmap of metrics across datasets × branches

Inputs read ONLY from artifacts:
  - artifacts/eval/metrics_*.json
  - artifacts/eval/predictions_*.csv
  - labels discovered via --labels-root (recursive) or alongside logs

Outputs:
  artifacts/eval/advanced_plots/
    - coherence_<stem>.png
    - scatter_<stem>.png
    - heatmap_<metric>.png  (for metric in {auc, prauc, f1, precision, recall, fpr, alert_rate})

Usage examples (from project root):
  python scripts/advanced_plots.py --labels-root "D:\\path\\to\\logs_or_bundle"
  # limit to specific datasets:
  python scripts/advanced_plots.py --labels-root "D:\\..." --stems access_eval_mix_2000,access_eval_small_500
  # pick which heatmap metric to render (default f1):
  python scripts/advanced_plots.py --labels-root "D:\\..." --heatmap-metric prauc
"""

from pathlib import Path
import argparse, json, os
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SciPy is used for coherence. If missing, raise a friendly error.
try:
    from scipy.signal import coherence
except Exception as e:
    raise SystemExit("This script requires SciPy. Please:\n  pip install scipy\n(original error: %s)" % e)

EVAL_DIR = Path("artifacts/eval")
OUT_DIR = EVAL_DIR / "advanced_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BRANCHES = ["fused", "content", "session"]

# ---------------- Utilities ----------------

def load_all_metrics() -> pd.DataFrame:
    rows = []
    for p in sorted(EVAL_DIR.glob("metrics_*.json")):
        try:
            data = json.load(open(p, "r", encoding="utf-8"))
        except Exception:
            continue
        stem = p.stem[len("metrics_"):] if p.stem.startswith("metrics_") else p.stem
        n_lines = data.get("n_lines")
        labels_used = data.get("labels_used", False)
        thresholds = data.get("thresholds", {}) or {}
        M = data.get("metrics", {}) or {}
        for b in BRANCHES:
            m = M.get(b)
            row = dict(dataset=stem, branch=b, n_lines=n_lines, labels_used=labels_used)
            if m:
                row.update({
                    "auc": m.get("auc"),
                    "prauc": m.get("prauc"),
                    "f1": m.get("f1"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "fpr": m.get("fpr"),
                    "tn": m.get("tn"), "fp": m.get("fp"), "fn": m.get("fn"), "tp": m.get("tp"),
                    "threshold": m.get("threshold", thresholds.get(b)),
                    "alert_rate": m.get("alert_rate", np.nan),
                })
            else:
                row.update({
                    "auc": np.nan, "prauc": np.nan, "f1": np.nan, "precision": np.nan,
                    "recall": np.nan, "fpr": np.nan, "tn": np.nan, "fp": np.nan, "fn": np.nan,
                    "tp": np.nan, "threshold": thresholds.get(b), "alert_rate": np.nan
                })
            rows.append(row)
    return pd.DataFrame(rows)

def predictions_path(stem: str) -> Optional[Path]:
    p = EVAL_DIR / f"predictions_{stem}.csv"
    return p if p.exists() else None

def find_labels(stem: str, labels_root: Optional[Path]) -> Optional[Path]:
    # Accept both "<stem>.labels.txt" and "<stem>.log.labels.txt"
    candidates: List[Path] = []
    if labels_root and labels_root.exists():
        for pat in (f"{stem}.labels.txt", f"{stem}.log.labels.txt"):
            candidates.extend(labels_root.rglob(pat))
    # also try eval dir (rare)
    candidates.append(EVAL_DIR / f"{stem}.labels.txt")
    candidates.append(EVAL_DIR / f"{stem}.log.labels.txt")
    for c in candidates:
        if c.exists():
            return c
    return None

def load_scores(stem: str) -> Dict[str, np.ndarray]:
    """Return dict of arrays: content/session/fused score series (NaNs kept) and also index t."""
    p = predictions_path(stem)
    if not p:
        return {}
    cont, sess, fus, lines = [], [], [], 0
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        _ = next(f, None)  # header
        for ln in f:
            parts = ln.rstrip("\n").split(",", 3)
            if len(parts) < 4: 
                continue
            sc, ss, sf = parts[0].strip(), parts[1].strip(), parts[2].strip()
            cont.append(float(sc) if sc else np.nan)
            sess.append(float(ss) if ss else np.nan)
            fus.append(float(sf) if sf else np.nan)
            lines += 1
    t = np.arange(lines, dtype=float)  # pseudo-time = line index
    return {"t": t, "content": np.array(cont, dtype=float), "session": np.array(sess, dtype=float), "fused": np.array(fus, dtype=float)}

def load_labels_array(labels_path: Path, n: int) -> np.ndarray:
    y = []
    with open(labels_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip()
            if not s: continue
            try:
                y.append(int(s.split(",")[-1]))
            except:
                pass
    arr = np.array(y, dtype=int)
    if arr.size > n:
        return arr[:n]
    if arr.size < n:
        # pad with zeros to align (rare mismatch)
        pad = np.zeros(n - arr.size, dtype=int)
        return np.concatenate([arr, pad])
    return arr

# ---------------- Plots ----------------

def plot_coherence(stem: str, series: Dict[str, np.ndarray]):
    """
    Two-panel plot:
      top  : time traces of content/session/fused (normalized)
      bottom: magnitude-squared coherence for pairs (C-S, C-F, S-F)
    """
    t = series["t"]
    c = np.nan_to_num(series["content"], nan=np.nanmedian(series["content"]))
    s = np.nan_to_num(series["session"], nan=np.nanmedian(series["session"]))
    f = np.nan_to_num(series["fused"],   nan=np.nanmedian(series["fused"]))

    # Normalize each to zero-mean, unit-variance for comparability
    def z(x):
        x = x - np.mean(x)
        std = np.std(x) or 1.0
        return x / std
    c_z, s_z, f_z = z(c), z(s), z(f)

    # Compute coherence with fs=1.0 (index-step as "time")
    fs = 1.0
    freq1, coh_cs = coherence(c_z, s_z, fs=fs, nperseg=min(256, len(t)))
    freq2, coh_cf = coherence(c_z, f_z, fs=fs, nperseg=min(256, len(t)))
    freq3, coh_sf = coherence(s_z, f_z, fs=fs, nperseg=min(256, len(t)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,6), constrained_layout=True)

    ax1.plot(t, c_z, label="Content (z)", alpha=0.9)
    ax1.plot(t, s_z, label="Session (z)", alpha=0.9)
    ax1.plot(t, f_z, label="Fused (z)", alpha=0.9)
    ax1.set_title(f"Score time-series (z-scored) • {stem}")
    ax1.set_xlabel("Line index")
    ax1.set_ylabel("Score (z)")
    ax1.legend(loc="upper right")

    ax2.plot(freq1, coh_cs, label="Coh(Content, Session)")
    ax2.plot(freq2, coh_cf, label="Coh(Content, Fused)")
    ax2.plot(freq3, coh_sf, label="Coh(Session, Fused)")
    ax2.set_xlabel("Frequency")
    ax2.set_ylabel("Coherence")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Magnitude-squared coherence")
    ax2.legend(loc="upper right")

    out = OUT_DIR / f"coherence_{stem}.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print("[OK] coherence →", out)

def plot_scatter(stem: str, series: Dict[str, np.ndarray], labels: Optional[np.ndarray], thr_fused: Optional[float]):
    """
    Scatter of Content vs Session with legend.
    Color by labels if available, otherwise pseudo-label by fused > threshold.
    """
    c = series["content"]; s = series["session"]; f = series["fused"]
    mask = ~np.isnan(c) & ~np.isnan(s)
    c, s = c[mask], s[mask]
    if labels is not None:
        y = labels[:len(mask)][mask]
        classes = {0: "Benign (label=0)", 1: "Attack (label=1)"}
        colors = {0: "tab:green", 1: "tab:orange"}
    else:
        # derive pseudo-label from fused threshold or p95
        if thr_fused is None or np.isnan(thr_fused):
            thr_fused = np.percentile(np.nan_to_num(f, nan=np.nanmedian(f)), 95)
        y = (np.nan_to_num(f, nan=np.nanmedian(f)) > thr_fused)[mask].astype(int)
        classes = {0: "Pseudo: Normal", 1: "Pseudo: Alert"}
        colors = {0: "tab:blue", 1: "tab:red"}

    plt.figure(figsize=(8,6))
    for cls in [0,1]:
        sel = (y == cls)
        if np.any(sel):
            plt.scatter(c[sel], s[sel], s=18, alpha=0.5, label=classes[cls], color=colors[cls], edgecolors="none")
    plt.xlabel("Content score")
    plt.ylabel("Session score")
    plt.title(f"Content vs Session • {stem}")
    plt.grid(True, alpha=0.2)
    plt.legend()
    out = OUT_DIR / f"scatter_{stem}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print("[OK] scatter →", out)

def plot_heatmap_metric(df_metrics: pd.DataFrame, metric: str):
    """
    Annotated heatmap: rows=datasets, cols=branches, cells=metric value.
    """
    if df_metrics.empty:
        print("[WARN] no metrics to plot heatmap")
        return
    pivot = df_metrics.pivot_table(index="dataset", columns="branch", values=metric, aggfunc="first")
    # keep consistent order
    cols = [c for c in BRANCHES if c in pivot.columns]
    pivot = pivot[cols].sort_index()

    # build figure
    fig, ax = plt.subplots(figsize=(max(8, 0.9*len(cols)+5), max(6, 0.35*len(pivot)+3)))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c.capitalize() for c in cols])
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(list(pivot.index))
    ax.set_title(f"Heatmap • {metric.upper()} (datasets × branches)")
    # annotate
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            txt = "—" if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else f"{v:.3f}"
            ax.text(j, i, txt, ha="center", va="center", color="white" if im.norm(v) > 0.6 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=metric.upper())
    fig.tight_layout()
    out = OUT_DIR / f"heatmap_{metric}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("[OK] heatmap →", out)
def _norm_stem(s: str) -> str:
    # accept "name", "name.log", "metrics_name.json" etc.
    s = s.strip()
    s = Path(s).stem  # strips .log / .json
    if s.startswith("metrics_"):
        s = s[len("metrics_"):]
    return s
# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", type=str, default=None,
                    help="Folder that contains labels files (<stem>.labels.txt or <stem>.log.labels.txt), searched recursively.")
    ap.add_argument("--stems", type=str, default=None,
                    help="Comma-separated list of dataset stems to include (default: all with predictions).")
    ap.add_argument("--heatmap-metric", type=str, default="f1",
                    choices=["auc","prauc","f1","precision","recall","fpr","alert_rate"],
                    help="Which metric to use in the annotated heatmap.")
    args = ap.parse_args()

    labels_root = Path(args.labels_root) if args.labels_root else None

    # Load metrics list (to fetch thresholds + stems)
    dfm = load_all_metrics()
    all_stems = sorted(dfm["dataset"].unique())
    if args.stems:
        wanted = [_norm_stem(s) for s in args.stems.split(",") if s.strip()]
        stems = [s for s in all_stems if s in wanted]
    else:
        stems = all_stems

    # ⬇️ use only selected stems for the heatmap
    dfm_sel = dfm[dfm["dataset"].isin(stems)].copy()

    # 1) Heatmap across SELECTED stems
    plot_heatmap_metric(dfm_sel, args.heatmap_metric)

    # 2) Per-stem coherence + scatter
    for stem in stems:
        series = load_scores(stem)
        if not series:
            print(f"[WARN] predictions missing for {stem}; skipping plots.")
            continue

        # thresholds (for pseudo-labels if needed)
        thr_row = dfm[(dfm["dataset"]==stem) & (dfm["branch"]=="fused")]
        thr_fused = thr_row["threshold"].iloc[0] if not thr_row.empty else np.nan

        # optional labels
        labels_path = find_labels(stem, labels_root)
        labels_arr = None
        if labels_path:
            labels_arr = load_labels_array(labels_path, n=len(series["t"]))

        plot_coherence(stem, series)
        plot_scatter(stem, series, labels_arr, thr_fused)

    print(f"[DONE] Plots saved in {OUT_DIR}")

if __name__ == "__main__":
    main()
