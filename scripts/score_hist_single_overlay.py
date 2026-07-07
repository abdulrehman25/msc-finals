"""
One histogram that overlays Benign vs Attack for Content, Session, and Fused scores.
- Uses SAME BINS across all shown curves (computed from the union of scores).
- Draws a vertical threshold line at p95 for each branch (configurable).
- Concatenates data across ALL selected stems into one combined histogram.

Inputs:
  artifacts/eval/predictions_<stem>.csv
  labels under --labels-root:
    <stem>.label.txt | <stem>.labels.txt | <stem>.log.label.txt | <stem>.log.labels.txt

Output:
  artifacts/eval/advanced_plots/hists_threshold/hist_SINGLE_overlay_<branches>.png

Usage:
  python scripts/score_hist_single_overlay.py --labels-root "D:\\...\\web-log-dummy" ^
    --stems access_eval_mix_2000,nginx_json_eval_800 --branches all --bins 60 --percentile 95

  # only fused+content:
  python scripts/score_hist_single_overlay.py --labels-root "D:\\...\\web-log-dummy" ^
    --stems access_eval_mix_2000,access_eval_small_500 --branches fused,content
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

EVAL_DIR = Path("artifacts/eval")
OUT_DIR  = EVAL_DIR / "advanced_plots" / "hists_threshold"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_BRANCHES = ["content", "session", "fused"]

def _norm_stem(s: str) -> str:
    p = Path(s.strip())
    stem = p.stem  # strips .log/.json
    return stem[8:] if stem.startswith("metrics_") else stem

def _preds_path(stem: str) -> Path | None:
    p = EVAL_DIR / f"predictions_{stem}.csv"
    return p if p.exists() else None

def _labels_path(stem: str, labels_root: Path) -> Path | None:
    # Support .label.txt and .labels.txt variants (with/without .log)
    pats = [
        f"{stem}.label.txt", f"{stem}.labels.txt",
        f"{stem}.log.label.txt", f"{stem}.log.labels.txt",
    ]
    for pat in pats:
        for c in labels_root.rglob(pat):
            if c.exists():
                return c
    # also try alongside eval dir just in case
    for pat in pats:
        c = EVAL_DIR / pat
        if c.exists():
            return c
    return None

def _load_scores_and_labels(pred_csv: Path, labels_path: Path):
    # returns dict branch->scores (np.ndarray), labels (np.ndarray)
    sc_c, sc_s, sc_f = [], [], []
    with open(pred_csv, "r", encoding="utf-8", errors="ignore") as f:
        _ = next(f, None)  # header
        for ln in f:
            parts = ln.rstrip("\n").split(",", 3)
            if len(parts) < 3:
                continue
            sc, ss, sf = parts[0].strip(), parts[1].strip(), parts[2].strip()
            sc_c.append(float(sc) if sc else np.nan)
            sc_s.append(float(ss) if ss else np.nan)
            sc_f.append(float(sf) if sf else np.nan)
    scores = {
        "content": np.array(sc_c, dtype=float),
        "session": np.array(sc_s, dtype=float),
        "fused":   np.array(sc_f, dtype=float),
    }
    # labels (0/1)
    labs = []
    with open(labels_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            try:
                labs.append(int(s.split(",")[-1]))
            except Exception:
                pass
    y = np.array(labs, dtype=int)

    # align lengths
    n = len(scores["fused"])
    m = min(n, len(y))
    for k in scores:
        scores[k] = scores[k][:m]
    y = y[:m]
    return scores, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", required=True, type=str, help="Folder containing *.label.txt files (searched recursively).")
    ap.add_argument("--stems", required=True, type=str, help="Comma-separated stems (accepts names with .log/.json).")
    ap.add_argument("--branches", default="all", type=str,
                    help="Which branches to overlay: all OR comma list from {content,session,fused}. Default: all.")
    ap.add_argument("--bins", type=int, default=60, help="Number of histogram bins.")
    ap.add_argument("--percentile", type=float, default=95.0, help="Threshold percentile (pXX vertical line).")
    ap.add_argument("--density", action="store_true", help="Use density instead of counts.")
    args = ap.parse_args()

    labels_root = Path(args.labels_root)
    stems = [_norm_stem(s) for s in args.stems.split(",") if s.strip()]
    if args.branches.lower() == "all":
        branches = ALL_BRANCHES
    else:
        branches = [b.strip().lower() for b in args.branches.split(",") if b.strip().lower() in ALL_BRANCHES]
        if not branches:
            print("[ERR] No valid branches selected; choose from content,session,fused")
            return

    # Collect combined arrays per branch
    combined = {b: {"scores": np.empty(0, dtype=float), "labels": np.empty(0, dtype=int)} for b in branches}

    for stem in stems:
        pred_csv = _preds_path(stem)
        if not pred_csv:
            print(f"[WARN] predictions_{stem}.csv not found; skip")
            continue
        lab = _labels_path(stem, labels_root)
        if not lab:
            print(f"[WARN] labels for {stem} not found (.label.txt/.labels.txt); skip")
            continue
        scores, y = _load_scores_and_labels(pred_csv, lab)
        for b in branches:
            s = scores[b]
            mask = np.isfinite(s)
            combined[b]["scores"] = np.concatenate([combined[b]["scores"], s[mask]])
            combined[b]["labels"] = np.concatenate([combined[b]["labels"], y[mask]])

    # Ensure we have some data
    any_scores = np.concatenate([combined[b]["scores"] for b in branches if combined[b]["scores"].size])
    if any_scores.size == 0:
        print("[ERR] No scores found after loading/cleaning. Check stems and labels-root.")
        return

    # Same bins across all branches/classes: from union of all selected branches' scores
    smin, smax = float(np.min(any_scores)), float(np.max(any_scores))
    if not np.isfinite(smin) or not np.isfinite(smax) or smax <= smin:
        smax = smin + 1e-6
    edges = np.linspace(smin, smax, args.bins + 1)

    # Plot
    plt.figure(figsize=(10, 6))

    # Styles per branch (kept simple & readable)
    # We'll draw benign as solid line, attack as dashed, one color per branch.
    style = {
        "content": {"color": None},  # matplotlib will assign default distinct colors
        "session": {"color": None},
        "fused":   {"color": None},
    }

    # First pass: plot benign curves
    for b in branches:
        s = combined[b]["scores"]; y = combined[b]["labels"]
        if s.size == 0: 
            continue
        benign = s[y == 0]
        if benign.size:
            plt.hist(benign, bins=edges, density=args.density, histtype="step",
                     linewidth=1.8, label=f"Benign • {b}", **({} if style[b]["color"] is None else {"color": style[b]["color"]}))

    # Second pass: plot attack curves (dashed)
    for b in branches:
        s = combined[b]["scores"]; y = combined[b]["labels"]
        if s.size == 0: 
            continue
        attack = s[y == 1]
        if attack.size:
            plt.hist(attack, bins=edges, density=args.density, histtype="step",
                     linestyle="--", linewidth=1.8, label=f"Attack • {b}", **({} if style[b]["color"] is None else {"color": style[b]["color"]}))

    # Threshold lines: pXX per branch on ALL scores of that branch
    for b in branches:
        s = combined[b]["scores"]
        if s.size:
            thr = float(np.percentile(s, args.percentile))
            plt.axvline(thr, linestyle=":", linewidth=1.5, label=f"p{int(args.percentile)} • {b}={thr:.3f}")

    plt.xlabel("Anomaly score")
    plt.ylabel("Density" if args.density else "Count")
    stems_title = ", ".join(stems)
    plt.title(f"Benign vs Attack (overlay) • branches: {', '.join(branches)}\nStems: {stems_title}")
    plt.legend(ncol=2)
    plt.tight_layout()

    out_name = f"hist_SINGLE_overlay_{'-'.join(branches)}.png"
    out_path = OUT_DIR / out_name
    plt.savefig(out_path, dpi=160)
    plt.close()
    print("[OK]", out_path)

if __name__ == "__main__":
    main()
