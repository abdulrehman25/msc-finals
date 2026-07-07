"""
Histogram of anomaly scores with Benign vs Attack overlays and a vertical threshold line.

Features:
- Same bins for both classes (computed from all scores of the selected branch)
- Vertical threshold at p95 (configurable via --percentile)
- Branch: content | session | fused | all
- Handles multiple stems; writes one figure per (stem, branch)
- Falls back to unlabeled histogram if labels are missing (warns)

Inputs (read-only):
  artifacts/eval/predictions_<stem>.csv
  labels under --labels-root: <stem>.labels.txt OR <stem>.log.labels.txt (searched recursively)

Outputs:
  artifacts/eval/advanced_plots/hists_threshold/
    hist_<stem>_<branch>_benign_vs_attack.png
    (or hist_<stem>_<branch>_unlabeled.png if no labels)

Usage examples:
  python scripts/score_hist_threshold.py --labels-root "D:\\...\\web-log-dummy" \
    --stems access_eval_mix_2000,nginx_json_eval_800 --branch fused

  # all three branches:
  python scripts/score_hist_threshold.py --labels-root "D:\\...\\web-log-dummy" \
    --stems access_eval_mix_2000,access_eval_small_500 --branch all --bins 50 --percentile 95
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

EVAL_DIR = Path("artifacts/eval")
OUT_DIR  = EVAL_DIR / "advanced_plots" / "hists_threshold"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BRANCHES = ["content", "session", "fused"]

def _norm_stem(s: str) -> str:
    p = Path(s.strip())
    stem = p.stem  # strips .log/.json
    return stem[8:] if stem.startswith("metrics_") else stem

def _find_predictions(stem: str) -> Path | None:
    p = EVAL_DIR / f"predictions_{stem}.csv"
    return p if p.exists() else None

def _find_labels(stem: str, labels_root: Path | None) -> Path | None:
    cands = []
    if labels_root and labels_root.exists():
        cands += list(labels_root.rglob(f"{stem}.labels.txt"))
        cands += list(labels_root.rglob(f"{stem}.log.labels.txt"))
    cands += [EVAL_DIR / f"{stem}.labels.txt", EVAL_DIR / f"{stem}.log.labels.txt"]
    for c in cands:
        if c.exists():
            return c
    return None

def _load_scores_and_labels(pred_csv: Path, labels_path: Path | None):
    sc_c, sc_s, sc_f = [], [], []
    with open(pred_csv, "r", encoding="utf-8", errors="ignore") as f:
        _ = next(f, None)  # header
        for ln in f:
            parts = ln.rstrip("\n").split(",", 3)
            if len(parts) < 3: continue
            sc, ss, sf = parts[0].strip(), parts[1].strip(), parts[2].strip()
            sc_c.append(float(sc) if sc else np.nan)
            sc_s.append(float(ss) if ss else np.nan)
            sc_f.append(float(sf) if sf else np.nan)
    scores = {
        "content": np.array(sc_c, dtype=float),
        "session": np.array(sc_s, dtype=float),
        "fused":   np.array(sc_f, dtype=float),
    }

    y = None
    if labels_path:
        labs = []
        with open(labels_path, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                s = ln.strip()
                if not s: continue
                try:
                    labs.append(int(s.split(",")[-1]))
                except Exception:
                    pass
        y = np.array(labs, dtype=int)

    # Align lengths if needed
    n = len(scores["fused"])
    if y is not None and len(y) != n:
        m = min(n, len(y))
        for k in scores: scores[k] = scores[k][:m]
        y = y[:m]
    return scores, y

def _plot_hist_for_branch(stem: str, branch: str, scores: np.ndarray, labels: np.ndarray | None,
                          bins: int, percentile: float, density: bool):
    # Clean finite values
    s = scores[np.isfinite(scores)]
    if s.size == 0:
        print(f"[WARN] no finite {branch} scores for {stem}; skipping")
        return

    # Same bins for both classes = bins from all scores
    smin, smax = float(np.min(s)), float(np.max(s))
    if smax == smin:
        # avoid zero-width bins
        smax = smin + 1e-6
    edges = np.linspace(smin, smax, bins + 1)

    # Threshold at pXX across ALL scores
    thr = float(np.percentile(s, percentile))

    plt.figure(figsize=(8, 5))

    if labels is not None and labels.size == scores.size and np.unique(labels).size >= 1:
        benign = scores[(labels == 0) & np.isfinite(scores)]
        attack = scores[(labels == 1) & np.isfinite(scores)]

        # Draw histograms with the same bin edges
        if benign.size:
            plt.hist(benign, bins=edges, density=density, alpha=0.6, label=f"Benign (n={len(benign)})")
        if attack.size:
            plt.hist(attack, bins=edges, density=density, alpha=0.6, label=f"Attack (n={len(attack)})")

        fname = OUT_DIR / f"hist_{stem}_{branch}_benign_vs_attack.png"
        subtitle = "Benign vs Attack"
    else:
        # Unlabeled fallback
        plt.hist(s, bins=edges, density=density, alpha=0.8, label=f"All (n={len(s)})")
        fname = OUT_DIR / f"hist_{stem}_{branch}_unlabeled.png"
        subtitle = "Unlabeled (labels missing)"

    # Vertical threshold line
    plt.axvline(thr, linestyle="--", linewidth=1.5, label=f"p{int(percentile)}={thr:.3f}")

    plt.xlabel(f"{branch.capitalize()} score")
    plt.ylabel("Density" if density else "Count")
    plt.title(f"{stem} • {branch} • {subtitle}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=160)
    plt.close()
    print("[OK]", fname)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", type=str, required=True, help="Folder with labels files (recursive).")
    ap.add_argument("--stems", type=str, required=True,
                    help="Comma-separated stems (accepts names with .log/.json).")
    ap.add_argument("--branch", type=str, default="all", choices=["content","session","fused","all"],
                    help="Which branch to plot (default: all).")
    ap.add_argument("--bins", type=int, default=60, help="Number of histogram bins (default: 60).")
    ap.add_argument("--percentile", type=float, default=95.0, help="Threshold percentile (default: 95).")
    ap.add_argument("--density", action="store_true", help="Use density instead of counts.")
    args = ap.parse_args()

    labels_root = Path(args.labels_root)
    branches = BRANCHES if args.branch == "all" else [args.branch]
    stems = [_norm_stem(s) for s in args.stems.split(",") if s.strip()]

    for stem in stems:
        pred_csv = _find_predictions(stem)
        if not pred_csv:
            print(f"[WARN] predictions_{stem}.csv not found; skipping")
            continue
        labels_path = _find_labels(stem, labels_root)
        if not labels_path:
            print(f"[WARN] labels not found for {stem}; plotting unlabeled.")

        scores, y = _load_scores_and_labels(pred_csv, labels_path)

        for b in branches:
            _plot_hist_for_branch(stem, b, scores[b], y, bins=args.bins,
                                  percentile=args.percentile, density=args.density)

if __name__ == "__main__":
    main()
