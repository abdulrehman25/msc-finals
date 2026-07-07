"""
Histogram comparisons from existing evaluation artifacts.

Inputs (read-only):
  artifacts/eval/predictions_<stem>.csv
  labels under --labels-root: <stem>.labels.txt OR <stem>.log.labels.txt

Outputs:
  artifacts/eval/advanced_plots/hists/
    - hist_<stem>_<branch>.png              (per-dataset, per-branch; labeled if labels exist)
    - hist_<stem>_<branch>_unlabeled.png    (when labels are missing)
    - hist_overlap_<branch>.png             (optional cross-dataset overlays)

Usage examples:
  # pick specific datasets
  python scripts/histograms.py --labels-root "D:\\...\\web-log-dummy" --stems access_eval_mix_2000,access_eval_small_500

  # include overlays across datasets, and use fewer bins
  python scripts/histograms.py --labels-root "D:\\...\\web-log-dummy" --stems access_eval_mix_2000,nginx_json_eval_800 --overlays --bins 40
"""

from pathlib import Path
import argparse, json
import numpy as np
import matplotlib.pyplot as plt

EVAL_DIR = Path("artifacts/eval")
OUT_DIR  = EVAL_DIR / "advanced_plots" / "hists"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BRANCHES = ["content", "session", "fused"]

def _norm_stem(s: str) -> str:
    p = Path(s.strip())
    stem = p.stem  # strips .log/.json
    if stem.startswith("metrics_"):
        stem = stem[len("metrics_"):]
    return stem

def _find_predictions(stem: str) -> Path | None:
    p = EVAL_DIR / f"predictions_{stem}.csv"
    return p if p.exists() else None

def _find_labels(stem: str, labels_root: Path | None) -> Path | None:
    candidates = []
    if labels_root and labels_root.exists():
        candidates += list(labels_root.rglob(f"{stem}.labels.txt"))
        candidates += list(labels_root.rglob(f"{stem}.log.labels.txt"))
    candidates += [EVAL_DIR / f"{stem}.labels.txt", EVAL_DIR / f"{stem}.log.labels.txt"]
    for c in candidates:
        if c.exists():
            return c
    return None

def _load_scores_and_labels(pred_csv: Path, labels_path: Path | None):
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

    y = None
    if labels_path:
        y_vals = []
        with open(labels_path, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                s = ln.strip()
                if not s:
                    continue
                try:
                    y_vals.append(int(s.split(",")[-1]))
                except Exception:
                    pass
        y = np.array(y_vals, dtype=int)

    # Align lengths if needed
    n = len(scores["fused"])
    if y is not None and len(y) != n:
        m = min(n, len(y))
        for k in scores:
            scores[k] = scores[k][:m]
        y = y[:m]
    return scores, y

def _per_dataset_hists(stems, labels_root: Path | None, bins: int, density: bool):
    for stem in stems:
        pred_csv = _find_predictions(stem)
        if not pred_csv:
            print(f"[WARN] predictions missing for {stem}; skipping.")
            continue
        labels_path = _find_labels(stem, labels_root)
        scores, y = _load_scores_and_labels(pred_csv, labels_path)

        for branch in BRANCHES:
            s = scores[branch]
            s = s[np.isfinite(s)]
            if s.size == 0:
                print(f"[WARN] empty scores for {stem}/{branch}")
                continue

            if y is not None and y.size == s.size and np.unique(y).size >= 1:
                # labeled hist: benign vs attack
                benign = s[y == 0]
                attack = s[y == 1]
                plt.figure(figsize=(8, 5))
                if benign.size:
                    plt.hist(benign, bins=bins, density=density, alpha=0.5, label="Benign")
                if attack.size:
                    plt.hist(attack, bins=bins, density=density, alpha=0.5, label="Attack")
                plt.xlabel(f"{branch.capitalize()} score")
                plt.ylabel("Density" if density else "Count")
                plt.title(f"Histogram • {branch} • {stem}")
                plt.legend()
                plt.tight_layout()
                out = OUT_DIR / f"hist_{stem}_{branch}.png"
                plt.savefig(out, dpi=160); plt.close()
                print("[OK]", out)
            else:
                # unlabeled hist
                plt.figure(figsize=(8,5))
                plt.hist(s, bins=bins, density=density, alpha=0.8)
                plt.xlabel(f"{branch.capitalize()} score")
                plt.ylabel("Density" if density else "Count")
                plt.title(f"Histogram (unlabeled) • {branch} • {stem}")
                plt.tight_layout()
                out = OUT_DIR / f"hist_{stem}_{branch}_unlabeled.png"
                plt.savefig(out, dpi=160); plt.close()
                print("[OK]", out)

def _overlays(stems, labels_root: Path | None, bins: int, density: bool):
    # Cross-dataset overlay per branch
    for branch in BRANCHES:
        any_data = False
        plt.figure(figsize=(9,6))
        for stem in stems:
            pred_csv = _find_predictions(stem)
            if not pred_csv:
                continue
            scores, _ = _load_scores_and_labels(pred_csv, None)
            s = scores[branch]
            s = s[np.isfinite(s)]
            if s.size == 0:
                continue
            # use line histogram outlines for legibility across many datasets
            plt.hist(s, bins=bins, density=density, histtype="step", linewidth=1.5, label=stem)
            any_data = True
        if not any_data:
            plt.close()
            continue
        plt.xlabel(f"{branch.capitalize()} score")
        plt.ylabel("Density" if density else "Count")
        plt.title(f"Histogram overlay across datasets • {branch}")
        plt.legend()
        plt.tight_layout()
        out = OUT_DIR / f"hist_overlap_{branch}.png"
        plt.savefig(out, dpi=160); plt.close()
        print("[OK]", out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", type=str, default=None, help="Folder with labels files (recursive).")
    ap.add_argument("--stems", type=str, default=None,
                    help="Comma-separated stems (e.g., access_eval_mix_2000,nginx_json_eval_800). "
                         "You may also pass names with .log/.json.")
    ap.add_argument("--bins", type=int, default=50, help="Histogram bins.")
    ap.add_argument("--density", action="store_true", help="Plot density instead of counts.")
    ap.add_argument("--overlays", action="store_true", help="Also create cross-dataset overlays per branch.")
    args = ap.parse_args()

    labels_root = Path(args.labels_root) if args.labels_root else None

    # discover available stems from metrics_*.json if none provided
    metrics = sorted(EVAL_DIR.glob("metrics_*.json"))
    discovered = [m.stem[len("metrics_"):] for m in metrics if m.stem.startswith("metrics_")]

    if args.stems:
        stems = [_norm_stem(s) for s in args.stems.split(",") if s.strip()]
        # keep only those we have predictions for
        stems = [s for s in stems if _find_predictions(s)]
    else:
        # default to everything with predictions present
        stems = [s for s in discovered if _find_predictions(s)]

    if not stems:
        print("[ERR] No datasets found (need predictions_*.csv). Run evaluate_file.py first.")
        return

    _per_dataset_hists(stems, labels_root, bins=args.bins, density=args.density)

    if args.overlays:
        _overlays(stems, labels_root, bins=args.bins, density=args.density)

if __name__ == "__main__":
    main()
