"""
Histogram of AUC scores across selected datasets (stems), using metrics_* artifacts.

Usage examples:
  # fused branch (default), two stems
  python scripts/auc_histogram.py --stems access_eval_mix_2000,nginx_json_eval_800

  # session branch with more stems and custom bins
  python scripts/auc_histogram.py --branch session --stems access_eval_mix_2000,access_eval_small_500,nginx_json_eval_800 --bins 12
"""

from pathlib import Path
import argparse, json
import numpy as np
import matplotlib.pyplot as plt

EVAL_DIR = Path("artifacts/eval")
OUT_DIR  = EVAL_DIR / "advanced_plots" / "hists"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _norm_stem(s: str) -> str:
    p = Path(s.strip())
    stem = p.stem  # strips .log/.json
    if stem.startswith("metrics_"):
        stem = stem[len("metrics_"):]
    return stem

def _load_auc(stems, branch: str):
    """Return list of (stem, auc_float) for given branch."""
    out = []
    for stem in stems:
        mp = EVAL_DIR / f"metrics_{stem}.json"
        if not mp.exists():
            print(f"[WARN] missing {mp}; skipping")
            continue
        try:
            data = json.load(open(mp, "r", encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] could not read {mp}: {e}")
            continue
        m = (data.get("metrics") or {}).get(branch) or {}
        auc = m.get("auc", None)
        if auc is None or not isinstance(auc, (int, float)) or not np.isfinite(auc):
            print(f"[WARN] {stem}/{branch}: AUC not available; skipping")
            continue
        out.append((stem, float(auc)))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", required=True,
                    help="Comma-separated list of stems (e.g., access_eval_mix_2000,nginx_json_eval_800). "
                         "You may include names with .log/.json.")
    ap.add_argument("--branch", choices=["fused","content","session"], default="fused",
                    help="Which branch's AUC to plot (default: fused).")
    ap.add_argument("--bins", type=int, default=10, help="Histogram bins (default: 10).")
    args = ap.parse_args()

    stems = [_norm_stem(s) for s in args.stems.split(",") if s.strip()]
    if not stems:
        print("[ERR] No stems provided after normalization.")
        return

    pairs = _load_auc(stems, args.branch)
    if not pairs:
        print("[ERR] No valid AUC values found for the requested stems/branch.")
        return

    labels, aucs = zip(*pairs)
    aucs = np.array(aucs, dtype=float)

    # Build histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    counts, bins, _patches = ax.hist(aucs, bins=args.bins, alpha=0.8)
    ax.set_xlabel(f"AUC ({args.branch})")
    ax.set_ylabel("Count")
    ax.set_title(f"AUC histogram • branch = {args.branch} • stems = {len(aucs)}")

    # Rug ticks for individual stems (helps compare which stem is where)
    ymax = counts.max() if counts.size else 1.0
    tick_h = max(0.05 * ymax, 0.5)
    for x, name in zip(aucs, labels):
        ax.vlines(x, 0, tick_h, linewidth=1)
        # optional: tiny label—commented to avoid clutter; uncomment if desired
        # ax.text(x, tick_h * 1.05, name, rotation=90, va="bottom", ha="center", fontsize=7)

    # Mean / median lines
    mu = float(np.mean(aucs))
    med = float(np.median(aucs))
    ax.axvline(mu, linestyle="--", linewidth=1, label=f"mean={mu:.3f}")
    ax.axvline(med, linestyle=":", linewidth=1, label=f"median={med:.3f}")
    ax.legend()

    plt.tight_layout()
    out = OUT_DIR / f"auc_hist_{args.branch}.png"
    plt.savefig(out, dpi=160)
    plt.close(fig)
    print("[OK] wrote", out)

    # Also drop a quick CSV summary next to the figure
    csv_out = OUT_DIR / f"auc_hist_{args.branch}.csv"
    with open(csv_out, "w", encoding="utf-8") as f:
        f.write("stem,auc\n")
        for name, val in pairs:
            f.write(f"{name},{val:.6f}\n")
    print("[OK] wrote", csv_out)

if __name__ == "__main__":
    main()
