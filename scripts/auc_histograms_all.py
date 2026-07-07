"""
Make histograms of AUC scores for fused/content/session across selected stems.

Outputs (to artifacts/eval/advanced_plots/hists/):
  - auc_hist_overlaid_all.png     (all branches on one plot)
  - auc_hist_grid_all.png         (1×3 subplots: fused, content, session)
  - auc_hist_all.csv              (table of AUCs per stem & branch)

Usage:
  python scripts/auc_histograms_all.py --stems access_eval_mix_2000,nginx_json_eval_800
  python scripts/auc_histograms_all.py --stems access_eval_mix_2000,access_eval_small_500 --bins 12
"""

from pathlib import Path
import argparse, json
import numpy as np
import matplotlib.pyplot as plt

EVAL_DIR = Path("artifacts/eval")
OUT_DIR  = EVAL_DIR / "advanced_plots" / "hists"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BRANCHES = ["fused", "content", "session"]

def _norm_stem(s: str) -> str:
    p = Path(s.strip())
    stem = p.stem  # strips .log/.json
    if stem.startswith("metrics_"):
        stem = stem[len("metrics_"):]
    return stem

def _load_auc_for_branch(stems, branch: str):
    vals = []
    for stem in stems:
        mp = EVAL_DIR / f"metrics_{stem}.json"
        if not mp.exists():
            print(f"[WARN] missing {mp}; skip {stem}")
            continue
        try:
            data = json.load(open(mp, "r", encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] read error {mp}: {e}")
            continue
        auc = ((data.get("metrics") or {}).get(branch) or {}).get("auc", None)
        if isinstance(auc, (int, float)) and np.isfinite(auc):
            vals.append((stem, float(auc)))
        else:
            print(f"[WARN] {stem}/{branch}: AUC unavailable; skip")
    return vals  # list[(stem, auc)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", required=True,
                    help="Comma-separated stems (accepts names with .log/.json too).")
    ap.add_argument("--bins", type=int, default=10, help="Histogram bins.")
    ap.add_argument("--density", action="store_true", help="Use density instead of counts.")
    args = ap.parse_args()

    stems = [_norm_stem(s) for s in args.stems.split(",") if s.strip()]
    if not stems:
        print("[ERR] No stems provided.")
        return

    # Load AUC arrays per branch
    auc_map = {}  # branch -> (labels list, values np.array)
    for b in BRANCHES:
        pairs = _load_auc_for_branch(stems, b)
        if not pairs:
            auc_map[b] = ([], np.array([], dtype=float))
            continue
        names, vals = zip(*pairs)
        auc_map[b] = (list(names), np.array(vals, dtype=float))

    # Save CSV summary
    csv_out = OUT_DIR / "auc_hist_all.csv"
    with open(csv_out, "w", encoding="utf-8") as f:
        f.write("stem,fused,content,session\n")
        # unify across branches
        all_names = sorted(set(sum((names for names, _ in auc_map.values()), [])))
        for name in all_names:
            row = [name]
            for b in BRANCHES:
                names, vals = auc_map[b]
                if name in names:
                    row.append(f"{vals[names.index(name)]:.6f}")
                else:
                    row.append("")
            f.write(",".join(row) + "\n")
    print("[OK] wrote", csv_out)

    # Prepare common bins across all branches (based on combined values)
    combined = np.concatenate([v for _, v in auc_map.values() if v.size])
    if combined.size == 0:
        print("[ERR] No AUC values found.")
        return
    bins = np.linspace(combined.min(), combined.max(), args.bins + 1)

    # 1) Overlaid histogram (all branches together)
    fig, ax = plt.subplots(figsize=(9, 6))
    for b in BRANCHES:
        names, vals = auc_map[b]
        if vals.size:
            ax.hist(vals, bins=bins, density=args.density, histtype="step", linewidth=1.5, label=b)
    ax.set_xlabel("AUC")
    ax.set_ylabel("Density" if args.density else "Count")
    ax.set_title("AUC histogram (overlaid) • fused vs content vs session")
    ax.legend()
    # simple rugs at bottom (offset by small amounts per branch)
    ymax = (ax.get_ylim()[1] or 1.0)
    step = max(0.02 * ymax, 0.2)
    for i, b in enumerate(BRANCHES):
        _, vals = auc_map[b]
        if vals.size:
            ax.vlines(vals, 0 + i*step, (i+1)*step, linewidth=1)
    plt.tight_layout()
    out_over = OUT_DIR / "auc_hist_overlaid_all.png"
    plt.savefig(out_over, dpi=160); plt.close(fig)
    print("[OK] wrote", out_over)

    # 2) Grid (1×3) — one subplot per branch
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
    for j, b in enumerate(BRANCHES):
        ax = axes[0, j]
        names, vals = auc_map[b]
        if vals.size:
            ax.hist(vals, bins=bins, density=args.density, alpha=0.8)
            mu = float(np.mean(vals)); med = float(np.median(vals))
            ax.axvline(mu, linestyle="--", linewidth=1, label=f"mean={mu:.3f}")
            ax.axvline(med, linestyle=":", linewidth=1, label=f"median={med:.3f}")
            ax.legend()
        ax.set_xlabel("AUC"); ax.set_ylabel("Density" if args.density else "Count")
        ax.set_title(b)
    plt.tight_layout()
    out_grid = OUT_DIR / "auc_hist_grid_all.png"
    plt.savefig(out_grid, dpi=160); plt.close(fig)
    print("[OK] wrote", out_grid)

if __name__ == "__main__":
    main()
