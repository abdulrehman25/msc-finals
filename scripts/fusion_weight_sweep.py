"""
Fusion-weight sensitivity sweep: for each dataset, recompute the fused score
from the ALREADY-CACHED score_content/score_session columns in
artifacts/eval/predictions_<stem>.csv at several content/session weight
ratios, and report AUC/F1/FPR per ratio. No model inference needed.

Usage:
    python scripts/fusion_weight_sweep.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_file import compute_metrics, load_labels_sidecar, choose_threshold

EVAL_DIR = ROOT / "artifacts" / "eval"

CANONICAL_STEMS = [
    "csic_eval", "access_eval_mix_2000", "access_eval_small_500", "nginx_json_eval_800",
    "access_attacks_200", "access_mixed_500", "access_small_benign", "access_super_long_session_2000",
]

# stem -> label sidecar path
LABELS = {
    "csic_eval": "data/csic/csic_eval.log.labels.txt",
    "access_eval_mix_2000": "data/mixed/access_eval_mix_2000.log.labels.txt",
    "access_eval_small_500": "data/mixed/access_eval_small_500.log.labels.txt",
    "nginx_json_eval_800": "data/mixed/nginx_json_eval_800.log.labels.txt",
    "access_attacks_200": "data/raw/access_attacks_200.log.labels.txt",
    "access_mixed_500": "data/raw/access_mixed_500.log.labels.txt",
    "access_small_benign": "data/raw/access_small_benign.log.labels.txt",
    "access_super_long_session_2000": "data/raw/access_super_long_session_2000.log.labels.txt",
}

WEIGHT_RATIOS = [(1.0, 0.0), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.0, 1.0)]


def load_predictions(stem):
    path = EVAL_DIR / f"predictions_{stem}.csv"
    sc, ss = [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        next(f)  # header
        for ln in f:
            parts = ln.rstrip("\n").split(",", 3)
            if len(parts) < 4:
                continue
            sc.append(float(parts[0]) if parts[0] != "" else np.nan)
            ss.append(float(parts[1]) if parts[1] != "" else np.nan)
    return np.array(sc), np.array(ss)


def main():
    rows = []
    for stem in CANONICAL_STEMS:
        pred_path = EVAL_DIR / f"predictions_{stem}.csv"
        if not pred_path.exists():
            continue
        sc, ss = load_predictions(stem)
        n = len(sc)
        y_true = load_labels_sidecar(str(ROOT / LABELS[stem]), n_rows=n)
        if y_true is None:
            print(f"[SKIP] {stem}: labels unavailable/mismatched")
            continue

        has_session = not np.isnan(ss).all()

        for w_c, w_s in WEIGHT_RATIOS:
            if w_s > 0 and not has_session:
                # No session branch available for this dataset (see session_v1_vs_v2.csv) -
                # weighting toward a nonexistent branch is meaningless; skip.
                continue
            num = w_c * np.nan_to_num(sc, nan=0.0) + w_s * np.nan_to_num(ss, nan=0.0)
            den = w_c * (~np.isnan(sc)).astype(float) + w_s * (~np.isnan(ss)).astype(float)
            den[den == 0] = 1.0
            fused = num / den
            valid = ~np.isnan(sc) if w_s == 0 else (~np.isnan(sc) | ~np.isnan(ss))
            fused_valid = fused[valid]
            y_valid = y_true[valid]
            thr = choose_threshold(fused_valid, p=95.0)
            m = compute_metrics(y_valid, fused_valid, thr)
            rows.append({
                "dataset": stem, "w_content": w_c, "w_session": w_s,
                "auc": m["auc"], "f1": m["f1"], "precision": m["precision"],
                "recall": m["recall"], "fpr": m["fpr"],
            })
            print(f"{stem:32s} w=({w_c},{w_s})  auc={m['auc']}  f1={m['f1']:.4f}  fpr={m['fpr']:.4f}")

    out_csv = EVAL_DIR / "fusion_sweep.csv"
    cols = ["dataset", "w_content", "w_session", "auc", "f1", "precision", "recall", "fpr"]
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")
    print(f"\n[OK] wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
