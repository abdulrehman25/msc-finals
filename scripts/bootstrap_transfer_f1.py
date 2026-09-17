"""
Bootstrap 95% CIs for F1 under the cross-dataset threshold-transfer protocol
(Table 7 / tab:threshold_transfer): the decision threshold is frozen once,
from CSIC 2010's held-out benign rows, and applied UNCHANGED to each target
dataset's resampled scores. Mirrors scripts/threshold_transfer.py's frozen
thresholds exactly, but resamples the target dataset with replacement instead
of scoring it once, using the per-request scores already saved to disk.

Usage:
    python scripts/bootstrap_transfer_f1.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_file import load_labels_sidecar
from scripts.threshold_transfer import SOURCE_STEM, SOURCE_LABELS, TARGETS, BRANCHES, load_predictions

N_BOOT = 2000
SEED = 42
EVAL_DIR = ROOT / "artifacts" / "eval"


def bootstrap_f1_fixed_threshold(y, s, thr, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    f1s = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        yhat = (s[idx] > thr).astype(int)
        f1s.append(f1_score(yb, yhat, zero_division=0))
    f1s = np.array(f1s)
    return float(np.mean(f1s)), float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))


def main():
    src_scores = load_predictions(SOURCE_STEM)
    n_src = len(src_scores["content"])
    y_src = load_labels_sidecar(str(ROOT / SOURCE_LABELS), n_rows=n_src)
    benign_mask = (y_src == 0)

    tau = {}
    for branch in BRANCHES:
        arr = src_scores[branch]
        valid = benign_mask & ~np.isnan(arr)
        tau[branch] = float(np.percentile(arr[valid], 95)) if valid.sum() else None
    print(f"[SOURCE] Frozen thresholds: {tau}")

    rows = []
    for stem, labels_rel in TARGETS.items():
        tgt_scores = load_predictions(stem)
        n_tgt = len(tgt_scores["content"])
        y_tgt = load_labels_sidecar(str(ROOT / labels_rel), n_rows=n_tgt)
        if y_tgt is None:
            continue
        for branch in ("content", "fused"):
            arr = tgt_scores[branch]
            valid = ~np.isnan(arr)
            if valid.sum() == 0 or tau[branch] is None:
                continue
            yv, sv = y_tgt[valid], arr[valid]
            f1_mean, f1_lo, f1_hi = bootstrap_f1_fixed_threshold(yv, sv, tau[branch])
            rows.append({"target_dataset": stem, "branch": branch, "n": int(valid.sum()),
                         "threshold": tau[branch], "transferred_f1_mean": f1_mean,
                         "transferred_f1_lo": f1_lo, "transferred_f1_hi": f1_hi})
            print(f"{stem:24s} {branch:8s} n={valid.sum():5d}  "
                  f"transferred F1={f1_mean:.3f} [{f1_lo:.3f}, {f1_hi:.3f}]")

    out_csv = EVAL_DIR / "bootstrap_transfer_f1.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n[OK] wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
