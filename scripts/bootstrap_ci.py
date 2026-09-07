"""
Bootstrap 95% confidence intervals for AUC (all methods) and F1 (our own
content/session/fused branches, at their already-reported fixed threshold),
computed directly from the per-request scores already saved to disk by
evaluate_file.py and plot_and_baselines.py. No retraining is needed.

Usage:
    python scripts/bootstrap_ci.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_file import load_labels_sidecar, choose_threshold

EVAL_DIR = ROOT / "artifacts" / "eval"
N_BOOT = 2000
SEED = 42

# (label for output, predictions csv, labels sidecar)
OWN_BRANCH_SOURCES = {
    "csic_eval": (EVAL_DIR / "predictions_csic_eval.csv",
                  ROOT / "data/csic/csic_heldout_eval.log.labels.txt"),
    "access_eval_mix_2000": (EVAL_DIR / "predictions_access_eval_mix_2000.csv",
                              ROOT / "data/mixed/access_eval_mix_2000.log.labels.txt"),
    "access_eval_small_500": (EVAL_DIR / "predictions_access_eval_small_500.csv",
                               ROOT / "data/mixed/access_eval_small_500.log.labels.txt"),
    "nginx_json_eval_800": (EVAL_DIR / "predictions_nginx_json_eval_800.csv",
                             ROOT / "data/mixed/nginx_json_eval_800.log.labels.txt"),
    "access_attacks_200": (EVAL_DIR / "predictions_access_attacks_200.csv",
                            ROOT / "data/raw/access_attacks_200.log.labels.txt"),
    "access_mixed_500": (EVAL_DIR / "predictions_access_mixed_500.csv",
                          ROOT / "data/raw/access_mixed_500.log.labels.txt"),
    "access_small_benign": (EVAL_DIR / "predictions_access_small_benign.csv",
                             ROOT / "data/raw/access_small_benign.log.labels.txt"),
}

BASELINE_KINDS = ["rf", "iforest", "ocsvm"]


def _load_own_scores(pred_csv: Path):
    sc, ss, sf = [], [], []
    with open(pred_csv, encoding="utf-8", errors="ignore") as f:
        next(f)
        for ln in f:
            parts = ln.rstrip("\n").split(",", 3)
            sc.append(float(parts[0]) if parts[0] else np.nan)
            ss.append(float(parts[1]) if parts[1] else np.nan)
            sf.append(float(parts[2]) if parts[2] else np.nan)
    return np.array(sc), np.array(ss), np.array(sf)


def bootstrap_auc(y, s, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, sb = y[idx], s[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, sb))
    aucs = np.array(aucs)
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def bootstrap_f1(y, s, thr, n_boot=N_BOOT, seed=SEED):
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
    rows = []

    for stem, (pred_csv, labels_path) in OWN_BRANCH_SOURCES.items():
        if not pred_csv.exists() or not labels_path.exists():
            print(f"[SKIP] {stem}: missing predictions or labels")
            continue
        sc, ss, sf = _load_own_scores(pred_csv)
        n = len(sc)
        y = load_labels_sidecar(str(labels_path), n_rows=n)
        if y is None:
            print(f"[SKIP] {stem}: label mismatch")
            continue
        for branch, s in [("content", sc), ("session", ss), ("fused", sf)]:
            mask = ~np.isnan(s)
            if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
                continue
            yv, sv = y[mask], s[mask]
            auc_mean, auc_lo, auc_hi = bootstrap_auc(yv, sv)
            thr = choose_threshold(sv, p=95.0)
            f1_mean, f1_lo, f1_hi = bootstrap_f1(yv, sv, thr)
            rows.append({"dataset": stem, "method": branch, "n": int(mask.sum()),
                         "auc_mean": auc_mean, "auc_lo": auc_lo, "auc_hi": auc_hi,
                         "f1_mean": f1_mean, "f1_lo": f1_lo, "f1_hi": f1_hi})
            print(f"{stem:24s} {branch:8s} n={mask.sum():6d}  "
                  f"AUC={auc_mean:.3f} [{auc_lo:.3f}, {auc_hi:.3f}]  "
                  f"F1={f1_mean:.3f} [{f1_lo:.3f}, {f1_hi:.3f}]")

    for kind in BASELINE_KINDS:
        for stem in OWN_BRANCH_SOURCES.keys():
            tag = "csic_eval" if stem == "csic_eval" else stem
            score_csv = EVAL_DIR / f"baseline_scores_{kind}_{tag}.csv"
            if not score_csv.exists():
                continue
            df = pd.read_csv(score_csv)
            y, s = df["y_true"].to_numpy(), df["score"].to_numpy()
            if len(np.unique(y)) < 2:
                continue
            auc_mean, auc_lo, auc_hi = bootstrap_auc(y, s)
            rows.append({"dataset": stem, "method": kind, "n": len(y),
                         "auc_mean": auc_mean, "auc_lo": auc_lo, "auc_hi": auc_hi,
                         "f1_mean": None, "f1_lo": None, "f1_hi": None})
            print(f"{stem:24s} {kind:8s} n={len(y):6d}  AUC={auc_mean:.3f} [{auc_lo:.3f}, {auc_hi:.3f}]")

    out_csv = EVAL_DIR / "bootstrap_ci.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n[OK] wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
