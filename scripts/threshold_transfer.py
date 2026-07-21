"""
Cross-dataset threshold-transfer experiment (mirrors the peer paper's
train-on-source / freeze-threshold / apply-unchanged-to-targets design).

Source domain: CSIC 2010 (data/csic/csic_eval) - the only publicly-citable
benchmark among our datasets. Its benign-only rows' score distribution
(from the already-cached predictions_csic_eval.csv) defines a FIXED
threshold per branch, which is then applied UNCHANGED (no recalibration)
to three target datasets: Access Eval Mix 2000, Access Eval Small 500,
and Nginx JSON Eval 800.

Compares each target's self-calibrated (in-sample p95 percentile, as
already reported in metrics_<stem>.json) metrics against the transferred
(frozen source-domain threshold) metrics, to quantify the generalization
gap. AUC is threshold-independent, so it's reported once per branch (not
duplicated across self/transferred columns).

Usage:
    python scripts/threshold_transfer.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_file import compute_metrics, load_labels_sidecar

EVAL_DIR = ROOT / "artifacts" / "eval"

SOURCE_STEM = "csic_eval"
SOURCE_LABELS = "data/csic/csic_eval.log.labels.txt"

TARGETS = {
    "access_eval_mix_2000": "data/mixed/access_eval_mix_2000.log.labels.txt",
    "access_eval_small_500": "data/mixed/access_eval_small_500.log.labels.txt",
    "nginx_json_eval_800": "data/mixed/nginx_json_eval_800.log.labels.txt",
}

BRANCHES = ["content", "session", "fused"]


def load_predictions(stem):
    path = EVAL_DIR / f"predictions_{stem}.csv"
    sc, ss, fu = [], [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        next(f)
        for ln in f:
            parts = ln.rstrip("\n").split(",", 3)
            if len(parts) < 4:
                continue
            sc.append(float(parts[0]) if parts[0] != "" else np.nan)
            ss.append(float(parts[1]) if parts[1] != "" else np.nan)
            fu.append(float(parts[2]) if parts[2] != "" else np.nan)
    return {"content": np.array(sc), "session": np.array(ss), "fused": np.array(fu)}


def main():
    # --- Derive frozen source-domain thresholds from CSIC's BENIGN-ONLY rows ---
    src_scores = load_predictions(SOURCE_STEM)
    n_src = len(src_scores["content"])
    y_src = load_labels_sidecar(str(ROOT / SOURCE_LABELS), n_rows=n_src)
    if y_src is None:
        raise SystemExit("Could not load CSIC source labels.")
    benign_mask = (y_src == 0)

    tau = {}
    for branch in BRANCHES:
        arr = src_scores[branch]
        valid = benign_mask & ~np.isnan(arr)
        if valid.sum() == 0:
            tau[branch] = None
            continue
        tau[branch] = float(np.percentile(arr[valid], 95))
    print(f"[SOURCE] Frozen thresholds from {SOURCE_STEM} benign rows (n={benign_mask.sum()}): {tau}")

    rows = []
    for stem, labels_rel in TARGETS.items():
        tgt_scores = load_predictions(stem)
        n_tgt = len(tgt_scores["content"])
        y_tgt = load_labels_sidecar(str(ROOT / labels_rel), n_rows=n_tgt)
        if y_tgt is None:
            print(f"[SKIP] {stem}: labels unavailable/mismatched")
            continue

        # Self-calibrated numbers: reuse what's already in metrics_<stem>.json
        self_metrics = json.loads((EVAL_DIR / f"metrics_{stem}.json").read_text())["metrics"]

        for branch in BRANCHES:
            arr = tgt_scores[branch]
            valid = ~np.isnan(arr)
            if valid.sum() == 0 or tau[branch] is None:
                continue
            transferred = compute_metrics(y_tgt[valid], arr[valid], tau[branch])
            self_m = self_metrics.get(branch)
            if self_m is None:
                continue
            rows.append({
                "target_dataset": stem,
                "branch": branch,
                "auc": self_m["auc"],  # threshold-independent, same for both columns
                "self_f1": self_m["f1"], "self_precision": self_m["precision"],
                "self_recall": self_m["recall"], "self_fpr": self_m["fpr"],
                "transferred_f1": transferred["f1"], "transferred_precision": transferred["precision"],
                "transferred_recall": transferred["recall"], "transferred_fpr": transferred["fpr"],
                "generalization_gap_f1": self_m["f1"] - transferred["f1"],
            })
            print(f"{stem:24s} {branch:8s} self_f1={self_m['f1']:.4f}  transferred_f1={transferred['f1']:.4f}  "
                  f"gap={self_m['f1']-transferred['f1']:+.4f}")

    out_csv = EVAL_DIR / "threshold_transfer.csv"
    cols = ["target_dataset", "branch", "auc", "self_f1", "self_precision", "self_recall", "self_fpr",
            "transferred_f1", "transferred_precision", "transferred_recall", "transferred_fpr",
            "generalization_gap_f1"]
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")
    print(f"\n[OK] wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
