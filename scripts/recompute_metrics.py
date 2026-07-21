"""
Backfill metrics_<stem>.json (including the low-alert-budget operating-point
fields from src/pipeline/metrics_extra.py) from the already-cached
predictions_<stem>.csv, without rerunning model inference or reparsing logs.

Usage:
    python scripts/recompute_metrics.py
"""
import json
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
    return np.array(sc), np.array(ss), np.array(fu)


def main():
    for stem in CANONICAL_STEMS:
        mpath = EVAL_DIR / f"metrics_{stem}.json"
        pred_path = EVAL_DIR / f"predictions_{stem}.csv"
        if not mpath.exists() or not pred_path.exists():
            print(f"[SKIP] {stem}: missing metrics/predictions file")
            continue

        old = json.loads(mpath.read_text())
        sc, ss, fu = load_predictions(stem)
        n = len(sc)
        y_true = load_labels_sidecar(str(ROOT / LABELS[stem]), n_rows=n)
        if y_true is None:
            print(f"[SKIP] {stem}: labels unavailable/mismatched")
            continue

        metrics = {}
        for name, arr in (("content", sc), ("session", ss), ("fused", fu)):
            if np.isnan(arr).all():
                continue
            valid = ~np.isnan(arr)
            thr = choose_threshold(arr[valid], p=95.0)
            metrics[name] = compute_metrics(y_true[valid], arr[valid], thr)

        old["metrics"] = metrics
        mpath.write_text(json.dumps(old, indent=2))
        print(f"[OK] {stem}: recomputed metrics for branches {list(metrics.keys())}")


if __name__ == "__main__":
    main()
