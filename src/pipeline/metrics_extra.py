"""
Low-alert-budget operating-point metrics: F1 achieved when the detector is
only allowed to alert on a small, fixed fraction of traffic (1%, 0.1%).

These fields (f1_at_1pct, thr_at_1pct, f1_at_0_1pct, thr_at_0_1pct,
alerts_per_million) used to be present in some of this project's older
metrics_*.json artifacts and are still referenced by src/api/main.py's
dashboard, but the evaluator that produced them was lost. This restores
the computation so it can be reapplied consistently.
"""
import numpy as np
from sklearn.metrics import f1_score, roc_curve


def _rate_key(rate: float) -> str:
    # 0.01 -> "1", 0.001 -> "0_1" (matches legacy field naming convention)
    pct = rate * 100
    s = f"{pct:g}"
    return s.replace(".", "_")


def compute_low_alert_metrics(y_true, y_score, target_rates=(0.01, 0.001)):
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    n = min(len(y_true), len(y_score))
    y_true = y_true[:n]
    y_score = y_score[:n]

    out = {}
    for rate in target_rates:
        thr = float(np.percentile(y_score, 100 * (1 - rate)))
        y_pred = (y_score > thr).astype(int)
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        key = _rate_key(rate)
        out[f"f1_at_{key}pct"] = f1
        out[f"thr_at_{key}pct"] = thr

    # alerts_per_million: alert rate at the primary (p95) operating threshold, scaled.
    thr95 = float(np.percentile(y_score, 95))
    alert_rate = float((y_score > thr95).mean())
    out["alerts_per_million"] = alert_rate * 1_000_000

    return out


def compute_operating_points(y_true, y_score, target_fprs=(0.01, 0.05, 0.1), target_recalls=(0.5, 0.8)):
    """Threshold-independent operating-point summary: recall achievable at a
    fixed false-positive-rate budget, and the false-positive rate required to
    reach a fixed recall. Lets methods be compared without assuming they were
    thresholded the same way (unlike a single F1/FPR at one chosen cutoff)."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    n = min(len(y_true), len(y_score))
    y_true = y_true[:n]
    y_score = y_score[:n]

    if np.unique(y_true).size != 2:
        return {"recall_at_fpr": {}, "fpr_at_recall": {}}

    fpr, tpr, _ = roc_curve(y_true, y_score)

    recall_at_fpr = {str(f): float(np.interp(f, fpr, tpr)) for f in target_fprs}
    fpr_at_recall = {str(r): float(np.interp(r, tpr, fpr)) for r in target_recalls}

    return {"recall_at_fpr": recall_at_fpr, "fpr_at_recall": fpr_at_recall}
