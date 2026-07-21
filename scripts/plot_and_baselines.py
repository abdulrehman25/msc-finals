"""
Plots & baselines for your web-log anomaly project.

Modes:
  agg                     -> aggregate BAR + LINE plots from artifacts/eval/metrics_*.json
  sweep --stem <stem>     -> threshold sweep (F1/Precision/Recall vs percentile) for one dataset
  sweep-all [--labels-root <dir>]
                          -> threshold sweep for ALL datasets that have labels
  rf --log <log> --labels <labels> [--trees 300]
                          -> train & plot RF baseline for one dataset
  rf-all --logs-root <dir> [--trees 300]
                          -> train & plot RF baseline for ALL (log, labels) pairs under logs-root

Outputs:
  artifacts/eval/plots/
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM

# project imports (editable install recommended)
from featurization.features import line_to_vector
from featurization.parse import parse_line

EVAL_DIR = Path("artifacts/eval")
PLOTS_DIR = EVAL_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------- helpers -----------------------

def _stem_from_metrics_path(p: Path) -> str:
    name = p.stem
    if name.startswith("metrics_"):
        return name[len("metrics_"):]
    return name

def load_all_metrics() -> pd.DataFrame:
    rows = []
    for mp in sorted(EVAL_DIR.glob("metrics_*.json")):
        try:
            data = json.load(open(mp, "r", encoding="utf-8"))
        except Exception:
            continue
        stem = _stem_from_metrics_path(mp)
        n_lines = data.get("n_lines")
        labels_used = data.get("labels_used", False)
        thrs = data.get("thresholds", {}) or {}
        M = data.get("metrics", {}) or {}

        for branch in ["fused","content","session"]:
            m = M.get(branch)
            if not m:
                rows.append({
                    "dataset": stem, "branch": branch, "labels_used": labels_used, "n_lines": n_lines,
                    "auc": np.nan, "prauc": np.nan, "f1": np.nan, "precision": np.nan, "recall": np.nan,
                    "fpr": np.nan, "tn": np.nan, "fp": np.nan, "fn": np.nan, "tp": np.nan,
                    "threshold": thrs.get(branch), "alert_rate": np.nan
                })
            else:
                rows.append({
                    "dataset": stem, "branch": branch, "labels_used": labels_used, "n_lines": n_lines,
                    "auc": m.get("auc"), "prauc": m.get("prauc"), "f1": m.get("f1"),
                    "precision": m.get("precision"), "recall": m.get("recall"), "fpr": m.get("fpr"),
                    "tn": m.get("tn"), "fp": m.get("fp"), "fn": m.get("fn"), "tp": m.get("tp"),
                    "threshold": m.get("threshold", thrs.get(branch)),
                    "alert_rate": m.get("alert_rate", np.nan)
                })
    return pd.DataFrame(rows)

def _grouped_bar(df: pd.DataFrame, metric: str, fname: str):
    if df.empty:
        return
    pivot = df.pivot_table(index="dataset", columns="branch", values=metric, aggfunc="first")
    # Ensure column order
    cols = [c for c in ["fused","content","session"] if c in pivot.columns]
    pivot = pivot[cols]
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_xlabel("Dataset")
    ax.set_ylabel(metric.upper())
    ax.set_title(metric.upper())
    ax.legend(title="Branch")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname, dpi=160)
    plt.close()

def _line_trend(df: pd.DataFrame, metric: str, fname: str):
    if df.empty:
        return
    pivot = df.pivot_table(index="dataset", columns="branch", values=metric, aggfunc="first")
    cols = [c for c in ["fused","content","session"] if c in pivot.columns]
    pivot = pivot[cols]
    ax = pivot.plot(kind="line", marker="o", figsize=(10,5))
    ax.set_xlabel("Dataset (sorted by name)")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} trend across datasets")
    ax.legend(title="Branch")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname, dpi=160)
    plt.close()

def _find_predictions(stem: str) -> Optional[Path]:
    p = EVAL_DIR / f"predictions_{stem}.csv"
    return p if p.exists() else None

def _find_labels_for_stem(stem: str, labels_root: Optional[Path]) -> Optional[Path]:
    # Try explicit root first (recursive). Canonical sidecar naming is
    # "<stem>.log.labels.txt"; ".labels.txt" (no ".log") is also accepted
    # for backward compatibility with stray/duplicate files.
    candidates: List[Path] = []
    if labels_root and labels_root.exists():
        for pattern in (f"{stem}.log.labels.txt", f"{stem}.labels.txt"):
            for p in labels_root.rglob(pattern):
                candidates.append(p)
    # Try eval dir (rare)
    candidates.append(EVAL_DIR / f"{stem}.log.labels.txt")
    candidates.append(EVAL_DIR / f"{stem}.labels.txt")
    for c in candidates:
        if c.exists():
            return c
    return None

def _load_scores_and_labels(pred_csv: Path, labels_path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    y_true = []
    with open(labels_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            try:
                lab = int(s.split(",")[-1])
                y_true.append(lab)
            except Exception:
                pass
    y_true = np.array(y_true, dtype=int)

    sc_c, sc_s, sc_f = [], [], []
    with open(pred_csv, "r", encoding="utf-8", errors="ignore") as f:
        header = next(f, None)
        for ln in f:
            parts = ln.rstrip("\n").split(",", 3)
            if len(parts) < 4:
                continue
            sc, ss, sf = parts[0].strip(), parts[1].strip(), parts[2].strip()
            sc_c.append(float(sc) if sc else np.nan)
            sc_s.append(float(ss) if ss else np.nan)
            sc_f.append(float(sf) if sf else np.nan)
    sc_c = np.array(sc_c, dtype=float)
    sc_s = np.array(sc_s, dtype=float)
    sc_f = np.array(sc_f, dtype=float)

    n = min(len(y_true), len(sc_f))
    y_true = y_true[:n]
    sc_c = sc_c[:n]
    sc_s = sc_s[:n]
    sc_f = sc_f[:n]

    out = {
        "fused": (y_true[~np.isnan(sc_f)], sc_f[~np.isnan(sc_f)]),
        "content": (y_true[~np.isnan(sc_c)], sc_c[~np.isnan(sc_c)]),
        "session": (y_true[~np.isnan(sc_s)], sc_s[~np.isnan(sc_s)]),
    }
    return out

def _rf_train_and_plot(X: np.ndarray, y: np.ndarray, n_trees: int, tag: str):
    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    split = int(0.8 * n)
    tr, te = idx[:split], idx[split:]
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

    if len(np.unique(ytr)) < 2:
        print(f"[RF] {tag}: skipped (training split has a single class -- dataset has no attack samples)")
        return None

    rf = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced"
    )
    rf.fit(Xtr, ytr)
    prob = rf.predict_proba(Xte)[:,1]
    yhat = (prob >= 0.5).astype(int)

    cm = confusion_matrix(yte, yhat, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    try: auc_roc = roc_auc_score(yte, prob)
    except: auc_roc = float("nan")
    try: pr_auc = average_precision_score(yte, prob)
    except: pr_auc = float("nan")
    P = precision_score(yte, yhat, zero_division=0)
    R = recall_score(yte, yhat, zero_division=0)
    F1 = f1_score(yte, yhat, zero_division=0)
    fpr_val = fp / (fp + tn) if (fp+tn)>0 else 0.0

    # ROC curve
    if len(np.unique(yte)) >= 2:
        fpr_arr, tpr_arr, _ = roc_curve(yte, prob)
        plt.figure(figsize=(7,5))
        plt.plot(fpr_arr, tpr_arr, label=f"AUC={auc_roc:.3f}")
        plt.plot([0,1],[0,1],'--')
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"RF • ROC • {tag}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"rf_roc_{tag}.png", dpi=160)
        plt.close()

        # PR curve
        prec, rec, _ = precision_recall_curve(yte, prob)
        plt.figure(figsize=(7,5))
        plt.plot(rec, prec, label=f"PR-AUC={pr_auc:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"RF • PR • {tag}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"rf_pr_{tag}.png", dpi=160)
        plt.close()

    # Confusion matrix
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"RF Confusion • {tag}\nF1={F1:.3f}, P={P:.3f}, R={R:.3f}, FPR={fpr_val:.3f}")
    plt.xticks([0,1], ["Pred 0", "Pred 1"])
    plt.yticks([0,1], ["True 0", "True 1"])
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"rf_confusion_{tag}.png", dpi=160)
    plt.close()

    # Feature importances (top 20)
    importances = rf.feature_importances_
    idxs = np.argsort(importances)[::-1][:20]
    plt.figure(figsize=(8,6))
    plt.bar(range(len(idxs)), importances[idxs])
    plt.xlabel("Top feature index")
    plt.ylabel("Importance")
    plt.title(f"RF Feature Importances • {tag} (Top 20)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"rf_feature_importances_{tag}.png", dpi=160)
    plt.close()

    print(f"[RF] {tag}: tn={tn} fp={fp} fn={fn} tp={tp} | AUC={auc_roc:.4f} PR-AUC={pr_auc:.4f} F1={F1:.4f} P={P:.4f} R={R:.4f}")

    metrics = {"auc": float(auc_roc), "prauc": float(pr_auc), "f1": float(F1),
               "precision": float(P), "recall": float(R), "fpr": float(fpr_val)}
    (EVAL_DIR / f"baseline_metrics_rf_{tag}.json").write_text(json.dumps(metrics, indent=2))
    return metrics

def _unsupervised_train_and_plot(X: np.ndarray, y: np.ndarray, tag: str, kind: str, max_train_benign: Optional[int] = None):
    """Shared driver for Isolation Forest / One-Class SVM baselines.

    Unlike the supervised RF baseline, these are fit on BENIGN-ONLY training
    rows (the correct comparison point against the benign-trained content-AE
    / session-LSTM-VAE branches). Anomaly threshold is the p95 percentile of
    the training-benign fold's own score, applied to the held-out test fold
    -- the same in-sample percentile convention used elsewhere in this repo.
    """
    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    split = int(0.8 * n)
    tr, te = idx[:split], idx[split:]
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

    Xtr_benign = Xtr[ytr == 0]
    if max_train_benign and len(Xtr_benign) > max_train_benign:
        sel = rng.choice(len(Xtr_benign), size=max_train_benign, replace=False)
        Xtr_benign = Xtr_benign[sel]

    if len(Xtr_benign) == 0:
        print(f"[{kind}] {tag}: no benign rows in training split, skipping")
        return None

    if kind == "iforest":
        model = IsolationForest(n_estimators=200, random_state=42)
        model.fit(Xtr_benign)
        score_train = -model.score_samples(Xtr_benign)  # higher = more anomalous
        score_test = -model.score_samples(Xte)
    elif kind == "ocsvm":
        model = OneClassSVM(kernel="rbf", nu=0.05)
        model.fit(Xtr_benign)
        score_train = -model.decision_function(Xtr_benign)
        score_test = -model.decision_function(Xte)
    else:
        raise ValueError(kind)

    thr = float(np.percentile(score_train, 95))
    yhat = (score_test > thr).astype(int)

    cm = confusion_matrix(yte, yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    try: auc_roc = roc_auc_score(yte, score_test)
    except Exception: auc_roc = float("nan")
    try: pr_auc = average_precision_score(yte, score_test)
    except Exception: pr_auc = float("nan")
    P = precision_score(yte, yhat, zero_division=0)
    R = recall_score(yte, yhat, zero_division=0)
    F1 = f1_score(yte, yhat, zero_division=0)
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    if len(np.unique(yte)) >= 2:
        fpr_arr, tpr_arr, _ = roc_curve(yte, score_test)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr_arr, tpr_arr, label=f"AUC={auc_roc:.3f}")
        plt.plot([0, 1], [0, 1], '--')
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"{kind.upper()} • ROC • {tag}")
        plt.legend(); plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{kind}_roc_{tag}.png", dpi=160)
        plt.close()

        prec, rec, _ = precision_recall_curve(yte, score_test)
        plt.figure(figsize=(7, 5))
        plt.plot(rec, prec, label=f"PR-AUC={pr_auc:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"{kind.upper()} • PR • {tag}")
        plt.legend(); plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{kind}_pr_{tag}.png", dpi=160)
        plt.close()

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{kind.upper()} Confusion • {tag}\nF1={F1:.3f}, P={P:.3f}, R={R:.3f}, FPR={fpr_val:.3f}")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{kind}_confusion_{tag}.png", dpi=160)
    plt.close()

    print(f"[{kind.upper()}] {tag}: benign_train={len(Xtr_benign)}/{len(Xtr)} tn={tn} fp={fp} fn={fn} tp={tp} "
          f"| AUC={auc_roc:.4f} PR-AUC={pr_auc:.4f} F1={F1:.4f} P={P:.4f} R={R:.4f}")

    metrics = {"auc": float(auc_roc), "prauc": float(pr_auc), "f1": float(F1),
               "precision": float(P), "recall": float(R), "fpr": float(fpr_val),
               "n_train_benign": int(len(Xtr_benign)), "n_train_total": int(len(Xtr))}
    (EVAL_DIR / f"baseline_metrics_{kind}_{tag}.json").write_text(json.dumps(metrics, indent=2))
    return metrics

# -------------------- modes --------------------

def do_agg():
    df = load_all_metrics()
    if df.empty:
        print("[ERR] No metrics_*.json found in artifacts/eval/")
        return
    df.sort_values(["dataset","branch"]).to_csv(EVAL_DIR / "summary_metrics.csv", index=False)
    print("[OK] wrote", EVAL_DIR / "summary_metrics.csv")

    # BAR charts
    for metric, fname in [
        ("auc","auc_bar.png"), ("prauc","prauc_bar.png"), ("f1","f1_bar.png"),
        ("precision","precision_bar.png"), ("recall","recall_bar.png"), ("fpr","fpr_bar.png"),
        ("alert_rate","alert_rate_bar.png"),
    ]:
        _grouped_bar(df, metric, fname)
        print("[OK] plot", fname)

    # LINE trends
    for metric, fname in [
        ("auc","auc_line.png"), ("prauc","prauc_line.png"),
        ("f1","f1_line.png"), ("precision","precision_line.png"), ("recall","recall_line.png")
    ]:
        _line_trend(df, metric, fname)
        print("[OK] plot", fname)

def _sweep_one(stem: str, labels_root: Optional[Path]):
    pred_csv = _find_predictions(stem)
    if not pred_csv:
        print(f"[WARN] [sweep] no predictions for {stem}")
        return False
    labels = _find_labels_for_stem(stem, labels_root)
    if not labels:
        print(f"[WARN] [sweep] no labels for {stem} (provide --labels-root)")
        return False
    d = _load_scores_and_labels(pred_csv, labels)
    ok_any = False
    for branch in ["fused","content","session"]:
        y, s = d[branch]
        if y.size == 0 or len(np.unique(y)) < 2:
            print(f"[WARN] [sweep] skip {stem}/{branch} (no/degenerate labels)")
            continue
        ps = np.linspace(50, 99, 50)  # percentiles
        f1s, precs, recs = [], [], []
        for p in ps:
            thr = np.percentile(s, p)
            yhat = (s > thr).astype(int)
            f1s.append(f1_score(y, yhat, zero_division=0))
            precs.append(precision_score(y, yhat, zero_division=0))
            recs.append(recall_score(y, yhat, zero_division=0))
        # F1
        plt.figure(figsize=(7,5))
        plt.plot(ps, f1s, marker="o")
        plt.xlabel("Percentile threshold")
        plt.ylabel("F1")
        plt.title(f"F1 vs threshold • {branch} • {stem}")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"sweep_f1_{branch}_{stem}.png", dpi=160)
        plt.close()
        # Precision
        plt.figure(figsize=(7,5))
        plt.plot(ps, precs, marker="o")
        plt.xlabel("Percentile threshold")
        plt.ylabel("Precision")
        plt.title(f"Precision vs threshold • {branch} • {stem}")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"sweep_precision_{branch}_{stem}.png", dpi=160)
        plt.close()
        # Recall
        plt.figure(figsize=(7,5))
        plt.plot(ps, recs, marker="o")
        plt.xlabel("Percentile threshold")
        plt.ylabel("Recall")
        plt.title(f"Recall vs threshold • {branch} • {stem}")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"sweep_recall_{branch}_{stem}.png", dpi=160)
        plt.close()
        ok_any = True
    return ok_any

def do_sweep(stem: str, labels_root: Optional[Path]):
    ok = _sweep_one(stem, labels_root)
    if ok:
        print(f"[OK] sweep plots done for {stem}")

def do_sweep_all(labels_root: Optional[Path]):
    df = load_all_metrics()
    if df.empty:
        print("[ERR] No metrics_*.json in artifacts/eval/")
        return
    stems = sorted(df["dataset"].unique())
    count_ok = 0
    for stem in stems:
        if _sweep_one(stem, labels_root):
            count_ok += 1
    print(f"[DONE] sweep-all finished. Successful: {count_ok}/{len(stems)} stems")

def _load_log_with_labels(log_path: Path, labels_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f, \
         open(labels_path, "r", encoding="utf-8", errors="ignore") as g:
        labels = [int(ln.strip().split(",")[-1]) for ln in g if ln.strip()]
        i = 0
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            vec, _ = line_to_vector(ln)
            if vec is None: continue
            if i >= len(labels): break
            X.append(vec)
            y.append(labels[i])
            i += 1
    return np.array(X, dtype=float), np.array(y, dtype=int)

def do_rf(log_path: Path, labels_path: Path, n_trees: int):
    X, y = _load_log_with_labels(log_path, labels_path)
    if X.size == 0 or y.size == 0:
        print("[ERR] [rf] no data after parsing/aligning")
        return
    tag = Path(log_path).stem
    _rf_train_and_plot(X, y, n_trees, tag)

def do_rf_all(logs_root: Path, n_trees: int):
    """Find every *.log with sibling *.labels.txt under logs_root (recursive) and run RF."""
    pairs = []
    for p in logs_root.rglob("*.log"):
        lab = p.with_suffix(p.suffix + ".labels.txt")  # e.g. .log.labels.txt
        if lab.exists():
            pairs.append((p, lab))
        else:
            # also check <stem>.labels.txt (without .log)
            alt = p.with_suffix("").with_suffix(".labels.txt")
            if alt.exists():
                pairs.append((p, alt))
    if not pairs:
        print(f"[ERR] [rf-all] no (log,labels) pairs found under {logs_root}")
        return
    for log_path, labels_path in pairs:
        print(f"[rf-all] {log_path.name}  +  {labels_path.name}")
        X, y = _load_log_with_labels(log_path, labels_path)
        if X.size == 0 or y.size == 0:
            print("  [WARN] skipped (no data after parsing/aligning)")
            continue
        tag = Path(log_path).stem
        _rf_train_and_plot(X, y, n_trees, tag)
    print("[DONE] rf-all finished.")

def _find_log_label_pairs(logs_root: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for p in logs_root.rglob("*.log"):
        lab = p.with_suffix(p.suffix + ".labels.txt")
        if lab.exists():
            pairs.append((p, lab))
        else:
            alt = p.with_suffix("").with_suffix(".labels.txt")
            if alt.exists():
                pairs.append((p, alt))
    return pairs

def do_unsupervised(log_path: Path, labels_path: Path, kind: str, max_train_benign: Optional[int]):
    X, y = _load_log_with_labels(log_path, labels_path)
    if X.size == 0 or y.size == 0:
        print(f"[ERR] [{kind}] no data after parsing/aligning")
        return
    tag = Path(log_path).stem
    _unsupervised_train_and_plot(X, y, tag, kind, max_train_benign)

def do_unsupervised_all(logs_root: Path, kind: str, max_train_benign: Optional[int]):
    pairs = _find_log_label_pairs(logs_root)
    if not pairs:
        print(f"[ERR] [{kind}-all] no (log,labels) pairs found under {logs_root}")
        return
    for log_path, labels_path in pairs:
        print(f"[{kind}-all] {log_path.name}  +  {labels_path.name}")
        X, y = _load_log_with_labels(log_path, labels_path)
        if X.size == 0 or y.size == 0:
            print("  [WARN] skipped (no data after parsing/aligning)")
            continue
        tag = Path(log_path).stem
        _unsupervised_train_and_plot(X, y, tag, kind, max_train_benign)
    print(f"[DONE] {kind}-all finished.")

def do_baseline_comparison():
    """Compile content-AE / session-LSTM-VAE / fused (from metrics_*.json) and
    RF / IsolationForest / OneClassSVM (from baseline_metrics_*_<tag>.json)
    into one comparison CSV."""
    df = load_all_metrics()
    if df.empty:
        print("[ERR] No metrics_*.json found in artifacts/eval/")
        return
    # Exclude the orphaned access_long_session_300 artifact: it has no label
    # sidecar anywhere in the repo and predates the current metrics schema.
    stems = sorted(s for s in df["dataset"].unique() if s != "access_long_session_300")
    rows = []
    for stem in stems:
        row = {"dataset": stem}
        for branch in ["content", "session", "fused"]:
            sub = df[(df["dataset"] == stem) & (df["branch"] == branch)]
            row[f"{branch}_auc"] = sub["auc"].iloc[0] if not sub.empty else None
            row[f"{branch}_f1"] = sub["f1"].iloc[0] if not sub.empty else None
        for kind in ["rf", "iforest", "ocsvm"]:
            bpath = EVAL_DIR / f"baseline_metrics_{kind}_{stem}.json"
            if bpath.exists():
                bm = json.loads(bpath.read_text())
                row[f"{kind}_auc"] = bm.get("auc")
                row[f"{kind}_f1"] = bm.get("f1")
                row[f"{kind}_fpr"] = bm.get("fpr")
            else:
                row[f"{kind}_auc"] = row[f"{kind}_f1"] = row[f"{kind}_fpr"] = None
        rows.append(row)

    cols = (["dataset", "content_auc", "content_f1", "session_auc", "session_f1",
             "fused_auc", "fused_f1"] +
            [f"{k}_{m}" for k in ["rf", "iforest", "ocsvm"] for m in ["auc", "f1", "fpr"]])
    out_csv = EVAL_DIR / "baseline_comparison.csv"
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(c) is None else str(r[c]) for c in cols) + "\n")
    print(f"[OK] wrote {out_csv} ({len(rows)} datasets)")

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    agg = sub.add_parser("agg", help="Aggregate bar + line plots from previous evals (metrics_*.json)")

    sweep = sub.add_parser("sweep", help="Threshold sweep line charts for one dataset stem")
    sweep.add_argument("--stem", required=True, help="dataset stem e.g., access_eval_mix_2000")
    sweep.add_argument("--labels-root", type=str, default=None, help="root to search for <stem>.labels.txt")

    sweep_all = sub.add_parser("sweep-all", help="Threshold sweep for ALL datasets discovered in artifacts/eval/")
    sweep_all.add_argument("--labels-root", type=str, default=None, help="root to search for labels (recursive)")

    rf = sub.add_parser("rf", help="Train & plot Random Forest on a log+labels pair")
    rf.add_argument("--log", required=True)
    rf.add_argument("--labels", required=True)
    rf.add_argument("--trees", type=int, default=300)

    rf_all = sub.add_parser("rf-all", help="Train & plot RF on ALL (log,labels) pairs under a folder (recursive)")
    rf_all.add_argument("--logs-root", required=True, help="folder containing logs and labels (recursive)")
    rf_all.add_argument("--trees", type=int, default=300)

    for kind in ("iforest", "ocsvm"):
        p1 = sub.add_parser(kind, help=f"Train & plot {kind} (unsupervised, benign-only) baseline on a log+labels pair")
        p1.add_argument("--log", required=True)
        p1.add_argument("--labels", required=True)
        p1.add_argument("--max-train-benign", type=int, default=None, help="Cap benign training rows (useful for large datasets)")

        p2 = sub.add_parser(f"{kind}-all", help=f"Train & plot {kind} on ALL (log,labels) pairs under a folder (recursive)")
        p2.add_argument("--logs-root", required=True)
        p2.add_argument("--max-train-benign", type=int, default=None)

    sub.add_parser("baseline-comparison", help="Compile content/session/fused + RF/IForest/OCSVM into baseline_comparison.csv")

    args = ap.parse_args()

    if args.cmd == "agg":
        do_agg()
    elif args.cmd == "sweep":
        root = Path(args.labels_root) if args.labels_root else None
        do_sweep(args.stem, root)
    elif args.cmd == "sweep-all":
        root = Path(args.labels_root) if args.labels_root else None
        do_sweep_all(root)
    elif args.cmd == "rf":
        do_rf(Path(args.log), Path(args.labels), args.trees)
    elif args.cmd == "rf-all":
        do_rf_all(Path(args.logs_root), args.trees)
    elif args.cmd in ("iforest", "ocsvm"):
        do_unsupervised(Path(args.log), Path(args.labels), args.cmd, args.max_train_benign)
    elif args.cmd in ("iforest-all", "ocsvm-all"):
        do_unsupervised_all(Path(args.logs_root), args.cmd.replace("-all", ""), args.max_train_benign)
    elif args.cmd == "baseline-comparison":
        do_baseline_comparison()

if __name__ == "__main__":
    main()
