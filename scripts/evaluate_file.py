import os, sys
from pathlib import Path
# project root = folder that contains "src" and "scripts"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse, json, os, math
from pathlib import Path
from collections import deque
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
import math

import torch, joblib
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, confusion_matrix
from src.featurization.features import line_to_vector
from src.models.content_autoencoder import ContentAE
from src.models.session_lstm_vae import SessionLSTMVAE
from src.pipeline.metrics_extra import compute_low_alert_metrics, compute_operating_points

def _norm_fields(cfg):
    return {
        "train_percentiles": cfg.get("train_score_percentiles"),
        "train_mean": cfg.get("train_score_mean"),
        "train_std": cfg.get("train_score_std"),
    }

def load_content_bundle(content_dir="artifacts/content_v2"):
    p = Path(content_dir)
    if not p.exists(): return None
    scaler = joblib.load(p/"scaler.joblib")
    cfg = json.load(open(p/"config.json"))
    model = ContentAE(input_dim=cfg["input_dim"])
    model.load_state_dict(torch.load(p/"model.pt", map_location="cpu"))
    model.eval()
    return {"scaler":scaler, "model":model, "thr":float(cfg["threshold"]), **_norm_fields(cfg)}

def load_session_bundle(session_dir="artifacts/session"):
    p = Path(session_dir)
    if not p.exists(): return None
    scaler = joblib.load(p/"scaler.joblib")
    cfg = json.load(open(p/"config.json"))
    model = SessionLSTMVAE(input_dim=cfg["input_dim"])
    model.load_state_dict(torch.load(p/"model.pt", map_location="cpu"))
    model.eval()
    return {"scaler":scaler, "model":model, "thr":float(cfg["threshold"]), "window":int(cfg.get("window",20)),
            **_norm_fields(cfg)}

def normalize_score(score, bundle, method):
    """Map a raw reconstruction-error score onto a comparable scale before
    fusion, using only statistics of the branch's own training distribution
    (never the eval set's), so attack-heavy eval sets can't bias the mapping
    -- the same pitfall the threshold-transfer analysis found for thresholds."""
    if score is None or bundle is None or method == "none":
        return score
    if method == "percentile":
        pcts = bundle.get("train_percentiles")
        if not pcts:
            thr = bundle.get("thr")
            return score / thr if thr else score
        keys = sorted(pcts.keys(), key=lambda k: float(k))
        score_vals = [pcts[k] for k in keys]
        frac_vals = [float(k) / 100.0 for k in keys]
        return float(np.interp(score, score_vals, frac_vals))
    if method == "zscore":
        mean, std = bundle.get("train_mean"), bundle.get("train_std")
        if mean is None or std is None:
            return score
        return (score - mean) / std if std else (score - mean)
    raise ValueError(f"unknown fusion-norm: {method}")

def score_content(bundle, vec):
    if bundle is None or vec is None: return None
    Xs = bundle["scaler"].transform(vec.reshape(1,-1))
    with torch.no_grad():
        recon = bundle["model"](torch.tensor(Xs, dtype=torch.float32)).numpy()
    err = float(np.mean((Xs - recon)**2))
    return err

def score_session(bundle, vec):
    """Naive offline scorer: repeats a single line's vector to fill the window
    (no rolling buffer). Kept for backward-compat / reproducing old numbers.
    Prefer score_session_stateful for new runs (see --session-mode)."""
    if bundle is None or vec is None: return None
    win = bundle["window"]
    Xs = bundle["scaler"].transform(vec.reshape(1,-1))
    seq = np.repeat(Xs, win, axis=0).reshape(1, win, -1)
    with torch.no_grad():
        recon, _, _ = bundle["model"](torch.tensor(seq, dtype=torch.float32))
        err = float(np.mean((seq - recon.numpy())**2))
    return err

# Backward-compat alias, matches the plan's naming (score_session_naive).
score_session_naive = score_session

def score_session_stateful(bundle, key, vec, buffers):
    """Real per-(ip,ua) rolling-buffer scorer, matching src/api/main.py's
    SESSION_BUFFERS semantics: only scores once a full window of that
    session's own requests has accumulated (returns None until then)."""
    if bundle is None or vec is None:
        return None
    win = bundle["window"]
    Xs = bundle["scaler"].transform(vec.reshape(1, -1))[0]
    buf = buffers.setdefault(key, deque(maxlen=win))
    buf.append(Xs)
    if len(buf) < win:
        return None
    seq = np.stack(buf, axis=0).reshape(1, win, -1)
    with torch.no_grad():
        recon, _, _ = bundle["model"](torch.tensor(seq, dtype=torch.float32))
        err = float(np.mean((seq - recon.numpy())**2))
    return err

def load_labels_sidecar(labels_path, n_rows):
    # labels file is CSV with one integer per line (0/1) or two columns line,label
    # Gracefully accept plain list or "line,label"
    y = []
    if not labels_path or not os.path.exists(labels_path):
        return None
    with open(labels_path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip()
            if not s: continue
            parts = s.split(",")
            try:
                if len(parts)==1:
                    y.append(int(parts[0]))
                else:
                    y.append(int(parts[-1]))
            except:
                pass
    if len(y) != n_rows:
        print(f"[WARN] Labels len={len(y)} differs from lines={n_rows}. Ignoring labels.")
        return None
    return np.array(y, dtype=int)

def choose_threshold(scores, method="percentile", p=95.0):
    if method=="percentile":
        return float(np.percentile(scores, p))
    # You can add EVT here later.
    return float(np.percentile(scores, 95.0))

def compute_metrics(y_true, y_score, thr):
    """
    y_true  : 1-D array of {0,1}
    y_score : 1-D float anomaly scores
    thr     : float threshold; if nan/inf, uses p95 of y_score
    Returns: dict with auc, prauc, f1, precision, recall, fpr, tn/fp/fn/tp, threshold, alert_rate
    """
    # 1) shape & dtype safety
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    n = min(len(y_true), len(y_score))
    y_true = y_true[:n]
    y_score = y_score[:n]

    # 2) choose threshold if bad
    if thr is None or (isinstance(thr, float) and (math.isnan(thr) or math.isinf(thr))):
        thr = float(np.percentile(y_score, 95))

    # 3) build predictions from threshold
    y_pred = (y_score > thr).astype(int)

    # 4) guard single-class labels for AUC/PR-AUC
    has_both = np.unique(y_true).size == 2 and (y_true == 1).any() and (y_true == 0).any()
    if has_both:
        try:
            auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            auc = None
        try:
            prauc = float(average_precision_score(y_true, y_score))
        except Exception:
            prauc = None
    else:
        auc = None
        prauc = None

    # 5) thresholded metrics
    P = float(precision_score(y_true, y_pred, zero_division=0))
    R = float(recall_score(y_true, y_pred, zero_division=0))
    F1 = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    alert_rate = float((y_pred == 1).mean()) if len(y_pred) else 0.0

    result = {
        "auc": auc,
        "prauc": prauc,
        "precision": P,
        "recall": R,
        "f1": F1,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fpr": fpr,
        "threshold": float(thr),
        "alert_rate": alert_rate,
    }
    result.update(compute_low_alert_metrics(y_true, y_score))
    result["operating_points"] = compute_operating_points(y_true, y_score)
    return result

def _fmt4(x):
    """Format floats to 4dp; show 'n/a' for None/NaN/Inf."""
    if x is None:
        return "n/a"
    try:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "n/a"
        return f"{x:.4f}"
    except Exception:
        return str(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to access log")
    ap.add_argument("--labels", help="Optional labels file (0/1 per line or CSV with label in last column)")
    ap.add_argument("--outdir", default="artifacts/eval", help="Where to write predictions/metrics")
    ap.add_argument("--fuse-weights", default="0.5,0.5", help="content,session weights for fusion")
    ap.add_argument("--thr-percentile", type=float, default=95.0, help="percentile threshold fallback")
    ap.add_argument("--content-dir", default="artifacts/content_v2",
                     help="Where the content-branch model bundle lives. Default is the methodologically-"
                          "corrected v2 model (trained on normal-only NASA HTTP rows); pass 'artifacts/content' "
                          "to reproduce legacy numbers.")
    ap.add_argument("--session-dir", default="artifacts/session_v3",
                     help="Where the session-branch model bundle lives. Default is the methodologically-"
                          "corrected v3 model (real (ip,ua) sessions, trained on the CSIC held-out split); "
                          "pass 'artifacts/session' + --session-mode naive to reproduce legacy v1 numbers.")
    ap.add_argument("--session-mode", choices=["naive", "stateful"], default="stateful",
                     help="'stateful' = real per-(ip,ua) rolling buffer (matches training/serving); "
                          "'naive' = repeat-one-line-to-window (old behavior, for reproducing legacy numbers)")
    ap.add_argument("--fusion-norm", choices=["none", "percentile", "zscore"], default="none",
                     help="Normalize each branch's score against its own training-score distribution "
                          "before fusing, so branches on different numeric scales don't let one branch "
                          "dominate the weighted average. 'none' reproduces legacy raw-score fusion.")
    ap.add_argument("--out-stem", default=None,
                     help="Override the output filename stem (default: derived from --log). Used to write "
                          "e.g. predictions_csic_eval.csv from a held-out log file that has a different name.")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    cb = load_content_bundle(args.content_dir)
    sb = load_session_bundle(args.session_dir)
    if cb is None and sb is None:
        raise SystemExit("No trained models found under artifacts/. Train first.")

    w_content, w_session = [float(x) for x in args.fuse_weights.split(",")]

    # Score all lines
    session_buffers = {}
    lines, s_content, s_session, fused, fused_norm = [], [], [], [], []
    with open(args.log, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip()
            if not s: continue
            vec, meta = line_to_vector(s)
            sc = score_content(cb, vec) if cb else None
            if sb and args.session_mode == "stateful":
                key = (meta.get("ip", ""), meta.get("ua", ""))
                ss = score_session_stateful(sb, key, vec, session_buffers)
            elif sb:
                ss = score_session(sb, vec)
            else:
                ss = None
            if sc is None and ss is None:
                continue
            s_content.append(sc if sc is not None else np.nan)
            s_session.append(ss if ss is not None else np.nan)
            # raw fusion (legacy, unweighted-scale average)
            num, den = 0.0, 0.0
            if sc is not None: num += w_content*sc; den += w_content
            if ss is not None: num += w_session*ss; den += w_session
            fused.append(num/den if den>0 else (sc if sc is not None else ss))
            # normalized fusion: each branch mapped onto its own training-score
            # scale first, so a wide-scale branch can't numerically dominate
            scn = normalize_score(sc, cb, args.fusion_norm) if sc is not None else None
            ssn = normalize_score(ss, sb, args.fusion_norm) if ss is not None else None
            numn, denn = 0.0, 0.0
            if scn is not None: numn += w_content*scn; denn += w_content
            if ssn is not None: numn += w_session*ssn; denn += w_session
            fused_norm.append(numn/denn if denn>0 else (scn if scn is not None else ssn))
            lines.append(s)

    n = len(lines)
    if n==0:
        raise SystemExit("No parsable lines found.")
    y_true = load_labels_sidecar(args.labels, n_rows=n)  # can be None (unsupervised)

    # thresholds
    sc_arr = np.array(s_content)
    ss_arr = np.array(s_session)
    fu_arr = np.array(fused)
    fun_arr = np.array(fused_norm)

    # Replace NaNs where a branch is missing (won’t be used in metrics if NaNs and no labels)
    sc_valid = sc_arr[~np.isnan(sc_arr)]
    ss_valid = ss_arr[~np.isnan(ss_arr)]
    fu_valid = fu_arr[~np.isnan(fu_arr)]
    fun_valid = fun_arr[~np.isnan(fun_arr)]

    thr_c = choose_threshold(sc_valid, p=args.thr_percentile) if sc_valid.size>0 else None
    thr_s = choose_threshold(ss_valid, p=args.thr_percentile) if ss_valid.size>0 else None
    thr_f = choose_threshold(fu_valid, p=args.thr_percentile) if fu_valid.size>0 else None
    thr_fn = choose_threshold(fun_valid, p=args.thr_percentile) if fun_valid.size>0 else None

    # Save predictions CSV
    out_stem = args.out_stem or Path(args.log).stem
    pred_path = Path(args.outdir)/("predictions_" + out_stem + ".csv")
    with open(pred_path, "w", encoding="utf-8") as out:
        out.write("score_content,score_session,score_fused,score_fused_norm,line\n")
        for a,b,c,d,l in zip(s_content, s_session, fused, fused_norm, lines):
            out.write(f"{'' if math.isnan(a) else a},{'' if math.isnan(b) else b},{c},{d},{l}\n")

    # If labels provided: compute metrics per-branch + fused (raw and normalized)
    metrics = {}
    if y_true is not None:
        if thr_c is not None:
            metrics["content"] = compute_metrics(y_true, np.nan_to_num(sc_arr, nan=-1e9), thr_c)
        if thr_s is not None:
            metrics["session"] = compute_metrics(y_true, np.nan_to_num(ss_arr, nan=-1e9), thr_s)
        if thr_f is not None:
            metrics["fused"] = compute_metrics(y_true, np.nan_to_num(fu_arr, nan=-1e9), thr_f)
        if args.fusion_norm != "none" and thr_fn is not None:
            metrics["fused_norm"] = compute_metrics(y_true, np.nan_to_num(fun_arr, nan=-1e9), thr_fn)

    # Write metrics.json
    mpath = Path(args.outdir)/("metrics_" + out_stem + ".json")
    json.dump({
        "n_lines": n,
        "labels_used": y_true is not None,
        "threshold_percentile": args.thr_percentile,
        "fusion_norm": args.fusion_norm,
        "thresholds": {"content":thr_c, "session":thr_s, "fused":thr_f, "fused_norm":thr_fn},
        "metrics": metrics
    }, open(mpath,"w"), indent=2)

    print(f"[OK] wrote {pred_path}")
    print(f"[OK] wrote {mpath}")
    if metrics:
        for k, v in metrics.items():
            print(f"== {k.upper()} ==")
            print(
                "AUC={auc}  PR-AUC={prauc}  F1={f1}  P={p}  R={r}  FPR={fpr}".format(
                    auc=_fmt4(v["auc"]),
                    prauc=_fmt4(v.get("prauc") or v.get("pr_auc")),
                    f1=_fmt4(v["f1"]),
                    p=_fmt4(v["precision"]),
                    r=_fmt4(v["recall"]),
                    fpr=_fmt4(v["fpr"]),
                )
            )
            print(f"TN={v['tn']} FP={v['fp']}  FN={v['fn']} TP={v['tp']}  thr={_fmt4(v['threshold'])}")

if __name__ == "__main__":
    main()
