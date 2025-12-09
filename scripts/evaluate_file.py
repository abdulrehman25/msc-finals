import os, sys
from pathlib import Path
# project root = folder that contains "src" and "scripts"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse, json, os, math
from pathlib import Path
import numpy as np
import torch, joblib
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, confusion_matrix
from src.featurization.features import line_to_vector
from src.models.content_autoencoder import ContentAE
from src.models.session_lstm_vae import SessionLSTMVAE

def load_content_bundle():
    p = Path("artifacts/content")
    if not p.exists(): return None
    scaler = joblib.load(p/"scaler.joblib")
    cfg = json.load(open(p/"config.json"))
    model = ContentAE(input_dim=cfg["input_dim"])
    model.load_state_dict(torch.load(p/"model.pt", map_location="cpu"))
    model.eval()
    return {"scaler":scaler, "model":model, "thr":float(cfg["threshold"])}

def load_session_bundle():
    p = Path("artifacts/session")
    if not p.exists(): return None
    scaler = joblib.load(p/"scaler.joblib")
    cfg = json.load(open(p/"config.json"))
    model = SessionLSTMVAE(input_dim=cfg["input_dim"])
    model.load_state_dict(torch.load(p/"model.pt", map_location="cpu"))
    model.eval()
    return {"scaler":scaler, "model":model, "thr":float(cfg["threshold"]), "window":int(cfg.get("window",20))}

def score_content(bundle, vec):
    if bundle is None or vec is None: return None
    Xs = bundle["scaler"].transform(vec.reshape(1,-1))
    with torch.no_grad():
        recon = bundle["model"](torch.tensor(Xs, dtype=torch.float32)).numpy()
    err = float(np.mean((Xs - recon)**2))
    return err

def score_session(bundle, vec):
    if bundle is None or vec is None: return None
    # offline evaluator: repeat to window (no rolling buffer). This matches API’s pre-buffer behavior.
    win = bundle["window"]
    Xs = bundle["scaler"].transform(vec.reshape(1,-1))
    seq = np.repeat(Xs, win, axis=0).reshape(1, win, -1)
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
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, confusion_matrix

    y_pred = (y_score > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    uniq = np.unique(y_true)
    if len(uniq) == 1:
        auc = float("nan")
        prauc = float("nan")
    else:
        try: auc = float(roc_auc_score(y_true, y_score))
        except: auc = float("nan")
        try: prauc = float(average_precision_score(y_true, y_score))
        except: prauc = float("nan")

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))

    alert_rate = float((y_pred == 1).mean())
    alerts_per_million = float(alert_rate * 1_000_000)

    def f1_at_alert_rate(target_rate):
        if len(y_score) == 0: return float("nan"), float("nan")
        q = max(0.0, min(1.0, 1.0 - target_rate))
        t = float(np.quantile(y_score, q))
        yp = (y_score > t).astype(int)
        return float(f1_score(y_true, yp, zero_division=0)), t

    f1_at_1pct, thr_1pct = f1_at_alert_rate(0.01)
    f1_at_0_1pct, thr_0_1pct = f1_at_alert_rate(0.001)

    out = {
        "auc": auc,
        "prauc": prauc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "threshold": float(thr),
        "alert_rate": alert_rate,
        "alerts_per_million": alerts_per_million,
        "f1_at_1pct": f1_at_1pct,
        "thr_at_1pct": thr_1pct,
        "f1_at_0_1pct": f1_at_0_1pct,
        "thr_at_0_1pct": thr_0_1pct,
        "cm": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }
    # keep legacy flat keys so print/UI that expect them won’t break
    out["tn"], out["fp"], out["fn"], out["tp"] = int(tn), int(fp), int(fn), int(tp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to access log")
    ap.add_argument("--labels", help="Optional labels file (0/1 per line or CSV with label in last column)")
    ap.add_argument("--outdir", default="artifacts/eval", help="Where to write predictions/metrics")
    ap.add_argument("--fuse-weights", default="0.5,0.5", help="content,session weights for fusion")
    ap.add_argument("--thr-percentile", type=float, default=95.0, help="percentile threshold fallback")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    cb = load_content_bundle()
    sb = load_session_bundle()
    if cb is None and sb is None:
        raise SystemExit("No trained models found under artifacts/. Train first.")

    w_content, w_session = [float(x) for x in args.fuse_weights.split(",")]

    # Score all lines
    lines, s_content, s_session, fused = [], [], [], []
    with open(args.log, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip()
            if not s: continue
            vec, _ = line_to_vector(s)
            sc = score_content(cb, vec) if cb else None
            ss = score_session(sb, vec) if sb else None
            if sc is None and ss is None:
                continue
            s_content.append(sc if sc is not None else np.nan)
            s_session.append(ss if ss is not None else np.nan)
            # fusion over available scores
            num, den = 0.0, 0.0
            if sc is not None: num += w_content*sc; den += w_content
            if ss is not None: num += w_session*ss; den += w_session
            fused.append(num/den if den>0 else (sc if sc is not None else ss))
            lines.append(s)

    n = len(lines)
    if n==0:
        raise SystemExit("No parsable lines found.")
    y_true = load_labels_sidecar(args.labels, n_rows=n)  # can be None (unsupervised)

    # thresholds
    sc_arr = np.array(s_content)
    ss_arr = np.array(s_session)
    fu_arr = np.array(fused)

    # Replace NaNs where a branch is missing (won’t be used in metrics if NaNs and no labels)
    sc_valid = sc_arr[~np.isnan(sc_arr)]
    ss_valid = ss_arr[~np.isnan(ss_arr)]
    fu_valid = fu_arr[~np.isnan(fu_arr)]

    thr_c = choose_threshold(sc_valid, p=args.thr_percentile) if sc_valid.size>0 else None
    thr_s = choose_threshold(ss_valid, p=args.thr_percentile) if ss_valid.size>0 else None
    thr_f = choose_threshold(fu_valid, p=args.thr_percentile) if fu_valid.size>0 else None

    # Save predictions CSV
    pred_path = Path(args.outdir)/("predictions_" + Path(args.log).stem + ".csv")
    with open(pred_path, "w", encoding="utf-8") as out:
        out.write("score_content,score_session,score_fused,line\n")
        for a,b,c,l in zip(s_content, s_session, fused, lines):
            out.write(f"{'' if math.isnan(a) else a},{'' if math.isnan(b) else b},{c},{l}\n")

    # If labels provided: compute metrics per-branch + fused
    metrics = {}
    if y_true is not None:
        if thr_c is not None:
            metrics["content"] = compute_metrics(y_true, np.nan_to_num(sc_arr, nan=-1e9), thr_c)
        if thr_s is not None:
            metrics["session"] = compute_metrics(y_true, np.nan_to_num(ss_arr, nan=-1e9), thr_s)
        if thr_f is not None:
            metrics["fused"] = compute_metrics(y_true, np.nan_to_num(fu_arr, nan=-1e9), thr_f)

    # Write metrics.json
    mpath = Path(args.outdir)/("metrics_" + Path(args.log).stem + ".json")
    json.dump({
        "n_lines": n,
        "labels_used": y_true is not None,
        "threshold_percentile": args.thr_percentile,
        "thresholds": {"content":thr_c, "session":thr_s, "fused":thr_f},
        "metrics": metrics
    }, open(mpath,"w"), indent=2)

    print(f"[OK] wrote {pred_path}")
    print(f"[OK] wrote {mpath}")
    if metrics:
        for k,v in metrics.items():
            print(f"== {k.upper()} ==")
            print(f"AUC={v['auc']:.4f}  PR-AUC={v['prauc']:.4f}  F1={v['f1']:.4f}  P={v['precision']:.4f}  R={v['recall']:.4f}  FPR={v['fpr']:.4f}")
            print(f"TN={v['tn']} FP={v['fp']}  FN={v['fn']} TP={v['tp']}  thr={v['threshold']:.6f}")

if __name__ == "__main__":
    main()
