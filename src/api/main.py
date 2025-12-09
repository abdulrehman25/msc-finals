from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import numpy as np, torch, joblib, json, os, glob, subprocess, sys, io, urllib.parse
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, deque
from typing import Dict, Tuple, Optional

from featurization.features import line_to_vector
from featurization.parse import parse_line
from models.content_autoencoder import ContentAE
from models.session_lstm_vae import SessionLSTMVAE
from storage.db import init_db, insert_event, recent_events, stats_counts

app = FastAPI(title="Dual-Branch Log Anomaly API", version="0.4.0")

# ---- Rolling session buffer ----
SESSION_BUFFERS: Dict[Tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=20))
SESSION_WINDOW = 20

# ---- Model bundles ----
_content_bundle = None
_session_bundle = None
_fusion_cfg = {"weights": {"content": 0.5, "session": 0.5}}

class LinePayload(BaseModel):
    line: str

def load_content_branch():
    p = Path("artifacts/content")
    if not p.exists(): return None
    scaler = joblib.load(p / "scaler.joblib")
    cfg = json.load(open(p / "config.json"))
    model = ContentAE(input_dim=cfg["input_dim"])
    model.load_state_dict(torch.load(p / "model.pt", map_location="cpu"))
    model.eval()
    return {"scaler": scaler, "model": model, "thr": float(cfg["threshold"])}

def load_session_branch():
    p = Path("artifacts/session")
    if not p.exists(): return None
    scaler = joblib.load(p / "scaler.joblib")
    cfg = json.load(open(p / "config.json"))
    model = SessionLSTMVAE(input_dim=cfg["input_dim"])
    model.load_state_dict(torch.load(p / "model.pt", map_location="cpu"))
    model.eval()
    return {"scaler": scaler, "model": model, "thr": float(cfg["threshold"]), "window": int(cfg.get("window", 20))}

def load_fusion_cfg():
    p = Path("artifacts/fusion/config.json")
    if p.exists(): return json.load(open(p))
    return {"weights": {"content": 0.5, "session": 0.5}}

@app.on_event("startup")
def _startup():
    global _content_bundle, _session_bundle, _fusion_cfg, SESSION_WINDOW
    init_db()
    _content_bundle = load_content_branch()
    _session_bundle = load_session_branch()
    _fusion_cfg = load_fusion_cfg()
    if _session_bundle:
        SESSION_WINDOW = int(_session_bundle["window"]) or 20

@app.get("/health")
def health():
    b = {
        "content": bool(_content_bundle),
        "session": bool(_session_bundle),
        "fusion": _fusion_cfg,
        "session_window": SESSION_WINDOW
    }
    return {"status": "ok", "bundles": b}

def score_content(vec: np.ndarray):
    if not _content_bundle: return None
    scaler = _content_bundle["scaler"]; model = _content_bundle["model"]; thr = _content_bundle["thr"]
    Xs = scaler.transform(vec.reshape(1, -1))
    with torch.no_grad(): recon = model(torch.tensor(Xs, dtype=torch.float32)).numpy()
    err = float(np.mean((Xs - recon) ** 2))
    return {"score": err, "threshold": thr, "is_anomaly": err > thr}

def score_session(key: Tuple[str, str], vec: np.ndarray):
    if not _session_bundle: return None
    scaler = _session_bundle["scaler"]; model = _session_bundle["model"]; thr = _session_bundle["thr"]
    Xs = scaler.transform(vec.reshape(1, -1)).reshape(-1)
    buf = SESSION_BUFFERS[key]; buf.append(Xs)
    if len(buf) < SESSION_WINDOW: return None
    seq = np.stack(buf)[-SESSION_WINDOW:].reshape(1, SESSION_WINDOW, -1)
    with torch.no_grad(): recon, _, _ = model(torch.tensor(seq, dtype=torch.float32))
    err = float(np.mean((seq - recon.numpy()) ** 2))
    return {"score": err, "threshold": thr, "is_anomaly": err > thr}

def fuse_scores(scores: dict):
    w = _fusion_cfg.get("weights", {"content": 0.5, "session": 0.5})
    used = [(k, v) for k, v in scores.items() if v is not None and k in w]
    if not used: return None
    fused = sum(w[k]*v["score"] for k,v in used) / max(sum(w[k] for k,_ in used), 1e-6)
    thr_mean = float(np.mean([v["threshold"] for _,v in used]))
    return {"fused_score": fused, "fused_threshold": thr_mean, "is_anomaly": fused > thr_mean}

@app.post("/score")
def score(payload: LinePayload):
    parsed = parse_line(payload.line)
    vec, meta = line_to_vector(payload.line)
    if vec is None or parsed is None:
        raise HTTPException(400, "Could not parse line")
    ip = meta.get("ip") or ""; ua = meta.get("ua") or ""; key = (ip, ua)
    c = score_content(vec); s = score_session(key, vec)
    per_branch = {"content": c, "session": s}
    fused = fuse_scores(per_branch) or (c and {"fused_score": c["score"], "fused_threshold": c["threshold"], "is_anomaly": c["is_anomaly"]})
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "ts_utc": now, "ip": ip, "ua": ua, "method": parsed.get("method"), "path": parsed.get("path"),
        "status": int(parsed.get("status", "0") or 0),
        "size": 0 if parsed.get("size") in (None, "-", "") else int(parsed.get("size")),
        "content_score": c["score"] if c else None, "content_thr": c["threshold"] if c else None,
        "session_score": s["score"] if s else None, "session_thr": s["threshold"] if s else None,
        "fused_score": fused["fused_score"] if fused else None, "fused_thr": fused["fused_threshold"] if fused else None,
        "is_anomaly": fused["is_anomaly"] if fused else False, "raw_line": payload.line,
    }
    insert_event(row)
    return {"per_branch": per_branch, **fused, "session_context_len": len(SESSION_BUFFERS[key])}

# ---------- Evaluation helpers ----------
EVAL_DIR = Path("artifacts/eval")
def _latest_metrics_path() -> Optional[Path]:
    if not EVAL_DIR.exists(): return None
    files = list(EVAL_DIR.glob("metrics_*.json"))
    if not files: return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]

@app.get("/eval/last")
def get_last_metrics():
    p = _latest_metrics_path()
    if not p: return JSONResponse({"ok": False, "reason": "no metrics found under artifacts/eval"}, status_code=404)
    data = json.load(open(p)); data["file"] = str(p)
    return JSONResponse({"ok": True, "metrics": data})

class EvalRunPayload(BaseModel):
    log: str
    labels: Optional[str] = None
    thr_percentile: float = 95.0
    fuse_weights: str = "0.5,0.5"

@app.post("/eval/run")
def run_eval(payload: EvalRunPayload):
    script = Path("scripts/evaluate_file.py")
    if not script.exists(): raise HTTPException(500, "scripts/evaluate_file.py not found")
    cmd = [sys.executable, str(script), "--log", payload.log, "--thr-percentile", str(payload.thr_percentile), "--fuse-weights", payload.fuse_weights]
    if payload.labels: cmd += ["--labels", payload.labels]
    cp = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.cwd()))
    if cp.returncode != 0:
        return JSONResponse({"ok": False, "stderr": cp.stderr, "stdout": cp.stdout}, status_code=500)
    p = _latest_metrics_path()
    if not p: return JSONResponse({"ok": False, "reason": "eval finished but no metrics file produced"}, status_code=500)
    data = json.load(open(p)); data["file"] = str(p)
    return JSONResponse({"ok": True, "metrics": data})

# ---------- NEW: ROC/PR curve endpoint ----------
@app.get("/eval/curve.png")
def eval_curve_png(
    chart: str = Query(..., regex="^(roc|pr)$"),
    kind: str = Query(..., regex="^(fused|content|session)$"),
    log: str = Query(...),
    labels: str = Query(...)
):
    """
    Returns a PNG of ROC or PR curve for the given kind using:
    - predictions_{Path(log).stem}.csv (created by evaluate_file.py)
    - labels file path (0/1 per line or last-column CSV)
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    import matplotlib.pyplot as plt
    stem = Path(log).stem
    pred_csv = EVAL_DIR / f"predictions_{stem}.csv"
    if not pred_csv.exists():
        raise HTTPException(404, f"predictions file not found: {pred_csv}. Run /eval/run first.")
    # Load predictions
    scores = {"content": [], "session": [], "fused": []}
    with open(pred_csv, encoding="utf-8", errors="ignore") as f:
        next(f)  # header
        for ln in f:
            parts = ln.rstrip("\n").split(",", 3)
            if len(parts) < 4: continue
            sc = parts[0].strip(); ss = parts[1].strip(); sf = parts[2].strip()
            scores["content"].append(float(sc) if sc else np.nan)
            scores["session"].append(float(ss) if ss else np.nan)
            scores["fused"].append(float(sf) if sf else np.nan)
    y_true = []
    with open(labels, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip()
            if not s: continue
            try:
                # allow "line,label" or "label"
                lab = int(s.split(",")[-1])
                y_true.append(lab)
            except: pass
    if len(y_true) != len(scores["fused"]):
        raise HTTPException(400, f"labels length {len(y_true)} != predictions length {len(scores['fused'])}")

    y_true = np.array(y_true, dtype=int)
    y_score = np.array(scores[kind], dtype=float)
    mask = ~np.isnan(y_score)
    y_true = y_true[mask]; y_score = y_score[mask]
    if y_true.size == 0:
        raise HTTPException(400, "no valid scores for selected branch")

    fig = plt.figure(figsize=(5,4), dpi=160)
    ax = fig.add_subplot(111)

    if chart == "roc":
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
        ax.plot([0,1],[0,1],'--',lw=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC • {kind}")
        ax.legend(loc="lower right")
    else:
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, lw=2, label=f"PR-AUC = {pr_auc:.4f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(f"PR • {kind}")
        ax.legend(loc="lower left")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

# ---------- Dashboard ----------
DASHBOARD_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Log Anomalies Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    body { font: 14px/1.4 -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial; margin: 20px; }
    .row { display: flex; gap: 16px; align-items: stretch; margin-bottom: 12px; flex-wrap: wrap; }
    .card { padding: 12px 16px; border: 1px solid #ddd; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
    .card h3{ margin: 0 0 6px 0; font-size: 16px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border-bottom: 1px solid #eee; padding: 6px 8px; text-align: left; vertical-align: top; }
    tr.anom { background: #fff3f3; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; border: 1px solid #ccc; }
    .ok { background:#eefbf1; border-color:#bfe5cc; }
    .bad { background:#fde8e8; border-color:#f5b5b5; }
    button { padding: 8px 12px; border-radius: 10px; border: 1px solid #ccc; cursor: pointer; }
    .metrics { display: grid; grid-template-columns: repeat(3,minmax(260px,1fr)); gap: 12px; width: 100%; }
    .muted { color:#777; }
    @media (max-width: 900px){ .metrics{ grid-template-columns: 1fr; } }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
    .charts { display:grid; grid-template-columns: repeat(3,minmax(260px,1fr)); gap: 12px; }
    img.chart { width:100%; height:auto; border:1px solid #eee; border-radius:10px; }

    .modal{position:fixed;inset:0;background:rgba(0,0,0,.35);display:grid;place-items:center;}
.modal-body{background:#fff;padding:16px 20px;border-radius:10px;min-width:280px;box-shadow:0 10px 30px rgba(0,0,0,.2);}
.cm{border-collapse:collapse;margin:10px 0;}
.cm th,.cm td{border:1px solid #ddd;padding:6px 10px;text-align:center;}
.linklike{border:none;background:none;color:#2563eb;cursor:pointer;padding:0;}
  </style>
</head>
<body>
  <h2>Log Anomalies Dashboard</h2>

  <div class="row">
    <div class="card"><b>Total</b>: <span id="total">—</span></div>
    <div class="card"><b>Anomalies</b>: <span id="anoms">—</span></div>
    <button id="refresh">Refresh table</button>
    <span class="muted">Auto-refresh every 5s</span>
  </div>

  <div class="row">
    <div class="card" style="flex:1; min-width: 320px;">
      <h3>Evaluation Metrics</h3>
      <div id="metrics-when" class="muted">No metrics loaded</div>
      <div class="metrics" id="metrics-cards"></div>
      <div class="row" style="margin-top:10px">
        <input id="eval-log" placeholder="path\\to\\log.log" style="flex:1; padding:6px 8px; border:1px solid #ccc; border-radius:8px;">
        <input id="eval-labels" placeholder="(optional) path\\to\\labels.txt" style="flex:1; padding:6px 8px; border:1px solid #ccc; border-radius:8px;">
        <button id="make-labels">Create labels</button>
        <button id="run-eval">Run eval</button>
      </div>
      <div class="muted mono" id="eval-status"></div>

      <div id="charts-wrap" style="margin-top:12px; display:none">
        <div class="muted">Curves render only when labels are provided.</div>
        <h4>ROC</h4>
        <div class="charts">
          <img id="roc-fused" class="chart" />
          <img id="roc-content" class="chart" />
          <img id="roc-session" class="chart" />
        </div>
        <h4>Precision–Recall</h4>
        <div class="charts">
          <img id="pr-fused" class="chart" />
          <img id="pr-content" class="chart" />
          <img id="pr-session" class="chart" />
        </div>
      </div>
    </div>
  </div>

  <table>
    <thead>
      <tr>
        <th>Time (UTC)</th><th>IP</th><th>Method</th><th>Path</th><th>Status</th><th>Size</th>
        <th>Content</th><th>Session</th><th>Fused</th><th>Alert</th>
      </tr>
    </thead>
    <tbody id="rows"></tbody>
  </table>

<script>
async function loadTable(){
  const [events, stats] = await Promise.all([
    fetch('/recent?limit=100').then(r=>r.json()),
    fetch('/stats').then(r=>r.json()),
  ]);
  document.getElementById('total').textContent = stats.total ?? '—';
  document.getElementById('anoms').textContent = stats.anomalies ?? '—';
  const tb = document.getElementById('rows');
  tb.innerHTML = '';
  for (const e of events){
    const tr = document.createElement('tr');
    if (e.is_anomaly) tr.className = 'anom';
    tr.innerHTML = `
      <td>${e.ts_utc || ''}</td>
      <td>${e.ip || ''}</td>
      <td>${e.method || ''}</td>
      <td>${e.path || ''}</td>
      <td>${e.status ?? ''}</td>
      <td>${e.size ?? ''}</td>
      <td>${(e.content_score ?? '').toString().slice(0,8)}</td>
      <td>${(e.session_score ?? '').toString().slice(0,8)}</td>
      <td>${(e.fused_score ?? '').toString().slice(0,8)}</td>
      <td><span class="badge ${e.is_anomaly ? 'bad':'ok'}">${e.is_anomaly ? 'Anomaly' : 'OK'}</span></td>
    `;
    tb.appendChild(tr);
  }
}

function fmt(x, d=4){
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return (typeof x === "number") ? x.toFixed(d) : x;
}
function metricsCard(title, m) {
  if (!m) return `<div class="card"><b>${title}</b><div class="muted">No data</div></div>`;
  const thr = (m.threshold===null||m.threshold===undefined) ? '—' : Number(m.threshold).toFixed(6);
  const tn = m.tn ?? m.cm?.tn, fp = m.fp ?? m.cm?.fp, fn = m.fn ?? m.cm?.fn, tp = m.tp ?? m.cm?.tp;
  return `
    <div class="card">
      <b>${title}</b>
      <div>AUC: <b>${fmt(m.auc)}</b></div>
      <div>PR-AUC: <b>${fmt(m.prauc)}</b></div>
      <div>F1: <b>${fmt(m.f1)}</b> &nbsp; P: ${fmt(m.precision)} &nbsp; R: ${fmt(m.recall)}</div>
      <div>FPR: ${fmt(m.fpr)} &nbsp; thr: <span class="mono">${thr}</span></div>
      <div>A/M: <b>${fmt(m.alerts_per_million,0)}</b> &nbsp; (rate=${fmt(m.alert_rate)})</div>
      <div>F1@1%: <b>${fmt(m.f1_at_1pct)}</b> &nbsp; thr@1%=${fmt(m.thr_at_1pct,6)}</div>
      <div>F1@0.1%: <b>${fmt(m.f1_at_0_1pct)}</b> &nbsp; thr@0.1%=${fmt(m.thr_at_0_1pct,6)}</div>
      <div class="muted">TN ${tn ?? '—'} • FP ${fp ?? '—'} • FN ${fn ?? '—'} • TP ${tp ?? '—'}
        &nbsp; <button class="linklike" onclick='showCM(${JSON.stringify({tn,fp,fn,tp})})'>Confusion&nbsp;Matrix</button>
      </div>
    </div>`;
}

async function loadMetrics(){
  const box = document.getElementById('metrics-cards');
  const when = document.getElementById('metrics-when');
  try{
    const r = await fetch('/eval/last');
    if (!r.ok){
      box.innerHTML = '';
      when.textContent = 'No metrics yet. Use "Run eval" to generate.';
      return;
    }
    const j = await r.json();
    const M = j.metrics || {};
    const met = M.metrics || {};
    const file = M.file || '';
    when.textContent = `Loaded: ${file}  •  lines=${M.n_lines ?? '—'}  •  labels=${M.labels_used ? 'yes':'no'}`;
    box.innerHTML = [
      metricsCard('Fused', met.fused),
      metricsCard('Content', met.content),
      metricsCard('Session', met.session),
    ].join('');
  }catch(e){
    when.textContent = 'Error loading metrics';
  }
}

function showCurves(logPath, labelsPath){
  if (!labelsPath){ document.getElementById('charts-wrap').style.display = 'none'; return; }
  const wrap = document.getElementById('charts-wrap');
  const enc = s => encodeURIComponent(s);
  const set = (id, chart, kind) => {
    const url = `/eval/curve.png?chart=${chart}&kind=${kind}&log=${enc(logPath)}&labels=${enc(labelsPath)}`;
    document.getElementById(id).src = url;
  };
  set('roc-fused','roc','fused');
  set('roc-content','roc','content');
  set('roc-session','roc','session');
  set('pr-fused','pr','fused');
  set('pr-content','pr','content');
  set('pr-session','pr','session');
  wrap.style.display = 'block';
}

async function runEval(){
  const log = document.getElementById('eval-log').value.trim();
  const labels = document.getElementById('eval-labels').value.trim();
  const status = document.getElementById('eval-status');
  if (!log){ status.textContent = 'Please enter a log path.'; return; }
  status.textContent = 'Running evaluation...';
  const payload = labels ? {log, labels} : {log};
  const r = await fetch('/eval/run', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  const j = await r.json();
  if (!j.ok){
    status.textContent = 'Eval failed: ' + (j.reason || j.stderr || 'unknown error');
    return;
  }
  status.textContent = 'Eval complete: ' + (j.metrics?.file || '');
  loadMetrics();
  if (labels) showCurves(log, labels);
}
function showCM(cm){
  const html = `
    <div class="modal" onclick="this.remove()">
      <div class="modal-body" onclick="event.stopPropagation()">
        <h3>Confusion Matrix</h3>
        <table class="cm">
          <tr><th></th><th>Pred 0</th><th>Pred 1</th></tr>
          <tr><th>True 0</th><td>${cm.tn ?? '—'}</td><td>${cm.fp ?? '—'}</td></tr>
          <tr><th>True 1</th><td>${cm.fn ?? '—'}</td><td>${cm.tp ?? '—'}</td></tr>
        </table>
        <button onclick="document.querySelector('.modal').remove()">Close</button>
      </div>
    </div>`;
  document.body.insertAdjacentHTML('beforeend', html);
}
async function makeLabels(){
  const log = document.getElementById('eval-log').value.trim();
  const status = document.getElementById('eval-status');
  if (!log){ status.textContent = 'Enter a log path first.'; return; }
  status.textContent = 'Creating heuristic labels...';
  const r = await fetch('/labels/heuristic', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({log})
  });
  const j = await r.json();
  if (!j.ok){ status.textContent = 'Labeling failed: ' + (j.stderr || j.reason || 'unknown'); return; }
  document.getElementById('eval-labels').value = j.labels;
  status.textContent = 'Labels created: ' + j.labels;
}
document.getElementById('make-labels').onclick = makeLabels;

document.getElementById('refresh').onclick = loadTable;
document.getElementById('run-eval').onclick = runEval;

setInterval(loadTable, 5000);
loadTable();
loadMetrics();
</script>
</body>
</html>
"""

class LabelRunPayload(BaseModel):
    log: str
    out: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(DASHBOARD_HTML)

@app.get("/recent")
def recent(limit: int = 100):
    return JSONResponse(recent_events(limit=limit))

@app.get("/stats")
def stats():
    return JSONResponse(stats_counts())

@app.post("/labels/heuristic")
def create_labels(payload: LabelRunPayload):
    script = Path("scripts/heuristic_label_logs.py")
    if not script.exists():
        raise HTTPException(500, "scripts/heuristic_label_logs.py not found")
    out = payload.out or (payload.log + ".labels.txt")
    cp = subprocess.run(
        [sys.executable, str(script), "--log", payload.log, "--out", out],
        capture_output=True, text=True, cwd=str(Path.cwd())
    )
    if cp.returncode != 0:
        return JSONResponse({"ok": False, "stderr": cp.stderr, "stdout": cp.stdout}, status_code=500)
    return JSONResponse({"ok": True, "labels": out, "stdout": cp.stdout})