"""
Per-line inference latency / throughput micro-benchmark for the content-AE
and session-LSTM-VAE branches, in both streaming (single-sample) and batch
modes -- mirrors the peer paper's simulated-streaming evaluation section.

Usage:
    python scripts/benchmark_latency.py [--log data/mixed/access_eval_mix_2000.log]
"""
import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_file import load_content_bundle, load_session_bundle
from src.featurization.features import line_to_vector

WARMUP_LINES = 50


def percentile_stats(times_s):
    arr = np.array(times_s) * 1000.0  # -> ms
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="data/mixed/access_eval_mix_2000.log")
    ap.add_argument("--session-dir", default="artifacts/session_v2")
    ap.add_argument("--out", default="artifacts/eval/latency_benchmark.json")
    args = ap.parse_args()

    cb = load_content_bundle()
    sb = load_session_bundle(args.session_dir)
    if cb is None:
        raise SystemExit("No trained content-AE model found under artifacts/content.")

    lines = [ln.strip() for ln in open(ROOT / args.log, encoding="utf-8", errors="ignore") if ln.strip()]
    print(f"[INFO] Loaded {len(lines)} lines from {args.log}")

    # -------------------- Streaming (single-sample) mode --------------------
    t_vectorize, t_content, t_session, t_fuse = [], [], [], []
    session_buffers = {}
    win = sb["window"] if sb else 20

    for i, s in enumerate(lines):
        t0 = time.perf_counter()
        vec, meta = line_to_vector(s)
        t1 = time.perf_counter()
        if vec is None:
            continue

        Xs_c = cb["scaler"].transform(vec.reshape(1, -1))
        with torch.no_grad():
            recon_c = cb["model"](torch.tensor(Xs_c, dtype=torch.float32)).numpy()
        sc = float(np.mean((Xs_c - recon_c) ** 2))
        t2 = time.perf_counter()

        ss = None
        if sb:
            key = (meta.get("ip", ""), meta.get("ua", ""))
            Xs_s = sb["scaler"].transform(vec.reshape(1, -1))[0]
            buf = session_buffers.setdefault(key, deque(maxlen=win))
            buf.append(Xs_s)
            if len(buf) == win:
                seq = np.stack(buf, axis=0).reshape(1, win, -1)
                with torch.no_grad():
                    recon_s, _, _ = sb["model"](torch.tensor(seq, dtype=torch.float32))
                    ss = float(np.mean((seq - recon_s.numpy()) ** 2))
        t3 = time.perf_counter()

        _ = 0.5 * sc + 0.5 * (ss if ss is not None else sc)
        t4 = time.perf_counter()

        if i >= WARMUP_LINES:
            t_vectorize.append(t1 - t0)
            t_content.append(t2 - t1)
            t_session.append(t3 - t2)
            t_fuse.append(t4 - t3)

    streaming_total_s = sum(t_vectorize) + sum(t_content) + sum(t_session) + sum(t_fuse)
    n_timed = len(t_vectorize)

    # -------------------- Batch mode (content branch, one forward pass) --------------------
    vecs = []
    for s in lines:
        v, _ = line_to_vector(s)
        if v is not None:
            vecs.append(v)
    X = np.stack(vecs, axis=0)
    Xs = cb["scaler"].transform(X)
    Xt = torch.tensor(Xs, dtype=torch.float32)

    t0 = time.perf_counter()
    with torch.no_grad():
        recon = cb["model"](Xt).numpy()
    t1 = time.perf_counter()
    batch_total_s = t1 - t0
    batch_per_line_s = batch_total_s / len(vecs)

    results = {
        "log": args.log,
        "n_lines": len(lines),
        "n_timed_streaming": n_timed,
        "streaming": {
            "vectorize": percentile_stats(t_vectorize),
            "content_score": percentile_stats(t_content),
            "session_score": percentile_stats(t_session),
            "fusion": percentile_stats(t_fuse),
            "throughput_lines_per_sec": n_timed / streaming_total_s if streaming_total_s > 0 else None,
        },
        "batch": {
            "n_lines": len(vecs),
            "total_s": batch_total_s,
            "per_line_ms": batch_per_line_s * 1000.0,
            "throughput_lines_per_sec": len(vecs) / batch_total_s if batch_total_s > 0 else None,
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))

    print(f"[OK] wrote {args.out}")
    print(f"Streaming: mean={results['streaming']['content_score']['mean_ms']:.4f}ms/line (content), "
          f"throughput={results['streaming']['throughput_lines_per_sec']:.1f} lines/sec")
    print(f"Batch: {results['batch']['per_line_ms']:.4f}ms/line, "
          f"throughput={results['batch']['throughput_lines_per_sec']:.1f} lines/sec")


if __name__ == "__main__":
    main()
