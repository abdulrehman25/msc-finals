# --- PATH SHIM: allow imports from ./src without install ---
import sys
from pathlib import Path
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))

import argparse, json, os
import numpy as np
import torch
import torch.optim as optim
import joblib
from sklearn.preprocessing import StandardScaler

from featurization.loaders import read_vectors_from_log
from models.session_lstm_vae import SessionLSTMVAE, vae_loss
from pipeline.sessionize import build_sessions


def _load_labels(path: str, n_rows: int):
    """Same lenient parsing convention as scripts/evaluate_file.py:
    one int per line, or 'line,label' (label = last comma-separated field)."""
    y = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            parts = s.split(",")
            try:
                y.append(int(parts[0]) if len(parts) == 1 else int(parts[-1]))
            except ValueError:
                pass
    if len(y) != n_rows:
        raise SystemExit(f"Labels file {path} has {len(y)} rows, expected {n_rows}.")
    return np.array(y, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True,
                    help="Path to training log (txt or .gz). Comma-separate multiple paths to pool them.")
    ap.add_argument("--labels", default=None,
                    help="Optional label sidecar(s) (0/1 per line), comma-separated, aligned 1:1 with --log. "
                         "When given, only benign (label==0) rows are used for training.")
    ap.add_argument("--window", type=int, default=20, help="Session window length")
    ap.add_argument("--step", type=int, default=1, help="Stride for session windows")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-lines", type=int, default=None, help="Optional cap per log for quick runs")
    ap.add_argument("--bad-lines-out", default=None, help="Write unparsable lines here")
    ap.add_argument("--percentile", type=float, default=95.0, help="Threshold percentile for session error")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--outdir", default="artifacts/session", help="Where to save trained artifacts")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log_paths = [p.strip() for p in args.log.split(",") if p.strip()]
    label_paths = [p.strip() for p in args.labels.split(",")] if args.labels else [None] * len(log_paths)
    if len(label_paths) != len(log_paths):
        raise SystemExit(f"--labels has {len(label_paths)} entries but --log has {len(log_paths)}; must align 1:1.")

    # --- Load, featurize, and pool (real (ip,ua) session identity preserved per-log) ---
    all_vecs, all_metas = [], []
    for log_path, labels_path in zip(log_paths, label_paths):
        X, skipped, metas = read_vectors_from_log(
            log_path, max_lines=args.max_lines, bad_out=args.bad_lines_out, return_meta=True
        )
        print(f"[INFO] {log_path}: parsed {len(X)} vectors, skipped {skipped} bad lines.")
        if labels_path:
            y = _load_labels(labels_path, n_rows=len(X))
            keep = y == 0
            n_before = len(X)
            X = X[keep]
            metas = [m for m, k in zip(metas, keep) if k]
            print(f"[INFO] {log_path}: kept {len(X)}/{n_before} benign-labeled rows for training.")
        all_vecs.extend(list(X))
        all_metas.extend(metas)

    if not all_vecs:
        raise SystemExit("No training vectors after filtering — check --log/--labels inputs.")

    X = np.stack(all_vecs, axis=0)
    print(f"[INFO] Pooled {len(X)} vectors across {len(log_paths)} log(s) for session training.")

    # --- Scale ---
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)
    Xs_list = list(Xs)

    # --- Build REAL (ip,ua)-grouped session windows (sessionize.build_sessions), ---
    # --- instead of a global rolling window over the raw chronological stream.  ---
    S = build_sessions(Xs_list, all_metas, window=args.window, step=args.step)
    if S.shape[0] == 0:
        raise SystemExit(
            f"No (ip,ua) key had >= {args.window} pooled requests — cannot build session windows. "
            f"Try a smaller --window or pool more session-rich logs."
        )
    print(f"[INFO] Built {S.shape[0]} real (ip,ua)-grouped session windows of length {args.window} (stride {args.step}).")

    # --- Model ---
    input_dim = S.shape[2]
    model = SessionLSTMVAE(input_dim=input_dim)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    Xt = torch.tensor(S, dtype=torch.float32)

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        recon, mu, logvar = model(Xt)

        loss_out = vae_loss(recon, Xt, mu, logvar)
        # vae_loss may return a scalar tensor OR a tuple (total, recon, kl)
        if isinstance(loss_out, tuple):
            total_loss = loss_out[0]
            recon_loss = loss_out[1] if len(loss_out) > 1 else None
            kl_loss    = loss_out[2] if len(loss_out) > 2 else None
        else:
            total_loss = loss_out
            recon_loss = kl_loss = None

        total_loss.backward()
        opt.step()

        if recon_loss is not None and kl_loss is not None:
            print(f"[VAE] epoch {epoch} loss={total_loss.item():.6f} | recon={recon_loss.item():.6f} | kl={kl_loss.item():.6f}")
        else:
            print(f"[VAE] epoch {epoch} loss={total_loss.item():.6f}")

    # --- Threshold via sequence reconstruction error percentile ---
    model.eval()
    with torch.no_grad():
        recon, mu, logvar = model(Xt)
        errs = ((Xt - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()
    thr = float(np.percentile(errs, args.percentile))

    # --- Save artifacts ---
    os.makedirs(args.outdir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.outdir, "model.pt"))
    joblib.dump(scaler, os.path.join(args.outdir, "scaler.joblib"))
    json.dump(
        {
            "input_dim": int(input_dim),
            "threshold": thr,
            "window": int(args.window),
            "percentile": float(args.percentile),
            "session_mode": "grouped",  # real (ip,ua) sessions, not a global rolling window
            "n_sessions": int(S.shape[0]),
            "trained_on": log_paths,
        },
        open(os.path.join(args.outdir, "config.json"), "w"),
        indent=2,
    )
    print(f"Saved session LSTM-VAE artifacts to {args.outdir}. threshold={thr:.6f} (p{args.percentile}), window={args.window}")


if __name__ == "__main__":
    main()
