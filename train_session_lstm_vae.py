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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to training log (txt or .gz)")
    ap.add_argument("--window", type=int, default=20, help="Session window length")
    ap.add_argument("--step", type=int, default=1, help="Stride for rolling windows")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-lines", type=int, default=None, help="Optional cap for quick runs")
    ap.add_argument("--bad-lines-out", default=None, help="Write unparsable lines here")
    ap.add_argument("--percentile", type=float, default=95.0, help="Threshold percentile for session error")
    args = ap.parse_args()

    # --- Load and featurize (robust; skips broken lines) ---
    X, skipped = read_vectors_from_log(args.log, max_lines=args.max_lines, bad_out=args.bad_lines_out)
    print(f"[INFO] Parsed {len(X)} vectors for session training. Skipped {skipped} bad lines.")

    if len(X) < args.window:
        raise SystemExit(f"Not enough data to build sessions: need ≥{args.window}, got {len(X)}.")

    # --- Scale ---
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)

    # --- Build rolling session windows ---
    seqs = []
    for i in range(0, len(Xs) - args.window + 1, args.step):
        seqs.append(Xs[i : i + args.window])
    S = np.stack(seqs, axis=0)  # [n_seq, window, d]
    print(f"[INFO] Built {S.shape[0]} session windows of length {args.window} (stride {args.step}).")

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
    os.makedirs("artifacts/session", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/session/model.pt")
    joblib.dump(scaler, "artifacts/session/scaler.joblib")
    json.dump(
        {"input_dim": int(input_dim), "threshold": thr, "window": int(args.window), "percentile": float(args.percentile)},
        open("artifacts/session/config.json", "w"),
        indent=2,
    )
    print(f"Saved session LSTM-VAE artifacts. threshold={thr:.6f} (p{args.percentile}), window={args.window}")


if __name__ == "__main__":
    main()
