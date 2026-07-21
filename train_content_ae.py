# --- PATH SHIM: allow imports from ./src without install ---
import sys
from pathlib import Path
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))

import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.preprocessing import StandardScaler

from featurization.loaders import read_vectors_from_log
from models.content_autoencoder import ContentAE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to training log (txt or .gz)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-lines", type=int, default=None, help="Optional cap for quick runs")
    ap.add_argument("--bad-lines-out", default=None, help="Write unparsable lines here")
    ap.add_argument("--percentile", type=float, default=95.0, help="Threshold percentile (e.g., 95)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Load and featurize (robust; skips broken lines) ---
    X, skipped = read_vectors_from_log(args.log, max_lines=args.max_lines, bad_out=args.bad_lines_out)
    print(f"[INFO] Parsed {len(X)} vectors from {args.log}. Skipped {skipped} bad lines.")

    # --- Scale ---
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)

    # --- Model ---
    input_dim = Xs.shape[1]
    model = ContentAE(input_dim=input_dim)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(Xs, dtype=torch.float32)

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        recon = model(Xt)
        loss = loss_fn(recon, Xt)
        loss.backward()
        opt.step()
        print(f"[AE] epoch {epoch} loss={loss.item():.6f}")

    # --- Threshold via reconstruction error percentile ---
    model.eval()
    with torch.no_grad():
        recon = model(Xt).cpu().numpy()
    errs = np.mean((Xs - recon) ** 2, axis=1)
    thr = float(np.percentile(errs, args.percentile))

    # --- Save artifacts ---
    os.makedirs("artifacts/content", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/content/model.pt")
    joblib.dump(scaler, "artifacts/content/scaler.joblib")
    json.dump(
        {"input_dim": int(input_dim), "threshold": thr, "percentile": float(args.percentile)},
        open("artifacts/content/config.json", "w"),
        indent=2,
    )
    print(f"Saved content AE artifacts. threshold={thr:.6f} (p{args.percentile})")


if __name__ == "__main__":
    main()
