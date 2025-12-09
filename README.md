# Web Log Anomaly Detection — Dual Branch MVP

Implements:
- **Branch A (Sessions):** LSTM-VAE on session windows (IP+UA).
- **Branch B (Content):** MLP Autoencoder per request.
- **Fusion:** Weighted fusion of scores. Basic percentile thresholds (EVT-ready).

## Quick start
1) Put your log in `data/raw/access.log`.
2) Train both branches:
   - `python train_content_ae.py --log data/raw/access.log --epochs 5`
   - `python train_session_lstm_vae.py --log data/raw/access.log --window 10 --epochs 3`
   - `python src/pipeline/fuse_calibrate.py --method percentile --p 95`
3) Run API: `uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload`
4) Stream logs: `python scripts/tail_and_send.py --file data/raw/access.log --api http://127.0.0.1:8000`
