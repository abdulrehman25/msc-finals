.PHONY: train api stream
train:
	python train_content_ae.py --log data/raw/access.log --epochs 5
	python train_session_lstm_vae.py --log data/raw/access.log --window 10 --epochs 3
	python src/pipeline/fuse_calibrate.py --method percentile --p 95

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

stream:
	python scripts/tail_and_send.py --file data/raw/access.log --api http://127.0.0.1:8000
