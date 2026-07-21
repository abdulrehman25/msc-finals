"""
Reconstruct raw .log files from artifacts/eval/predictions_<stem>.csv.

Raw .log files are gitignored and not present on disk, but the "line" column
of each predictions_<stem>.csv preserves the original log text verbatim, in
original order, one row per input line. This rebuilds a working .log next to
each stem's canonical `<stem>.log.labels.txt` sidecar so the eval/training
pipeline can be rerun from a fresh clone.

Usage:
    python scripts/reconstruct_logs.py [--verify-only]
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# stem -> (predictions csv, output .log path, label sidecar path)
STEMS = {
    "csic_eval": ("artifacts/eval/predictions_csic_eval.csv", "data/csic/csic_eval.log", "data/csic/csic_eval.log.labels.txt"),
    "access_eval_mix_2000": ("artifacts/eval/predictions_access_eval_mix_2000.csv", "data/mixed/access_eval_mix_2000.log", "data/mixed/access_eval_mix_2000.log.labels.txt"),
    "access_eval_small_500": ("artifacts/eval/predictions_access_eval_small_500.csv", "data/mixed/access_eval_small_500.log", "data/mixed/access_eval_small_500.log.labels.txt"),
    "nginx_json_eval_800": ("artifacts/eval/predictions_nginx_json_eval_800.csv", "data/mixed/nginx_json_eval_800.log", "data/mixed/nginx_json_eval_800.log.labels.txt"),
    "access_attacks_200": ("artifacts/eval/predictions_access_attacks_200.csv", "data/raw/access_attacks_200.log", "data/raw/access_attacks_200.log.labels.txt"),
    "access_mixed_500": ("artifacts/eval/predictions_access_mixed_500.csv", "data/raw/access_mixed_500.log", "data/raw/access_mixed_500.log.labels.txt"),
    "access_small_benign": ("artifacts/eval/predictions_access_small_benign.csv", "data/raw/access_small_benign.log", "data/raw/access_small_benign.log.labels.txt"),
    "access_super_long_session_2000": ("artifacts/eval/predictions_access_super_long_session_2000.csv", "data/raw/access_super_long_session_2000.log", "data/raw/access_super_long_session_2000.log.labels.txt"),
    # access_long_session_300 intentionally excluded: orphaned artifact, no label sidecar exists anywhere.
}


def reconstruct_one(stem: str, pred_rel: str, out_rel: str, labels_rel: str) -> int:
    pred_path = ROOT / pred_rel
    out_path = ROOT / out_rel
    labels_path = ROOT / labels_rel

    if not pred_path.exists():
        print(f"[SKIP] {stem}: predictions file missing ({pred_rel})")
        return 0
    if not labels_path.exists():
        print(f"[SKIP] {stem}: label sidecar missing ({labels_rel})")
        return 0

    lines = []
    with open(pred_path, encoding="utf-8", errors="ignore") as f:
        header = f.readline()  # "score_content,score_session,score_fused,line"
        for ln in f:
            parts = ln.rstrip("\n").split(",", 3)
            if len(parts) < 4:
                continue
            lines.append(parts[3])

    with open(labels_path, encoding="utf-8", errors="ignore") as f:
        n_labels = sum(1 for ln in f if ln.strip())

    if len(lines) != n_labels:
        print(f"[WARN] {stem}: reconstructed {len(lines)} lines but labels file has {n_labels} — NOT writing (mismatch).")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for l in lines:
            out.write(l + "\n")

    print(f"[OK] {stem}: wrote {len(lines)} lines -> {out_rel}")
    return len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true", help="Only check counts, don't write files")
    args = ap.parse_args()

    total = 0
    for stem, (pred_rel, out_rel, labels_rel) in STEMS.items():
        if args.verify_only:
            pred_path = ROOT / pred_rel
            labels_path = ROOT / labels_rel
            if not pred_path.exists() or not labels_path.exists():
                print(f"[MISSING] {stem}")
                continue
            with open(pred_path, encoding="utf-8", errors="ignore") as f:
                n_pred = sum(1 for _ in f) - 1
            with open(labels_path, encoding="utf-8", errors="ignore") as f:
                n_lab = sum(1 for ln in f if ln.strip())
            status = "OK" if n_pred == n_lab else "MISMATCH"
            print(f"[{status}] {stem}: predictions={n_pred} labels={n_lab}")
            continue
        total += reconstruct_one(stem, pred_rel, out_rel, labels_rel)

    if not args.verify_only:
        print(f"\nReconstructed {total} total lines across {len(STEMS)} stems.")


if __name__ == "__main__":
    main()
