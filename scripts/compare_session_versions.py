"""
Compare the session-branch v1 (global rolling window, naive repeat-vector
scoring) against v2 (real (ip,ua)-grouped sessions, stateful rolling-buffer
scoring) across all reconstructable datasets.

Runs scripts/evaluate_file.py twice per dataset (v1+naive, v2+stateful) into a
scratch outdir and compiles the resulting metrics into one CSV.

Usage:
    python scripts/compare_session_versions.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "artifacts" / "_scratch_session_compare"

# stem -> (log path, labels path)
STEMS = {
    "csic_eval": ("data/csic/csic_eval.log", "data/csic/csic_eval.log.labels.txt"),
    "access_eval_mix_2000": ("data/mixed/access_eval_mix_2000.log", "data/mixed/access_eval_mix_2000.log.labels.txt"),
    "access_eval_small_500": ("data/mixed/access_eval_small_500.log", "data/mixed/access_eval_small_500.log.labels.txt"),
    "nginx_json_eval_800": ("data/mixed/nginx_json_eval_800.log", "data/mixed/nginx_json_eval_800.log.labels.txt"),
    "access_attacks_200": ("data/raw/access_attacks_200.log", "data/raw/access_attacks_200.log.labels.txt"),
    "access_mixed_500": ("data/raw/access_mixed_500.log", "data/raw/access_mixed_500.log.labels.txt"),
    "access_small_benign": ("data/raw/access_small_benign.log", "data/raw/access_small_benign.log.labels.txt"),
    "access_super_long_session_2000": ("data/raw/access_super_long_session_2000.log", "data/raw/access_super_long_session_2000.log.labels.txt"),
}

VERSIONS = [
    ("v1_naive", "artifacts/session", "naive"),
    ("v2_stateful", "artifacts/session_v2", "stateful"),
]


def run_one(stem, log, labels, session_dir, mode, outdir):
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate_file.py"),
         "--log", str(ROOT / log), "--labels", str(ROOT / labels),
         "--outdir", str(outdir), "--session-dir", str(ROOT / session_dir),
         "--session-mode", mode, "--thr-percentile", "95.0"],
        cwd=str(ROOT), check=True, capture_output=True, text=True,
    )
    mfile = outdir / f"metrics_{Path(log).stem}.json"
    return json.loads(mfile.read_text())


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    rows = []
    for stem, (log, labels) in STEMS.items():
        for version_name, session_dir, mode in VERSIONS:
            outdir = SCRATCH / version_name
            outdir.mkdir(parents=True, exist_ok=True)
            try:
                m = run_one(stem, log, labels, session_dir, mode, outdir)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] {stem}/{version_name}: {e.stderr[-500:]}", file=sys.stderr)
                continue
            sess = m["metrics"].get("session")
            fused = m["metrics"].get("fused")
            n_session_scored = None
            if sess is not None:
                n_session_scored = sess["tn"] + sess["fp"] + sess["fn"] + sess["tp"]
            rows.append({
                "dataset": stem,
                "session_version": version_name,
                "n_lines": m["n_lines"],
                "n_session_scored": n_session_scored,
                "session_auc": sess["auc"] if sess else None,
                "session_f1": sess["f1"] if sess else None,
                "session_precision": sess["precision"] if sess else None,
                "session_recall": sess["recall"] if sess else None,
                "fused_auc": fused["auc"] if fused else None,
                "fused_f1": fused["f1"] if fused else None,
            })
            print(f"[OK] {stem} / {version_name}: session_auc={rows[-1]['session_auc']} n_scored={n_session_scored}")

    out_csv = ROOT / "artifacts" / "eval" / "session_v1_vs_v2.csv"
    cols = ["dataset", "session_version", "n_lines", "n_session_scored",
            "session_auc", "session_f1", "session_precision", "session_recall",
            "fused_auc", "fused_f1"]
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")
    print(f"\n[OK] wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
