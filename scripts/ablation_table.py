"""
Compile a per-dataset branch-ablation table (content-only vs session-only vs
fused) from artifacts/eval/metrics_*.json into one wide CSV.

Usage:
    python scripts/ablation_table.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "artifacts" / "eval"

BRANCHES = ["content", "session", "fused"]
FIELDS = ["auc", "prauc", "f1", "precision", "recall", "fpr"]

# The 8 reconstructable stems (see scripts/reconstruct_logs.py). Excludes the
# orphaned access_long_session_300 artifact, which has no label sidecar
# anywhere in the repo and predates the current metrics schema.
CANONICAL_STEMS = [
    "csic_eval", "access_eval_mix_2000", "access_eval_small_500", "nginx_json_eval_800",
    "access_attacks_200", "access_mixed_500", "access_small_benign", "access_super_long_session_2000",
]


def main():
    rows = []
    for stem in CANONICAL_STEMS:
        mpath = EVAL_DIR / f"metrics_{stem}.json"
        if not mpath.exists():
            print(f"[SKIP] {stem}: no metrics file")
            continue
        m = json.loads(mpath.read_text())
        metrics = m.get("metrics", {})
        row = {"dataset": stem, "n_lines": m.get("n_lines")}
        for branch in BRANCHES:
            b = metrics.get(branch)
            for field in FIELDS:
                row[f"{branch}_{field}"] = b[field] if b else None
        rows.append(row)

    cols = ["dataset", "n_lines"] + [f"{b}_{f}" for b in BRANCHES for f in FIELDS]
    out_csv = EVAL_DIR / "ablation_table.csv"
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")

    print(f"[OK] wrote {out_csv} ({len(rows)} datasets)")
    for r in rows:
        print(f"  {r['dataset']:35s} content_auc={r['content_auc']}  session_auc={r['session_auc']}  fused_auc={r['fused_auc']}")


if __name__ == "__main__":
    main()
