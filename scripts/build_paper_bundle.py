"""
Consolidate final tables and figures into artifacts/paper/{tables,figures}/
for the paper-writing phase to reference directly. Idempotent -- rerunning
this after any upstream data change just re-copies the current state.

Usage:
    python scripts/build_paper_bundle.py
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "artifacts" / "eval"
PAPER_DIR = ROOT / "artifacts" / "paper"
TABLES_DIR = PAPER_DIR / "tables"
FIGURES_DIR = PAPER_DIR / "figures"

TABLE_FILES = [
    "ablation_table.csv",
    "fusion_sweep.csv",
    "threshold_transfer.csv",
    "baseline_comparison.csv",
    "session_v1_vs_v2.csv",
    "summary_metrics.csv",
    "latency_benchmark.json",
]

CANONICAL_STEMS = [
    "csic_eval", "access_eval_mix_2000", "access_eval_small_500", "nginx_json_eval_800",
    "access_attacks_200", "access_mixed_500", "access_small_benign", "access_super_long_session_2000",
]

# Figure glob patterns to pull in, relative to artifacts/eval/
FIGURE_GLOBS = [
    "plots/*_bar.png",
    "plots/*_line.png",
    "plots/rf_*.png",
    "plots/iforest_*.png",
    "plots/ocsvm_*.png",
    "advanced_plots/heatmap_*.png",
]


def clean_dir(d: Path):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


def main():
    clean_dir(TABLES_DIR)
    clean_dir(FIGURES_DIR)

    n_tables = 0
    for name in TABLE_FILES:
        src = EVAL_DIR / name
        if src.exists():
            shutil.copy2(src, TABLES_DIR / name)
            n_tables += 1
        else:
            print(f"[WARN] missing table: {name}")

    n_figs = 0
    for pattern in FIGURE_GLOBS:
        for src in sorted(EVAL_DIR.glob(pattern)):
            shutil.copy2(src, FIGURES_DIR / src.name)
            n_figs += 1

    # Threshold-sweep curves for the source (CSIC) + 3 transfer-target datasets only
    # (not every dataset, to keep the bundle focused on the paper's headline datasets).
    sweep_stems = ["csic_eval", "access_eval_mix_2000", "access_eval_small_500", "nginx_json_eval_800"]
    for stem in sweep_stems:
        for src in sorted(EVAL_DIR.glob(f"plots/sweep_*_{stem}.png")):
            shutil.copy2(src, FIGURES_DIR / src.name)
            n_figs += 1

    # coherence/scatter for all canonical stems
    for stem in CANONICAL_STEMS:
        for pattern in (f"advanced_plots/coherence_{stem}.png", f"advanced_plots/scatter_{stem}.png"):
            src = EVAL_DIR / pattern
            if src.exists():
                shutil.copy2(src, FIGURES_DIR / src.name)
                n_figs += 1

    print(f"[OK] {n_tables}/{len(TABLE_FILES)} tables -> {TABLES_DIR}")
    print(f"[OK] {n_figs} figures -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
