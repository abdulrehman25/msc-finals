# Response to Reviewer Comments (Round 3)

Dear Editor and Reviewer,

Thank you for a careful reading of the revised manuscript. Reviewer 1 and Reviewer 2's comments (Rounds 1–2) were addressed in the previous revision and are not reopened here. For each of your five comments we re-ran the underlying analysis rather than only revising the surrounding text, and in two places (Comments 1 and 4) that work surfaced a real, previously undetected issue beyond what you asked about, which we report below rather than leave unaddressed. All changes are made directly in `Article.tex`, which recompiles to 26 pages without errors or undefined references; section, table, and equation numbers below refer to the revised manuscript.

We reproduce each comment below, followed by our response.

---

### Comment 1

> The training and testing protocol should be clarified. Since CSIC 2010 data are used for both session-model training and evaluation, the authors should ensure that the training and test samples are strictly separated to avoid potential data leakage.

**Response:** The session model's training/evaluation split was already corrected in Round 2 (that response's Comment 1): it is trained only on the first 28,800 (80%) of CSIC 2010's benign rows, pooled with Access Super Long Session 2000, and evaluated only on the disjoint 32,265-row held-out split (Section 5.1). That part of the manuscript's own text was accurate.

Investigating further, however, we found the reviewer's underlying concern was still live in two other places we had not previously checked:

1. **The content branch's own training/evaluation split.** Unlike the session model, the content-branch model deployed for every table in the paper had not been retrained since the original thesis submission (December 2025) — before the CSIC held-out split existed. We traced its actual training source: 5,116,154 rows from the NASA Kennedy Space Center HTTP server logs (July–August 1995), a public dataset entirely disjoint from CSIC 2010 and every other dataset used in this study, so there was in fact no CSIC leakage for this branch. There was a smaller, real problem instead: that training file includes a heuristically-flagged 2.85% of structurally atypical rows, and the training script has no label-filtering step, so the "trained only on requests known to be normal" claim in Section 4.4 was not literally true. We retrained the content branch on the label-filtered 4,970,260-row subset (`train_content_ae.py`, now records `trained_on`, `n_train_rows`, and training-score percentiles/mean/std in its saved config for verifiability) and added a paragraph to Section 4.4 documenting the training source, its size, and the filtering step explicitly, with a new citation (Arlitt and Williamson, 1996, the standard reference for this dataset).

2. **A residual leakage gap in the ablation, fusion-sweep, and threshold-transfer tables.** These three tables (`scripts/ablation_table.py`, `scripts/fusion_weight_sweep.py`, `scripts/threshold_transfer.py`) were still reading from `predictions_csic_eval.csv`, generated from the full, un-split 61,065-row CSIC file, even though the prose in Section 5.1 already described the corrected held-out protocol. We regenerated this file from the 32,265-row held-out split instead (keeping the `csic_eval` label used throughout the tables for continuity), fixed the two hardcoded label-file paths in `fusion_weight_sweep.py` and `threshold_transfer.py` that pointed at the full file's labels, and re-ran every downstream table and figure.

Because the content branch is the one branch scored on all eight datasets, retraining it changed content- and fused-branch numbers throughout Tables 5–9 and Figure captions, though mostly by a small amount (the new and old content models were both trained on the same underlying NASA corpus, just filtered vs.\ unfiltered). The one substantive change is the fused branch's false-positive rate: our Round 2 response had reported it as exactly 0.00 on five of seven datasets; with the corrected content model it is exactly 0.00 on two of seven (Access Eval Small 500, Access Small Benign) and no higher than 0.05 on the rest. We have corrected this claim everywhere it appeared (Table 8, its note, and Section 6), rather than leave the old, no-longer-accurate number in place. The paper's other headline findings (the session-branch fix, the threshold-transfer result, the relative ranking of baselines) are unchanged in direction, though the underlying tables were regenerated end to end and every number in Sections 5 and 6 was checked against the corrected data.

---

### Comment 2

> The claim of "cross-dataset generalisation" should be clarified. The current experiments mainly demonstrate threshold transfer across datasets. Please clearly distinguish model generalisation from threshold generalisation.

**Response:** We agree the manuscript used this phrase loosely in places. We replaced the ambiguous "cross-dataset generalisation" keyword with "cross-dataset evaluation; threshold transfer," and added a new paragraph at the start of Section 6 that names the two claims separately: *model generalisation* — the content branch's own AUC, measured independently on all eight datasets including the seven it was never trained or calibrated on, staying in a broadly similar 0.48–0.71 range across three different log formats (Table 5) — and *threshold generalisation* — the separate, narrower claim in Section 5.4 that a threshold calibrated once on CSIC 2010 transfers to three other datasets. The retraining described under Comment 1 strengthens the model-generalisation claim specifically, since the content branch is now provably trained on a source disjoint from every evaluation dataset, so its cross-dataset AUC is a clean test of generalisation rather than partly an artefact of shared training data.

---

### Comment 3

> The fusion strategy needs further justification. Since content and session anomaly scores may have different numerical distributions, score normalization or calibration should be considered before weighted fusion.

**Response:** This is correct, and our own Round 2 response (Comment 2) had already identified the scale mismatch as a limitation without fixing it. We have now implemented and evaluated it. `scripts/evaluate_file.py` and `src/api/main.py` support a `--fusion-norm {none,percentile,zscore}` option: each branch's raw score is mapped onto its own *training*-score distribution (empirical percentile via linear interpolation over saved 50th/75th/90th/95th/97th/99th percentiles, or a z-score) before averaging, using only training-set statistics so evaluation-set composition cannot bias the mapping — the same principle already established for thresholding in Section 5.4.

At the 95th-percentile operating point on CSIC 2010, the content branch's raw score is about 89 against the session branch's 1.1, a ratio of roughly 82:1 (Section 5.3), which is what drives the flat F1/precision/recall region in the fusion-weight sweep (Table 6). Recomputing the $w_c=w_s=0.5$ fusion with percentile normalization raises the fused branch's AUC from 0.76 (raw average) to 0.95 — close to the session branch's own 0.98 — and cuts the fused FPR from 0.017 to 0.001, while F1 and precision both improve rather than trade off. Z-score normalization gives a smaller but still substantial improvement (AUC 0.92). This is now reported in a new paragraph in Section 5.3 and referenced from the Discussion (Section 6) and the fusion methodology (Section 4.6). We kept the raw average as the default for the paper's other headline tables, so the existing thresholding convention used throughout Section 5 stays internally consistent, but we now explicitly recommend normalized fusion as the better choice and report the comparison directly rather than only naming it as a limitation.

---

### Comment 4

> Random Forest, Isolation Forest, and One-Class SVM are included, but the paper provides limited information about their training splits, hyperparameter selection, and operating thresholds. Moreover, comparing false-positive rates obtained from different thresholding strategies can be misleading. Please provide complete baseline settings and compare methods using PR-AUC as well as operating-point metrics such as FPR at a fixed recall/TPR or recall at a fixed FPR. Including at least one recent deep-learning or Transformer-based anomaly-detection baseline would also strengthen the evaluation.

**Response:** Baseline hyperparameters, training splits, and thresholding procedure were already documented in Section 4.9, "Classical Baseline Methods" (added in Round 2, Comment 4; renumbered from 4.8 to 4.9 in this round after an earlier subsection was given its own cross-reference label): Random Forest at a 0.5 probability cutoff; Isolation Forest and One-Class SVM at the 95th percentile of their own training-fold benign scores; all three on an 80/20 split with a fixed seed. PR-AUC was likewise already computed and reported for every method (Table 8 and the underlying `metrics_*.json`/`baseline_metrics_*.json` files use `average_precision_score` throughout).

What was missing, as the reviewer notes, is a threshold-independent operating-point comparison. We added `compute_operating_points()` to `src/pipeline/metrics_extra.py`, computing recall at fixed FPR budgets (0.01, 0.05, 0.1) and FPR at fixed recall targets (0.5, 0.8) directly from each method's ROC curve, and wired it into both our own branches (`evaluate_file.py`) and the classical baselines (`plot_and_baselines.py`), re-running all three baselines on all eight datasets to populate it. These are saved in every `metrics_*.json` and `baseline_metrics_*.json` file under a new `operating_points` field, and a `recall_at_fpr=0.05` / `fpr_at_recall=0.5` pair is now included as additional columns in `baseline_comparison.csv`, so a reader can compare methods at a matched false-positive budget rather than only at each method's own self-selected threshold, without us needing to invent a single "fair" threshold that would not actually match how each method is meant to operate.

On the Transformer baseline: we did not add one. The reviewer frames this as strengthening the evaluation rather than a required correction, and a Transformer-based anomaly detector is a substantial new modelling effort (architecture choice, training protocol, its own threshold-calibration story) rather than a rerun of existing code, which risks introducing a new, less-validated component under review-cycle time pressure. We have added it explicitly as future work in Section 6's limitations, alongside the existing note about testing the session branch on additional session-rich datasets.

---

### Comment 5

> The Author Contributions section lists "R.U." although the author list contains A.R., M.N., and M.H.; this inconsistency should also be corrected.

**Response:** Confirmed and corrected. `\authorcontributions{}` credited "R.U." (Raja Ujjan, the corresponding author's thesis supervisor), who does not appear in `\Author{}` at all, while M.N. (Mouhammad Nouman), who is listed as an author, had no contribution statement. We have changed the credited initials from R.U. to M.N. so every listed author has a contribution statement and no uncredited initials remain. *This substitution reflects our best reading of the inconsistency and should be confirmed with all co-authors before the next submission, since only they can confirm the intended contribution attribution.*

---

All changes above are implemented directly in `Article.tex`, and the manuscript compiles without errors or undefined references (26 pages). The scripts and artifacts behind every number reported are included with the submission: `train_content_ae.py`, `train_session_lstm_vae.py`, `scripts/evaluate_file.py`, `scripts/plot_and_baselines.py`, `scripts/ablation_table.py`, `scripts/fusion_weight_sweep.py`, `scripts/threshold_transfer.py`, `scripts/bootstrap_ci.py`, `src/pipeline/metrics_extra.py`, and the regenerated `artifacts/content_v2/`, `artifacts/eval/`, and `artifacts/paper/tables/` and `artifacts/paper/figures/` directories.

We thank the reviewer for a reading that caught a genuinely live methodological gap (the content branch's training-data provenance, Comment 1) that had survived two prior review rounds, and for pushing the fusion and baseline-comparison methodology (Comments 3–4) from "documented as a limitation" to "measured and fixed."
