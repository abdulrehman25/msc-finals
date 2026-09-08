# Response to Reviewer Comments (Round 3, Reviewer 4)

Dear Editor and Reviewer,

Thank you for a positive and constructive reading of the manuscript. The three "Disadvantages" you raise are concerns we had already identified and addressed explicitly in the manuscript as limitations (we point to exactly where below, rather than treat the comment as new). The two "Suggestions," however, asked for something the manuscript did not yet have: an actual check of run-to-run variance, and a named path to testing generalisation beyond CSIC 2010. We have added the first and scoped the second as concrete future work. All changes are made directly in `Article.tex`, which recompiles to 27 pages without errors or undefined references.

---

### Disadvantage 1

> The key positive result (improving session AUC from 0.63 to 0.97) is practically based on a single dataset (CSIC 2010), so generalizing conclusions about the "strength" of a session branch is limited.

**Response:** Agreed, and already stated as the first limitation in Section 6: "the session branch's strong result is demonstrated on only one dataset, CSIC 2010, because it is the only dataset in this study with enough per-client requests to build real session windows... it has not been shown to help on genuinely session-rich traffic of a different shape or scale." (Note: the manuscript now reports 0.98, not 0.97, for the corrected model — see Round 2's Comment 1 response, which explains the held-out-split correction that produced this figure.) See our response to Suggestion 2 below for what we are doing about this specific limitation now.

---

### Disadvantage 2

> Some datasets are very small (120–500 requests), which weakens the statistical reliability of the reported metrics.

**Response:** Agreed, and already addressed on two fronts. It is named directly as the second limitation in Section 6, and — more substantively — Round 2's Comment 5 added 2,000-resample bootstrap 95% confidence intervals for exactly this reason (Table 9, Section 5.5), which show that most of the point-estimate comparisons against classical baselines at these small sample sizes are not statistically distinguishable and should be read as indicative rather than settled.

---

### Disadvantage 3

> The features used by the model are purely structural (path length, status code, etc.) and do not analyze the request content — the model may miss attacks not visible at the structural level.

**Response:** Agreed, and already stated as a deliberate, named scope choice rather than an oversight, in both the Methods (Section 4.3, immediately after the feature table) and Discussion (third limitation, Section 6): the content branch "is not well suited to attacks in which the request looks structurally ordinary but carries a malicious value inside the body or a query parameter, such as a syntactically well-formed SQL injection or cross-site-scripting payload." The future-work paragraph in Section 6 also names extending the feature set with lightweight payload signals as a next step.

---

### Suggestion 1

> Consider adding cross-validation or running the experiments multiple times to assess the variance of the results.

**Response:** This was a genuine gap: the manuscript previously *admitted* single-run variance was unchecked (the fifth limitation in Section 6) rather than checking it. We have now closed it. We retrained both the content and session branches from scratch five times, using five independent random seeds, and re-evaluated every retrained pair on all seven labelled datasets — a new Results subsection, "Training-Run Variance" (Section 5.7), reports the resulting AUC as mean ± standard deviation (new Table 10).

The result is reassuring: variance is small everywhere, and smallest exactly where it matters most for the paper's central claim. The session branch's AUC on CSIC 2010 is 0.972 ± 0.010 across five independently trained models, so the single-run figure of 0.98 used throughout the Results section sits well within one standard deviation of the five-run mean — not an unrepresentative extreme. The content branch's AUC varies even less (standard deviation ≤ 0.002 on every dataset), despite the five retrained models having verifiably different learned weights (we confirmed this by comparing model checkpoints directly, to rule out the seed silently not taking effect). This directly answers the specific concern raised in Disadvantage 1: the reported session-branch improvement is not an artefact of one favourable random initialisation. We have been careful in the text (Section 5.7 and the revised fifth limitation in Section 6) to state what this does and does not establish — it is evidence of *training-run stability*, not of *cross-dataset generalisation*, which remains the separate, open question addressed by Suggestion 2 below.

We did not additionally run k-fold cross-validation, since both branches are trained semi-supervised on a fixed benign-only split whose held-out evaluation set must stay disjoint from training (Section 5.1); k-fold splitting the training data would either shrink the held-out evaluation set or reintroduce the training/evaluation overlap corrected in Round 2. Five-seed retraining tests the same underlying question — is the point estimate representative? — without that trade-off.

---

### Suggestion 2

> If possible, test the model on additional representative datasets with actual session structure to see if the CSIC 2010 result generalizes.

**Response:** We looked for a suitable dataset rather than treat this as impossible in principle. The requirement is unusual: a public dataset needs *both* genuine multi-request session structure (repeated requests from the same client, identifiable via IP/user-agent) *and* real, labelled attack traffic, and the two rarely coincide — organic traffic captures (like the NASA HTTP logs now used to train our content branch, Section 4.4) have session structure but no genuine attacks, while most labelled attack datasets (like CSIC 2010 itself) are constructed rather than organic and rarely have enough repeated-client structure.

We identified one concrete, recently published candidate that plausibly satisfies both requirements: Biblio-US17, a labelled, real-world HTTP log dataset from a university library website (47 million requests over six months, published in *Cybersecurity*, 2025). We have added it by name, with citation, to the future-work paragraph in Section 6. We have not attempted to acquire and integrate it in this revision round: its publicly documented field list does not confirm whether client identifiers survive its anonymisation process, which would need to be verified before it could support session-window construction at all, and integrating a new 47-million-row dataset into the pipeline (parsing, labelling convention, and a fresh session-structure audit like the one in Section 5.1) is a substantial undertaking better scoped as a follow-up study than fitted into the current review cycle. We would rather flag this honestly as a named, verifiable next step than claim a generalisation result we have not actually established.

---

All changes above are implemented directly in `Article.tex` (new Section 5.7, new Table 10, revised fifth limitation in Section 6, new future-work sentence with citation), and the manuscript compiles without errors or undefined references (27 pages). The scripts and artifacts behind the new numbers are included: the five retrained model checkpoints (`artifacts/content_v2_seed{42,7,13,99,123}/`, `artifacts/session_v3_seed{42,7,13,99,123}/`), their per-seed evaluation outputs (`artifacts/eval_seed{42,7,13,99,123}/`), and the aggregated variance table (`artifacts/eval/seed_variance.csv`).

We thank the reviewer for a comment that turned an acknowledged gap (run-to-run variance) into an actual measurement, and for pushing us to name a concrete, checkable path for the cross-dataset generalisation question rather than leave it as an open-ended limitation.
