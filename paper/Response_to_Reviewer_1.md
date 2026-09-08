# Response to Reviewer Comments

We thank the reviewer for the constructive comments. Below we reproduce each comment and describe the corresponding revision. All changes have been made directly in `Article.tex`; section and line numbers refer to the revised manuscript.

---

**Comment 1: Title should reflect the work also.**

Response: We have shortened and refocused the title. The previous title ("Content- and Session-Aware Anomaly Detection for Web Server Logs Using a Fused Autoencoder and Variational LSTM Framework: A Cross-Dataset Evaluation") was long and led with the evaluation protocol rather than the method. The revised title is:

"Hybrid Content- and Session-Based Anomaly Detection for Web Server Logs"

(Article.tex, line 55). This leads with the two-branch hybrid design, which is the paper's central contribution, and drops the subtitle so the title itself is easier to parse.

---

**Comment 2: Abstract is not consistent.**

Response: The abstract has been rewritten (Article.tex, line 79) to follow a fixed order: problem, method, dataset/evaluation setup, main findings, main takeaway. Previously the abstract mixed a limitation ("scores nothing for clients with only one or two requests") into the middle of the summary of results, which broke the flow between the method description and the findings. This has been moved out of the summary sentence and rephrased as a direct property of the session branch, stated once and immediately followed by the fusion and threshold-transfer findings in the order they appear in the Results section. The abstract now closes with a single takeaway sentence, matching the structure of the Conclusions.

---

**Comment 3: Features built from each parsed request has not been justified.**

Response: We added a justification paragraph directly after Table `tab:features` (Section 4.3, "Data Preprocessing and Feature Extraction"). It explains three reasons for the feature choice: (1) all 11 features can be read from fields common to Apache Combined, Apache Common, and structured JSON logs, so one feature pipeline works across all three formats; (2) they are cheap to compute, which keeps per-request scoring fast enough for the near-real-time results in Section 5; (3) restricting features to structural properties (not payload/query content) avoids tying the model to any one application's schema. We also now state explicitly what this buys and what it costs: the feature set is well suited to attacks that change a request's shape (unusual paths, methods, status/size patterns, scanning behaviour) and not well suited to attacks where the request is structurally ordinary but carries a malicious payload (e.g., a well-formed SQL injection string), a trade-off that is cross-referenced to the Discussion (Section 6) limitations.

---

**Comment 4: Fusion and Anomaly Scoring has not been explained.**

Response: Section 4.6 ("Fusion and Anomaly Scoring") has been expanded. We now state explicitly that both branch scores are the same quantity — mean squared reconstruction error, computed identically for the content branch (Section 4.4) and the session branch (Section 4.5) — which is why they are combined directly with no separate normalisation step. We clarify the fallback rule (if only one branch has a score, the fused score is that branch's score, not a missing value), and we justify weighted averaging over alternatives such as taking the maximum score or learning a fusion weight from labelled data: it needs no attack labels to fit, it is simple to compute, and the fused score stays interpretable as a direct blend of two comparable quantities. We also added a one-sentence definition of the session score to Section 4.5 (previously the score itself was never stated, only the training loss), so both branch scores are now defined before they are combined.

---

**Comment 5: Session branch AUC and F1 on CSIC 2010 has not been validated.**

Response: We added an explicit statement at the start of Section 5.1 ("Effect of the Session Construction Fix") establishing why CSIC 2010 is the correct dataset for this validation: it is the only one of the eight datasets where enough requests come from the same client to build genuine 20-request session windows (almost all of its 61,065 requests come from the same two clients, as detailed in Section 4.5). (The specific training-window count cited here in the original Round 1 response, 7,594, was from the session model as it stood before the Round 2 held-out-split correction; the current, corrected session model is trained on 6,154 windows, pooled from Access Super Long Session 2000 and CSIC 2010's 80% benign training split, per `Response_to_Reviewer_2.md` Comment 1.) This is stated as the reason the AUC/F1 improvement can be attributed to the corrected model rather than to an absence of session structure elsewhere. We also added an explicit caveat at the end of that subsection: because CSIC 2010 is the only dataset in this study with genuine multi-request session structure, the result validates the fix on a session-rich dataset, and it does not by itself establish that the same gain would appear on other traffic with a different session-length distribution. This caveat is cross-referenced to the Discussion (Section 6), where it is listed as a limitation and as a direction for future work (testing on additional session-rich datasets).

---

**Comment 6: AUC by dataset and branch has not been clarified and needs more justification.**

Response: We added an explanatory paragraph immediately before Table `tab:ablation` in Section 5.2 ("Branch Ablation") that makes three structural points explicit before the numbers are read: (1) the content branch scores every request from its own features alone, so its AUC differences across datasets reflect differences in how separable normal and attack traffic are in each dataset, not differences in data volume; (2) the session branch only produces a value where a client sends at least 20 consecutive requests, which is why the session column is populated only for CSIC 2010; (3) the fused branch reduces exactly to the content branch wherever no session score exists, by the fallback rule defined in Section 4.6. We also state directly that AUC is reported because it is threshold-independent (it measures ranking quality, not alerts raised by a specific percentile cutoff), and we add an explicit caveat that comparing AUC across datasets is informative but not a controlled comparison, since the datasets differ in size, format, and class balance (Table `tab:datasets`). The heatmap figure caption (Figure `fig:heatmap_auc`) was also updated to state directly that the session column is blank for every dataset except CSIC 2010 because none of the others has a client with enough consecutive requests to form a valid session window.

---

**Comment 7: Conclusion is like a story and needs to have outcomes clearly.**

Response: The Conclusions section (Section 7) has been rewritten into four short, function-specific paragraphs instead of three narrative-style paragraphs: one paragraph stating the method and evaluation design, one paragraph stating the three key findings directly (with supporting numbers retained), one paragraph stating the single practical implication, and one paragraph stating the limitations and future work. Repeated framing language and restated context from earlier sections have been removed so each paragraph adds new information rather than re-arguing a point already made.

---

We also carried out a general language pass, removing informal, conversational asides in the Results and Discussion sections (e.g., "The result is the opposite of what we expected going in," "it is worth stating plainly rather than glossing over," "This is a useful, real finding") in favour of direct statements, and confirmed the revised manuscript compiles without errors.
