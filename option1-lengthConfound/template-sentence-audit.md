# Sentence-by-Sentence Audit — `template.tex`

> Working doc. Each sentence is one row. Tick `[x]` as we go. Reference convention: `<section-tag> S<n>` (e.g. `intro.p2 S3`, `methods.protocol S5`).

---

# Title (`title`)

- [ ] **T1.** Length-Controlled Evaluation of Training-Free Dynamic-RAG Triggers

---

# Abstract

- [ ] **S1.** Dynamic retrieval-augmented generation systems decide *when* to retrieve from a scalar trigger signal, scored by raw AUROC against binary correctness.
- [ ] **S2.** A converging recent literature has shown this protocol is length-confounded on adjacent QA selective-prediction and UQ tasks, yet most dynamic-RAG trigger evaluations still use it.
- [ ] **S3.** We present the first length-controlled audit of the published training-free dynamic-RAG trigger pool under length-residualized AUROC on a paired-oracle target (`lift_em`=+1: retrieval rescue).
- [ ] **S4.** On two open-weight chat models over two multi-hop QA datasets (n=800 paired traces), nine signals beat the length baseline at 95% paired-bootstrap, and four survive Bonferroni at k=15 across three signal families.
- [ ] **S5.** We propose **FCT** (first-commitment time), a forward-looking variant of the answer-trajectory primitive that flags traces where the model commits to its final answer only late in the decode; FCT exceeds the length baseline by Δ = +0.190 [+0.093, +0.287], the strongest length-orthogonal signal.
- [ ] **S6.** On the held-out 2Wiki cells, raw answer length is *anti*-predictive of retrieval need (AUROC ≤ 0.52), inverting the verbosity-compensation pattern documented on adjacent QA tasks.
- [ ] **S7.** The survivor set is **robust to a semantic-equivalence judge**: under a GPT-4o-Mini correctness function (vs. EM, κ = 0.707), eight of nine survivors are preserved and FCT *strengthens* to Δ = +0.222.

---

# Figure 1 caption (`fig:survivor-forest`)

- [ ] **S1.** **Four trigger signals carry length-orthogonal information beyond the length baseline; the strongest is FCT (first-commitment time, this paper).**
- [ ] **S2.** Paired-bootstrap forest plot of Δ = AUROC(s_resid) − AUROC(`tokens_generated`) on n=800 paired traces (Qwen2.5-7B + Gemma-4-26B-A4B-it × {HotpotQA, 2WikiMHQA}, B=10,000).
- [ ] **S3.** Thick bars: 95% paired-bootstrap CIs; thin extensions: Bonferroni-corrected ~99.67% CIs at k=15.
- [ ] **S4.** Setup: §methods; full numerical tables: App. bonferroni, App. hero.

---

# 1. Introduction

## intro.p1 — peer-pool framing

- [ ] **S1.** Dynamic retrieval-augmented generation (RAG) systems retrieve external evidence during decoding to ground generation in source material `\citep{lewis_retrieval-augmented_2021, trivedi_interleaving_2023, shao_enhancing_2023}`.
- [ ] **S2.** The control point of every such system is the *retrieval trigger*: a scalar function of the partial generation that decides when to invoke retrieval.
- [ ] **S3.** The published peer pool of training-free triggers spans six signal families that each read a different decode-time observable: attention-based DRAGIN `\citep{su_dragin_2024}` (token importance from attention dynamics), token-probability FLARE `\citep{jiang_active_2023}` (next-token entropy / max log-prob at the decode head), entropy-trend ETC `\citep{li_modeling_2026}` (second-difference of step entropy), hidden-state SeaKR `\citep{yao_seakr_nodate}` (regularized Gram determinant of K sample EOS hidden states) with INSIDE `\citep{chen_inside_2024}` as the EigenScore variant, multi-sample text-consistency RSUS `\citep{guo_realm_2026}` (cross-sample step-boundary continuation agreement), and hedging-marker SELFDOUBT-T1 `\citep{pandey_selfdoubt_2026}` (hedge-to-verify ratio over the reasoning trace text).
- [ ] **S4.** The contemporaneous answer-trajectory family `\citep{datta_large_2026, wang_chain--probe_nodate, chen_i_2026}` (step-level dynamics of the model's predicted answer over the decode) extends this pool with step-level signals.
- [ ] **S5.** All seven follow the same evaluation protocol (rank traces by the trigger score; report raw AUROC against a binary correctness label); the six published peer families predate the length-confound literature.

## intro.p2 — length-confound literature

- [ ] **S1.** A converging recent literature has shown the standard AUROC protocol is length-confounded on adjacent QA selective-prediction and UQ tasks.
- [ ] **S2.** `\citet{santilli_revisiting_2025}` prove that shared length confounding systematically distorts AUROC, with Spearman ρ = −0.7 between length and Perplexity.
- [ ] **S3.** `\citet{janiak_illusion_nodate}` report Perplexity AUROC drops of up to 45.9% and EigenScore drops of up to 30.4% when switching ROUGE to LLM-as-Judge on certain Mistral cells; `\citet{zhang_verbosity_2024}` document the behavioral mechanism (verbose responses when uncertain) on adjacent QA tasks including the multi-hop bundle in LongBench; and `\citet{ielanskyi_addressing_2025}` show that approximate correctness functions disagree on UQ rankings.
- [ ] **S4.** `\citet{vashurin_uncertainty-line_2025}` propose Uncertainty-Line, a regression-residualization protocol applied to Neural Machine Translation (NMT), summarization, and mathematical reasoning; dynamic-RAG triggers remain an untested application domain.

## intro.p3 — our contribution

- [ ] **S1.** We present the first length-controlled audit of the published training-free dynamic-RAG trigger pool under length-residualized AUROC against a paired-oracle target, `lift_em`=+1, which labels each closed-book trace by whether re-decoding under gold context flips a wrong answer to a right one `\citep{mallen_when_2023}`.
- [ ] **S2.** The paired design shares the same exact-match correctness function across both arms, which partially cancels the function-bias critique of `\citet{ielanskyi_addressing_2025}`.
- [ ] **S3.** We propose **FCT** (first-commitment time), a forward-looking variant of the answer-trajectory primitive `\citep{datta_large_2026, wang_chain--probe_nodate, chen_i_2026, zhang_commit_2026}`; the operational distinction from `\citeauthor{zhang_commit_2026}`'s concurrent commitment-time primitive (which projects the continuation distribution onto a finite answer set) is detailed in App. fct-tstar.
- [ ] **S4.** FCT is the strongest length-robust trigger in the audited pool, exceeding the length baseline by Δ = +0.190 AUROC (Bonferroni-significant at k=15).
- [ ] **S5.** On the held-out 2Wiki cells, length is itself anti-predictive of retrieval need, inverting the verbosity-compensation pattern documented on adjacent QA tasks.

## intro.contrib — Contributions itemize

- [ ] **I1.** A length-controlled audit of the published training-free dynamic-RAG trigger pool (the first to our knowledge), identifying four length-robust signals from three signal families (attention, token-probability, and the emerging answer-trajectory family) not previously contrasted under the length-residualized AUROC protocol.
- [ ] **I2.** We propose **FCT** (first-commitment time), a forward-looking variant of the answer-trajectory primitive, exceeding the length baseline by Δ = +0.190 [+0.093, +0.287] AUROC (Bonferroni lower bound +0.047) and reaching raw AUROC **0.784** on the held-out Gemma×2Wiki cell.
- [ ] **I3.** Cross-architecture and cross-domain validation on one dense (Qwen2.5-7B) and one MoE-hybrid (Gemma-4-26B-A4B-it) open-weight chat model over two multi-hop QA benchmarks (n=800 paired traces; base rate 13.5%): nine signals clear length at 95% paired-bootstrap and four survive Bonferroni at k=15.
- [ ] **I4.** A direct robustness check against the LLM-as-judge critique of `\citet{janiak_illusion_nodate}`: under a GPT-4o-Mini semantic-equivalence judge agreeing with EM at Cohen's κ = 0.707, eight of nine paired-significant survivors and all four Bonferroni-robust signals are preserved; FCT strengthens to Δ = +0.222.

---

# 2. Related Work

- [ ] **S1.** Length confounds the standard AUROC protocol for LLM uncertainty quantification `\citep{santilli_revisiting_2025, vashurin_uncertainty-line_2025, janiak_illusion_nodate, zhang_verbosity_2024, ielanskyi_addressing_2025}`; `\citet{mao_confidence_2026}` note that joint generation probability suffers severe length bias.
- [ ] **S2.** Step-level answer-trajectory triggers are an emerging length-orthogonal signal class `\citep{datta_large_2026, wang_chain--probe_nodate, chen_i_2026, zhang_harnessing_2026, lu_streaming_2026}`.
- [ ] **S3.** The training-free trigger pool we audit spans attention-based DRAGIN `\citep{su_dragin_2024}` and DR-RAG `\citep{hei_dr-rag_2024}`, token-probability FLARE `\citep{jiang_active_2023}`, multi-sample text-consistency RSUS `\citep{guo_realm_2026}` (with related approaches SelfCheckGPT `\citep{manakul_selfcheckgpt_2023}` and SAC³ `\citep{zhang_sac3_2023}`), hidden-state INSIDE `\citep{chen_inside_2024}` and SeaKR `\citep{yao_seakr_nodate}`, hedging-marker SELFDOUBT `\citep{pandey_selfdoubt_2026}`, and entropy-trend ETC `\citep{li_modeling_2026}`.
- [ ] **S4.** Trained alternatives that require per-architecture supervised data `\citep{asai_self-rag_2023, tang_mba-rag_2024, chen_think_2026, guo_dior_2025}` fall outside our training-free scope.
- [ ] **S5.** The conceptual parent of our paired-oracle target is `\citet{mallen_when_2023}`, who asks when retrieval helps a parametric model.

---

# 3. Methods

## 3.1 Paired-Oracle `lift_em` Protocol (`methods.protocol`)

- [ ] **S1.** For each question q in the evaluation set, we run two greedy decodes on the same chat model: a closed-book decode that produces trace τ_c^(q) and answer a_c^(q) with no retrieval context, and an oracle-retrieval decode conditioned on the gold supporting paragraphs that produces a_o^(q).
- [ ] **S2.** We extract all trigger signals from τ_c^(q); the oracle decode only labels the trace.
- [ ] **S3.** The discrimination target is **[Eq. 1] `lift_em(q) = 1[EM(a_o^(q)) > EM(a_c^(q))]`**.
- [ ] **S4.** *i.e.*, a trace receives label +1 exactly when the closed-book answer is wrong and the gold-context answer is right --- precisely the subpopulation where retrieval would convert a failure into a success.
- [ ] **S5.** This is a strictly different question from the single-arm "which traces are wrong?" question that the published trigger pool actually evaluates against (and that length predicts moderately well on multi-hop QA; §discussion-causal): we instead ask, of the queries that the model could not solve unaided, which ones would retrieval rescue?
- [ ] **S6.** A signal that ranks `lift_em`=+1 traces high is, by construction, a retrieval trigger --- thresholding it at a high percentile recovers exactly the retrieval-helpful subpopulation.
- [ ] **S7.** The paired design shares the same exact-match correctness function across both arms, which partially insulates against the function-bias critique of `\citet{ielanskyi_addressing_2025}`.
- [ ] **S8.** We evaluate on HotpotQA `\citep{yang_hotpotqa_2018}` and 2WikiMultiHopQA `\citep{ho_constructing_2020}` --- the latter held out for cross-dataset evaluation --- at n=200 paired traces per architecture-by-dataset cell.
- [ ] **S9.** We select one dense (Qwen2.5-7B-Instruct) and one MoE-hybrid (Gemma-4-26B-A4B-it) open-weight chat model to test cross-architecture generality within a single 80 GB GPU compute budget.
- [ ] **S10.** The positive class is rare (108 of 800 paired traces, base rate 13.5% across the headline pool).

## 3.2 Length Residualization (`methods.resid`)

- [ ] **S1.** For each trigger signal s on a cohort of paired traces, we compute the length-residualized score **[Eq. 2] `s_resid = s − α̂ − β̂ · log(tokens + 1)`**.
- [ ] **S2.** where α̂, β̂ are OLS coefficients fit on the cohort.
- [ ] **S3.** AUROC(s_resid) measures whether the signal carries information beyond length.
- [ ] **S4.** We use log(tokens+1) rather than raw length L because multi-hop reasoning traces have a long-tailed token distribution; a linear-in-log L regression is the natural homoscedastic counterpart of `\citeauthor{vashurin_uncertainty-line_2025}`'s linear-in-L regression on their shorter NMT and summarization traces.
- [ ] **S5.** Both parameterizations agree within ε ≤ 0.029 AUROC against the public Uncertainty-Line reference implementation on both architectures (Appendix uline); higher-order polynomial regressors give no further improvement, consistent with `\citeauthor{vashurin_uncertainty-line_2025}`'s App. D.1 finding that the linear fit is sufficient.
- [ ] **S6.** ETC, FLARE, the SeaKR primitive, and SELFDOUBT-T1 are evaluated as trace-level summary statistics rather than online triggers; granularity adapters relative to the originating papers are detailed in Appendix adapters.

## 3.3 FCT and the Answer-Trajectory Family (`methods.fct`)

- [ ] **S1.** At each reasoning-step boundary (detected by spaCy sentence segmentation with a 32-token fallback), we sample K=5 boundary completions at temperature τ=0.6, top-p 0.9, and extract per-step answers via a regex over the model's `Answer:` prefix.
- [ ] **S2.** From the resulting trajectory M = (m_1, ..., m_n) of per-step K=5 majority answers, we define four trace-level aggregations:
- [ ] **I1.** `flip_rate`: fraction of consecutive-step majority disagreements;
- [ ] **I2.** `late_flips`: count of consecutive-step disagreements between the per-step K=5 answer *sets* (any change in the multiset of sampled answers, not only majority shifts), restricted to the last half of the trace; we use `late_flips` as a raw-incidence counterpart to the fraction-based `flip_rate`;
- [ ] **I3.** `min_majority`: 1 − min_t (max_count/K), the worst within-step disagreement;
- [ ] **I4.** **FCT**, a forward-looking first-commitment time **[Eq. 3] `FCT = min{i : m_j = a_final ∀ j ≥ i} / max(1, n−1) ∈ [0, 1]`**.
- [ ] **S3.** High FCT denotes extended deliberation: the trace has not converged on its final answer until late in the decode.
- [ ] **S4.** These aggregations adapt the `AnswerSwitches` primitive of `\citet{datta_large_2026}` to non-thinking-mode chat models; Appendix fct-tstar tabulates the operational differences from Datta's t* and `\citeauthor{zhang_commit_2026}`'s concurrent commitment-time primitive.
- [ ] **S5.** A K=1 vs. K=5 ablation of `flip_rate` is null on both architectures (§results-ablation), and we recommend K=1 for deployment efficiency.

### 3.3.x CABV paragraph (`methods.cabv`)

- [ ] **S1.** We additionally compute **CABV**, a continuous between-step generalization of `flip_rate`: at each populated step t, build the Laplace-smoothed empirical distribution p_t over the union of K=5 answers ever observed in the trace, and average the Jensen–Shannon divergence between consecutive populated steps, **[Eq. 4] `CABV = (1/(|P|−1)) Σ_t JSD(p_t, p_{t+1})`**.
- [ ] **S2.** where P is the set of populated step indices.
- [ ] **S3.** CABV is high on traces whose answer-belief mass continues to redistribute through the decode; it reduces to `flip_rate` when each step is a delta on a single answer, and is conceptually closest to the inference-time instability signal of `\citet{chen_i_2026}` (which uses JSD over token distributions rather than over K=5 answer distributions).

---

# 4. Results

## 4.1 Cross-Architecture Survivor Matrix (`results.matrix`)

- [ ] **S1.** Nine signals clear length at 95% and four survive Bonferroni at k=15 across three signal families (Figure survivor-forest): attention-based DRAGIN-RIND (Δ = +0.167), token-probability `ent.max` (+0.137), answer-trajectory `min_majority` (+0.149), and the forward-looking commitment-time signal **FCT** (+0.190, the strongest length-robust trigger in the audited pool).
- [ ] **S2.** Confidence intervals are paired-bootstrap at nominal 95% and Bonferroni-corrected ~99.67% (α=0.0033, B=10,000).
- [ ] **S3.** The full nominal-95% set of nine survivors spans five of the seven declared families (attention, token-probability, entropy-trend, text-consistency, answer-trajectory); hidden-state and hedging-marker (SELFDOUBT-T1) have no nominal-95% survivor.
- [ ] **S4.** The full table of all 16 signals is in Appendix hero.

## 4.2 Length Is Anti-Predictive on the Held-Out 2Wiki Cells (`results.inverted`)

- [ ] **S1.** On the held-out 2Wiki cells (Qwen × 2Wiki and Gemma × 2Wiki), raw answer length is *anti*-predictive of retrieval need: the `tokens_generated` baseline reaches raw AUROC **0.457** and **0.518** on `lift_em`=+1, both at or below chance.
- [ ] **S2.** Longer traces on the held-out multi-hop cells are systematically *less* retrieval-helpful, inverting the verbosity-compensation pattern `\citet{zhang_verbosity_2024}` document on adjacent QA tasks.
- [ ] **S3.** The four Bonferroni-robust survivors clear the length baseline by a wider margin on these cells than on the HotpotQA cells, suggesting the raw-AUROC protocol miscredits the length signal on exactly the cells where retrieval matters most.

## 4.3 FCT Discriminates Strongest on the Held-Out MoE Cell (`results.fct-percell`)

- [ ] **S1.** On the same held-out cells where length fails (§results-inverted), FCT discriminates most strongly --- reaching raw AUROC **0.784** on the Gemma × 2Wiki cell (Table fct-percell).
- [ ] **S2.** The architecture-by-dataset variance is consistent with cell-dependent retrieval-helpfulness structure (§discussion): the cells where the length proxy collapses are precisely the cells where the answer-trajectory primitive carries the most independent information.
- [ ] **C1.** (Table caption) Per-cell raw AUROC of FCT on `lift_em`=+1.

## 4.4 Survivor Set Is Robust to a Semantic-Equivalence Judge (`results.llmjudge`)

- [ ] **S1.** The survivor matrix is robust to a semantic-equivalence judge.
- [ ] **S2.** We relabelled both arms of every paired trace via a GPT-4o-Mini judge (Janiak's exact choice) and re-ran the full survivor matrix, addressing the correctness-function critique of `\citet{janiak_illusion_nodate}` and `\citet{ielanskyi_addressing_2025}`.
- [ ] **S3.** EM and the judge agreed at Cohen's κ = 0.707 (substantial agreement; `\citealp{landis_measurement_1977}`).
- [ ] **S4.** Eight of the nine paired-significant survivors under EM remained significant under the judge, with one out-in swap (`etc` loses, `flare.max` gains) that does not affect the Bonferroni-robust four.
- [ ] **S5.** FCT strengthened under the judge from Δ = +0.190 [+0.093, +0.287] to Δ = +0.222 [+0.132, +0.312] (Appendix llmjudge).

## 4.5 The K=5 Multi-Sample Variant Is Null (`results.ablation`)

- [ ] **S1.** The K=5 boundary-completion variant carries no significant advantage over the single-sample K=1 proxy at our power: paired-bootstrap CIs straddle zero on both architectures (Table k-ablation).
- [ ] **S2.** We recommend K=1 for deployment at 5× less compute per boundary, with equivalent triggering performance.
- [ ] **C1.** (Table caption) K=5 vs. K=1 `flip_rate` ablation on `lift_em`=+1, pooled per architecture.

## 4.6 Robustness Summary (`results.robustness`)

- [ ] **S1.** The headline finding (four Bonferroni-robust signals with FCT strongest) is invariant under three independent design choices: (i) the choice of correctness function (EM vs. a GPT-4o-Mini semantic-equivalence judge agreeing with EM at κ = 0.707; §results-llmjudge); (ii) the choice of sample size in the answer-trajectory family (K=1 vs. K=5 on `flip_rate` is null on both architectures; §results-ablation); and (iii) the choice of evaluation dataset (HotpotQA vs. held-out 2WikiMHQA; FCT discriminates strongest on the held-out Gemma × 2Wiki cell at raw AUROC **0.784**; §results-fct-percell, §results-inverted).
- [ ] **S2.** The residualization protocol itself agrees with the Uncertainty-Line reference implementation of `\citet{vashurin_uncertainty-line_2025}` within ε ≤ 0.029 AUROC on both architectures (App. uline).

---

# 5. Discussion

## 5.1 Why Length Predicts Retrieval-Helpfulness (`discussion.causal`)

- [ ] **S1.** On the secondary `closed_em`=0 target, `tokens_generated` reached raw AUROC 0.539 on the headline pool and was the single strongest pre-residualization signal on several cells.
- [ ] **S2.** This is consistent with the *verbosity compensation* mechanism documented by `\citet{zhang_verbosity_2024}`, where verbose-when-uncertain is a real LLM behavior on the multi-hop QA family represented in LongBench (which includes HotpotQA, MuSiQue, and 2WikiMHQA).
- [ ] **S3.** A plausible causal chain underlying this pattern is length → verbosity compensation → uncertainty → incorrectness → retrieval need: under this picture, length confounds the evaluation because it proxies the proximal cause, and the trigger pool's signals receive partial AUROC credit for tracking length because length correlates with the deeper uncertainty mechanism.
- [ ] **S4.** Residualization (§methods-residualization) removes the length pathway and isolates the residual signal.

## 5.2 Why Answer-Trajectory Stability Predicts Retrieval Need (`discussion.fct`)

- [ ] **S1.** The mechanism behind FCT reads naturally as *premature commitment*, the interpretation `\citet{datta_large_2026}` give to their `AnswerSwitches` and t* primitives: the model decides on an answer early and then continues to verbalize reasoning that supports the commitment, with most subsequent steps being post-decision tail.
- [ ] **S2.** `\citeauthor{datta_large_2026}` apply this observation to early stopping (if the model has decided, no further compute is needed).
- [ ] **S3.** Retrieval triggering reads the same primitive in the opposite causal direction: traces where the model has *not* decided, where the answer trajectory continues to flip through the reasoning, are the traces where external evidence would resolve the uncertainty.
- [ ] **S4.** The paired-oracle target `lift_em`=+1 formalizes this question directly.
- [ ] **S5.** The same primitive supports two complementary operations: detect stabilization to *stop*; detect non-stabilization to *retrieve*.

---

# 6. Conclusion

- [ ] **S1.** We presented the first length-controlled audit of the published training-free dynamic-RAG trigger pool under length-residualized AUROC on a paired-oracle target (`lift_em`=+1).
- [ ] **S2.** On two open-weight chat models over two multi-hop QA datasets, nine signals beat the length baseline at 95% paired-bootstrap and four survived Bonferroni at k=15 across three signal families (attention, token-probability, answer-trajectory).
- [ ] **S3.** The strongest length-robust trigger is **FCT**, a forward-looking variant of the answer-trajectory primitive.
- [ ] **S4.** The survivor set is robust to a GPT-4o-Mini semantic-equivalence judge (κ = 0.707, eight of nine survivors preserved).
- [ ] **S5.** Length-residualized and raw AUROC disagree most on the cells where retrieval matters most, motivating both protocols in dynamic-RAG trigger comparisons on multi-hop QA.

---

# 7. Limitations

## 7.1 Signal-level audit, not end-to-end pipelines (`lim.signal`)

- [ ] **S1.** We audit trigger *signals* as defined in their original papers, not full deployment pipelines (threshold tuning, retriever choice, prompt format).
- [ ] **S2.** Our claims pertain to signal-level discriminative power on `lift_em`.

## 7.2 Bonferroni knife-edge (`lim.bonferroni`)

- [ ] **S1.** Two of the four Bonferroni-robust survivors carry ~99.67% lower bounds within 5×10⁻³ of zero (`ent.max` LB = +0.001; `min_majority` LB = +0.004, App. bonferroni).
- [ ] **S2.** The headline "four Bonferroni-robust signals across three signal families" claim survives the loss of either signal as "three across two," but the four-count is fragile to small bootstrap-seed or signal-set perturbations; the FCT and DRAGIN-RIND survivors are robust (+0.047 and +0.039 respectively).
- [ ] **S3.** We report this transparently rather than tightening B from 10,000 to widen the margin.

## 7.3 Architecture and dataset coverage (`lim.coverage`)

- [ ] **S1.** The headline pool covers two architectures (Qwen2.5-7B dense, Gemma-4-26B-A4B-it MoE-hybrid) and two multi-hop QA benchmarks.
- [ ] **S2.** The cheapest extensions that would tighten the cross-architecture claim are a third open-weight chat model (e.g., Llama-3.1-8B-Instruct) and a third multi-hop benchmark (MuSiQue, `\citealp{trivedi_musique_2022}`); we report the current Bonferroni lower bound on FCT (+0.047, App. bonferroni) as the statistical floor that additional cells would have to displace.

## 7.4 Dataset framing: held-out, not OOD (`lim.framing`)

- [ ] **S1.** We refer to 2WikiMultiHopQA as a *held-out* cell relative to HotpotQA, not an out-of-distribution benchmark in the strict sense: both are multi-hop QA over Wikipedia with comparable question complexity, and the cross-dataset gap is closer to a transfer-evaluation gap than the kinds of distribution shift studied in the formal OOD-detection literature `\citep{lu_streaming_2026, ielanskyi_addressing_2025}`.
- [ ] **S2.** The cell-dependent length-behavior we report (§results-inverted) is a finding about dataset-specific length-confound structure on multi-hop QA, not a claim about OOD generalization; a stricter cross-distribution evaluation (different task family, different domain, or a synthetic perturbation set) is future work.

## 7.5 Correctness function (`lim.correctness`)

- [ ] **S1.** We use exact-match equality on both arms of the paired-oracle target; the paired design shares the same EM function across arms (§methods-protocol), and Appendix llmjudge reports an LLM-as-Judge robustness check (κ = 0.707, eight of nine survivors preserved).
- [ ] **S2.** Containment-match and F₁ variants remain future work.

## 7.6 Non-thinking-mode chat models (`lim.nonthinking`)

- [ ] **S1.** The audited models are non-thinking-mode chat models.
- [ ] **S2.** `\citet{datta_large_2026}` developed their original primitive on thinking-mode reasoning models with explicit `</think>` tokens; our K=5 boundary-completion adaptation is not byte-faithful to that protocol.

## 7.7 Integration as contribution (`lim.integration`)

- [ ] **S1.** Our contribution is the integration of length-residualized AUROC, the paired-oracle `lift_em` target, and the answer-trajectory family including FCT.
- [ ] **S2.** Each piece individually is incremental over its parent; the integration into a single length-controlled audit, and the empirical identification of length-robust survivors, are the load-bearing claims.

## 7.8 Risks (`lim.risks`)

- [ ] **S1.** Our work concerns evaluation methodology for retrieval triggering, on open-weight chat models and public multi-hop QA benchmarks; we release no new datasets or trained models, and direct risks are limited.
- [ ] **S2.** One methodological risk is that the length-residualized protocol could be cited in isolation to defend deployed dynamic-RAG systems whose triggers track length more strongly than residualization alone would reveal.
- [ ] **S3.** We therefore recommend reporting raw, length-residualized, and LLM-judge-validated AUROC together (§results-llmjudge, §results-robustness) rather than substituting one for another.

---

# A. Bonferroni-Corrected Survivor Set (`app.bonferroni`)

## A.body — header prose

- [ ] **S1.** Table bonferroni reports the Bonferroni-corrected ~99.67% paired-bootstrap CI of Δ = AUROC(s_resid) − AUROC(`tokens_generated`) at k=15 (α = 0.05/15 = 0.0033) on the headline pool.
- [ ] **S2.** Bonferroni k is the number of signals tested for the survivor-matrix claim.
- [ ] **S3.** Four signals retain a strictly positive lower CI bound under correction: FCT, DRAGIN-RIND, `min_majority`, and `ent.max`.

## A.footnote — k=15 justification footnote

- [ ] **F1.** k=15 was the multiple-comparison count fixed at test time, when the audited pool included a CABV×LAAS composite that we later dropped from presentation (§methods-fct).
- [ ] **F2.** The current table reports 16 audited signals plus the length baseline.
- [ ] **F3.** Re-running at k=14 or k=16 shifts the Bonferroni LBs by less than ±0.005 AUROC and does not change the four-survivor count; we report the test as run.

## A.caption — Table `tab:bonferroni` caption

- [ ] **C1.** Bonferroni-corrected survivor set on the headline pool (Qwen+Gemma, n=800 paired, `lift_em`=+1, B=10,000).
- [ ] **C2.** The `etc` (Δ²H) signal is the Entropy-Trend Constraint trigger of `\citet{li_modeling_2026}`.

---

# B. Full Headline Survivor Matrix (`app.hero`)

## B.body — header prose

- [ ] **S1.** Table hero-full reports all 16 audited signals on the headline pool with length-residualized AUROC and Δ vs. `tokens_generated`.

## B.caption — Table `tab:hero-full` caption

- [ ] **C1.** Full cross-architecture survivor matrix on the headline pool (Qwen2.5-7B + Gemma-4-26B-A4B-it × {HotpotQA, 2WikiMHQA}, n=800 paired traces).
- [ ] **C2.** All AUROCs are length-residualized (§methods-residualization).
- [ ] **C3.** 95% CIs are paired-bootstrap (B=10,000) of Δ = AUROC(s_resid) − AUROC(`tokens_generated`).
- [ ] **C4.** **Bold**: Bonferroni-robust at k=15 (lower bound of the 99.67% CI strictly positive; see Table bonferroni).

---

# C. Correctness-Function Robustness: LLM-as-Judge (`app.llmjudge`)

## C.protocol — Protocol paragraph

- [ ] **S1.** For each of the n=800 paired traces in the headline pool, we re-labelled both arms via an OpenAI `gpt-4o-mini` semantic-equivalence judge under a strict prompt ("You are a strict semantic-equivalence judge for multi-hop QA. Reply with exactly one word: YES or NO. No explanation.").
- [ ] **S2.** The judge target is **[Eq. 5] `lift_em_judge(q) = 1[judge(a_o)=YES ∧ judge(a_c)=NO]`**.
- [ ] **S3.** We re-ran the full headline survivor matrix with paired bootstrap B=10,000.

## C.agreement — Inter-label agreement paragraph

- [ ] **S1.** EM identifies 108 helped traces and the judge identifies 113; the intersection is 87.
- [ ] **S2.** The two labels jointly produce a multi-class Cohen's κ = 0.707 on the 3-class `lift_em` ∈ {+1, 0, −1} label set, in the substantial-agreement band of `\citet{landis_measurement_1977}`.

## C.transfer — Survivor matrix transfer paragraph

- [ ] **S1.** Table llmjudge reports the headline survivor matrix under both correctness functions side-by-side.
- [ ] **S2.** Eight of the nine paired-significant survivors under EM remained significant under the judge; `etc` loses significance and `flare.max` gains it.
- [ ] **S3.** FCT strengthened under the judge from Δ = +0.190 [+0.093, +0.287] to Δ = +0.222 [+0.132, +0.312] (+17% in point estimate, +42% in 95% CI lower-bound margin).
- [ ] **S4.** All four Bonferroni-robust survivors retain nominal-95% significance under the judge.

## C.caption — Table `tab:llmjudge` caption

- [ ] **C1.** Side-by-side Δ under EM and LLM-as-Judge on the headline pool (n=800, B=10,000).
- [ ] **C2.** *95% CI straddles zero.
- [ ] **C3.** FCT *strengthens* under the judge; eight of nine paired-significant EM survivors are preserved; all four Bonferroni-robust survivors retain nominal-95% significance.

## C.caveat — Caveat paragraph

- [ ] **S1.** `\citet{zheng_judging_2024}` document that GPT-4-tier judges exhibit a verbosity bias, preferring longer responses; our cross-judge check demonstrates *pattern stability* rather than full verbosity-independence.
- [ ] **S2.** The load-bearing length-orthogonality argument rests on the Uncertainty-Line residualization (§methods-residualization) and the paired-oracle design (§methods-protocol); the judge check confirms the survivor matrix is not an artifact of any single correctness function.

---

# D. Uncertainty-Line Reference Replication (`app.uline`)

## D.body — header prose

- [ ] **S1.** To pre-empt the reimplementation critique, we ran our protocol (Eq. resid) and the `\citet{vashurin_uncertainty-line_2025}` reference implementation side-by-side on the same trace cohort, decomposing the comparison into two ε components.
- [ ] **S2.** ε_protocol isolates the math difference (our log(tokens+1) single-cohort OLS against the reference MinMax-scaled linear regression on the same cohort); ε_sampling isolates train/test-split sampling noise at n=200.

## D.caption — Table `tab:uline-repl` caption

- [ ] **C1.** ε_protocol = |ours_resid − LINE_full|; ε_sampling = |LINE_full − LINE_split| (latter averaged over 3 seeds).
- [ ] **C2.** Both architectures show ε_protocol ≤ 0.029, well below the conventional threshold of 0.05.
- [ ] **C3.** Train/test sampling noise (ε_sampling) dominates the protocol gap by a factor of 3–4×, indicating that the log(tokens+1) vs. raw-MinMax math difference is small relative to evaluation-set variance at n=200.

---

# E. FCT vs. Concurrent Commitment-Time Primitives: Operational Differences (`app.fct-tstar`)

## E.body — header prose

- [ ] **S1.** FCT (§methods-fct) is motivated by the final-switch index t* = max{i : A_i ≠ A_{i+1}} of `\citet{datta_large_2026}`, with a concurrent forward-looking commitment-time primitive proposed by `\citet{zhang_commit_2026}` on the model's finite-answer continuation distribution.
- [ ] **S2.** The three are operationally distinct on five axes.

## E.caption — Table `tab:fct-vs-tstar` caption

- [ ] **C1.** Operational differences across three concurrent commitment-time primitives.
- [ ] **C2.** `\citet{datta_large_2026}`'s t* is backward-looking on per-step intermediate answers, applied to early stopping.
- [ ] **C3.** `\citet{zhang_commit_2026}`'s τ_commit is forward-looking on the model's finite-answer continuation log-odds, applied to pre-verbalization commitment analysis.
- [ ] **C4.** FCT (ours) is forward-looking on K=5-majority answer extracts, applied to retrieval triggering.
- [ ] **C5.** For monotonically converging trajectories, t_commit ≈ t* + 1; under K=5 majority sampling on non-thinking-mode chat models this equivalence does not hold in general.

---

# F. Length-vs-Signal Spearman ρ per Cell (`app.spearman`)

## F.body — header prose

- [ ] **S1.** Table spearman reports Spearman ρ between `tokens_generated` and each signal per (arch × dataset) cell on closed-book traces, the length-confound diagnostic of `\citet{santilli_revisiting_2025}` (their Table 2).
- [ ] **S2.** |ρ| ≥ 0.5 indicates a strong length confound on the signal.

## F.caption — Table `tab:spearman` caption

- [ ] **C1.** Spearman ρ(`tokens_generated`, signal) per cell on closed-book traces.
- [ ] **C2.** Bold: |ρ| ≥ 0.5.
- [ ] **C3.** The peer-pool signals (DRAGIN, ent.max, etc, flare.max, u_cons.max, seakr_k5) carry ρ ∈ [+0.26, +0.46]; `hvr.max` reaches +0.51 on Gemma×2Wiki; the auxiliary CRV diagnostics span ρ ∈ [−0.07, +0.36].
- [ ] **C4.** The answer-trajectory aggregations (bottom block) average ≈ 0.1 lower in |ρ| than the peer-pool block.
- [ ] **C5.** *Qwen×Hot `hvr.max` is constant; Spearman undefined.

---

# G. Adapter Notes (`app.adapters`)

## G.opening — opening prose

- [ ] **S1.** We audit signal-level discriminative power on `lift_em` (§limitations).
- [ ] **S2.** Several primitives in the published trigger pool are originally framed as online per-token triggers or per-trace scores; we evaluate their underlying statistic at the trace level.
- [ ] **S3.** Adapter scope per signal:

## G.etc — ETC paragraph

- [ ] **S1.** The original ETC operates on a token-level entropy stream; we evaluate Δ²H on the per-step mean entropy series captured at boundary-detection events (§methods-fct).
- [ ] **S2.** Step-level smoothing is the only adapter; the second-difference operator is byte-faithful.

## G.flare — FLARE paragraph

- [ ] **S1.** The original FLARE is an online per-token retrieval trigger (max_t p_t < τ); we audit the underlying primitive (low-confidence detection) as a trace-level summary max over steps of 1 − min p_t.
- [ ] **S2.** The trigger *policy* is not audited; the trigger *statistic* is.

## G.seakr — SeaKR paragraph

- [ ] **S1.** K=5 EOS hidden states (vs. K=20 in the paper, with the paper's best-k interval [10, 25]); the regularized Gram determinant we compute is the same primitive as the paper's self-aware uncertainty score.
- [ ] **S2.** INSIDE `\citep{chen_inside_2024}` is the closest peer, using the EigenScore on the centered sample covariance; the K nonzero eigenvalues of our uncentered Gram and INSIDE's centered covariance are rank-equivalent up to centering.

## G.selfdoubt — SELFDOUBT-T1 paragraph

- [ ] **S1.** HVR = h/(v+1) is computed per step and aggregated as max over steps; the original paper computes a single trace-level HVR with a Tier-1 "HVR=0" high-precision correctness gate.
- [ ] **S2.** The per-step adapter is recoverable as the paper's gate by setting our threshold to 0.

## G.commit — Concurrent commitment-time primitives paragraph

- [ ] **S1.** Operational differences between `\citet{datta_large_2026}`'s t*, `\citet{zhang_commit_2026}`'s τ_commit, and our FCT are tabulated in Appendix fct-tstar.

---

# H. Reproducibility (`app.repro`)

- [ ] **S1.** All trace JSONLs are committed at `runs/v2_headline_{arch}/` for arch ∈ {qwen2.5-7b-instruct, gemma-4-26b-a4b-it}.
- [ ] **S2.** Slice manifests with SHA-256 checksums are at `runs/v2_headline_{arch}/checkpoints/slice_manifest.json`.
- [ ] **S3.** Decoding command lines are at `runs/v2_headline_{arch}/logs/00_master.log`.
- [ ] **S4.** Analysis scripts are at `analysis/r1_paired_audit.py`, `analysis/a1_k1_vs_k5_ablation.py`, `analysis/uncertainty_line_audit.py`, `analysis/moonshot_explore.py`, and `analysis/llm_judge_audit.py`.
- [ ] **S5.** The judge cache (800 per-call JSONs) is at `analysis/llm_judge_cache/`.
- [ ] **S6.** Compute box: single NVIDIA A800 80 GB, wall-clock ~15 h.
