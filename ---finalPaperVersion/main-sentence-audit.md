# Sentence-by-Sentence Audit: `main.tex`

> Source: `---finalPaperVersion/main.tex`
> Convention: one bullet = one sentence (as printed). Display equations, table contents, and quoted prompt text are noted but not split. Citations are kept inline as `\citet{...}` / `\citep{...}` so we can audit them in place. Status markers (`[ ]`) are for our review pass.

---

## Title

- [ ] Length Is the Floor: A Length-Residualized Audit of Training-Free Dynamic-RAG Triggers

---

## Abstract

- [ ] Dynamic retrieval-augmented generation systems decide *when* to retrieve from a scalar trigger signal, scored by raw AUROC against binary correctness.
- [ ] Recent work has shown this protocol is length-confounded on adjacent question-answering (QA) selective-prediction and uncertainty-quantification (UQ) tasks.
- [ ] We present the first length-residualized audit of the training-free dynamic-RAG trigger pool on a paired-oracle target (`lift_em`=+1) that asks whether retrieval flips a wrong answer to a right one, isolating the causal retrieval-helpfulness signal from length-correlated incorrectness.
- [ ] Across two open-weight chat models on HotpotQA and 2WikiMultiHopQA (n=800 paired traces), nine signals beat the length baseline at 95% paired-bootstrap and only four survive Bonferroni at k=15.
- [ ] On the held-out 2Wiki cells, raw length collapses to at-or-below chance (AUROC ≤ 0.52), and our **FCT** (first-commitment time, a forward-looking answer-trajectory signal) reaches raw AUROC **0.784** on Gemma × 2Wiki, the strongest length-residualized signal in the pool.
- [ ] The survivor set is robust to a GPT-4o-Mini semantic-equivalence judge (κ = 0.707), and FCT itself *strengthens* from Δ = +0.190 to Δ = +0.222 under the judge, the opposite of what a string-level (exact-match) EM artifact would predict.

---

## Figure 1 caption (`fig:survivor-forest`)

- [ ] **Four trigger signals from three families carry length-orthogonal information beyond the length baseline; FCT (this paper) is strongest.**
- [ ] Forest plot of Δ = AUROC(s_resid) − AUROC(`tokens_generated`) on `lift_em`=+1 (n=800 paired traces).
- [ ] Thick bars: 95% paired-bootstrap CIs; thin extensions: Bonferroni-corrected 99.67% CIs at k=15.
- [ ] Setup: §Methods. Tables: App. Bonferroni, Hero.

---

## 1. Introduction

- [ ] Dynamic retrieval-augmented generation (RAG) systems retrieve external evidence during decoding to ground generation in source material \citep{lewis_retrieval-augmented_2021, trivedi_interleaving_2023, shao_enhancing_2023}.
- [ ] At the core of every such system sits a *retrieval trigger*: a scalar function of the partial generation that decides when to fire retrieval, evaluated by ranking traces by the trigger score and reporting raw AUROC against a binary correctness label.
- [ ] We restrict our audit to *training-free* triggers.
- [ ] Prior work spans six signal families: attention-based DRAGIN \citep{su_dragin_2024}, token-probability FLARE \citep{jiang_active_2023}, entropy-trend ETC \citep{li_modeling_2026}, hidden-state SeaKR \citep{yao_seakr_2025} and INSIDE \citep{chen_inside_2024}, multi-sample text-consistency (U_cons component of RSUS, \citealp{guo_when_2026}), and hedging-marker SELFDOUBT-T1 \citep{pandey_selfdoubt_2026}.
- [ ] A contemporaneous answer-trajectory family tracks step-level dynamics of the model's own predicted answer \citep{datta_large_2026, wang_chain--probe_2025, chen_i_2026}.

- [ ] Recent work has shown this evaluation convention is length-confounded on adjacent QA selective-prediction and uncertainty-quantification (UQ) tasks.
- [ ] \citet{santilli_revisiting_2025} prove that mutual length-bias on UQ scores and correctness functions systematically distorts AUROC.
- [ ] \citet{janiak_illusion_2025} show the empirical consequence: detection AUROC drops by up to 45.9% when switching from ROUGE to an LLM-as-Judge label.
- [ ] \citet{zhang_verbosity_2024} attribute the underlying behavior to verbosity compensation on multi-hop QA, and \citet{ielanskyi_addressing_2025} document a related correctness-function failure across approximate metrics.
- [ ] The methodological response is \citet{vashurin_uncertainty-line_2025}'s **Uncertainty-Line**, which regresses each UQ score on output length and uses the residual as a length-invariant estimate.
- [ ] The dynamic-RAG trigger pool is the missing application.

- [ ] We extend the length-residualization protocol of \citet{vashurin_uncertainty-line_2025} to the training-free dynamic-RAG trigger pool.
- [ ] Rather than score triggers against single-arm correctness, we re-decode each closed-book trace under gold context and ask whether retrieval flips a wrong answer to a right one, defining a paired-oracle target `lift_em`=+1 conceptually motivated by \citet{mallen_when_2023}.
- [ ] This isolates the *causal* retrieval-helpfulness signal from the length-correlated incorrectness signal that the published pool actually measures (§Discussion-Causal).
- [ ] We also propose **FCT** (first-commitment time), a forward-looking variant of the answer-trajectory primitive \citep{datta_large_2026, wang_chain--probe_2025, chen_i_2026, zhang_when_2026}.
- [ ] FCT reads K=5 majority answers from non-thinking-mode chat models, where neither Datta's `</think>`-anchored forced completion nor \citeauthor{zhang_when_2026}'s finite-set continuation projection applies directly (App. FCT-tstar).

### 1.1 Contributions paragraph

- [ ] (1) A length-residualized audit of the dynamic-RAG trigger pool, the first to our knowledge, scored against a paired-oracle target (`lift_em`=+1).
- [ ] (2) **FCT** (first-commitment time), our forward-looking variant of the answer-trajectory primitive and the strongest length-residualized trigger in the audited pool.
- [ ] (3) A correctness-function robustness check against \citet{janiak_illusion_2025} and \citet{ielanskyi_addressing_2025}, under which the survivor matrix is preserved and FCT *strengthens*.

---

## 2. Related Work

- [ ] The length-confound critique of standard UQ AUROC evaluation comes from QA UQ and hallucination detection \citep{santilli_revisiting_2025, janiak_illusion_2025, zhang_verbosity_2024, ielanskyi_addressing_2025, vashurin_uncertainty-line_2025, mao_confidence_2026}.
- [ ] The answer-trajectory family \citep{datta_large_2026, wang_chain--probe_2025, chen_i_2026} provides the parent primitives for FCT.
- [ ] Two related supervised methods sit outside our training-free scope: \citet{lu_streaming_2026} train hidden-state probes for streaming hallucination detection, and \citet{zhang_harnessing_2026} contrastively shape answer embeddings under latent perturbation at the trace boundary, where their "answer agreement" differs from our K=5 boundary-completion "answer-trajectory".
- [ ] Trained adaptive-retrieval policies likewise fall outside our audit \citep{asai_self-rag_2023, guo_dior_2025}.
- [ ] Our paired-oracle target operationalizes the question of when retrieval helps a parametric model \citep{mallen_when_2023}, replacing static entity-popularity proxies with per-trace closed-vs-oracle re-decode.

---

## 3. Methods

### 3.1 Paired-Oracle `lift_em` Protocol

- [ ] The published trigger pool scores against single-arm correctness, a target length predicts moderately well on multi-hop QA (§Discussion-Causal).
- [ ] We instead define a paired target that flags traces where retrieval would convert a wrong answer to right, isolating the *causal* retrieval-helpfulness signal from length-correlated incorrectness.
- [ ] Concretely, for each question q, we run two greedy decodes on the same chat model: a closed-book pass that produces trace τ_c^(q) and answer a_c^(q), and an oracle pass conditioned on the gold supporting paragraphs that produces a_o^(q) (decoding prompts and answer-extraction details: App. Prompts).
- [ ] We read all trigger signals from τ_c^(q); the oracle decode only supplies the label.
- [ ] The discrimination target is `lift_em(q) = 1[EM(a_o) > EM(a_c)]` (Eq. lift-em).

- [ ] We evaluate on HotpotQA \citep{yang_hotpotqa_2018} and 2WikiMultiHopQA \citep{ho_constructing_2020}, holding the latter out as a cross-dataset cell, at n=200 paired traces per architecture-by-dataset cell.
- [ ] The two models are Qwen2.5-7B-Instruct (dense) and Gemma-4-26B-A4B-it (MoE-hybrid), chosen for cross-architecture coverage.
- [ ] The positive class is rare: 108 of 800 paired traces, base rate 13.5%.

### 3.2 Length Residualization

- [ ] We adopt the residualization protocol of \citet{vashurin_uncertainty-line_2025}.
- [ ] For each trigger signal s on a cohort of paired traces, we fit OLS of s on log(tokens+1) and evaluate AUROC on the residual: `s_resid = s − α̂ − β̂·log(tokens+1)` (Eq. residualize).
- [ ] Intuitively, s_resid keeps only the part of s that length cannot linearly explain; its AUROC therefore measures whether the signal carries information beyond length.
- [ ] Our log(tokens+1) parameterization agrees within ε ≤ 0.029 AUROC with the public reference implementation on our cohorts (App. uline).
- [ ] Several signals in the audited pool (ETC, FLARE, SeaKR, SELFDOUBT-T1) are originally online or per-token, and we evaluate each as a trace-level summary statistic with granularity adapters in Appendix Adapters.

### 3.3 FCT and the Answer-Trajectory Family

- [ ] At each reasoning-step boundary (detected by tokenizer-level sentence-end token matchings), we sample K=5 boundary completions at temperature τ=0.6, top-p 0.9, and extract a per-step answer from each via a regex over the model's `Answer:` prefix (sampler config and extractor regex: App. Prompts).
- [ ] From the resulting per-step K=5 majority sequence M = (m_1, …, m_n) we compute four trace-level aggregations:
  - [ ] `flip_rate`: fraction of consecutive-step majority disagreements;
  - [ ] `late_flips`: count of K=5 answer-*set* disagreements in the last half of the trace;
  - [ ] `min_majority`: 1 − min_t(max_count/K), the worst within-step disagreement;
  - [ ] **FCT**, a forward-looking first-commitment time (Eq. fct), with k ∈ {0, …, n−1}.
- [ ] Intuitively, FCT is the (normalized) decode position at which the model first committed to its final answer.
- [ ] For example, on a 5-step trace with per-step majorities (m_1, …, m_5) = (X, X, Y, Y, Y) and final answer Y, the smallest cutoff k such that m_j=Y for all j>k is t_commit=2, so FCT = 2/4 = 0.5 (a worked computation on a real `lift_em`=+1 trace from the headline pool is in App. Prompts).
- [ ] Low FCT denotes early commitment; high FCT denotes a trace that kept flipping until close to the end of the decode, the traces where retrieval is most likely to help.
- [ ] All four aggregations adapt the `AnswerSwitches` primitive of \citet{datta_large_2026} to non-thinking-mode chat models with K=5 majority-answer extracts.
- [ ] Neither Datta's `</think>`-anchored forced completion nor \citeauthor{zhang_when_2026}'s finite-set continuation projection applies directly in this regime (App. FCT-tstar).

#### Continuous variant paragraph

- [ ] We additionally evaluate **CABV** (continuous answer-belief velocity), a between-step generalization of `flip_rate` defined in App. CABV and reported in Tables Bonferroni and Hero-full.

---

## 4. Results

### 4.1 Cross-Architecture Survivor Matrix

- [ ] Of the 16 audited signals, nine clear the length baseline at nominal 95% paired-bootstrap, and only four survive Bonferroni correction at k=15 (Figure 1): attention-based DRAGIN-RIND (Δ = +0.167), token-probability `ent.max` (+0.137), answer-trajectory `min_majority` (+0.149), and **FCT** (+0.190, the strongest length-residualized trigger in the pool).
- [ ] The four survivors span three of the seven declared families, which suggests the length-orthogonal signal is not a quirk of any one mechanism.
- [ ] Two families have no nominal-95% survivor: hidden-state (SeaKR, INSIDE, plus three geometry variants) and hedging-marker (SELFDOUBT-T1) are absorbed by length (App. Spearman).
- [ ] The full table is in App. Hero.

### 4.2 Length Is Anti-Predictive on the Held-Out 2Wiki Cells

- [ ] The Bonferroni-robust four answer one question: which signals beat length?
- [ ] A sharper question is when length itself stops being informative.
- [ ] On the two held-out 2Wiki cells, raw answer length is at or below chance for predicting retrieval need: `tokens_generated` reaches AUROC **0.457** on Qwen × 2Wiki and **0.518** on Gemma × 2Wiki.
- [ ] The verbosity-compensation pattern \citet{zhang_verbosity_2024} document on the same LongBench bundle does not transfer to `lift_em`: longer multi-hop traces are no longer reliably retrieval-helpful.
- [ ] The four Bonferroni-robust survivors clear the length baseline by a wider margin on these cells than on HotpotQA, meaning the raw-AUROC protocol miscounts the length signal on exactly the cells where retrieval matters most.

### 4.3 FCT Discriminates Strongest on the Held-Out MoE Cell

- [ ] The cells where length fails are the cells where FCT succeeds.
- [ ] On Gemma × 2Wiki, the held-out cell where length is closest to chance (`tokens_generated` 0.518), FCT reaches raw AUROC **0.784**, a +0.266 absolute gap over length (Table fct-percell).
- [ ] The same pattern repeats on Qwen × 2Wiki: where the length proxy is below chance (0.457), FCT's margin (+0.147) is wider than on either HotpotQA cell.
- [ ] The answer-trajectory primitive carries the most independent information where the length proxy collapses.

#### Table `tab:fct-percell` caption

- [ ] Per-cell raw AUROC of FCT on `lift_em`=+1.

### 4.4 Survivor Set Is Robust to a Semantic-Equivalence Judge

- [ ] The natural alternative explanation for the survivor matrix is that it reflects EM's string-matching artifact rather than a length-orthogonal signal.
- [ ] To rule this out we relabelled both arms of every paired trace via a GPT-4o-Mini judge (\citeauthor{janiak_illusion_2025}'s exact choice) and re-ran the audit, addressing the correctness-function critique of \citet{janiak_illusion_2025} and \citet{ielanskyi_addressing_2025}.
- [ ] EM and the judge agree at Cohen's κ = 0.707 (substantial agreement; \citealp{landis_measurement_1977}).
- [ ] Eight of the nine paired-significant survivors under EM remain significant under the judge, with a single swap (`etc` loses, `flare.max` gains) that leaves the Bonferroni-robust four intact.
- [ ] FCT itself *strengthens* under the judge from Δ = +0.190 to Δ = +0.222, the opposite of what a string-level EM artifact would predict under semantic relabeling (App. llmjudge).

### 4.5 The K=5 Multi-Sample Variant Is Null

- [ ] The paired-bootstrap CI on Δ_{K=5 − K=1} for `flip_rate` straddles zero on both architectures (Qwen −0.002, [−0.040, +0.031]; Gemma +0.033, [−0.025, +0.092]; App. k-ablation).
- [ ] The audit reports K=5 throughout because that was the capture protocol when the ablation returned null; for deployment K=1 delivers the same triggering performance at 5× less compute per boundary.
- [ ] K=1 here is a single stochastic sample at the same (τ=0.6, top-p=0.9); a deterministic-greedy K=1 ablation is left to future work.

---

## 5. Discussion

### 5.1 Why Length Predicts Retrieval-Helpfulness

- [ ] On the secondary `closed_em`=0 target, `tokens_generated` reaches raw AUROC 0.539 on the headline pool, competitive with several internal signals.
- [ ] This pattern matches \citet{zhang_verbosity_2024}'s verbosity-compensation mechanism.
- [ ] A plausible causal chain is length → verbosity compensation → uncertainty → incorrectness → retrieval need, and length predicts `closed_em`=0 moderately well because it proxies the upstream incorrectness step.
- [ ] The paired-oracle target `lift_em`=+1 strips that step by conditioning on closed-book wrongness, which is why length collapses on the held-out 2Wiki cells (§Results-Inverted).

### 5.2 Why Answer-Trajectory Stability Predicts Retrieval Need

- [ ] The mechanism behind FCT reads naturally as *premature commitment*, the interpretation \citet{datta_large_2026} give to their `AnswerSwitches` and t* primitives.
- [ ] The model decides on an answer early and continues to verbalize reasoning that supports the commitment, with most subsequent steps as post-decision tail.
- [ ] \citeauthor{datta_large_2026} apply this observation to early stopping: if the model has decided, no further compute is needed.
- [ ] Retrieval triggering reads the same primitive in the opposite causal direction.
- [ ] The traces where the model has not decided, where the answer trajectory continues to flip through the reasoning, are exactly the traces where external evidence would resolve the uncertainty.
- [ ] The paired-oracle target `lift_em`=+1 formalizes that question directly.
- [ ] The same primitive supports two complementary operations: detect stabilization to *stop*, detect non-stabilization to *retrieve*.

---

## 6. Conclusion

- [ ] We presented the first length-residualized audit of the dynamic-RAG trigger pool, extending the protocol of \citet{vashurin_uncertainty-line_2025} to a paired-oracle target (`lift_em`=+1).
- [ ] On two open-weight chat models over HotpotQA and 2WikiMultiHopQA, nine signals beat the length baseline at 95% paired-bootstrap and four survived Bonferroni at k=15 across three families.
- [ ] The strongest length-residualized trigger is **FCT**, our forward-looking variant of the answer-trajectory primitive, exceeding length by Δ = +0.190 AUROC and reaching raw 0.784 on the held-out Gemma × 2Wiki cell.
- [ ] The survivor set is robust to a GPT-4o-Mini semantic-equivalence judge (κ = 0.707), with FCT itself strengthening under the judge.
- [ ] The same answer-trajectory primitive supports two complementary deployment operations: detect stabilization to *stop*, detect non-stabilization to *retrieve*.

---

## 7. Limitations

### Signal-level audit, not end-to-end pipelines

- [ ] We audit trigger *signals* as defined in their original papers, not full deployment pipelines (threshold tuning, retriever choice, prompt format).
- [ ] Our claims pertain to signal-level discriminative power on `lift_em`.

### Bonferroni knife-edge

- [ ] Two of the four Bonferroni-robust survivors (`ent.max`, `min_majority`) carry ~99.67% lower bounds within 5×10⁻³ of zero (App. Bonferroni).
- [ ] The four-survivor count is therefore fragile to small perturbations.
- [ ] The FCT and DRAGIN-RIND lower bounds are robust at +0.047 and +0.039.

### Architecture and dataset coverage

- [ ] The headline pool covers two architectures and two multi-hop QA benchmarks.
- [ ] The cheapest extensions are a third open-weight chat model (e.g., Llama-3.1-8B-Instruct) and a third multi-hop benchmark (MuSiQue, \citealp{trivedi__2022}).
- [ ] The current FCT Bonferroni lower bound (+0.047) is the statistical floor additional cells would have to displace.

### Dataset framing: held-out, not OOD

- [ ] We treat 2WikiMultiHopQA as a *held-out* cell relative to HotpotQA, not as out-of-distribution in the strict sense: both are multi-hop QA over Wikipedia with comparable question complexity.
- [ ] The cell-dependent length behavior (§Results-Inverted) is a dataset-specific finding on multi-hop QA rather than a claim about OOD generalization, and a stricter cross-distribution evaluation is future work.

### Correctness function

- [ ] We use exact-match equality on both arms of the paired-oracle target.
- [ ] The paired design (§Methods-Protocol) and the LLM-as-Judge robustness check (App. llmjudge) partially insulate the headline against this choice.
- [ ] Containment-match and F1 variants remain future work.

### Non-thinking-mode chat models

- [ ] The audited models are non-thinking-mode chat variants.
- [ ] \citet{datta_large_2026} developed their answer-trajectory primitive on thinking-mode reasoning models with explicit `</think>` tokens.
- [ ] Our K=5 boundary-completion adaptation is not byte-faithful to that protocol.

### Risks

- [ ] Our work concerns evaluation methodology for retrieval triggering on open-weight chat models and public multi-hop QA benchmarks.
- [ ] We release no new datasets or trained models, and direct risks are limited.
- [ ] One methodological risk is that the length-residualized protocol could be cited in isolation to defend deployed dynamic-RAG systems whose triggers track length more strongly than residualization alone would reveal.
- [ ] We therefore recommend reporting raw, length-residualized, and LLM-judge-validated AUROC together (§Results-LLMJudge) rather than substituting one for another.

---

# Appendices

## A. Bonferroni-Corrected Survivor Set

- [ ] Table tab:bonferroni reports the Bonferroni-corrected ~99.67% paired-bootstrap CI of Δ = AUROC(s_resid) − AUROC(`tokens_generated`) at k=15 (α = 0.05/15 = 0.0033) on the headline pool.
- [ ] Bonferroni k is the number of signals tested for the survivor-matrix claim.
- [ ] Four signals retain a strictly positive lower CI bound under correction: FCT, DRAGIN-RIND, `min_majority`, and `ent.max`.
  - [ ] Footnote: k=15 was the multiple-comparison count fixed at test time, when the audited pool included a CABV×LAAS composite that we later dropped from presentation (§Methods-FCT).
  - [ ] Footnote (cont.): The current table reports 16 audited signals plus the length baseline.
  - [ ] Footnote (cont.): The Bonferroni-corrected CI half-widths in the table (≈ 0.12–0.14 AUROC at B=10,000) dominate any plausible k-sensitivity at this signal pool size.
  - [ ] Footnote (cont.): We report the test as run.

### Table `tab:bonferroni` caption

- [ ] Bonferroni-corrected survivor set on the headline pool (Qwen+Gemma, n=800 paired, `lift_em`=+1, B=10,000).
- [ ] The `etc` (Δ²H) signal is the Entropy-Trend Constraint trigger of \citet{li_modeling_2026}.

---

## B. Full Headline Survivor Matrix

- [ ] Table tab:hero-full reports all 16 audited signals on the headline pool with length-residualized AUROC and Δ vs. `tokens_generated`.

### Table `tab:hero-full` caption

- [ ] Full cross-architecture survivor matrix on the headline pool (Qwen2.5-7B + Gemma-4-26B-A4B-it × {HotpotQA, 2WikiMHQA}, n=800 paired traces).
- [ ] All AUROCs are length-residualized (§Methods-Residualization).
- [ ] 95% CIs are paired-bootstrap (B=10,000) of Δ = AUROC(s_resid) − AUROC(`tokens_generated`).
- [ ] **Bold**: Bonferroni-robust at k=15 (lower bound of the 99.67% CI strictly positive; see Table bonferroni).

---

## C. Correctness-Function Robustness: LLM-as-Judge

### Protocol

- [ ] For each of the n=800 paired traces in the headline pool, we re-labelled both arms via an OpenAI `gpt-4o-mini` semantic-equivalence judge under a strict three-sentence system prompt: "You are a strict semantic-equivalence judge for multi-hop QA. Your task is to decide whether a model-predicted answer is semantically equivalent to a gold answer. Reply with exactly one word: YES or NO. No explanation."
- [ ] The judge target is `lift_em_judge(q) = 1[judge(a_o)=YES ∧ judge(a_c)=NO]` (Eq. lift-em-judge).
- [ ] We re-ran the full headline survivor matrix with paired bootstrap B=10,000.

### Inter-label agreement

- [ ] EM identifies 108 helped traces and the judge identifies 113; the intersection is 87.
- [ ] The two labels jointly produce a multi-class Cohen's κ = 0.707 on the 3-class `lift_em` ∈ {+1, 0, −1} label set, in the substantial-agreement band of \citet{landis_measurement_1977}.

### Survivor matrix transfer

- [ ] Table tab:llmjudge reports the headline survivor matrix under both correctness functions side-by-side.
- [ ] Eight of the nine paired-significant survivors under EM remain significant under the judge, with a single swap: `etc` loses significance and `flare.max` gains it.
- [ ] FCT strengthens under the judge from Δ = +0.190 [+0.093, +0.287] to Δ = +0.222 [+0.132, +0.312] (+17% in point estimate, +42% in 95% CI lower-bound margin), the opposite of what a string-level EM artifact would predict.
- [ ] All four Bonferroni-robust survivors retain nominal-95% significance under the judge.

### Table `tab:llmjudge` caption

- [ ] Side-by-side Δ under EM and LLM-as-Judge on the headline pool (n=800, B=10,000).
- [ ] *: 95% CI straddles zero.
- [ ] FCT *strengthens* under the judge.
- [ ] Eight of nine paired-significant EM survivors are preserved, and all four Bonferroni-robust survivors retain nominal-95% significance.

### Caveat

- [ ] \citet{zheng_judging_2023} document that GPT-4-tier judges exhibit a verbosity bias, preferring longer responses; our cross-judge check demonstrates *pattern stability* rather than full verbosity-independence.
- [ ] The load-bearing length-orthogonality argument rests on the **Uncertainty-Line** residualization (§Methods-Residualization) and the paired-oracle design (§Methods-Protocol); the judge check confirms the survivor matrix is not an artifact of any single correctness function.

---

## D. Decoding Prompts, Answer Extraction, and a Worked Example

### Closed-book and oracle decoding

- [ ] Both arms of the paired-oracle target (§Methods-Protocol) use a single chat-template message pair with a system role and a user role; the oracle arm differs only by paragraph filtering (gold supporting paragraphs only, distractors removed).
- [ ] Decoding is greedy (argmax over post-repetition-penalty logits with `repetition_penalty`=1.0 on both architectures), `max_new_tokens`=1024, with early stop on EOS or the first `Answer: <…>\n` line (checked every 16 generated tokens to amortize the decode cost).
- [ ] The shared system instruction is [quoted prompt block; not split].
- [ ] [User payload rendering shown as a quoted template block; not split].
- [ ] Both messages are rendered through each model's native chat template via Hugging Face `apply_chat_template` with default kwargs (no thinking-mode toggle on either model; Qwen2.5 has none, Gemma-4-26B-A4B-it leaves the thought channel empty under defaults).
- [ ] On HotpotQA the closed-book arm sees all 10 distractor paragraphs; the oracle arm sees only paragraphs whose title appears in the gold `supporting_facts` (typically 2).
- [ ] Per-step boundary detection fires on tokenizer-level sentence-end tokens (period, newline, etc.) with a 32-token cap.

### K=5 boundary completions

- [ ] At each detected boundary, the model state is forked from the cached `past_key_values` and 5 continuations are sampled at τ=0.6, top-p=0.9, `max_new_tokens`=40, multinomial over the nucleus, with the cache truncated back to the boundary on exit.
- [ ] Random seed 20260511 is offset per step.
- [ ] The same K=5 samples feed both the answer-trajectory family (§Methods-FCT) and the SeaKR-K=5 hidden proxy (App. Adapters).

### Trace-level answer extraction

- [ ] The final-answer extractor scans for the *last* occurrence of one of {`final answer:`, `final_answer:`, `answer:`, `## answer`, `**answer**`, `**final answer**`} (case-insensitive) whose payload does not begin with a placeholder echo (`<` or `[`), takes the first non-empty line, truncates at the first sentence-end punctuation followed by whitespace, splits at the first lower→upper seam past character 1 (defensive against decode-loop concatenation artefacts), and hard-caps at 100 characters.
- [ ] EM uses HotpotQA-standard SQuAD normalization: lowercase, strip articles ({a, an, the}), drop punctuation, collapse whitespace.

### Per-step answer extraction (FCT input)

- [ ] The per-step answer regex is intentionally tighter and faster, since it runs 5n times per trace: [regex block, not split].
- [ ] Per-step normalization keeps only `[a-z0-9 ]`, lowercases, and truncates at 30 characters.
- [ ] Per-step majority is `Counter(answers).most_common(1)[0][0]`; FCT (Eq. fct) is then the smallest index t_commit such that every populated step at or after t_commit has majority equal to the trace-level final answer, normalized by max(1, n_populated−1).

### Worked example

- [ ] Qwen2.5-7B-Instruct × HotpotQA, question `5abd01205542992ac4f3819b` (a real `lift_em`=+1 trace from the headline pool):
  - [ ] Q: "Mummulgum is a stop on the Bruxner Highway before reaching the town on the banks of which river?"
  - [ ] Gold: *Richmond*
  - [ ] Closed-book final answer: *Casino* (EM=0)
  - [ ] Oracle final answer: *Richmond* (EM=1)
  - [ ] ⇒ `lift_em`=+1.
- [ ] The closed-book trace has n=3 step boundaries.
- [ ] Per-step K=5 majority answers extracted from the boundary completions: [table block, not split].
- [ ] All three step-majorities equal the trace-level final answer (`casino`), so t_commit=0 and **FCT** = 0/(3−1) = 0.000.
- [ ] This trace is a per-trace *false negative* for FCT: the model committed to a wrong answer early, a mode FCT cannot distinguish from confident-correct early commitment.
- [ ] Aggregate FCT raw AUROC on `lift_em`=+1 is 0.784 on Gemma × 2Wiki (Table fct-percell); the Bonferroni 99.67% lower bound is +0.047 (Table bonferroni).
- [ ] FCT is the strongest length-residualized discriminator in the audited pool, not a per-trace oracle.
- [ ] The full trace JSON (closed and oracle arms, all 5 boundary samples per step, per-token logprobs, per-step entropies, hidden-state captures, gate decisions) will be released with the code (App. Repro).

---

## E. Uncertainty-Line Reference Replication

- [ ] To pre-empt the reimplementation critique, we ran our protocol (Eq. residualize) and the \citet{vashurin_uncertainty-line_2025} reference implementation side-by-side on the same trace cohort, decomposing the comparison into two ε components.
- [ ] ε_protocol isolates the math difference (our log(tokens+1) single-cohort OLS against the reference MinMax-scaled linear regression on the same cohort); ε_sampling isolates train/test-split sampling noise at n=200 (Table uline-repl).

### Table `tab:uline-repl` caption

- [ ] ε_protocol = |ours_resid − LINE_full|; ε_sampling = |LINE_full − LINE_split| (latter averaged over 3 seeds).
- [ ] Both architectures show ε_protocol ≤ 0.029, well below the conventional threshold of 0.05.
- [ ] Train/test sampling noise (ε_sampling) dominates the protocol gap by a factor of 3–4×, indicating that the log(tokens+1) vs. raw-MinMax math difference is small relative to evaluation-set variance at n=200.

---

## F. FCT vs. Concurrent Commitment-Time Primitives: Operational Differences

- [ ] FCT (§Methods-FCT) is motivated by the final-switch index t* = max{i : A_i ≠ A_{i+1}} of \citet{datta_large_2026}, with a concurrent forward-looking commitment-time primitive proposed by \citet{zhang_when_2026} on the model's finite-answer continuation distribution.
- [ ] The three are operationally distinct on five axes (Table fct-vs-tstar).

### Table `tab:fct-vs-tstar` caption

- [ ] Operational differences across three concurrent commitment-time primitives.
- [ ] \citet{datta_large_2026}'s t* is backward-looking on per-step intermediate answers, applied to early stopping.
- [ ] \citet{zhang_when_2026}'s τ_commit is forward-looking on the model's finite-answer continuation distribution (log-odds in binary tasks), applied to pre-verbalization commitment analysis.
- [ ] FCT (ours) is forward-looking on K=5-majority answer extracts, applied to retrieval triggering.
- [ ] For monotonically converging trajectories, t_commit ≈ t* + 1; under K=5 majority sampling on non-thinking-mode chat models this equivalence does not hold in general.

---

## G. CABV: Continuous Answer-Belief Velocity

- [ ] CABV is a continuous between-step generalization of `flip_rate`: at each populated step we build a Laplace-smoothed empirical distribution over the trace's K=5 answers and average the Jensen–Shannon divergence between consecutive steps.
- [ ] CABV reduces to `flip_rate` when each step concentrates on a single answer, and is closest in spirit to the inference-time instability signal of \citet{chen_i_2026}, defined on JSD between token distributions rather than K=5 answer distributions.

---

## H. Length-vs-Signal Spearman ρ per Cell

- [ ] Table tab:spearman reports Spearman ρ between `tokens_generated` and each signal per (arch × dataset) cell on closed-book traces, the length-confound diagnostic of \citet{santilli_revisiting_2025} (their Table 2).
- [ ] |ρ| ≥ 0.5 indicates a strong length confound on the signal.

### Table `tab:spearman` caption

- [ ] Spearman ρ(`tokens_generated`, signal) per cell on closed-book traces.
- [ ] Bold: |ρ| ≥ 0.5.
- [ ] The peer-pool signals (DRAGIN, ent.max, etc, flare.max, u_cons.max, seakr_k5) carry ρ ∈ [+0.26, +0.46], `hvr.max` reaches +0.51 on Gemma×2Wiki, and the auxiliary hidden-state geometry diagnostics span ρ ∈ [−0.07, +0.36].
- [ ] The answer-trajectory aggregations (bottom block) average ≈ 0.15 lower in |ρ| than the peer-pool block, the per-cell quantitative version of the family-level finding from §Results-Matrix.
- [ ] *: Qwen×Hot `hvr.max` is constant, so Spearman is undefined.

---

## I. K=5 vs. K=1 Ablation

- [ ] Table tab:k-ablation reports the paired-bootstrap Δ_{K=5 − K=1} for `flip_rate` on `lift_em`=+1, pooled per architecture.
- [ ] Both CIs straddle zero (§Results-Ablation).

### Table `tab:k-ablation` caption

- [ ] K=5 vs. K=1 `flip_rate` ablation on `lift_em`=+1, pooled per architecture.

---

## J. Adapter Notes

- [ ] We audit signal-level discriminative power on `lift_em` (§Limitations).
- [ ] Several primitives in the published trigger pool are originally framed as online per-token triggers or per-trace scores, and we evaluate their underlying statistic at the trace level on the closed-book trace τ_c^(q) decoded with the prompt of App. Prompts.
- [ ] The per-signal notes below state the original primitive, our trace-level adaptation, and any sign or granularity drift.

### ETC \citep{li_modeling_2026}

- [ ] The original ETC operates on a token-level entropy stream; we evaluate Δ²H on the per-step mean entropy series captured at boundary-detection events (§Methods-FCT).
- [ ] We score max|Δ²H| rather than the signed second-difference, so the adapter is magnitude-faithful but not sign-faithful.

### FLARE \citep{jiang_active_2023}

- [ ] The original FLARE is an online per-token retrieval trigger (max_t p_t < τ); we audit the underlying primitive (low-confidence detection) as a trace-level summary max over steps of 1 − min p_t.
- [ ] The trigger *policy* is not audited; the trigger *statistic* is.

### SeaKR \citep{yao_seakr_2025}

- [ ] We use K=5 EOS hidden states for compute efficiency, smaller than the paper's reported best-k interval K ∈ [10, 25].
- [ ] The regularized Gram determinant we compute is the same primitive as the paper's self-aware uncertainty score.
- [ ] INSIDE \citep{chen_inside_2024} is the closest peer, using the EigenScore on the centered sample covariance; the K nonzero eigenvalues of our uncentered Gram and INSIDE's centered covariance are rank-equivalent up to centering.

### SELFDOUBT-T1 \citep{pandey_selfdoubt_2026}

- [ ] HVR = h/(v+1) is computed per step from a fixed seed marker dictionary (\citeauthor{pandey_selfdoubt_2026} use a per-model Stage-2 expansion from the same seeds); we aggregate as max over steps and report AUROC of the continuous score.
- [ ] The original paper's Tier-1 "HVR=0" high-precision *gate* is the binary degenerate ranking induced by threshold 0 on this same statistic; we do not separately report gate precision/coverage.
- [ ] \citeauthor{pandey_selfdoubt_2026}'s evaluation is on multi-choice reasoning benchmarks (BBH, GPQA-Diamond, MMLU-Pro) over long thinking-mode traces, and they explicitly flag free-form generation as untested; our regime is short non-thinking-mode chat traces on multi-hop short-answer QA, so the audited result here measures HVR's transfer to a regime its authors flagged as untested.

### Concurrent commitment-time primitives

- [ ] Operational differences between \citet{datta_large_2026}'s t*, \citet{zhang_when_2026}'s τ_commit, and our FCT are tabulated in Appendix FCT-tstar.

---

## K. Reproducibility

- [ ] All decoding traces, paired closed-vs-oracle JSONLs, slice manifests with SHA-256 checksums, decoding command lines, and the LLM-as-Judge call cache (1,600 per-call JSONs, oracle and closed arm for each of 800 paired traces) will be released alongside the analysis scripts that produce every table and figure in this paper, including the worked example trace of App. Prompts.
- [ ] Compute box: single NVIDIA A800 80 GB, wall-clock ~15 h end-to-end.
