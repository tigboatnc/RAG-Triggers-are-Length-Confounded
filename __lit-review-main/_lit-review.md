# Literature Review — RQ1 (Dynamic-RAG Triggers, Length-Controlled Evaluation, Step-Level Answer Instability)

> **Purpose.** Situate our experimental campaign (P0, P0.5, P0.6, R1 Phi/Qwen/Gemma, A1 K=1 vs K=5 ablation, planned R2/E7/E8) within the published literature.
> Each Section 1 entry is a self-contained capsule of what each paper does. Section 2 reorganizes those papers by the topics our work touches: (i) step-level answer-trajectory primitives, (ii) length-as-confound in UQ evaluation, (iii) stepwise / streaming uncertainty in CoT, (iv) verbosity / length compensation, (v) hallucination-detection evaluation methodology, (vi) UQ-evaluation methodology pitfalls, (vii) representation/trajectory-based hallucination signals.
>
> **Our anchoring code & data (one-line index)**
> - `analysis/a1_k1_vs_k5_ablation.py` — K=1 (Datta-style proxy) vs K=5 (boundary-completion majority) `flip_rate` ablation, paired bootstrap B=10,000, OLS residualization on `log(tokens+1)`.
> - `analysis/r1_phi_paired_audit.py`, `analysis/r1_phi_d1_closed_audit.py`, `analysis/r1_qwen_paired_audit.py` — full trigger-pool audits at n=200/cell × 3 archs × 2 datasets, both `lift_em=+1` (paired oracle re-decode) and `closed_em=0` targets, length-residualized AUROC.
> - `analysis/length_audit_v2.py` — origin V2 audit on Phi (n=393 HotpotQA / n=452 2Wiki).
> - `analysis/_explore_*.py` — P0.6 deep exploration that surfaced AID (Answer Instability Detection) family (`flip_rate`, `late_flips`, `min_majority`, `flip_rate K=1`).
> - **AID family** = step-to-step disagreement across K=5 boundary-completion answers; primary headline-survivor on Qwen pooled (resid AUROC 0.648, paired-bootstrap-significantly beats `tokens_generated`, CI [+0.020, +0.224]).
> - **DRAGIN-RIND** = co-survivor on Qwen pooled (resid AUROC 0.605, CI [+0.003, +0.168]).
> - **Length residualization protocol** = OLS `signal ~ log(tokens_generated + 1)`, evaluate AUROC on residual; mirrors Belem/Vashurin UNCERTAINTY-LINE.

---

# Section 1 — General overview (paper-by-paper capsules)

_Filled paper-by-paper as the review proceeds. Order matches the alphabetical PDF inventory in `__lit-review-main/`._

## \cite{Chen_DynamicInstabilityLLMReasoning} — Chen, Cheng, Han, Keselj (2026), *"I May Not Have Articulated Myself Clearly": Diagnosing Dynamic Instability in LLM Reasoning at Inference Time*, arXiv:2602.02863, Feb 2026.

**What they do.** Define a black-box, training-free, *single-trace* "instability" signal on top-k token log-probs:
\[
I_t = D_t + \lambda H_t, \qquad D_t = \mathrm{JSD}(\tilde p_t, \tilde p_{t-1}),\ H_t = \text{entropy}(\tilde p_t),\ \lambda = 1\ (\text{fixed, no tuning}),
\]
summarize each trace by **peak instability strength** \(S = \max_t I_t\) (and early-window controls \(S_{w}\) for \(w \in \{10,20,50,100\}\)).
**Operationalize timing** via the relative peak position \(\rho = t^\star / T\): "early peak" \(\rho < 0.25\) = **corrective instability** (recoverable, more often correct); "late peak" \(\rho > 0.5\) = **destructive instability** (insufficient remaining decoding horizon, more often wrong).

**Setup.** GSM8K (first 300 test + full 1319 test) and HotpotQA distractor validation (7405 ex). Models 0.5B–8B: Qwen2.5-1.5B-Instruct, Llama-3.2-1B/3B-Instruct, Llama-3.1-8B-Instruct. Greedy + small temperature grid \(\tau \in \{0, 0.3, 0.7\}\), top-p=0.9, top-k=50 logged, **128 new-token cap to reduce length confounding**.

**Headline numbers.** \(\mathrm{AUC}_{\text{wrong}}\) 0.66–0.74 across model × decoding (e.g., Qwen2.5-1.5B greedy 0.681; Llama-3.2-1B \(\tau=0.7\) **0.741**). Bucketed accuracy is monotonically declining across 5 equal-sized strength quintiles. Peak-position table: early \(\rho<0.25\) → 46% accuracy (57% of traces), late \(\rho>0.5\) → 14% accuracy (14% of traces) — **>3× accuracy gap conditional only on peak timing**. Fixed-window \(S_{50}\) preserves AUC ≈ 0.605 (Qwen) / 0.665 (Llama), proving the signal is **not** an artifact of longer traces.

**Ablation.** JSD-only (\(\lambda=0\)) collapses to chance (AUC ≈ 0.52); entropy alone is the weak component but the **combination** is what works. (Aligns with our finding that single-component logit signals fail residualization but compositional ones survive.)

**Scope claim.** Diagnostic only — explicitly not a controller, calibrator, or hallucination detector. Compatible with retrieval-trigger downstream use but not the paper's stated framing.

**Why it matters for us.**
1. **Direct precedent for our `late_flips` mechanic** — same early-vs-late corrective/destructive logic, but on token-distribution dynamics rather than on answer-set majority. We MUST cite this in §2 / §5; the agent's lit-precedent block already flags it as "Hou et al. 2602.02863" (this is the same paper — ID matches).
2. **External validation of length-confound concern** — they cap to 128 new tokens AND report fixed-window \(S_w\) controls. Our `log(tokens+1)` OLS residualization is the regression-based analogue; same intent, different mechanic.
3. **Parallel signal class but disjoint instrumentation** — they need per-step **top-k token distributions**; we currently log only `per_token_max_logprob` (top-1). A faithful in-paper audit replication would require a re-decode with top-k≥50 logging. **Out of scope for current campaign**, but flag as future work.
4. **Same datasets (HotpotQA), partially-overlapping model size band** (Phi-4-mini 4B, Qwen2.5-7B vs their 0.5–8B). Their cross-model AUC stability (0.66–0.74) is **directly comparable in regime** to our cross-arch AID-family resid AUROC of 0.560–0.690.

**Code-audit verdict.** Cannot byte-for-byte replicate without a re-decode (we lack the top-k distributions). Our `etc (Δ²H max)` signal — second difference of mean step entropy — is the *closest existing audit-pool proxy* (resid AUROC 0.470 Phi-D1, 0.582 Qwen-pooled). It survives residualization on Qwen but is weaker than AID; **the gap suggests the JSD-distributional-shift component carries signal beyond entropy alone**, which is exactly what Chen et al.'s Section 5.4 ablation reports. Concretely actionable: add a §5 paragraph noting that our strongest AID variant (`min_majority`, semantic-level analogue of \(D_t\) at the answer level) is consistent with the stronger signal Chen et al. attribute to the JSD term.

**Honest caveats.**
- Their reported AUC is on **incorrectness** target (`y=1` if wrong); ours is on the orthogonal **`lift_em=+1`** retrieval-helpfulness target. AUC bands are not directly comparable; only direction-of-effect is.
- Their largest model is 8B; our Gemma cell (26B MoE) extends beyond their probed range.
- Their 128-token cap is much shorter than our typical 250–500 token traces; their length-control is *truncation*, ours is *regression-residualization*. Different statistical posture; both are valid.

---

## \cite{Datta_DecideEarlyExplainLater} — Datta, Zhao, Verma, Mamidi, Marreddy, Mehler (2026), *Large Language Models Decide Early and Explain Later*, arXiv:2604.22266 (ICML 2026), Apr 24 2026.

**What they do.** Define **forced answer completion**: at each partial reasoning prefix \((R_1, \ldots, R_i)\) of a thinking-model trace, append the closing `</think>` token plus a task-specific answer prefix \(P\), then sample to obtain an intermediate predicted answer \(A_i\). Repeating after every reasoning step (sentences as units) yields an **answer trajectory** \(T = (A_0, A_1, \ldots, A_n)\). On top of \(T\) they define:
- **AnswerSwitches**(T) \(= \sum_{i=0}^{n-1} \mathbb{1}[A_i \ne A_{i+1}]\) — total switch count.
- **Final answer-switch index** \(t^\star = \max\{i : A_i \ne A_{i+1}\}\) (\(-1\) if none).
- **Transient answer flips (TAFs)** with default window \(k=3\): brief deviations that revert.
- **Hold-for-\(k\) causal smoothing** (Eq. 4): require an answer to persist \(k\) consecutive steps before counting as a switch — denoises trajectories online.
- **Tokens after final switch** \(T_{\text{after}} = T_{\text{total}} - T_{t^\star+1}\) — the wasteable tail.

**Setup.** Qwen3-4B (main), Qwen3-8B, Qwen3-30B-A3B (MoE). Four task families: MCQ (Wang et al. 2024), numeric (Lightman et al. 2023, PRM800K), search-query (Monteiro et al. 2024; cosine-sim ≥ 0.9 between embedded queries used as equality), tool-selection (QuotientAI 2025). Reasoning unit = sentence (consistent with `_rq1-plan.md` boundary detector spaCy-sentence convention). The intermediate prediction is **single-sample** (the model sampled once after the forced answer prefix).

**Headline numbers.**
- Predicted answers change in only **32% of queries** averaged over datasets on Qwen3-4B → answer is fixed from the earliest stages in 68% of traces.
- After the final answer switch, the model generates **760 additional reasoning tokens on average**, up to **1000 tokens** in some cases.
- Probe-based early stopping at the detected final switch reduces reasoning tokens by **~500 per query at only 2% accuracy drop**.

**Why it matters for us — this is the most paper-defining precedent in our pool.**

1. **AnswerSwitches ≡ our `flip_count` primitive; \(t^\star\) ≡ our `late_flips` localization mechanic.** The mechanical primitive of measuring step-to-step answer change at intermediate reasoning prefixes is **identical** at the formal level. The agent's action item #2 is to credit Datta et al. in the AID family field documentation; this lit review locks in that obligation. **Drop any "novel — no published trigger uses step-to-step answer stability" framing.**
2. **The K=5 vs K=1 contrast IS the cross-paper distinction.** Datta uses a single sample per intermediate prediction (forced-completion → sample once → record). We use K=5 boundary-completion samples per step and majority-vote. **Our A1 ablation is the formal experimental discriminator between the two specifications.** A1 on P0 + R1 Phi + R1 Qwen verdict = **regime 2 (K=5 ≈ K=1, CI straddles 0)** on `lift_em=+1` for n=800 paired traces. That means the K=5 multi-sample contribution is **empirically null** at headline-grade power. **The honest framing is: "we extend Datta et al.'s forced-answer-completion to a retrieval-trigger setting; K is empirically free."** This is exactly the framing the agent committed to in the R1 Qwen readout.
3. **Application target is orthogonal — they go for early stopping, we go for retrieval triggering.** That is the genuine contribution wedge that survives the precedent walkback. Datta's stopping question is "when has the model decided?"; ours is "when is retrieval needed?". On the same primitive, opposite causal directions: their `t^\star` indicates *no further compute is needed*; our `flip_rate` / `late_flips` indicate *external information would help*. Both can be true on the same trajectory family.
4. **Models are disjoint.** They use the Qwen3 family (4B/8B/30B); we explicitly excluded Qwen3.6 (P0 SKIPPED — chain-of-thought collapses under our matched-protocol greedy + `enable_thinking=False`). They use thinking-mode protocol with `<think>...</think>`; we use Option-A non-thinking greedy. **The capture protocols are NOT directly comparable** — their forced-completion requires the explicit `</think>` token; ours uses spaCy-sentence boundary detection on a non-think-tag-bearing trace. Worth noting in §3.
5. **Hold-for-\(k\) smoothing is something we do NOT do.** Our `flip_rate` is raw transitions. The agent has flagged in `_rq1-agent.md` that AID variants beat raw `flip_rate` (`min_majority` on Qwen 2Wiki resid 0.672 vs `flip_rate K=5` resid 0.549) — Datta-style hold-for-3 smoothing on `flip_rate` is a **drop-in additional AID variant** to test post-hoc on R1 traces. **Pre-registered actionable for §5: add `flip_rate_hold3` to the analyzer; cost = 1 hour CPU.**
6. **The "model decides early" finding is consistent with our "oracle eliminates multi-step reasoning entirely" P0.6 finding** (oracle traces drop n_steps from 7.3 → 0.0, tokens from 132 → 111 on helpful traces). Different framing, same underlying observation that models can answer immediately when context is available — they don't *need* the long CoT, they just generate it. **Our retrieval-trigger angle is the operational question that answers Datta's diagnostic question: which queries are the 32% where the answer keeps changing? That is exactly the retrieval-helpful subpopulation.** This is a powerful synthesis sentence for the §1 framing.

**Code-audit verdict.**
- Their forced-answer-completion intervention with `</think>` + prefix is **partially captured** by our cancelled `prefix_answer_now` (N2) field (rq1-pilot-v2 schema, never run at scale). N2 was: at each boundary, fork cache → prefill `Answer: ` → decode 10 tokens greedy → save. The architectural difference: they use the model-specific thinking-end token; we use a plain `Answer:` prefix. For non-thinking models (our pool: Phi-4-mini, Qwen2.5-7B-Instruct, Gemma-4-26B-A4B-it) the `Answer:` prefix is the correct analogue. **Recommend re-launching N2 capture on R1 Gemma cell only (~1.5 h compute) to obtain a faithful Datta-replication AID variant on at least one architecture; this is the cleanest defense against "K=5 is just stochastic noise on top of a Datta-style probe" reviewer attack.**
- The `extract_answer` regex in our `analysis/a1_k1_vs_k5_ablation.py` is `r"\bAnswer\s*:\s*(.+?)(?:[\n.]|$)"`, normalize lowercase + strip non-alphanumeric, truncate to 30 chars. Datta's per-task answer extractor is task-specific (MCQ option, numeric token, query cosine-sim ≥ 0.9, tool identifier). **For HotpotQA + 2Wiki our regex is appropriate**; no audit fix required. Document in §3 that the answer-equality check is **stricter than Datta's** (we require exact normalized string match; they allow cosine-sim equivalence on search queries).

**Honest caveats / what would refute us.**
- If a Qwen3-thinking-mode A1 replication on the same multi-hop QA datasets produces **K=1 paired-significantly beats K=5** (regime 3), that re-instates the Datta primitive as the field-canonical and reduces our contribution to "we apply Datta to retrieval." That is still publishable but in a substantially weaker frame.
- Their 760-token post-decision-tail finding implies our K=5 boundary captures may be **dominated by post-decision content** on the substantial fraction of traces where the model decided early — i.e., our `flip_rate` is computed over steps that mostly do nothing. This is consistent with the **Phi sparse-flip limitation** the agent already flagged (only 52/200 R1 Phi traces have ≥2 answer-bearing steps). **Specifically actionable: report the fraction of "decided-early" traces (`t^\star ≤ 1`) per arch × dataset cell** so reviewers can see this is the same phenomenon, not an instrumentation artifact.

---

## \cite{Ielanskyi_PitfallsUQEvaluationNLG} — Ielanskyi, Schweighofer, Aichberger, Hochreiter (2025), *Addressing Pitfalls in the Evaluation of Uncertainty Estimation Methods for Natural Language Generation*, arXiv:2510.02279, Oct 2025.

> **Attribution correction.** `_rq1-agent.md` cites this arXiv ID as "Castaneda et al." — that is incorrect. The actual authors are Ielanskyi, Schweighofer, Aichberger, Hochreiter (ELLIS Linz / NXAI). Fix the attribution before submission.

**What they do.** Critique the standard selective-prediction evaluation protocol for UQ in NLG. Three concrete pitfalls:
1. **Approximate correctness functions disagree substantially.** ROUGE-1/2/L/Lsum, BLEU, BERTScore, BLEURT, LLM-as-judge variants (Llama-3-70B-IT, Llama-3-8B-IT, MOJI) produce mutually-inconsistent binary labels at threshold 0.5 — pairwise mutual AUROC as low as 0.53 on COQA / TRIVIA / SQUAD / Llama-3-8B-IT.
2. **Method ranking changes with the correctness function.** Spearman ρ between rankings under different correctness functions is sometimes **negative** (e.g., BLEU vs ROUGE on COQA: ρ = −0.66). This means *the published winners of UQ-NLG papers are partly an artifact of which correctness function the authors picked.*
3. **Length confound.** They explicitly invoke Santilli et al. (2024) on length as a confound and treat correctness-function-specific length sensitivity as one of the bias channels. (Their figures 1c on the right show generated/reference answer length distributions overlaid with correctness-function agreement.)

**Their fixes.**
- **Marginalize over multiple LLM-as-a-judge variants** for QA selective-prediction.
- Add **structured tasks** (with exact correctness), **OOD detection**, and **perturbation detection** as additional risk-correlation experiments — the three properties \(\hat u\) should satisfy.
- **Elo rating aggregation** across all evaluation settings to summarize method utility.

**Setup.** QA datasets COQA, TRIVIA-QA, SQUAD; Llama-3-8B-IT (and others in appendix). Multiple existing UQ methods compared (Kuhn semantic entropy, Farquhar, Duan, Aichberger, Bakman, Kossen, Nikitin, etc. — full list in their Apx. B.1).

**Why it matters for us.**
1. **Direct support for our methodology contribution scope.** Our `lift_em ∈ {+1, 0, -1}` paired closed-vs-oracle target avoids the correctness-function-disagreement problem entirely *because both `closed_em` and `oracle_em` use the same exact-match correctness function*. The label is `1[oracle_em > closed_em]` — even if EM is biased, the bias is shared between the two arms and cancels in the lift label. **This is a paper-strengthening framing argument we should make explicitly in §3 / §4.**
2. **Refutes any "EM is enough" framing.** Our `closed_em = 0` secondary target IS subject to the EM correctness-function-disagreement Ielanskyi flags. We should down-weight closed_em-based claims accordingly and headline lift_em.
3. **External authority for our R1 D1 closed_em finding** that `tokens_generated` AUROC = 0.645 beats every internal signal — Ielanskyi catalogues exactly this kind of confound (length-correlated correctness label) as one of the pitfalls. Cite together with Janiak's Illusion of Progress (next paper) as the dual external-authority reference.
4. **Structured / OOD / perturbation risk indicators are an evaluation gap our paper does NOT close.** Future work — they advocate going beyond QA selective prediction; we are still in QA. Acknowledge in §6 limitations.
5. **Elo aggregation across (arch × dataset × signal × question-type) cells is not something we currently do.** Worth considering for the survivor matrix in §4 — instead of "AID family wins 4/4 cells", we could compute an Elo over the survivor pair on the full (arch × dataset × question-type) grid. **Cost = 1 hour of analysis; yields a single number reviewers can latch onto.**

**Code-audit verdict.** N/A — they critique evaluation protocol, not a method to replicate. Useful only as framing / methodology citation. Their LLM-as-judge marginalization recipe is **not** applicable to our `lift_em` target (which doesn't use a judge).

**Honest caveats.**
- They focus on short-answer QA selective prediction. Our setting is multi-hop QA with paired oracle re-decode — partially overlapping but not identical.
- Their critique applies most strongly to *single-correctness-function* papers. We use one (EM); we don't compare across correctness functions, so we are technically vulnerable to their critique #2 (rank changes under alternative correctness). **Pre-empt by reporting EM, F1, and a containment-match correctness function in an appendix table** — cheap insurance, no compute cost.

---

## \cite{Janiak_IllusionProgressHallucinationDetection} — Janiak, Binkowski, Sawczyn, Gabrys, Shwartz-Ziv, Kajdanowicz (2025), *The Illusion of Progress: Re-evaluating Hallucination Detection in LLMs*, EMNLP 2025 (pp. 34728–34745).

**What they do.** Three concentric findings on hallucination-detection benchmarking:
1. **ROUGE-L (threshold 0.3) is poorly aligned with human factual-correctness judgments** in QA hallucination detection. Human study: 200 NQ-Open Mistral answers, 3 annotators per item, Cohen's Kappa = 0.799 inter-annotator agreement; ROUGE has high recall, **alarmingly low precision**.
2. **Re-ranking the published trigger pool under LLM-as-Judge (GPT-4o-Mini) labels collapses performance** — Perplexity drops 45.9%, Eigenscore drops 30.4% from ROUGE to LLM-as-Judge labels (Figure 1 across NQ-Open, SQuAD, TriviaQA on Llama-3.1-8B-IT and Mistral-7B-IT).
3. **"Simple heuristics based on response length can rival complex hallucination detection techniques."** Length matches or surpasses Semantic Entropy on at least some cells. Causal manipulations: longer / more ambiguous responses are more prone to hallucination; ROUGE can be gamed by trivial repetition.

**Setup.** NQ-Open (3,610), TriviaQA validation subset (3,842), SQuAD validation rc.nocontext (4,150). Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3. Methods: Perplexity, Length-Normalized Entropy (LN-Entropy), Semantic Entropy (Farquhar et al. 2024), EigenScore (Chen et al. 2024 INSIDE), LogDet (Sriramanan et al. 2024a), eRank (Roy & Vetterli 2007 / Garrido 2023, repurposed). Sampling: n=10 at temp 1.0 + one "best answer" at temp 0.1. Both zero-shot and 5-shot.

**Why it matters for us.**
1. **This is the dual external authority that backs our R1 D1 closed_em finding** (Phi × HotpotQA, n=200, `tokens_generated` AUROC = 0.645, beats every internal signal raw and residualized; resid AUROC ≥ 0.5 only for AID `late_flips` 0.529 and 4 borderline). The Illusion of Progress essentially says: *"under length manipulation and judge-aware labels, length is the floor every method must beat."* Our `lift_em = +1` paired-oracle target is the per-instance generalization of their concern: we don't ask "is this answer wrong?" (which length predicts), we ask "would retrieval flip this answer?" (which length predicts much more weakly). **Both target the same lurking confound from opposite ends. Cite together with Vashurin's UNCERTAINTY-LINE in the §1 framing paragraph.**
2. **External validation that simple-length-rivals-fancy-internal-signals is real, not just our pipeline.** Our V2 Phi finding (`tokens` 0.700 raw, beats DRAGIN 0.670 raw on HotpotQA) is in regime with their finding. **The agent's framing — "length-confound is the dominant single phenomenon in QA UQ" — is now externally validated. Use this in the §1 stake-in-the-ground paragraph.**
3. **Their methods evaluated do NOT include DRAGIN, FLARE, SeaKR or AID.** They cover the UQ-classical pool (PPL, LN-Entropy, SemEntropy, EigenScore, LogDet, eRank); we cover the dynamic-RAG-trigger pool (DRAGIN, FLARE, SeaKR-K5, AID variants, `tokens_generated`, etc). **Our paper extends Janiak's critique to a different method family.** This is a contribution argument: *"We replicate Janiak's `length-rivals-everything` finding in the dynamic-RAG-trigger family on multi-hop QA, and identify the two specific exceptions — DRAGIN-RIND and the AID family — that survive the length confound on at least one architecture."*
4. **Their ROUGE / LLM-as-Judge correctness disagreement is the same critique Ielanskyi et al. levels.** We use EM (a stricter relative of ROUGE-L) and our paired-oracle setup partially insulates us via shared bias (see Ielanskyi entry above). **Pre-empt by adding LLM-as-Judge label-comparison rows in an appendix table on a 200-trace subset (~30 min CPU + small GPT-4o-Mini cost).** This is the cheapest single thing that materially raises acceptance probability.
5. **Their "manipulating verbosity" causal experiment is something we do NOT do.** They prove length is causally implicated, not just correlated. Our regression-residualization protocol controls for length; their causal interventions establish *why* you have to control for it. Cite as the foundational motivation in §3.

**Code-audit verdict.** N/A — methodology critique paper. Our use is purely as a citation anchor.

**Honest caveats.**
- Their finding is on **single-hop QA** (NQ-Open, TriviaQA, SQuAD); our setting is **multi-hop QA** (HotpotQA, 2Wiki, MuSiQue planned). The length-confound is documented in single-hop; our R1 results extend it to multi-hop. **Direction-of-effect aligns; magnitudes likely differ.**
- They use a single open-weights setup (Llama-3.1-8B-IT + Mistral-7B-IT). We span Phi-4 + Qwen2.5-7B + Gemma-4-26B-MoE. Cross-arch generality is *our* contribution beyond theirs.
- They identify length as a problem; we **propose the residualization fix** (jointly with Vashurin/Belem) — they do not propose a fix beyond "use LLM-as-Judge for labels."

---

## \cite{Lu_StreamingHallucinationDetectionCoT} — Lu, Pan, Li, Nan, Zhuang, Zhao, Sun, Wang, Liu (2026), *Streaming Hallucination Detection in Long Chain-of-Thought Reasoning*, arXiv:2601.02170, Jan 2026.

**What they do.** Recast hallucination in long-CoT as an **evolving latent state**, not a one-off event. Two probes on hidden states:
- **Step-level probe** \(c^{\text{step}}_t = P(z^{\text{step}}_t = 1 \mid h_t)\) — locally identifies the current step as introducing unsupported content.
- **Prefix-level probe** \(c^{\text{prefix}}_t = P(z^{\text{prefix}}_t = 1 \mid h_t, c^{\text{step}}_t)\) — globally tracks whether the entire reasoning prefix has entered a hallucinated regime.

Step representation: per-step token-hidden-state aggregation with **fixed time-aware exponential weighting within the step** \(w_n = (n-1)/(L_t - 1)\), then \(\ell_2\) normalization. Designed to fix two failure modes of additive aggregation: (i) cross-step saturation (later prefixes saturate, lose discriminability) and (ii) within-step imbalance (later tokens carry more information than uniform-mean assumes).

**Setup.** 10,000+ long-CoT trajectories, 200k+ reasoning steps, 40k+ hallucinated steps. Three base models: **Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, DeepSeek-R1-Distill-Llama-8B**. Queries from Trivedi et al. (2022) and Suzgun et al. (2022). Annotation by **Claude 4.5** with three validation passes (answer-aware semantic check + logical consistency + manual verification on stratified samples). Final-answer hallucination rates: 73% Llama, 67% Qwen, 48% DeepSeek; step-level rates 38.8% / 36.0% / 13.7% respectively.

**Headline numbers.**
- Step-level AUROC **87.83% on Llama-3.1-8B, 86.70% Qwen2.5-7B, 93.27% DeepSeek-R1-Distill-8B** (their best step-level probe). Beats SAPLMA (Azaria & Mitchell 2023) and TTPD (Bürger et al. 2024) by 4–6%.
- Layer-wise: strongest signal at **intermediate layers**; gap across layers is small (cross-step exponential aggregation makes shallow layers nearly as informative as deep ones).
- **Temporal sensitivity along CoT: AUC drops from ~0.89 (early third) to ~0.82 (late third); ACC drops from 0.89 to 0.74; F1 INCREASES from <0.35 to ~0.78.** This is the inverse of the Chen et al. early-vs-late corrective/destructive distinction — they find that step-level *detection* gets harder later, but the *prevalence-weighted F1* improves because hallucinations are denser late.
- Prefix-level: 78% online correct identification of CoT instances "as reasoning unfolds."

**Why it matters for us.**
1. **Closest temporal-state framing in the literature, but on a strictly different method family.** They train probes on hidden states (supervised, with Claude-4.5 labels); we are explicitly **parameter-free, training-free, decode-time** (RQ1 statement, `_rq1-plan.md` §"RQ1 statement"). **Our paper EXCLUDES this family** (DioR, CtrlA, MBA-RAG, Self-RAG already dropped on the same principled grounds in `_rq1-plan.md`). Cite as "out-of-family but conceptually parallel."
2. **Same arch overlap (Qwen2.5-7B-Instruct).** Their step-level Qwen2.5 AUROC of 0.867 against Claude-4.5 hallucination labels is **a strong external upper bound on what is achievable with hidden-state-trained probes on the same backbone we use.** Our best parameter-free signal on Qwen2.5 (AID `min_majority` resid 0.648 on `lift_em=+1`, or 0.672 raw on Qwen 2Wiki) is ~0.20 AUROC below their *trained* upper bound, **but on the orthogonal `lift_em` target**. Direction-of-magnitude expected: trained probes on the same task dominate parameter-free signals; our value-add is *training-free + retrieval-trigger framing*, not raw hallucination detection.
3. **Step-level vs prefix-level decomposition is conceptually parallel to our (per-step `flip_rate` aggregation × per-trace `late_flips` localization).** Both parses of the same idea: local evidence + temporal aggregation. Their evidence that **prefix-level probe with step-level guidance > prefix-level alone** validates the family-of-aggregations approach we already adopt with AID (4 aggregations: `flip_rate`, `late_flips`, `min_majority`, `flip_rate_K=1`). Cite in §3 / §4 as "parallel evidence that step-level + global-aggregation outperforms either alone."
4. **Their `Late 1/3 AUC drops` finding contradicts our `late_flips` direction-of-effect on Phi 2Wiki/comparison (raw AUROC 0.950).** This is a **real cross-paper tension worth a §5 discussion paragraph**: their probe-based detection signal weakens late, while our answer-instability signal *strengthens* in the late window on at least one cell. Possible explanation: their c_step measures whether the model is *introducing* false content; our late_flips measures whether the model is *changing its answer*. These are different events that can dissociate. Cite Lu and the Datta `t^\star` 760-tokens-after-final-switch finding together as evidence that "late-trajectory dynamics are both stabilization-dominated AND informative — different signals capture different aspects."
5. **Their dataset construction (10k+ long-CoT trajectories with Claude-4.5 step-level labels) is a model for what an ideal MuSiQue-extension dataset would look like for us.** Out of scope for the EMNLP-short submission; explicit future-work citation.
6. **Their 8 logic-based dynamic metrics + 7 empirical observations on hallucination dynamics are the closest published analog to our 10-angle exploratory analysis (`analysis/_explore_*.py`).** Worth comparing analytical frames in §5 — they catalog hallucination dynamics, we catalog answer-instability dynamics; both find timing matters.

**Code-audit verdict.**
- We **cannot** replicate their step-level probe without (a) per-step per-token full hidden states (currently we capture only EOS hidden + sample-EOS-hidden), (b) the Claude-4.5 annotated step-level hallucination labels (out of compute scope), (c) a probe-training pipeline. **Out of scope.**
- Their exponential within-step aggregation \(w_n = (n-1)/(L_t-1)\) is a **drop-in alternative aggregation** for our `dragin_rind_max.max` and `drift_cosine.max` if we ever want to do parametric aggregation. **Pre-registered actionable for §5 future work, not for current campaign.**

**Honest caveats.**
- Their results are on **long-CoT thinking-mode traces** (DeepSeek-R1-Distill is reasoning-tuned). Our pool is mostly **non-thinking-mode greedy** (Phi-4-mini, Qwen2.5-7B, Gemma-4 in Option-A). The trajectory length distributions and step-density are not comparable. They average 18.2–32.0 steps per CoT; we average ~7 steps on Phi/Qwen.
- Final-answer hallucination rate baseline (48–73%) is far higher than our `closed_em = 0` rate (~59% on Phi HotpotQA; comparable but on a different task family).
- Annotated by Claude 4.5 — Janiak's critique of single-judge labels applies (cite the tension).

---

## \cite{Mao_ConfidenceOverTimeSTL} — Mao, Venkat, Bisliouk, Kothiyal, Kumbakonam Subramanian, Singhu, Ruchkin (2026), *Confidence over Time: Confidence Calibration with Temporal Logic for Large Language Model Reasoning*, arXiv:2601.13387 (ACL 2026), Jan 19 2026.

**What they do.** Treat per-token probability sequence as a temporal signal; segment into reasoning steps; aggregate per-step as \(s_j = \frac{1}{L_j}\sum_{k} P(y_{t_j+k} \mid y_{<t_j+k}, x)\). Apply **Signal Temporal Logic (STL)** discriminative mining (TeLEx-style) to discover formulas like \(\Box_{[a,b]}(s_t \ge \mu)\), \(\Diamond_{[a,b]}(s_t \ge \mu)\) whose **robustness scores** \(\rho(\varphi, S, t) \in \mathbb{R}\) discriminate correct from incorrect traces. Two formula sets: \(\Phi_{\text{pos}}\) (correct-reasoning patterns) and \(\Phi_{\text{neg}}\) (failure-mode patterns). Extend with a **hypernetwork** that predicts STL parameters per-instance from prompt + signal-summary statistics.

**Setup.** Backbones: **Qwen3-8B, Gemma-3-12B, Llama-3-8B**. Tasks: GAOKAO-Math, CLadder (causal MCQ), SciQ, BBH (Big-Bench-Hard). All MCQ / binary correctness — *not* multi-hop QA. Calibration metrics: ECE, Brier Score, AUROC. Single-sample inference except self-consistency baselines.

**Headline numbers.**
- **AUROC of the STL approach is ~0.61–0.62** (Table 2 hyperparameter study, 10-template balanced config). This is calibration-headline-equivalent — they prioritize ECE / Brier, not selective prediction.
- ECE down from baselines (AveLogit 0.05–0.55, Self-Eval 0.04–0.59, Self-Consistency 0.04–0.47) to **0.005–0.054 with their full hypernetwork-STL method** across all 4 benchmarks × 3 backbones.
- **Failure-mode (Φ_neg) patterns transfer cross-task; correct-mode (Φ_pos) patterns are task-specific** (Figure 3 Jaccard similarity). Inference time 0.55s/example.

**Why it matters for us.**
1. **Closest paper in spirit to our AID family.** They argue confidence should be modeled as a **temporal signal**, not a scalar; we argue answer-stability should be modeled as a step-level temporal signal, not a single end-of-trace score. Same epistemic move, different signal class. **Cite as the most direct theoretical kin in §2.** The agent's lit-precedent block already names it as "closest in spirit, different formalism (STL vs flip-rate)."
2. **Their AUROC of 0.61–0.62 is in the same band as our AID-family residualized AUROCs (0.560–0.690 cell-level, 0.605–0.648 pooled).** The order-of-magnitude is identical; *our methods are competitive with the strongest STL-on-confidence approach without requiring an STL-formula-mining pipeline or hypernetwork training.* This is a quietly powerful framing: "We achieve comparable temporal-signal AUROC with a far simpler, training-free, single-aggregation primitive."
3. **Their finding that failure-mode patterns generalize cross-task while correct-mode patterns are task-specific is consistent with our per-cell heterogeneity finding** (`AID late_flips` 2Wiki/comparison 0.950 vs HotpotQA/comparison 0.250). Failure modes are universal; success modes are task-dependent. **Cite together with the per-question-type heterogeneity discussion in §5.**
4. **They DO acknowledge length bias** explicitly in Section 2 ("joint generation probability suffers from severe length bias, assigning smaller values to longer responses regardless of correctness, making it unsuitable for long-form reasoning") and cite Li et al. 2025b. They mention length-normalized variants (Malinin & Gales 2021) but do not propose a residualization fix. **We cite Mao for "the field already knew length is a problem," and we cite UNCERTAINTY-LINE / Verbosity≠Veracity / Janiak / Santilli for "and the field still has not fixed it" — our protocol then closes the loop.**
5. **Tasks are NOT multi-hop QA** (math, causal, science, Big-Bench-Hard). Our HotpotQA/2Wiki/MuSiQue setting extends a different task family. Direction-of-effect comparable; absolute numbers not.
6. **Targeting calibration ≠ selective prediction.** They optimize ECE/Brier; we optimize AUROC for retrieval triggering. **The correct comparison cell across the two papers is `closed_em = 0` AUROC and `lift_em = +1` AUROC, not ECE/Brier.** Be careful not to mix metrics in the comparison table.
7. **Their hypernetwork is supervised (per-question parameter adaptation requires training data).** Our AID is parameter-free. **We win on data efficiency; they win on per-question adaptivity.** Honest trade-off to flag in §5.

**Code-audit verdict.**
- We can compute their **stepwise confidence signal** \(s_j\) directly from our existing `per_token_max_logprob` capture (rq1-pilot-v1 schema). The within-step aggregation (arithmetic mean of per-token probabilities) is **identical** to one of the candidate aggregations in our analyzer. **Drop-in addition: a `mao_step_conf_min`, `mao_step_conf_decline_slope`, `mao_step_conf_late_low_count` triple of free signals on R1 traces — pre-registered actionable; cost ~30 minutes CPU.**
- We **cannot** replicate the STL discriminative mining or hypernetwork training without an additional supervised pipeline. Out of scope.
- The Φ_neg cross-task generalization can be partially audited: are our AID-family signals in the same direction across (HotpotQA, 2Wiki, planned MuSiQue) cells? Yes (`min_majority` and `late_flips` both > 0.5 resid AUROC on 4/4 cells). **This consistency is itself an argument that our family is operating in a similar regime to Mao's Φ_neg patterns.**

**Honest caveats.**
- Their reported numbers are calibration not selective-prediction; cross-paper AUROC comparison must be done on AUROC rows of Table 2 only (their reported AUROC at 10-template config = 0.617).
- They use Qwen3 (we excluded Qwen3.6 due to thinking-mode collapse under our protocol; they use Qwen3-8B without the same constraint).
- Their backbones partially overlap (Gemma-3-12B vs our Gemma-4-26B-A4B-it; Llama-3-8B vs our excluded Llama-2-7B). Direct cell-level replication impossible.

---

## \cite{Santilli_RevisitingUQEvaluationLengthBias} — Santilli, Goliński, Kirchhof, Danieli, Blaas, Xiong, Zappella, Williamson (2025), *Revisiting Uncertainty Quantification Evaluation in Language Models: Spurious Interactions with Response Length Bias Results*, arXiv:2504.13677 v2, Jun 2025 (Apple + Sapienza + NUS).

> **Attribution correction.** `_rq1-agent.md` cites this arXiv ID as "Belem et al. (ACL 2025 short)." The actual authors are Santilli et al. (Apple). Fix the attribution before submission.

**What they do.** Provide the formal proof of *why* length confounds UQ evaluation, then back it with a 4 × 4 × 8 × 7 empirical matrix.
- **Formal result (their App. C).** Decompose AUROC estimation under noisy correctness labels:
  - (i) If correctness-function errors are **independent** of UQ scores, \(\widehat{\mathrm{AUROC}}\) is **noisy but unbiased**.
  - (ii) If correctness-function errors are **mutually biased** with UQ scores (both depend on the same factor — e.g., length), \(\widehat{\mathrm{AUROC}}\) is **systematically skewed**: some methods appear better than they are, some worse, and rankings flip across correctness functions.
- **Empirical confirmation.** Negative Sequence Probability \(\hat g \propto -\prod_i \hat p(\hat y_i \mid \cdot)\) **mechanically increases with length** \(L\). ROUGE-L / SQuAD-F1 / BERTScore correctness scores **also depend on length** through their lexical-overlap mechanics. Mutual bias is therefore inevitable for the dominant published UQ-method × correctness-function combinations.

**Setup.** 4 datasets × 4 models × 8 UQ methods × 7 correctness functions. UQ methods: Token Length (yes, as a baseline!), Character Length, Perplexity, Negative Sequence Probability, Naive Entropy, Mean Token Entropy, Probe (RougeL-F1>0.5), Probe (LM-as-a-judge), Semantic Entropy with/without length normalization. Correctness functions: ROUGE-1 (F1), ROUGE-L F1 / Recall at multiple thresholds, SQuAD-F1, BERTScore, SentenceBERT, AlignScore, LM-as-a-judge (Prompt). Human study with 450 LM samples × 4 annotator trials. Cohen's Kappa: human–human 0.83–0.86; human–LM-as-judge 0.71–0.75; human–ROUGE-L only 0.45–0.47.

**Headline numbers / spearman correlations** (their Figure 4):
- token-length × Perplexity correlation: **−0.7**
- token-length × Naive Entropy: **−0.6**
- token-length × Mean Token Entropy: **−0.6**
- token-length × Semantic Entropy (without length norm): **−0.7**
- token-length × ROUGE-L (F1) at t=0.5: **0.7** (positive — long answers more likely scored as wrong by lexical metrics).

These are **strong** confounds, comparable in magnitude to UQ-method-vs-UQ-method correlations.

**Why it matters for us — this is the formal foundation.**
1. **This paper supplies the formal proof underlying our entire residualization protocol.** Their scenario (ii) — mutual bias on \(L\) — is *literally what we observe* in V2 Phi (`tokens` AUROC 0.700 raw on HotpotQA) and R1 Phi closed_em (`tokens` 0.645). They prove this bias systematically distorts AUROC; we propose `signal ~ log(tokens+1)` OLS residualization as the regression-based fix. **Cite as the canonical theoretical justification in §3 (residualization protocol section).** Together with UNCERTAINTY-LINE (Vashurin) which proposes the residualization fix, Santilli is the *why*, Vashurin is the *how*.
2. **Their finding "Token Length itself is a UQ method on the leaderboard" directly mirrors our headline-grade finding** that on Phi HotpotQA closed_em, `tokens_generated` (AUROC 0.645) beats every internal signal raw and residualized. They literally include "Token Length" as a UQ method baseline in their Figure 1 leaderboard. **Powerful framing for §1: "Santilli et al. (2025) report token-length as a competitive UQ-method baseline; Janiak et al. (2025) report length heuristics rivaling complex hallucination detectors. We replicate this in the dynamic-RAG-trigger family on multi-hop QA and propose a residualization protocol that surfaces the two signals (DRAGIN-RIND, AID family) that survive the length confound."**
3. **Their LM-as-a-judge endorsement applies to evaluation labels, not to UQ scores.** Our `lift_em` paired-oracle target uses identical (EM) correctness function on both arms, sharing the bias and partially canceling it (see Ielanskyi entry §1). **Their proof + our paired design = cleanest defense against the mutual-bias critique.**
4. **Their datasets are SimpleQA, TriviaQA, NQ-Open, etc. — single-hop QA.** We are multi-hop. Their length confounds are present at single-hop scale; ours are present (and per-cell heterogeneous, e.g., Qwen 2Wiki inverted) at multi-hop scale. **Direction-of-effect aligns; magnitudes per-cell.**
5. **They do NOT propose a residualization fix; they recommend "use LM-as-a-judge for correctness labels."** That fixes one half (the correctness-function-bias half). The UQ-method-bias half is **still open**, and is exactly the gap UNCERTAINTY-LINE / our protocol close. **Frame our paper as the regression-based complement to Santilli's judge-based recommendation.**

**Code-audit verdict.**
- Their scenario-(ii) formalism (App. C) is **directly importable** as a 1-paragraph theoretical motivation in our §3. We should cite the formal result as "Santilli et al. (2025) Theorem [N]." (Pull the exact theorem number on a re-read of their Appendix C; the Read tool's PDF extraction was sparse on the appendix.)
- Their `Probe (LM-as-a-judge)` baseline is a *trained* binary classifier with judge-label supervision; we don't have an analog. Out of scope for replication; useful as reviewer-defense citation.
- Their length-correlation table is a **drop-in template** for our reviewer-prep appendix table: "for each of our 16 trigger-pool signals, report Spearman \(\rho\) with `tokens_generated` and `n_steps` per (arch × dataset) cell." We **already have this** as a side-output of `length_audit_v2.py` and `r1_phi_paired_audit.py`. **Cost = 0; just need to format the table.**

**Honest caveats.**
- Their analysis is single-hop QA + fixed-prompt 4-model setup. Our cross-arch + multi-hop + paired-oracle design is more demanding.
- They include `Token Length` in the leaderboard as a baseline; we use `tokens_generated` as the residualization regressor and as a baseline. **Naming/usage difference; cite carefully.**

---

## \cite{Vashurin_UncertaintyLineLengthInvariant} — Vashurin, Goloburda, Nakov, Panov (2025), *UNCERTAINTY-LINE: Length-INvariant Estimation of Uncertainty for Large Language Models*, arXiv:2505.19060 v2, Sep 25 2025 (MBZUAI).

> **Attribution correction.** `_rq1-agent.md` cites this arXiv ID as "Belem et al. UNCERTAINTY-LINE." The actual authors are Vashurin, Goloburda, Nakov, Panov. Fix attribution before submission. Code released at `https://github.com/stat-ml/uncertainty-line`.

**What they do.** Direct response to Santilli et al. 2025 (which they cite explicitly): even length-*normalized* uncertainty measures (Perplexity, Mean Token Entropy, Token-SAR, MCNSE) **still** exhibit length bias. Propose a regression-based fix:
\[
\text{UNCERTAINTY-LINE: } \hat u_{\text{LINE}}(\hat y, x) = \hat u(\hat y, x) - \hat f(L(\hat y))
\]
where \(\hat f\) is a regression of uncertainty score on output length \(L\). Use the residual as the length-invariant uncertainty estimate. **Post-hoc, model-agnostic, applicable to any UQ measure.** Two variants: unsupervised (regression on length alone) and supervised (when length correlates with quality, regress on both).

**Setup.** Llama 3.1 8B-Instruct. Tasks: neural machine translation (NMT), summarization, mathematical reasoning. **NOT direct QA** — they pick tasks that have varying response length. They show **slope of every UQ method vs. length is statistically significant** (p < 0.001) on every task. Examples (Figure 2-equivalent reads): MSP vs length slope **+0.45** (math); PPL vs length slope **−0.08** (translation), **−0.11** to **−0.26** (other tasks); MTE vs length slope **−0.11**. Length normalization helps but **doesn't eliminate** the slope.

**Why it matters for us — this is the methodological twin of our protocol.**
1. **UNCERTAINTY-LINE = our `signal ~ log(tokens_generated + 1)` OLS residualization.** Same mechanic, different regressor parameterization (they use raw \(L\), we use \(\log(L+1)\) for the well-known long-tailed length distribution). **We MUST credit Vashurin et al. as the originator of the residualization protocol; reframe our methodology contribution as "we adopt and extend the UNCERTAINTY-LINE protocol of Vashurin et al. (2025) to dynamic-RAG triggering with the `lift_em = +1` paired-oracle target."** This is exactly the action item the agent already locked in.
2. **Their unsupervised variant is what our `length_audit_v2.py` and `r1_phi_paired_audit.py` implement.** Our `_residualize_signal` function (in the analyzer code) computes OLS residuals of `signal_value ~ log(tokens_generated + 1)` per cohort, then evaluates AUROC on the residual. **This is byte-for-byte UNCERTAINTY-LINE-unsupervised. Acknowledge in the paper's methodology section.**
3. **Their supervised variant (when length is informative for quality) maps onto a refinement we have NOT yet adopted.** On Phi HotpotQA closed_em where `tokens_generated` AUROC = 0.645, length is *itself* informative for the target. Vashurin's supervised variant would say: residualize signal on length AND quality jointly to extract the residual signal that adds *beyond* what length alone provides. This is essentially the paired-bootstrap-vs-tokens framework we already use (we test if signal beats `tokens_generated` on raw AUROC, which is the discrete analog of "does signal add information beyond length?"). **Cite the supervised variant as the formal version of our paired-Δ-vs-tokens test.**
4. **Their tasks (NMT, summarization, math) do NOT include QA, and definitely not multi-hop QA with paired oracle re-decode.** Our setting is therefore a **non-trivial extension of the UNCERTAINTY-LINE protocol** to a new task family with a new target (`lift_em` rather than QA EM/F1). This justifies the methodology contribution surviving the precedent.
5. **They demonstrate length bias on UQ measures (PPL, MTE, MSP, MCSE/MCNSE, Token-SAR, Semantic Entropy).** We extend the catalogue to dynamic-RAG-trigger signals: DRAGIN-RIND, FLARE, SeaKR-K5-proxy, AID family, SELFDOUBT-T1, drift_cosine, variance_l2_window, hidden_norm — none of which appear in their evaluated pool. **Cross-method generalization of the same finding.**
6. **They critique naive length-normalization as overcorrecting** (citing Gupta et al. 2024). Our `log(tokens+1)` residualization is a regression, not a division — naturally avoids the overcorrection trap they describe. **Worth a sentence in §3 protocol description.**

**Code-audit verdict — direct head-to-head replication possible.**
- We can run **their public implementation** (`github.com/stat-ml/uncertainty-line`) on our R1 traces with `tokens_generated` as the length variable and our 16+4 trigger signals as the UQ scores. This produces a **paired-replication appendix table** showing our `_residualize_signal` function gives the same residualized AUROCs as their reference implementation. **Cost = 2-3 hours of integration work + 30 min CPU. Pre-registered actionable; massively reviewer-defense-positive.**
- Their supervised variant is a 30-line addition to the analyzer; we should add it as `_residualize_signal_supervised(signal, log_tokens, target)` for an §A.X appendix table.
- The choice of regressor (raw \(L\) vs \(\log(L+1)\) vs spline) is a free degree of freedom; report all three in the appendix to head off "you cherry-picked the regressor" reviews.

**Honest caveats.**
- Their primary backbone is Llama 3.1 8B (we excluded Llama-2-7B from current campaign; Llama 3.1 not in our pool). Direct cell-level replication is not possible. Use their implementation on our traces as the bridge.
- They focus on tasks where length varies across instances within a model; we focus on length variance within a fixed (arch × dataset) cell. Different inferential frame, same statistical machinery.
- Their unsupervised regression assumes that length-quality correlation is *not* the source of the signal. On Phi HotpotQA closed_em, length **is** informative for correctness, so the supervised variant (or our paired-Δ-vs-tokens test) is more appropriate. **Be precise about which variant we use per cell.**

---

## \cite{Wang_ChainOfProbeCoTStepByStep} — Z. Wang, Zeng, W. Liu, Y. Wang, L. Li, Y. Wang, Shang, Jiang, Q. Liu, Wong (2025), *Chain-of-Probe: Examining the Necessity and Accuracy of CoT Step-by-Step*, Findings of NAACL 2025 (pp. 2586–2606).

**What they do.** Probe per-step model confidence by appending a fixed probing string \(a^* = \) `" So, the answer is ("` to the partial reasoning trace, take the softmax over a target token set \(V = \{A, B, C, D\}\) (for MCQ), and define the per-step confidence as
\[
c_i = [p_v^i : v \in V], \quad p_v^i = P(y = v \mid x, s_{1:i}, a^*).
\]
\(c_0\) (no CoT) is the direct-answer probe. Sentence-level segmentation between probes.

**Findings.**
1. **Early answering is widespread**: across multiple LLMs on MCQ datasets, in many cases the model commits to the same answer throughout the entire reasoning trace.
2. **Early-answering accuracy is HIGHER than changing-during-reasoning accuracy by >20%** — when the model changes its choice mid-CoT, accuracy is ~20% lower than when it commits early. CoT often hurts on simple questions and is "unnecessary" for them.
3. **Question simplicity predicts early answering**: easy questions trigger early commit; hard questions get more answer-changing behavior.
4. **Higher mid-reasoning confidence correlates with higher correctness**; the confidence-change pattern can be used to score CoT validity.
5. They build **CoP Score** (rank candidates by confidence trajectory; competitive with majority voting) and **CoP Tree** (decision tree on extracted CoP features; identify error-prone CoT and trigger resampling; +13% average accuracy).

**Why it matters for us.**
1. **Strong "soft" precedent for our `flip_rate` and the AID family.** CoP measures *soft* confidence over a closed answer set per step; our AID measures *hard* answer-set transitions per step (extracted from K=5 sample texts). The CoP-Score-vs-CoP-Tree distinction parallels our `flip_rate` (raw rate) vs `min_majority` (within-step variability) variants. **Cite as the third precedent in the answer-trajectory family alongside Datta and Hou/Chen.**
2. **Their finding that "changing-during-reasoning has 20% lower accuracy than early-answering" is the EXACT same direction as our P0.6 deep-exploration finding** that no-flip traces are 61% correct (3% helped), late-flip traces are 32% correct (18% helped), high-flip traces are 20% correct (20% helped). **The mechanism mirrors precisely**: answer-changing during CoT is associated with both lower correctness AND higher retrieval-helpfulness. **This is the cleanest cross-paper concordance available to us; use it as the §1 motivation paragraph: "Wang et al. (2025) and Datta et al. (2026) independently find that LLMs that flip their answer during reasoning are systematically less accurate; we propose this same flip behavior as a retrieval trigger."**
3. **CoP's probe is forced-answer-completion-lite** — fixed `So, the answer is (` prefix + closed token set, instead of Datta's `</think>` + general answer prefix + sampling. Our boundary-completion K=5 sampling is yet a third specification of "what the model would answer right now." **The three specifications form a methodological gradient:** Wang (CoP) = closed-token-soft-probe; Datta = forced-completion-hard-greedy; ours = K=5 stochastic boundary completion + answer extraction. **Pre-registered actionable for §3: report the gradient and explain why we chose K=5 stochastic for retrieval-trigger application (we want to capture sampling variability across plausible boundary trajectories, not just deterministic snapshot answers).**
4. **They use MCQ datasets exclusively.** We use HotpotQA / 2Wiki / MuSiQue (open short-answer multi-hop). The closed-answer-set assumption \(V = \{A, B, C, D\}\) does NOT apply to our setting; we extract free-form answers from K=5 sample texts. **CoP cannot be directly applied to our task family without an answer-set inference step (which our `extract_answer` regex effectively performs by collapsing diverse string answers to a normalized 30-char canonical form).**
5. **CoP Tree's "trigger resampling on error-prone CoT" is conceptually adjacent to retrieval triggering, but uses internal resampling rather than external retrieval.** Our framework substitutes external oracle retrieval for their internal resampling. **Cite together with Datta as evidence that the field already converges on "answer-trajectory dynamics → trigger some intervention" — we differ in *what* is triggered.**
6. **Confidence change pattern → CoT correctness validation** is closely related to the Lu et al. step-level + prefix-level probing framework, but Wang's is *parameter-free at runtime* (decision tree learned offline on extracted features) while Lu's requires per-arch hidden-state probe training. Wang sits between us (fully training-free) and Lu (probe-trained); cite the spectrum.

**Code-audit verdict.**
- Their probing string mechanism is **a different beast than our boundary-completion sampling.** Their probe uses a single greedy forward pass with a fixed prefix; ours uses K=5 sampled completions. To replicate CoP on our traces, we would need to log the per-step softmax distribution over a closed answer set after appending `So, the answer is (`. **For multi-hop QA with free-form answers, this requires constructing a per-question candidate-answer set, which is non-trivial.** Out of scope for current campaign.
- Their CoP Tree feature extractor (max confidence change, position of confidence drop, monotonicity, etc.) is conceptually a multi-feature aggregation of step-level confidence change. **Several of these features have direct analogs in our exploration: `dragin_late_spike` (P0.6 angle 1, AUROC_help 0.709), `late_flips` (AID variant V3), `consecutive_flips` (V8). Pre-registered: add 2-3 CoP-style aggregations to the analyzer for §A appendix; cost ~1 hour CPU.**

**Honest caveats.**
- Their backbones are not specified in the abstract section we read; their setup is multiple-choice reasoning datasets, not retrieval-augmented multi-hop QA. **Direction-of-effect alignment is the strongest cross-paper signal we have; magnitudes do not transfer.**
- They compete on accuracy / answer-selection, not retrieval triggering. The application is orthogonal but the underlying signal family is the same.
- Their CoP Tree achieves +13% average accuracy via *resampling*; we achieve our headline AID result without any resampling intervention (parameter-free retrieval trigger). **We are weaker on the intervention side; equal-or-stronger on the methodology-novelty side.**

---

## \cite{Zhang_VerbosityVeracityVCBehavior} — Y. Zhang, Das, R. Zhang (2024), *Verbosity ≠ Veracity: Demystify Verbosity Compensation Behavior of Large Language Models*, arXiv:2411.07858 v2, Dec 2024 (ACL 2025 UncertainLP).

**What they do.** Define **Verbosity Compensation (VC)** = LLM behavior of generating responses *that can be compressed without information loss when prompted to respond concisely*. Operationalized via a **length-based detector** \(V(x, y, r) = \mathbb{1}[\text{response is verbose}]\) — for their gold-answer-≤-4-words filtered samples, any longer response triggers VC. Five categories: repeating questions, enumerating, ambiguity, verbose details, verbose format. Then test:
1. **VC frequency across 14 LLMs × 5 datasets.**
2. **Performance gap** \(\Delta = \text{recall}_{\text{concise}} - \text{recall}_{\text{verbose}}\) and relative \(\delta = \Delta / \overline{\text{recall}}\).
3. **Uncertainty gap**: perplexity (open) / Laplacian-eigenvalue-sum (closed) of verbose vs concise splits.
4. **Cascade Model Selection (CaSel)** algorithm: route weak-to-strong, fall back when VC detected on weak model.

**Setup.** Datasets: Qasper, **LongBench (HotpotQA + MuSiQue + 2WikiMultihopQA bundled)**, NarrativeQA, NaturalQuestions-30 (lost-in-the-middle distractor), MMLU (modified). Filter to gold-answer-≤-4-words for clean VC detection. 14 LLMs (Gemma, Mistral, Llama, Gemini, GPT-4, Claude, etc.).

**Headline numbers.**
- **GPT-4 VC frequency = 50.40%** — even the strongest model is verbose half the time on hard QA.
- **Performance gap = 27.61% on Qasper** between verbose and concise responses (concise is much better).
- **Verbose responses exhibit higher uncertainty across all 5 datasets** — strong, consistent connection between VC and model uncertainty.
- **CaSel cascade**: Mistral VC frequency 63.81% → 16.16% on Qasper (weak-to-strong routing with VC fallback).

**Why it matters for us.**
1. **They use the same multi-hop QA datasets we use** (HotpotQA, MuSiQue, 2Wiki — bundled as their LongBench), making this the **single highest-overlap empirical paper** in our pool. **Cite this in §3 dataset description.**
2. **Their finding "verbose responses have higher uncertainty across all 5 datasets" is the direct multi-hop-QA equivalent of the length-confound finding** we replicate at scale on R1. **Direction-of-effect alignment is exact:** longer = more uncertain = more often wrong (their dimension) = more often retrievable-helpful (our dimension, where uncertainty drives retrieval need). Cite together with Janiak / Santilli / UNCERTAINTY-LINE in the §1 length-confound paragraph.
3. **Their performance-gap mechanism (verbose → wrong) is the *behavioral driver* of our length-confound observation.** Why does `tokens_generated` AUROC = 0.645 on Phi HotpotQA? Because verbose-when-uncertain is a real LLM behavior (Y. Zhang et al.); the trigger pool's signals get partial credit for tracking length, but length itself is the proxy variable for the verbose-when-uncertain behavior. **This gives us the *causal* story for §5 discussion: length is a proxy for verbosity-compensation, which is a proxy for uncertainty, which is a proxy for incorrectness/retrieval-need.** Powerful causal chain to write up.
4. **They use length as their VC detector but also point out that length-based detection is brittle.** Their CaSel uses length as the signal but its real value is in the cascade structure. **Our `tokens_generated` baseline is similar in spirit; our value-add is the residualization protocol that goes beyond length to find length-orthogonal signals (DRAGIN, AID).**
5. **CaSel is a model-cascade approach (route to bigger model when VC detected); our work is a retrieval-trigger approach (call retrieval when AID/DRAGIN trigger).** Both substitute external compute for the weak model's uncertainty. **Cite as a parallel-but-orthogonal mitigation of the same underlying phenomenon (verbose-when-uncertain). Future work: combine retrieval-trigger and model-cascade into a unified policy.**
6. **They restrict to gold-answer ≤ 4 words.** Our HotpotQA/2Wiki gold answers are typically short (1-5 words for entity-answer questions); the restriction roughly aligns. **Direct comparability of VC-detector mechanics is plausible.** If we wanted to add an explicit VC detector to our analyzer, applying their |response| > k threshold per cell would take 30 minutes of code.
7. **Their uncertainty measures are perplexity and graph-Laplacian eigenvalues** — neither is in our trigger pool directly, but `etc (Δ²H max)` and `seakr_k5_proxy.logdet` are conceptually adjacent. **Worth a comparison row: do their uncertainty signals cluster with ours under residualization?** Out of scope but pre-registerable for follow-up.

**Code-audit verdict.**
- Their length-based VC detector \(V(x, y, r) = \mathbb{1}[|r| > k \text{ given } |y| \le 4]\) is a 5-line addition to our analyzer. We can compute VC frequency per (arch × dataset × question-type) cell on R1 traces and report it. **Pre-registered actionable for §A appendix; cost = 30 min CPU.** This would give us a **direct tie-back to their behavioral framework.**
- We can NOT replicate their CaSel — it requires multi-model serving and the assumption that a stronger model is available. Future work.

**Honest caveats.**
- Their backbones are mostly proprietary (GPT-4, Gemini, Claude) plus a few open (Mistral, Gemma, Llama-2). Our pool is 100% open-weights (Phi-4, Qwen2.5, Gemma-4). Direct cell-level replication impossible.
- They use **recall** as the score function (not EM); we use EM (and F1 in side reports). Their numbers are not directly comparable to ours.
- They prompt the model **to be concise**; our prompts allow free-form CoT. Their VC measure assumes the conciseness instruction was honored or violated; ours doesn't have that anchor. **We should NOT directly report their VC frequencies as comparable to ours.**

---

## \cite{Zhang_AnswerAgreementRepresentationShaping} — J. Zhang, Guo, Y. Jiang, H. Wang, An, Du (2026), *Harnessing Reasoning Trajectories for Hallucination Detection via Answer-agreement Representation Shaping*, ICML 2026 (arXiv:2601.17467 v3, May 5 2026; NTU + Sichuan + Zhejiang).

**What they do.** Frame hallucination detection in Large Reasoning Models (LRMs) as a representation-shaping problem. Method:
1. Take the model's penultimate-layer hidden state \(h\) at the **trace boundary** (last token of reasoning trace).
2. Apply small **latent perturbations** \(\tilde h = h + \delta\) (noise injection).
3. Continue decoding from \(\tilde h\) to obtain a **counterfactual answer** \(\tilde a\).
4. Label the (a, \(\tilde a\)) pair via an agreement function \(\text{Agr}(a, \tilde a) \in \{0, 1\}\).
5. Train a **lightweight mapping** \(g_\phi\) on answer representations to **pull agreement pairs together, push disagreement pairs apart** (contrastive shaping).
6. Plug shaped embeddings into existing embedding-based detectors (Azaria & Mitchell, Burns CCS, Du subspace scoring, Chen INSIDE EigenScore).

**Theoretical claim (their Proposition 4.2).** Agreement-driven shaping yields representations whose stability under latent perturbation is informative about answer correctness — internally-stable answers are more likely truthful; fragile internal states (counterfactuals diverge under small perturbations) indicate hallucination risk.

**Setup.** Qwen3-8B on TruthfulQA. Code at `https://github.com/radiolab-ntu/ars_icml2026`. Layer-wise analysis (Figure 1): with-vs-without-trace probing AUROC across 32 layers — vanilla trace inclusion sometimes *hurts* probing performance (the motivation for ARS).

**Headline numbers.**
- ARS-shaped embeddings improve TruthfulQA detection AUROC by **+19.79%** vs vanilla LRM embeddings.
- Achieves **86.64% AUROC** state-of-the-art on TruthfulQA detection benchmarks.

**Why it matters for us — and the terminology-collision risk.**
1. **They use the term "answer stability" for what is mechanically VERY different from our `flip_rate`.** Their "stability" = robustness of decoded answer to small **latent perturbations** at the trace boundary. Our `flip_rate` / AID family = **frequency of step-to-step answer changes** under K=5 boundary-completion **sampling**. Both name the underlying construct "answer stability"; the operationalizations are completely orthogonal:
   - **ARS:** ONE trace, perturb hidden state at boundary, decode once → counterfactual answer; repeat with different \(\delta\) noise samples.
   - **Ours (AID):** ONE trace, K=5 stochastic completions per intermediate step → answer-set-per-step → step-to-step transitions.
   - **Datta:** ONE trace, append `</think>` + prefix at intermediate prefixes → single greedy answer per prefix → answer trajectory.
   These are three different intervention classes (latent / boundary-completion-sampling / forced-text-completion). **The agent's lit-precedent block already flagged this as "Zhang et al. ARS — terminological collision; cite to head off vocabulary-overlap critique." This lit review locks in the obligation.** Mandatory §2 paragraph: "We use 'answer instability' to denote step-to-step disagreement under K=5 boundary-completion sampling, distinct from the latent-perturbation-based notion of Zhang et al. (2026)."
2. **Their family is supervised representation-shaping; ours is parameter-free decode-time.** They train \(g_\phi\) (a learned MLP/projection); we use raw step-level transitions. **Categorically different families; both excluded from our trigger pool by the parameter-free RQ1 scope (`_rq1-plan.md` §"RQ1 statement"). Cite as "out-of-scope but conceptually adjacent."**
3. **Their Figure 1 finding "trace inclusion can HURT detection performance at some layers"** is consistent with our P0.6 finding that DRAGIN (which uses last-layer attention) collapses on Phi while AID (answer-level, layer-agnostic) survives. **Their interpretation: "trace can obscure answer-level signal."** Our interpretation: "different signal classes have different (arch × layer × trace-position) sensitivities." Compatible framings; cite together in §5.
4. **Their backbone is Qwen3-8B (a reasoning model).** We excluded Qwen3.6 (collapses under our matched-protocol greedy + non-thinking-mode). Their work uses thinking-mode operationally; not a comparable cell-level baseline.
5. **Their counterfactual-via-latent-perturbation has a relative we have NOT explored: capture the per-perturbation answer agreement rate as a free signal.** Concretely: at the EOS hidden state per step, inject K Gaussian noises \(\delta_k\), decode K answers, count agreement with the original. **This would be a 4-novel-trigger N5 capture in the rq1-pilot-v2 schema.** Pre-registered for future-work mention; out of scope for current campaign.
6. **TruthfulQA target is incorrectness/hallucination, not retrieval-helpfulness.** Their AUROC is on a different question than our `lift_em = +1`. Numbers not directly comparable.

**Code-audit verdict.**
- Their ARS framework is in PyTorch; we **cannot replicate** without (a) per-step trace-boundary penultimate-layer hidden state capture (we have only EOS hidden + sample-EOS-hidden), (b) a counterfactual-decoding pipeline, (c) contrastive training infrastructure, (d) the Qwen3-8B backbone we excluded. **Out of scope for current campaign.** Future-work future-work.
- Their public code (`radiolab-ntu/ars_icml2026`) is available; for a follow-up paper, integrating ARS-shaped representations as a *post-hoc* trigger candidate on our trace family is technically feasible. Defer.

**Honest caveats.**
- The terminology-collision risk is real and material. **A reviewer who reads "answer stability" in our paper without context will assume we mean ARS.** Mandatory disambiguation.
- Their reported AUROC of 86.64% is on TruthfulQA single-hop; our pool is multi-hop QA with paired oracle. The +19.79% improvement number is on a fundamentally different task.
- Their method requires *training* the lightweight mapping; ours requires no training at any stage. Their detection-quality ceiling is therefore higher (trained probes dominate parameter-free signals on the same task; consistent with Lu et al. 2026 discussion above).

---

# Section 2 — Topic-linked synthesis (how each paper supports / refutes / contextualizes our experiments)

## §2.1 Step-level answer-trajectory primitives — the AID family lineage

\cite{Datta_DecideEarlyExplainLater} states that **AnswerSwitches**(T) \(= \sum_i \mathbb{1}[A_i \ne A_{i+1}]\) and the **final answer-switch index** \(t^\star = \max\{i : A_i \ne A_{i+1}\}\) computed over **forced-answer-completion** intermediate predictions are direct, mechanical analogues of our `flip_count` and `late_flips`. They report 32% of Qwen3-4B traces ever change answer; 760 tokens generated after the final switch; probe-based early stopping saves ~500 tokens at 2% accuracy cost. **Our extension is the K=5 stochastic boundary completion (vs their K=1 forced greedy) and the application target (retrieval triggering vs early stopping).** Our A1 ablation on Phi+Qwen R1 (n=800 paired) shows **regime 2** (K=5 ≈ K=1, CIs straddle 0 on `lift_em=+1`); the K-value is empirically free, not a contribution. **Re-frame our methodology contribution as the residualized + paired-oracle + retrieval-trigger application of the Datta primitive.**

\cite{Wang_ChainOfProbeCoTStepByStep} states that *Chain-of-Probe* — appending the probing string `So, the answer is (` after each reasoning step and reading the softmax over a closed answer set \(V = \{A,B,C,D\}\) — finds the same phenomenon Datta later formalized: many traces commit early, and the early-commit accuracy is **>20% higher than the change-during-reasoning accuracy.** This is **the cleanest external concordance with our P0.6 mechanism finding**: no-flip traces 61% correct / 3% helped; late-flip 32% correct / 18% helped; high-flip 20% correct / 20% helped. The mechanism (answer-changing during CoT correlates with both lower correctness AND higher retrieval-helpfulness) is now established by two independent groups. Use this as the §1 motivation paragraph.

\cite{Chen_DynamicInstabilityLLMReasoning} states that **timing matters more than magnitude**: the relative peak position \(\rho = t^\star / T\) of an instability signal \(I_t = \text{JSD}(\tilde p_t, \tilde p_{t-1}) + \lambda H_t\) discriminates corrective (early, \(\rho < 0.25\), 46% accuracy) from destructive (late, \(\rho > 0.5\), 14% accuracy) instability — a >3× accuracy gap conditional on timing alone, at S strength similar. **This directly justifies our `late_flips` AID variant** and explains why on Phi 2Wiki/comparison the raw `late_flips` AUROC is **0.950**, the strongest cell-level finding in our R1 campaign. Different signal class (token-distribution JSD vs answer-set transitions); identical timing logic.

\cite{Zhang_AnswerAgreementRepresentationShaping} states that **"answer stability" via small latent perturbations at the trace boundary** discriminates truthful from hallucinated answers, achieving 86.64% AUROC on TruthfulQA after representation shaping. **This is a TERMINOLOGICAL COLLISION risk for our paper.** Their "stability" = robustness to latent noise injection at boundary; ours = step-to-step agreement under K=5 boundary-completion sampling. Mandatory disambiguation in §2 of the paper: clearly distinguish "Answer Instability Detection (AID, ours, decode-time, training-free)" from "Answer-agreement Representation Shaping (Zhang et al. 2026, supervised, latent-perturbation-based)."

\cite{Datta_DecideEarlyExplainLater}, \cite{Wang_ChainOfProbeCoTStepByStep}, \cite{Chen_DynamicInstabilityLLMReasoning}, \cite{Zhang_AnswerAgreementRepresentationShaping} together establish that **the answer-trajectory family is now an active, multi-paper research front (Wang NAACL 2025 → Chen Feb 2026 → Datta Apr 2026 → J. Zhang ICML 2026)**, and our paper's contribution is honestly positioned as: **(i)** the first application to retrieval triggering with paired-oracle `lift_em` target, **(ii)** under length-residualization protocol, **(iii)** with K-value empirically resolved across 3 architectures.

---

## §2.2 Length / verbosity as a confound in UQ evaluation — the residualization-protocol lineage

\cite{Santilli_RevisitingUQEvaluationLengthBias} states that mutual length-bias on both UQ scores and correctness functions **systematically skews AUROC rankings** (their App. C formal proof, Scenario ii). Empirical: Spearman \(\rho\) of token-length vs Perplexity = −0.7, vs Naive Entropy = −0.6, vs Mean Token Entropy = −0.6, vs ROUGE-L (F1) at t=0.5 = +0.7. Token Length itself is a competitive UQ-method baseline on their leaderboard. **This supplies the formal proof underlying our entire residualization protocol.**

\cite{Vashurin_UncertaintyLineLengthInvariant} states that even nominally length-normalized UQ measures (Perplexity, MTE, MCNSE, Token-SAR) exhibit residual length bias, and proposes **UNCERTAINTY-LINE: \(\hat u_{\text{LINE}} = \hat u - \hat f(L)\) via OLS regression on length, residuals as length-invariant estimates**, post-hoc and model-agnostic. **This IS our residualization protocol.** Our `_residualize_signal(signal, log_tokens)` in `analysis/length_audit_v2.py` and downstream R1 analyzers is the unsupervised UNCERTAINTY-LINE variant. Our paired-bootstrap-vs-tokens test corresponds to their supervised variant. **We must credit Vashurin et al. as the originator and re-frame our methodology contribution as "we adopt and extend the UNCERTAINTY-LINE protocol of Vashurin et al. (2025) to dynamic-RAG triggering with the `lift_em = +1` paired-oracle target." Replicate via their public code (`github.com/stat-ml/uncertainty-line`) on R1 traces in the appendix.**

\cite{Janiak_IllusionProgressHallucinationDetection} states **directly** that *"simple heuristics based on response length can rival complex hallucination detection techniques"* — a 45.9% Perplexity drop and 30.4% Eigenscore drop when switching ROUGE → LLM-as-Judge labels, and length-based heuristics matching or surpassing Semantic Entropy on 3 single-hop QA benchmarks. **This is the dual external authority for our R1 D1 closed_em finding** that `tokens_generated` AUROC = 0.645 beats every internal signal raw and residualized on Phi HotpotQA. Cite together with Santilli + Vashurin as the §1 length-confound triplet.

\cite{Zhang_VerbosityVeracityVCBehavior} states that **verbosity compensation is pervasive** (GPT-4 50.4% VC frequency) and **verbose responses exhibit higher uncertainty across all 5 datasets** including HotpotQA / MuSiQue / 2Wiki bundled as their LongBench. Verbose-vs-concise performance gap = 27.61% on Qasper. **This supplies the *behavioral causal story*** for §5 discussion: length is a proxy for verbosity-compensation, which is a proxy for uncertainty, which is a proxy for incorrectness/retrieval-need. Powerful causal chain.

\cite{Ielanskyi_PitfallsUQEvaluationNLG} states that approximate correctness functions (ROUGE-1/2/L, BLEU, BERTScore, BLEURT, LLM-as-judge variants) **disagree substantially** on UQ method ranking, with pairwise mutual AUROC as low as 0.53 and Spearman ρ between rankings sometimes negative (BLEU vs ROUGE on COQA: ρ = −0.66). Recommends **marginalizing over LLM-as-a-judge variants**, structured/OOD/perturbation tasks, Elo aggregation. **Defends our `lift_em` paired-design choice** (same EM correctness function on both arms cancels mutual bias). **Pre-empt by reporting EM, F1, and containment-match rows in an appendix table; cheap insurance.**

\cite{Mao_ConfidenceOverTimeSTL} acknowledges length bias explicitly in §2 ("joint generation probability suffers from severe length bias, making it unsuitable for long-form reasoning") and cites Li et al. 2025b. They use length-normalized Perplexity but **do not propose a residualization fix.** This is the exact gap UNCERTAINTY-LINE / our protocol fill.

**Synthesis for our §1 framing paragraph:**
> "Length confounds are the dominant single phenomenon in UQ and hallucination-detection evaluation on QA tasks (\cite{Santilli_RevisitingUQEvaluationLengthBias}, \cite{Janiak_IllusionProgressHallucinationDetection}, \cite{Zhang_VerbosityVeracityVCBehavior}, \cite{Mao_ConfidenceOverTimeSTL}, \cite{Ielanskyi_PitfallsUQEvaluationNLG}). \cite{Vashurin_UncertaintyLineLengthInvariant} propose UNCERTAINTY-LINE — regression-residualization on length — as a post-hoc fix on NMT, summarization, and math reasoning. **We adopt and extend UNCERTAINTY-LINE to dynamic-RAG triggering on multi-hop QA paired with oracle re-decode**, and identify the two signals that survive the length-residualization on at least one of three architectures (DRAGIN-RIND, the AID family)."

---

## §2.3 Stepwise / streaming uncertainty in long-form CoT reasoning

\cite{Lu_StreamingHallucinationDetectionCoT} states that **hallucination in long-CoT is an evolving latent state**; trains a step-level + prefix-level probe pair on hidden states with Claude-4.5 labels, achieving **86–93% AUROC** on Llama-3.1-8B / Qwen2.5-7B / DeepSeek-R1-Distill-8B. Step-level performance drops along the trace (AUC 0.89 → 0.82 across early/mid/late thirds); F1 increases (0.35 → 0.78) due to prevalence shift. **Out-of-family for our parameter-free RQ1 scope but conceptually parallel** — establishes the trained-probe upper bound on the same backbone (Qwen2.5-7B). Our parameter-free signals (AID family resid AUROC 0.605–0.690 on Qwen `lift_em`) sit ~0.20 AUROC below their *trained* upper bound, on the orthogonal `lift_em` target.

\cite{Mao_ConfidenceOverTimeSTL} states that stepwise confidence trajectories admit **discriminative STL formula mining**: \(\Box_{[a,b]}(s_t \ge \mu)\) and similar give AUROC ~0.61–0.62 across BBH / CLadder / SciQ / GAOKAO-Math on Qwen3-8B / Gemma-3-12B / Llama-3-8B. **Failure-mode patterns transfer cross-task; correct-mode patterns are task-specific** — directly consistent with our per-cell heterogeneity finding (`AID late_flips` 2Wiki/comparison 0.950 vs HotpotQA/comparison 0.250).

\cite{Datta_DecideEarlyExplainLater} provides an explicit **"streaming"-compatible operationalization** via hold-for-\(k\) causal smoothing of the answer trajectory (Eq. 4) — the smoothed trajectory depends only on current+previous steps, enabling on-the-fly monitoring. **This is a drop-in additional AID variant we should add to the analyzer**: `flip_rate_hold3` from hold-for-3 smoothing. Cost ~1 hour CPU; pre-registered actionable.

\cite{Lu_StreamingHallucinationDetectionCoT}, \cite{Mao_ConfidenceOverTimeSTL}, \cite{Datta_DecideEarlyExplainLater}, \cite{Chen_DynamicInstabilityLLMReasoning} establish the broad consensus that **stepwise / streaming / temporal modeling of LLM reasoning uncertainty is the field's center of gravity in 2025–2026.** Our paper is **part of this wave** in the parameter-free trigger sub-family.

---

## §2.4 Multi-sample answer-agreement vs single-sample answer probing — the K=5 vs K=1 question

\cite{Datta_DecideEarlyExplainLater} uses K=1 sampling at each forced-answer-completion probe. \cite{Wang_ChainOfProbeCoTStepByStep} uses a single greedy forward pass with closed-answer-set softmax probing. Our K=5 stochastic boundary-completion is the **multi-sample extension**. **Our A1 ablation result on R1 Phi (n=105) and R1 Qwen (n=305) is regime 2: K=5 ≈ K=1**, point estimates near zero, CI straddles 0 on `lift_em=+1` and on `closed_em=0`. **Empirically the K=5 multi-sample contribution is null at headline-grade power**; we recommend K=1 for compute-efficiency reasons. Frees a third of the §4 word budget; defensible and honest.

\cite{Mao_ConfidenceOverTimeSTL} uses single-sample inference for their main method (only self-consistency baselines draw multiple samples). \cite{Chen_DynamicInstabilityLLMReasoning} is single-trace by design (trajectory diagnostic). \cite{Janiak_IllusionProgressHallucinationDetection} uses **n=10** samples at temp 1.0 for Semantic Entropy and similar consistency methods. \cite{Lu_StreamingHallucinationDetectionCoT} uses single-trace probes. **The field is split**: trajectory-based methods (us, Chen, Datta, Wang, Mao, Lu) trend toward K=1 / single-sample because trajectory itself provides temporal information; consistency-based methods (Semantic Entropy, SAC³, SelfCheckGPT) need K≥5. **Our finding that K=5 ≈ K=1 in the AID family is consistent with this split**: AID is a trajectory method, not a consistency method, and inherits the single-sample sufficiency of the trajectory family.

---

## §2.5 Hallucination-detection benchmarking and evaluation pitfalls — what reviewers will demand

\cite{Janiak_IllusionProgressHallucinationDetection} demands **LLM-as-Judge labels** in addition to ROUGE/EM. **Pre-registered actionable for our submission**: add an LLM-as-Judge label-comparison row in an appendix table on a 200-trace subset. Cost ~30 min CPU + small GPT-4o-Mini API spend.

\cite{Ielanskyi_PitfallsUQEvaluationNLG} demands **marginalization over multiple correctness functions**, structured/OOD/perturbation tasks, and Elo aggregation. **Our `lift_em` paired-design** (shared correctness function on both arms) is partially insulated; report EM + F1 + containment-match rows in the appendix to fully address.

\cite{Santilli_RevisitingUQEvaluationLengthBias} demands explicit reporting of length-vs-signal Spearman \(\rho\) per cell. **We already have this** as a side output of `length_audit_v2.py` and `r1_phi_paired_audit.py`. **Cost = 0; just need to format the reviewer-prep table.**

\cite{Zhang_VerbosityVeracityVCBehavior} would demand a **VC frequency report per (arch × dataset)** in our setting. The detector \(V(x, y, r) = \mathbb{1}[|r| > k \text{ given } |y| \le 4]\) is a 5-line addition to our analyzer. Cost ~30 min CPU; pre-registered for the §A appendix.

\cite{Vashurin_UncertaintyLineLengthInvariant} would demand a **head-to-head replication of our `_residualize_signal` against their reference implementation.** Pull `github.com/stat-ml/uncertainty-line`, run on R1 traces with our 16+4 trigger signals, report AUROC delta in the appendix. Cost ~2-3 hours integration + 30 min CPU. **Massively reviewer-defense-positive.**

---

## §2.6 Representation-level / hidden-state trajectory signals — the family we adjoin but do NOT join

\cite{Lu_StreamingHallucinationDetectionCoT} trains step-level + prefix-level probes on hidden states with Claude-4.5 labels (supervised). \cite{Zhang_AnswerAgreementRepresentationShaping} trains a contrastive lightweight mapping \(g_\phi\) on counterfactual-perturbed embeddings (supervised). **Both are excluded from our trigger pool** by the parameter-free RQ1 scope (`_rq1-plan.md` §"RQ1 statement"), alongside DioR, CtrlA, MBA-RAG, Self-RAG.

Our `dragin_rind_max.max`, `drift_cosine.max`, `variance_l2_window.max`, `hidden_norm.range`, `seakr_k5_proxy.logdet` family is the **parameter-free hidden-state trajectory cousin** of these supervised methods. Performance comparison: \cite{Lu_StreamingHallucinationDetectionCoT} reports trained-probe AUROC 0.867 on Qwen2.5-7B; our `dragin_rind_max.max` resid AUROC on Qwen pooled = 0.605 (paired-significantly beats `tokens_generated`, CI [+0.003, +0.168]). The 0.26 AUROC gap is the **price of parameter-free decoding**: trained probes dominate, but at the cost of arch-specific labeled data + training infrastructure + lack of cross-arch portability. **Our value-add is portability + zero training cost; their value-add is absolute detection performance.**

\cite{Chen_DynamicInstabilityLLMReasoning} provides the **closest parameter-free probing analog**: top-k token distributions (no hidden states needed). They do not include their JSD signal in the trigger-pool comparison; we cannot directly compare AUROC. Conceptual closeness: their \(D_t = \text{JSD}(\tilde p_t, \tilde p_{t-1})\) is the per-token-distribution version of our `etc (Δ²H max)` (per-step entropy second-difference). Both target distributional dynamics; they at token granularity, ours at step granularity.

---

## §2.7 Code-audit / replication priorities

| Priority | Action | Cost | Defensive value |
|---|---|---:|---|
| **P1** | Run UNCERTAINTY-LINE reference implementation on R1 traces, replicate `_residualize_signal` AUROCs head-to-head | 3 h | Massively positive — settles "is your residualization implemented correctly" |
| **P1** | Add `flip_rate_hold3` AID variant (Datta hold-for-3 smoothing) to analyzer; report on R1 Phi + Qwen + Gemma | 1 h | High — directly addresses the K-value-vs-Datta critique |
| **P1** | Add VC frequency per (arch × dataset × question-type) cell from \cite{Zhang_VerbosityVeracityVCBehavior} detector | 30 min | High — ties to verbosity-uncertainty causal story |
| **P1** | Report length-vs-signal Spearman \(\rho\) table per cell (already computed) | 0 (formatting only) | Mandatory per Santilli |
| **P2** | LLM-as-Judge label-comparison appendix on 200-trace subset | 30 min CPU + GPT-4o-Mini API | High — addresses Janiak / Ielanskyi |
| **P2** | Add EM / F1 / containment-match rows to correctness function comparison appendix | 1 h | Medium — addresses Ielanskyi |
| **P2** | Fraction of "decided-early" traces (\(t^\star \le 1\)) per cell; ties to Datta's 32%-of-traces-ever-change finding | 30 min | Medium — disarms the Phi sparse-flip criticism |
| **P3** | Add Mao stepwise-confidence aggregations (`mao_step_conf_min`, `mao_step_conf_decline_slope`, `mao_step_conf_late_low_count`) as free-signal trigger candidates | 30 min | Low — adds a comparison row in §A |
| **P3** | Add Wang CoP-Tree-style aggregation features (max conf change, monotonicity) | 1 h | Low — broadens the trigger pool comparison |
| **P4 / future work** | Run N2 `prefix_answer_now` capture (Datta-faithful single-sample forced-completion) on R1 Gemma cell | 1.5 h compute | Medium — only if reviewers demand byte-faithful Datta replication |
| **P4 / future work** | Replicate Chen et al. JSD signal via re-decode with top-k=50 logging; compare to AID late_flips on same cohort | 4-6 h compute | Low — out of scope for current campaign |

**Combined cost of all P1 + P2 actionables: ~8 hours including writing.** This is the cheapest single block of work that materially raises EMNLP-short acceptance probability per `_rq1-outlook.md` decomposition.

---

## §2.8 Attribution corrections required before submission

`_rq1-agent.md` and `_rq1-plan.md` contain three misattributions discovered during this lit review pass. Fix before submission:

1. **arXiv:2510.02279** — agent cites as "Castaneda et al. 2025"; actual authors are **Ielanskyi, Schweighofer, Aichberger, Hochreiter** (ELLIS Linz / NXAI / JKU).
2. **arXiv:2504.13677** — agent cites as "Belem et al. (ACL 2025 short)"; actual authors are **Santilli, Goliński, Kirchhof, Danieli, Blaas, Xiong, Zappella, Williamson** (Apple + Sapienza + NUS).
3. **arXiv:2505.19060** — agent cites as "Belem et al. UNCERTAINTY-LINE"; actual authors are **Vashurin, Goloburda, Nakov, Panov** (MBZUAI).

There appears to be no "Belem et al." paper in our pool at all. The agent's lit-precedent search may have mis-resolved an autocomplete or search-result snippet. **Audit the rest of the agent's citations in `_rq1-plan.md` §"Lit precedent / positioning — LENGTH-CONFOUND lineage" against the actual arXiv records before any draft submission.**

