# `final_auditNotes.md` — Palette Cleanser

> **Purpose.** Before the top-down audit of `template.tex`, lay out what is *good* about this paper, what its load-bearing strengths are, and how academic norms in the surrounding literature map to those strengths. The goal is to enter the audit with a clear sense of what NOT to weaken — so when the audit finds inconsistencies, the fixes preserve the paper's actual contribution rather than over-reading the literature.
>
> **Structure.**
> 1. Story spine and load-bearing strengths
> 2. What the literature actually does — and what it does NOT do
> 3. Where this paper *defensibly* extends prior work
> 4. Audit posture: what to keep robust, what to soften

---

## 1. Story spine and load-bearing strengths

### 1.1 Story (one-line)
*Length confounds AUROC for dynamic-RAG triggers; we run the first length-controlled audit on a paired-oracle target, find that an answer-trajectory primitive (FCT) is the strongest length-robust trigger, and show this survives an LLM-judge robustness check.*

### 1.2 Load-bearing strengths

These are the parts that survive any honest scrutiny and should *not* be weakened by audit edits:

**S1. The paired-oracle `lift_em` target is mechanism-clean.**
Most published trigger evaluations score a signal against `closed_em=0` ("which traces are wrong?") — but length predicts wrongness moderately well on multi-hop QA, so a signal can earn AUROC by tracking length. The paper instead asks `lift_em=+1`: "which closed-book wrong traces become right under gold-context?" — i.e., which traces are *retrieval-helpful*. This is exactly the question a deployed dynamic-RAG trigger is asked to answer, and the paired EM equality across both arms partially insulates against the correctness-function-bias critique of Ielanskyi 2025. Mallen 2023 introduced the conceptual parent (when does retrieval help a parametric model?), and `lift_em=+1` operationalizes it cleanly.

**S2. Length-residualized AUROC is an established protocol that is *novel as applied to dynamic-RAG triggers*.**
Vashurin et al.'s Uncertainty-Line is a real published method; this paper reuses its math and pre-empts the reimplementation critique with App. D's `ε_protocol ≤ 0.029` agreement check. The honest claim is: the *math* is established; the *application domain* (dynamic-RAG triggers, multi-hop QA) is new. This is a defensible contribution shape.

**S3. FCT is a real new primitive with the largest effect size.**
`Δ = +0.190` length-residualized, Bonferroni LB `+0.047`, raw AUROC `0.784` on the held-out Gemma×2Wiki cell. Forward-looking variant of Datta 2026's `t*` (which is backward-looking, intended for early-stopping). Causal direction matters: stop vs. retrieve are dual operations on the same primitive. This is the paper's clearest novel contribution and the headline result.

**S4. The held-out 2Wiki inversion is a strong sanity check.**
On `lift_em=+1`, raw `tokens_generated` AUROC is 0.457 (Qwen×2Wiki) and 0.518 (Gemma×2Wiki) — at or below chance. The verbosity-compensation pattern (Zhang Verbosity 2024) inverts on the held-out cells. This is empirical evidence that length-residualization isn't just adding noise — it's removing real predictive content that doesn't generalize.

**S5. LLM-judge robustness check directly addresses Janiak's critique.**
Janiak documents that switching ROUGE → LLM-as-Judge collapses Perplexity AUROC by ~46%. The paper checks this exact failure mode on its own headline pool: 8/9 paired-significant survivors preserved under GPT-4o-Mini judge, FCT *strengthens* to `+0.222`, all 4 Bonferroni-robust signals retain nominal-95% under the judge. The judge-vs-EM `κ = 0.707` is reported. This is a methodologically sharp pre-emption.

**S6. K=1 vs K=5 ablation is a deployment-relevant honesty move.**
Paper recommends K=1 because K=5 is null at this power. This is unusual self-deprecation for a paper introducing a K=5-flavored signal (FCT itself is K=5 majority). It builds reviewer trust.

**S7. Bonferroni honesty.**
4 signals survive at k=15 with positive lower bounds. Two are knife-edge (`ent.max` LB `+0.001`, `min_majority` LB `+0.004`); the new Limitations paragraph documents this transparently. FCT (`+0.047`) and DRAGIN-RIND (`+0.039`) are robust.

**S8. Pre-empts every standard reviewer attack.**
- "You missed Uncertainty-Line" → App. D ε-replication.
- "You missed length confound" → §3.2 OLS residualization, App. F per-cell Spearman.
- "You missed correctness-function bias" → §4.4 LLM-judge robustness.
- "You did multiple comparisons" → Bonferroni at k=15.
- "Your K=5 inflates compute" → §4.5 K=1 ablation.
- "Datta-derived, not novel" → App. E operational-axes table.

### 1.3 What is *not* the story

The paper is **not** claiming:
- That answer-trajectory is a new family in the literature (Datta, Wang, Chen, Lu, Zhang ARS predate or are concurrent).
- That FCT's primitive is unrelated to Datta's `t*` (App. E spells out the relationship).
- That length-residualization is a new method (Vashurin's contribution; ours is the application).
- That dynamic-RAG triggering is a new task (DRAGIN, FLARE, SeaKR, etc. are the pool).
- That EM is the right correctness function in general (LLM-judge check + Limitations explicitly walk this back).

The actual contribution shape is **integration**: residualization + paired-oracle + answer-trajectory family + FCT as a new variant within the family. Each piece is incremental over its parent; the integration is what's load-bearing.

---

## 2. What the literature actually does — and what it does NOT do

This section grounds each citation against the source paper. Items marked **CHECK** are places where the audited paper's framing is verified; items marked **REFINE** are places where the framing is technically true but should be tightened to avoid over-reading the source.

### 2.1 Vashurin et al. 2025 (Uncertainty-Line)

**What they actually do.**
- OLS residualization on **raw** token count `|y|` (Eq. 1: `û(y) = α|y| + β`), **not log-tokens**. This is mathematically equivalent in rank for monotonic single-cohort fits, and the paper's `ε_protocol ≤ 0.029` check confirms this.
- Train/test split (no ratio specified in paper; reference code uses 70/30). Paper says "5 seeds"; the audited paper says "5 seeds" for averaging — the local code (`uncertainty_line_audit_*.out`) uses 3 seeds.
- Domains: **NMT (WMT14/19), XSum (summarization), GSM8K (math reasoning)**. Evaluation metric: **Prediction-Rejection Ratio (PRR)** — selective prediction, not retrieval triggering.
- App. D.1: linear fit is preferred over polynomial; "do not provide any significant or consistent improvement over the linear fit" — **CHECK**, this is exactly what the audited paper §3.2 cites.
- Honest caveat from Vashurin: *"in tasks such as multi-step reasoning, where uncertainty may follow phase-specific patterns, a linear fit may be insufficient"* — relevant for the audited paper's multi-hop QA application; should optionally be mentioned in Limitations or App. F.

**Implication for the audit.**
- The paper's claim "dynamic-RAG triggers remain an untested application domain" — **CHECK**, Vashurin does not test on RAG at all.
- The paper's `ε_protocol ≤ 0.029` claim — **CHECK** against `uncertainty_line_audit_*.out`: Qwen `ε_max=0.0287`, Gemma `ε_max=0.0270`. Numbers match.
- The paper's `ε_sampling_max = 0.062 (Qwen), 0.070 (Gemma)` — **MISMATCH** with code: Qwen `ε_max=0.1113`, Gemma `ε_max=0.1063`. Either re-run the audit at 5 seeds (paper's claim) or update to the 3-seed numbers from code, and update the caption.
- The paper's "n_seeds = 5" — **MISMATCH** with code (`n_seeds: 3`). Update one or the other.

### 2.2 Santilli et al. 2025 (length confounds AUROC)

**What they actually do.**
- 4 datasets (TriviaQA, SQuAD, NQ-Open, SimpleQA) × 4 models × 8 UQ × 7 correctness functions; **single-hop QA, not multi-hop**.
- Headline finding: changing the correctness function reshuffles UQ rankings; length-based baselines rank competitively under lexical/embedding judges but last under LM-as-judge.
- **Spearman ρ = −0.7 between length and Perplexity** — verified in Figure 4 (pooled correlation matrix). Paper's claim is correct.
- Length-normalized Semantic Entropy: ρ = −0.7; non-normalized: ρ = −0.3 (only relevant if SE ever cited; not in audited paper).

**Implication for the audit.**
- The paper's `\citet{santilli_revisiting_2025}` for "Spearman ρ = −0.7 between length and Perplexity" — **CHECK**, accurate.
- Note: Santilli is single-hop QA; our application is multi-hop. The audited paper's framing "adjacent QA selective-prediction and UQ tasks" is correct (adjacent, not identical).

### 2.3 Janiak et al. (LLM-as-judge, Illusion of Progress)

**What they actually do.**
- Datasets: NQ-Open, TriviaQA, SQuAD v2 nocontext. Models: Llama-3.1-8B, Mistral-7B-v0.3. Judge: GPT-4o-Mini. **Single-hop QA**.
- Headline: detector AUROC drops up to 45.9% switching ROUGE → LLM-as-Judge.
- Paper's "45.9% Perplexity, 30.4% EigenScore" — **CHECK**, exactly matches Janiak Table 2 Mistral × NQ-Open zero-shot.
- **REFINE**: these are *cherry-picked worst-case* numbers (Mistral × NQ-Open zero-shot). LLAMA on the same dataset shows Perplexity drop = −1.2%, EigenScore = −19.0%. The audited paper currently cites the worst-case as if representative. Consider one-line softening (e.g., "up to 45.9% on certain Mistral cells").

**Implication for the audit.**
- The audited paper's `\kappa = 0.707` → "substantial agreement" frame, citing Landis 1977 — Landis-Koch land-mark thresholds are: 0.41–0.60 moderate; 0.61–0.80 substantial; 0.81+ almost perfect. ✓
- Janiak uses **GPT-4o-Mini** judge → audited paper uses GPT-4o-Mini; **same judge model**, this is "Janiak's exact choice" as the paper claims. **CHECK**.

### 2.4 Zhang Verbosity 2024 (verbosity compensation)

**What they actually do.**
- 5 benchmarks: Qasper, **LongBench (HotpotQA + MuSiQue + 2WikiMultiHopQA bundled, 449 samples)**, NarrativeQA, NQ30, MMLU.
- Mechanism: Verbosity Compensation (VC) — verbose-when-uncertain. Δ recall (concise vs verbose) up to 27.61% (Qasper, llama3-70b).
- Multi-hop coverage is **only as a LongBench bundle**, not standalone benchmarks.

**Implication for the audit.**
- The paper's framing (§5.1): "verbose-when-uncertain is a real LLM behavior on the HotpotQA, MuSiQue, and 2WikiMHQA family" — **REFINE**: technically true (LongBench bundles them) but Zhang doesn't run them as standalone benchmarks. The current phrasing is close enough; consider softening to "the multi-hop QA family represented in LongBench" if a strict reviewer reads.
- The paper's Figure 1 / Abstract claim "inverting the verbosity-compensation pattern observed on single-hop QA" — Zhang does NOT explicitly contrast multi-hop vs single-hop on this axis. The "single-hop QA" framing here is supported by Zhang's NQ30 cell and TriviaQA-style claim, but Zhang's comparison is concise-vs-verbose, not single-vs-multi-hop. **REFINE**: phrase as "the verbosity-compensation pattern documented on QA more broadly" rather than asserting Zhang showed it specifically on single-hop.

### 2.5 Mao 2026 (Confidence over Time)

**What they actually do.**
- Datasets: GAOKAO-Math, CLadder, SciQ, BBH — **structured MCQ/binary reasoning, not multi-hop QA**.
- Quote: *"This quantity suffers from severe length bias, assigning smaller values to longer responses regardless of correctness"* — **CHECK**, verbatim verifiable.

**Implication.** Citation is correct. Could optionally note Mao's domain is math/MCQ, not RAG, in §2.

### 2.6 Ielanskyi 2025 (correctness-function bias)

**What they actually do.**
- CoQA, TriviaQA, SQuAD, BigCodeBench, COLLIE.
- Spearman ρ disagreement up to −0.66 (BLEU vs ROUGE on CoQA); mutual AUROC down to 0.53.
- Sequence-length top-3 frequency: 0.312 → 0.562 under adversarial correctness-function selection.

**Implication.** Citation is correct. The paper's framing "approximate correctness functions disagree on UQ rankings" matches Ielanskyi's claim verbatim.

### 2.7 Answer-trajectory family (Datta, Wang, Chen, Lu, Zhang ARS)

**What they actually do.**
- **Datta 2026** (`arXiv:2604.22266`, ICML 2026): forced-answer-completion, K=1 sampling on thinking-mode reasoning models with `</think>` tokens. Primitives: `t* = max{i : A_i ≠ A_{i+1}}` (last switch index, backward-looking) and `AnswerSwitches(T) = Σ [A_i ≠ A_{i+1}]`. Application: **early stopping**, not retrieval triggering.
- **Wang Chain-of-Probe** (NAACL Findings 2025): step-boundary confidence-probing. Application: **CoT necessity / accuracy**, not RAG.
- **Chen 2026** (`arXiv:2602.02863`): inference-time instability `I_t = D_t + λH_t`, where `D_t = JSD(p̃_t, p̃_{t-1})` is on **next-token distributions** (not answer distributions). Application: **failure diagnosis**, not RAG.
- **Lu 2026** (`arXiv:2601.02170`): cumulative prefix-level hallucination state. Application: **streaming hallucination detection**, not RAG.
- **Zhang ARS 2026** (`arXiv:2601.17467`): answer-agreement representation shaping via latent perturbations. Application: **hallucination detection**, not RAG.

**Implication for the audit.**
- The paper's "first length-controlled audit of the published training-free dynamic-RAG trigger pool" — **CHECK**: none of the answer-trajectory papers above touch dynamic-RAG.
- The paper's CABV definition cites Chen 2026 ("conceptually closest to the inference-time instability signal of Chen") — **REFINE**: Chen's JSD is on **token distributions**, CABV's JSD is on **K=5 answer distributions**. The paper's clause "(which uses JSD over token distributions rather than over K=5 answer distributions)" already correctly disambiguates.
- The paper's "emerging answer-trajectory family" framing for `\citep{datta_large_2026, wang_chain--probe_nodate, chen_i_2026}` — **CHECK**: this is the right citation set; Lu and Zhang ARS could be added to broaden the family if word-budget permits.

### 2.8 Trigger-pool peer methods (DRAGIN, FLARE, SeaKR, INSIDE, RSUS, SELFDOUBT, ETC)

**Verified out-of-band.**
- **DRAGIN (Su 2024)** — RIND = entropy × max-attention × stopword-filter, byte-faithful.
- **SeaKR (Yao ACL 2025)** — regularized Gram determinant of K=20 EOS hidden states from middle layer; our `seakr_k5.logdet` is K=5 same primitive.
- **INSIDE (Chen ICLR 2024)** — EigenScore on centered sample covariance; K nonzero eigenvalues are rank-equivalent to Gram log-det up to centering.
- **SELFDOUBT (Pandey 2026, arXiv:2604.06389)** — `HVR = h/(v+1)`, single-trace, Tier-1 high-precision gate at HVR=0 (96.1% accuracy at 25.4% coverage).
- **FLARE (Jiang 2023)** — online per-token trigger `max p_t < τ`.
- **ETC (Li 2026)** — token-level entropy second-difference.
- **RSUS (Guo 2026)** — multi-sample step-boundary continuation consistency.

All citation framings in §1 ¶1 / §2 are verified against either the local PDF (where available) or web search.

---

## 3. Where this paper *defensibly* extends prior work

The contribution shape is **integration**, with FCT as the headline primitive:

| Component | Prior art | What this paper adds |
|---|---|---|
| **Length residualization** | Vashurin 2025 (NMT/summarization/math) | First application to dynamic-RAG triggers; ε-replication audit pre-empts the reimplementation critique |
| **Paired-oracle target** | Mallen 2023 (when does retrieval help) | Operationalizes Mallen as `lift_em=+1`; paired-EM-equality answers Ielanskyi's correctness-function critique |
| **Answer-trajectory family** | Datta 2026 (early-stopping), Wang 2024 (CoT), Chen 2026 (failure-diagnosis), Lu 2026 (hallucination), Zhang ARS 2026 | First audit of these primitives as dynamic-RAG triggers under length control |
| **FCT** | Datta's `t*` (backward, K=1, early-stopping) | Forward-looking variant, K=5 majority, retrieval-triggering target, non-thinking-mode chat models |
| **CABV** | Chen 2026's instability `I_t = D_t + λH_t` (token-distribution JSD) | Answer-distribution analogue (K=5 answer JSD), defined in §3.3 |
| **LLM-judge robustness** | Janiak 2025 (single-hop QA, ROUGE→GPT-4o-Mini) | Same judge model on multi-hop QA; addresses Janiak's worst case directly |

**The "first" claims that survive scrutiny:**
- "First length-controlled audit of the published training-free dynamic-RAG trigger pool" ✓
- FCT as "the strongest length-robust trigger in the audited pool" ✓ (Δ = +0.190, Bonferroni LB +0.047)
- "Length is anti-predictive of retrieval need on the held-out 2Wiki cells" ✓ (raw AUROC ≤ 0.52)

**Claims that need softening:**
- *"All six (now seven) follow the same evaluation protocol and predate the length-confound literature"* — answer-trajectory family is **contemporaneous**, not predating. Restructure to "the six published peer families predate" (already fixed in last revision).
- *"The verbosity-compensation pattern observed on single-hop QA"* — Zhang 2024 doesn't explicitly contrast single vs multi-hop on this axis; technically supported but loose. Optional refinement.
- *Janiak's "45.9% / 30.4%"* — these are Mistral × NQ-Open worst-case, not dataset-mean. Optional softening to "up to 45.9% in certain cells."

---

## 4. Audit posture: what to keep robust, what to soften

### Keep robust (the load-bearing claims)
- The 4 Bonferroni-robust survivors and their Δ / 95% CI / Bonferroni LB numbers (verified against `e7_bonferroni.out`).
- The 9 nominal-95% survivors and their family classification.
- FCT raw AUROC per cell (0.677 / 0.604 / 0.642 / 0.784) — verified against `moonshot_explore.out`.
- LLM-judge κ = 0.707, 8/9 survivors preserved, FCT strengthens to +0.222 — verified against `llm_judge_audit_both.out`.
- K=1 vs K=5 ablation Δ and CIs — verified against `a1_k1_vs_k5_r1_*.out`.
- Held-out 2Wiki tokens_generated raw AUROC 0.457 / 0.518 — verified against `uncertainty_line_audit_*.out`.

### Soften (where prior work doesn't fully license the framing)
- "All six predate the length-confound literature" → six of seven (already done).
- "single-hop QA" framing for Zhang verbosity-compensation → optional.
- "45.9% drop" framing → optional.

### Flag for fix (real numerical inconsistencies with code)
- App. D Table 6 `ε^samp_max` numbers: paper 0.062 / 0.070, code 0.111 / 0.106 — **mismatch**.
- App. D caption "averaged over 5 seeds": code uses 3 seeds — **mismatch**.

### Verify (additional checks needed during audit)
- Bibliography keys: `guo_realm_2026`, `pandey_selfdoubt_2026` need to exist in `custom.bib`.
- Bibliography year fields for `yao_seakr_nodate`, `wang_chain--probe_nodate`, `janiak_illusion_nodate`.
- The Spearman App. F caption (`ρ ∈ [+0.18, +0.46]`) doesn't quite match the actual data — pre-existing.
- The Adapter Notes appendix references: `\ref{sec:appendix-adapters}` resolves OK.

---

*End of palette cleanser. Proceed to `final_audit.md` for the section-by-section top-down audit.*
