# How much weight will "length is a confound" carry in the paper?

> Concise reading doc: which papers support the claim, with the specific numbers, what each one *actually* says, and what the field has NOT yet shown.
>
> **Verdict up-front: the claim is well-supported in UQ / hallucination-detection at large, with strong direct quotes from 5 papers between 2024 and 2026 covering both formal proof, empirical replications, behavioral mechanism, and a methodological fix. The gap in the literature — and our paper's wedge — is that none of them has tested the claim on the published *dynamic-RAG trigger* pool specifically (DRAGIN, FLARE, SeaKR, SelfCheckGPT). That's exactly what our V2 and R1 cohorts do.**

---

## 1. The five-paper foundation (in order of citation priority)

### 1.1 Santilli et al. 2024 — the FORMAL PROOF

**Paper.** Santilli, Goliński, Kirchhof, Danieli, Blaas, Xiong, Zappella, Williamson — *Revisiting Uncertainty Quantification Evaluation in Language Models: Spurious Interactions with Response Length Bias Confound Conclusions*. arXiv:2504.13677 (Apr 2025).

**What they prove (Appendix C, Scenario ii).** When both the UQ score $\hat u$ and the noisy correctness function $\tilde Q$ share a common confounder (response length $L$), the AUROC estimator is **systematically skewed** — not just noisy. Ranking among UQ methods *flips* across correctness functions.

**Specific empirical numbers.** Spearman correlations between token-length and standard UQ measures (their Table 2 / leaderboard):

| UQ measure | Spearman ρ vs token length |
|---|---:|
| Perplexity | **−0.7** |
| Naive Entropy | **−0.6** |
| Mean Token Entropy | **−0.6** |
| Semantic Entropy | **−0.7** |
| Length-only baseline | (competitive with the above) |

**Bottom line.** *"Token Length itself is a competitive UQ-method baseline on our leaderboard."* This is the formal anchor for the entire residualization protocol family.

### 1.2 Vashurin et al. 2025 UNCERTAINTY-LINE — the FIX

**Paper.** Vashurin, Goloburda, Nakov, Panov — *UNCERTAINTY-LINE: Length-Invariant Estimation of Uncertainty for Large Language Models*. arXiv:2505.19060 (May 2025). Code: [github.com/stat-ml/uncertainty-line](https://github.com/stat-ml/uncertainty-line).

**What they propose.** Even nominally length-normalized UQ measures (Perplexity, Mean Token Entropy, Mean Conditional Negative Log-likelihood, Token-SAR) retain residual length bias. Fix:

$$\hat u_{\text{LINE}}(\hat y, x) = \hat u(\hat y, x) - \hat f(L(\hat y))$$

OLS regression of the UQ score on length, take the residual as the length-invariant estimate. Post-hoc, model-agnostic, applicable to any scalar UQ measure.

**Their empirical claim.** Consistent improvements across NMT (WMT14), summarization (CNN/DailyMail, XSum), and mathematical reasoning datasets.

**Why this matters for our paper.** **This IS our `_residualize_signal` protocol.** Our `analysis/uncertainty_line_audit.py` (run 2026-05-15) verified our implementation agrees with theirs within **ε_protocol = 0.029 AUROC max, mean 0.012** across 20 signal-cells on Qwen R1. **Direct attribution upgrade for the paper.**

### 1.3 Janiak et al. EMNLP 2025 — the DIRECT EMPIRICAL VALIDATION

**Paper.** Janiak, Binkowski, Sawczyn, Gabrys, Shwartz-Ziv, Kajdanowicz — *The Illusion of Progress: Re-evaluating Hallucination Detection in LLMs*. EMNLP 2025 (pp. 34728–34745).

**The headline quote you can cite verbatim.** *"Simple heuristics based on response length can rival complex hallucination detection techniques."*

**Specific drops when correctness function changes (their Table / §4).** Switching ROUGE → LLM-as-Judge labels on NQ-Open / SQuAD / TriviaQA:

| method | drop in AUROC |
|---|---:|
| **Perplexity** | **−45.9%** |
| **EigenScore** | **−30.4%** |
| Semantic Entropy | (matches length heuristic on multiple cells) |

**Why this matters.** This is **published external authority** for the same finding our V2 R1 D1 closed_em audit produced (`tokens_generated` AUROC = 0.645 on Phi HotpotQA beats every internal signal). Cite them as direct empirical concordance.

### 1.4 Y. Zhang et al. 2024 Verbosity ≠ Veracity — the BEHAVIORAL MECHANISM

**Paper.** Y. Zhang, Das, R. Zhang — *Verbosity ≠ Veracity: Demystify Verbosity Compensation Behavior of Large Language Models*. arXiv:2411.07858 v2 (Dec 2024; ACL 2025 UncertainLP).

**The phenomenon (Verbosity Compensation, VC).** LLMs systematically generate longer, wordier responses when uncertain. Five subcategories: question-repeating, enumerating, ambiguity, verbose details, verbose format.

**Specific numbers across 14 LLMs × 5 datasets (their §4):**

| metric | value |
|---|---|
| GPT-4 VC frequency on hard QA | **50.4%** |
| Performance gap (concise − verbose) on Qasper | **27.6%** EM |
| Verbosity-uncertainty correlation | strong and consistent across all 5 evaluated datasets, including **HotpotQA**, **MuSiQue**, **2WikiMHQA** (bundled as LongBench) |

**Why this matters.** Supplies the **causal story** for §5 discussion. The chain is:

> length ↔ verbosity-compensation ↔ uncertainty ↔ incorrectness ↔ retrieval-need

You can write this as a single causal sentence in §1 / §5 with citations to Zhang for the verbosity-uncertainty link and to your own R1 data for the retrieval-need link. **Highest-overlap paper with your evaluation datasets** — directly comparable benchmarks.

### 1.5 Ielanskyi et al. 2025 — the BROADER METHODOLOGY PITFALL

**Paper.** Ielanskyi, Schweighofer, Aichberger, Hochreiter — *Addressing Pitfalls in the Evaluation of Uncertainty Estimation Methods for Natural Language Generation*. arXiv:2510.02279 (Oct 2025).

**What they show (§3-§4).** Approximate correctness functions (ROUGE-1/2/L, BLEU, BERTScore, BLEURT, multiple LLM-as-Judge variants) **disagree substantially** on UQ method rankings. Pairwise mutual AUROC as low as **0.53** (BLEU vs ROUGE on CoQA + Llama-3-8B-IT); Spearman ρ between rankings sometimes **negative** (ρ = −0.66 for BLEU vs ROUGE on CoQA).

**Length is identified as one driver of this disagreement.**

**Why this matters for our paper.** This paper *defends* our `lift_em` paired-design specifically — both arms use the **same** EM correctness function, so the correctness-function-disagreement pitfall is structurally cancelled. **Cite for the §3 protocol-justification paragraph: *"our paired-oracle target is partially insulated from the correctness-function-disagreement critique of Ielanskyi et al. (2025) because the two arms share the same correctness function."***

---

## 2. Supporting / corroborating papers (cite as a footnote bundle)

### 2.1 Mao et al. ACL 2026 (Confidence Over Time, arXiv:2601.13387)

Explicitly state: *"Joint generation probability suffers from severe length bias, making it unsuitable for long-form reasoning."* They propose stepwise STL formulas but do NOT propose a residualization fix. **Acknowledged-gap citation.**

### 2.2 Chen et al. 2026 (Dynamic Instability, arXiv:2602.02863)

Cap output to 128 tokens *specifically as a length control* (their §3 / §4). Report fixed-window controls $S_w$ for $w \in \{10, 20, 50, 100\}$. Different statistical posture (truncation vs regression) but same intent. **Methodologically convergent citation.**

### 2.3 Mallen et al. 2023 (When Not to Trust Language Models)

The parent of the dynamic-RAG-triggering question: "when is retrieval needed for this question?" They use popularity as a proxy; we use the AID family + DRAGIN. **Cite as the conceptual parent for `lift_em`.**

---

## 3. The GAP — what nobody has done

| paper | length confound tested on... |
|---|---|
| Santilli 2024 | UQ leaderboard methods (Perplexity, Semantic Entropy, Naive Entropy) on NQ/CoQA/TriviaQA |
| Vashurin 2025 | UQ on NMT, summarization, math reasoning |
| Janiak 2025 | Hallucination detection on NQ-Open, SQuAD, TriviaQA |
| Y. Zhang 2024 | Verbose-vs-concise gap on 5 QA datasets |
| Ielanskyi 2025 | Correctness function disagreement on COQA, TRIVIA, SQUAD |
| **OUR PAPER** | **the published *dynamic-RAG trigger* pool — DRAGIN, FLARE, SeaKR, SelfCheckGPT, etc. — at signal-level, on multi-hop QA with paired oracle re-decode** |

**Nobody has run the length-residualization protocol on the published RAG-trigger pool. This is the wedge.** All five foundation papers establish that length is a confound in *some* UQ / hallucination-detection setting. Our paper is the first to apply the residualization protocol to the specific population of *triggers* used in dynamic-RAG systems, on multi-hop QA, with the paired-oracle target. The protocol is theirs (Vashurin); the application domain is ours.

---

## 4. Our own data confirming length-confound on the RAG-trigger pool

Direct quotes you can cite from `_rq1-agent.md`:

| cohort | finding |
|---|---|
| **V2 Phi (n=393 HotpotQA / n=452 2Wiki)** | `tokens_generated` AUROC = **0.749** on HotpotQA, **0.582** on 2Wiki; **beats every internal signal raw and residualized** on HotpotQA. Only DRAGIN-RIND survives length-residualization on both. |
| **R1 Phi D1 closed (n=200 HotpotQA)** | `tokens_generated` AUROC = **0.645** on `closed_em=0`; under residualization every internal signal collapses to <0.5 (drift_cosine, ent.max, dragin, FLARE, etc. all paired-significantly worse than tokens). |
| **R1 Qwen2.5 × 2WikiMHQA (n=200)** | `tokens_generated` AUROC = **0.457** — **inverted-length-correlation cell** where shorter answers are *worse*. Internal signals (AID min_majority + DRAGIN-RIND) paired-bootstrap-significantly beat tokens here under residualization. |
| **R1 Gemma D1 (n=200 HotpotQA)** | `tokens_generated` AUROC = **0.594** (much lower than P0 small-n estimate of 0.750) — heterogeneity across cells confirms length-dominance is not universal. |

These four numbers directly mirror the literature: length is dominant in many cells (V2 / R1 Phi HotpotQA — *à la* Janiak / Santilli), but heterogeneity exists and can even *invert* (Qwen 2Wiki — making length below-chance and giving internal signals headroom).

---

## 5. How strong is the claim, on a paper-defense scale?

**Verdict: VERY STRONG.** Five concentric layers of support:

| layer | citation | what it gives you |
|---|---|---|
| **Formal proof** | Santilli 2024 | mathematical anchor: length-confound systematically skews AUROC |
| **Methodological fix** | Vashurin 2025 (UNCERTAINTY-LINE) | the residualization protocol you adopt; **public code that you verified your implementation against (ε=0.029)** |
| **Direct empirical authority** | Janiak EMNLP 2025 | "length rivals complex methods" — published at the venue you're targeting |
| **Behavioral mechanism** | Y. Zhang 2024 (Verbosity ≠ Veracity) | verbose-when-uncertain across HotpotQA/MuSiQue/2Wiki — same datasets as us |
| **Methodological pitfall context** | Ielanskyi 2025 | length is one driver of correctness-function-disagreement — defends our `lift_em` design |

Any reviewer who challenges the length-confound claim is challenging five 2024-2026 papers from established UQ/hallucination-detection groups (Apple/Sapienza/NUS, MBZUAI, EMNLP, JKU, ACL UncertainLP). **You're on the right side of a converged consensus, not a fringe position.**

The only honest caveat: every one of these papers establishes the claim in general UQ / hallucination-detection. **You're the first to establish it on the dynamic-RAG trigger pool specifically.** That's a feature, not a bug — it's exactly the "new application domain" that makes a method paper worth publishing.

---

## 6. Naming sidebar — "AID" concern

You're right to be cautious about naming. **"AID" is unambiguously an *adaptation* of the Datta-2026 AnswerSwitches primitive + Wang-2025 Chain-of-Probe family**, not an invention. Three honest options:

1. **Drop the acronym entirely.** Refer to it as *"step-level answer-trajectory triggers"* or *"answer-flip-rate triggers"* — descriptive, no acronym, no naming claim. **Safest and most defensible.**
2. **Borrow Datta's vocabulary explicitly.** *"Answer-trajectory triggers (AnswerSwitches-derived, Datta et al. 2026)."* Acknowledges parentage; no original claim.
3. **Name only the aggregations, not the family.** Refer to `flip_rate`, `late_flips`, `min_majority` as the four aggregations of "the Datta-style answer-trajectory signal." Each aggregation is descriptive; no umbrella term needed.

My recommendation: **Option 1 or 3.** If you must use an umbrella term, *"step-level answer-trajectory triggers"* is more accurate and less proprietary-feeling than "AID." If you keep "AID" only as a within-paper shorthand for compactness, define it in §3 explicitly as *"step-level answer-trajectory triggers, hereafter AID for brevity; derived from the AnswerSwitches primitive of Datta et al. (2026)."* Reviewers will not object to a transparently-credited shorthand.

---

## 7. One-sentence form for the §1 abstract / intro

> *"A convergent literature establishes that response length confounds uncertainty-quantification evaluation in language models (Santilli et al., 2024; Vashurin et al., 2025; Janiak et al., 2025; Y. Zhang et al., 2024; Ielanskyi et al., 2025). We are the first to test whether this confound extends to the published dynamic-RAG trigger pool (DRAGIN, FLARE, SeaKR, SelfCheckGPT), on multi-hop QA with paired oracle re-decode, under the residualization protocol of Vashurin et al. (2025); two signal families survive."*

That sentence carries the claim with five citations and a clean novelty wedge. Drop into §1 verbatim if you want.

---

## 8. Action items (drop-in for the paper draft)

1. **§1 paragraph** — use sentence in §7 above.
2. **§2 (related work)** — one paragraph per foundation paper, in the order of §1.1–§1.5 above. Total ~250 words.
3. **§3 (methodology)** — cite Vashurin 2025 as the protocol parent; cite our ε_protocol=0.029 head-to-head replication as the appendix calibration.
4. **§5 (discussion)** — cite Y. Zhang 2024 as the behavioral causal story for length-confound.
5. **§6 (limitations)** — cite Ielanskyi 2025 for the correctness-function-disagreement caveat and note that `lift_em` is partially insulated.
6. **Appendix** — reference paper-defense replication numbers from `analysis/uncertainty_line_audit_{arch}.out`.

Total citations you must add to `litReview/-rq1-v2-litReview.md` and the .bib: Santilli 2024 (2504.13677), Vashurin 2025 (2505.19060), Janiak 2025 (EMNLP — find the proceedings citation), Y. Zhang 2024 (2411.07858), Ielanskyi 2025 (2510.02279). All five are already in `__lit-review-main/_lit-review.md` with full capsules.
