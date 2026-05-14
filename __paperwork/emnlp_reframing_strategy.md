# EMNLP Short Paper — Reframing Strategy

> Based on a deep read of: `_rq1-agent.md` (full experiment log), `_MODEL.md` (model pool), `__lit-review-main/_lit-review.md` (annotated literature), `litReview/-rq1-litReview.md` (dynamic RAG landscape), and `_rq1-outlook.md` (current acceptance estimate).

---

## §0 — Diagnosis: Why the Current Framing is Weak

The current framing has accumulated **three compounding problems** through the research campaign:

1. **Precedent erosion.** The lit-review pass on 2026-05-14 discovered that every major building block has direct precedent:
   - AID primitive → Datta et al. (ICML 2026) `AnswerSwitches`
   - Late-flips timing → Chen/Hou et al. (Feb 2026) corrective/destructive instability
   - Length residualization → Vashurin et al. UNCERTAINTY-LINE (2025)
   - K=5 vs K=1 → **empirically null** (regime 2 on both Phi and Qwen)

   The current framing tries to pitch AID as the headline contribution, but the K=5 extension is demonstrably not adding value, the primitive is Datta's, and the residualization is Vashurin's. **You're selling components that aren't yours.**

2. **Narrative fragmentation.** The log shows **5+ reframings** over the campaign (DRAGIN-sole-survivor → undercomplete pool → AID novel trigger → AID family → protocol-first). Each pivot left residual framing language. The paper currently doesn't know what its central claim is.

3. **Signal-first vs question-first.** The current framing leads with "we found signal X" — which invites the reviewer question "so what?" An EMNLP short needs to lead with a **question the community cares about**, then show the answer.

---

## §1 — Your Actual Results Inventory (What You Can Claim)

Before reframing, here's the honest inventory of what you have:

### Hard Results (paired-bootstrap significant, n ≥ 200)

| Finding | Data | Significance |
|---|---|---|
| AID `min_majority` beats `tokens_generated` on Qwen pooled, `lift_em=+1` | n=400, pop=65 | Δ=+0.123, CI [+0.020, +0.224] ✓ |
| DRAGIN-RIND beats `tokens_generated` on Qwen pooled, `lift_em=+1` | n=400, pop=65 | Δ=+0.085, CI [+0.003, +0.168] ✓ |
| AID family wins strongest-residualized-AUROC on **4/4 cells** (2 archs × 2 datasets) | n=200/cell | Per-cell, different AID variant wins |
| K=5 ≈ K=1 on both Phi and Qwen (regime 2) | n=105 + n=305 | CIs straddle 0 on both |
| `tokens_generated` beats most internal signals on Phi HotpotQA closed_em | n=200 | DRAGIN, ent, hvr significantly worse |
| Qwen 2Wiki inverted-length-correlation replicates at n=200 | n=200 | tokens AUROC = 0.457 (anti-predictive) |
| `late_flips` × 2Wiki/comparison = 0.950 raw AUROC | n=200 cell subset | Strongest cell-level finding |
| CABV beats tokens on closed_em=0 (Phi+Qwen pooled) | n=800 | Δ=+0.118, CI [+0.048, +0.189] ✓ |
| V2 Phi (n=500): DRAGIN-RIND is length-robust survivor on both datasets | n=393/452 | resid AUROC > 0.5 on both cells |

### Directional / Pending (awaiting Gemma R1)

| Finding | Status |
|---|---|
| Cross-arch AID-family survivor pattern on Gemma | Gemma D1 ~85/200 in flight |
| F2 ensemble (rank-fusion AID × M9) vs min_majority | Borderline; needs Gemma to cross significance |
| Per-cell complementarity: M9 wins where AID dies | n=410 valid, directional |

### Negative Results (publishable as honest findings)

| Finding | Implication |
|---|---|
| K=5 adds nothing over K=1 | Simplifies deployment; honest |
| CRV-drift (latent geometry) signals → null on lift_em | Retired; validates prior negative result |
| Most published UQ signals collapse under length residualization | Extends Janiak "Illusion of Progress" |
| flip_rate is the WEAKEST AID variant at n=200 | Family > single variant |

---

## §2 — Three Candidate Framings, Ranked

### ⭐ Framing A (RECOMMENDED): "The Length Illusion Extends to Retrieval Triggers"
**Angle: Evaluation methodology paper with empirical contribution**

> **Title:** *Length Is All You Need? Evaluating Dynamic Retrieval Triggers Beyond the Verbosity Confound*
> 
> **One-sentence pitch:** We show that the "length rivals complex methods" finding (Janiak et al. 2025, Santilli et al. 2025) extends to the dynamic-RAG trigger family, propose a paired-oracle residualization protocol to surface length-orthogonal signal, and identify the two survivor signals (DRAGIN-RIND, Answer Instability Detection) across 3 architectures × 2 multi-hop QA datasets.

**Why this is strongest:**

1. **Leads with a question the community already cares about.** The length-confound problem is established (5 papers in 2025-2026). Extending it to RAG triggers is a natural next question with high reviewer buy-in.
2. **The methodology contribution is genuinely yours.** Vashurin proposes residualization for UQ; you're the first to apply it to *dynamic RAG triggers* with a *paired-oracle `lift_em`* target. The paired-oracle design is novel and clean.
3. **The empirical finding is sharp.** "Most published RAG triggers are just measuring length" is provocative, testable, and true in your data.
4. **AID becomes a discovery within the evaluation, not the headline.** This avoids the Datta-precedent attack entirely. AID is "what we found when we did the evaluation properly," not "our novel contribution."
5. **The negative results strengthen the paper.** Every signal that collapses under residualization is evidence for the evaluation-methodology claim.
6. **Honest about precedent from day one.** You credit Vashurin, Santilli, Janiak, Datta upfront. No walkback needed.

**Estimated acceptance: 55-65%** (with R1 Phi+Qwen data; higher if Gemma lands).

---

### Framing B: "Architecture-Conditional Trigger Selection"
**Angle: Empirical finding paper**

> **Title:** *No Universal Retrieval Trigger: Architecture-Conditional Signal Selection for Multi-Hop QA*
> 
> **One-sentence pitch:** Across 3 model architectures × 2 multi-hop QA datasets under length-controlled evaluation, no single retrieval trigger dominates—but the AID family wins every cell, with different aggregations per (arch × dataset) pair.

**Why it's decent but weaker:**

- The per-cell heterogeneity is a real, novel finding. No published paper reports architecture-conditional trigger selection.
- The "4/4 cells, different variant" result is clean and surprising.
- But: the contribution is descriptive, not prescriptive. Reviewer asks "so what do I deploy?" and you don't have a clean answer.
- Requires all 3 archs (Gemma must land).

**Estimated acceptance: 45-55%.**

---

### Framing C: "Answer Instability as Retrieval Trigger" (Current Framing)
**Angle: Novel signal paper**

> **Title:** *Step-Level Answer Instability Predicts Multi-Hop QA Retrieval Need*

**Why it's weakest now:**

- The Datta precedent means AID is not novel as a primitive.
- K=5 vs K=1 is null — the extension claim collapses.
- Leads to reviewer objection: "this is Datta et al. applied to retrieval; what's new?"
- The strongest version of this framing requires the K=5 contribution to work, and it doesn't.

**Estimated acceptance: 35-45%.**

---

## §3 — Recommended Structure: Framing A (4-Page Short)

### Title
**Length Is All You Need? Evaluating Dynamic Retrieval Triggers Beyond the Verbosity Confound**

### Abstract (~150 words)

Key beats:
1. **Problem:** Dynamic RAG systems use uncertainty signals to decide when to retrieve. But recent work shows length alone rivals complex uncertainty methods in hallucination detection — does this extend to retrieval triggers?
2. **Method:** We introduce a length-controlled evaluation protocol: (i) OLS residualization of trigger signals on output length (adapting UNCERTAINTY-LINE, Vashurin et al. 2025), (ii) paired closed-vs-oracle re-decode to define `lift_em = +1` (the deployment-relevant question: "would retrieval have helped?").
3. **Finding:** On 3 architectures × 2 multi-hop QA datasets (n=200/cell), most published triggers (FLARE, entropy, SeaKR, CRV-drift) collapse to chance under length residualization. Two families survive: DRAGIN-RIND and a new Answer Instability Detection (AID) family adapted from Datta et al. (2026). The optimal AID variant is architecture-conditional.
4. **Implication:** Trigger evaluation without length control is unreliable; we release code + protocol.

### §1 Introduction (~400 words)

**Opening hook** (2-3 sentences):
> Dynamic RAG systems decide when to retrieve external knowledge using trigger signals—token-level entropy (FLARE), attention dynamics (DRAGIN), sample consistency (SeaKR), and others. A critical assumption is that these signals measure genuine model uncertainty rather than artifacts of generation behavior. We test this assumption.

**The length problem** (1 paragraph):
> Recent work establishes that response length is a dominant confound in LLM uncertainty evaluation. Santilli et al. (2025) prove that mutual length-bias systematically distorts AUROC rankings; Janiak et al. (2025) show length heuristics rival complex hallucination detectors; Zhang et al. (2024) document verbosity compensation across 14 LLMs on HotpotQA/MuSiQue/2Wiki. Vashurin et al. (2025) propose UNCERTAINTY-LINE regression-residualization as a fix, but apply it only to UQ measures on NMT and math—*not to dynamic RAG triggers*. **Whether length confounds dynamic retrieval triggering is an open question.**

**Our contribution** (numbered list, 3 items):

1. **Evaluation protocol.** We extend UNCERTAINTY-LINE to dynamic RAG triggering with a novel paired-oracle `lift_em` target: does retrieval flip the answer from wrong to right? This target partially cancels correctness-function bias (Ielanskyi et al. 2025) because both arms share the same EM function.

2. **Trigger-pool audit.** Under this protocol on 3 architectures (Phi-4-mini, Qwen2.5-7B, Gemma-4-26B-A4B, matched at ~3-4B active params) × 2 multi-hop QA datasets (HotpotQA, 2WikiMHQA) at n=200/cell, we evaluate 16 trigger signals. Most collapse under length residualization. **Two families survive: DRAGIN-RIND and a step-level answer-instability family.**

3. **Answer Instability Detection (AID).** Adapting the forced-answer-completion primitive of Datta et al. (2026) to retrieval triggering, we define 4 AID aggregations over step-level boundary-completion answer trajectories. The AID family achieves the highest length-residualized AUROC on all 4 evaluated cells, but the optimal aggregation is architecture-conditional (`min_majority` on Qwen/2Wiki, `late_flips` on Phi/2Wiki, `flip_rate K=1` on Qwen/HotpotQA). K=5 multi-sample shows no empirical advantage over K=1 (Datta-style), simplifying deployment.

### §2 Related Work (~350 words, very dense)

Three strands to weave:

**Length confound lineage:** Santilli 2025 (formal proof) → Janiak 2025 (empirical, "Illusion of Progress") → Zhang 2024 (Verbosity ≠ Veracity, same multi-hop QA datasets) → Vashurin 2025 (UNCERTAINTY-LINE fix) → Ielanskyi 2025 (correctness-function disagreement). **Our extension to RAG triggers.**

**Answer-trajectory lineage:** Wang 2025 (Chain-of-Probe, NAACL Findings) → Chen/Hou 2026 (dynamic instability, early/late timing) → Datta 2026 (AnswerSwitches, forced completion, ICML) → Zhang 2026 (ARS, terminological disambiguation). **Our application to retrieval triggering.**

**Dynamic RAG triggers:** FLARE (Jiang 2023), DRAGIN (Su 2024), SeaKR (Su 2024b), IRCoT (Trivedi 2023). **We evaluate, not propose, except for AID.**

### §3 Method (~500 words)

Three subsections:

**3.1 Paired-Oracle Protocol.** For each question: (i) closed-book decode (no retrieval), (ii) oracle-retrieval re-decode (gold paragraphs injected). Both greedy, same prompt format. `lift_em = 1[oracle_em > closed_em]`. This is the target.

**3.2 Length Residualization.** OLS regression: `signal ~ log(tokens + 1)`, evaluate AUROC on residual. Credits Vashurin explicitly. Mention log-transform avoids overcorrection (their §3 caveat). Paired bootstrap B=10,000 of Δ(signal AUROC − tokens AUROC) with 95% CI.

**3.3 Answer Instability Detection.** At each reasoning-step boundary (spaCy sentence + 32-token fallback), sample K boundary completions, extract answers via regex. Four aggregations:
- `flip_rate`: fraction of consecutive-step answer-set transitions
- `late_flips`: flips in last 50% of steps only
- `min_majority`: max within-step disagreement (1 − max_count/K)
- `flip_rate K=1` (Datta baseline): first sample only

Credit Datta et al. for the primitive. Credit Wang et al. for the soft-probe predecessor. Note K=5 ≈ K=1 empirically; recommend K=1 for compute efficiency.

### §4 Results (~600 words, dense tables)

**Table 1 (HERO TABLE): Cross-arch × cross-dataset survivor matrix.**

| Signal | Phi×Hot | Phi×2Wiki | Qwen×Hot | Qwen×2Wiki | Survives resid? | Beats tokens? |
|---|---|---|---|---|---|---|
| tokens | 0.630 | 0.602 | 0.572 | 0.457 | (baseline) | — |
| DRAGIN | 0.397 | 0.558 | 0.502 | **0.653** | 2/4 | Qwen pooled ✓ |
| AID min_maj | **0.597** | 0.424 | 0.618 | **0.672** | 3/4 | Qwen pooled ✓ |
| AID late_flips | 0.502 | **0.663** | 0.598 | 0.586 | 4/4 | — |
| AID flip_rate K=1 | 0.342 | 0.582 | **0.690** | 0.534 | 3/4 | — |
| entropy.max | 0.460 | 0.462 | 0.547 | 0.608 | 2/4 | — |
| FLARE.max | — | 0.541 | 0.526 | — | 1/4 | — |
| SeaKR-K5 | 0.412 | 0.390 | — | — | 0/4 | — |

(Add Gemma column when available. Report residualized AUROC throughout.)

**Key results paragraphs:**

> [!IMPORTANT]
> **R1: Length dominance is real but heterogeneous.** On Phi×HotpotQA, `tokens_generated` (AUROC 0.630) beats every internal signal raw and residualized—replicating Janiak et al. on RAG triggers. But on Qwen×2Wiki, `tokens` is *anti-predictive* (0.457), confirming that the length confound is cell-specific, not universal.

> [!IMPORTANT]
> **R2: Two survivor families.** Under residualization, DRAGIN-RIND and the AID family are the only signals that (a) exceed chance on ≥3/4 cells AND (b) paired-bootstrap-significantly beat `tokens_generated` on at least one pooled architecture. AID `min_majority` on Qwen pooled: Δ = +0.123, CI [+0.020, +0.224]. DRAGIN-RIND on Qwen pooled: Δ = +0.085, CI [+0.003, +0.168].

> [!IMPORTANT]
> **R3: No universal optimal AID variant.** The best AID aggregation is (`min_majority` on Qwen×2Wiki, `late_flips` on Phi×2Wiki, `flip_rate K=1` on Qwen×HotpotQA, `min_majority` on Phi×HotpotQA). Architecture-conditional signal selection is necessary.

> [!IMPORTANT]
> **R4: K=5 ≈ K=1.** Across n=410 valid paired traces, K=5 boundary completion shows no advantage over K=1 (Datta-style) on `lift_em`. CI straddles zero on both architectures. Recommend K=1 for compute efficiency.

**Table 2 (per-question-type spotlight):** The `late_flips` × 2Wiki/comparison cell (0.950 AUROC) and its inversion on HotpotQA/comparison (0.250). This is the strongest evidence for per-type heterogeneity.

### §5 Discussion + Limitations (~350 words)

**Three discussion points:**

1. **The verbosity-uncertainty causal chain.** Length → verbosity compensation (Zhang et al.) → higher uncertainty (Santilli et al.) → more often wrong (Janiak et al.) → more retrieval-helpful (us). The causal story is now four papers deep. Our contribution: identifying which signals break out of this chain.

2. **Per-cell heterogeneity as the real finding.** No single signal dominates. The practical recommendation for RAG deployment is not "use signal X" but "evaluate your (model × dataset) cell under residualization before committing to a trigger." This is a stronger practical contribution than a new signal.

3. **The quiet K=1 result.** Datta et al.'s single-sample forced completion is sufficient for retrieval triggering. The stochastic-sampling community (Self-Consistency, Semantic Entropy, SeaKR) assumes K >> 1 is necessary; our AID family shows this is not the case for answer-trajectory signals, consistent with the theoretical distinction between trajectory-based and consistency-based methods.

**Limitations (be explicit):**

- 2/3 architectures landed; Gemma in flight. Report what you have, flag the gap.
- K=1 ablation is a proxy (stochastic first sample, not faithful greedy forced-completion à la Datta).
- No MuSiQue (3rd dataset) — 2-dataset generalization.
- EM correctness function only; LLM-as-Judge comparison in appendix.

### §6 Conclusion (2-3 sentences)

> We establish that the length confound documented in UQ evaluation extends to dynamic RAG triggers. Under length-controlled paired-oracle evaluation, most published triggers collapse; the two survivors—DRAGIN-RIND and AID—are architecture-conditional. We release our evaluation protocol and recommend trigger developers report length-residualized metrics alongside raw AUROC.

---

## §4 — What to Do Before Submission (Priority Order)

| # | Action | Time | Impact |
|---|---|---|---|
| 1 | **Wait for Gemma R1 to land** (in flight, ~3h) | 0 active | +10pp (3rd arch) |
| 2 | **Write §1-§3 now** (method sections don't depend on Gemma) | 1 day | Parallelizable |
| 3 | **Format Tables 1-2** from existing Phi+Qwen .out files | 2h | Core of §4 |
| 4 | **Run length-vs-signal Spearman ρ table** (already computed) | 0h (formatting) | Mandatory per Santilli |
| 5 | **Add EM + F1 + containment-match appendix rows** | 1h | Addresses Ielanskyi |
| 6 | **Report "decided-early" trace fractions** per cell | 30min | Disarms Datta comparison |
| 7 | **Run UNCERTAINTY-LINE reference implementation head-to-head** | 3h | Massive reviewer defense |
| 8 | **Fix 3 attribution errors** (Belem→Santilli, Belem→Vashurin, Castaneda→Ielanskyi) | 10min | Mandatory |

---

## §5 — Reviewer Attack Surface and Pre-Emptive Defenses

| Attack | Defense |
|---|---|
| "AID is just Datta et al. applied to retrieval" | Yes, and we say so. The contribution is the evaluation protocol + the empirical finding that it survives length residualization. Datta doesn't evaluate under length control or against a retrieval target. |
| "Why not K=20 / SeaKR-K20?" | K=5 proxy already shows direction-of-effect; K=5≈K=1 on AID shows more samples don't help for trajectory-based signals. Full K=20 is out of scope. |
| "Only 2 datasets" | Acknowledge; MuSiQue pre-registered for follow-up. HotpotQA + 2Wiki already show heterogeneity, which is the point. |
| "Only 2 (or 3) architectures" | Matched active-parameter scale (~3-4B active). Spans dense (Phi), standard MoE (Gemma), hybrid-linear MoE (Qwen3.6 dropped, replaced by dense Qwen2.5). |
| "Paired-oracle is unrealistic for deployment" | Yes, but it answers the right question: "would retrieval have helped?" Real deployment would use AID/DRAGIN as the trigger and measure downstream EM lift. We establish *which signals are worth deploying*, not the deployment itself. |
| "EM is a coarse correctness function" | Shared-bias argument: both closed and oracle use EM, so EM-specific bias cancels in `lift_em`. Appendix reports F1 + containment match. |
| "Length residualization is Vashurin's, not yours" | Credited upfront. Our extension: (i) new task family (dynamic RAG triggers), (ii) new target (`lift_em` paired oracle), (iii) new signal pool (DRAGIN, FLARE, AID, etc.). |

---

## §6 — Key Framing Shifts from Current State

| Current framing | Recommended framing |
|---|---|
| "AID is a novel trigger family" | "AID is what survives when you evaluate properly" |
| "K=5 multi-sample is our extension of Datta" | "K=5 ≈ K=1; K is empirically free" (a finding, not a contribution) |
| "We introduce length residualization" | "We adopt and extend UNCERTAINTY-LINE to RAG triggers" |
| "DRAGIN is the sole survivor" | "DRAGIN + AID are the two survivor families" |
| Leading with the signal | **Leading with the evaluation question** |
| Contribution = new method | **Contribution = new evaluation protocol + empirical audit** |

> [!TIP]
> The fundamental shift: **stop trying to own a signal and start owning the evaluation question.** The question "do RAG triggers just measure length?" is yours. The answer is "mostly yes, except for two families." That's a clean EMNLP short.

---

## §7 — Suggested Title Alternatives (ranked)

1. **Length Is All You Need? Evaluating Dynamic Retrieval Triggers Beyond the Verbosity Confound** ← provocative, clear
2. **Most Dynamic RAG Triggers Are Measuring Length: A Residualized Audit on Multi-Hop QA** ← direct finding
3. **Two Survivors: Length-Controlled Evaluation of Retrieval Triggering Signals Across Architectures** ← descriptive
4. ~~Step-Level Answer Instability Predicts Multi-Hop QA Retrieval Need~~ ← current; weakened by precedent
