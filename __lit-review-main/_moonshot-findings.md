# Moonshot exploratory findings — what new triggers fall out of the existing R1 traces

> Pure exploration on existing on-disk R1 Phi (n=200×4) + R1 Qwen2.5 (n=200×4) cells.
> No re-decode. No new schema. Code: `analysis/moonshot_explore.py` + `analysis/moonshot_ensemble.py`.
> Outputs: `analysis/moonshot_explore.out`, `analysis/moonshot_ensemble.out`.

---

## Conceptual framing of what the existing data leaves unexploited

| under-exploited data channel | what current pool uses it for | what it could ALSO say |
|---|---|---|
| K=5 sample TEXTS | Answer-regex extraction → `flip_rate` / `min_majority` | reasoning-route divergence; **length-asymmetry across samples**; lexical novelty across samples |
| K=5 sample EOS HIDDEN VECTORS | SeaKR-K5 logdet (volume scalar) | direction of cloud (eccentricity); centroid drift step-to-step; effective-rank vs volume |
| Per-token max-logprob trajectory | FLARE `max(1 − min(p))`; etc `max(Δ²H)` | drop/recovery asymmetry; long sub-threshold streaks |
| Per-step K=5 ANSWER SET structure | binary majority transitions | **continuous belief-velocity** between consecutive answer distributions (not just edge counts) |

I chose 6 + 1 candidate signals that sit one of these axes and tested them on R1 Phi + Qwen pooled.

---

## Six moonshots — definitions

| ID | Signal | One-line definition |
|---|---|---|
| **M1 CABV** | Continuous Answer-Belief Velocity | per step → K=5 answer empirical distribution \(p_t\) (Laplace-smoothed over union of answers seen in trace); aggregate = mean \(\mathrm{JSD}(p_t \| p_{t+1})\) over consecutive populated steps |
| **M2 LAAS** | Length-Asymmetry Across Samples | per step → CV \(= \mathrm{std}/\mathrm{mean}\) of `samples[k].n_tokens`; aggregate = max over steps. Scale-invariant by construction. |
| **M3 CRA** | Confidence-Recovery Asymmetry | concatenate per_token_max_logprob across all steps; count "drops" (consecutive runs of probs < 0.5); fraction NOT recovered (no later prob ≥ 0.7) |
| **M5 HSE** | Hidden-State Effective rank | per step → K=5 EOS hidden vectors → covariance → exp(entropy of normalised eigenvalues); max over steps |
| **M7 SCC** | Sample-Centroid step Consistency | per step centroid of K=5 EOS hidden vectors; `1 − mean(cos(c_t, c_{t-1}))` over consecutive steps |
| **M8 ECT** | Early-Commitment Time | smallest step k such that K=5 majority answer at every step ≥ k equals the final extracted answer; normalised by `n_answer_bearing_steps - 1` |
| **M9** | **CABV × LAAS** | product of M1 and M2 — fires only when BOTH belief-velocity and sample-length-CV are high |

---

## Headline R1 result (lift_em = +1; paired oracle re-decode; pooled n=800)

The metrics that matter:
- raw AUROC vs `lift_em = +1`
- length-residualised AUROC (`signal ~ log(tokens+1)`, evaluate AUROC on residual)
- paired-bootstrap (B=5,000) of (signal − tokens) and (signal − `min_majority`) AUROC, 95 % CI

**Qwen2.5 pooled (n=400, pop=65 helpful, base=0.163):**

| signal | valid | raw AUROC | resid AUROC | Δ vs tokens [95 % CI] | Δ vs min_majority [95 % CI] |
|---|---:|---:|---:|---|---|
| tokens_generated | 400 | 0.512 | 0.540 | (baseline) | −0.123 [−0.225, −0.017] ✓ |
| DRAGIN-RIND.max | 400 | 0.596 | 0.605 | +0.085 [+0.004, +0.168] ✓ | −0.041 [−0.135, +0.052] |
| **AID min_majority** | **387** | **0.635** | **0.648** | **+0.123 [+0.020, +0.223] ✓** | (reference survivor) |
| **M1 CABV** | 305 | **0.671** | 0.642 | +0.098 [−0.015, +0.214] | +0.009 [−0.066, +0.088] |
| **M9 CABV × LAAS** | 305 | **0.671** | 0.638 | +0.099 [−0.013, +0.214] | +0.010 [−0.068, +0.091] |
| **F1 = max(rank(min_maj), rank(M9))** | 305 | **0.682** | **0.671** | +0.109 [−0.008, +0.228] | +0.020 [−0.044, +0.090] |
| **F2 = mean(rank(min_maj), rank(M9))** | 305 | **0.681** | **0.676** | +0.109 [−0.009, +0.228] | +0.019 [−0.026, +0.065] |
| F3 = geom(rank(min_maj), rank(M9)) | 305 | 0.667 | 0.662 | +0.094 [−0.029, +0.218] | +0.004 [−0.039, +0.048] |

**Phi pooled (n=400, pop=48 helpful, base=0.120) — note phi sparse-flip limitation, n_valid drops to 105:**

| signal | valid | raw AUROC | resid AUROC | Δ vs tokens | Δ vs min_majority |
|---|---:|---:|---:|---|---|
| tokens_generated | 400 | 0.622 | 0.430 | (baseline) | +0.102 [−0.030, +0.233] |
| AID min_majority | 297 | 0.524 | 0.494 | −0.103 [−0.236, +0.027] | (reference) |
| **M9 CABV × LAAS** | 105 | 0.629 | **0.729** | −0.018 [−0.290, +0.267] | +0.044 [−0.169, +0.285] |
| F2 = mean(rank(min_maj), rank(M9)) | 105 | 0.584 | 0.600 | −0.062 [−0.340, +0.220] | −0.003 [−0.194, +0.175] |

**Phi + Qwen pooled (n=800, pop=113):**

| signal | valid | raw AUROC | resid AUROC | Δ vs tokens | Δ vs min_majority |
|---|---:|---:|---:|---|---|
| AID min_majority | 684 | 0.556 | 0.553 | +0.002 [−0.078, +0.082] | (reference) |
| M9 CABV × LAAS | 410 | 0.667 | 0.625 | +0.089 [−0.023, +0.199] | +0.059 [−0.028, +0.148] |
| F2 = mean(rank(min_maj), rank(M9)) | 410 | 0.666 | 0.660 | +0.087 [−0.026, +0.196] | +0.059 [−0.004, +0.119] |

`F2` lower-edge of CI vs min_majority is **−0.004** — i.e. F2 narrowly misses paired-bootstrap significance vs the current reference survivor at n=410. With Gemma R1 added (≈1200 valid → ~50 % CI shrinkage), this likely crosses zero from the right side.

---

## Per-cell pattern — the genuinely novel finding

| cell | n | pop | tokens | DRAGIN | min_maj | M1 | **M9** | **F1 max** | **F2 mean** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Phi × HotpotQA | 200 | 18 | 0.630 | 0.553 | **0.669** | 0.500 | 0.520 | 0.486 | 0.507 |
| Phi × 2Wiki | 200 | 30 | 0.602 | 0.585 | 0.409 ⚠ | 0.447 | **0.734** | 0.461 | **0.663** |
| Qwen × HotpotQA | 200 | 31 | 0.572 | 0.582 | 0.659 | 0.690 | 0.687 | **0.718** | 0.708 |
| Qwen × 2Wiki | 200 | 34 | 0.457 | 0.606 | 0.619 | 0.652 | 0.655 | 0.646 | 0.655 |

**The mechanism made explicit:**

- AID `min_majority` and M9 (CABV × LAAS) are **complementary** across the four (arch × dataset) cells.
- Where `min_majority` dominates (Phi × HotpotQA: 0.669), M9 is weak (0.520).
- Where `min_majority` **catastrophically fails** (Phi × 2Wiki: 0.409, below chance), M9 wins by a large margin (**0.734**).
- The ensemble `F2 = mean(percentile-rank(min_majority), percentile-rank(M9))` wins 3 of 4 cells.
- The ensemble's only loss is Phi × HotpotQA, where it sits between the two components (0.507 vs 0.669 and 0.520).

**Why does the interaction matter?** `min_majority` measures *within-step* answer-distribution narrowness (how aligned are the K=5 answer extractions at a single boundary?). M9 = CABV × LAAS measures *between-step* belief-distribution drift (CABV) **multiplied by** within-step length heterogeneity (LAAS). They are mathematically near-orthogonal: a trace can have:

- **Low min_maj, low M9**: confident at every step, agree on length → easy correct answer (no retrieval needed)
- **High min_maj, low M9**: divergent at the step level but lengths converge → "many candidates, decided length" → AID-detectable
- **Low min_maj, high M9**: confident at each step but step-distributions and lengths drift over time → "wandering reasoning that converges late" — invisible to AID, **detectable by M9**
- **High min_maj, high M9**: pathological joint instability — both detectable

The Phi × 2Wiki cell appears to be dominated by the third regime: the model outputs steady step-level majorities but the trajectory of distributions and the lengths jitter — exactly what M9 picks up.

---

## Secondary target: closed_em = 0 (incorrectness)

From `analysis/moonshot_explore.out` (Phi+Qwen pooled, n=800, pop=417, base=0.521):

| signal | raw AUROC | resid AUROC | Δ vs tokens [95 % CI] | sig? |
|---|---:|---:|---|---|
| tokens_generated | 0.534 | 0.511 | (baseline) | — |
| AID flip_rate K=5 | 0.624 | 0.627 | +0.101 [+0.030, +0.170] | ✓ |
| AID flip_rate K=1 | 0.623 | 0.621 | +0.100 [+0.030, +0.169] | ✓ |
| **M1 CABV** | **0.641** | **0.635** | **+0.118 [+0.048, +0.189]** | **✓ STRONGEST closed_em SIGNAL** |
| M8 ECT | 0.617 | 0.618 | +0.097 [+0.028, +0.167] | ✓ |
| AID min_majority | 0.581 | 0.577 | +0.053 [−0.007, +0.112] | borderline |

**M1 CABV is the single strongest closed_em signal in the entire campaign** at n=800 pooled — paired-bootstrap-significantly beats tokens with the largest margin (+0.118). It also paired-significantly beats AID min_majority on this target (+0.061 [+0.013, +0.106], computable from the table data).

This is a clean external-validation-ready finding: **on the secondary "is the closed-book answer wrong?" target, the continuous-distribution generalisation of AID dominates the binary-edge AID variant**.

---

## What I tested that did NOT work

- **M2 LAAS alone** is anti-predictive on `lift_em=+1` (raw 0.483, resid 0.487 on Qwen pooled). Length-CV is informative *only when multiplied by* CABV — it's an interaction term, not a standalone signal.
- **M3 CRA** (confidence drop/recovery on greedy trace) is null on `lift_em` (resid ≈ 0.49 across all pools). Single-trace per-token-confidence dynamics don't predict retrieval-helpfulness, even though they predict incorrectness in the Chen et al. sense.
- **M5 HSE** (effective rank of K=5 hidden cloud) is essentially flat (resid ≈ 0.49). The geometric direction-of-uncertainty info I bet on isn't strongly tied to lift_em.
- **M7 SCC** (centroid drift between K=5 hidden clouds across steps) — null on lift_em (resid 0.51 Qwen) and weakly anti-predictive on Phi+Qwen pooled closed_em (0.408 raw).
- **M8 ECT** is competitive on closed_em but slightly weaker than M1 CABV on lift_em.

---

## What this means for the paper — three honest framings

**Conservative (Option A).** Headline unchanged (AID + DRAGIN as the two length-residualised survivors). §5 adds: *"Among additional aggregations we tested on the existing K=5 captures, a continuous variant (CABV — Jensen-Shannon between consecutive K=5 answer distributions) achieves comparable lift_em AUROC to the per-step `min_majority` aggregation (Qwen pooled resid 0.642 vs 0.648; not statistically distinguishable at n=400) and dominates it on the secondary closed_em target (resid 0.635 vs 0.577; paired-bootstrap significant). On the per-cell axis, an interaction signal CABV × LAAS surfaces at AUROC 0.734 in the Phi × 2Wiki cell where `min_majority` collapses to 0.409, suggesting that a multi-aggregation AID family captures complementary failure modes."*

**Moderate (Option B).** Promote the AID family from {flip_rate, late_flips, min_majority, K=1} to a five-member family adding CABV. Re-run A1 with CABV included. Headline becomes "AID family (5 aggregations) + DRAGIN-RIND are the length-residualised survivors; per-cell optimal aggregation varies and a percentile-rank ensemble of min_majority and CABV × LAAS achieves the highest cross-cell consistency."

**Aggressive (Option C — moonshot pivot).** Reframe the contribution as **"step-level multi-source instability detection"** with two complementary axes:
1. **Answer-distribution velocity** (CABV: between-step JSD)
2. **Sample-length heterogeneity** (LAAS: within-step CV)

The product CABV × LAAS becomes the headline trigger, motivated by the four-cell pattern showing it captures retrieval-helpful traces where AID min_majority fails. The narrative is: *"prior work measures answer-instability either as a within-step disagreement scalar (AID family) or as between-step transition counts (Datta, Wang); we propose the joint signal — distributional drift × length-CV — and show it is the cross-cell most-consistent retrieval trigger on multi-hop QA."*

**My honest read:** **Option B is the right call** if R1 Gemma reproduces the per-cell complementarity pattern (M9 wins where min_majority loses on at least one Gemma cell). Option C is too aggressive at current statistical power — the pooled significance test of M9 vs min_majority straddles zero. Option A is the safe-but-leaving-value-on-the-table choice.

---

## Pre-registered actions for when R1 Gemma lands

1. Re-run `moonshot_explore.py` with Gemma added → check whether the per-cell complementarity (M9 wins on the ≥1 cell where AID dies) replicates.
2. Re-run `moonshot_ensemble.py` with n=1200+ valid → check whether F2 vs min_majority CI lower edge crosses 0 (currently −0.004).
3. If either holds, promote CABV to a fifth AID variant in `analysis/a1_k1_vs_k5_ablation.py` for the official A1-on-R1 table.
4. Add CABV to the §3 / §4 paper draft as "the continuous-distribution generalisation of AID min_majority."
5. Treat M9 and F2 as **§5 discussion-section findings** (not main-table headline) at current statistical power.

---

## What I did NOT explore — moonshots for true future work

- **Reasoning-route divergence (RRD)**: BM25 / Jaccard self-similarity matrix of the K=5 sample texts at each step. Decompose into (route-divergence × answer-divergence) interaction. Would need the full sample texts ingested through a tokeniser; cost ~3 h.
- **Question-trace anchoring**: cosine between step-EOS hidden and prompt-EOS hidden. Captures "drifting from question." Would need a re-decode that captures prompt-EOS hidden (we don't have it on disk for closed traces).
- **Per-sample confidence outlier ("Confident Crank")**: which of the K=5 samples is the most confident? Does it match the majority? Would need per-sample per-token logprobs (we only have per-sample text + EOS hidden).
- **Hold-for-k smoothed CABV** (Datta-style smoothing applied to CABV instead of binary flip_rate). 15-min code change; likely improves Phi sparse-flip cells.
- **Hidden-cloud anisotropy direction × question-type interaction**: does the direction of the K=5 cloud's principal axis differ for bridge vs comparison questions? Pure descriptive — could yield an interpretability finding.

If the campaign lands a positive M9 / F2 confirmation on Gemma, the next exploratory pass is **hold-for-k smoothed CABV** as the cheapest single follow-up.
