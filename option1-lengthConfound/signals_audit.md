# `signals_audit.md` — Per-Signal Ground Truth Audit

> **Purpose.** A precise, file-cited audit of every signal in the 17-signal headline pool of the EMNLP 2026 short paper. For each signal: (1) the source paper or origin, (2) the file + line where it is computed, (3) the implementation formula, (4) what the paper currently claims about it, and (5) an honest verdict on whether the claim, citation, and code agree.
>
> **Why this exists.** A strict reviewer who reads `template.tex` will check that each signal is correctly attributed and implemented as described. This audit identifies all known mismatches before the next revision pass.
>
> **Scope.** Headline pool of 17 signals as reported in `analysis/e7_bonferroni.out` and `template.tex` App. B Table 5. Excludes the length baseline (`tokens_generated`) which is well-attributed.
>
> **Verdict counts.** Of the 17 audited signals: **6 are byte-faithful or honest-proxy implementations correctly cited**; **3 are misattributed** (CRV-family); **2 have citation gaps or mismatches** (u_cons, hvr); **3 are author-novel but not defined in main text** (CABV, CABV×LAAS, partial LAAS); **1 has a paper-code semantic discrepancy** (late_flips); **2 match paper but with honest scope/granularity adapter footnotes warranted** (etc, flare.max); the remainder match paper exactly.
>
> **Co-researcher cross-check (May 23, 2026, second pass).** Independent verification against `signal_lib.py`, `moonshot_explore.py`, `vectors.py`, `sota_capture.py`, `baseline_gates.py`, `e7_bonferroni.out`, and the Datta 2026 PDF. Four additional issues surfaced that this audit had not flagged:
> 1. **App. B Table 5 family-column for `u_cons.max` says `$K{=}5$ text` but the code is K=3** (`sota_capture.py:528`, `n_samples=3`, per Guo 2026 §3.2). New paper-code discrepancy added in §3.4.
> 2. **The code-canonical family map (`signal_lib.py:299-318`) already labels the three CRV signals `"hidden CRV"`** — it is the paper's App. B Table 5 that demotes them to `"hidden"`, *creating* the misattribution flagged in §3.2. Not an authoring oversight; a paper-side regression.
> 3. **Two of four Bonferroni-robust survivors are knife-edge:** `ent.max` Bonferroni LB = +0.001 and `min_majority` Bonferroni LB = +0.004 (`e7_bonferroni.out` rows 12 and 24). Worth a Limitations sentence; documented in §2.2 and §2.14.
> 4. **§1 ¶1's "five signal families" enumeration omits answer-trajectory entirely**, even though 5 of 9 nominal-95% survivors and 2 of 4 Bonferroni survivors are answer-trajectory/answer-velocity. The framing burden, not just the family count, is wrong (§4).
>
> Two additional refinements to existing entries: (a) §2.6 SeaKR proxy's real adapter is *uncentered Gram* vs *centered covariance* — the K-sample spectra are rank-equivalent up to centering, so the framing "Gram log-det vs SVD top-k" overstates the math gap; (b) §2.11 should not call Datta's protocol "greedy" — Datta §3.2 says "Sampling", which commits only to K=1, not to a decode mode.

---

## Table of contents

- [0. Methodology](#0-methodology)
- [1. Authoritative signal index](#1-authoritative-signal-index)
- [2. Per-signal ground truth](#2-per-signal-ground-truth)
  - [2.1 DRAGIN-RIND.max](#21-dragin-rindmax)
  - [2.2 ent.max](#22-entmax)
  - [2.3 etc (Δ²H max)](#23-etc-δh-max)
  - [2.4 flare.max (1-min p)](#24-flaremax-1-min-p)
  - [2.5 u_cons.max](#25-u_consmax)
  - [2.6 seakr_k5_proxy.logdet](#26-seakr_k5_proxylogdet)
  - [2.7 drift_cosine.max](#27-drift_cosinemax)
  - [2.8 var_l2_window.max](#28-var_l2_windowmax)
  - [2.9 hidden_norm.range](#29-hidden_normrange)
  - [2.10 hvr.max (SELFDOUBT-T1)](#210-hvrmax-selfdoubt-t1)
  - [2.11 flip_rate K=5](#211-flip_rate-k5)
  - [2.12 flip_rate K=1](#212-flip_rate-k1)
  - [2.13 late_flips](#213-late_flips)
  - [2.14 min_majority](#214-min_majority)
  - [2.15 CABV (M1)](#215-cabv-m1)
  - [2.16 FCT (M8)](#216-fct-m8)
  - [2.17 CABV × LAAS (M9)](#217-cabv--laas-m9)
- [3. Cross-cutting issues](#3-cross-cutting-issues)
  - [3.1 Citation mismatches](#31-citation-mismatches)
  - [3.2 Misattributed author-designed signals](#32-misattributed-author-designed-signals)
  - [3.3 Paper coverage gaps](#33-paper-coverage-gaps)
  - [3.4 Paper-code discrepancies](#34-paper-code-discrepancies)
  - [3.5 Honest scope/granularity adapters](#35-honest-scopegranularity-adapters)
- [4. Implications for family-count claim](#4-implications-for-family-count-claim)
  - [4.1 Framing burden — §1 ¶1 hides the answer-trajectory family entirely](#41-framing-burden--1-1-hides-the-answer-trajectory-family-entirely)
- [5. Recommended honest pool](#5-recommended-honest-pool)
- [6. File location index](#6-file-location-index)

---

## 0. Methodology

For each signal in `analysis/e7_bonferroni.out` (the 17-signal Bonferroni-corrected paired-bootstrap report on the headline pool):

1. **Origin search.** Located the signal's computation in `emnlp_26_lengthConfound/analysis/` (aggregation layer) and `emnlp_26_lengthConfound/scripts/crv/` (trace-generation layer).
2. **Source paper attribution.** Identified the published peer paper (if any) that introduces the underlying primitive. Cross-referenced against `__lit-review-main/`, `litReview/-rq1-litReview-sources.csv`, `litReview/-rq1-v2-litReview-sources.csv`, and `_rq1-agent.md`.
3. **Formula comparison.** Compared the code's computation against the source paper's stated formula (where applicable) and against the paper's §3.3 main-text description (for novel signals).
4. **Citation cross-check.** Verified that `template.tex` §1 ¶1 / §2 cites the correct source for each audited signal.

Audit performed against:
- `option1-lengthConfound/template.tex` (current revision, post-Round-3-cascade).
- `emnlp_26_lengthConfound/analysis/signal_lib.py` (319 lines; authoritative for analysis-layer formulas).
- `emnlp_26_lengthConfound/analysis/moonshot_explore.py` (389 lines; authoritative for novel signal formulas).
- `emnlp_26_lengthConfound/scripts/crv/sota_capture.py` (876 lines; authoritative for SOTA baseline implementations).
- `emnlp_26_lengthConfound/scripts/crv/baseline_gates.py` (referenced for SELFDOUBT-T1).
- `emnlp_26_lengthConfound/scripts/crv/vectors.py` (CRVTracker, authoritative for the CRV-family signals).
- `emnlp_26_lengthConfound/analysis/e7_bonferroni.out` (authoritative headline-pool report).

---

## 1. Authoritative signal index

The 17-signal headline pool exactly as it appears in `analysis/e7_bonferroni.out` (and matching `template.tex` App. B Table 5):

| Row | Signal name (in output) | Status (✓ / ⚠️ / ❌) | Issue type |
|---|---|---|---|
| 1 | `DRAGIN-RIND.max` | ✓ | none — byte-faithful |
| 2 | `ent.max` | ✓ | generic baseline |
| 3 | `etc (Δ²H max)` | ⚠️ | granularity adapter (token→step) |
| 4 | `flare.max (1-min p)` | ⚠️ | scope adapter (online→trace-summary) |
| 5 | `u_cons.max` | ❌ | citation mismatch (RSUS, not SelfCheckGPT) + App. B family-column label `$K{=}5$ text` should be `$K{=}3$` |
| 6 | `seakr_k5_proxy.logdet` | ⚠️ | acknowledged proxy (K=5 vs K=20; full log-det vs top-k EigenScore) |
| 7 | `drift_cosine.max` | ❌ | misattributed (CRV, not peer-pool) |
| 8 | `var_l2_window.max` | ❌ | misattributed (CRV, not peer-pool) |
| 9 | `hidden_norm.range` | ❌ | misattributed (CRV, not peer-pool) |
| 10 | `hvr.max` | ❌ | citation gap (Pandey 2026 SELFDOUBT-T1 not in paper) |
| 11 | `flip_rate K=5` | ✓ | matches paper |
| 12 | `flip_rate K=1` | ✓ | matches paper (Datta protocol) |
| 13 | `late_flips` | ❌ | paper-code semantic discrepancy |
| 14 | `min_majority` | ✓ | matches paper |
| 15 | `CABV` | ❌ | not defined in §3.3 |
| 16 | `FCT` | ✓ | matches paper §3.3 Eq. 3 |
| 17 | `CABV × LAAS` | ❌ | not defined in §3.3 (LAAS standalone not in pool) |

---

## 2. Per-signal ground truth

### 2.1 `DRAGIN-RIND.max`

**Source paper.** Su et al. 2024, *DRAGIN: Dynamic Retrieval Augmented Generation based on the Real-time Information Needs of Large Language Models*. arXiv:2403.10081v3. ACL 2024 main.

**Trace-generation file.** `emnlp_26_lengthConfound/scripts/crv/sota_capture.py:147-215` — class `RindAccumulator`. Docstring at L7-13 explicitly cites the paper.

**Analysis aggregation file.** `emnlp_26_lengthConfound/analysis/signal_lib.py:217` via `moonshot_explore.py` `tmax(c, "dragin_rind_max")`.

**Code formula.** Per DRAGIN paper §3.2:
- `RIND_i = H_i · a_max(i) · s_i`
- `H_i` = next-token entropy (Shannon, nats; computed at decode in `compute_token_entropy_nats`).
- `a_max(i)` = max attention received by generated-token `i` from any later token `j > i` in the last full-attention layer (accumulated online via `ingest_attention_row`).
- `s_i` = stopword filter ∈ {0,1}, computed via spaCy stopword set (`_SPACY_STOPWORDS`, ~330 entries) + punctuation filter.
- Per-step `rind_max` = `max` over content tokens (s_i = 1) within the step.
- Trace-level audit value = `max` over reasoning steps of per-step `rind_max`.

**Paper claim.** `template.tex:121` (§1 ¶1): "attention-based DRAGIN \citep{su_dragin_2024} (token importance from attention dynamics)". `template.tex:138` (§2): "attention-based DRAGIN \citep{su_dragin_2024}".

**Verdict.** ✓ **Byte-faithful per-token formula.** Stopword set, attention-receiver convention, and entropy units all match the DRAGIN paper §3.2. Only adaptation: we compute a **trace-level summary statistic** (`max` over reasoning steps) for AUROC evaluation; DRAGIN itself is an **online per-token trigger**. This adaptation is covered by the paper's "signal-level audit" scope statement in Limitations (template.tex:264).

**Honest framing footnote (not yet in paper):** none needed; current framing is correct.

---

### 2.2 `ent.max`

**Source paper.** None specific — generic mean-token-entropy baseline.

**Trace-generation file.** Mean token entropy is captured during decoding in `scripts/04_run_qa.py` (referenced from `sota_capture.py:55-65`: "matches `_entropy_nats` in `scripts/04_run_qa.py`").

**Analysis aggregation file.** `signal_lib.py:304` (entry `ent_max` → display `ent.max`). Computed as `tmax(c, "mean_token_entropy")`.

**Code formula.** Per step: `mean_token_entropy` = mean Shannon entropy in nats over the step's tokens. Trace-level audit value = `max` over reasoning steps.

**Paper claim.** Not cited to a specific paper. Implicitly part of the FLARE / token-probability family per `template.tex:121` (§1 ¶1) — "token-probability FLARE \citep{jiang_active_2023} (**next-token entropy** / max log-prob at the decode head)".

**Verdict.** ✓ **OK as a generic baseline within the FLARE / token-probability family.** No method paper to cite. Worth one sentence in §3 acknowledging this is a generic single-step entropy baseline within the family — it is not a separate "method", just the FLARE family's per-step entropy aggregation.

**⚠️ Strength caveat (knife-edge Bonferroni LB).** Per `e7_bonferroni.out` row 12: `ent.max` Bonferroni 99.67% CI = `[+0.001, +0.263]`. **Lower bound = +0.001.** This is the most fragile of the four Bonferroni-robust survivors — any small change to `B`, the bootstrap seed, the residualization parameterization, or the sample exclusion criterion could push it below zero. Worth a Limitations sentence acknowledging the knife-edge status, or re-running with `B=50000` and reporting the tighter interval. The headline "four Bonferroni-robust survivors" claim does not fail without `ent.max`, but the §1 contribution-bullet count would drop to three.

---

### 2.3 `etc (Δ²H max)`

**Source paper.** Li et al. 2026, *ETC: Entropy-Trend Constraint*. arXiv:2511.09980 (AAAI 2026 per `_AUDIT-REPORT.md` and `litReview/-rq1-litReview-sources.csv`).

**Trace-generation file.** Source data is per-step `mean_token_entropy` captured during decoding (same source as `ent.max`).

**Analysis aggregation file.** `signal_lib.py:69-81` — `etc_score(r)`. Inline docstring (L70-76): "ETC (Li et al. 2025): max absolute second-difference of step entropy. Δ²H_i = H_{i+1} − 2 H_i + H_{i−1}".

**Code formula.** `max_i |H_{i+1} - 2*H_i + H_{i-1}|` over step-level mean entropies `H`.

**Paper claim.** `template.tex:121` (§1 ¶1): "entropy-trend ETC \citep{li_modeling_2026} (second-difference of step entropy)". `template.tex:138` (§2): "entropy-trend ETC \citep{li_modeling_2026}".

**Verdict.** ⚠️ **Honest granularity adapter.** ETC paper operates on **token-level** entropy stream; our code operates on **step-level mean** entropy. The paper §1 already says "second-difference of **step** entropy" — so the claim is internally consistent with the code. **No further action needed**, but a one-line §3 footnote acknowledging the step-vs-token granularity adaptation would pre-empt strict-reviewer attack.

---

### 2.4 `flare.max (1-min p)`

**Source paper.** Jiang et al. 2023, *Active Retrieval Augmented Generation (FLARE)*. arXiv:2305.06983, EMNLP 2023 main.

**Trace-generation file.** Source data is `per_token_max_logprob_alt` (pre-repetition-penalty distribution) and `per_token_max_logprob` (post-penalty) captured during decoding.

**Analysis aggregation file.** `signal_lib.py:89-104` — `flare_max_per_step(r)`.

**Code formula.** Per step: `1 - min(per_token_max_logprob_alt OR per_token_max_logprob)` (uses the pre-repetition-penalty distribution `_alt` when available, falls back otherwise). Trace-level audit value = `max` over reasoning steps.

**Paper claim.** `template.tex:121` (§1 ¶1): "token-probability FLARE \citep{jiang_active_2023} (next-token entropy / max log-prob at the decode head)".

**Verdict.** ⚠️ **Honest scope adapter.** FLARE paper specifies an **online per-token trigger**: retrieve when `max_t p_t < τ` for some threshold τ. Our code computes a **trace-level summary statistic** (`max` over steps of `1 - min p` per step) for AUROC evaluation. We test FLARE's *primitive* (low-confidence detection) but not FLARE's *triggering policy*. Already covered by Limitations "signal-level audit, not end-to-end pipelines" (template.tex:264). **No further action needed**, but one §3 footnote acknowledging the online→trace-summary adapter would pre-empt strict-reviewer attack.

---

### 2.5 `u_cons.max`

**Source paper.** Guo et al. 2026, *ReaLM-Retrieve* (RSUS scoring framework). Citation key not yet in `template.tex` (`_rq1-agent.md` references arXiv:2604.XXXXX placeholder).

**❌ NOT SelfCheckGPT.** The paper §1 currently cites Manakul 2023 SelfCheckGPT as the multi-sample text-consistency family — but the implemented `u_cons.max` signal is from Guo 2026 RSUS, not Manakul 2023.

**Trace-generation file.** `emnlp_26_lengthConfound/scripts/crv/sota_capture.py:508-705` — class `ConsistencySampler`. Docstring at L509-522: "Guo et al. §3.2: sample K=3 continuations of ~40 tokens at T=0.6, top-p=0.9 from the current state. `U_cons` is `1 - mean pairwise cosine similarity` between the continuations' sentence-BERT embeddings."

**Code formula.**
1. At each step boundary, fork the KV cache.
2. Sample K=3 continuations of ~40 tokens at temperature 0.6, top-p 0.9 (per Guo et al. §3.2).
3. Embed each continuation with `sentence-transformers/all-MiniLM-L6-v2` (L2-normalized).
4. `mean_pairwise_cosine` = mean of `dot(emb_i, emb_j)` for all i < j.
5. `u_cons = max(0, min(1, 1 - mean_pairwise_cosine))`.
6. Restore KV cache to pre-fork sequence length.
7. Trace-level audit value = `max` over reasoning steps.

**SelfCheckGPT's actual formula (for comparison).** Manakul et al. 2023 samples K full responses and computes BERTScore / n-gram / LLM-judge agreement at the **final-answer** level. Different mechanism, different granularity (response-level, not step-boundary), and different paper.

**Paper claim.** `template.tex:121` (§1 ¶1): "**multi-sample text-consistency SelfCheckGPT (Manakul, 2023)** (cross-sample answer agreement on the generated text)". `template.tex:138` (§2): "multi-sample text-consistency **SelfCheckGPT (Manakul 2023) and SAC³ (Zhang 2023)**".

**Verdict.** ❌ **CITATION MISMATCH.** The implemented `u_cons.max` is RSUS U_cons (Guo 2026) — a step-boundary K=3 continuation-consistency signal — not SelfCheckGPT's response-level multi-sample text consistency. Both are conceptually "multi-sample text consistency" but the **formula, granularity, and source paper differ**.

**Action options.**
- **(a)** Re-cite §1 ¶1 to Guo 2026 RSUS instead of SelfCheckGPT, and acknowledge u_cons as RSUS-derived.
- **(b)** Add a separate SelfCheckGPT implementation to the audit pool (extra work).
- **(c)** Drop `u_cons.max` from the headline pool.
- **Recommended:** (a). Most honest and lowest cost.

---

### 2.6 `seakr_k5_proxy.logdet`

**Source papers.** Yao et al. 2024 *SeaKR: Self-aware Knowledge Retrieval* (ACL 2025 main, arXiv:2406.19215) AND Chen et al. 2024 *INSIDE* (ICLR 2024 spotlight, arXiv:2402.03744). **One signal proxying both.**

**Trace-generation file.** K=5 sample EOS hidden states captured per step boundary in the SOTA capture pipeline.

**Analysis aggregation file.** `signal_lib.py:107-134` — `seakr_k5_logdet(r, alpha=1e-3)`.

**Code formula.**
1. Per step, collect K=5 sample EOS hidden states `H = [h_1, …, h_K]`.
2. `M = stack(H)` shape `K × d`.
3. Gram matrix `G = M @ M.T + α·I` with α = 1e-3 (ridge regularizer).
4. `slogdet(G)` → `(sign, logdet)`; keep `logdet` if `sign > 0` and finite.
5. Trace-level audit value = `max` over reasoning steps of per-step `logdet`.

**SeaKR's actual formula.** Yao et al. 2024 uses K=20 samples and computes the SVD-based EigenScore = `(1/k) sum log λ_i` over the **top-k** eigenvalues of the SVD-derived covariance.

**INSIDE's actual formula.** Chen et al. 2024 computes EigenScore on the covariance matrix of K samples' last-layer hidden states.

**Code's variable name says `_proxy`.** Explicit acknowledgement that this is NOT byte-faithful to either paper.

**Paper claim.** `template.tex:121` (§1 ¶1): "hidden-state SeaKR \citep{yao_seakr_nodate} (internal-state covariance across K samples)". `template.tex:138` (§2): "hidden-state INSIDE \citep{chen_inside_2024} and SeaKR \citep{yao_seakr_nodate}".

**Verdict.** ⚠️ **Acknowledged proxy doing double duty.**
- **Adaptation 1 (sample budget):** K=5 not K=20.
- **Adaptation 2 (math, refined):** uncentered Gram log-det vs centered covariance EigenScore. For K samples in d-dim with K ≪ d, the K nonzero eigenvalues of `M @ M.T` (Gram) and `M.T @ M / K` (sample covariance) coincide; AUROC of `slogdet(M @ M.T)` vs `(1/K) Σ log λ_i` is **rank-equivalent up to centering**. The honest gap is therefore **mean-centering** (we don't), not "Gram vs SVD". The "top-k of K=20" choice in SeaKR collapses to "all-K of K=5" trivially, since K=5 has only 5 nonzero eigenvalues.
- **One-signal-two-papers:** the paper §2 names INSIDE and SeaKR as two distinct methods; we audit one signal that approximates both.

**Action options.**
- **(a)** One §3 or App. F footnote: "We compute a single uncentered Gram log-det on K=5 sample EOS hidden states. The K nonzero spectrum is rank-equivalent to a covariance EigenScore of the same samples; the honest adapters are (i) K=5 vs K=20 sample budget and (ii) no mean-centering. The signal jointly approximates SeaKR (Yao 2024) and INSIDE (Chen 2024)."
- **(b)** Implement K=20 and centered EigenScore separately (extra compute).
- **Recommended:** (a). One sentence pre-empts the attack.

---

### 2.7 `drift_cosine.max`

**Source paper.** ❌ **None.** This signal is from **this paper's own CRV (Concept Representation Vector) framework**, not from any cited peer-pool paper.

**Trace-generation file.** `emnlp_26_lengthConfound/scripts/crv/vectors.py:31-184` — class `CRVTracker`. Docstring at L1-21: "Concept Representation Vector tracker (Phase 2.3). Per logical step, accumulate the per-token hidden states (one per model.forward call within that step), mean-pool to produce a single CRV vector, and compute the drift-and-variance signals that downstream gating uses."

**Trigger logic file.** `emnlp_26_lengthConfound/scripts/crv/triggers.py:28-95` — class `CRVTrigger`.

**Analysis aggregation file.** `signal_lib.py:308` — entry `drift_max` → display `drift_cosine.max`. Computed as `tmax(c, "drift_cosine")`.

**Code formula.**
1. Per step, mean-pool per-token hidden states across selected layer subset → per-step CRV vector.
2. `drift_t = 1 - cos_sim(CRV_t, CRV_{t-1})`, range [0, 2].
3. Trace-level audit value = `max` over reasoning steps.

**Paper claim.** `template.tex` does NOT cite this signal to any source paper. App. B Table 5 (template.tex:344) labels family `hidden`, implicitly grouping it with the SeaKR/INSIDE family.

**Verdict.** ❌ **MISATTRIBUTED.** The signal is presented in the audit as if it is part of the published "hidden-state" trigger family (SeaKR/INSIDE-style) but is in fact **this paper's own CRV-framework diagnostic**. A strict reviewer who reads `vectors.py` will see this is the author's design, not a peer-pool baseline.

**Action options.**
- **(a)** Drop `drift_cosine.max` from the headline pool entirely. Cleanest.
- **(b)** Re-label in App. B Table 5 family column as "auxiliary CRV-derived (this paper)" and add one §3 sentence: "We additionally include three auxiliary hidden-state diagnostics from our CRV framework (drift, variance, norm-range) as exploratory baselines beyond the SeaKR-proxy."
- **Recommended:** (b) if drift_cosine.max appears anywhere as a clean negative result (it does — App. B Table 5 row, Δ = +0.020, not significant); (a) if you don't need it for the negative-finding narrative.

---

### 2.8 `var_l2_window.max`

**Source paper.** ❌ **None.** Same as `drift_cosine.max` — from this paper's CRV framework.

**Trace-generation file.** `emnlp_26_lengthConfound/scripts/crv/vectors.py` — same `CRVTracker`. Rolling variance computation managed via `_norm_history` deque (L74, `window_size=10`).

**Trigger logic file.** `emnlp_26_lengthConfound/scripts/crv/triggers.py:25-26` — `_VARIANCE_KEY = "variance_l2_window"`.

**Analysis aggregation file.** `signal_lib.py:309` — entry `var_max` → display `var_l2_window.max`. Computed as `tmax(c, "variance_l2_window")`.

**Code formula.**
1. Per step, compute `||CRV_t||_2` (L2 norm of the CRV vector).
2. Maintain rolling deque of last `window_size=10` step L2 norms.
3. `variance_l2_window_t` = variance of the rolling window at step `t`.
4. Trace-level audit value = `max` over reasoning steps.

**Paper claim.** Same as `drift_cosine.max` — not cited to a source paper; App. B Table 5 (template.tex:346) labels family `hidden`.

**Verdict.** ❌ **MISATTRIBUTED.** Same issue as `drift_cosine.max`. Author's CRV signal presented as if peer-pool.

**Action options.** Same as 2.7.

---

### 2.9 `hidden_norm.range`

**Source paper.** ❌ **None.** Same as the other CRV signals.

**Trace-generation file.** Derived from CRV norm series (`vectors.py`).

**Analysis aggregation file.** `signal_lib.py:310` — entry `hidden_range` → display `hidden_norm.range`. Computed as `trange(c, "hidden_norm")`.

**Code formula.** `max_t ||CRV_t||_2 - min_t ||CRV_t||_2` over the trace (function `trange` at `signal_lib.py:62-66`).

**Paper claim.** Same as `drift_cosine.max` — not cited; App. B Table 5 (template.tex:343) labels family `hidden`.

**Verdict.** ❌ **MISATTRIBUTED.** Author's CRV signal presented as if peer-pool.

**Action options.** Same as 2.7.

---

### 2.10 `hvr.max (SELFDOUBT-T1)`

**Source paper.** Pandey et al. 2026, *SELFDOUBT* (arXiv:2604.06389v1). Cited explicitly in `scripts/crv/baseline_gates.py:5-9` and `sota_capture.py:21-23`.

**❌ NOT CITED IN `template.tex`.** Neither §1 ¶1 nor §2 mentions Pandey et al. 2026 or the SELFDOUBT framework. The signal appears in App. B Table 5 (template.tex:342) with family `text` and the parenthetical "(SELFDOUBT-T1)" in the `display` name, but no actual citation.

**Trace-generation file.** `emnlp_26_lengthConfound/scripts/crv/baseline_gates.py:508-565` — class `SelfDoubtTier1Gate`. Docstring at L508-528 explicitly cites Pandey et al. 2026.

**Driver file.** `emnlp_26_lengthConfound/scripts/04f_run_qa_sota.py:331` computes `hvr_val = h_count / (v_count + 1)` per step.

**Analysis aggregation file.** `signal_lib.py:311` — entry `hvr_max` → display `hvr.max (SELFDOUBT-T1)`. Computed as `hvr_max(r)` defined at `signal_lib.py:84-86`: `max` over steps of trace's `hvr` field values.

**Code formula.** Per SELFDOUBT paper §3:
- `HVR_t = h_t / (v_t + 1)`
- `h_t` = count of hedge marker words in step `t` text (e.g., "maybe", "possibly", "unsure"; full dictionary from Pandey et al. seeded in `baseline_gates.py:458-501`).
- `v_t` = count of verify marker words in step `t` text (e.g., "definitely", "certain", "confirmed").
- Trace-level audit value = `max` over reasoning steps.

**Adaptation note (acknowledged in code, L521-528).** Pandey et al. paper computes HVR over a **complete** trace; we compute it **per step**. Adaptation documented in code; not currently documented in paper.

**Paper claim.** ❌ **No citation.** Signal appears in App. B Table 5 as if it's a generic "text" family signal without identifying its source.

**Verdict.** ❌ **CITATION GAP.** Real signal from a real published paper, byte-faithfully implemented to the SELFDOUBT-T1 spec (with documented per-step vs per-trace adaptation), but template.tex never cites Pandey et al. 2026.

**Action options.**
- **(a)** Cite Pandey et al. 2026 in §1 ¶1 (add SELFDOUBT-T1 to the multi-sample text-consistency family, OR add a "hedging-marker" family) and in §2 Related Work.
- **(b)** Drop `hvr.max` from the audit pool.
- **(c)** Re-label as a generic hedging-marker baseline without source attribution (NOT recommended — strict reviewer can grep the codebase and find the citation).
- **Recommended:** (a). The signal is byte-faithful and the citation is trivial to add.

---

### 2.11 `flip_rate K=5`

**Source paper.** Datta et al. 2026, *Large Language Models Decide Early and Explain Later* (ICML 2026, arXiv:2604.22266). Primitive: `AnswerSwitches`(T) and `t*` over forced-answer-completion trajectories.

**This-paper extension.** Datta uses K=1 forced-answer-completion (Datta §3.2: "Sampling the model conditioned on this prompt produces an intermediate answer A_i" — single-sample, decode mode unspecified, **not stated as greedy in the paper**). We extend with K=5 majority-sampling at each step boundary.

**Trace-generation file.** K=5 sample answers per step boundary captured in `scripts/04_run_qa.py` via boundary detection (`scripts/crv/boundaries.py`) + per-step K=5 sampling.

**Analysis aggregation file.** `signal_lib.py:168-177` — `flip_rate_majority(per_step)`.

**Code formula.**
1. Build sequence `seq` = list of per-step K=5 majority answers (skip empty steps).
2. `flips` = `sum(1 for i in 1..n-1 if seq[i] != seq[i-1])`.
3. Return `flips / max(1, len(seq) - 1)`.

**Paper claim.** `template.tex:168` (§3.3): "flip_rate: fraction of consecutive-step majority disagreements". Matches code.

**Verdict.** ✓ **Matches paper. Datta primitive K=5-extended.**

---

### 2.12 `flip_rate K=1`

**Source paper.** Datta et al. 2026 (Datta's exact K=1 protocol — single-sample forced-answer-completion at each step; Datta §3.2 specifies "Sampling" without committing to greedy decoding).

**Trace-generation file.** Same K=5 capture; this signal uses only the first sample (`ans[0]`).

**Analysis aggregation file.** `signal_lib.py:180-189` — `flip_rate_single(per_step)`.

**Code formula.** Identical to `flip_rate K=5` but uses `ans[0]` (single sample per step) instead of K=5 majority.

**Paper claim.** Not explicitly defined in §3.3 (line 177 mentions "K=1 vs K=5 ablation" — implies the K=1 version is the obvious single-sample analogue).

**Verdict.** ✓ **Matches Datta's K=1 protocol exactly.** App. B Table 5 (template.tex:336) family `answer-trajectory` is correct.

---

### 2.13 `late_flips`

**Source.** This paper (variant of flip_rate, restricted to last half of trace).

**Trace-generation file.** Same K=5 capture as flip_rate.

**Analysis aggregation file.** `signal_lib.py:192-205` — `late_flips_count(per_step, n_steps_total)`.

**Code formula.**
1. Build `answer_steps = [(i, set(ans)) for i, ans in enumerate(per_step) if ans]`. **Note: uses `set(ans)`, NOT `Counter.most_common(1)`.**
2. `mid = n_steps_total // 2`.
3. `late = [(i, s) for i, s in answer_steps if i >= mid]`.
4. Return `float(sum(1 for j in 1..len(late)-1 if late[j][1] != late[j-1][1]))`. **Note: raw COUNT, not divided by `len(late) - 1`.**

**Paper claim.** `template.tex:169` (§3.3): "late_flips: **same** [as flip_rate], restricted to the last half of the trace."

**Verdict.** ❌ **PAPER-CODE SEMANTIC DISCREPANCY (two divergences):**

1. **Set vs majority.** Code compares `set(ans)` per step (any K=5 sample change between adjacent steps); paper "same as flip_rate" implies majority comparison. Set is strictly looser — every majority-flip is also a set-flip, but not vice versa.
2. **Count vs fraction.** Code returns raw count; paper "same as flip_rate" implies fraction `flips / (len - 1)`.

**Practical impact.** AUROC is rank-invariant, so count-vs-fraction does not change rankings. The set-vs-majority semantic IS a real difference — at the same trace, set-based late_flips will be ≥ majority-based late_flips.

**Action options.**
- **(a)** Update §3.3 line 169 to describe the actual code behavior: "late_flips: count of consecutive-step K=5 sample-set disagreements over the last half of the trace." (Recommended — code behavior is arguably what was intended: sensitive to ANY K=5 sample shift, not just majority shifts.)
- **(b)** Update CODE to match §3.3 (use majority comparison, divide by `len(late) - 1`). Would change `late_flips` Δ in `e7_bonferroni.out` — re-run pipeline.
- **(c)** Drop `late_flips` from the headline pool (it's not Bonferroni-robust anyway; Δ = +0.081 [-0.016, +0.175], 95% CI straddles 0).
- **Recommended:** (a). Cheapest, paper-side fix.

---

### 2.14 `min_majority`

**Source.** This paper (within-step K=5 sample dispersion).

**Trace-generation file.** Same K=5 capture.

**Analysis aggregation file.** `signal_lib.py:208-218` — `min_majority_inv(per_step)`.

**Code formula.**
1. Per step, compute `c / len(ans)` where `c` = count of K=5 majority answer.
2. Return `1.0 - min(c / len(ans))` over populated steps.

**Paper claim.** `template.tex:170` (§3.3): "`min_majority`: 1 − min_t (max_count/K), the worst within-step disagreement."

**Verdict.** ✓ **Matches paper formula.**

**⚠️ Strength caveat (knife-edge Bonferroni LB).** Per `e7_bonferroni.out` row 24: `min_majority` Bonferroni 99.67% CI = `[+0.004, +0.284]`. **Lower bound = +0.004.** Together with `ent.max` (LB = +0.001), this is one of two knife-edge survivors. Same recommendation as §2.2: re-run with larger `B` to tighten, and acknowledge fragility in Limitations. The "four Bonferroni-robust signals across four families" claim is structurally fragile to either of these two losing significance under a stricter bootstrap.

---

### 2.15 `CABV` (M1)

**Source.** This paper, NOVEL. Not in any published peer paper.

**Trace-generation file.** K=5 per-step sample answers (same as flip_rate's source).

**Analysis aggregation file.** `emnlp_26_lengthConfound/analysis/moonshot_explore.py:75-109` — `cabv_score(steps, k_top=5, eps=1e-3)`.

**Code formula.**
1. Build per-step K=5 answer list.
2. Union of all answers ever seen in trace → universe of size `K_uni`.
3. Per populated step `t`: compute Laplace-smoothed empirical distribution `p_t` over the universe (`eps = 1e-3` added uniformly).
4. Mean Jensen-Shannon divergence between consecutive populated steps: `mean(JSD(p_t, p_{t+1}))`.
5. Trace-level value = the mean JSD itself.

**Docstring (`moonshot_explore.py:12-16`):** "**CABV — Continuous Answer-Belief Velocity.** Per step → K=5 answer empirical distribution p_t (Laplace-smoothed over the union of answers seen in this trace) → mean JSD(p_t, p_{t+1}) over consecutive populated steps. Generalises the binary flip_rate edge-count to a continuous between-step measure."

**Paper claim.** ❌ **NOT DEFINED IN §3.3.** Appears in App. B Table 5 (template.tex:333) under family `answer-velocity` with `Δ = +0.137`, and in App. F Spearman table (template.tex:458). Never introduced or defined in main text.

**Verdict.** ❌ **PAPER COVERAGE GAP.** A strict reviewer reading App. B Table 5 will see "CABV: answer-velocity, Δ = +0.137 [+0.041, +0.229]" — significant at 95% — and ask "What is CABV?"

**Action options.**
- **(a)** Add a 2-3 sentence definition of CABV to §3.3 (right after `min_majority`). Then `flip_rate K=5` is the discrete edge-count primitive, `CABV` is the continuous JSD generalization. Adds ~30 words.
- **(b)** Remove CABV from App. B Table 5 + App. F. Loses one 95%-survivor from the pool.
- **Recommended:** (a). CABV is empirically a 95% survivor and worth defining.

---

### 2.16 `FCT` (M8)

**Source.** This paper. Forward-looking variant of Datta 2026's `t*` primitive.

**Trace-generation file.** K=5 per-step sample answers + per-trace final answer (extracted via regex `ANSWER_RE = r"\bAnswer\s*:\s*(.+?)(?:[\n.]|$)"`).

**Analysis aggregation file.** `emnlp_26_lengthConfound/analysis/moonshot_explore.py:141-169` — `fct_score(steps, final_ans, k_top=5)`.

**Code formula.**
1. Build per-step K=5 majority answers `majorities = [(i, m)]`.
2. `n = len(majorities)`.
3. `t_commit = n - 1` (default: committed only at last step).
4. For `k` in `0..n-1`: if all of `majorities[k:]` have majority = `final_ans_n`, set `t_commit = k` and break.
5. Return `t_commit / (n - 1)` if `n > 1`, else `0.0`.
6. **Edge case:** if no step matches `final_ans`, `t_commit` stays at `n - 1`, return `1.0`.

**Paper claim.** `template.tex:172-174` (§3.3, Eq. 3): "FCT = min{i : m_j = a_final ∀ j ≥ i} / max(1, n - 1) ∈ [0, 1]".

**Verdict.** ✓ **Matches paper Eq. 3.** Edge case (no match → 1.0) not yet documented in §3.3; the `_AUDIT-REPORT.md` (line 514-522) already flagged this as a one-footnote fix.

**Adaptation note.** The paper describes FCT as a "forward-looking variant of the answer-trajectory primitive (Datta 2026 `t*`)." Datta's `t*` is backward-looking (the **last** answer-switch index); FCT is forward-looking (the **first** stable-commitment step). The relationship is documented in App. E (template.tex:410-429).

---

### 2.17 `CABV × LAAS` (M9)

**Source.** This paper, NOVEL composite. Not in any published peer paper.

**Component 1 (CABV)** — see [§2.15](#215-cabv-m1).

**Component 2 (LAAS — "Length-Asymmetry Across Samples").** Not in App. B Table 5 standalone. Docstring (`moonshot_explore.py:18-21`): "Per step → coefficient of variation std(len)/mean(len) of K=5 sample lengths; max over steps. Scale-invariant by construction."

**Trace-generation file.** K=5 sample lengths captured per step (`n_tokens` field per sample).

**Analysis aggregation file.** `moonshot_explore.py:112-138` — `laas_score(steps)`.

**LAAS code formula.**
1. Per step, collect K=5 sample lengths.
2. `cv = std(lengths) / mean(lengths)` (population std, ddof=0).
3. Trace-level LAAS = `max` over steps of per-step `cv`.

**CABV × LAAS code formula.** `cabv * laas` (simple product). Returns `None` if either component is `None`.

**Paper claim.** ❌ **NOT DEFINED IN §3.3.** Appears in App. B Table 5 (template.tex:339) under family `answer-velocity` with `Δ = +0.089`, 95% CI straddles 0 — not significant. Never introduced or defined in main text. **LAAS standalone does not appear in App. B Table 5 at all** (only its product with CABV).

**Verdict.** ❌ **PAPER COVERAGE GAP.** Same as CABV — appears in App. B Table 5 but never defined in main text. Additionally, LAAS itself is invisible (only the composite is reported).

**Action options.**
- **(a)** Add 2-3 sentences to §3.3 defining LAAS and the CABV × LAAS composite. Adds ~40 words.
- **(b)** Remove CABV × LAAS from App. B Table 5. (Not Bonferroni-robust, 95% straddles 0 — easy to drop.)
- **Recommended:** (b). It's not a positive result, defining it just to show a negative finding isn't worth the §3.3 word budget.

---

## 3. Cross-cutting issues

### 3.1 Citation mismatches

| Signal | Currently cited in paper as | Actual implementation source |
|---|---|---|
| `u_cons.max` | Manakul 2023 (SelfCheckGPT), §1 ¶1 + §2 | Guo et al. 2026 (ReaLM-Retrieve / RSUS U_cons) |
| `hvr.max` | Not cited at all | Pandey et al. 2026 (SELFDOUBT-T1) |

Both are real published peer papers with byte-faithful (or near-byte-faithful) implementations in `scripts/crv/sota_capture.py` and `scripts/crv/baseline_gates.py`. The fix is paper-side citation updates.

### 3.2 Misattributed author-designed signals

Three signals from `scripts/crv/vectors.py` (the author's own Concept Representation Vector framework) are presented in App. B Table 5 under family `hidden` as if they were tested peer-published hidden-state methods:

- `drift_cosine.max` (App. B Table 5 row at template.tex:344)
- `var_l2_window.max` (App. B Table 5 row at template.tex:346)
- `hidden_norm.range` (App. B Table 5 row at template.tex:343)

None of them is from SeaKR, INSIDE, or any other paper cited in §2. They are **author-designed CRV diagnostics**.

A strict reviewer who reads `vectors.py` will see this. Treatment options:
- **Drop from headline pool** (cleanest), OR
- **Relabel in App. B Table 5 family column** as "auxiliary CRV (this paper)" and add one §3 acknowledgement sentence

**Note: this is a paper-side regression, not a code-side oversight.** The code-canonical family map at `analysis/signal_lib.py:308-310` already labels these three signals as `"hidden CRV"`:

```
("drift_max",      "drift_cosine.max",  "hidden CRV"),
("var_max",        "var_l2_window.max", "hidden CRV"),
("hidden_range",   "hidden_norm.range", "hidden CRV"),
```

The misattribution arose during paper-writing when App. B Table 5 (`template.tex:343-346`) demoted `"hidden CRV"` to plain `"hidden"`, collapsing the author-novel signals into the same family bucket as the SeaKR/INSIDE peer pool. The fix is therefore to **restore App. B Table 5's family column to match the code** — either `"hidden (CRV, this paper)"` or `"auxiliary CRV"` is consistent with `signal_lib.py`. The `seakr_k5_proxy.logdet` row at `template.tex:345` already uses `$K{=}5$ hidden` (a peer-pool label), so a clean three-way distinction in App. B is achievable: peer-pool hidden (seakr), CRV (drift/var/range), and external/length (tokens).

### 3.3 Paper coverage gaps

Two signals appear in App. B Table 5 / App. F but are never defined in main text (§3.3):

- `CABV` (App. B Table 5 row at template.tex:333; App. F at template.tex:458) — **95% paired-bootstrap survivor**, Δ = +0.137
- `CABV × LAAS` (App. B Table 5 row at template.tex:339) — **not significant**, Δ = +0.089

Plus `LAAS` standalone never appears anywhere — its product is in App. B but the component itself is not.

`hvr.max` also appears in App. B Table 5 (template.tex:342) without a §1/§2 citation to Pandey 2026 (covered by [§3.1](#31-citation-mismatches)).

### 3.4 Paper-code discrepancies

**`late_flips` (signal 13).** Two divergences between code and §3.3 line 169:

| Aspect | §3.3 says | Code does |
|---|---|---|
| Per-step comparison | "majority disagreements" (implied by "same as flip_rate") | `set(ans)` comparison |
| Return type | Fraction (implied by "same as flip_rate") | Raw count |

Practical impact: rank-invariant (no AUROC change), but semantically real (set is strictly looser than majority).

**`u_cons.max` family-column label (signal 5).** App. B Table 5 row at `template.tex:335` labels the family as `$K{=}5$ text`, but the code (`scripts/crv/sota_capture.py:528`, `n_samples: int = 3`) and Guo et al. 2026 §3.2 both specify **K=3** continuation samples. The "K=5" appears to be a copy-paste from the `seakr_k5_proxy.logdet` row directly above (which is genuinely K=5).

| Aspect | App. B Table 5 says | Code does |
|---|---|---|
| Sample budget | `$K{=}5$ text` | K=3 (`ConsistencySampler(n_samples=3)`) |

Practical impact: pure relabeling, no recomputation needed. Fix App. B Table 5 row to `$K{=}3$ text` (or, more honestly, `multi-sample text-consistency`). Pairs with the §3.1 citation fix (re-cite to Guo 2026 RSUS, not SelfCheckGPT).

### 3.5 Honest scope/granularity adapters

These are NOT bugs — they're acknowledged adaptations from the source paper's mechanism to our trace-level summary-statistic framework. Listed for completeness, all defensible with one footnote each:

| Signal | Adaptation |
|---|---|
| `etc (Δ²H max)` | Step-level mean entropy (paper: token-level entropy) |
| `flare.max` | Trace-level summary (paper: online per-token trigger) |
| `seakr_k5_proxy.logdet` | K=5 + full Gram log-det (papers: K=20 SVD top-k EigenScore for SeaKR; covariance EigenScore for INSIDE) |
| `hvr.max` (SELFDOUBT-T1) | Per-step HVR + max-over-steps (paper: per-trace HVR) — documented in `baseline_gates.py:521-528` |

---

## 4. Implications for family-count claim

Current `template.tex` claims (post Round-3 cascade):
- Abstract S4: "four survive Bonferroni at $k{=}15$ across four signal families"
- §1 Contribution Bullet 1: "from four signal families not previously distinguished"
- §4.1 (template.tex:185): "across four signal families"
- §6: "across four signal families"

The 4 Bonferroni-robust survivors per `e7_bonferroni.out`:

| Signal | App. B Table 5 family | §1 ¶1 family (cleanest) |
|---|---|---|
| FCT | answer-trajectory | answer-trajectory (this paper) |
| DRAGIN-RIND.max | attention | attention (Su 2024) |
| min_majority | answer-trajectory | answer-trajectory (this paper) |
| ent.max | logit | token-probability (FLARE-family) |

**Distinct families: 3, not 4** under any honest classification (App. B's grouping, §1 ¶1's grouping, or the code-canonical `signal_lib.py` grouping).

The 9 nominal-95% survivors:

| Signal | §1 ¶1 family (cleanest, after citation fixes) |
|---|---|
| FCT | answer-trajectory (this paper) |
| DRAGIN-RIND.max | attention (Su 2024) |
| ent.max | token-probability (FLARE-family, Jiang 2023) |
| min_majority | answer-trajectory (this paper) |
| CABV | answer-velocity (this paper, novel) |
| flip_rate K=5 | answer-trajectory (this paper, Datta-extended) |
| u_cons.max | multi-sample text-consistency (**Guo 2026** RSUS, not SelfCheckGPT) |
| flip_rate K=1 | answer-trajectory (Datta 2026, K=1 exact) |
| etc | entropy-trend (Li 2026) |

**5 distinct families if we lump this-paper's "answer-trajectory + answer-velocity" as one umbrella:**
- attention, token-probability, entropy-trend, text-consistency, step-level-answer-dynamics-this-paper

**6 distinct families if we keep App. B's split:**
- attention, token-probability, entropy-trend, text-consistency, answer-trajectory, answer-velocity

Path D's "9 across 5; 4 across 3" claim is **honestly defensible under the umbrella lumping**, but requires fixing the u_cons citation (cite Guo 2026 not Manakul 2023) and the hvr.max citation (add Pandey 2026).

### 4.1 Framing burden — §1 ¶1 hides the answer-trajectory family entirely

Beyond the count problem, there is a **structural framing burden** in §1 ¶1 that this audit had not previously flagged. The §1 ¶1 enumeration of the published trigger pool lists exactly five families:

> "attention-based DRAGIN, token-probability FLARE, hidden-state SeaKR, multi-sample text-consistency SelfCheckGPT, and entropy-trend ETC."

Of the **9 nominal-95% survivors** in `e7_bonferroni.out`:

| Survivor | §1 ¶1 family? | Comment |
|---|---|---|
| FCT | ❌ | answer-trajectory — never named in §1 ¶1 |
| DRAGIN-RIND.max | ✓ | attention |
| min_majority | ❌ | answer-trajectory — never named |
| ent.max | ✓ | token-probability (FLARE-family) |
| CABV | ❌ | answer-velocity — never named |
| flip_rate K=5 | ❌ | answer-trajectory — never named |
| u_cons.max | ✓ | text-consistency (under correct citation Guo 2026) |
| flip_rate K=1 | ❌ | answer-trajectory — never named |
| etc | ✓ | entropy-trend |

**5 of 9 survivors come from a family the §1 ¶1 enumeration does not name.** Of the **4 Bonferroni-robust survivors**, two (FCT and min_majority) are answer-trajectory. The reader has to wait until §3.3 to even hear "answer-trajectory" pronounced as a family name, and CABV/LAAS never appear in §3.3 at all.

This is a coverage gap that compounds the family-count error: the paper is not just over-counting from 3 to 4, it is also **silently importing a sixth family (answer-trajectory) that §1 ¶1 never declared the audit-pool would test**. A strict reviewer who maps the §1 ¶1 enumeration to the survivor matrix will notice that more than half the headline survivors are from an unnamed family.

**Recommended fix.** Update §1 ¶1 to name **six** trigger-pool families: attention, token-probability, hidden-state, text-consistency, entropy-trend, *and* answer-trajectory (Datta 2026, Wang 2024 chain-of-probe, Chen 2026). This is consistent with §2 Related Work (`template.tex:138`), which already cites Datta, Wang, and Chen for "step-level answer-trajectory triggers as an emerging length-orthogonal signal class." §1 ¶1 just needs to mirror that framing, and then the family-count claim becomes "four signals across **three** of these six families" — which is what the headline figure caption and survivor table actually report.

---

## 5. Recommended honest pool

If the audited pool is strictly limited to "signals with correct attribution and main-text definitions", honestly cleaned up:

### Tier A — Published-pool peer signals (correctly attributed)

| Signal | Family | Cite |
|---|---|---|
| `DRAGIN-RIND.max` | attention | Su et al. 2024 ✓ |
| `flare.max (1-min p)` | token-probability | Jiang et al. 2023 ✓ |
| `ent.max` | token-probability (FLARE-family baseline) | — (generic baseline, ack in §3) |
| `etc (Δ²H max)` | entropy-trend | Li et al. 2026 ✓ |
| `seakr_k5_proxy.logdet` | hidden-state | Yao 2024 + Chen 2024 INSIDE, proxy ✓ |
| `u_cons.max` | multi-sample text-consistency | **Guo et al. 2026 RSUS** (FIX citation in §1 / §2) |
| `hvr.max` | hedging-marker / text-consistency | **Pandey et al. 2026 SELFDOUBT-T1** (ADD citation in §1 / §2) |

### Tier B — This-paper novel signals (defined in §3.3)

| Signal | Family | Source |
|---|---|---|
| `flip_rate K=5` | answer-trajectory | This paper, Datta-extended ✓ |
| `flip_rate K=1` | answer-trajectory | Datta 2026 K=1 baseline ✓ |
| `min_majority` | answer-trajectory (within-step) | This paper ✓ |
| `late_flips` | answer-trajectory (late-window) | **Fix paper §3.3 line 169** to describe set-comparison, count-return |
| `FCT` | answer-trajectory | This paper, Datta-derived ✓ |
| `CABV` | answer-velocity | **Add 2-3 sentence definition to §3.3** |

### Tier C — Drop or relabel

| Signal | Issue | Action |
|---|---|---|
| `drift_cosine.max` | Author's CRV signal misattributed to hidden-state peer family | Drop from headline pool, OR relabel "auxiliary CRV (this paper)" in App. B |
| `var_l2_window.max` | Same | Same |
| `hidden_norm.range` | Same | Same |
| `CABV × LAAS` | Not significant + LAAS standalone not in pool | Drop from headline pool |

### Net effect

**Tier A + B = 13 signals** (down from 17). All correctly attributed, all defined in main text.

**Bonferroni-robust survivors unchanged:** FCT, DRAGIN-RIND, min_majority, ent.max (4 signals, 3 families per any honest accounting).

**Nominal-95% survivors unchanged:** all 9 are in Tier A or B. The "9 across 5 families" claim becomes fully defensible.

**Removed from pool:** 4 signals (3 CRV diagnostics + 1 CABV × LAAS composite).

---

## 6. File location index

### Trace-generation layer (`emnlp_26_lengthConfound/scripts/`)

| File | Lines | Purpose |
|---|---|---|
| `scripts/04_run_qa.py` | (full) | Closed-book QA decode driver |
| `scripts/04f_run_qa_sota.py` | (full); L73-74, 174, 180, 331, 384 | SOTA baseline decode driver, instantiates gates |
| `scripts/04l_oracle_retrieve_decode.py` | (full) | Oracle-retrieve decode (gold context) |
| `scripts/crv/__init__.py` | — | Package init |
| `scripts/crv/boundaries.py` | (full) | Reasoning-step boundary detection (spaCy + 32-token fallback per §3.3) |
| `scripts/crv/hooks.py` | (full) | Hidden-state capture hooks |
| `scripts/crv/vectors.py` | L1-184 | **CRVTracker** — this paper's CRV framework. **Authoritative source for `drift_cosine.max`, `var_l2_window.max`, `hidden_norm.range`** |
| `scripts/crv/triggers.py` | L24-95 | **CRVTrigger** — CRV-based gating logic |
| `scripts/crv/sota_capture.py` | (876 lines) | **Authoritative source for DRAGIN, U_cons (RSUS), U_verb, U_ent** |
| `scripts/crv/sota_capture.py` | L7-13, L147-215 | `RindAccumulator` — DRAGIN-RIND byte-faithful implementation, Su 2024 cited |
| `scripts/crv/sota_capture.py` | L218-501 | Verbalized confidence probe + entity-entropy scorer (U_verb + U_ent from Guo 2026 RSUS) |
| `scripts/crv/sota_capture.py` | L503-705 | `ConsistencySampler` — **U_cons (RSUS, Guo 2026) — NOT SelfCheckGPT** |
| `scripts/crv/sota_capture.py` | L707-741 | RSUS combiner |
| `scripts/crv/baseline_gates.py` | L1-15, L456-565 | `SelfDoubtTier1Gate` — **HVR byte-faithful implementation, Pandey 2026 cited (arXiv:2604.06389v1)** |
| `scripts/crv/baseline_gates.py` | L822+ | `SelfDoubtFullGate` — full SELFDOUBT (Tier-1 + verbalized) |

### Analysis layer (`emnlp_26_lengthConfound/analysis/`)

| File | Lines | Purpose |
|---|---|---|
| `analysis/signal_lib.py` | (319 lines) | **Authoritative source for shared signal aggregators + the canonical `SIGNALS` tuple at L299-318** |
| `analysis/signal_lib.py` | L40 | `ANSWER_RE` regex for answer extraction |
| `analysis/signal_lib.py` | L53-66 | `step_vals`, `tmax`, `trange` — per-step value accessors |
| `analysis/signal_lib.py` | L69-81 | `etc_score` (ETC, Li 2026) |
| `analysis/signal_lib.py` | L84-86 | `hvr_max` (SELFDOUBT-T1 aggregation) |
| `analysis/signal_lib.py` | L89-104 | `flare_max_per_step` (FLARE, Jiang 2023) |
| `analysis/signal_lib.py` | L107-134 | `seakr_k5_logdet` (SeaKR/INSIDE proxy) |
| `analysis/signal_lib.py` | L140-152 | Answer extraction + normalization |
| `analysis/signal_lib.py` | L155-165 | `per_step_answers` — K=5 answer extraction per step |
| `analysis/signal_lib.py` | L168-177 | `flip_rate_majority` (flip_rate K=5) |
| `analysis/signal_lib.py` | L180-189 | `flip_rate_single` (flip_rate K=1, Datta protocol) |
| `analysis/signal_lib.py` | L192-205 | `late_flips_count` — **CODE-PAPER DISCREPANCY** (uses set, returns count) |
| `analysis/signal_lib.py` | L208-218 | `min_majority_inv` (matches paper) |
| `analysis/signal_lib.py` | L224-229 | `auroc_pos` — AUROC helper |
| `analysis/signal_lib.py` | L245-263 | `ols_residual` — UNCERTAINTY-LINE-style residualization |
| `analysis/signal_lib.py` | L266-292 | `bootstrap_auroc_diff` — paired bootstrap |
| `analysis/signal_lib.py` | L299-318 | **`SIGNALS` tuple — code-canonical signal-to-family map (17 signals)** |
| `analysis/moonshot_explore.py` | (389 lines) | **Authoritative source for CABV, LAAS, FCT, CABV×LAAS** |
| `analysis/moonshot_explore.py` | L69 | `HEADLINE_ARCHS` — Qwen2.5-7B + Gemma-4-26B-A4B-it |
| `analysis/moonshot_explore.py` | L75-109 | `cabv_score` (M1 CABV) |
| `analysis/moonshot_explore.py` | L112-138 | `laas_score` (M2 LAAS, not in headline pool standalone) |
| `analysis/moonshot_explore.py` | L141-169 | `fct_score` (M8 FCT) — matches paper §3.3 Eq. 3 |
| `analysis/moonshot_explore.py` | L181-232 | `load_paired` — paired-oracle trace loader; row builder |
| `analysis/moonshot_explore.py` | L246-262 | `SIGNALS_HEAD2HEAD` — full signal list for the headline report |
| `analysis/e7_bonferroni_and_spearman.py` | (full) | Bonferroni-corrected paired-bootstrap script |
| `analysis/e7_bonferroni.out` | (33 lines) | **Authoritative headline-pool report (17 signals × ✓/✓✓ markers)** |
| `analysis/e7_spearman.out` | (full) | Per-cell Spearman ρ (length vs signal) report |
| `analysis/uncertainty_line_audit.py` | (full) | UNCERTAINTY-LINE reference replication (App. D) |
| `analysis/llm_judge_audit.py` | (full) | GPT-4o-Mini LLM-as-judge robustness (App. C) |
| `analysis/r1_paired_audit.py` | (full) | Per-architecture paired audit |
| `analysis/a1_k1_vs_k5_ablation.py` | (full) | K=1 vs K=5 ablation (§4.5) |
| `analysis/plot_survivor_forest.py` | (234 lines) | Figure 1 generation |

### Paper / writing layer (`option1-lengthConfound/`)

| File | Section | Lines | Purpose |
|---|---|---|---|
| `option1-lengthConfound/template.tex` | Abstract | L107-109 | Headline narrative |
| `option1-lengthConfound/template.tex` | Figure 1 caption | L114 | Headline figure caption |
| `option1-lengthConfound/template.tex` | §1 Introduction | L118-133 | Field framing + contribution bullets |
| `option1-lengthConfound/template.tex` | §1 ¶1 | L121 | **5-family taxonomy claim** |
| `option1-lengthConfound/template.tex` | §2 Related Work | L135-138 | Citation list per family |
| `option1-lengthConfound/template.tex` | §3 Methods | L140-177 | Protocol + residualization + FCT definition |
| `option1-lengthConfound/template.tex` | §3.3 (line 169) | L169 | **late_flips definition — CODE DISCREPANCY** |
| `option1-lengthConfound/template.tex` | §3.3 | L166-176 | Four trace-level aggregations (flip_rate, late_flips, min_majority, FCT) — **CABV / LAAS not mentioned** |
| `option1-lengthConfound/template.tex` | §4.1 | L185 | "across four signal families" (over-count) |
| `option1-lengthConfound/template.tex` | §6 Conclusion | L259 | "across four signal families" (over-count) |
| `option1-lengthConfound/template.tex` | App. B Table 5 | L321-351 | Full 17-signal matrix |
| `option1-lengthConfound/template.tex` | App. B Table 5 row | L342 | hvr.max — **no citation** |
| `option1-lengthConfound/template.tex` | App. B Table 5 rows | L343-346 | drift_cosine, var_l2, hidden_norm — **misattributed to "hidden" family** |
| `option1-lengthConfound/template.tex` | App. B Table 5 rows | L333, L339 | CABV, CABV×LAAS — **App. B only, no §3.3 definition** |
| `option1-lengthConfound/template.tex` | App. F Table | L431-466 | Spearman ρ per cell |

### Logs

| File | Lines | Purpose |
|---|---|---|
| `_rq1-agent.md` | L73-77 | Origin of "4 families" claim — agent log carves FCT out from min_majority |
| `_rq1-agent.md` | L75 | Verbatim claim: "4 different signal families (answer-trajectory, attention, K-sample answer, single-step logit)" |
| `_rq1-agent.md` | L446-447 | Family-layer table with CABV / CABV×LAAS as "extended K-sample-derived continuous family (this work, novel)" |
| `__paperwork/v1-log-archived/_agent_v2.md` | (2211 lines) | Predecessor log |
| `__paperwork/v1-log-archived/_AGENT.md` | (3126 lines) | Earliest agent log |

### Audit / reference

| File | Lines | Purpose |
|---|---|---|
| `option1-lengthConfound/_AUDIT-REPORT.md` | (936 lines) | Prior fresh-pass audit (pre-FCT-rename, post-ECT) |
| `option1-lengthConfound/audit_final.md` | (1010 lines) | Super-reviewer audit (Round-3 cascade applied) |
| `litReview/-rq1-litReview-sources.csv` | (52 lines) | Dynamic-RAG paper inventory |
| `litReview/-rq1-v2-litReview-sources.csv` | (52 lines) | v2 inventory (newer dynamic-RAG papers — DTR, QuCo-RAG, etc.) |
| `litReview/-rq1-v2-litReview.md` | (181 lines) | v2 narrative summary |
| `__lit-review-main/` | — | PDF corpus (Janiak, Vashurin, Datta, Chen, Lu, Mao, Santilli, Ielanskyi, Wang, Zhang Verbosity, Zhang ARS) |

---

*End of `signals_audit.md`. Generated by per-signal trace-generation-to-analysis-to-paper audit on May 23, 2026. Updated May 23, 2026 (second-pass co-researcher cross-check) to add: (i) u_cons App. B `$K{=}5$ text` mislabel (§3.4); (ii) `"hidden CRV"` → `"hidden"` paper-side regression note (§3.2); (iii) knife-edge Bonferroni LBs for `ent.max` and `min_majority` (§2.2, §2.14); (iv) §1 ¶1 framing-burden gap and answer-trajectory invisibility (§4.1); plus refinements to the SeaKR-proxy adapter framing (§2.6) and the Datta sampling-vs-greedy claim (§2.11–§2.12). Standing as ground-truth reference for the signal-pool revisions in upcoming paper rounds.*
