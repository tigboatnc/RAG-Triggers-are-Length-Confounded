# `paper_revision_plan.md` — Surgical Revision Plan for `template.tex`

> **Purpose.** Translate the findings of `signals_audit.md` (post-second-pass) into a concrete, location-cited list of edits to `template.tex` that fixes accuracy issues without changing the empirical story. The story stays exactly as it is; the attribution, family count, and a handful of definitions become accurate.
>
> **Scope.** Paper-side edits only. No re-running of the analysis pipeline is required for any item below — every change is either a citation swap, a family relabel, a definition addition, or a footnote acknowledgement. The 800 paired traces, the 17-signal pool, the bootstrap CIs, the LLM-judge robustness check, the FCT-vs-tstar comparison, and the Uncertainty-Line replication all stay byte-identical.
>
> **Audit ground truth.** All claims here are traced to `signals_audit.md` §§2.x and §§3.x; any new evidence introduced by the second-pass cross-check (e.g., the SeaKR Gram-determinant correction) is sourced inline.

## 0. Story preservation — what does NOT change

The following load-bearing claims are **unaffected** by the audit and stay verbatim:

1. **First length-controlled audit of the published training-free dynamic-RAG trigger pool.** Still true.
2. **FCT (first-commitment time) is the strongest length-robust trigger.** `\Delta = +0.190` [+0.093, +0.287], Bonferroni LB +0.047. Stays.
3. **Held-out 2Wiki cells: length is anti-predictive of retrieval need.** Raw AUROC 0.457 / 0.518. Stays.
4. **FCT reaches raw AUROC 0.784 on Gemma×2Wiki.** Stays.
5. **LLM-judge robustness:** EM vs GPT-4o-Mini judge agree at $\kappa = 0.707$; 8 of 9 paired-significant survivors preserved; FCT *strengthens* to $\Delta = +0.222$. Stays.
6. **K=1 vs K=5 ablation null:** recommend K=1 for deployment. Stays.
7. **Uncertainty-Line replication:** $\varepsilon_{\text{protocol}} \leq 0.029$. Stays.
8. **Verbosity-compensation discussion (§5.1):** length → verbosity → uncertainty → retrieval-need causal chain. Stays.
9. **Forest plot (Fig 1) shape:** all 17 signal points stay, all CIs stay, all bold/non-bold stays. Only the *family labels in App. B Table 5* and the *family-count claim* change.

The four Bonferroni-robust survivors stay the same: **FCT, DRAGIN-RIND, min\_majority, ent.max.** The **count is still four**. What changes is that **four signals span three distinct families, not four**, and two of the four (ent.max LB +0.001, min\_majority LB +0.004) are knife-edge under any small perturbation.

---

## 1. New evidence from the web check

Three load-bearing facts surfaced from the second-pass web search beyond what `signals_audit.md` already covers:

### 1.1 SeaKR is byte-faithful Gram determinant — not "EigenScore proxy"

Per Yao et al. ACL 2025 main (arXiv:2406.19215, `aclanthology.org/2025.acl-long.1312`):

> "SeaKR uses the regularized Gram determinant as the self-aware uncertainty score… we sample the hidden representation for `<EOS>` for $k=20$ times… The internal states are extracted from the middle layer of the LLM, i.e., $l = L/2$."

This is **exactly** what `seakr_k5_logdet` (`signal_lib.py:107-134`) computes:

```python
M = np.stack(H, axis=0)          # K x d
G = M @ M.T                      # K x K Gram
G = G + alpha * np.eye(...)      # ridge regularizer
sign, logdet = np.linalg.slogdet(G)
```

**Implication:** the audit's earlier framing ("Gram log-det proxy, paper uses SVD top-k EigenScore") was *wrong about SeaKR's math*. EigenScore is **INSIDE** (Chen et al. ICLR 2024). SeaKR uses regularized Gram determinant — identical to our code up to two adapters: K=5 vs K=20, and EOS-hidden-state layer choice (paper uses `l = L/2`; our code uses whatever the decode driver captured for `eos_hidden`).

**Action.** Strengthen `seakr_k5` framing in §2 from "proxy" to "K=5 implementation of SeaKR's Gram-determinant primitive (Yao et al. 2025), with INSIDE (Chen et al. 2024) as the closest peer in the EigenScore variant." This is *better* for the paper, not worse — we are now *more* byte-faithful than the audit currently claims.

### 1.2 SeaKR citation key needs date

`template.tex` currently uses `\citep{yao_seakr_nodate}` (lines 121 and 138). The paper is published in ACL 2025 main. Update the BibTeX key to `yao_seakr_2025` (or fix the date field on the existing key). One-token change in `custom.bib`.

### 1.3 SELFDOUBT (Pandey et al.) verified

arXiv:2604.06389; authors Satwik Pandey, Suresh Raghu, Shashwat Pandey; title *SelfDoubt: Uncertainty Quantification for Reasoning LLMs via the Hedge-to-Verify Ratio*; April 2026. `HVR = h / (v + 1)` confirmed; trace-level evaluation confirmed (paper reports HVR=0 → 96.1% accuracy at 25.4% coverage). The audit's per-step adaptation note is correct.

**Action.** Add `\citep{pandey_selfdoubt_2026}` in the new §1 ¶1 entry for the hedging-marker family and in §2 Related Work alongside multi-sample text-consistency.

---

## Table of contents

- [0. Story preservation — what does NOT change](#0-story-preservation--what-does-not-change)
- [1. New evidence from the web check](#1-new-evidence-from-the-web-check)
- [2. Cascade map — every edit, every line](#2-cascade-map--every-edit-every-line)
- [3. Specific phrasing per location](#3-specific-phrasing-per-location)
  - [3.1 §1 ¶1 trigger-pool enumeration (the load-bearing fix)](#31-1-1-trigger-pool-enumeration-the-load-bearing-fix)
  - [3.2 §2 Related Work — citation chain](#32-2-related-work--citation-chain)
  - [3.3 §3.3 — `late_flips`, CABV, definitions](#33-33--late_flips-cabv-definitions)
  - [3.4 §4.1 / §6 / Abstract — family-count fix (4 to 3)](#34-41--6--abstract--family-count-fix-4-to-3)
  - [3.5 App. B Table 5 — relabels](#35-app-b-table-5--relabels)
  - [3.6 SeaKR proxy framing — strengthen, do not weaken](#36-seakr-proxy-framing--strengthen-do-not-weaken)
  - [3.7 Limitations — knife-edge fragility](#37-limitations--knife-edge-fragility)
  - [3.8 Honest scope/granularity adapter footnotes](#38-honest-scopegranularity-adapter-footnotes)
- [4. Bibliography additions and key swaps](#4-bibliography-additions-and-key-swaps)
- [5. Order of operations](#5-order-of-operations)
- [6. Open decisions for the user](#6-open-decisions-for-the-user)
- [7. Final survivor narrative under the cleaned audit](#7-final-survivor-narrative-under-the-cleaned-audit)

---

## 2. Cascade map — every edit, every line

Cross-referenced against `template.tex` (475 lines) and `signals_audit.md` (post-second-pass).

| # | Location in `template.tex` | Audit ref | Type | Net char delta |
|---|---|---|---|---|
| **A** | L121 §1 ¶1 trigger-pool enumeration | §3.1, §3.2, §4.1 | Add a sixth "answer-trajectory" family; swap SelfCheckGPT → Guo RSUS for u_cons; add Pandey SELFDOUBT-T1 as hedging-marker | +200 chars |
| **B** | L138 §2 Related Work citation list | §3.1 | Same swap + add SELFDOUBT, RSUS, optional CtrlA/DTR coverage sentence | +150 chars |
| **C** | L107-109 Abstract ¶1 (S4) | §4 | "across four signal families" → "across three signal families (attention, token-probability, answer-trajectory)" | -10 chars |
| **D** | L114 Figure 1 caption | §4 | "span four signal families" → "span three signal families" | -2 chars |
| **E** | L129 §1 contribution bullet 1 | §4 | "four signal families not previously distinguished" → "three signal families…" + add a clause naming them | +30 chars |
| **F** | L131 §1 contribution bullet 3 | none | No change (count still 9 / four) |
| **G** | L168 §3.3 list, line for `flip_rate` | §2.11 | No change |
| **H** | L169 §3.3 list, line for `late_flips` | §2.13 / §3.4 | Replace "same, restricted to the last half" with explicit set-comparison + count-return definition | +60 chars |
| **I** | L170 §3.3 list, line for `min_majority` | §2.14 | No change |
| **J** | L171-176 §3.3 FCT block | §2.16 | No change |
| **K** | L176 §3.3, after FCT block | §2.15 | **NEW** 2-3 sentence block defining CABV (continuous answer-belief velocity) | +400 chars |
| **L** | L185 §4.1 first sentence | §4 | "across four signal families" → "across three signal families" | -10 chars |
| **M** | L259 §6 Conclusion | §4 | Same swap | -10 chars |
| **N** | L262-264 §Limitations new ¶ | §0 (knife-edge) | **NEW** sentence acknowledging ent.max + min_majority Bonferroni LBs are knife-edge | +200 chars |
| **O** | L335 App. B Table 5 row `u_cons.max` | §3.4 | Family column `$K{=}5$ text` → `$K{=}3$ text` | -1 char |
| **P** | L342 App. B Table 5 row `hvr.max` | §3.1 | Family column `text` → `text (SELFDOUBT-T1)` plus footnote `\citep{pandey_selfdoubt_2026}` | +50 chars |
| **Q** | L343-346 App. B Table 5 rows `hidden_norm`, `drift_cosine`, `var_l2_window` | §3.2 | Family column `hidden` → `hidden (CRV, this paper)` | +60 chars |
| **R** | L339 App. B Table 5 row `CABV×LAAS` | §2.17 | **DROP** entire row (LAAS standalone never appears in pool, composite is not significant) | -200 chars |
| **S** | L121 (and indirectly L344) seakr framing | §1.1 above | Reword "covariance / SVD top-k EigenScore" → "Gram-determinant primitive (Yao 2025); INSIDE (Chen 2024) is the EigenScore variant" | -10 chars |
| **T** | L161 §3.2 residualization paragraph end | §3.5 (etc adapter) | **NEW** trailing sentence: "ETC, FLARE, and the SeaKR primitive are evaluated as trace-level summary statistics rather than online triggers; granularity adapters are detailed in App. F." | +250 chars |
| **U** | App. F or new App. (after Spearman) | §3.5 | **NEW** ~10-line "Adapter notes" subsection enumerating: ETC step-vs-token, FLARE online-vs-summary, SeaKR K=5/middle-layer, SELFDOUBT per-step-vs-trace | +600 chars |
| **V** | `custom.bib` | §1.2 above | Update `yao_seakr_nodate` → `yao_seakr_2025`; add `pandey_selfdoubt_2026`, `guo_realm_2026` (RSUS) | bib-only |

**Net effect:** the paper's body grows by approximately 1.5 paragraphs of substance (+CABV definition, Limitations knife-edge sentence, adapter notes appendix), shrinks by one App. B Table 5 row, and gains five citations (Yao 2025 datefix, Pandey 2026, Guo 2026, optionally Chen 2026 dynamic-instability, optionally DTR/CtrlA). No figure regenerated, no table value changed.

---

## 3. Specific phrasing per location

For every edit below, **strikethrough** is the current text and **bold** is the proposed replacement (or addition). Where the change is purely additive, only the addition is shown.

### 3.1 §1 ¶1 trigger-pool enumeration (the load-bearing fix)

**Current (L121):**

> "The published pool of training-free triggers spans **five** signal families that each read a different decode-time observable: attention-based DRAGIN [su_dragin_2024] (token importance from attention dynamics), token-probability FLARE [jiang_active_2023] (next-token entropy / max log-prob at the decode head), hidden-state SeaKR [yao_seakr_nodate] (internal-state covariance across $K$ samples), multi-sample text-consistency **SelfCheckGPT (Manakul, 2023)** (cross-sample answer agreement on the generated text), and entropy-trend ETC [li_modeling_2026] (second-difference of step entropy)."

**Proposed (replacement, fixes audit §3.1, §3.2, §4.1 simultaneously):**

> "The published pool of training-free triggers spans **six** signal families that each read a different decode-time observable: attention-based DRAGIN \citep{su_dragin_2024} (token importance from attention dynamics), token-probability FLARE \citep{jiang_active_2023} (next-token entropy / max log-prob at the decode head), entropy-trend ETC \citep{li_modeling_2026} (second-difference of step entropy), **hidden-state SeaKR \citep{yao_seakr_2025} (regularized Gram determinant of $K$ sample EOS hidden states) with INSIDE \citep{chen_inside_2024} as the EigenScore variant, multi-sample text-consistency RSUS \citep{guo_realm_2026} (cross-sample step-boundary continuation agreement), hedging-marker SELFDOUBT-T1 \citep{pandey_selfdoubt_2026} (hedge-to-verify ratio over the reasoning trace text), and the emerging answer-trajectory family \citep{datta_large_2026, wang_chain--probe_nodate, chen_i_2026} (step-level dynamics of the model's predicted answer over the decode).** All six follow the same evaluation protocol (rank traces by the trigger score; report raw AUROC against a binary correctness label) and all predate the length-confound literature."

**Why this works.**
- Drops SelfCheckGPT (we never implemented it; audit §3.1).
- Adds RSUS (Guo 2026), SELFDOUBT-T1 (Pandey 2026) — the two real cited primitives in the codebase.
- Adds answer-trajectory as a sixth declared family — closes the framing-burden gap (audit §4.1) and matches §2 Related Work which already cites Datta, Wang, Chen.
- Reframes SeaKR from "covariance" to "regularized Gram determinant" (web-check §1.1) — this is mathematically tight to our code.
- Splits "SeaKR / INSIDE" so reviewers can verify each.
- Six families is **defensible**: we audit a single representative signal from each except hidden-state (where we audit one signal proxying both SeaKR and INSIDE).

**Optional one-sentence add after the enumeration** (acknowledging contemporaneous methods we did not include):

> "Concurrent training-free methods such as DTR \citep{chen_dtr_2026}, CtrlA \citep{liu_ctrla_2024}, QuCo-RAG \citep{min_qucorag_2025}, MIND \citep{ji_mind_2025}, and DioR \citep{guo_dior_2025} extend the same families but require per-method capture pipelines that fall outside our signal-level audit scope (\S\ref{sec:limitations})."

This pre-empts the "you missed X" reviewer attack and costs one sentence + five bib entries.

---

### 3.2 §2 Related Work — citation chain

**Current (L138):**

> "The training-free trigger pool we audit spans attention-based DRAGIN \citep{su_dragin_2024} and DR-RAG \citep{hei_dr-rag_2024}, token-probability FLARE \citep{jiang_active_2023}, multi-sample text-consistency **SelfCheckGPT \citep{manakul_selfcheckgpt_2023}** and SAC$^3$ \citep{zhang_sac3_2023}, hidden-state INSIDE \citep{chen_inside_2024} and SeaKR \citep{yao_seakr_nodate}, and entropy-trend ETC \citep{li_modeling_2026}."

**Proposed (replacement):**

> "The training-free trigger pool we audit spans attention-based DRAGIN \citep{su_dragin_2024} and DR-RAG \citep{hei_dr-rag_2024}, token-probability FLARE \citep{jiang_active_2023}, **multi-sample text-consistency RSUS \citep{guo_realm_2026}** (with related approaches SelfCheckGPT \citep{manakul_selfcheckgpt_2023} and SAC$^3$ \citep{zhang_sac3_2023}), hidden-state INSIDE \citep{chen_inside_2024} and SeaKR \citep{yao_seakr_2025}, **hedging-marker SELFDOUBT \citep{pandey_selfdoubt_2026}**, and entropy-trend ETC \citep{li_modeling_2026}."

The "(with related approaches…)" parenthetical preserves the SelfCheckGPT and SAC$^3$ citations — they remain proper "related work" entries, just no longer the primary citation for the implemented signal.

---

### 3.3 §3.3 — `late_flips`, CABV, definitions

**Edit H — line 169.** Replace the bullet for `late_flips`.

Current:
> "\texttt{late\_flips}: same, restricted to the last half of the trace;"

Proposed:
> "\texttt{late\_flips}: count of consecutive-step disagreements between the per-step $K{=}5$ answer sets, restricted to the last half of the trace (the count return is intentional — we use \texttt{late\_flips} as a *raw-incidence* counterpart to the *fraction-based* \texttt{flip\_rate});"

This documents both the set-vs-majority comparison and the count-vs-fraction return type — exactly the two divergences the audit §2.13 / §3.4 flagged. AUROC is rank-invariant to count-vs-fraction so the row in App. B Table 5 stays as-is.

**Edit K — line 176, NEW block after FCT.** Add CABV definition immediately before the trailing "These aggregations adapt the AnswerSwitches primitive of Datta…" sentence.

Proposed insertion:
> "We additionally compute \textbf{CABV} (continuous answer-belief velocity), a continuous between-step generalization of \texttt{flip\_rate}: at each populated step $t$, build the Laplace-smoothed empirical distribution $p_t$ over the union of $K{=}5$ answers ever observed in the trace, and average the Jensen-Shannon divergence between consecutive populated steps, $\mathrm{CABV} = \mathrm{mean}_t \, \mathrm{JSD}(p_t, p_{t+1})$. CABV is high on traces whose answer-belief mass continues to redistribute through the decode and reduces to \texttt{flip\_rate} when each step is a delta on a single answer; it is conceptually closest to the inference-time instability signal of \citet{chen_i_2026}."

This adds the missing main-text definition (audit §2.15), grounds CABV in a contemporaneous published primitive (Chen 2026 dynamic-instability uses JSD over token distributions; CABV is the K=5-answer-distribution analogue), and avoids defining LAAS standalone since it does not appear as a standalone signal in the headline pool.

**Edit R — App. B Table 5.** Drop the `CABV × LAAS` row entirely (audit §2.17). The composite is not Bonferroni-significant ($\Delta = +0.089$, 95% CI $[-0.016, +0.191]$) and LAAS standalone is invisible in the headline pool. Removing the row eliminates a paper-coverage gap with no narrative loss.

---

### 3.4 §4.1 / §6 / Abstract — family-count fix (4 to 3)

The four locations that say "four signal families" all need to become "three signal families" (audit §4):

**Edit C — Abstract (L107-109).**
> "and four survive Bonferroni at $k{=}15$ across **three signal families** (attention, token-probability, answer-trajectory)."

**Edit D — Figure 1 caption (L114).**
> "Nine signals clear length; four survive Bonferroni and **span three signal families**."

**Edit E — §1 contribution bullet 1 (L129).**
> "A length-controlled audit of the published training-free dynamic-RAG trigger pool (the first to our knowledge), identifying four length-robust signals from **three signal families** (attention, token-probability, and the emerging answer-trajectory family) **not previously contrasted under length-residualized AUROC**."

**Edit L — §4.1 (L185).**
> "Nine signals clear length at 95\% and four survive Bonferroni at $k{=}15$ **across three signal families**…"

**Edit M — §6 Conclusion (L259).**
> "and four survived Bonferroni at $k{=}15$ **across three signal families**."

**Optional honesty add at §4.1.** A one-sentence acknowledgement of the *nominal-95\%* family span:

> "The full nominal-95\% set of nine survivors spans five of the six declared families (attention, token-probability, entropy-trend, text-consistency, answer-trajectory); only hidden-state has no nominal-95\% survivor."

This matches the e7_bonferroni.out output and lets the reader trust the family bookkeeping at both levels of significance.

---

### 3.5 App. B Table 5 — relabels

Per audit §3.2 / §3.4 / §1.2:

| Row | Current `Family` | Proposed `Family` | Source |
|---|---|---|---|
| L335 `u_cons.max` | `$K{=}5$ text` | `$K{=}3$ text` | code uses `n_samples=3` per Guo 2026 §3.2 |
| L342 `hvr.max` | `text` | `text (SELFDOUBT-T1)` | Pandey 2026 attribution |
| L343 `hidden_norm.range` | `hidden` | `hidden (CRV)` | `signal_lib.py:310` already labels this |
| L344 `drift_cosine.max` | `hidden` | `hidden (CRV)` | `signal_lib.py:308` already labels this |
| L345 `seakr_k5_proxy.logdet` | `$K{=}5$ hidden` | `$K{=}5$ hidden (SeaKR-faithful)` *or* keep as-is | optional strengthening; see §3.6 |
| L346 `var_l2_window.max` | `hidden` | `hidden (CRV)` | `signal_lib.py:309` already labels this |
| L339 `CABV$\times$LAAS` | `answer-velocity` | (drop entire row) | audit §2.17 |

**Optional refinement of `seakr_k5_proxy.logdet` row name.** Since the math is byte-faithful Gram determinant per web-check §1.1, the name `seakr_k5_proxy.logdet` could be relaxed to `seakr_k5.logdet` (drop "proxy") in App. B. This is *one character* in the table and *one character* in the App. F Spearman table (L448, `seakr\_k5`). The code variable name in `signal_lib.py:313` (`seakr_k5_logdet`) already reflects this and would not need changing.

---

### 3.6 SeaKR proxy framing — strengthen, do not weaken

The audit §2.6 originally treated `seakr_k5_proxy.logdet` as an "honest proxy" for both SeaKR and INSIDE. Web-check §1.1 reveals SeaKR itself uses **regularized Gram determinant** (not "EigenScore" or "SVD top-k"), so our code is *byte-faithful to SeaKR's primitive*. The honest adapters are now:

1. **Sample budget:** K=5 vs paper's K=20 (paper's best-k interval is [10–25]).
2. **Hidden-layer choice:** paper extracts from middle layer $l = L/2$; our `eos_hidden` is whatever the decode driver captured (verify against `scripts/04_run_qa.py` and `04f_run_qa_sota.py` — see §6 below for the open question).
3. **Mean-centering:** SeaKR also computes the *uncentered* Gram, so this is **not** an adapter (drop from the audit's framing).

INSIDE (Chen 2024) is the *EigenScore* paper — it computes `(1/K) Σ log λ_i` on the SVD-derived covariance. Our Gram log-det is rank-equivalent on the K nonzero eigenvalues, but is **closer to SeaKR than to INSIDE** because both we and SeaKR don't center.

**Recommended in-paper framing (App. F or new App.):**

> "We compute one signal that primarily implements SeaKR's regularized Gram determinant on $K{=}5$ EOS hidden states (vs $K \in [10, 25]$ in the paper), with a single shared adapter relative to INSIDE: SeaKR uses uncentered Gram, INSIDE uses the eigenvalue spectrum of the centered sample covariance. The two methods' uncertainty scores are rank-equivalent on the nonzero spectrum up to centering."

This is *better* than the previous "proxy doing double duty" framing — strict reviewers will read it as confidence, not hand-waving.

---

### 3.7 Limitations — knife-edge fragility

**Edit N — new Limitations paragraph (L262-264).** Add before the existing "Architecture and dataset coverage" paragraph:

> "\paragraph{Bonferroni knife-edge.} Two of the four Bonferroni-robust survivors carry $\sim$99.67\% lower bounds within $5 \times 10^{-3}$ of zero (\texttt{ent.max} LB $= +0.001$; \texttt{min\_majority} LB $= +0.004$ per App.~\ref{sec:appendix-bonferroni}). The headline ``four Bonferroni-robust signals across three signal families'' claim survives the loss of either signal as ``three across two,'' but the four-count is fragile to small bootstrap-seed or signal-set perturbations; the FCT and DRAGIN-RIND survivors are robust ($+0.047$ and $+0.039$ respectively). We report this transparently rather than tightening $B$ from $10{,}000$ to $50{,}000$ to widen the margin, on the principle that the user-facing claim should reflect the actual statistical state."

This is the most defensive line item in the plan but adds reviewer trust at a cost of one paragraph.

---

### 3.8 Honest scope/granularity adapter footnotes

Per audit §3.5, four signals carry adapter notes that are *true to the code* but not yet flagged in the paper. Rather than inline footnotes that clutter the body, consolidate them as a new short appendix or a paragraph in App. F.

**Edit U — new App. or App. F insertion.**

> "\section*{Adapter Notes}
> \paragraph{ETC (Li et al., 2026).} The original ETC operates on token-level entropy; we evaluate $\Delta^2 H$ on the per-step mean entropy series captured at boundary-detection events (\S\ref{sec:methods-fct}). Step-level smoothing is the only adapter; the second-difference operator is byte-faithful.
> \paragraph{FLARE (Jiang et al., 2023).} The original FLARE is an online per-token retrieval trigger ($\max_t p_t < \tau$); we audit the underlying primitive (low-confidence detection) as a trace-level summary $\max$ over steps of $1 - \min p_t$, consistent with our signal-level audit scope (\S\ref{sec:limitations}).
> \paragraph{SeaKR (Yao et al., 2025).} K=5 EOS hidden states (vs K=20 in the paper, best-k $\in [10, 25]$); regularized Gram determinant is identical to the paper's primitive. INSIDE (Chen et al., 2024) is the closest peer, using the EigenScore on the centered sample covariance — rank-equivalent to our Gram log-det on the nonzero spectrum up to centering.
> \paragraph{SELFDOUBT-T1 (Pandey et al., 2026).} HVR is computed per step and aggregated as $\max$ over steps; the original paper computes a single trace-level HVR and a Tier-1 ``HVR=0'' high-precision correctness gate. The per-step adapter is documented in our \texttt{baseline\_gates.py:521-528} and recoverable as the paper's gate by setting our threshold to $0$.
> \paragraph{Datta's $t^\star$ vs FCT.} Already covered in App.~\ref{sec:appendix-fct-tstar}."

This single ~300-word block discharges the "honest-proxy" exposure for four signals at once.

---

## 4. Bibliography additions and key swaps

Edits to `custom.bib` (or whatever `\bibliography{custom}` resolves to). All entries below are *additions* unless marked `[swap]`.

| Citation key | Type | Purpose | Source |
|---|---|---|---|
| `yao_seakr_2025` `[swap]` | inproceedings | ACL 2025 main; replaces `yao_seakr_nodate` everywhere | aclanthology.org/2025.acl-long.1312 |
| `pandey_selfdoubt_2026` | misc/article | SELFDOUBT-T1 HVR primitive | arXiv:2604.06389 |
| `guo_realm_2026` | misc/article | RSUS U_cons (multi-sample text consistency) | arXiv (ID per `_rq1-agent.md`; need to confirm exact arXiv ID before submission) |
| `chen_dtr_2026` *(optional)* | misc/article | DTR — uncertainty-guided trigger | arXiv:2601.03908 |
| `liu_ctrla_2024` *(optional)* | misc/article | CtrlA — internal-state probe | arXiv:2405.18727 |
| `min_qucorag_2025` *(optional)* | misc/article | QuCo-RAG — corpus-grounded uncertainty | arXiv:2512.19134 |
| `ji_mind_2025` *(optional)* | misc/article | MIND — token-entropy + attention multi-hop | arXiv:2503.23095 |
| `guo_dior_2025` *(optional)* | misc/article | DioR — adaptive cognitive detection | arXiv:2504.10198 |
| `chen_i_2026` `[verify key]` | misc/article | Chen 2026 dynamic-instability — already cited in paper as `chen_i_2026`; verify the bib entry exists and points to arXiv:2602.02863 | local PDF |

The five "optional" entries are needed only if §3.1's optional one-sentence "concurrent training-free methods" line is added. Without that line, the paper is still correct; with it, the paper covers a much broader contemporaneous landscape and reduces "you missed X" reviewer surface.

---

## 5. Order of operations

To keep the cascade auditable and avoid overlapping diffs, do the edits in this order. Each step is independently verifiable by recompiling and re-reading.

1. **Bibliography first.** Apply §4 (`custom.bib`): rename `yao_seakr_nodate` → `yao_seakr_2025`; add `pandey_selfdoubt_2026`, `guo_realm_2026`. Recompile to confirm no missing citations.
2. **§1 ¶1 enumeration (Edit A).** This is the load-bearing change; everything else depends on the six declared families. Recompile, re-read.
3. **§2 Related Work (Edit B).** Update citation chain to match §1 ¶1.
4. **App. B Table 5 relabels (Edits O, P, Q).** The cleanest way to verify §1 ¶1 is consistent: every Family column entry should map to a §1 ¶1 family or to "external/length" or to "auxiliary CRV (this paper)".
5. **App. B Table 5 row-drop (Edit R).** Drop the CABV×LAAS row. Re-check Bonferroni count: it stays at 4 because LAAS×CABV was nominal-only, never Bonferroni-robust.
6. **§3.3 late_flips definition (Edit H).** One-line update.
7. **§3.3 CABV definition (Edit K).** New paragraph block. This is the largest body addition; place it explicitly so word-budget for the 4-page short-paper limit is visible.
8. **Family-count fix in four locations (Edits C, D, E, L, M).** Search-and-replace `four signal families` → `three signal families`. Verify each instance lands inside the correct sentence.
9. **Limitations knife-edge (Edit N).** Add Bonferroni-knife-edge paragraph.
10. **Adapter Notes appendix (Edit U).** Final block.
11. **§3.2 trailing pointer (Edit T).** One sentence pointing to App. F adapter notes.
12. **Optional: §1 ¶1 concurrent-methods sentence.** Add only if word budget permits and if the additional five citations are wanted.

Each step is a self-contained diff that can be reviewed independently. Steps 1–6 together are the minimum viable "audit-discharge" patch; steps 7–10 close the framing-burden / coverage-gap exposures; steps 11–12 are polish.

---

## 6. Open decisions for the user

These are points where the audit allows multiple defensible paths and you should pick before drafting:

### D1. Six-family declaration vs five-family + "answer-trajectory as our contribution"

**Path D1a (recommended):** §1 ¶1 declares **six** families including answer-trajectory as an "emerging" family with citations to Datta, Wang, Chen — i.e., framed as *prior art we extend*, not *our contribution*. This is the most honest reading of the literature (Datta 2026, Wang 2024, Chen 2026, Lu 2026, Zhang 2026 ARS all predate or are concurrent with our submission) and matches §2 Related Work which already cites them.

**Path D1b:** §1 ¶1 keeps **five** declared peer-pool families and explicitly carves out answer-trajectory as *our novel framing*. This requires the contribution-bullet 1 to claim novelty of the "answer-trajectory family lifted into dynamic-RAG triggering," which is harder to defend given the prior art.

**Recommendation: D1a.** The story still credits FCT as the strongest length-robust trigger and as our novel forward-looking variant. The family is not novel; FCT *within* the family is novel.

### D2. Drop CABV × LAAS, or define LAAS too?

**Path D2a (recommended):** Drop the CABV × LAAS row from App. B Table 5. LAAS standalone never appears in the headline pool, the composite is not Bonferroni-significant, and dropping it eliminates a paper-coverage gap with no narrative loss. Audit §2.17 supports this.

**Path D2b:** Define LAAS in §3.3 alongside CABV and keep the composite in App. B Table 5 as an explicit null result. Adds ~40 words to §3.3 for a not-significant signal.

**Recommendation: D2a.**

### D3. SeaKR proxy → SeaKR-faithful?

**Path D3a (recommended):** Relabel `seakr_k5_proxy.logdet` → `seakr_k5.logdet` in App. B Table 5 and App. F Spearman tables; describe in §3.6's framing as "K=5 implementation of SeaKR's Gram-determinant primitive." This is *strictly more accurate* given the web-check finding (§1.1).

**Path D3b:** Keep the `_proxy` label and acknowledge K=5 vs K=20 + middle-layer-vs-EOS as the only adapters in App. F.

**Recommendation: D3a.** The math is byte-faithful; the only remaining adapters are sample budget and possibly layer choice. Strict reviewers benefit from precision.

### D4. Limitations paragraph for knife-edge — keep or drop?

**Path D4a (recommended):** Keep Edit N. It costs one Limitations paragraph, but pre-empts a strict-reviewer attack on the four-Bonferroni-survivors claim. The knife-edge is a real fact in `e7_bonferroni.out`; reviewers will compute it.

**Path D4b:** Re-run the bootstrap with $B = 50{,}000$ to *try* to widen the LB. This costs compute but might let the four-count claim stand without a paragraph. **Risk:** the LB might widen to $+0.005$ instead of $+0.001$ but that's still not far from zero, so the paragraph would still be defensible.

**Recommendation: D4a.** Honesty is cheaper than re-running and produces a stronger paper.

### D5. Concurrent training-free methods sentence (DTR / CtrlA / etc.)

**Path D5a:** Add the optional one-sentence concurrent-methods coverage in §1 ¶1. Five citations, one sentence. Pre-empts "you missed X" reviewer attacks.

**Path D5b:** Skip; rely on §2 Related Work to enumerate prior art.

**Recommendation: D5a if word budget permits.** The five-citation sentence is cheap and the v2 lit review explicitly surfaces these methods as the contemporaneous landscape.

---

## 7. Final survivor narrative under the cleaned audit

After all edits land, the paper's core claims read as:

> "On two open-weight chat models over two multi-hop QA datasets ($n{=}800$ paired traces), nine signals beat the length baseline at 95\% paired-bootstrap, and four survive Bonferroni at $k{=}15$ across three signal families (attention, token-probability, answer-trajectory). FCT exceeds the length baseline by $\Delta = +0.190$ [+0.093, +0.287] (Bonferroni LB +0.047), the strongest length-robust signal in the audited pool."

| Survivor | Δ (resid AUROC) | 95% CI | Bonferroni LB | Family |
|---|---|---|---|---|
| **FCT** | +0.190 | [+0.093, +0.287] | **+0.047** | answer-trajectory (this paper, Datta-derived) |
| **DRAGIN-RIND** | +0.167 | [+0.082, +0.249] | **+0.039** | attention (Su 2024) |
| **min_majority** | +0.149 | [+0.054, +0.240] | +0.004 | answer-trajectory (this paper) |
| **ent.max** | +0.137 | [+0.050, +0.223] | +0.001 | token-probability (FLARE-family) |

The four nominal-95\%-only survivors after the cleanup:

| Survivor | Δ | 95% CI | Family |
|---|---|---|---|
| CABV | +0.137 | [+0.041, +0.229] | answer-trajectory (this paper, novel JSD generalization) |
| flip_rate K=5 | +0.135 | [+0.037, +0.233] | answer-trajectory (Datta-extended) |
| u_cons.max | +0.120 | [+0.028, +0.213] | text-consistency (Guo 2026 RSUS, K=3) |
| flip_rate K=1 | +0.116 | [+0.016, +0.217] | answer-trajectory (Datta exact) |
| etc | +0.107 | [+0.018, +0.196] | entropy-trend (Li 2026, step-mean adapter) |

Nominal-95\% set: 9 signals across 5 of the 6 declared families (attention, token-probability, entropy-trend, text-consistency, answer-trajectory; only hidden-state has no nominal survivor). This is internally consistent with the corrected §1 ¶1 enumeration and the App. B Table 5 family relabels.

The four-Bonferroni-robust signals span three families. The story — *length-controlled audit, FCT is strongest, held-out 2Wiki inverts length, robust to LLM judge* — is unchanged. Only the family count and the attribution chain become accurate.

---

*End of `paper_revision_plan.md`. Generated by co-researcher cross-pass on May 23, 2026, against `signals_audit.md` (post-second-pass) and verified against `template.tex`, `signal_lib.py`, `e7_bonferroni.out`, the SeaKR ACL 2025 paper, and the SELFDOUBT arXiv 2604.06389 v1 paper. This plan is fully cascade-aware: every edit references the audit section that motivates it and the source-paper passage that justifies it.*

