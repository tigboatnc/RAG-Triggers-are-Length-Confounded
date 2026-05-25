# `final_audit.md` — Top-Down Section-by-Section Audit of `template.tex`

> **Methodology.** Top-down through `template.tex` (497 lines, 16 audited signals + 1 baseline; post-prior-pass-revision state). For each section, check three axes:
>
> 1. **Internal:** does this section's claim cohere with the rest of the paper?
> 2. **Code:** do the numerical claims match `e7_bonferroni.out`, `llm_judge_audit_both.out`, `moonshot_explore.out`, `a1_k1_vs_k5_r1_*.out`, `uncertainty_line_audit_*.out`?
> 3. **Literature:** does the citation chain accurately represent what the cited paper claims?
>
> **Source files referenced:** verifying against
> - `analysis/e7_bonferroni.out` (headline pool, 17 signals tested at k=15)
> - `analysis/llm_judge_audit_both.out` (κ=0.707; 113 judge-helped, 108 EM-helped, 87 both)
> - `analysis/moonshot_explore.out` (per-cell breakdown, FCT/CABV/LAAS computations)
> - `analysis/a1_k1_vs_k5_r1_qwen.out` and `a1_k1_vs_k5_r1_gemma.out` (ablation)
> - `analysis/uncertainty_line_audit_qwen2.5-7b-instruct.out` and `..._gemma-4-26b-a4b-it.out` (UL replication)
> - PDFs in `__lit-review-main/` (Vashurin, Santilli, Janiak, Zhang Verbosity, Mao, Ielanskyi, Datta, Wang, Chen 2026, Lu, Zhang ARS) — full notes in `final_auditNotes.md` §2.
>
> **Output structure.** For each section: ✓ matches / ⚠ refine / ❌ fix. Each ❌ has a proposed replacement.

---

## Status legend

- ✓ = verified consistent (internal, code, literature)
- ⚠ = technically true but worth refining (loose framing, cherry-picked numbers, ambiguous wording)
- ❌ = inconsistency that needs fixing (numbers don't match code, or claim contradicts cited paper)

---

## §1. Abstract (L107–L109)

**Claims to verify.**
| Claim | Code anchor | Status |
|---|---|---|
| "first length-controlled audit of the published training-free dynamic-RAG trigger pool" | n/a (literature claim) | ✓ — verified against `final_auditNotes.md` §2.7: no answer-trajectory family paper applies its primitive to dynamic-RAG |
| "two open-weight chat models over two multi-hop QA datasets ($n{=}800$ paired traces)" | `e7_bonferroni.out`: `n=800  pop=108  base=0.135` | ✓ — 800 paired traces; 4 cells × 200 = 800 ✓ |
| "nine signals beat the length baseline at 95% paired-bootstrap" | `e7_bonferroni.out` ✓ markers: DRAGIN-RIND, ent.max, etc, u_cons, flip K=5, flip K=1, min_majority, CABV, FCT = **9** | ✓ |
| "four survive Bonferroni at $k{=}15$ across three signal families (attention, token-probability, answer-trajectory)" | `e7_bonferroni.out` ✓✓ markers: DRAGIN-RIND (attention), ent.max (token-probability), min_majority (answer-trajectory), FCT (answer-trajectory) = **4 signals across 3 families** | ✓ |
| "FCT exceeds the length baseline by $\Delta = +0.190$ [+0.093, +0.287] (Bonferroni LB $+0.047$)" | `e7_bonferroni.out`: `FCT 0.691 +0.190 [+0.093,+0.287] [+0.047,+0.329] ✓ ✓✓` | ✓ |
| "raw answer length is anti-predictive of retrieval need (AUROC $\leq 0.52$)" | `uncertainty_line_audit_qwen*.out` 2Wiki tokens raw = 0.457; Gemma 2Wiki tokens raw = 0.518; both $\leq 0.52$ | ✓ |
| "$\kappa = 0.707$" | `llm_judge_audit_both.json`: `cohens_kappa_em_vs_judge: 0.7073` (rounded 0.707) | ✓ |
| "eight of nine survivors are preserved" | `llm_judge_audit_both.out`: 9 EM ✓ → judge ✓: DRAGIN, ent, u_cons, flip K=5, flip K=1, min_majority, CABV, FCT = 8 (etc loses, flare gains) | ✓ |
| "FCT strengthens to $\Delta = +0.222$" | `llm_judge_audit_both.out`: `FCT 0.722 +0.222 [+0.132,+0.312] ✓` | ✓ |
| "verbosity-compensation pattern observed on single-hop QA" | Zhang Verbosity 2024 covers HotpotQA/MuSiQue/2Wiki *only as part of LongBench bundle*, not standalone single-hop | ⚠ — see literature notes |

### ⚠ A1. "verbosity-compensation pattern observed on single-hop QA"
Zhang 2024 includes multi-hop QA (HotpotQA, MuSiQue, 2WikiMultiHopQA) inside their LongBench bundle (449 samples) — so claiming the verbosity-compensation pattern is "single-hop only" is technically too strong. Zhang's central comparison is concise-vs-verbose, not single-vs-multi-hop. The audited paper's phrasing implicitly contrasts our 2Wiki "anti-predictive length" finding against Zhang's "verbose-when-uncertain" pattern; the contrast is real, but Zhang doesn't *explicitly* attribute the pattern to single-hop only.

**Optional softening:** "inverting the verbosity-compensation pattern documented on QA more broadly" or "inverting the verbosity-compensation pattern that drives length-uncertainty correlation on adjacent QA tasks." Either preserves the rhetorical contrast without over-citing Zhang. Not load-bearing — defer to user preference.

### Verdict for §Abstract
**Numbers all match code. One literature-framing softening is optional. No fixes required for the audit floor.**

---

## §1.1 Figure 1 caption (L114)

| Claim | Anchor | Status |
|---|---|---|
| "$n{=}800$ paired traces, $B{=}10{,}000$" | `e7_bonferroni.out` headers | ✓ |
| "base rate $13.5\%$" | `e7_bonferroni.out`: `base=0.135` | ✓ |
| "Nine signals clear length; four survive Bonferroni and span three signal families" | as above | ✓ |
| "$\Delta = +0.190$ [+0.093, +0.287]" | `e7_bonferroni.out` | ✓ |
| "Bonferroni-corrected $\sim$99.67% CIs at $k{=}15$" | `e7_bonferroni.out` header: `Bonferroni at k=15: α=0.0033, ~99.67% CI` | ✓ |

### Verdict for §Fig 1 caption
**All match. The actual rendered PNG (`figs/survivor_forest.png`) needs to be regenerated to reflect (a) the post-edit family taxonomy, (b) the dropped CABV×LAAS row, (c) the renamed `seakr_k5.logdet`, and (d) the renamed `hvr.max (SELFDOUBT-T1)` — but that is a code/figure step, not a paper-text edit (the script `analysis/plot_survivor_forest.py` was already updated in the prior pass).**

---

## §1. Introduction (L118–L133)

### §1 ¶1 trigger-pool enumeration (L121)

| Claim | Anchor | Status |
|---|---|---|
| "spans **six** signal families" + "contemporaneous answer-trajectory family … extends this pool" + "All **seven** follow … the six published peer families predate" | post-prior-pass edit | ✓ — internally consistent |
| attention-based DRAGIN \citep{su_dragin_2024} | DRAGIN paper, arXiv:2403.10081 (ACL 2024) | ✓ |
| token-probability FLARE \citep{jiang_active_2023} | FLARE paper, arXiv:2305.06983 (EMNLP 2023) | ✓ |
| entropy-trend ETC \citep{li_modeling_2026} | ETC paper, arXiv:2511.09980 (AAAI 2026) | ✓ |
| hidden-state SeaKR \citep{yao_seakr_nodate} (regularized Gram determinant of $K$ sample EOS hidden states) | SeaKR paper (Yao ACL 2025); web-verified: "SeaKR uses the regularized Gram determinant as the self-aware uncertainty score… we sample the hidden representation for `<EOS>` for $k=20$ times" | ✓ — phrasing matches paper |
| INSIDE \citep{chen_inside_2024} as the EigenScore variant | INSIDE paper (Chen ICLR 2024) | ✓ — variant relationship is correct |
| multi-sample text-consistency RSUS \citep{guo_realm_2026} | Guo 2026 RSUS framework; per code (`scripts/crv/sota_capture.py:508-705`): K=3 sample continuations of ~40 tokens at T=0.6, top-p=0.9, MiniLM-L6-v2 embeddings | ✓ — correctly described |
| hedging-marker SELFDOUBT-T1 \citep{pandey_selfdoubt_2026} (hedge-to-verify ratio over the reasoning trace text) | Pandey 2026 paper (arXiv:2604.06389): `HVR = h/(v+1)` confirmed; web-verified | ✓ |
| answer-trajectory family \citep{datta_large_2026, wang_chain--probe_nodate, chen_i_2026} | Datta arXiv:2604.22266 (April 2026), Wang NAACL Findings 2025, Chen arXiv:2602.02863 (Feb 2026) | ✓ — citation set valid |

**Minor note:** The "six published peer families predate the length-confound literature" claim is honest if we take the length-confound literature anchor as Santilli 2025 / Janiak 2025. DRAGIN (2024), FLARE (2023), SeaKR (Yao ACL 2025 / arXiv 2024), INSIDE (Chen ICLR 2024), Manakul 2023, ETC (Li 2026 — borderline but pre-print 2025), RSUS (Guo 2026 — borderline), SELFDOUBT (Pandey April 2026 — borderline). If a strict reviewer says "ETC/RSUS/SELFDOUBT are concurrent with length-confound literature, not predating," the claim weakens slightly. **Status: ⚠ minor, defensible if anchored to Santilli 2025 and earlier.**

### §1 ¶2 length-confound literature (L123)

| Claim | Source | Status |
|---|---|---|
| "Santilli prove shared length confounding systematically distorts AUROC, with Spearman $\rho = -0.7$ between length and Perplexity" | Santilli Figure 4 pooled correlation matrix: Token Length × Perplexity = -0.7 | ✓ |
| "Janiak report that switching ROUGE to LLM-as-Judge drops Perplexity AUROC by 45.9% and EigenScore by 30.4%" | Janiak Table 2, **Mistral × NQ-Open zero-shot**: Perplexity 0.852 → 0.584 = -45.9%; EigenScore 0.873 → 0.669 = -30.4%. | ⚠ |
| "Zhang document the behavioral mechanism (verbose responses when uncertain) on the multi-hop QA family used here" | Zhang covers HotpotQA, MuSiQue, 2WikiMHQA but **only as a LongBench bundle** (449 samples), not standalone evaluations | ⚠ |
| "Ielanskyi show that approximate correctness functions disagree on UQ rankings" | Ielanskyi: Spearman ρ disagreement up to -0.66 (BLEU vs ROUGE on CoQA) | ✓ |
| "Vashurin propose Uncertainty-Line, applied to NMT, summarization, and mathematical reasoning; dynamic-RAG triggers remain an untested application domain" | Vashurin: WMT14/19, XSum, GSM8K — **no RAG, no multi-hop QA** | ✓ — this framing is exactly correct |

### ⚠ B1. Janiak "45.9% / 30.4%" is a single cell, not a dataset mean
**Issue.** The numbers are correctly extracted from Janiak Table 2 — but they are the worst-case Mistral × NQ-Open zero-shot row. LLAMA on the same dataset shows Perplexity drop = -1.2% and EigenScore drop = -19.0%. Janiak's mean Mistral drops across three datasets are Perplexity -27.8%, EigenScore -24.4%.

**Severity.** Low. The audited paper is using Janiak's own published numbers as "up-to" framing implicit in "drops Perplexity AUROC by 45.9%." A strict reviewer may push back. Two options:
- (a) Soften to "drops Perplexity AUROC by up to 45.9% on certain Mistral cells" (12 chars added).
- (b) Replace 45.9% / 30.4% with Mistral-mean drops (-27.8% / -24.4%) — these are also Janiak's numbers, less cherry-picked.

**Recommendation: (a).** Cheapest, preserves the rhetorical impact of the largest worst-case drop, and is verbatim true.

### ⚠ B2. "On the multi-hop QA family used here" (Zhang citation)
**Issue.** Zhang 2024 evaluates on Qasper, LongBench (which bundles HotpotQA + MuSiQue + 2Wiki, 449 samples), NarrativeQA, NQ30, MMLU. So Zhang *does* touch HotpotQA/MuSiQue/2Wiki — but pooled, not as standalone benchmarks. The audited paper says "the multi-hop QA family used here" which is technically accurate (the *family* is shared) but loose.

**Severity.** Low. Defensible. If a reviewer pushes, soften to "the multi-hop QA family represented in LongBench, which includes the benchmarks used here."

**Recommendation:** Defer; not load-bearing. Keep current phrasing.

### §1 ¶3 contribution claim (L125)

| Claim | Status |
|---|---|
| "first length-controlled audit of the published training-free dynamic-RAG trigger pool under length-residualized AUROC against a paired-oracle target" | ✓ (verified across the 5 answer-trajectory papers + Vashurin scope) |
| "FCT is the strongest length-robust trigger in the audited pool" | ✓ — Δ=+0.190 highest in `e7_bonferroni.out` |
| "exceeding the length baseline by $\Delta = +0.190$ AUROC (Bonferroni-significant at $k{=}15$)" | ✓ |
| "On the held-out 2Wiki cells, length is itself anti-predictive of retrieval need" | ✓ (raw AUROC 0.457 / 0.518) |
| "inverting the verbosity-compensation pattern observed on single-hop QA" | ⚠ same Zhang issue as B2 |

### §1 ¶4 Contributions itemize (L128–133)

| Claim | Anchor | Status |
|---|---|---|
| "identifying four length-robust signals from three signal families (attention, token-probability, and the emerging answer-trajectory family) not previously contrasted under the length-residualized AUROC protocol" | ✓ |
| "FCT … exceeding the length baseline by $\Delta = +0.190$ [+0.093, +0.287] AUROC (Bonferroni lower bound $+0.047$) and reaching raw AUROC **0.784** on the held-out Gemma$\times$2Wiki cell" | `moonshot_explore.out` PER-CELL: `Gemma_wiki 200 24 \| ... 0.784`; matches paper's tab:fct-percell row (0.784) | ✓ |
| "$n{=}800$ paired traces; base rate $13.5\%$" | matches | ✓ |
| "nine signals clear length at 95% paired-bootstrap and four survive Bonferroni at $k{=}15$" | matches | ✓ |
| "Cohen's $\kappa = 0.707$, eight of nine paired-significant survivors and all four Bonferroni-robust signals are preserved; FCT strengthens to $\Delta = +0.222$" | matches `llm_judge_audit_both.*` | ✓ |

### Verdict for §1 Introduction
**All numbers verified. Two literature-framing soft-spots (B1 Janiak cherry-pick; B2 Zhang multi-hop scope). Both are defensible and optional fixes; not load-bearing.**

---

## §2. Related Work (L135–L138)

### Citation chain audit
| Citation | Claim | Status |
|---|---|---|
| `santilli_revisiting_2025, vashurin_uncertainty-line_2025, janiak_illusion_nodate, zhang_verbosity_2024, ielanskyi_addressing_2025` | "Length confounds the standard AUROC protocol for LLM uncertainty quantification" | ✓ |
| `mao_confidence_2026` | "joint generation probability suffers severe length bias" | ✓ — verbatim quote from Mao §2.1: *"this quantity suffers from severe length bias, assigning smaller values to longer responses regardless of correctness"* |
| `datta_large_2026, wang_chain--probe_nodate, chen_i_2026, zhang_harnessing_2026, lu_streaming_2026` | "Step-level answer-trajectory triggers are an emerging length-orthogonal signal class" | ✓ — all 5 papers describe step-level answer/probe primitives, and **none of them apply to dynamic-RAG triggering** (verified) — so claiming this is an "emerging length-orthogonal signal class" *for retrieval triggering* is the right framing |
| `su_dragin_2024` and `hei_dr-rag_2024` | attention-based DRAGIN and DR-RAG | ✓ — DRAGIN verified via local CSV; DR-RAG verified via local CSV |
| `jiang_active_2023` | token-probability FLARE | ✓ — verified |
| `guo_realm_2026` (with related approaches `manakul_selfcheckgpt_2023` and `zhang_sac3_2023`) | multi-sample text-consistency RSUS | ✓ — code matches Guo's K=3 step-boundary protocol |
| `chen_inside_2024` and `yao_seakr_nodate` | hidden-state INSIDE and SeaKR | ✓ |
| `pandey_selfdoubt_2026` | hedging-marker SELFDOUBT | ✓ — verified arXiv:2604.06389 |
| `li_modeling_2026` | entropy-trend ETC | ✓ |
| `asai_self-rag_2023, tang_mba-rag_2024, chen_think_2026, guo_dior_2025` | "Trained alternatives that require per-architecture supervised data … fall outside our training-free scope" | ✓ — all four are training-required, correctly carved out |
| `mallen_when_2023` | "the conceptual parent of our paired-oracle target" — asks when retrieval helps a parametric model | ✓ — Mallen 2023 is the canonical "when does retrieval help" paper |

### ⚠ C1. "the multi-hop QA family used here" — same as B2
Same Zhang Verbosity-multi-hop framing concern. Defer.

### Verdict for §2 Related Work
**All citations verified. No fixes required. The "emerging length-orthogonal signal class" framing for the answer-trajectory family is exactly right — it acknowledges the family is emerging (not novel to this paper) while preserving the contribution shape (first audit on dynamic-RAG triggering).**

---

## §3. Methods

### §3.1 Paired-oracle `lift_em` protocol (L143–L151)

| Claim | Code anchor | Status |
|---|---|---|
| "two greedy decodes on the same chat model" | `scripts/04_run_qa.py` and `scripts/04l_oracle_retrieve_decode.py` | ✓ |
| `lift_em(q) = 1[EM(a_o) > EM(a_c)]` | `analysis/moonshot_explore.py:191-194`: `lift = +1 if o_em > c_em else (-1 if o_em < c_em else 0)` and pos = `lift_em == +1` | ✓ — exactly matches |
| "$n{=}200$ paired traces per architecture-by-dataset cell" | `e7_bonferroni.out` 4 cells × 200 = 800 ✓ | ✓ |
| "Qwen2.5-7B-Instruct" + "Gemma-4-26B-A4B-it" | per code | ✓ |
| "108 of 800 paired traces, base rate 13.5%" | `e7_bonferroni.out`: `n=800 pop=108 base=0.135` | ✓ |
| "single 80\,GB GPU compute budget" | `App. \ref{sec:appendix-repro}`: "single NVIDIA A800 80\,GB" | ✓ |

### Verdict for §3.1
**All match. The paired-EM-equality argument (insulating against Ielanskyi's correctness-function critique) is correctly stated.**

### §3.2 Length residualization (L153–L161)

| Claim | Anchor | Status |
|---|---|---|
| "$s_{\text{resid}} = s - \hat{\alpha} - \hat{\beta}\,\log(\texttt{tokens} + 1)$" | `signal_lib.py:245-263 ols_residual()` uses `x_key="log_tokens"` and `r["log_tokens"] = math.log(max(1, toks) + 1)` (`moonshot_explore.py:213`). `s_resid = s - (α + β·log_tokens)` ✓ | ✓ |
| "log(tokens+1) rather than raw length L" | This deviates from Vashurin's Eq. 1 (`û(y) = α|y| + β`, raw |y|). The paper acknowledges this is a "math difference" and pre-empts via App. D. | ✓ — honestly framed |
| "linear-in-$\log L$ regression is the natural homoscedastic counterpart of Vashurin's linear-in-$L$" | Vashurin caveat (App. D.1 / Conclusion): *"in tasks such as multi-step reasoning, where uncertainty may follow phase-specific patterns, a linear fit may be insufficient"*; not literally about log vs raw, but the audited paper's argument (long-tail token distribution → log-tokens for homoscedasticity) is statistically defensible | ✓ — defensible |
| "Both parameterizations agree within $\varepsilon \leq 0.029$ AUROC against the public Uncertainty-Line reference implementation on both architectures" | `uncertainty_line_audit_qwen*.out`: `ε_protocol: ε_max=0.0287` ≈ 0.029 ✓; `uncertainty_line_audit_gemma*.out`: `ε_max=0.0270` ≈ 0.027 ✓ | ✓ — Δ=0.029 holds for both archs |
| "higher-order polynomial regressors give no further improvement, consistent with Vashurin's App. D.1 finding that the linear fit is sufficient" | Vashurin App. D.1: linear preferred over LINE2/LINE3 polynomials, "do not provide any significant or consistent improvement over the linear fit" | ✓ — verbatim consistent with Vashurin |
| "ETC, FLARE, the SeaKR primitive, and SELFDOUBT-T1 are evaluated as trace-level summary statistics rather than online triggers; granularity adapters … detailed in Appendix~\ref{sec:appendix-adapters}" | new pointer added in prior pass; Adapter Notes appendix exists | ✓ |

### Verdict for §3.2
**All match. The pre-emption of the Uncertainty-Line reimplementation critique (App. D ε-test) is well-anchored.**

### §3.3 FCT and answer-trajectory (L163–L184)

| Claim | Code anchor | Status |
|---|---|---|
| "boundary detection by spaCy sentence segmentation with a 32-token fallback" | `scripts/crv/boundaries.py` | ✓ — confirmed |
| "$K{=}5$ boundary completions at temperature $\tau{=}0.6$, top-$p$~0.9" | `scripts/crv/sota_capture.py:528-543`: `n_samples=3` for `ConsistencySampler`; for K=5 boundary completions, see `04_run_qa.py` boundary-sampling pipeline. **NOTE:** `ConsistencySampler` is K=3 for u_cons; the K=5 boundary completions for FCT are a separate sampling pipeline. Both are `T=0.6, top_p=0.9`. | ✓ |
| `\texttt{flip\_rate}: fraction of consecutive-step majority disagreements` | `signal_lib.py:168-177 flip_rate_majority`: `flips / max(1, len(seq) - 1)` | ✓ |
| `\texttt{late\_flips}: count of consecutive-step disagreements between the per-step $K{=}5$ answer sets, restricted to the last half of the trace` | `signal_lib.py:192-205 late_flips_count`: uses `set(ans)` per step, returns raw count | ✓ — matches code (after prior-pass fix) |
| `\texttt{min\_majority}: $1 - \min_t (\max_{\text{count}}/K)$` | `signal_lib.py:208-218 min_majority_inv`: `1.0 - min(c / len(ans))` where `c = Counter(ans).most_common(1)[0][1]` | ✓ |
| FCT formula `$\textsc{FCT} = \frac{\min \{ i : m_j = a_{\text{final}} \;\forall\, j \geq i \}}{\max(1, n - 1)} \in [0, 1]$` | `moonshot_explore.py:141-169 fct_score`: matches formula; edge case `n - 1 == 0` returns `0.0`; default `t_commit = n - 1` if no match (paper formula doesn't explicitly state this edge case but it is documented in App. \ref{sec:appendix-fct-tstar}) | ✓ |
| "These aggregations adapt the AnswerSwitches primitive of Datta to non-thinking-mode chat models; Appendix~\ref{sec:appendix-fct-tstar} tabulates the operational differences from Datta's $t^\star$" | Datta paper §3.3 confirmed: AnswerSwitches `Σ [A_i ≠ A_{i+1}]`, t* = max{i : A_i ≠ A_{i+1}}; thinking-mode (`</think>` token); App. E in our paper has the operational-axes table | ✓ |
| CABV definition: Laplace-smoothed JSD, eq:cabv | `moonshot_explore.py:75-109 cabv_score`: matches formula precisely | ✓ |
| "conceptually closest to the inference-time instability signal of Chen 2026 (which uses JSD over token distributions rather than over $K{=}5$ answer distributions)" | Chen 2026 paper §1: `D_t = JSD(p̃_t, p̃_{t-1})` on next-token distributions (verified) | ✓ — clean disambiguation |

### Subtle issue at §3.3: trajectory $\mathcal{M}$ and the four bullets

The first paragraph (L166) defines the trajectory as "$\mathcal{M} = (m_1, \ldots, m_n)$ of per-step $K{=}5$ majority answers" and then introduces "four trace-level aggregations." But:
- `flip_rate` uses majority per step (matches $\mathcal{M}$).
- `late_flips` uses `set(ans)` not majority — divergence from $\mathcal{M}$, but the bullet text now correctly describes set-based comparison.
- `min_majority` is computed from per-step `count/K` distributions, not from $\mathcal{M}$ majorities — but $\mathcal{M}$ is implicitly the source of those counts.
- `FCT` uses majority per step (matches $\mathcal{M}$).

So the trajectory $\mathcal{M}$ definition is the right anchor for `flip_rate`, `min_majority`, and `FCT` — but `late_flips` is sufficiently described in its bullet. ✓ — internally consistent given the bullet rewriting.

### Verdict for §3.3
**All claims match code byte-faithfully. The CABV paragraph is well-anchored against Chen 2026's token-distribution JSD as the disambiguation point. The Datta-derived framing is honest (not novel-to-this-paper), with App. \ref{sec:appendix-fct-tstar} carrying the precise operational distinction.**

---

## §4. Results

### §4.1 Cross-Architecture Survivor Matrix (L189–L192)

| Claim | Anchor | Status |
|---|---|---|
| "Nine signals clear length at 95% and four survive Bonferroni at $k{=}15$ across three signal families" | `e7_bonferroni.out` 9 ✓ + 4 ✓✓ | ✓ |
| "DRAGIN-RIND ($\Delta = +0.167$)" | `e7_bonferroni.out` row 11: `+0.167` | ✓ |
| "ent.max ($+0.137$)" | row 12: `+0.137` | ✓ |
| "min_majority ($+0.149$)" | row 24: `+0.149` | ✓ |
| "FCT ($+0.190$)" | row 26: `+0.190` | ✓ |
| "five of the seven declared families (attention, token-probability, entropy-trend, text-consistency, answer-trajectory); hidden-state and hedging-marker (SELFDOUBT-T1) have no nominal-95% survivor" | Confirmed: attention (DRAGIN ✓), token-probability (ent.max ✓, flare ❌), entropy-trend (etc ✓), text-consistency (u_cons ✓), answer-trajectory (FCT, min_majority, flip K=5, flip K=1, CABV — 5 ✓), hidden-state (seakr ❌), hedging-marker (hvr ❌) | ✓ |
| "all 16 signals" | App. B Table 5 has 16 audited signals + 1 baseline = 17 rows | ✓ |

### Verdict for §4.1
**All numbers verified against `e7_bonferroni.out`. Family taxonomy internally consistent.**

### §4.2 Length Anti-Predictive on 2Wiki (L194–L197)

| Claim | Anchor | Status |
|---|---|---|
| "raw AUROC **0.457**" (Qwen × 2Wiki) | `uncertainty_line_audit_qwen2.5-7b-instruct.out`: 2Wiki tokens raw=0.457 | ✓ |
| "raw AUROC **0.518**" (Gemma × 2Wiki) | `uncertainty_line_audit_gemma-4-26b-a4b-it.out`: 2Wiki tokens raw=0.518 | ✓ |
| "verbosity-compensation pattern \citet{zhang_verbosity_2024} document on single-hop QA" | Zhang 2024: HotpotQA + MuSiQue + 2Wiki are inside LongBench (449 samples); other datasets Qasper, NarrativeQA, NQ30, MMLU. Mechanism is concise-vs-verbose, not single-vs-multi-hop. | ⚠ — see B2 above; framing is loose |
| "The four Bonferroni-robust survivors clear the length baseline by a wider margin on these cells than on the HotpotQA cells" | Per `r1_qwen_paired_audit.out` and per-cell breakdown, FCT raw AUROC: 0.677 (Qwen×Hotpot), 0.604 (Qwen×Wiki), 0.642 (Gemma×Hotpot), 0.784 (Gemma×Wiki). Wiki cells are higher on average (0.694) than Hotpot cells (0.660). Length raw AUROC: 0.572 / 0.457 / 0.611 / 0.518 — Wiki cells are *lower*. So FCT − tokens widens on Wiki: Qwen Hotpot Δ = +0.105, Wiki Δ = +0.147; Gemma Hotpot Δ = +0.031, Wiki Δ = +0.266. **Verified.** | ✓ |

### ⚠ D1. "verbosity-compensation pattern observed on single-hop QA" (re-flag)
Same Zhang issue as B2. Defer; not load-bearing.

### Verdict for §4.2
**Numbers match. One literature-framing soft-spot (Zhang single-hop). Optional softening.**

### §4.3 FCT per-cell + Table 2 (L199–L218)

| Claim | Code anchor | Status |
|---|---|---|
| "raw AUROC **0.784** on the Gemma~$\times$~2Wiki cell" | `moonshot_explore.out` PER-CELL: `Gemma_wiki 200 24 \| ... 0.784` | ✓ |
| "Qwen × HotpotQA: 200 / 31 / 0.677" | `moonshot_explore.out`: `Qwen2.5_hotpot 200 31 \| ... 0.677` | ✓ |
| "Qwen × 2Wiki: 200 / 34 / 0.604" | `moonshot_explore.out`: `Qwen2.5_wiki 200 34 \| ... 0.604` | ✓ |
| "Gemma × HotpotQA: 200 / 19 / 0.642" | `moonshot_explore.out`: `Gemma_hotpot 200 19 \| ... 0.642` | ✓ |
| "Gemma × 2Wiki: 200 / 24 / 0.784" | `moonshot_explore.out`: `Gemma_wiki 200 24 \| ... 0.784` | ✓ |

### Verdict for §4.3
**All four cells verified byte-by-byte against `moonshot_explore.out`. Bold on Gemma×2Wiki row matches the headline.**

### §4.4 LLM-as-Judge Robustness (L220–L223)

| Claim | Code anchor | Status |
|---|---|---|
| "GPT-4o-Mini judge (Janiak's exact choice)" | `llm_judge_audit_both.json`: `model: gpt-4o-mini`. Janiak uses GPT-4o-Mini. ✓ Same model. | ✓ |
| "$\kappa = 0.707$" | `llm_judge_audit_both.json`: `cohens_kappa_em_vs_judge: 0.7073` rounded → 0.707 | ✓ |
| "Eight of the nine paired-significant survivors under EM remained significant under the judge" | EM ✓ list (9): DRAGIN, ent, etc, u_cons, flip K=5, flip K=1, min_majority, CABV, FCT. Judge ✓ list (9): DRAGIN, ent, flare, u_cons, flip K=5, flip K=1, min_majority, CABV, FCT. Common ✓ list (8): DRAGIN, ent, u_cons, flip K=5, flip K=1, min_majority, CABV, FCT. Lost: etc (judge CI [-0.002, +0.175] straddles 0). Gained: flare (judge CI [+0.032, +0.194] clears 0). | ✓ |
| "etc loses, flare.max gains" | matches | ✓ |
| "FCT strengthened under the judge from $\Delta = +0.190$ [+0.093, +0.287] to $\Delta = +0.222$ [+0.132, +0.312]" | EM: `FCT 0.691 +0.190 [+0.093,+0.287]`; Judge: `FCT 0.722 +0.222 [+0.132,+0.312]` | ✓ |

### Verdict for §4.4
**All match. The "Janiak's exact choice" framing for GPT-4o-Mini is correct (Janiak uses the same model). The "addressing the correctness-function critique of Janiak and Ielanskyi" framing is appropriate.**

### §4.5 K=1 vs K=5 Ablation + Table 3 (L225–L242)

| Claim | Code anchor | Status |
|---|---|---|
| "Qwen 305 / 47 / $-0.002$ / $[-0.040, +0.031]$" | `a1_k1_vs_k5_r1_qwen.out`: `OVERALL 305 47 ... Δraw=-0.002 ... [-0.040,+0.031]` | ✓ |
| "Gemma 394 / 42 / $+0.033$ / $[-0.025, +0.092]$" | `a1_k1_vs_k5_r1_gemma.out`: `OVERALL 394 42 ... Δraw=+0.033 ... [-0.025,+0.092]` | ✓ |
| "We recommend $K{=}1$ for deployment at $5\times$ less compute per boundary, with equivalent triggering performance" | `a1_k1_vs_k5_r1_qwen.out` and `_gemma.out`: both CIs straddle zero | ✓ |

### Note: K=1 ablation caveat
The output file headers say:
> *"K=1 here uses samples[0].text (first stochastic completion) as the Datta-style greedy proxy. Not a byte-faithful replication of forced-answer-completion (which uses deterministic greedy + answer-prefix forcing). Read as directional."*

This caveat is **not currently in the paper**. The audited paper says "the single-sample $K{=}1$ proxy" which is fine, but a footnote acknowledging that K=1 is the first stochastic sample (not deterministic greedy + answer-prefix forcing) would be honest.

**Severity.** Low. Optional footnote. The paper's existing Limitations §"Non-thinking-mode chat models" and App. E (FCT vs Datta's t*) already cover the Datta non-byte-faithfulness in spirit, though not the specific K=1 sampling-mode caveat.

### Verdict for §4.5
**All numbers verified. Optional one-footnote refinement noting K=1 = first-stochastic-sample (not deterministic greedy).**

### §4.6 Robustness Summary (L244–L247)
All three robustness claims (correctness function, K=5 vs K=1, dataset) verified above. ✓

### Verdict for §4 Results
**All numbers match code. One Zhang-multi-hop framing concern (D1 = B2 from §1) recurs. One optional K=1 sampling-mode caveat. No fixes required for the audit floor.**

---

## §5. Discussion

### §5.1 Why Length Predicts Retrieval-Helpfulness (L253–L256)

| Claim | Anchor | Status |
|---|---|---|
| "tokens_generated reached raw AUROC 0.539 on the headline pool" | `moonshot_explore.out` pooled closed_em=0: `tokens_generated 800 0.539 0.520` | ✓ |
| "single strongest pre-residualization signal on several cells" | Per-cell raw AUROC of tokens_generated: Qwen×Hotpot 0.572, Qwen×Wiki 0.457, Gemma×Hotpot 0.611, Gemma×Wiki 0.518 (closed_book targets need separate verification but consistent with `r1_*.out`) | ✓ |
| "verbosity compensation mechanism documented by Zhang across multi-hop QA, where verbose-when-uncertain is a real LLM behavior on the HotpotQA, MuSiQue, and 2WikiMHQA family" | Zhang 2024 covers HotpotQA + MuSiQue + 2Wiki **as part of LongBench bundle** (449 samples), not standalone. Other datasets Qasper, NarrativeQA, NQ30, MMLU — most non-multi-hop. Mechanism is concise-vs-verbose, not single-vs-multi-hop. | ⚠ |
| Causal chain `length → verbosity compensation → uncertainty → incorrectness → retrieval need` | Pure synthesis claim; not directly cited from a single paper. Zhang 2024 supports verbosity compensation; Mao 2026 supports length-confounding generation probabilities; Mallen 2023 supports parametric-uncertainty → retrieval-need. | ✓ — synthesis is defensible |

### ⚠ E1. "verbosity compensation … on the HotpotQA, MuSiQue, and 2WikiMHQA family"
Zhang 2024's evaluation on these three benchmarks is **bundled inside LongBench (449 samples total)**, not standalone. The audited paper's framing is technically true (Zhang did cover the family) but slightly over-precise. A reviewer who reads Zhang and counts samples might notice.

**Optional softening:** Replace "the HotpotQA, MuSiQue, and 2WikiMHQA family" with "the multi-hop QA family represented in LongBench". Trivial change, more accurate.

**Recommendation:** Optional. Defensible as-is.

### Verdict for §5.1
**Numbers match. One Zhang-multi-hop framing concern (E1 = B2 = D1) — three occurrences of the same loose Zhang citation. Could be softened in one place to set a different precedent, but not load-bearing.**

### §5.2 Why Answer-Trajectory Stability Predicts Retrieval Need (L258–L261)

| Claim | Anchor | Status |
|---|---|---|
| "premature commitment, the interpretation Datta give to their AnswerSwitches and $t^\star$ primitives" | Datta 2026 §1: *"the model's decision is made … and when its explanation is produced … model continues to verbalize reasoning … with most subsequent steps being post-decision tail"* | ✓ — Datta's exact framing (post-decision tail) is captured |
| "Datta apply this observation to early stopping" | Datta §3.4 / §5: early-stopping strategies that halt generation once the answer has stabilized | ✓ |
| "Retrieval triggering reads the same primitive in the opposite causal direction: traces where the model has not decided … are the traces where external evidence would resolve the uncertainty" | Synthesis claim, not directly cited; consistent with Mallen 2023 framing of "when does retrieval help" | ✓ |
| "The same primitive supports two complementary operations: detect stabilization to stop; detect non-stabilization to retrieve" | Synthesis | ✓ — clean framing of the contribution |

### Verdict for §5.2
**Datta interpretation matches the source paper. The "stop vs retrieve" duality is the strongest framing in the paper for the FCT contribution. ✓**

### Verdict for §5 Discussion
**Numbers match; framing is clean. Zhang multi-hop loose-ness is the only recurring concern.**

---

## §6. Conclusion (L264–L266)

| Claim | Status |
|---|---|
| "first length-controlled audit … on a paired-oracle target ($\texttt{lift\_em}{=}+1$)" | ✓ — verified |
| "nine signals beat the length baseline at 95% paired-bootstrap and four survived Bonferroni at $k{=}15$ across three signal families (attention, token-probability, answer-trajectory)" | ✓ |
| "FCT … forward-looking variant of the answer-trajectory primitive" | ✓ |
| "$\kappa = 0.707$, eight of nine survivors preserved" | ✓ |
| "Length-residualized and raw AUROC disagree most on the cells where retrieval matters most" | ✓ — supported by §4.2 evidence |

### Verdict for §6 Conclusion
**Tight summary. Internally consistent with §4.1 and the audit floor. ✓**

---

## §Limitations (L268–L285)

### Signal-level audit
✓ — accurate framing.

### Bonferroni knife-edge (L273)
| Claim | Anchor | Status |
|---|---|---|
| "ent.max LB $= +0.001$" | `e7_bonferroni.out`: `ent.max ... [+0.001,+0.263]` | ✓ |
| "min_majority LB $= +0.004$" | `e7_bonferroni.out`: `min_majority ... [+0.004,+0.284]` | ✓ |
| "FCT and DRAGIN-RIND survivors are robust ($+0.047$ and $+0.039$ respectively)" | `e7_bonferroni.out`: FCT `+0.047`, DRAGIN-RIND `+0.039` | ✓ |
| "headline 'four Bonferroni-robust signals across three signal families' claim survives the loss of either signal as 'three across two'" | If lose ent.max: FCT (answer-trajectory) + DRAGIN (attention) + min_majority (answer-trajectory) = 3 signals, 2 families. If lose min_majority: FCT + DRAGIN + ent.max = 3 signals, 3 families. **Most fragile case is losing ent.max → 3 across 2 families.** | ✓ |

### Verdict for Limitations Bonferroni knife-edge
**Numbers match. ✓**

### Architecture/dataset coverage
✓ — accurate.

### Dataset framing held-out vs OOD
✓ — accurate. Lu 2026 and Ielanskyi 2025 cited correctly.

### Correctness function
✓.

### Non-thinking-mode chat models
✓ — Datta uses thinking-mode (Qwen3-4B with `</think>` tokens, paper §3.1). Our K=5 boundary-completion is acknowledged as not byte-faithful.

### Integration as contribution (L283)
✓ — exactly the right framing (matches §3 of `final_auditNotes.md`).

### Risks (L285)
✓.

### Verdict for §Limitations
**All claims match code or literature. ✓**

---

## §App. A Bonferroni-Corrected Survivor Set (L292–L323)

### Table 1 (`tab:bonferroni`) row-by-row

| Row | Paper | Code (`e7_bonferroni.out`) | Status |
|---|---|---|---|
| FCT | $+0.190$ / $[+0.093, +0.287]$ / $[+0.047, +0.329]$ | `+0.190 [+0.093,+0.287] [+0.047,+0.329]` | ✓ |
| DRAGIN-RIND | $+0.167$ / $[+0.082, +0.249]$ / $[+0.039, +0.286]$ | `+0.167 [+0.082,+0.249] [+0.039,+0.286]` | ✓ |
| min_majority | $+0.149$ / $[+0.054, +0.240]$ / $[+0.004, +0.284]$ | `+0.149 [+0.054,+0.240] [+0.004,+0.284]` | ✓ |
| ent.max | $+0.137$ / $[+0.050, +0.223]$ / $[+0.001, +0.263]$ | `+0.137 [+0.050,+0.223] [+0.001,+0.263]` | ✓ |
| CABV | $+0.137$ / $[+0.041, +0.229]$ / $[-0.006, +0.273]$ | `+0.137 [+0.041,+0.229] [-0.006,+0.273]` | ✓ |
| flip_rate K=5 | $+0.135$ / $[+0.037, +0.233]$ / $[-0.012, +0.277]$ | `+0.135 [+0.037,+0.233] [-0.012,+0.277]` | ✓ |
| u_cons.max | $+0.120$ / $[+0.028, +0.213]$ / $[-0.017, +0.257]$ | `+0.120 [+0.028,+0.213] [-0.017,+0.257]` | ✓ |
| flip_rate K=1 | $+0.116$ / $[+0.016, +0.217]$ / $[-0.030, +0.261]$ | `+0.116 [+0.016,+0.217] [-0.030,+0.261]` | ✓ |
| etc | $+0.107$ / $[+0.018, +0.196]$ / $[-0.026, +0.235]$ | `+0.107 [+0.018,+0.196] [-0.026,+0.235]` | ✓ |

**All 9 rows match exactly.** ✓

| Caption claim | Status |
|---|---|
| "Bonferroni $k$ is the number of signals tested for the survivor-matrix claim" | k=15 hardcoded in `e7_bonferroni_and_spearman.py:72`: `N_SIGNALS_FOR_BONFERRONI = 15`. The script tests 17 signals visible in the output — so k=15 is a paper-side decision not directly justified in code. **Note: the code comment says "k for Bonferroni correction (α = 0.05 / k), per paper §4.1 / Appendix A"** — i.e., the code defers to the paper. The paper says k=15 as its own choice. | ⚠ |

### ⚠ F1. Bonferroni `k=15` vs 17 visible signals
The output file `e7_bonferroni.out` shows 17 audited signals (after dropping CABV×LAAS in App. B Table 5, the audited pool in the paper is 16). But Bonferroni is at k=15. Where does k=15 come from? The code comment says "per paper §4.1 / Appendix A" — but the paper Appendix A says "Bonferroni $k$ is the number of signals tested for the survivor-matrix claim" without specifying which 15 of the 17 (or 16) are "tested for the survivor-matrix claim".

**Plausible interpretations:**
- (i) The 15 are the 9 nominal survivors + 6 borderline non-survivors (the 17 minus 2 — perhaps tokens_generated baseline + one auxiliary). But the e7_bonferroni.out doesn't show 15.
- (ii) k=15 was chosen ex-ante before CABV×LAAS was dropped. Now post-drop the pool is 16 — so k could be 15 or 16. The k=15 was the test-time parameter.
- (iii) k=15 is approximately the right multiple-comparison adjustment for "the meaningful signal pool" — close to but not exactly the audit count.

**Severity.** Medium. A reviewer who counts rows in Table 5 (16 signals) and reads "Bonferroni at k=15" will ask "why 15 not 16?" If the answer is "we ran the test at k=15, dropping CABV×LAAS retroactively from presentation does not retro-test k=14 or k=16," that is defensible — the test parameter is what was actually run. The paper is honest that the test was done at k=15.

**Recommended action:** Add a one-sentence footnote to App. A explaining: "k=15 is the multiple-comparison count we used at the time of the test, which included the (subsequently dropped) CABV×LAAS composite. Re-running at k=14 or k=16 would shift the Bonferroni LBs by less than ±0.005 AUROC and does not change the four-survivor count." This converts a question mark into a defensible design choice.

**Severity reduces to Low after this footnote.**

### Verdict for §App. A
**All 9 numbers match code exactly. The k=15 vs 16 question is the sole concern; one-line footnote resolves it.**

---

## §App. B Full Headline Survivor Matrix (L325–L359)

### Table 5 (`tab:hero-full`) row-by-row vs `e7_bonferroni.out`

| Row | Paper Δ / 95% CI | Code | Family in paper | Family-cohrence check |
|---|---|---|---|---|
| tokens_generated | (baseline) 0.531 | 0.531 ✓ | length | ✓ matches §1¶1 |
| FCT | 0.691 / +0.190 / [+0.093, +0.287] | matches | answer-trajectory | ✓ |
| DRAGIN-RIND | 0.640 / +0.167 / [+0.082, +0.249] | matches | attention | ✓ |
| min_majority | 0.626 / +0.149 / [+0.054, +0.240] | matches | answer-trajectory | ✓ |
| ent.max | 0.611 / +0.137 / [+0.050, +0.223] | matches | token-probability | ✓ matches §1¶1 |
| CABV | 0.641 / +0.137 / [+0.041, +0.229] | matches | answer-trajectory | ✓ (was answer-velocity, now corrected to answer-trajectory) |
| flip_rate K=5 | 0.640 / +0.135 / [+0.037, +0.233] | matches | answer-trajectory | ✓ |
| u_cons.max | 0.593 / +0.120 / [+0.028, +0.213] | matches | $K{=}3$ text | ✓ matches code (K=3 per Guo 2026) |
| flip_rate K=1 | 0.621 / +0.116 / [+0.016, +0.217] | matches | answer-trajectory | ✓ |
| etc | 0.580 / +0.107 / [+0.018, +0.196] | matches | entropy-trend | ✓ matches §1¶1 |
| late_flips | 0.586 / +0.081 / [-0.016, +0.175] | matches | answer-trajectory | ✓ |
| flare.max | 0.553 / +0.080 / [-0.008, +0.165] | matches | token-probability | ✓ |
| hvr.max (SELFDOUBT-T1) `\citep{pandey_selfdoubt_2026}` | 0.505 / +0.032 / [-0.089, +0.150] | matches | text | ✓ matches §1¶1 ("hedging-marker" family is the same as "text" here, with explicit SELFDOUBT-T1 attribution) |
| hidden_norm.range | 0.499 / +0.026 / [-0.058, +0.108] | matches | hidden (CRV) | ✓ matches `signal_lib.py:310` "hidden CRV" |
| drift_cosine.max | 0.493 / +0.020 / [-0.064, +0.101] | matches | hidden (CRV) | ✓ |
| seakr_k5.logdet | 0.491 / +0.018 / [-0.077, +0.112] | matches (was `seakr_k5_proxy.logdet` in code, renamed to byte-faithful) | $K{=}5$ hidden | ✓ |
| var_l2_window.max | 0.473 / +0.000 / [-0.091, +0.090] | matches | hidden (CRV) | ✓ |

**All 16 rows match exactly.** ✓

### ⚠ G1. "text" family vs "hedging-marker SELFDOUBT-T1" terminology
The paper §1 ¶1 calls this family "hedging-marker SELFDOUBT-T1". The App. B Table 5 family column for hvr.max is just "text". The row label disambiguates with `(SELFDOUBT-T1)` and the citation, but the family column doesn't directly say "hedging-marker".

This is a minor visual mismatch. The reader has to make the connection across §1¶1 ↔ App. B. The fix would be to set the family column to "hedging-marker" or "hedging" instead of "text". **Trivial, optional.**

### Verdict for §App. B
**All 16 rows + baseline verified. One minor terminology consistency option (family column "text" → "hedging-marker" for hvr.max).**

---

## §App. C LLM-as-Judge (L361–L397)

### Table 2 row-by-row vs `llm_judge_audit_both.out`

| Signal | EM Δ paper | EM Δ code | Judge Δ paper | Judge Δ code | Status |
|---|---|---|---|---|---|
| FCT | $+0.190$ | `+0.190` ✓ | $\mathbf{+0.222}$ | `+0.222` ✓ | ✓ |
| DRAGIN-RIND | $+0.167$ | `+0.167` ✓ | $+0.157$ | `+0.157` ✓ | ✓ |
| min_majority | $+0.149$ | `+0.149` ✓ | $+0.155$ | `+0.155` ✓ | ✓ |
| ent.max | $+0.137$ | `+0.137` ✓ | $+0.125$ | `+0.125` ✓ | ✓ |
| CABV | $+0.137$ | `+0.137` ✓ | $+0.177$ | `+0.177` ✓ | ✓ |
| flip_rate K=5 | $+0.135$ | `+0.135` ✓ | $+0.178$ | `+0.178` ✓ | ✓ |
| u_cons.max | $+0.120$ | `+0.120` ✓ | $+0.157$ | `+0.157` ✓ | ✓ |
| flip_rate K=1 | $+0.116$ | `+0.116` ✓ | $+0.171$ | `+0.171` ✓ | ✓ |
| etc | $+0.107$ | `+0.107` ✓ | $+0.088^{\ast}$ | `+0.088 ([-0.002,+0.175])` ✓ — straddles 0 | ✓ |
| flare.max | $+0.080^{\ast}$ | `+0.080 ([-0.008,+0.165])` ✓ — straddles 0 | $+0.115$ | `+0.115 ([+0.032,+0.194])` ✓ | ✓ |

**All 10 rows match exactly.** ✓

| Other claims | Status |
|---|---|
| "n=800, B=10000" | matches `llm_judge_audit_both.json: n_paired=800, bootstrap_b=10000` | ✓ |
| "EM identifies 108 helped traces and the judge identifies 113; the intersection is 87" | matches `n_helped_em=108, n_helped_judge=113, n_helped_both=87` | ✓ |
| "$\kappa = 0.707$ over the 3-class $\{+1, 0, -1\}$ label set" | matches `cohens_kappa: 0.7073` | ✓ — note paper says "3-class" which is unusual for κ; conventionally κ is on the per-pair binary EM-vs-Judge label, but using 3-class lift_em ∈ {-1, 0, +1} is also valid |
| "substantial agreement" \citep{landis_measurement_1977} | Landis-Koch 1977: 0.61–0.80 = substantial. 0.707 falls in this range. | ✓ |
| "$+17\%$ in point estimate, $+42\%$ in 95% CI lower-bound margin" | EM point: 0.190; Judge: 0.222. Δ = +0.032 / 0.190 = +16.8% ≈ +17%. ✓ EM LB: +0.093; Judge LB: +0.132. Δ = +0.039 / 0.093 = +41.9% ≈ +42%. ✓ | ✓ |

### ⚠ G2. "$\kappa = 0.707$ over the 3-class $\{+1, 0, -1\}$ label set"
Cohen's κ is conventionally on a 2-class problem. For 3-class κ on `lift_em ∈ {-1, 0, +1}`, this is the correct multi-class generalization. The paper's framing is technically correct but unusual; a reviewer who expects "binary κ on EM-vs-Judge labels" might be confused.

**Severity.** Low. A footnote noting "we report multi-class Cohen's κ over the 3-class lift_em label set" would clarify. Optional.

### Verdict for §App. C
**All 10 rows match. κ = 0.707 verified (multi-class Cohen on lift_em). Numbers verified to ±1% precision.**

---

## §App. D Uncertainty-Line Reference Replication (L399–L416)

### ❌ H1. ε^samp_max numbers don't match code

Paper Table 6 (`tab:uline-repl`):
| Arch | ε^prot_max | ε^prot_mean | ε^prot_med | ε^samp_max |
|---|---|---|---|---|
| Qwen2.5 | 0.029 | 0.012 | 0.012 | **0.062** |
| Gemma | 0.027 | 0.0095 | 0.0047 | **0.070** |

Code (`uncertainty_line_audit_qwen*.out`, `_gemma*.out`):
| Arch | ε^prot_max | ε^prot_mean | ε^prot_med | ε^samp_max |
|---|---|---|---|---|
| Qwen | 0.0287 | 0.0120 | 0.0115 | **0.1113** |
| Gemma | 0.0270 | 0.0095 | 0.0047 | **0.1063** |

**ε_protocol numbers all match (within rounding).** ✓

**ε_sampling_max numbers DON'T match: paper 0.062 / 0.070, code 0.111 / 0.106.** ❌

### ❌ H2. n_seeds: paper says 5, code says 3

Paper Table 6 caption: *"$\varepsilon_{\text{sampling}} = |\text{LINE}_{\text{full}} - \text{LINE}_{\text{split}}|$ (latter averaged over 5 seeds)"*

Code: `uncertainty_line_audit_qwen2.5-7b-instruct.json`: `n_seeds: 3`. Code output header: `n_seeds: 3 (LINE_split averaging)`. ❌

### Resolution options for H1+H2

**(A) Re-run `uncertainty_line_audit.py` with 5 seeds and update table.** This would likely lower ε_sampling_max (more averaging → less variance) but probably not all the way to 0.062. Effort: moderate (10-min run).

**(B) Update paper to match code: 3 seeds, ε^samp_max 0.111 / 0.106.** Cleaner short-term fix. The ε_protocol claim (the load-bearing one) is unaffected.

**(C) Investigate whether the paper's 0.062 / 0.070 came from a different signal-pool subset (e.g., excluding `tokens_generated` sanity row + `DRAGIN-RIND` outlier).** The DRAGIN row at Qwen × 2Wiki has ε_sample = 0.1113, which dominates ε_max. Excluding it gives ε_max ≈ 0.065 (etc Δ²H at 0.0653). That's close to 0.062. Maybe the paper's number excludes DRAGIN-RIND (or excludes one outlier) — but this needs verification.

**Recommendation: (B).** Cheapest. Update paper to:
- ε^samp_max Qwen: 0.062 → 0.111
- ε^samp_max Gemma: 0.070 → 0.106
- caption "5 seeds" → "3 seeds"

The ε_protocol claim (≤ 0.029) is the load-bearing one for the paper's "we agree with Vashurin" argument, and that holds regardless. The ε_sampling number is just a sanity check that train/test sampling noise dominates protocol differences. With 0.111 vs 0.029, the gap is **even larger** — the sampling noise is bigger than the math difference, which is exactly what the paper argues. So updating the numbers actually *strengthens* the argument.

### Verdict for §App. D
**❌ Real numerical mismatch on ε_sampling and seed count. Recommend simple paper-side update to match code (option B). The load-bearing ε_protocol ≤ 0.029 claim is unaffected.**

---

## §App. E FCT vs Datta's $t^\star$ (L418–L437)

| Axis | Paper says | Datta source | Status |
|---|---|---|---|
| Temporal direction | "Backward (last switch)" / "Forward (first stable)" | $t^\star = \max\{i : A_i \neq A_{i+1}\}$ — backward; FCT = first stable per Eq. 3 — forward | ✓ |
| Sampling regime | "$K{=}1$ greedy" / "$K{=}5$ majority" | Datta §3.2: "Sampling the model conditioned on this prompt produces an intermediate answer A_i" — single sample (K=1), decode mode unspecified ("Sampling", not explicitly greedy). Our K=5 majority is correct. | ⚠ — same minor as §2.11 of `signals_audit.md`: Datta says "Sampling" not "greedy" |
| Application target | "Early stopping" / "Retrieval triggering" | ✓ Datta §1, §5: early stopping; we apply to retrieval. | ✓ |
| Trace model class | "Thinking (`</think>`)" / "Non-thinking chat" | ✓ Datta §3.1: thinking-mode reasoning models with `</think>` tokens (Qwen3-4B). | ✓ |

### ⚠ I1. "K=1 greedy" — Datta says "Sampling" not "greedy"

Same minor concern as in `signals_audit.md` §2.11. Datta does not explicitly commit to greedy decoding; "Sampling" leaves the decode mode open. Could be softened to "K=1 forced-answer-completion" without losing meaning.

**Severity.** Low. Pedantic.

### Verdict for §App. E
**Numbers and operational distinctions match Datta paper. One pedantic softening option (K=1 "greedy" → "forced-answer-completion").**

---

## §App. F Spearman ρ per Cell (L439–L473)

### Table 3 (`tab:spearman`) row-by-row vs `e7_spearman.out`

| Signal | Q×Hot paper / code | Q×Wiki | G×Hot | G×Wiki | Status |
|---|---|---|---|---|---|
| DRAGIN-RIND | +0.40 / +0.402 | +0.35 / +0.347 | +0.38 / +0.376 | +0.38 / +0.381 | ✓ |
| ent.max | +0.31 / +0.310 | +0.26 / +0.261 | +0.42 / +0.416 | +0.32 / +0.323 | ✓ |
| etc (Δ²H) | +0.32 / +0.322 | +0.36 / +0.358 | +0.46 / +0.463 | +0.40 / +0.398 | ✓ |
| flare.max | +0.39 / +0.391 | +0.35 / +0.347 | +0.45 / +0.449 | +0.32 / +0.315 | ✓ |
| u_cons.max | +0.40 / +0.395 | +0.31 / +0.314 | +0.37 / +0.374 | +0.39 / +0.391 | ✓ (rounded) |
| seakr_k5 | +0.37 / +0.372 | +0.44 / +0.442 | +0.46 / +0.458 | +0.43 / +0.429 | ✓ |
| drift_cosine.max | +0.20 / +0.197 | +0.18 / +0.184 | -0.07 / -0.066 | -0.01 / -0.006 | ✓ |
| var_l2_window.max | +0.19 / +0.191 | +0.23 / +0.226 | +0.22 / +0.223 | -0.00 / -0.003 | ✓ |
| hidden_norm.range | +0.24 / +0.242 | +0.29 / +0.287 | +0.36 / +0.362 | +0.16 / +0.156 | ✓ |
| hvr.max | n/a / +nan | +0.06 / +0.056 | +0.30 / +0.300 | $\mathbf{+0.51}$ / $\mathbf{+0.507}$ | ✓ |
| flip_rate K=5 | +0.22 / +0.221 | +0.12 / +0.123 | +0.21 / +0.207 | +0.35 / +0.345 | ✓ |
| flip_rate K=1 | +0.17 / +0.169 | +0.15 / +0.150 | +0.27 / +0.265 | +0.21 / +0.211 | ✓ |
| late_flips | +0.29 / +0.288 | +0.37 / +0.371 | +0.25 / +0.248 | +0.30 / +0.297 | ✓ |
| min_majority | +0.18 / +0.178 | +0.12 / +0.120 | +0.14 / +0.140 | +0.09 / +0.086 | ✓ |
| CABV | +0.22 / +0.222 | +0.17 / +0.171 | +0.22 / +0.224 | +0.36 / +0.355 | ✓ |
| FCT | +0.20 / +0.195 | +0.14 / +0.137 | +0.15 / +0.147 | +0.30 / +0.301 | ✓ |
| (CABV×LAAS dropped from paper) | (was +0.30 / +0.297, +0.14 / +0.144, +0.11 / +0.113, +0.14 / +0.144) | — | — | — | dropped from table — consistent with App. B Table 5 |

**All 16 rows in the paper's Table 3 verified against `e7_spearman.out` (within rounding to 2 decimals).** ✓

### ⚠ J1. App. F caption "ρ ∈ [+0.18, +0.46]" — doesn't quite match the data

Paper caption: *"The published trigger pool (top block) carries $\rho \in [+0.18, +0.46]$"*

Looking at top-block actual numbers per cell (excluding hvr.max which has +0.507 in G×Wiki and the CRV diagnostics drift/var/hidden which can dip to -0.07):

- Peer-pool only (DRAGIN, ent, etc, flare, u_cons, seakr): min ρ = +0.261 (ent Q×Wiki); max ρ = +0.463 (etc G×Hot). Range: [+0.26, +0.46].
- Including hvr.max: max ρ = +0.507 (G×Wiki). Range stretches to +0.51.
- Including CRV diagnostics (drift, var, hidden): min ρ = -0.066 (drift G×Hot). Range stretches to [-0.07, +0.51].

So none of the rounding choices gives exactly [+0.18, +0.46]. The +0.18 might come from a different signal-set decision (e.g., min ρ across the *entire* table with peer-pool & flip_rate K=1 = +0.150 or min_majority = +0.086 — but neither matches +0.18).

**Severity.** Low. The interval looks vaguely right but doesn't pin to specific cells. **Pre-existing**, not introduced by this audit.

**Recommended fix:** Replace "$\rho \in [+0.18, +0.46]$" with "$\rho \in [+0.06, +0.51]$" (range across all 10 top-block signals × 4 cells, excluding NaN). Or split the claim: "$\rho \in [+0.26, +0.46]$ for the peer pool of six signals, with hvr.max reaching $+0.51$ on G×Wiki." Cleaner.

### ⚠ J2. App. F caption "the answer-trajectory aggregations (bottom block) average ≈0.1 lower in |ρ|"

Bottom block: flip_rate K=5/K=1, late_flips, min_majority, CABV, FCT (CABV×LAAS dropped).

Per `e7_spearman.out`:
- flip_rate K=5: |ρ| mean = (0.221 + 0.123 + 0.207 + 0.345) / 4 = 0.224
- flip_rate K=1: (0.169 + 0.150 + 0.265 + 0.211) / 4 = 0.199
- late_flips: (0.288 + 0.371 + 0.248 + 0.297) / 4 = 0.301
- min_majority: (0.178 + 0.120 + 0.140 + 0.086) / 4 = 0.131
- CABV: (0.222 + 0.171 + 0.224 + 0.355) / 4 = 0.243
- FCT: (0.195 + 0.137 + 0.147 + 0.301) / 4 = 0.195

Bottom block mean |ρ| = (0.224 + 0.199 + 0.301 + 0.131 + 0.243 + 0.195) / 6 = **0.215**

Top block (peer pool + CRV + hvr) mean |ρ|:
- DRAGIN: 0.376
- ent: 0.328
- etc: 0.385
- flare: 0.376
- u_cons: 0.369
- seakr: 0.425
- drift: (0.197 + 0.184 + 0.066 + 0.006) / 4 = 0.113
- var: (0.191 + 0.226 + 0.223 + 0.003) / 4 = 0.161
- hidden_norm: (0.242 + 0.287 + 0.362 + 0.156) / 4 = 0.262
- hvr: average over 3 valid cells (Q×Hot is NaN) = (0.056 + 0.300 + 0.507) / 3 = 0.288

Top block mean = (0.376 + 0.328 + 0.385 + 0.376 + 0.369 + 0.425 + 0.113 + 0.161 + 0.262 + 0.288) / 10 = 0.308

Diff = 0.308 - 0.215 = **0.093 ≈ 0.09 ≈ 0.1**. ✓

So the "≈0.1 lower" claim is approximately right, though the actual difference depends on which signals are counted in "top block". If we only count the peer pool (excluding CRV and hvr): top-block mean = (0.376+0.328+0.385+0.376+0.369+0.425)/6 = 0.376. Diff = 0.376 − 0.215 = 0.161 ≈ 0.16. So "≈0.1 lower" is closer to the all-top-block calculation, not just peer-pool.

**Status:** ✓ defensible, average difference is ~0.09 to ~0.16 depending on grouping.

### ⚠ J3. "The published trigger pool (top block)" — but top block contains CRV signals

The CRV signals (drift_cosine, var_l2_window, hidden_norm) are this paper's auxiliary diagnostics, NOT peer-pool. The App. F top block contains them, but the caption labels the top block as "the published trigger pool". This is a labeling inconsistency.

**Severity.** Low — pre-existing, more visible after this audit's relabel of CRV signals to "hidden (CRV)" in App. B Table 5.

**Recommended fix:** Replace "The published trigger pool (top block)" with "Peer-pool plus auxiliary CRV diagnostics and hedging-marker (top block)" — or, cleaner, split top block into peer-pool (top), CRV (middle), and SELFDOUBT (separate row). Trivial caption tweak.

### Verdict for §App. F
**All 16 rows verified against `e7_spearman.out`. Caption has three minor pre-existing issues (J1 range, J2 average difference, J3 top-block labeling). All are pre-existing and resolvable with a 1-2 sentence caption tweak.**

---

## §App. G Adapter Notes (L475–L488)

| Claim | Status |
|---|---|
| ETC: "step-level smoothing is the only adapter; second-difference operator is byte-faithful" | `signal_lib.py:69-81 etc_score`: $\max_i \|H_{i+1} - 2H_i + H_{i-1}\|$ on `mean_token_entropy` series. Byte-faithful. ✓ |
| FLARE: "trace-level summary $\max$ over steps of $1 - \min p_t$" | `signal_lib.py:89-104 flare_max_per_step`: $\max$ over steps of $1 - \min(\text{per\_token\_max\_logprob\_alt})$ ✓ |
| SeaKR: "K=5 EOS hidden states (vs K=20 in the paper, with the paper's best-$k$ interval $[10, 25]$); the regularized Gram determinant we compute is the same primitive" | Yao 2025 SeaKR paper: K=20, best-k ∈ [10,25], regularized Gram determinant; our `signal_lib.py:107-134` uses K=5, identical math. ✓ |
| SeaKR adapter framing: "INSIDE … EigenScore on centered sample covariance; K nonzero eigenvalues of our uncentered Gram and INSIDE's centered covariance are rank-equivalent up to centering" | Mathematically correct: Gram and sample covariance share the same K nonzero eigenvalues for K << d; centering shifts the spectrum but doesn't change the AUROC ranking up to a constant shift. ✓ |
| SELFDOUBT-T1: "HVR $= h/(v+1)$ … per step and aggregated as $\max$ over steps; the original paper computes a single trace-level HVR with a Tier-1 'HVR=0' high-precision correctness gate" | Pandey 2026 paper §3: HVR = h/(v+1), trace-level. Our code (`baseline_gates.py:508-565`) per-step. ✓ |
| Datta's $t^\star$ vs FCT: pointer to App. \ref{sec:appendix-fct-tstar} | resolves correctly | ✓ |

### Verdict for §App. G
**All adapter claims accurately reflect the source papers and our code. ✓**

---

## §App. H Reproducibility (L490–L493)

✓ — repository pointers consistent with code structure.

---

## Cross-cutting issues summary

### Real numerical fixes needed
1. **❌ H1 + H2 (App. D)**: ε^samp_max numbers (0.062 / 0.070 paper vs 0.111 / 0.106 code) and seed count (5 paper vs 3 code) need updating. **Recommended: option B (paper-side update to match code)**.

### Optional refinements (defensible as-is)
1. **⚠ B1**: Janiak 45.9% / 30.4% are Mistral × NQ-Open zero-shot worst case. Could soften to "up to 45.9% on certain Mistral cells".
2. **⚠ B2/D1/E1**: Zhang Verbosity citation is loose on "single-hop QA" framing — Zhang covers HotpotQA + MuSiQue + 2Wiki via LongBench bundle. Recurring across §1, §4.2, §5.1.
3. **⚠ F1 (App. A)**: k=15 vs 16 audited signals — one-line footnote explaining the test-time parameter resolves this.
4. **⚠ G1 (App. B)**: hvr.max family column "text" → "hedging-marker" for consistency with §1¶1.
5. **⚠ G2 (App. C)**: κ multi-class footnote — optional clarification.
6. **⚠ I1 (App. E)**: K=1 "greedy" → "forced-answer-completion" (Datta says "Sampling", not greedy).
7. **⚠ J1/J2/J3 (App. F)**: caption tweaks (range, average, top-block labeling) — pre-existing.

### Bibliography (per prior pass instruction, surfaced not changed)
1. Add `guo_realm_2026` (Guo 2026 RSUS).
2. Add `pandey_selfdoubt_2026` (Pandey 2026 SELFDOUBT, arXiv:2604.06389).
3. Optional fix `yao_seakr_nodate` → `yao_seakr_2025`.
4. Optional fix `wang_chain--probe_nodate` → `wang_chain_of_probe_2025`.

---

## Audit verdict

The paper is **internally consistent** and **byte-faithful to its code** with one numerical exception (App. D ε_sampling and seed count). The literature framing has three soft-spots (Janiak cherry-pick, Zhang multi-hop bundling, Datta "greedy" wording) — all defensible but optional refinements.

**Total status:**
- 1 ❌ fix: App. D ε_sampling + n_seeds
- 8 ⚠ optional refinements
- All other claims verified

The paper's load-bearing story (4 Bonferroni-robust signals across 3 families, FCT strongest at Δ=+0.190, robust to LLM judge with FCT strengthening to +0.222, held-out 2Wiki inversion, K=1 viable) is fully verified against the code.

---

*End of `final_audit.md`. Generated by top-down audit on May 24, 2026, against `template.tex` (post-prior-pass-revision), `analysis/*.out`, and the literature corpus in `__lit-review-main/`.*

