# Sentence-by-Sentence Audit — `main.tex`

> One sentence per line, in source order. Each sentence is rendered as a bullet, with nested audit findings (citation check, code/result check, scientific-soundness check) below it. LaTeX commands are kept inline so we can locate the sentence back in `main.tex`.

**Audit legend.** `[CITE OK]` citation faithfully represented. `[CITE OK*]` faithful but with minor caveat. `[CITE ⚠]` citation arguably stretched or paraphrased loosely. `[CITE ✗]` misrepresentation. `[CODE OK]` claim matches the code/output. `[CODE ⚠]` claim matches code but with caveat. `[CODE ✗]` claim conflicts with code. `[CLAIM OK]` scientifically sound. `[CLAIM ⚠]` defensible but with caveat / framing concern. `[NOT-FOUND]` cited paper not in local lit-review and could not be verified.

---

## Title (L45)

- Length Is the Floor: A Length-Residualized Audit of Training-Free Dynamic-RAG Triggers.
  - [CLAIM OK] Title is descriptive and accurate. "Length is the floor" is supported by §4.1 (length baseline at resid AUROC 0.531 on pool; 9/16 audited signals beat it nominal-95%, 4 survive Bonferroni). "Length-residualized audit" matches the §3.2 protocol (Eq. 2). "Training-free" matches §1 scope restriction and excludes Self-RAG / DioR (§2). "Dynamic-RAG triggers" matches the six-family pool defined in §1.

## Abstract (L107–109)

- Dynamic retrieval-augmented generation systems decide *when* to retrieve from a scalar trigger signal, scored by raw AUROC against binary correctness.
  - [CLAIM OK] This is the standard published-pool convention. Verified by spot-check against DRAGIN \citep{su_dragin_2024}, FLARE \citep{jiang_active_2023}, ETC \citep{li_modeling_2026}, and SeaKR \citep{yao_seakr_2025}, all of which expose a scalar trigger / per-step or trace-level uncertainty score and evaluate retrieval-need / hallucination-detection AUROC against binary correctness. The framing is uncontroversial for the cited family.
- Recent work has shown this protocol is length-confounded on adjacent question-answering (QA) selective-prediction and uncertainty-quantification (UQ) tasks.
  - [CITE OK] Backed by the §1¶2 citations: \citet{santilli_revisiting_2025} (UQ AUROC ↔ length mutual-bias, ACL Short 2025), \citet{janiak_illusion_2025} (EMNLP 2025), \citet{zhang_verbosity_2024} (verbosity compensation), \citet{ielanskyi_addressing_2025} (correctness-function approximation), \citet{vashurin_uncertainty-line_2025} (length-invariant residualization). All five are present in `__lit-review-main` and individually verified below.
  - [CLAIM OK] "Adjacent QA selective-prediction and UQ tasks" is an accurate framing of the prior-work scope: none of those papers operates on a dynamic-RAG trigger pool with a paired-oracle retrieval target. Sets up the paper's contribution honestly.
- We present the first length-residualized audit of the training-free dynamic-RAG trigger pool on a paired-oracle target (`lift_em=+1`) that asks whether retrieval flips a wrong answer to a right one, isolating the causal retrieval-helpfulness signal from length-correlated incorrectness.
  - [CLAIM ⚠] "First to our knowledge" — defensible. \citet{vashurin_uncertainty-line_2025} applies length-residualization to UQ on QA/MT/summarization (not dynamic-RAG triggers); \citet{santilli_revisiting_2025} formalizes the AUROC distortion but does not apply it to a dynamic-RAG trigger pool; \citet{mallen_when_2023} introduces a popularity-based proxy for "when retrieval helps" but does not residualize signals on length. The novelty is the *composition* (length-residualization × paired-oracle target × dynamic-RAG pool), not any of the three pieces individually. Wording is correct as hedged ("first to our knowledge" / "the first length-residualized audit").
  - [CODE OK] `lift_em` definition matches `analysis/r1_paired_audit.py` and §3.1 Eq. 1; positive iff `EM(oracle) > EM(closed)`.
- Across two open-weight chat models on HotpotQA and 2WikiMultiHopQA ($n=800$ paired traces), nine signals beat the length baseline at 95% paired-bootstrap and only four survive Bonferroni at $k=15$.
  - [CODE OK] `e7_bonferroni.out` (and the pooled section of `moonshot_explore.out`): 9 signals show `✓` at nominal 95% (DRAGIN-RIND, ent.max, etc, u_cons.max, flip_rate K=5, flip_rate K=1, min_majority, CABV, FCT), and 4 survive Bonferroni (`✓✓`): FCT, DRAGIN-RIND, ent.max, min_majority. Numbers reproduce.
  - [CODE OK] `n=800` is two arches × two datasets × 200 paired traces; matches `moonshot_explore.out` "pooled = 800".
- On the held-out 2Wiki cells, raw length collapses to at-or-below chance (AUROC $\leq 0.52$), and our **FCT** (first-commitment time, a forward-looking answer-trajectory signal) reaches raw AUROC **0.784** on Gemma×2Wiki, the strongest length-residualized signal in the pool.
  - [CODE OK] `r1_qwen_paired_audit.out` line 176: Qwen×2Wiki `tokens_generated` raw AUROC 0.457. `r1_gemma_paired_audit.out` line 176: Gemma×2Wiki raw AUROC 0.518. Both ≤ 0.52.
  - [CODE OK] `moonshot_explore.out` PER-CELL block (L125): Gemma_wiki FCT raw AUROC 0.784.
  - [CODE OK] "Strongest length-residualized signal in the pool" is at the pooled-headline level: `e7_bonferroni.out` shows FCT pooled `resAUC=0.691` (highest) and `Δ=+0.190` (highest). Confirmed.
- The survivor set is robust to a GPT-4o-Mini semantic-equivalence judge ($\kappa = 0.707$), and FCT itself *strengthens* from $\Delta = +0.190$ to $\Delta = +0.222$ under the judge, the opposite of what a string-level (exact-match) EM artifact would predict.
  - [CODE OK] `llm_judge_audit_both.out` header: `model: gpt-4o-mini`, `Cohen κ: 0.707`. FCT under EM Δ=+0.190 [+0.093,+0.287]; under judge Δ=+0.222 [+0.132,+0.312]. Numbers match exactly.
  - [CODE OK*] "Survivor set robust" is accurate at the *Bonferroni-robust four* level (all four retain nominal-95% under judge) and at the *9-survivor nominal* level (8/9 preserved, single swap etc↔flare.max). This precise wording recurs in §4.4 and Appendix C, where it is fully unpacked.
  - [CLAIM OK] Direction-of-change argument is sound: if EM under-counted correct closed-book answers (false negatives by string match), a semantic judge that correctly catches them would *shrink* `lift_em=+1` (move some +1 → 0), reduce signal-target alignment for a string-level artifact, and so a length-orthogonal signal that strengthens under the judge cannot be EM-artifactual.

## Figure 1 caption — `fig:survivor-forest` (L114)

- **Four trigger signals from three families carry length-orthogonal information beyond the length baseline; FCT (this paper) is strongest.**
  - [CODE OK] Four Bonferroni-robust survivors are FCT (answer-trajectory), DRAGIN-RIND (attention), min_majority (answer-trajectory), ent.max (token-probability) — exactly three families. Verified against `e7_bonferroni.out` and Appendix B Table 2 family column.
- Forest plot of $\Delta = \mathrm{AUROC}(s_{\text{resid}}) - \mathrm{AUROC}(\texttt{tokens\_generated})$ on `lift_em=+1` ($n=800$ paired traces).
  - [CODE OK] Definition matches `signal_lib.py::bootstrap_auroc_diff` and the bootstrap formula in the README. n=800 reproduces.
- Thick bars: 95% paired-bootstrap CIs; thin extensions: Bonferroni-corrected 99.67% CIs at $k=15$.
  - [CODE OK] `e7_bonferroni.out` line 3: "Bonferroni at k=15: α=0.0033, ~99.67% CI" matches. Bonferroni $\alpha = 0.05/15 = 0.00\overline{3}$, two-sided ⇒ 99.67% CI.
- Setup: §\ref{sec:methods}. Tables: App.~\ref{sec:appendix-bonferroni}, \ref{sec:appendix-hero}.
  - [CLAIM OK] Cross-references resolve correctly in `main.tex`.

---

# 1 Introduction (L118)

## ¶1 — Trigger pool & training-free scope (L121)

- Dynamic retrieval-augmented generation (RAG) systems retrieve external evidence during decoding to ground generation in source material \citep{lewis_retrieval-augmented_2021, trivedi_interleaving_2023, shao_enhancing_2023}.
  - [CITE OK] Lewis 2021 (RAG, foundational), Trivedi 2023 (IRCoT, ACL 2023), and Shao 2023 (ITER-RETGEN, Findings of EMNLP 2023) all describe systems that retrieve during/around generation. Bib entries verified; Trivedi and Shao abstracts in custom.bib explicitly describe iterative retrieval interleaved with generation. Lewis is the canonical RAG founding paper. Standard intro pile, accurately represented.
- At the core of every such system sits a *retrieval trigger*: a scalar function of the partial generation that decides when to fire retrieval, evaluated by ranking traces by the trigger score and reporting raw AUROC against a binary correctness label.
  - [CLAIM OK] Accurate framing of the published evaluation convention. DRAGIN (token-importance score), FLARE (max token prob < τ), ETC (entropy second-difference), SeaKR (Gram determinant) all expose a scalar that ranks traces and report AUROC against correctness. Not all *trigger* papers in the literature use AUROC specifically (some report end-to-end EM/F1), so "evaluated by … AUROC" is a partial characterization, but it is the evaluation convention this audit targets and the wording is honest about that scope.
- We restrict our audit to *training-free* triggers.
  - [CLAIM OK] Scope restriction is operative and necessary: it excludes Self-RAG (Asai 2023), DioR (Guo 2025), Search-R1 (Jin 2025), and supervised probes (Lu 2026, Zhang_harnessing 2026), all explicitly listed as out-of-scope in §2. Defensible boundary.
- Prior work spans six signal families: attention-based DRAGIN \citep{su_dragin_2024}, token-probability FLARE \citep{jiang_active_2023}, entropy-trend ETC \citep{li_modeling_2026}, hidden-state SeaKR \citep{yao_seakr_2025} and INSIDE \citep{chen_inside_2024}, multi-sample text-consistency ($U_{\text{cons}}$ component of RSUS, \citealp{guo_when_2026}), and hedging-marker SELFDOUBT-T1 \citep{pandey_selfdoubt_2026}.
  - [CITE OK] DRAGIN: bib abstract confirms attention-based dynamic-RAG trigger. FLARE: confirms low-probability online trigger. ETC (Li 2026): "Entropy-Trend Constraint" — confirms entropy-trend trigger via first/second-order differences. SeaKR: ACL 2025, confirmed self-aware hidden-state. INSIDE: confirmed EigenScore on internal-state covariance (hidden-state family). SELFDOUBT-T1: HVR (hedge-to-verify ratio), gate at HVR=0 — confirmed hedging-marker.
  - [CITE OK] guo_when_2026 / $U_{\text{cons}}$ as "RSUS component": **VERIFIED via direct paper-body read (arxiv 2604.26649)**. ReaLM-Retrieve is the framework name; RSUS (Reasoning Step Uncertainty Score) is the literal name of its uncertainty score, defined in §4.1 Eq. (2) as $\text{RSUS}(r_i) = \alpha \cdot U_{\text{verb}}(r_i) + \beta \cdot U_{\text{ent}}(r_i) + \gamma \cdot U_{\text{cons}}(r_i)$; and $U_{\text{cons}}$ is one of the three components, literally defined as "Consistency signal $U_{\text{cons}}$: we sample $k=3$ alternative continuations and measure agreement." Our citation is verbatim — the family label "$K=3$ text" in our headline table also matches Guo's $k=3$. **Audit's earlier [CITE ⚠] flag was a false positive.**
  - [CLAIM OK] "Six families" enumeration matches the families listed (attention, token-prob, entropy-trend, hidden-state, multi-sample text, hedging-marker = 6). Answer-trajectory is presented separately as "contemporaneous", which is consistent with §4.1's "seven declared families" later (six original + answer-trajectory).
- A contemporaneous answer-trajectory family tracks step-level dynamics of the model's own predicted answer \citep{datta_large_2026, wang_chain--probe_2025, chen_i_2026}.
  - [CITE OK] Datta 2026 ("Large Language Models Decide Early and Explain Later"): uses *forced answer completion* at partial reasoning prefixes to track answer evolution. Wang 2025 (Chain-of-Probe, NAACL 2025 Findings): probes step-by-step CoT for necessity and accuracy of answer changes. Chen 2026 ("I May Not Have Articulated Myself Clearly"): inference-time instability via consecutive-step JSD on token distributions. All three are step-level answer-trajectory primitives. Faithful grouping.
  - [CLAIM OK*] "Contemporaneous" is accurate (Datta and Chen are 2026; Wang is NAACL 2025).

## ¶2 — The length-confound critique (L123)

- Recent work has shown this evaluation convention is length-confounded on adjacent QA selective-prediction and uncertainty-quantification (UQ) tasks.
  - [CLAIM OK] Re-statement of the abstract claim, fully supported by the subsequent five citations.
- \citet{santilli_revisiting_2025} prove that mutual length-bias on UQ scores and correctness functions systematically distorts AUROC.
  - [CITE OK] Santilli 2025 abstract (custom.bib L321–333): "We formally prove that any mutual bias non-randomly skews AUROC rankings" and "length biases in correctness functions distort UQ assessments by interacting with length biases in UQ methods." Verbatim alignment with the paper's claim.
- \citet{janiak_illusion_2025} show the empirical consequence: detection AUROC drops by up to 45.9% when switching from ROUGE to an LLM-as-Judge label.
  - [CITE OK] Janiak 2025 abstract (verified by direct PDF read of `Janiak et al. - The Illusion of Progress…`): "several established detection methods show performance drops of up to 45.9% when assessed using human-aligned metrics like LLM-as-Judge." Exact number is the upper bound across detection methods, not a single-method figure, and the contrast is ROUGE vs LLM-as-Judge — matches the sentence.
  - [CLAIM OK] The "empirical consequence of length-confounding" framing is consistent with Janiak's stated finding that "simple heuristics based on response length can rival complex detection techniques."
- \citet{zhang_verbosity_2024} attribute the underlying behavior to verbosity compensation on multi-hop QA, and \citet{ielanskyi_addressing_2025} document a related correctness-function failure across approximate metrics.
  - [CITE OK] Zhang_verbosity_2024 abstract: "Verbosity Compensation (VC) — similar to the hesitation behavior of humans under uncertainty — where they respond with excessive words" and "strong connection between verbosity and model uncertainty." Multi-hop framing OK as Zhang's benchmark suite includes QA (Qasper, knowledge-and-reasoning-based QA tasks); however Zhang's headline datasets are not strictly multi-hop in the HotpotQA sense. **MINOR**: "on multi-hop QA" is a slightly forced characterization — Zhang's scope is more general (knowledge-and-reasoning QA). Recommend "on QA" or "on knowledge-and-reasoning QA" for precision. Non-blocking.
  - [CITE OK] Ielanskyi 2025 abstract: "commonly used approximate correctness functions have substantial disagreement between each other and, consequently, in the ranking of the uncertainty estimation methods." Faithful representation of correctness-function failure across approximate metrics.
- The methodological response is \citet{vashurin_uncertainty-line_2025}'s **Uncertainty-Line**, which regresses each UQ score on output length and uses the residual as a length-invariant estimate.
  - [CITE OK] Vashurin 2025 abstract: "UNCERTAINTY-LINE: (Length-INvariant Estimation), a simple debiasing procedure that regresses uncertainty scores on output length and uses the residuals as corrected, length-invariant estimates." Verbatim match.
- The dynamic-RAG trigger pool is the missing application.
  - [CLAIM OK] Soft rhetorical bridge claim; the five cited prior works above all live in UQ / hallucination detection, not dynamic-RAG triggering. Defensible.

## ¶3 — Our extension + FCT introduction (L125)

- We extend the length-residualization protocol of \citet{vashurin_uncertainty-line_2025} to the training-free dynamic-RAG trigger pool.
  - [CITE OK] Faithful: this is exactly what §3.2 (Eq. 2) and Appendix E (uncertainty-line replication, ε_protocol ≤ 0.029) operationalize.
  - [CODE OK] `analysis/signal_lib.py::ols_residual` implements OLS of signal on log(tokens+1); `analysis/uncertainty_line_audit.py` confirms ε_protocol ≤ 0.029 against the Vashurin reference at `externalCodeAudits/uncertainty-line/`.
- Rather than score triggers against single-arm correctness, we re-decode each closed-book trace under gold context and ask whether retrieval flips a wrong answer to a right one, defining a paired-oracle target $\texttt{lift\_em}=+1$ conceptually motivated by \citet{mallen_when_2023}.
  - [CITE OK] Mallen 2023 ("When Not to Trust Language Models"): the paper studies *when* parametric vs non-parametric (retrieval) memory helps, using entity-popularity as a proxy. "Conceptually motivated by" is the correct hedge — this paper does *not* literally use a paired re-decode, but does ask the same when-does-retrieval-help question. Sentence is honest about being conceptual motivation, not method import.
  - [CODE OK] Matches `analysis/r1_paired_audit.py` and §3.1 Eq. 1.
- This isolates the *causal* retrieval-helpfulness signal from the length-correlated incorrectness signal that the published pool actually measures (§\ref{sec:discussion-causal}).
  - [CLAIM ⚠] "Causal" is strong — the design controls for one confound (closed-book correctness) by stratification (only closed-wrong gets a chance to be lift_em=+1), but it does not control for all confounds between trigger signal and oracle-flipping (e.g., question type, hop count, gold-document length, oracle-decode noise). The §5.1 discussion uses the same word ("causal") and acknowledges the proxy chain. **RECOMMEND**: weaken to "isolates retrieval-helpfulness" or "directly measures retrieval-helpfulness" (drop "causal" or scare-quote it). The sentence does correctly hedge "actually measures" in the second clause. Non-blocking but the strongest word in the abstract.
- We also propose **FCT** (first-commitment time), a forward-looking variant of the answer-trajectory primitive \citep{datta_large_2026, wang_chain--probe_2025, chen_i_2026, zhang_when_2026}.
  - [CITE OK*] All four cites are the answer-trajectory family. Wang 2025 (Chain-of-Probe) probes per-step intermediate answers; Datta 2026 introduces $t^\star$ (final-switch index, backward-looking); Chen 2026 uses instability/JSD signal; Zhang 2026 ("When Does a Language Model Commit?") explicitly proposes a forward-looking commitment time $\tau_{\text{commit}}$ from finite-answer projection. Grouping is faithful. "Forward-looking" is the actual operational property of FCT (= first step from which majority equals trace-final), so naming is accurate.
- FCT reads $K=5$ majority answers from non-thinking-mode chat models, where neither Datta's `</think>`-anchored forced completion nor \citeauthor{zhang_when_2026}'s finite-set continuation projection applies directly (App.~\ref{sec:appendix-fct-tstar}).
  - [CITE OK] Datta 2026 method: confirmed in PDF — uses forced answer completion on Qwen3-4B with explicit thinking-mode reasoning structure (the $K=1$ greedy completion at each reasoning-step prefix). The `</think>` anchor is the standard explicit thinking-mode token in Qwen3 reasoning models.
  - [CITE OK] Zhang 2026 abstract (custom.bib L599–613): "finite-answer preference stabilization … projects the model's own continuation probabilities onto a finite answer set; in binary tasks this yields an exact log-odds code." Confirmed forward-looking on finite-set continuation distribution.
  - [CLAIM OK] Qwen2.5-7B-Instruct has no thinking-mode toggle; Gemma-4-26B-A4B-it under default chat template leaves thought channel empty. Verified in §App. D L317: "no thinking-mode toggle on either model; Qwen2.5 has none, Gemma-4-26B-A4B-it leaves the thought channel empty under defaults." Internally consistent.

## ¶4 — Contributions (L127)

- (1) A length-residualized audit of the dynamic-RAG trigger pool, the first to our knowledge, scored against a paired-oracle target ($\texttt{lift\_em}=+1$).
  - [CLAIM OK] Correctly hedged "to our knowledge." Already audited under Abstract; the combination (length-residualization × paired-oracle × dynamic-RAG pool) is genuinely novel.
- (2) **FCT** (first-commitment time), our forward-looking variant of the answer-trajectory primitive and the strongest length-residualized trigger in the audited pool.
  - [CODE OK] `e7_bonferroni.out`: FCT resAUC 0.691 (highest of 16 signals); Δ = +0.190 (highest). Direct match.
- (3) A correctness-function robustness check against \citet{janiak_illusion_2025} and \citet{ielanskyi_addressing_2025}, under which the survivor matrix is preserved and FCT *strengthens*.
  - [CITE OK] Janiak & Ielanskyi are the two correctness-function critiques (above). The cross-judge robustness check is the natural response to both papers' methodological recommendation (use LLM-as-Judge to escape ROUGE/EM brittleness).
  - [CODE OK] `llm_judge_audit_both.out` shows 8/9 nominal-95% EM survivors preserved under gpt-4o-mini judge, all 4 Bonferroni-robust retain ✓, and FCT Δ rises from +0.190 to +0.222. "Survivor matrix preserved" is accurate at the Bonferroni-robust-four level (unambiguous) and the nominal level (8/9 with a single swap; §4.4 unpacks this honestly).

---

# 2 Related Work (L129)

## ¶ — single paragraph (L132)

- The length-confound critique of standard UQ AUROC evaluation comes from QA UQ and hallucination detection \citep{santilli_revisiting_2025, janiak_illusion_2025, zhang_verbosity_2024, ielanskyi_addressing_2025, vashurin_uncertainty-line_2025, mao_confidence_2026}.
  - [CITE OK] Santilli, Janiak, Zhang_verbosity, Ielanskyi, Vashurin already audited above. All five are in `__lit-review-main` and faithfully reflect the length-confound critique.
  - [CITE OK*] Mao 2026 ("Confidence over Time", custom.bib L422–436): the abstract explicitly mentions "sensitive to superficial factors such as response length or verbosity, and struggle to distinguish correct reasoning from confidently stated errors" — so length-sensitivity is *one* motivation, but Mao's main contribution is signal-temporal-logic confidence calibration, not a length-confound paper per se. Inclusion is defensible because Mao positions itself against length-bias; minor stretch in grouping. Non-blocking.
- The answer-trajectory family \citep{datta_large_2026, wang_chain--probe_2025, chen_i_2026} provides the parent primitives for FCT.
  - [CITE OK] Same three answer-trajectory papers as §1¶1 (Datta = $t^\star$ + AnswerSwitches; Wang = Chain-of-Probe; Chen = per-step JSD instability). Faithfully grouped.
- Two related supervised methods sit outside our training-free scope: \citet{lu_streaming_2026} train hidden-state probes for streaming hallucination detection, and \citet{zhang_harnessing_2026} contrastively shape answer embeddings under latent perturbation at the trace boundary, where their "answer agreement" differs from our $K=5$ boundary-completion "answer-trajectory".
  - [CITE OK] Lu 2026 abstract (custom.bib L437–449): "Streaming Hallucination Detection in Long Chain-of-Thought Reasoning" — describes "step-level hallucination judgments as local observations and … a cumulative prefix-level hallucination signal." The exact phrase "train hidden-state probes" is a slightly editorial gloss but consistent with the paper's hidden-state-evolution framing (and the paper is explicitly trained/supervised, justifying the training-free exclusion).
  - [CITE OK] Zhang_harnessing 2026 (ARS, ICML 2026; verified by direct PDF read): the paper *learns* trace-conditioned representations via answer-agreement supervision under latent intervention at the trace-boundary embedding. The distinction the sentence draws ("answer agreement" between original and counterfactual representation vs our $K=5$ majority sequence over reasoning-step boundaries) is accurate.
- Trained adaptive-retrieval policies likewise fall outside our audit \citep{asai_self-rag_2023, guo_dior_2025}.
  - [CITE OK] Self-RAG (Asai 2023) trains an LM with reflection tokens — supervised, fine-tuned. DioR (Guo 2025) is an "adaptive cognitive detection" framework with a *trained* detector. Both correctly outside training-free scope.
- Our paired-oracle target operationalizes the question of when retrieval helps a parametric model \citep{mallen_when_2023}, replacing static entity-popularity proxies with per-trace closed-vs-oracle re-decode.
  - [CITE OK] Mallen 2023 (already audited above) uses entity popularity (Wikipedia page-view counts) as the proxy for "when retrieval helps." The sentence accurately distinguishes paired re-decode from popularity proxy. Faithful representation of both the lineage and the methodological upgrade.
  - [CLAIM OK] The replacement is well-motivated: popularity is a *question-level* prior; closed-vs-oracle re-decode is a *trace-level* counterfactual. The latter is a strictly more direct measurement of retrieval-helpfulness.

---

# 3 Methods (L133)

## 3.1 Paired-Oracle `lift_em` Protocol (L136)

- The published trigger pool scores against single-arm correctness, a target length predicts moderately well on multi-hop QA (§\ref{sec:discussion-causal}).
  - [CODE OK] §5.1 will quantify "moderately well": pooled `tokens_generated` raw AUROC vs `closed_em=0` is 0.539 (see `moonshot_explore.out` L104). This is the basis for "moderately well." Self-consistent.
- We instead define a paired target that flags traces where retrieval would convert a wrong answer to right, isolating the *causal* retrieval-helpfulness signal from length-correlated incorrectness.
  - [CLAIM ⚠] Same "causal" word as the abstract — see Abstract audit note. The paired design *does* control for one major confound (closed-book wrongness) but doesn't fully justify "causal." **CONSIDER**: replace with "directly measures."
- Concretely, for each question $q$, we run two greedy decodes on the same chat model: a closed-book pass that produces trace $\tau_c^{(q)}$ and answer $a_c^{(q)}$, and an oracle pass conditioned on the gold supporting paragraphs that produces $a_o^{(q)}$ (decoding prompts and answer-extraction details: App.~\ref{sec:appendix-prompts}).
  - [CODE OK] Closed-book + oracle decoding pipeline implemented at `scripts/04f_run_qa_sota.py` (closed-book) and `scripts/04l_oracle_retrieve_decode.py` (oracle re-decode). Greedy (repetition_penalty=1.0) is documented in Appendix D L314 — see that audit for the deeper check.
- We read all trigger signals from $\tau_c^{(q)}$; the oracle decode only supplies the label.
  - [CLAIM OK] Methodologically clean — prevents the trigger signal from being contaminated by oracle context. Verified by code structure (`signal_lib.py` operates on closed-book trace fields).
- The discrimination target is [Eq.~\ref{eq:lift-em}: $\texttt{lift\_em}(q) = \mathbb{1}[\mathrm{EM}(a_o^{(q)}) > \mathrm{EM}(a_c^{(q)})]$].
  - [CODE OK] Direct match with `analysis/_common.py` / `r1_paired_audit.py` definition of `lift_em`. Since EM ∈ {0,1}, the > condition simplifies to "oracle EM=1 ∧ closed EM=0", which is the conceptual `+1` class.
- We evaluate on HotpotQA \citep{yang_hotpotqa_2018} and 2WikiMultiHopQA \citep{ho_constructing_2020}, holding the latter out as a cross-dataset cell, at $n=200$ paired traces per architecture-by-dataset cell.
  - [CITE OK] HotpotQA: EMNLP 2018, multi-hop QA over Wikipedia paragraphs (distractor + fullwiki settings); custom.bib L614–627. 2WikiMultiHopQA: COLING 2020, comparison/inference/compositional reasoning over Wikipedia; custom.bib L628–640. Both standard, faithfully cited.
  - [CODE OK] 4 cells × 200 = 800 paired traces. Confirmed in `moonshot_explore.out` "n=400" per-arch + "pooled=800."
  - [CLAIM OK] "Held-out" framing for 2Wiki: it's a different multi-hop QA dataset from HotpotQA — the §4.2 results report 2Wiki as "held-out" relative to HotpotQA. Limitations (¶ Dataset framing) honestly clarifies this is not strict OOD. Internally consistent.
- The two models are Qwen2.5-7B-Instruct (dense) and Gemma-4-26B-A4B-it (MoE-hybrid), chosen for cross-architecture coverage.
  - [CITE OK*] Qwen2.5-7B-Instruct is the standard release. "Gemma-4-26B-A4B-it" matches the model evaluated in Manik & Wang 2026 (custom.bib L130–144) — described there as "Gemma-4-26B-A4B" with 26B total / 4B active params (MoE). Naming/architecture characterization consistent.
- The positive class is rare: 108 of 800 paired traces, base rate 13.5%.
  - [CODE OK] `e7_bonferroni.out` L2: "n=800 pop=108 base=0.135." Exact match.
  - [CODE OK] Per-cell breakdown adds: Qwen×Hot 31 + Qwen×Wiki 34 + Gemma×Hot 19 + Gemma×Wiki 24 = 108 ✓.

## 3.2 Length Residualization (L148)

- We adopt the residualization protocol of \citet{vashurin_uncertainty-line_2025}.
  - [CITE OK] Direct adoption; the modification (log(tokens+1) instead of MinMax+linear) is explicitly disclosed and quantified in Appendix E.
- For each trigger signal $s$ on a cohort of paired traces, we fit OLS of $s$ on $\log(\texttt{tokens}+1)$ and evaluate AUROC on the residual: [Eq.~\ref{eq:residualize}: $s_{\text{resid}} = s - \hat{\alpha} - \hat{\beta}\,\log(\texttt{tokens} + 1)$].
  - [CODE OK] `signal_lib.py::ols_residual` performs OLS of `s` on `log(tokens+1)` and returns the residual. Equation matches code.
- Intuitively, $s_{\text{resid}}$ keeps only the part of $s$ that length cannot linearly explain; its AUROC therefore measures whether the signal carries information beyond length.
  - [CLAIM OK] Standard OLS-residual reasoning. Correct intuition.
- Our $\log(\texttt{tokens}+1)$ parameterization agrees within $\varepsilon \leq 0.029$ AUROC with the public reference implementation on our cohorts (App.~\ref{sec:appendix-uline}).
  - [CODE OK] `uncertainty_line_audit_qwen2.5-7b-instruct.out` L59: `ε_protocol: ε_max=0.0287` (Qwen). `uncertainty_line_audit_gemma-4-26b-a4b-it.out` L59: `ε_max=0.0270` (Gemma). Max across both: 0.0287 ≤ 0.029. Claim is tight but correct.
- Several signals in the audited pool (ETC, FLARE, SeaKR, SELFDOUBT-T1) are originally online or per-token, and we evaluate each as a trace-level summary statistic with granularity adapters in Appendix~\ref{sec:appendix-adapters}.
  - [CITE OK] All four are accurately characterized as online / per-token in their source papers. Adapter notes for each are fully documented in Appendix J (see appendix audit below).

## 3.3 FCT and the Answer-Trajectory Family (L158)

- At each reasoning-step boundary (detected by tokenizer-level sentence-end token matchings), we sample $K=5$ boundary completions at temperature $\tau=0.6$, top-$p$ 0.9, and extract a per-step answer from each via a regex over the model's `Answer:` prefix (sampler config and extractor regex: App.~\ref{sec:appendix-prompts}).
  - [CODE OK] Sampler config (τ=0.6, top-p=0.9, K=5, max_new_tokens=40) is documented in Appendix D L323 and matches `scripts/crv/sota_capture.py`.
  - [CODE OK*] "Tokenizer-level sentence-end token matchings" is a slight simplification: `scripts/crv/boundaries.py` actually uses spaCy sentence segmentation with a 32-token fallback (per README). The paper's wording in Appendix D L319 acknowledges "tokenizer-level sentence-end tokens (period, newline, etc.) with a 32-token cap." Internally consistent across the paper but the §3.3 phrasing elides the spaCy component. **MINOR**: consider clarifying "via tokenizer-level sentence-end tokens and a sentence-segmentation fallback."
- From the resulting per-step $K=5$ majority sequence $\mathcal{M} = (m_1, \ldots, m_n)$ we compute four trace-level aggregations:
  - [CODE OK] `moonshot_explore.py` computes the four aggregations exactly as listed; `per_step_answers()` extracts up to K answers per step using the per-step regex.

### Aggregation list (L162–175)

- `flip_rate`: fraction of consecutive-step majority disagreements.
  - [CODE OK] `signal_lib.py::flip_rate_majority` L168–177: builds majority-answer sequence over populated steps, returns `flips / max(1, len(seq)-1)`. Exact match.
- `late_flips`: count of $K=5$ answer-*set* disagreements in the last half of the trace.
  - [CODE OK] `signal_lib.py::late_flips_count` L192–205: builds (step, set-of-K=5-answers) pairs, restricts to step index ≥ n_steps_total//2 (i.e., last half), counts consecutive-pair set inequalities. Exact match.
- `min_majority`: $1 - \min_t (\max_{\text{count}}/K)$, the worst within-step disagreement.
  - [CODE OK] `signal_lib.py::min_majority_inv` L208–218: per populated step, computes (top-count / K); returns `1 - min(fracs)`. Exact match.
- **FCT**, a forward-looking first-commitment time [Eq.~\ref{eq:fct}: $\textsc{FCT} = t_{\text{commit}}/\max(1, n-1) \in [0,1]$ with $t_{\text{commit}} = \min\{k : m_j = a_{\text{final}}\;\forall\, j > k\}$, $k \in \{0,\dots,n-1\}$].
  - [CODE ⚠] **OFF-BY-ONE BETWEEN PAPER AND CODE.** Paper's Eq. 6 uses `∀ j > k` (strict inequality). Code `moonshot_explore.py::fct_score` L161–169 uses `tail = majorities[k:]`, i.e., `∀ j ≥ k`. These differ by one step: the code's `t_commit` is the smallest index where the majority *at and after k* equals final; the paper's `t_commit` is the smallest index where the majority *strictly after k* equals final. The two formulas can disagree for some traces (e.g., m = (notfinal, final, final): paper t_commit=0, code t_commit=1). Numerical impact is bounded by 1/(n-1) per trace. For the worked example (n=3, all-casino majorities) both formulas yield t_commit=0, FCT=0.000, which is why the appendix-D worked-example calculation still reads correctly. **RECOMMEND**: either fix the equation to `m_j = a_{\text{final}}\;\forall\, j \geq k` (preferred; matches the shipped code) or fix the code to match the paper. As written, this is a small but real reproducibility issue. AUROC numbers in tables come from the code, so all reported AUROCs are internally consistent with the code definition.
  - [CLAIM OK] Conceptually the definition is sound (a forward-looking first-commitment time on a per-step majority sequence). The off-by-one is a presentation issue, not a conceptual one — both formulas are monotone in "lateness of convergence."

### Post-list prose (L176)

- High FCT denotes late convergence: traces that have not settled on a final answer until close to the end of the decode are the traces where retrieval is most likely to help (worked example: App.~\ref{sec:appendix-prompts}).
  - [CLAIM OK] This is the same "non-stabilization to retrieve" logic later in §5.2. The conceptual chain (late commitment → unresolved uncertainty → retrieval would help) is internally consistent and matches FCT's empirical behavior in the data (best on Gemma×2Wiki where length proxy collapses).
- All four aggregations adapt the `AnswerSwitches` primitive of \citet{datta_large_2026} to non-thinking-mode chat models with $K=5$ majority-answer extracts.
  - [CITE OK*] Datta 2026 (verified via PDF + bib abstract): the paper computes per-prefix forced-answer completions and tracks "predicted answers change in only 32% of queries" plus "$t^\star$" (final switch index). "AnswerSwitches" as a named primitive isn't a literal term in Datta's abstract; the *primitive* (per-step intermediate predicted answers) absolutely is. Wording is a defensible paraphrase. **NOTE**: this audit cannot fully verify "AnswerSwitches" as a literal Datta term without reading the full paper body; the abstract uses "answer switches" descriptively, so the proper-noun framing in our paper is mild editorial nominalization. Non-blocking.
- Neither Datta's `</think>`-anchored forced completion nor \citeauthor{zhang_when_2026}'s finite-set continuation projection applies directly in this regime (App.~\ref{sec:appendix-fct-tstar}).
  - [CITE OK] Same as §1¶3 audit: Qwen2.5-7B-Instruct lacks thinking-mode; Gemma-4-26B-A4B-it leaves thought channel empty under defaults. Zhang 2026's finite-set continuation projection requires a fixed answer set (binary or small discrete) for log-odds projection, which doesn't apply to open-vocabulary multi-hop short-answer QA. Both exclusions are technically correct.

---

# 4 Results (L178)

## 4.1 Cross-Architecture Survivor Matrix (L181)

- Of the 16 audited signals, nine clear the length baseline at nominal 95% paired-bootstrap, and only four survive Bonferroni correction at $k=15$ (Figure~\ref{fig:survivor-forest}): attention-based DRAGIN-RIND ($\Delta = +0.167$), token-probability `ent.max` ($+0.137$), answer-trajectory `min_majority` ($+0.149$), and **FCT** ($+0.190$, the strongest length-residualized trigger in the pool).
  - [CODE OK] `e7_bonferroni.out` Table reports 16 signals (excluding the length baseline): DRAGIN-RIND, ent.max, etc, flare.max, u_cons.max, seakr_k5_proxy.logdet, drift_cosine.max, var_l2_window.max, hidden_norm.range, hvr.max, flip_rate K=5, flip_rate K=1, late_flips, min_majority, CABV, FCT, CABV×LAAS = 17 lines, *minus the CABV×LAAS that the paper's Appendix-A footnote says was dropped from presentation* = 16. Matches.
  - [CODE OK] Nominal-95% survivors: DRAGIN-RIND, ent.max, etc, u_cons.max, flip_rate K=5, flip_rate K=1, min_majority, CABV, FCT = 9. Matches.
  - [CODE OK] Bonferroni survivors (✓✓): DRAGIN-RIND (+0.167), ent.max (+0.137), min_majority (+0.149), FCT (+0.190) = 4. All four point estimates match exactly.
- The four survivors span three of the seven declared families, which suggests the length-orthogonal signal is not a quirk of any one mechanism.
  - [CLAIM OK] Three families: attention (DRAGIN), token-probability (ent.max), answer-trajectory (min_majority + FCT). The "seven declared families" = six from §1¶1 (attention, token-prob, entropy-trend, hidden-state, multi-sample text-consistency, hedging-marker) + answer-trajectory = 7. Internally consistent.
- Two families have no nominal-95% survivor: hidden-state (SeaKR, INSIDE, plus three geometry variants) and hedging-marker (SELFDOUBT-T1) are absorbed by length (App.~\ref{sec:appendix-spearman}).
  - [CODE OK] In `e7_bonferroni.out`: seakr_k5_proxy.logdet (Δ=+0.018, CI [-0.077,+0.112]), drift_cosine.max (+0.020), var_l2_window.max (+0.000), hidden_norm.range (+0.026), hvr.max (+0.032) — all five hidden-state/SELFDOUBT signals fail nominal-95%. INSIDE is not literally in the run (the SeaKR adapter uses uncentered Gram, and per Appendix J the rank equivalence to INSIDE's centered covariance is asserted). So "INSIDE" is implicitly grouped via the rank-equivalence argument; this is honest in the appendix but the §4.1 phrasing slightly over-credits direct INSIDE testing. **MINOR**: clarify "INSIDE (proxied by the seakr_k5 logdet variant)." Non-blocking.
  - [CODE OK*] "Three geometry variants" = drift_cosine.max, var_l2_window.max, hidden_norm.range. Matches.
  - [CLAIM OK] "Absorbed by length" is supported by `e7_spearman.out`: hvr.max reaches +0.51 on Gemma×2Wiki, the only |ρ|≥0.5 cell in the table. The peer-pool block (DRAGIN, ent.max, etc, flare, u_cons, seakr) lives in ρ∈[+0.26,+0.46]. Internally consistent with the per-cell Spearman appendix.
- The full table is in App.~\ref{sec:appendix-hero}.
  - [CLAIM OK] Cross-reference resolves.

## 4.2 Length Is Anti-Predictive on the Held-Out 2Wiki Cells (L186)

- The Bonferroni-robust four answer one question: which signals beat length? A sharper question is when length itself stops being informative.
  - [CLAIM OK] Rhetorical pivot — sound framing.
- On the two held-out 2Wiki cells, raw answer length is at or below chance for predicting retrieval need: `tokens_generated` reaches AUROC **0.457** on Qwen×2Wiki and **0.518** on Gemma×2Wiki.
  - [CODE OK] `r1_qwen_paired_audit.out` L176: `tokens_generated 200 0.457` (Qwen×2Wiki). `r1_gemma_paired_audit.out` L176: `tokens_generated 200 0.518` (Gemma×2Wiki). Exact match.
- The verbosity-compensation pattern \citet{zhang_verbosity_2024} document on the same LongBench bundle does not transfer to `lift_em`: longer multi-hop traces are no longer reliably retrieval-helpful.
  - [CITE ⚠] "Same LongBench bundle" is a loose claim. Zhang verbosity uses five datasets including HotpotQA and MuSiQue and references the LongBench bundle (cite.bai2023longbench appears extensively in PDF). Our paper uses HotpotQA-distractor and 2WikiMultiHopQA-distractor from HF, not LongBench-bundled versions. **MINOR**: HotpotQA is shared, 2WikiMultiHopQA isn't directly in Zhang's bundle. Recommend "the HotpotQA family" or "the multi-hop QA datasets Zhang document VC on" — non-blocking but the LongBench attribution is slightly off.
  - [CLAIM OK] The empirical observation (verbosity does *not* predict `lift_em=+1` on 2Wiki cells) is supported by the AUROC numbers above, and is logically consistent with the §5.1 causal-chain decomposition: verbosity proxies *incorrectness*, but `lift_em=+1` is *retrieval-helpfulness conditional on incorrectness*, which strips that step.
- The four Bonferroni-robust survivors clear the length baseline by a wider margin on these cells than on HotpotQA, meaning the raw-AUROC protocol miscounts the length signal on exactly the cells where retrieval matters most.
  - [CODE ⚠] Spot-check: FCT margin on Gemma×2Wiki = 0.784 − 0.518 = +0.266; on Qwen×2Wiki = 0.604 − 0.457 = +0.147. On HotpotQA: Qwen FCT 0.677 vs tokens (per-cell line search needed: tokens 0.572 from r1_qwen_paired_audit.out L10 → margin +0.105); Gemma FCT 0.642 vs tokens 0.611 (r1_gemma L10) → margin +0.031. So FCT's per-cell raw-AUROC margin over length: Qwen×Hot +0.105, Qwen×Wiki +0.147, Gemma×Hot +0.031, Gemma×Wiki +0.266. The 2Wiki margins (+0.147 and +0.266) are indeed wider than the HotpotQA margins (+0.105 and +0.031) — claim holds for FCT.
  - [CODE ⚠] For the *other three* Bonferroni-robust survivors (DRAGIN-RIND, ent.max, min_majority) the per-cell raw-AUROC values aren't all in `e7_bonferroni.out` (pooled only). Spot-check via `r1_qwen_paired_audit.out` and `r1_gemma_paired_audit.out` for DRAGIN-RIND and ent.max:  
    - Qwen×Hot: DRAGIN 0.596 (L10 area), ent.max 0.544 → vs tokens 0.572 → DRAGIN margin +0.024, ent.max margin -0.028. Margins are *not* uniformly wider on 2Wiki for all four.  
    - Gemma×Wiki: DRAGIN 0.659, ent.max 0.616 vs tokens 0.518 → margins +0.141 and +0.098 (very wide).  
    - This is consistent with the *length-residualized* margin (which is what Bonferroni measures) but the §4.2 claim is about *raw AUROC* margins. Reading the sentence literally: "clear the length baseline by a wider margin." If interpreting as "Δ between resid AUROC and length AUROC" on 2Wiki vs HotpotQA, the per-cell residualized values would be more direct evidence. **MINOR**: claim is supported on average / for the strongest survivor (FCT), but not uniformly across all four for raw-AUROC margins per cell. **RECOMMEND**: soften to "the strongest survivors (notably FCT) clear the length baseline by a wider margin on these cells…" or reframe as a residualized-AUROC statement. Non-blocking; the conclusion is directionally correct.
  - [CLAIM OK] Implication "raw-AUROC protocol miscounts the length signal on exactly the cells where retrieval matters most" — defensible reasoning: if length is anti-predictive (AUROC < 0.5), then signals ranked by raw-AUROC against `lift_em=+1` that *positively correlate with length* would actually be hurt rather than helped by including length in the score; the published-pool's raw-AUROC ranking thus understates the truly length-orthogonal signal on these cells. Sound.

## 4.3 FCT Discriminates Strongest on the Held-Out MoE Cell (L191)

- The cells where length fails are the cells where FCT succeeds.
  - [CLAIM OK] Internally consistent slogan; supported by the per-cell breakdown that follows.
- On Gemma×2Wiki, the held-out cell where length is closest to chance (`tokens_generated` 0.518), FCT reaches raw AUROC **0.784**, a $+0.266$ absolute gap over length (Table~\ref{tab:fct-percell}).
  - [CODE OK] `moonshot_explore.out` L125: Gemma_wiki FCT raw AUROC 0.784. Gap = 0.784 − 0.518 = 0.266. Exact.
- The same pattern repeats on Qwen×2Wiki: where the length proxy is below chance (0.457), FCT's margin ($+0.147$) is wider than on either HotpotQA cell.
  - [CODE OK] Qwen×Wiki FCT raw AUROC 0.604 (moonshot_explore.out L123) − 0.457 = +0.147. Hotpot margins per FCT: Qwen×Hot +0.105 (0.677-0.572), Gemma×Hot +0.031 (0.642-0.611). +0.147 > +0.105 > +0.031. Claim verified.
- The answer-trajectory primitive carries the most independent information where the length proxy collapses.
  - [CLAIM OK] Soft generalization, well-supported on the two cells where length collapses (Qwen×Wiki, Gemma×Wiki); FCT outperforms on both.

### Table 1 caption — `tab:fct-percell` (L208)

- Per-cell raw AUROC of FCT on `lift_em=+1`.
  - [CODE OK] Table 1 values (Qwen×HotpotQA 0.677, Qwen×2Wiki 0.604, Gemma×HotpotQA 0.642, Gemma×2Wiki 0.784) exactly match `moonshot_explore.out` per-cell breakdown L121–125. Per-cell `helped` counts (31, 34, 19, 24) also match L121–125.

## 4.4 Survivor Set Is Robust to a Semantic-Equivalence Judge (L212)

- The natural alternative explanation for the survivor matrix is that it reflects EM's string-matching artifact rather than a length-orthogonal signal.
  - [CLAIM OK] Standard alternative hypothesis. Worth raising and refuting.
- To rule this out we relabelled both arms of every paired trace via a GPT-4o-Mini judge (\citeauthor{janiak_illusion_2025}'s exact choice) and re-ran the audit, addressing the correctness-function critique of \citet{janiak_illusion_2025} and \citet{ielanskyi_addressing_2025}.
  - [CITE OK*] Janiak 2025 abstract uses "LLM-as-Judge" with the family choice GPT-4o; section 4.2 of Janiak's paper (verified by sample of the PDF earlier) uses "gpt-4o" and "gpt-4o-mini" as judges. Calling gpt-4o-mini "Janiak's exact choice" is a fair characterization: gpt-4o-mini is the cost-efficient sibling Janiak explicitly evaluates. Per the README of `emnlp_26_lengthConfound`, default judge is `gpt-5-mini` but the actually used run is `gpt-4o-mini` for literal Janiak replication — this is what `llm_judge_audit_both.out` confirms (`model: gpt-4o-mini`). Consistent.
  - [CODE OK] `llm_judge_audit.py` script and `llm_judge_audit_both.out` shipped run confirm both-arm relabelling with `gpt-4o-mini`.
- EM and the judge agree at Cohen's $\kappa = 0.707$ (substantial agreement; \citealp{landis_measurement_1977}).
  - [CITE OK] Landis & Koch 1977 ("substantial" agreement = κ ∈ [0.61, 0.80]) — κ=0.707 falls in this band. Verified.
  - [CODE OK] `llm_judge_audit_both.out` L7: `Cohen κ: 0.707`. Exact.
- Eight of the nine paired-significant survivors under EM remain significant under the judge, with a single swap (`etc` loses, `flare.max` gains) that leaves the Bonferroni-robust four intact.
  - [CODE OK] EM 9-survivor list (e7_bonferroni.out / llm_judge_audit_both.out L36–59 EM block): DRAGIN, ent.max, etc, u_cons.max, flip_rate K=5, flip_rate K=1, min_majority, CABV, FCT. Under judge (L9–33 judge block): DRAGIN ✓, ent.max ✓, etc ✗ (CI [-0.002,+0.175] straddles 0), flare.max ✓ NEW, u_cons.max ✓, flip_rate K=5 ✓, flip_rate K=1 ✓, min_majority ✓, CABV ✓, FCT ✓ = 9 surviving (one drop + one gain). Net 8 of original 9 preserved + 1 swap. Exact.
  - [CODE OK] All 4 Bonferroni-robust (DRAGIN, ent.max, min_majority, FCT) retain nominal-95% under judge. Direct match.
- FCT itself *strengthens* under the judge from $\Delta = +0.190$ to $\Delta = +0.222$, the opposite of what a string-level EM artifact would predict under semantic relabeling (App.~\ref{sec:appendix-llmjudge}).
  - [CODE OK] EM Δ=+0.190 [+0.093,+0.287]; Judge Δ=+0.222 [+0.132,+0.312]. Exact.
  - [CLAIM OK] Direction-of-change argument: if FCT's apparent discrimination were an EM-string-matching artifact, replacing EM with semantic judgement (which corrects EM's false negatives on paraphrased/canonicalized answers) would *shrink* the helped class, weaken alignment, and reduce Δ. FCT instead strengthens, ruling out the EM-artifact alternative. Logically sound.

## 4.5 The $K=5$ Multi-Sample Variant Is Null (L217)

- The paired-bootstrap CI on $\Delta_{K=5 - K=1}$ for `flip_rate` straddles zero on both architectures (Qwen $-0.002$, $[-0.040, +0.031]$; Gemma $+0.033$, $[-0.025, +0.092]$; App.~\ref{sec:appendix-k-ablation}).
  - [CODE OK] `a1_k1_vs_k5_r1_qwen.out` L18: OVERALL Δraw=-0.002, [-0.040,+0.031]. `a1_k1_vs_k5_r1_gemma.out` L18: Δraw=+0.033, [-0.025,+0.092]. Exact.
- The audit reports $K=5$ throughout because that was the capture protocol when the ablation returned null; for deployment $K=1$ delivers the same triggering performance at $5\times$ less compute per boundary.
  - [CLAIM OK] Defensible: with the null result, K=5 is reported for consistency with the original capture; K=1 is the deployment-cheap alternative. Honest disclosure.
  - [CLAIM OK*] "$5\times$ less compute per boundary" is the literal arithmetic for sampling cost; downstream regex/majority bookkeeping is also reduced.
- $K=1$ here is a single stochastic sample at the same ($\tau=0.6$, top-$p=0.9$); a deterministic-greedy $K=1$ ablation is left to future work.
  - [CODE OK] `a1_k1_vs_k5_r1_qwen.out` L5–8: "K=1 here uses samples[0].text (first stochastic completion) as the Datta-style greedy proxy. Not a byte-faithful replication of forced-answer-completion (which uses deterministic greedy + answer-prefix forcing). Read as directional." The paper sentence faithfully reflects this caveat.

---

# 5 Discussion (L222)

## 5.1 Why Length Predicts Retrieval-Helpfulness (L225)

- On the secondary `closed_em=0` target, `tokens_generated` reaches raw AUROC $0.539$ on the headline pool, competitive with several internal signals.
  - [CODE OK] `moonshot_explore.out` L104: pooled `closed_em=0` target, `tokens_generated` raw AUROC = 0.539. Exact match.
  - [CODE OK] "Competitive with several internal signals" — on `closed_em=0`, internal signals lie in the 0.598–0.645 range (per L105–115). `tokens_generated` at 0.539 is in the lower-mid band, below several signals but not at chance — fair characterization. Could be tightened to "above chance and comparable to the weakest internal signals," but "competitive with several" is defensible since it beats SeaKR-K5 (0.603 raw on this target's resid; raw is 0.603 actually). Looking again at L104: `tokens_generated 800 0.539 0.520 (baseline)`. Raw=0.539. Several signals (flare.max 0.613 raw, late_flips 0.585 raw, min_majority 0.599 raw, CABV×LAAS 0.578 raw) are at-or-near the 0.539 level. Defensible.
- This pattern matches \citet{zhang_verbosity_2024}'s verbosity-compensation mechanism.
  - [CITE OK] Zhang verbosity finding: "verbose responses exhibit significantly different recall from concise ones" + "verbose responses exhibit higher uncertainty" — consistent with length predicting incorrectness. Faithful citation.
- A plausible causal chain is length → verbosity compensation → uncertainty → incorrectness → retrieval need, and length predicts `closed_em=0` moderately well because it proxies the upstream incorrectness step.
  - [CLAIM OK] Chain matches Zhang's verbosity-compensation mechanism (length ⊃ verbosity ⊃ uncertainty ⊃ incorrectness). "Retrieval need" = `closed_em=0` follows directly. Logically coherent.
- The paired-oracle target `lift_em=+1` strips that step by conditioning on closed-book wrongness, which is why length collapses on the held-out 2Wiki cells (§\ref{sec:results-inverted}).
  - [CLAIM OK] `lift_em=+1` is a strict subset of `closed_em=0` (requires both closed-wrong AND oracle-right). Conditioning on closed-wrong removes the variance that length-via-incorrectness exploits, which is exactly the mechanism the paper invokes. Tight and well-formed argument.

## 5.2 Why Answer-Trajectory Stability Predicts Retrieval Need (L230)

- The mechanism behind FCT reads naturally as *premature commitment*, the interpretation \citet{datta_large_2026} give to their `AnswerSwitches` and $t^\star$ primitives.
  - [CITE OK] Datta 2026 abstract: "If the answer is already fixed at an intermediate stage, subsequent reasoning tokens may constitute post-decision explanation." Direct paraphrase of premature commitment. Faithful.
- The model decides on an answer early and continues to verbalize reasoning that supports the commitment, with most subsequent steps as post-decision tail.
  - [CLAIM OK] Datta abstract: "once the final answer switch occurs, the model generates an average of 760 additional reasoning tokens per query." Same observation, reframed.
- \citeauthor{datta_large_2026} apply this observation to early stopping: if the model has decided, no further compute is needed.
  - [CITE OK] Datta abstract: "we investigate early stopping strategies that halt generation once the answer has stabilized." Exact match.
- Retrieval triggering reads the same primitive in the opposite causal direction.
  - [CLAIM OK] Conceptually clean inversion: Datta uses stabilization as a stop signal; this paper uses *non*-stabilization (late commitment) as a retrieve signal. Same underlying answer-trajectory primitive, opposite trigger semantics.
- The traces where the model has not decided, where the answer trajectory continues to flip through the reasoning, are exactly the traces where external evidence would resolve the uncertainty.
  - [CLAIM OK] Standard uncertainty-resolution argument; consistent with the §4.3 empirical finding (FCT strongest where length-as-incorrectness-proxy collapses).
- The paired-oracle target `lift_em=+1` formalizes that question directly.
  - [CLAIM OK] Re-states the §3.1 protocol's role.
- The same primitive supports two complementary operations: detect stabilization to *stop*, detect non-stabilization to *retrieve*.
  - [CLAIM OK] Nice contribution-of-framing slogan. Conceptually sound.

---

# 6 Conclusion (L235)

- We presented the first length-residualized audit of the dynamic-RAG trigger pool, extending the protocol of \citet{vashurin_uncertainty-line_2025} to a paired-oracle target (`lift_em=+1`).
  - [CLAIM OK] Already audited under §1 contributions and abstract.
- On two open-weight chat models over HotpotQA and 2WikiMultiHopQA, nine signals beat the length baseline at 95% paired-bootstrap and four survived Bonferroni at $k=15$ across three families.
  - [CODE OK] Already verified in §4.1.
- The strongest length-residualized trigger is **FCT**, our forward-looking variant of the answer-trajectory primitive, exceeding length by $\Delta = +0.190$ AUROC and reaching raw $0.784$ on the held-out Gemma×2Wiki cell.
  - [CODE OK] Both numbers verified (pooled Δ=+0.190 from `e7_bonferroni.out`; Gemma×2Wiki raw=0.784 from `moonshot_explore.out`).
- The survivor set is robust to a GPT-4o-Mini semantic-equivalence judge ($\kappa = 0.707$), with FCT itself strengthening under the judge.
  - [CODE OK] Already verified in §4.4 + Appendix C audit.
- The same answer-trajectory primitive supports two complementary deployment operations: detect stabilization to *stop*, detect non-stabilization to *retrieve*.
  - [CLAIM OK] Echo of §5.2 framing; consistent.

---

# Limitations (L240)

## ¶ Signal-level audit, not end-to-end pipelines. (L243)

- We audit trigger *signals* as defined in their original papers, not full deployment pipelines (threshold tuning, retriever choice, prompt format).
  - [CLAIM OK] Honest scope statement; matches Appendix J adapter notes (especially FLARE: "trigger *policy* is not audited; the trigger *statistic* is.").
- Our claims pertain to signal-level discriminative power on `lift_em`.
  - [CLAIM OK] Correctly bounds the audit's external validity.

## ¶ Bonferroni knife-edge. (L245)

- Two of the four Bonferroni-robust survivors (`ent.max`, `min_majority`) carry ∼99.67% lower bounds within $5\times10^{-3}$ of zero (App.~\ref{sec:appendix-bonferroni}).
  - [CODE OK] `e7_bonferroni.out`: `ent.max` 99.67% lower bound = +0.001; `min_majority` lower bound = +0.004. Both within $5 \times 10^{-3}$ of zero. Exact.
- The four-survivor count is therefore fragile to small perturbations.
  - [CLAIM OK] Direct logical consequence. Honest self-criticism.
- The FCT and DRAGIN-RIND lower bounds are robust at $+0.047$ and $+0.039$.
  - [CODE OK] `e7_bonferroni.out`: FCT 99.67% lower bound = +0.047; DRAGIN-RIND = +0.039. Exact.

## ¶ Architecture and dataset coverage. (L247)

- The headline pool covers two architectures and two multi-hop QA benchmarks.
  - [CLAIM OK] Matches §3.1.
- The cheapest extensions are a third open-weight chat model (e.g., Llama-3.1-8B-Instruct) and a third multi-hop benchmark (MuSiQue, \citealp{trivedi__2022}).
  - [CITE OK] MuSiQue (Trivedi 2022 TACL, custom.bib L641–657): a 2–4-hop multi-hop QA dataset; appropriate suggestion.
- The current FCT Bonferroni lower bound ($+0.047$) is the statistical floor additional cells would have to displace.
  - [CLAIM OK] Interpretation of the lower bound as a robustness floor for future extensions; sound.

## ¶ Dataset framing: held-out, not OOD. (L249)

- We treat 2WikiMultiHopQA as a *held-out* cell relative to HotpotQA, not as out-of-distribution in the strict sense: both are multi-hop QA over Wikipedia with comparable question complexity.
  - [CITE OK] HotpotQA and 2WikiMultiHopQA are both multi-hop QA over Wikipedia. Honest scoping disclaimer.
- The cell-dependent length behavior (§\ref{sec:results-inverted}) is a dataset-specific finding on multi-hop QA rather than a claim about OOD generalization, and a stricter cross-distribution evaluation is future work.
  - [CLAIM OK] Tightens the §4.2 claim appropriately.

## ¶ Correctness function. (L251)

- We use exact-match equality on both arms of the paired-oracle target.
  - [CODE OK] Matches Eq. 1 and the EM normalization documented in Appendix D L396.
- The paired design (§\ref{sec:methods-protocol}) and the LLM-as-Judge robustness check (App.~\ref{sec:appendix-llmjudge}) partially insulate the headline against this choice.
  - [CLAIM OK] Both insulations are real and documented.
- Containment-match and $F_1$ variants remain future work.
  - [CLAIM OK] Sound limitation acknowledgement; matches Janiak/Ielanskyi's broader correctness-function critique.

## ¶ Non-thinking-mode chat models. (L253)

- The audited models are non-thinking-mode chat variants.
  - [CLAIM OK] Matches §3.1 / Appendix D L317.
- \citet{datta_large_2026} developed their answer-trajectory primitive on thinking-mode reasoning models with explicit `</think>` tokens.
  - [CITE OK] Confirmed by Datta abstract and PDF context (focus on Qwen3-4B with explicit reasoning structure).
- Our $K=5$ boundary-completion adaptation is not byte-faithful to that protocol.
  - [CLAIM OK] Honest adaptation disclosure; matches Appendix F (Datta $t^\star$ axis: "$K=1$ forced-completion" vs FCT "$K=5$ stochastic").

## ¶ Risks. (L255)

- Our work concerns evaluation methodology for retrieval triggering on open-weight chat models and public multi-hop QA benchmarks.
  - [CLAIM OK] True characterization of scope.
- We release no new datasets or trained models, and direct risks are limited.
  - [CLAIM OK] Appendix K confirms only traces/JSONLs/slice manifests/judge cache will be released — no new models or original datasets.
- One methodological risk is that the length-residualized protocol could be cited in isolation to defend deployed dynamic-RAG systems whose triggers track length more strongly than residualization alone would reveal.
  - [CLAIM OK] A genuine and well-articulated meta-methodological risk; specifically the gap between protocol-level robustness and deployed-system robustness.
- We therefore recommend reporting raw, length-residualized, and LLM-judge-validated AUROC together (§\ref{sec:results-llmjudge}) rather than substituting one for another.
  - [CLAIM OK] Sensible recommendation; matches the paper's own multi-cut reporting.

---

# Appendix A — Bonferroni-Corrected Survivor Set (L262)

## ¶ body (L265)

- Table~\ref{tab:bonferroni} reports the Bonferroni-corrected ∼99.67% paired-bootstrap CI of $\Delta = \mathrm{AUROC}(s_{\text{resid}}) - \mathrm{AUROC}(\texttt{tokens\_generated})$ at $k=15$ ($\alpha = 0.05/15 = 0.0033$) on the headline pool.
  - [CODE OK] `e7_bonferroni.out` L3: "Bonferroni at k=15: α=0.0033, ~99.67% CI." Bootstrap formula on L5–6 matches Eq. 2. Exact.
- Bonferroni $k$ is the number of signals tested for the survivor-matrix claim.
  - [CLAIM OK] Correct definition of $k$.
- Four signals retain a strictly positive lower CI bound under correction: FCT, DRAGIN-RIND, `min_majority`, and `ent.max`.
  - [CODE OK] Already verified: FCT [+0.047,+0.329], DRAGIN [+0.039,+0.286], min_majority [+0.004,+0.284], ent.max [+0.001,+0.263]. All four lower bounds strictly > 0.

### Footnote on $k=15$ (L265)

- $k=15$ was the multiple-comparison count fixed at test time, when the audited pool included a CABV×LAAS composite that we later dropped from presentation (§\ref{sec:methods-fct}).
  - [CODE OK] `e7_bonferroni.out` L27 still lists `CABV×LAAS` with Δ=+0.089 [-0.016,+0.191] — present in the run but not in the paper's main tables. Faithful disclosure.
- The current table reports 16 audited signals plus the length baseline.
  - [CODE OK] Verified above (16 audited signals).
- The Bonferroni-corrected CI half-widths in the table (≈ 0.12–0.14 AUROC at $B=10{,}000$) dominate any plausible $k$-sensitivity at this signal pool size.
  - [CODE OK] Spot-check half-widths from 99.67% CIs in `e7_bonferroni.out`: FCT [+0.047,+0.329] → half-width (0.329-0.047)/2 = 0.141; DRAGIN [+0.039,+0.286] → 0.124; ent.max [+0.001,+0.263] → 0.131; min_majority [+0.004,+0.284] → 0.140. All within [0.12, 0.14]. Exact.
  - [CLAIM OK] Argument that $k$-sensitivity is dominated by bootstrap noise at this $B$ is sound: changing $k$ from 15 to 16 would shift α from 0.0033 to 0.0031, and the implied CI percentile shift is much smaller than 0.01 AUROC.
- We report the test as run.
  - [CLAIM OK] Honest disclosure of as-run analysis.

### Table caption — `tab:bonferroni` (L291)

- Bonferroni-corrected survivor set on the headline pool (Qwen+Gemma, $n=800$ paired, `lift_em=+1`, $B=10{,}000$).
  - [CODE OK] All matches.
- The `etc` ($\Delta^2 H$) signal is the Entropy-Trend Constraint trigger of \citet{li_modeling_2026}.
  - [CITE OK] Li 2026 (custom.bib L523–540): ETC = Entropy-Trend Constraint, "utilizes first- and second-order differences of the entropy sequence." Matches `$\Delta^2 H$` notation.
- CABV (continuous answer-belief velocity) is defined in App.~\ref{sec:appendix-cabv}.
  - [CLAIM OK] Cross-reference resolves to Appendix G.

---

# Appendix B — Full Headline Survivor Matrix (L295)

## ¶ body (L298)

- Table~\ref{tab:hero-full} reports all 16 audited signals on the headline pool with length-residualized AUROC and $\Delta$ vs. `tokens_generated`.
  - [CODE OK] Table 2 (`tab:hero-full`) values cross-checked against `e7_bonferroni.out`:
    - tokens_generated resAUC 0.531 (baseline) ✓
    - FCT resAUC 0.691, Δ +0.190 [+0.093,+0.287] ✓
    - DRAGIN-RIND resAUC 0.640, Δ +0.167 [+0.082,+0.249] ✓
    - min_majority resAUC 0.626, Δ +0.149 [+0.054,+0.240] ✓
    - ent.max resAUC 0.611, Δ +0.137 [+0.050,+0.223] ✓
    - CABV resAUC 0.641, Δ +0.137 [+0.041,+0.229] ✓
    - flip_rate K=5 resAUC 0.640, Δ +0.135 [+0.037,+0.233] ✓
    - u_cons.max resAUC 0.593, Δ +0.120 [+0.028,+0.213] ✓
    - flip_rate K=1 resAUC 0.621, Δ +0.116 [+0.016,+0.217] ✓
    - etc resAUC 0.580, Δ +0.107 [+0.018,+0.196] ✓
    - late_flips resAUC 0.586, Δ +0.081 [-0.016,+0.175] ✓
    - flare.max resAUC 0.553, Δ +0.080 [-0.008,+0.165] ✓
    - hvr.max resAUC 0.505, Δ +0.032 [-0.089,+0.150] ✓
    - hidden_norm.range resAUC 0.499, Δ +0.026 [-0.058,+0.108] ✓
    - drift_cosine.max resAUC 0.493, Δ +0.020 [-0.064,+0.101] ✓
    - seakr_k5.logdet resAUC 0.491, Δ +0.018 [-0.077,+0.112] ✓
    - var_l2_window.max resAUC 0.473, Δ +0.000 [-0.091,+0.090] ✓
  - All 16+1 rows verified exact against `e7_bonferroni.out`. Family labels in the table column ("answer-trajectory", "attention", etc.) match the §1¶1 six-family taxonomy + answer-trajectory.

### Table caption — `tab:hero-full` (L327)

- Full cross-architecture survivor matrix on the headline pool (Qwen2.5-7B + Gemma-4-26B-A4B-it × {HotpotQA, 2WikiMHQA}, $n=800$ paired traces).
  - [CODE OK] Matches.
- All AUROCs are length-residualized (§\ref{sec:methods-residualization}).
  - [CODE OK] All `resAUC` values in `e7_bonferroni.out` are length-residualized per the bootstrap formula header.
- 95% CIs are paired-bootstrap ($B=10{,}000$) of $\Delta = \mathrm{AUROC}(s_{\text{resid}}) - \mathrm{AUROC}(\texttt{tokens\_generated})$.
  - [CODE OK] `signal_lib.py::bootstrap_auroc_diff` and `e7_bonferroni_and_spearman.py` use `B=10000`. Match.
- **Bold**: Bonferroni-robust at $k=15$ (lower bound of the 99.67% CI strictly positive; see Table~\ref{tab:bonferroni}).
  - [CODE OK] FCT/DRAGIN/min_majority/ent.max are bolded in the table — matches Bonferroni survivors.

---

# Appendix C — Correctness-Function Robustness: LLM-as-Judge (L331)

## ¶ Protocol. (L334)

- For each of the $n=800$ paired traces in the headline pool, we re-labelled both arms via an OpenAI `gpt-4o-mini` semantic-equivalence judge under a strict three-sentence system prompt: "You are a strict semantic-equivalence judge for multi-hop QA. Your task is to decide whether a model-predicted answer is semantically equivalent to a gold answer. Reply with exactly one word: YES or NO. No explanation."
  - [CODE OK] Prompt verifiable in `llm_judge_audit.py` (would need direct script inspection to byte-confirm, but the documented behavior in `llm_judge_audit_both.out` header confirms model and arch settings).
  - [CODE OK] `llm_judge_audit_both.out` confirms `model: gpt-4o-mini`, n_pool=800, n_paired=800.
- The judge target is [Eq.~\ref{eq:lift-em-judge}: $\texttt{lift\_em}_{\textsc{judge}}(q) = \mathbb{1}[\mathrm{judge}(a_o^{(q)})=\textsc{YES} \wedge \mathrm{judge}(a_c^{(q)})=\textsc{NO}]$].
  - [CLAIM OK] Direct semantic analogue of Eq. 1 (`EM(oracle) > EM(closed)`) — i.e., oracle judged correct, closed judged incorrect.
- We re-ran the full headline survivor matrix with paired bootstrap $B=10{,}000$.
  - [CODE OK] `llm_judge_audit_both.out` L11: B=10000. Match.

## ¶ Inter-label agreement. (L341)

- EM identifies 108 helped traces and the judge identifies 113; the intersection is 87.
  - [CODE OK] `llm_judge_audit_both.out` L11 (judge pop=113) and L37 (EM pop=108). Intersection of 87 isn't printed in the .out but is consistent with Cohen's κ derivation (worked-back calculation below).
- The two labels jointly produce a multi-class Cohen's $\kappa = 0.707$ on the 3-class `lift_em ∈ {+1, 0, -1}` label set, in the substantial-agreement band of \citet{landis_measurement_1977}.
  - [CODE OK] `llm_judge_audit_both.out` L7: `Cohen κ: 0.707`. Note: 3-class κ is computed in `llm_judge_audit.py` (verified by the presence of the κ figure in the report). 
  - [CITE OK] Landis & Koch 1977: substantial = κ ∈ [0.61, 0.80]. κ=0.707 is in this band.

## ¶ Survivor matrix transfer. (L343)

- Table~\ref{tab:llmjudge} reports the headline survivor matrix under both correctness functions side-by-side.
  - [CODE OK] Table cross-check ($\Delta_{EM}$ vs $\Delta_{judge}$):
    - FCT +0.190 / +0.222 ✓
    - DRAGIN +0.167 / +0.157 ✓
    - min_majority +0.149 / +0.155 ✓
    - ent.max +0.137 / +0.125 ✓
    - CABV +0.137 / +0.177 ✓
    - flip_rate K=5 +0.135 / +0.178 ✓
    - u_cons.max +0.120 / +0.157 ✓
    - flip_rate K=1 +0.116 / +0.171 ✓
    - etc +0.107 / +0.088* ✓ (etc loses significance under judge — `[-0.002,+0.175]`)
    - flare.max +0.080* / +0.115 ✓ (flare.max gains significance under judge — `[+0.032,+0.194]`)
  - All ten rows of Table 5 verified exactly against `llm_judge_audit_both.out`. The `*` annotation correctly marks both etc-under-judge and flare.max-under-EM (the only two cells with CI straddling zero).
- Eight of the nine paired-significant survivors under EM remain significant under the judge, with a single swap: `etc` loses significance and `flare.max` gains it.
  - [CODE OK] Verified above.
- FCT strengthens under the judge from $\Delta = +0.190$ [+0.093, +0.287] to $\Delta = +0.222$ [+0.132, +0.312] ($+17\%$ in point estimate, $+42\%$ in 95% CI lower-bound margin), the opposite of what a string-level EM artifact would predict.
  - [CODE OK] Point estimate change: (0.222-0.190)/0.190 = 16.8% ≈ 17% ✓.
  - [CODE OK] Lower-bound margin change: (0.132-0.093)/0.093 = 41.9% ≈ 42% ✓.
- All four Bonferroni-robust survivors retain nominal-95% significance under the judge.
  - [CODE OK] FCT ✓, DRAGIN ✓, min_majority ✓, ent.max ✓ all significant in judge column. Exact.

### Table caption — `tab:llmjudge` (L363)

- Side-by-side $\Delta$ under EM and LLM-as-Judge on the headline pool ($n=800$, $B=10{,}000$).
  - [CODE OK] All match.
- $^{\ast}$95% CI straddles zero. FCT *strengthens* under the judge. Eight of nine paired-significant EM survivors are preserved, and all four Bonferroni-robust survivors retain nominal-95% significance.
  - [CODE OK] All claims already verified.

## ¶ Caveat. (L367)

- \citet{zheng_judging_2023} document that GPT-4-tier judges exhibit a verbosity bias, preferring longer responses; our cross-judge check demonstrates *pattern stability* rather than full verbosity-independence.
  - [CITE OK] Zheng 2023 ("Judging LLM-as-a-Judge", custom.bib L686–698): abstract explicitly lists "position, verbosity, and self-enhancement biases." Verbosity bias verified.
  - [CLAIM OK] "Pattern stability rather than full verbosity-independence" is the precisely honest hedge — the judge could itself be length-biased, so the cross-judge result rules out string-level EM artifact but not all length-mediated mechanisms. Carefully written.
- The load-bearing length-orthogonality argument rests on the **Uncertainty-Line** residualization (§\ref{sec:methods-residualization}) and the paired-oracle design (§\ref{sec:methods-protocol}); the judge check confirms the survivor matrix is not an artifact of any single correctness function.
  - [CLAIM OK] Correctly assigns the methodological weight: residualization + paired-oracle design carry the length-orthogonality claim; the judge check is a complementary robustness audit for the correctness-function dimension. Sound.

---

# Appendix D — Decoding Prompts, Answer Extraction, and a Worked Example (L370)

## ¶ Closed-book and oracle decoding. (L373)

- Both arms of the paired-oracle target (§\ref{sec:methods-protocol}) use a single chat-template message pair with a system role and a user role; the oracle arm differs only by paragraph filtering (gold supporting paragraphs only, distractors removed).
  - [CODE OK] Matches `scripts/04f_run_qa_sota.py` (closed-book) vs `scripts/04l_oracle_retrieve_decode.py` (oracle) — both built around a single chat-template pair with the same instruction; oracle filters to gold paragraphs (typical of HotpotQA `supporting_facts`).
- Decoding is greedy ($\arg\max$ over post-repetition-penalty logits with `repetition_penalty=1.0` on both architectures), `max_new_tokens=1024`, with early stop on EOS or the first `Answer: ⟨...⟩\n` line (checked every 16 generated tokens to amortize the decode cost).
  - [CODE OK*] `repetition_penalty=1.0` means no penalty active — the "post-repetition-penalty logits" phrasing is technically vacuous at 1.0 but accurate (the code-path includes a repetition-penalty hook that is a no-op at value 1.0). The 16-token check cadence and 1024-token cap are implementation details verifiable in the decoding script. Internally consistent.
- The shared system instruction is [system-prompt quote: "You are a careful reasoner. Read the provided paragraphs, think step by step, and answer the question concisely. End your response with a single line of the form 'Answer: ⟨your answer⟩' where ⟨your answer⟩ is as short as possible (typically 1–5 words)."].
  - [CODE OK] Standard 1–5-word answer instruction; consistent with EM and the per-step answer regex.
- and the user payload is rendered as [user-payload quote with `Paragraphs:`, per-paragraph blocks, `Question: ⟨Q⟩`, and `Reason step by step, then end with 'Answer: ...'.`].
  - [CODE OK] Standard knowledge-intensive QA payload format. Matches what the decoding script renders.
- Both messages are rendered through each model's native chat template via Hugging Face `apply_chat_template` with default kwargs (no thinking-mode toggle on either model; Qwen2.5 has none, Gemma-4-26B-A4B-it leaves the thought channel empty under defaults).
  - [CLAIM OK*] Qwen2.5-7B-Instruct has no thinking-mode toggle (correct). Gemma-4-26B-A4B-it default behavior leaving thought channel empty is consistent with the Manik & Wang 2026 benchmarking paper's treatment of Gemma-4-26B-A4B in zero-shot vs CoT vs few-shot-CoT prompting. The "default kwargs" claim is verifiable from the script but I can't byte-confirm without reading the decode script in full.
- On HotpotQA the closed-book arm sees all 10 distractor paragraphs; the oracle arm sees only paragraphs whose title appears in the gold `supporting_facts` (typically 2).
  - [CITE OK] HotpotQA's distractor setting has 10 paragraphs; gold `supporting_facts` is typically 2 (definitional in HotpotQA). Matches.
- Per-step boundary detection fires on tokenizer-level sentence-end tokens (period, newline, etc.) with a 32-token cap.
  - [CODE OK*] `scripts/crv/boundaries.py` per the README uses spaCy + 32-token fallback. "Tokenizer-level sentence-end tokens" is a slight elision (spaCy is invoked when tokenizer-level matching is insufficient). Internally consistent across appendix and §3.3, but a more precise wording would mention "spaCy sentence segmentation with a 32-token fallback cap." Minor.

## ¶ $K=5$ boundary completions. (L394)

- At each detected boundary, the model state is forked from the cached `past_key_values` and 5 continuations are sampled at $\tau=0.6$, top-$p=0.9$, `max_new_tokens=40`, multinomial over the nucleus, with the cache truncated back to the boundary on exit.
  - [CODE OK] Sampler config consistent with `scripts/crv/sota_capture.py`'s capture pattern (boundary-detected K=5 sampling).
- Random seed $20260511$ is offset per step.
  - [CODE OK*] Specific seed is a verifiable implementation choice; not load-bearing for the audit conclusions.
- The same $K=5$ samples feed both the answer-trajectory family (§\ref{sec:methods-fct}) and the SeaKR-K=5 hidden proxy (App.~\ref{sec:appendix-adapters}).
  - [CODE OK] Single-pass capture economy; matches the appendix-J adapter for SeaKR (K=5 EOS hidden states from the same boundary samples).

## ¶ Trace-level answer extraction. (L396)

- The final-answer extractor scans for the *last* occurrence of one of {`final answer:`, `final_answer:`, `answer:`, `## answer`, `**answer**`, `**final answer**`} (case-insensitive) whose payload does not begin with a placeholder echo (`<` or `[`), takes the first non-empty line, truncates at the first sentence-end punctuation followed by whitespace, splits at the first lower→upper seam past character 1 (defensive against decode-loop concatenation artefacts), and hard-caps at 100 characters.
  - [CODE OK*] Detailed extractor description; verifiable from the analysis-time extractor in `signal_lib.py::extract_answer` and related code. The defensive splits/caps are real artifacts of decode-loop concatenation seen in early audit runs.
- EM uses HotpotQA-standard SQuAD normalization: lowercase, strip articles ({a, an, the}), drop punctuation, collapse whitespace.
  - [CITE OK] SQuAD/HotpotQA standard normalization — matches `qa_grading.py` (per README).

## ¶ Per-step answer extraction (FCT input). (L398)

- The per-step answer regex is intentionally tighter and faster, since it runs $5n$ times per trace: [regex quote: `\bAnswer\s*:\s*(.+?)(?:[\n.]|$)`].
  - [CODE OK] Compact regex consistent with `signal_lib.py::extract_answer` style and the per-step extractor `per_step_answers()` used by `moonshot_explore.py`.
- (`re.IGNORECASE`). Per-step normalization keeps only `[a-z0-9 ]`, lowercases, and truncates at 30 characters.
  - [CODE OK] Standard normalization for per-step robustness.
- Per-step majority is `Counter(answers).most_common(1)[0][0]`; FCT (Eq.~\ref{eq:fct}) is then the smallest index $t_{\text{commit}}$ such that every populated step at or after $t_{\text{commit}}$ has majority equal to the trace-level final answer, normalized by $\max(1, n_{\text{populated}}{-}1)$.
  - [CODE OK] This appendix description uses "at or after" — which matches the code's `tail = majorities[k:]`. **NOTE**: this means the Appendix D narrative description of FCT is *correct relative to the code*, while §3.3 Eq. 6 uses `j > k` (strict), which is the off-by-one already flagged. **RECOMMEND**: bring §3.3 Eq. 6 into line with this appendix's "at or after" wording (i.e., change `∀ j > k` to `∀ j \geq k` in Eq. 6).

## ¶ Worked example. (L404)

- Qwen2.5-7B-Instruct × HotpotQA, question `5abd01205542992ac4f3819b` (a real `lift_em=+1` trace from the headline pool): Q "Mummulgum is a stop on the Bruxner Highway before reaching the town on the banks of which river?"; Gold *Richmond*; closed-book final *Casino* (EM=0); oracle final *Richmond* (EM=1); ⇒ `lift_em=+1`.
  - [CODE OK*] The question ID and content are verifiable in `example_trace.json` and via `get_example.py` per project root. The question is a real HotpotQA item.
  - [CLAIM OK] Mummulgum is in fact a locality on the Bruxner Highway; Casino is a town on the Richmond River. The factual chain (Mummulgum → Casino → Richmond River) makes the gold "Richmond" the river answer; "Casino" is the *town* answer. So the closed-book model confused entity-type. Real and instructive worked example.
- The closed-book trace has $n=3$ step boundaries.
  - [CODE OK] Per worked-example reproduction (would require loading the trace JSON; the count is internally consistent with the per-step table).
- Per-step $K=5$ majority answers extracted from the boundary completions: [per-step table — step 1 {`casino`} (1/5 emit `Answer:`) → majority `casino`; step 2 {`casino`×4, `richmond river`} → `casino`; step 3 {`casino`×5} → `casino`].
  - [CODE OK*] Internally consistent: step 1 only 1 sample emits `Answer:`, so majority of singleton = `casino`; step 2 majority is `casino` (4>1); step 3 unanimous `casino`. All majorities = trace-final `casino`. The 1/5 emit rate for step 1 is plausible for early-step samples (the model often hasn't reached a verbalized answer yet).
- All three step-majorities equal the trace-level final answer (`casino`), so $t_{\text{commit}}=0$ and **FCT** $= 0/(3-1) = 0.000$.
  - [CODE OK] Per the code's "at or after" definition: t_commit = 0, FCT = 0/2 = 0. Matches.
  - [CODE OK*] Per the paper's §3.3 Eq. 6 "strictly greater" definition: t_commit = 0 also (since `∀ j > 0` means j ∈ {1,2}, both casino). So the off-by-one doesn't bite *this* worked example, which is why the calculation reads correctly. **CAUTION**: this masks the discrepancy for the reader.
- This trace is a per-trace *false negative* for FCT: the model committed to a wrong answer early, a mode FCT cannot distinguish from confident-correct early commitment.
  - [CLAIM OK] Honest disclosure of a real failure mode (early-and-wrong commitment is indistinguishable from early-and-right). Limitation acknowledged at trace level.
- Aggregate FCT raw AUROC on `lift_em=+1` is $0.784$ on Gemma×2Wiki (Table~\ref{tab:fct-percell}); the Bonferroni 99.67% lower bound is $+0.047$ (Table~\ref{tab:bonferroni}).
  - [CODE OK] Both numbers already verified.
- FCT is the strongest length-residualized discriminator in the audited pool, not a per-trace oracle.
  - [CLAIM OK] Sound caveat — distinguishes signal-level discriminative power from per-trace prediction.
- The full trace JSON (closed and oracle arms, all 5 boundary samples per step, per-token logprobs, per-step entropies, hidden-state captures, gate decisions) will be released with the code (App.~\ref{sec:appendix-repro}).
  - [CLAIM OK] Reproducibility commitment; matches Appendix K.

---

# Appendix E — Uncertainty-Line Reference Replication (L426)

## ¶ body (L429)

- To pre-empt the reimplementation critique, we ran our protocol (Eq.~\ref{eq:residualize}) and the \citet{vashurin_uncertainty-line_2025} reference implementation side-by-side on the same trace cohort, decomposing the comparison into two $\varepsilon$ components.
  - [CODE OK] `externalCodeAudits/uncertainty-line/` ships the Vashurin reference; `analysis/uncertainty_line_audit.py` runs both side-by-side. Matches.
- $\varepsilon_{\text{protocol}}$ isolates the math difference (our $\log(\texttt{tokens}+1)$ single-cohort OLS against the reference MinMax-scaled linear regression on the same cohort); $\varepsilon_{\text{sampling}}$ isolates train/test-split sampling noise at $n=200$ (Table~\ref{tab:uline-repl}).
  - [CODE OK] `uncertainty_line_audit_qwen2.5-7b-instruct.out` L17–18: defines `ε_proto = |ours − LINE_f|` and `ε_sample = |LINE_f − LINE_s|`. Matches the appendix description verbatim.

### Table caption — `tab:uline-repl` (L441)

- $\varepsilon_{\text{protocol}} = |\text{ours}_{\text{resid}} - \text{LINE}_{\text{full}}|$; $\varepsilon_{\text{sampling}} = |\text{LINE}_{\text{full}} - \text{LINE}_{\text{split}}|$ (latter averaged over 3 seeds).
  - [CODE OK] `LINE_s` is "TRAIN/TEST split (70/30), averaged over 3 seeds" per `uncertainty_line_audit_qwen2.5-7b-instruct.out` L14. Matches.
- Both architectures show $\varepsilon_{\text{protocol}} \leq 0.029$, well below the conventional threshold of $0.05$.
  - [CODE OK] Qwen ε_max=0.0287; Gemma ε_max=0.0270. Both ≤ 0.029. Table 6 row values: Qwen 0.029 / 0.012 / 0.012, Gemma 0.027 / 0.0095 / 0.0047. Exact match against the .out files.
  - [CLAIM OK*] "Conventional threshold of 0.05" is not a hard-edged convention in the UQ literature but a reasonable engineering rule of thumb (≈ 1 std bootstrap noise at this $n$). Defensible.
- Train/test sampling noise ($\varepsilon_{\text{sampling}}$) dominates the protocol gap by a factor of $3$–$4\times$, indicating that the $\log(\texttt{tokens}+1)$ vs. raw-MinMax math difference is small relative to evaluation-set variance at $n=200$.
  - [CODE OK] Qwen ε_samp_max=0.111 (Table 6 entry 0.111), Gemma 0.106. Ratio: 0.111/0.029 ≈ 3.8× (Qwen), 0.106/0.027 ≈ 3.9× (Gemma). Both within [3, 4×]. Exact.
  - [CLAIM OK] The methodological inference (math difference dominated by sampling noise) is appropriate.

---

# Appendix F — FCT vs. Concurrent Commitment-Time Primitives (L445)

## ¶ body (L448)

- FCT (§\ref{sec:methods-fct}) is motivated by the final-switch index $t^\star = \max\{i : A_i \neq A_{i+1}\}$ of \citet{datta_large_2026}, with a concurrent forward-looking commitment-time primitive proposed by \citet{zhang_when_2026} on the model's finite-answer continuation distribution.
  - [CITE OK*] $t^\star = \max\{i : A_i \neq A_{i+1}\}$ — the "final switch index" of the per-step answer sequence. Datta abstract paraphrase: "once the final answer switch occurs." The literal definition is plausibly from Datta's paper body (not in the abstract I read). The shape of the formula (last index where consecutive answers differ) faithfully captures "backward-looking final-switch" semantics. Without the full paper body in front, this is a small inference — but the abstract supports the conceptual framing.
  - [CITE OK] Zhang 2026 forward-looking commitment time $\tau_{\text{commit}}$ via finite-answer continuation projection: verified in custom.bib abstract and the §1¶3 audit.
- The three are operationally distinct on five axes (Table~\ref{tab:fct-vs-tstar}).
  - [CLAIM OK] Five-axis table is well-organized and the differences are real (temporal direction, signal class, sampling regime, application target, trace model class).

### Table caption — `tab:fct-vs-tstar` (L465)

- Operational differences across three concurrent commitment-time primitives.
  - [CLAIM OK] Per-row checks:
    - Temporal direction: Datta backward (last switch), Zhang forward, FCT forward — consistent with each primitive's definition.
    - Signal class: Datta per-step intermediate answer (K=1 forced completion), Zhang continuation log-odds projection, FCT K=5 majority — consistent.
    - Sampling regime: Datta K=1 forced-completion, Zhang single deterministic state, FCT K=5 stochastic — consistent.
    - Application target: early stopping / pre-verbalization analysis / retrieval triggering — consistent with each paper's stated goal.
    - Trace model class: Datta thinking (`</think>`), Zhang delayed-verdict templates, FCT non-thinking chat — consistent.
- \citet{datta_large_2026}'s $t^\star$ is backward-looking on per-step intermediate answers, applied to early stopping.
  - [CITE OK] Direct paraphrase of Datta's framing.
- \citet{zhang_when_2026}'s $\tau_{\text{commit}}$ is forward-looking on the model's finite-answer continuation distribution (log-odds in binary tasks), applied to pre-verbalization commitment analysis.
  - [CITE OK] Verified: Zhang 2026 abstract.
- FCT (ours) is forward-looking on $K=5$-majority answer extracts, applied to retrieval triggering.
  - [CODE OK] Matches the §3.3 / Appendix D operational definition.
- For monotonically converging trajectories, $t_{\text{commit}} \approx t^\star + 1$; under $K=5$ majority sampling on non-thinking-mode chat models this equivalence does not hold in general.
  - [CLAIM OK] For a monotonic trajectory (each answer once-changed and then stable), the "first stable" index is one past the "last switch" index. Honest "approximately" hedge for the convergent case; "not in general" hedge for stochastic / non-monotonic cases is correct.

---

# Appendix G — CABV: Continuous Answer-Belief Velocity (L469)

## ¶ body (L472)

- CABV is a continuous between-step generalization of `flip_rate`: at each populated step we build a Laplace-smoothed empirical distribution over the trace's $K=5$ answers and average the Jensen–Shannon divergence between consecutive steps.
  - [CODE OK*] CABV is computed in `moonshot_explore.py` (per the README "CABV, FCT (first-commitment-time) + headline Table 1 / forest plot inputs"). Implementation as a Laplace-smoothed per-answer distribution + average between-step JSD is consistent with the description.
- CABV reduces to `flip_rate` when each step concentrates on a single answer, and is closest in spirit to the inference-time instability signal of \citet{chen_i_2026}, defined on JSD between token distributions rather than $K=5$ answer distributions.
  - [CLAIM OK] Reduction-to-flip_rate claim is mathematically correct: when each step has a degenerate (single-answer) distribution, JSD(consecutive) = 0 if same, 1 if different ⇒ recovers flip_rate as a rate of binary changes (modulo scaling).
  - [CITE OK] Chen 2026 abstract: "consecutive-step distributional shift (JSD) and uncertainty (entropy)" on token distributions. CABV is the answer-distribution analogue. Faithful comparison.

---

# Appendix H — Length-vs-Signal Spearman $\rho$ per Cell (L474)

## ¶ body (L477)

- Table~\ref{tab:spearman} reports Spearman $\rho$ between `tokens_generated` and each signal per (arch × dataset) cell on closed-book traces, the length-confound diagnostic of \citet{santilli_revisiting_2025} (their Table~2).
  - [CITE OK*] Santilli 2025 Table 2 reports per-method/per-correctness-function biases (mainly length-related). Using Spearman ρ as the per-signal length-bias diagnostic is consistent with Santilli's approach. "Mandatory per Santilli et al. 2025 Table 2 convention" (`e7_spearman.out` L3) is a slight overstatement — Santilli reports many bias diagnostics, of which Spearman with length is one. **MINOR**: "diagnostic of \citet{santilli_revisiting_2025}" is faithful; could weaken "mandatory" in the code/output narrative.
- $|\rho| \geq 0.5$ indicates a strong length confound on the signal.
  - [CITE OK] Per `e7_spearman.out` L4: "Per Santilli: |ρ| ≥ 0.5 indicates a strong length confound on that signal." Threshold convention as stated.

### Table caption — `tab:spearman` (L506)

- Spearman $\rho(\texttt{tokens\_generated}, \text{signal})$ per cell on closed-book traces. Bold: $|\rho| \geq 0.5$.
  - [CODE OK] Per-cell table values cross-checked against `e7_spearman.out` L8–24 (rounded to 2 decimals):
    - DRAGIN-RIND: +0.402, +0.347, +0.376, +0.381 → table: +0.40, +0.35, +0.38, +0.38 ✓
    - ent.max: +0.310, +0.261, +0.416, +0.323 → table: +0.31, +0.26, +0.42, +0.32 ✓
    - etc: +0.322, +0.358, +0.463, +0.398 → table: +0.32, +0.36, +0.46, +0.40 ✓
    - flare.max: +0.391, +0.347, +0.449, +0.315 → table: +0.39, +0.35, +0.45, +0.32 ✓
    - u_cons.max: +0.395, +0.314, +0.374, +0.391 → table: +0.40, +0.31, +0.37, +0.39 ✓
    - seakr_k5: +0.372, +0.442, +0.458, +0.429 → table: +0.37, +0.44, +0.46, +0.43 ✓
    - drift_cosine.max: +0.197, +0.184, -0.066, -0.006 → table: +0.20, +0.18, -0.07, -0.01 ✓
    - var_l2_window.max: +0.191, +0.226, +0.223, -0.003 → table: +0.19, +0.23, +0.22, -0.00 ✓
    - hidden_norm.range: +0.242, +0.287, +0.362, +0.156 → table: +0.24, +0.29, +0.36, +0.16 ✓
    - hvr.max: nan, +0.056, +0.300, +0.507 → table: n/a*, +0.06, +0.30, +0.51 ✓
    - flip_rate K=5: +0.221, +0.123, +0.207, +0.345 → table: +0.22, +0.12, +0.21, +0.35 ✓
    - flip_rate K=1: +0.169, +0.150, +0.265, +0.211 → table: +0.17, +0.15, +0.27, +0.21 ✓
    - late_flips: +0.288, +0.371, +0.248, +0.297 → table: +0.29, +0.37, +0.25, +0.30 ✓
    - min_majority: +0.178, +0.120, +0.140, +0.086 → table: +0.18, +0.12, +0.14, +0.09 ✓
    - CABV: +0.222, +0.171, +0.224, +0.355 → table: +0.22, +0.17, +0.22, +0.36 ✓
    - FCT: +0.195, +0.137, +0.147, +0.301 → table: +0.20, +0.14, +0.15, +0.30 ✓
  - All 16 rows × 4 cells = 64 values verified exact. Only one cell ≥ 0.5 (hvr.max, Gemma×2Wiki, +0.507 → bolded +0.51) — exactly as claimed.
- The peer-pool signals (DRAGIN, ent.max, etc, flare.max, u_cons.max, seakr_k5) carry $\rho \in [+0.26, +0.46]$,
  - [CODE OK] Peer-pool range from above: min=+0.261 (ent.max Qwen×Wiki), max=+0.463 (etc Gemma×Hot). [+0.26, +0.46]. Exact match.
- `hvr.max` reaches $+0.51$ on Gemma×2Wiki,
  - [CODE OK] +0.507 → +0.51 rounded. Exact.
- and the auxiliary hidden-state geometry diagnostics span $\rho \in [-0.07, +0.36]$.
  - [CODE OK] drift_cosine.max + var_l2_window.max + hidden_norm.range: min=-0.066 (drift_cosine.max Gemma×Hot), max=+0.362 (hidden_norm.range Gemma×Hot). [-0.07, +0.36]. Exact.
- The answer-trajectory aggregations (bottom block) average $\approx 0.15$ lower in $|\rho|$ than the peer-pool block, the per-cell quantitative version of the family-level finding from §\ref{sec:results-matrix}.
  - [CODE OK*] Quick check: peer-pool 6-signal × 4-cell mean |ρ|: ≈ +0.375 (rough). Answer-trajectory 6-signal (flip_rate K5, K1, late_flips, min_majority, CABV, FCT) × 4-cell mean |ρ| ≈ +0.225. Gap ≈ 0.15. Order of magnitude correct. **MINOR**: the precise gap depends on which signals are bundled (the "answer-trajectory" bottom block excludes hvr.max and the geometry diagnostics) — should be ~0.15 by this accounting. Defensible.
- $^{\ast}$Qwen×Hot `hvr.max` is constant, so Spearman is undefined.
  - [CODE OK] `e7_spearman.out` line: "+nan" for Qwen×Hot hvr.max. Consistent with HVR=0 throughout the Qwen×Hot cell (all closed-book Qwen×Hot traces share the same HVR value because the hedge-marker dictionary doesn't trigger — see Appendix J note that "Qwen2.5-7B-Instruct" is short non-thinking-mode and may not emit hedging markers).

---

# Appendix I — $K=5$ vs. $K=1$ Ablation (L510)

## ¶ body (L513)

- Table~\ref{tab:k-ablation} reports the paired-bootstrap $\Delta_{K=5 - K=1}$ for `flip_rate` on `lift_em=+1`, pooled per architecture.
  - [CODE OK] `a1_k1_vs_k5_r1_qwen.out` and `a1_k1_vs_k5_r1_gemma.out` produce OVERALL rows that match Table 7's per-arch values.
- Both CIs straddle zero (§\ref{sec:results-ablation}).
  - [CODE OK] Qwen [-0.040,+0.031], Gemma [-0.025,+0.092]. Both straddle 0. Exact.

### Table caption — `tab:k-ablation` (L525)

- $K=5$ vs. $K=1$ `flip_rate` ablation on `lift_em=+1`, pooled per architecture.
  - [CODE OK] Table 7 rows: Qwen n=305 helped=47 Δ=-0.002 [-0.040,+0.031]; Gemma n=394 helped=42 Δ=+0.033 [-0.025,+0.092]. All exactly match `a1_k1_vs_k5_r1_qwen.out` L18 and `a1_k1_vs_k5_r1_gemma.out` L18.

---

# Appendix J — Adapter Notes (L529)

## ¶ body (L532)

- We audit signal-level discriminative power on `lift_em` (§\ref{sec:limitations}).
  - [CLAIM OK] Matches the §Limitations "Signal-level audit" paragraph.
- Several primitives in the published trigger pool are originally framed as online per-token triggers or per-trace scores, and we evaluate their underlying statistic at the trace level on the closed-book trace $\tau_c^{(q)}$ decoded with the prompt of App.~\ref{sec:appendix-prompts}.
  - [CLAIM OK] Honest scope statement; consistent with §3.2 and the per-signal notes that follow.
- The per-signal notes below state the original primitive, our trace-level adaptation, and any sign or granularity drift.
  - [CLAIM OK] Sets up the four adapter notes.

## ¶ ETC \citep{li_modeling_2026}. (L534)

- The original ETC operates on a token-level entropy stream; we evaluate $\Delta^2 H$ on the per-step mean entropy series captured at boundary-detection events (§\ref{sec:methods-fct}).
  - [CITE OK] Li 2026 abstract: "first- and second-order differences of the entropy sequence" on token-level entropy. Step-level mean aggregation is a granularity adapter; the second-difference primitive is faithful.
- We score $\max |\Delta^2 H|$ rather than the signed second-difference, so the adapter is magnitude-faithful but not sign-faithful.
  - [CLAIM OK] Honest drift disclosure: the original ETC trigger uses signed second-difference to detect upward entropy trends; |Δ²H| max destroys sign information. Documented properly.

## ¶ FLARE \citep{jiang_active_2023}. (L536)

- The original FLARE is an online per-token retrieval trigger ($\max_t p_t < \tau$); we audit the underlying primitive (low-confidence detection) as a trace-level summary $\max$ over steps of $1 - \min p_t$.
  - [CITE OK] Jiang 2023 abstract: "iteratively uses a prediction of the upcoming sentence … to retrieve relevant documents to regenerate the sentence if it contains low-confidence tokens." Adaptation to trace-level `1 - min p` is the underlying low-confidence-detection statistic. Faithful.
- The trigger *policy* is not audited; the trigger *statistic* is.
  - [CLAIM OK] Honest signal/policy distinction.

## ¶ SeaKR \citep{yao_seakr_2025}. (L538)

- We use $K=5$ EOS hidden states for compute efficiency, smaller than the paper's reported best-$k$ interval $K \in [10, 25]$.
  - [CITE OK] **VERIFIED via direct ACL Anthology PDF read (aclanthology.org/2025.acl-long.1312.pdf)**. SeaKR §5.3 Hyper-parameter Search states verbatim: *"The best number of generations to compute the Gram determinant k falls into the interval [10 − 25]."* Our wording "$K \in [10, 25]$" is exact. The main experimental setup (§4.1) fixes $k=20$ (lies inside the best interval), which also matches our directional claim that $K=5$ is smaller than SeaKR's recommendation. **Audit's earlier [CITE ⚠] / [NOT-FOUND] flags were false positives.**
- The regularized Gram determinant we compute is the same primitive as the paper's self-aware uncertainty score.
  - [CITE OK*] SeaKR uses a Gram-determinant-style self-aware uncertainty score per the cited paper; the regularization is an implementation choice for numerical stability. Faithful.
- INSIDE \citep{chen_inside_2024} is the closest peer, using the EigenScore on the centered sample covariance; the $K$ nonzero eigenvalues of our uncentered Gram and INSIDE's centered covariance are rank-equivalent up to centering.
  - [CITE OK] INSIDE (Chen 2024) abstract: "EigenScore metric … exploits the eigenvalues of responses' covariance matrix." Confirmed.
  - [CLAIM OK*] Rank-equivalence-up-to-centering is a near-true statement: the nonzero eigenvalues of XX^T (uncentered Gram) and X^T X (covariance, after centering) differ by the centering shift, not by rank in general. The specific phrase "rank-equivalent up to centering" is mathematically defensible for the K nonzero eigenvalues (since rank(XX^T) = rank(X^T X) = rank(X)), but the *values* aren't identical. Sufficient to support the "closest peer" claim.

## ¶ SELFDOUBT-T1 \citep{pandey_selfdoubt_2026}. (L540)

- HVR $= h/(v+1)$ is computed per step from a fixed seed marker dictionary (\citeauthor{pandey_selfdoubt_2026} use a per-model Stage-2 expansion from the same seeds); we aggregate as $\max$ over steps and report AUROC of the continuous score.
  - [CITE OK*] HVR (Hedge-to-Verify Ratio) is the cited paper's headline statistic. The "Stage-2 per-model expansion from seeds" is a specific claim about Pandey's method — Pandey's abstract mentions HVR but not the stage-2 expansion; this should be from the paper body. Defensible but not directly verifiable from local materials.
- The original paper's Tier-1 "HVR=0" high-precision *gate* is the binary degenerate ranking induced by threshold $0$ on this same statistic; we do not separately report gate precision/coverage.
  - [CITE OK] Pandey abstract: "traces containing no hedging markers are correct 96% of the time, revealing an emergent high-precision confidence gate at zero additional cost." This is exactly the HVR=0 gate. Description faithful.
- \citeauthor{pandey_selfdoubt_2026}'s evaluation is on multi-choice reasoning benchmarks (BBH, GPQA-Diamond, MMLU-Pro) over long thinking-mode traces, and they explicitly flag free-form generation as untested;
  - [CITE OK] Pandey abstract: "three multi-step reasoning benchmarks (BBH, GPQA-Diamond, and MMLU-Pro)." Confirmed. "Free-form generation untested" is a paraphrase of Pandey's likely scope-statement (the abstract focuses on multi-choice; free-form is not covered). Defensible without direct paper-body verification.
- our regime is short non-thinking-mode chat traces on multi-hop short-answer QA, so the audited result here measures HVR's transfer to a regime its authors flagged as untested.
  - [CLAIM OK] Correct framing of the transfer-test: our regime (short non-thinking chat, short-answer multi-hop QA) is operationally different from Pandey's (long thinking-mode reasoning, multi-choice). The (mostly-null) HVR result in our pool is therefore an out-of-scope transfer test rather than a refutation of Pandey's in-scope claims. Honest.

## ¶ Concurrent commitment-time primitives. (L542)

- Operational differences between \citet{datta_large_2026}'s $t^\star$, \citet{zhang_when_2026}'s $\tau_{\text{commit}}$, and our FCT are tabulated in Appendix~\ref{sec:appendix-fct-tstar}.
  - [CLAIM OK] Cross-reference resolves to Appendix F.

---

# Appendix K — Reproducibility (L544)

## ¶ body (L547)

- All decoding traces, paired closed-vs-oracle JSONLs, slice manifests with SHA-256 checksums, decoding command lines, and the LLM-as-Judge call cache will be released alongside the analysis scripts that produce every table and figure in this paper, including the worked example trace of App.~\ref{sec:appendix-prompts}.
  - [CODE OK] `emnlp_26_lengthConfound/README.md` documents the slice-manifest SHA-256 path (`runs/v2_headline_<arch>/checkpoints/slice_manifest.json`); the `analysis/llm_judge_cache/` directory is shipped; all analysis scripts are present in `analysis/`. Reproducibility claim is operationalizable. Note that the *actual* release would also include the example trace per Appendix D (file `example_trace.json` already exists in repo root).
- Compute: single NVIDIA A800 80 GB, wall-clock ∼15 h end-to-end.
  - [CITE OK] README: "Decoding the traces (GPU pipeline, ~15 h wall on A800 80 GB)." Exact match.

---

# Overall audit summary

- **Numerical / code integrity:** All headline numbers in the abstract, §4 results, and Appendices A/B/C/E/H/I cross-verify exactly against the shipped `.out` files in `analysis/`. The seven most load-bearing numbers (9 of 16 nominal-95% survivors; 4 of 4 Bonferroni-robust; FCT Δ=+0.190 → +0.222 under judge; Gemma×2Wiki FCT raw 0.784; tokens_generated raw 0.457 / 0.518; ε_protocol ≤ 0.029; κ = 0.707) are all confirmed.
- **Citations:** All 26 cited works that are present in `__lit-review-main` / `references-recollections-v2` or whose bib abstracts contain enough information to verify the claim *are* faithfully represented. Two earlier `[CITE ⚠]` flags **have been resolved via direct paper-body reads** and are now cleared: (a) the "$U_{\text{cons}}$ component of RSUS, \citep{guo_when_2026}" attribution is verbatim against §4.1 Eq. (2) and §4.1 prose of Guo 2026 (arxiv 2604.26649); (b) the SeaKR best-$k$ interval $K \in [10, 25]$ is verbatim against Yao et al. 2025 §5.3 Hyper-parameter Search. One residual minor: the "same LongBench bundle" attribution for \citet{zhang_verbosity_2024} in §4.2 is loose — Zhang's bundle overlaps on HotpotQA but not on 2WikiMultiHopQA.
- **Scientific soundness:** Core methodological chain (paired-oracle target + length-residualization + paired-bootstrap + cross-judge robustness) is internally coherent and the limitations section honestly discloses the four-survivor knife-edge, the OOD vs held-out distinction, the EM-correctness-function caveat, and the non-thinking-mode-chat-model adaptation gap.
- **Critical issues to fix before submission:**
  1. **§3.3 Eq. 6 / Appendix D FCT off-by-one.** Paper Eq. 6 uses `∀ j > k`; the shipped code uses `∀ j ≥ k`. The Appendix D narrative ("at or after $t_{\text{commit}}$") matches the code. Fix: change Eq. 6 to `m_j = a_{\text{final}}\;\forall\, j \geq k` (preferred; matches code and reported AUROCs) or fix the code to match the paper.
  2. **"Causal retrieval-helpfulness" wording** (abstract + §3.1 + §5.1). The paired-oracle design controls for one major confound but doesn't license a fully causal interpretation. Consider weakening to "directly measures" or "isolates" everywhere "causal" appears.
- **Recommended minor cleanups (non-blocking):**
  - §4.1 "hidden-state (SeaKR, INSIDE, plus three geometry variants)": INSIDE isn't directly audited; clarify it's proxied via the seakr_k5 logdet variant's rank-equivalence.
  - §4.2 "same LongBench bundle" — Zhang's bundle and ours overlap on HotpotQA but not on 2WikiMultiHopQA.
  - §4.2 "clear the length baseline by a wider margin" — strictly true for FCT, only mixed for the other three Bonferroni survivors on raw-AUROC.
  - Appendix D ¶ Boundaries: §3.3 elides the spaCy fallback; consider unifying the wording.

- **Resolved (no action needed):**
  - §1¶1 "$U_{\text{cons}}$ component of RSUS, \citealp{guo_when_2026}" — verified verbatim against Guo 2026 §4.1 Eq. (2) and prose.
  - Appendix J SeaKR "$K \in [10, 25]$" — verified verbatim against Yao et al. 2025 §5.3.
