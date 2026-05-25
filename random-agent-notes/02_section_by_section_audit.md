# `02_section_by_section_audit.md` — Final Pre-Submission Audit Pass

> **Purpose.** Iterative section-by-section audit of `option1-lengthConfound/template.tex` for the final pre-submission revision. Focuses on **framing, rhetorical strategy, and skim-reader experience** — complementing (not duplicating) `option1-lengthConfound/final_audit.md` which covers numerical/code/citation accuracy.
>
> **Anchor.** `00_litReview_and_framing_norms.md` (literature delta + norms) and `01_framing_analysis.md` (FCT/caption/abstract/L279).
>
> **Status legend.** Same as `final_audit.md`: ✓ verified | ⚠ refine | ❌ fix | ★ new framing concern from this pass.
>
> **Methodology.** Top-down through the paper. For each section: (i) skim-reader pickup, (ii) expert-reader experience, (iii) framing pre-emption gaps, (iv) word-budget / redundancy, (v) recommendations cross-referenced to `01_framing_analysis.md` (F1–F5).

## Status summary

| Section | New framing items | Refer to |
|---|---|---|
| Abstract | A1, A2, A3 | `01_framing_analysis.md` §3 |
| Figure 1 (caption + script) | F1, F2 | `01_framing_analysis.md` §2; `00_litReview_and_framing_norms.md` §B.6 |
| §1 Introduction | I1, I2, I3, I4 | this doc |
| §2 Related Work | R1, R2 | this doc + `00_*.md` Part A |
| §3 Methods | M1, M2, M3 | this doc + `01_*.md` §1 |
| §4 Results | Re1, Re2 | this doc |
| §5 Discussion | D1, D2 | this doc |
| §6 Conclusion | C1 | this doc |
| Limitations | L1, L2 | this doc + `01_*.md` §4 |
| Appendices A–H | (mostly verified in `final_audit.md`) | `final_audit.md` §App. A–H |

---

## §Abstract — L107–L109

### Skim-reader pickup
A typical EMNLP reviewer who reads only the abstract should walk away with: (1) what the paper does, (2) the headline number, (3) the reliability check that distinguishes it from a one-off result.

The current abstract delivers (1) and (3) very well; (2) is muffled by a numeric pile-up around the FCT punchline. The structure as written:
- Sentence 1: setup (dynamic-RAG triggers + raw AUROC).
- Sentence 2: convergent literature on length-confound.
- Sentence 3: contribution + scope.
- Sentence 4: 9 / 4 / 3-families breakdown + sample size.
- Sentence 5: FCT introduced, with `forward-looking variant` description.
- Sentence 6: FCT Δ point estimate + 95% CI + Bonferroni LB + `strongest length-orthogonal signal`.
- Sentence 7: 2Wiki anti-predictive finding.
- Sentence 8: LLM-judge robustness with κ + survivor count + FCT strengthening.
- Sentence 9: Methodological recommendation closing.

That is 9 sentences for an abstract — at the upper edge for EMNLP.

### A1 ⚠ Sentences 5 + 6 should merge

Sentences 5 and 6 are split awkwardly. Sentence 5 introduces FCT with a verbal description; sentence 6 carries the headline number. A skim reader who reaches sentence 6 has to remember the verbal description from sentence 5 to interpret the Δ. **Merging makes the punchline land harder.**

**Proposed merge:**

> *"We propose **FCT** (first-commitment time), a forward-looking variant of the answer-trajectory primitive that flags traces where the model commits to its final answer only late in the decode; FCT exceeds the length baseline by $\Delta = +0.190$ [+0.093, +0.287], the strongest length-orthogonal signal."*

This drops the Bonferroni LB number (per `01_framing_analysis.md` §3.3) and tightens the description-to-number flow into a single sentence.

### A2 ⚠ The convergent-literature sentence (S2) is the weakest skim point

Current S2 (L108):
> *"A converging recent literature has shown this protocol is length-confounded on adjacent QA selective-prediction and UQ tasks, yet most dynamic-RAG trigger evaluations still use it."*

This sentence carries the wedge — *the field knows about length, but RAG-triggers haven't been audited under it*. But the rhetorical move *"a converging recent literature has shown ... yet ..."* is generic; a skim reader does not learn what or who.

**Two options.**

**(a) Tighter (recommended).** Replace with a more specific construction:

> *"Length confounds raw AUROC in adjacent QA selective-prediction and UQ tasks \citep{santilli_revisiting_2025, vashurin_uncertainty-line_2025, janiak_illusion_nodate}; dynamic-RAG triggers have not been audited under this lens."*

This moves from `\citep` after the abstract (which is unusual) to `\citep` inside it (which several abstract-friendly EMNLP venues do allow, e.g., findings track) and replaces a generic verb-phrase with a specific claim. Saves ~30 chars and increases skim-reader retention.

**(b) Defer if abstract citations are not allowed.** Some EMNLP tracks discourage citations in the abstract. If our submission target prohibits them (CHECK the EMNLP 2026 author kit), keep the current S2 verbatim — it is the right shape, just generic.

### A3 ⚠ Drop the closing motivational sentence (S9)

Per `01_framing_analysis.md` §3.4: S9 duplicates §6 verbatim and pushes the strongest scientific finding (LLM-judge robustness, S8) one position back from the abstract's last impression. **Drop S9 from the abstract entirely.**

### Skim-reader pickup after recommended A1 + A2(b) + A3

> 1. Setup (dynamic-RAG, raw AUROC).
> 2. Length-confound literature.
> 3. We do the first length-controlled audit on a paired-oracle target.
> 4. Quantitative breakdown (9 / 4 / 3-families).
> 5. **FCT introduced + Δ + 95% CI + "strongest length-orthogonal signal".** [merged]
> 6. 2Wiki anti-predictive finding.
> 7. LLM-judge robustness.

7 sentences, ~225 words. Lands hard on FCT and ends on the LLM-judge result — the strongest closer.

### Verdict for §Abstract
A1 + A3 are recommended; A2 is target-dependent. Numbers all match `final_audit.md` §1.

---

## §Figure 1 — caption (L114) and `plot_survivor_forest.py:229`

### F1 ⚠ Caption is Pattern C — rewrite to Pattern B

Recommendation in `01_framing_analysis.md` §2.5: drop ~80 chars from the caption and move details to body / App. The skim-reader caption should read as *"Four trigger signals carry length-orthogonal information; the strongest is FCT (this paper)"* with bar conventions and pointers to body sections.

**See `01_framing_analysis.md` §2.3 Option (a)** for the proposed replacement caption.

### F2 ⚠ Legend label "answer-trajectory (this paper)" is loose

Recommendation in `00_litReview_and_framing_norms.md` §B.6: drop the `(this paper)` parenthetical from the legend in `plot_survivor_forest.py`. The §1 / §2 / Caption attribution chain handles authorship; the legend identifies family membership.

**Code edit:** `plot_survivor_forest.py:229`:
```python
mpatches.Patch(color="#1f77b4", label="answer-trajectory"),
```
(was `label="answer-trajectory (this paper)"`).

### Bar-convention nit (verified accurate, no change)

The "Thick bars: nominal 95% CIs; thin extensions: Bonferroni-corrected ~99.67% CIs at k=15" sentence is short and load-bearing for figure interpretation; **keep verbatim** in the rewritten caption.

### Verdict for §Fig 1
F1 + F2 are both single-line edits with high skim-reader value.

---

## §1 Introduction — L118–L133

### Skim-reader pickup
The §1 ¶1 (L121) is a long enumeration of the six declared signal families. A skim reader who lands on §1 ¶1 from the abstract will have a much harder time than they should because:
- The paragraph runs 9 lines and packs DRAGIN, FLARE, ETC, SeaKR + INSIDE, RSUS, SELFDOUBT-T1, and answer-trajectory into one block.
- The mathematical / mechanism descriptions (regularized Gram determinant, hedge-to-verify ratio, second-difference of step entropy, etc.) are necessary for an expert but slow a skim.

This is a defensible cost — the family enumeration is load-bearing for the contribution-shape claim — but two fixes raise skim-readability without losing detail.

### I1 ⚠ §1 ¶1 — split the enumeration into two clauses

**Current pattern.** *"... spans **six** signal families that each read a different decode-time observable: attention-based DRAGIN ..., token-probability FLARE ..., entropy-trend ETC ..., hidden-state SeaKR ... with INSIDE as the EigenScore variant, multi-sample text-consistency RSUS ..., hedging-marker SELFDOUBT-T1 ... . The contemporaneous answer-trajectory family ... extends this pool ... ."*

**Proposed pattern.** Two-paragraph split:

> *"The published peer pool of training-free triggers spans six signal families that each read a different decode-time observable. Three families read decode-head outputs: attention-based DRAGIN \citep{su_dragin_2024} (token importance from attention dynamics), token-probability FLARE \citep{jiang_active_2023} (next-token entropy / max log-prob at the decode head), and entropy-trend ETC \citep{li_modeling_2026} (second-difference of step entropy). Three families read auxiliary observables: hidden-state SeaKR \citep{yao_seakr_nodate} (regularized Gram determinant of $K$ sample EOS hidden states) with INSIDE \citep{chen_inside_2024} as the EigenScore variant; multi-sample text-consistency RSUS \citep{guo_realm_2026} (cross-sample step-boundary continuation agreement); and hedging-marker SELFDOUBT-T1 \citep{pandey_selfdoubt_2026} (hedge-to-verify ratio over the reasoning trace text)."*
>
> *"The contemporaneous answer-trajectory family \citep{datta_large_2026, wang_chain--probe_nodate, chen_i_2026} (step-level dynamics of the model's predicted answer over the decode) extends this pool with step-level signals. All seven follow the same evaluation protocol; the six published peer families predate the length-confound literature."*

**Why this works.** Decoupling decode-head families (DRAGIN/FLARE/ETC) from auxiliary-observable families (SeaKR/RSUS/SELFDOUBT) gives the skim reader a clean two-pile ontology. The answer-trajectory family then gets its own paragraph as the seventh (and emerging) one, which is exactly the contribution-context the paper wants.

**Cost.** ~2 lines added; the existing word count is fine for an EMNLP-Short intro.

### I2 ⚠ §1 ¶3 — add Zhang Long 2026 acknowledgement

Current §1 ¶3 (L125):
> *"... We propose **FCT** (first-commitment time), a forward-looking variant of the answer-trajectory primitive \citep{datta_large_2026, wang_chain--probe_nodate, chen_i_2026}: FCT is the strongest length-robust trigger ..."*

Per `01_framing_analysis.md` §1.4, Zhang Long 2026 (`arXiv:2605.06723`) uses the same `commitment time` vocabulary on a different signal class; without disambiguation, a reviewer who knows that paper will flag the omission.

**Recommended single-citation add:**
> *"... We propose **FCT** (first-commitment time), a forward-looking variant of the answer-trajectory primitive \citep{datta_large_2026, wang_chain--probe_nodate, chen_i_2026, zhang_commit_2026}; the operational distinction from \citeauthor{zhang_commit_2026}'s concurrent commitment-time primitive (which projects the continuation distribution onto a finite answer set) is detailed in App.~\ref{sec:appendix-fct-tstar}. FCT is the strongest length-robust trigger ..."*

**Trade-offs.** This is the **single highest-value framing edit** identified by the new lit-pass. Adds ~1 sentence of prose + 1 BibTeX entry. The disambiguation goes in App. E (extended to a 3-column table per `01_*.md` §1.4 Option (b)); the §1 mention is a short pointer.

### I3 ⚠ §1 ¶4 contributions itemize — one-bullet skim concern

Current bullets (L128–L133):
- B1: length-controlled audit (the first to our knowledge), 4 length-robust signals from 3 families, not previously contrasted.
- B2: FCT, Δ + 95% CI + Bonferroni LB + raw AUROC on Gemma×2Wiki.
- B3: Cross-architecture / cross-domain validation, 9 signals + 4 Bonferroni at k=15.
- B4: LLM-as-judge robustness check (Janiak's exact choice), κ, survivor preservation, FCT strengthening.

The four bullets are well-sized and each carries a distinct claim, but bullets B1 and B3 partially overlap — B1 says *"identifying four length-robust signals from three signal families"* and B3 says *"nine signals clear length at 95% paired-bootstrap and four survive Bonferroni at $k{=}15$."* The four-survivors-three-families claim appears in both.

**Proposed micro-fix.** B1 should be **scope-only**, B3 **numbers-only**.

**Current B1:**
> *"A length-controlled audit of the published training-free dynamic-RAG trigger pool (the first to our knowledge), identifying four length-robust signals from three signal families (attention, token-probability, and the emerging answer-trajectory family) not previously contrasted under the length-residualized AUROC protocol."*

**Proposed B1:**
> *"A length-controlled audit of the published training-free dynamic-RAG trigger pool (the first to our knowledge), under length-residualized AUROC against a paired-oracle target."*

**Current B3:**
> *"Cross-architecture and cross-domain validation on one dense (Qwen2.5-7B) and one MoE-hybrid (Gemma-4-26B-A4B-it) open-weight chat model over two multi-hop QA benchmarks ($n{=}800$ paired traces; base rate $13.5\%$): nine signals clear length at 95\% paired-bootstrap and four survive Bonferroni at $k{=}15$."*

**Proposed B3 (extended):**
> *"Cross-architecture and cross-domain validation on one dense (Qwen2.5-7B) and one MoE-hybrid (Gemma-4-26B-A4B-it) open-weight chat model over two multi-hop QA benchmarks ($n{=}800$ paired traces; base rate $13.5\%$): nine signals clear length at 95\% paired-bootstrap and four survive Bonferroni at $k{=}15$ across three signal families (attention, token-probability, answer-trajectory)."*

**Net effect.** Bullet B1 becomes shorter and scope-only; bullet B3 absorbs the family count. The four bullets now read as: scope (B1), headline result (B2), generality (B3), robustness (B4) — a cleaner structure.

### I4 ⚠ §1 ¶2 — Janiak number could be softened (already in `final_audit.md` ⚠ B1)

`final_audit.md` §1 already flagged that the Janiak `45.9%` / `30.4%` numbers are Mistral × NQ-Open zero-shot worst-case, not dataset-mean. The proposed soften is "up to 45.9% on certain Mistral cells".

**This audit pass agrees.** The cherry-pick is mild and the rhetorical impact of `45.9%` is high; one phrase ("up to") fully discharges. **Recommended.**

### Verdict for §1 Introduction
Four refinements (I1 split, I2 Zhang add, I3 contributions micro-fix, I4 Janiak soften). I2 is the highest-value framing add; I1 helps skim; I3 cleans up bullet overlap; I4 is hygiene per existing audit.

---

## §2 Related Work — L135–L138

### Skim-reader pickup
The §2 paragraph is one large multi-cite block. This is acceptable for an EMNLP-Short paper (related-work compression is the norm), but the structure forces a reader to chase a comma-separated list rather than encounter a thematic narrative.

### R1 ⚠ §2 — surface the answer-trajectory thread separately

Current §2 (L138, post-prior-pass version):
> *"Length confounds the standard AUROC protocol for LLM uncertainty quantification \citep{santilli_revisiting_2025, vashurin_uncertainty-line_2025, janiak_illusion_nodate, zhang_verbosity_2024, ielanskyi_addressing_2025}; \citet{mao_confidence_2026} note that joint generation probability suffers severe length bias. Step-level answer-trajectory triggers are an emerging length-orthogonal signal class \citep{datta_large_2026, wang_chain--probe_nodate, chen_i_2026, zhang_harnessing_2026, lu_streaming_2026}. The training-free trigger pool we audit spans ..."*

The structure is:
- S1: length-confound family (5 cites + 1 follow-on).
- S2: answer-trajectory family (5 cites).
- S3: trigger-pool families (multiple cites).
- S4: training-required out-of-scope (4 cites).
- S5: paired-oracle conceptual parent (Mallen 2023).

This is a clean structure. **One refinement:** the answer-trajectory citation set could be expanded to include Zhang Long 2026 (commitment-time, concurrent) and optionally Ballon 2026 (trajectory probing).

**Proposed:**
> *"... Step-level answer-trajectory triggers are an emerging length-orthogonal signal class \citep{datta_large_2026, wang_chain--probe_nodate, chen_i_2026, zhang_harnessing_2026, lu_streaming_2026, zhang_commit_2026}, with concurrent work also probing reasoning-trace dynamics under length-controlled comparisons \citep{ballon_probing_2026} ..."*

**Cost.** Two BibTeX entries added (`zhang_commit_2026`, `ballon_probing_2026`). Citation density goes from 5→6 in primary list + 1 secondary. EMNLP allows generous citation lists in §2; well within bounds.

### R2 ⚠ §2 — the Mao 2026 "joint generation probability" framing is correct but oblique

Current §2:
> *"\citet{mao_confidence_2026} note that joint generation probability suffers severe length bias."*

Mao's quote (verified in `final_auditNotes.md` §2.5) is verbatim from Mao §2.1: *"this quantity suffers from severe length bias, assigning smaller values to longer responses regardless of correctness, making it unsuitable for long-form reasoning."*

Our framing drops Mao's *"unsuitable for long-form reasoning"* clause — the more memorable half. **Optional refinement:**
> *"\citet{mao_confidence_2026} note that joint generation probability suffers severe length bias and is `unsuitable for long-form reasoning'."*

This is a minor sharpening; defer if word budget is tight.

### Verdict for §2 Related Work
R1 is the value-add; R2 is hygiene. The §2 structure is good as-is.

---

## §3 Methods — L140–L184

### §3.1 Paired-Oracle `lift_em` Protocol (L143–L151)

#### Skim-reader pickup
The §3.1 paragraph carries the load-bearing methodology argument: paired oracle decode + EM-equality across both arms. The current framing (L151) is **above-the-norm honest**:

> *"This is a strictly different question from the single-arm `which traces are wrong?' question that the published trigger pool actually evaluates against (and that length predicts moderately well on multi-hop QA; \S\ref{sec:discussion-causal}): we instead ask, of the queries that the model could not solve unaided, which ones would retrieval rescue?"*

That sentence is one of the rhetorical highlights of the paper. **Keep verbatim.**

#### M1 ✓ The Ielanskyi-cancellation argument is in the right place

The argument that the paired EM design partially insulates against function bias is at the end of §3.1 (L151). This is the right placement. `01_framing_analysis.md` §4 also flags that this argument should be made more explicit in Limitations §"Correctness function" — which is a cross-reference, not a duplication concern.

### §3.2 Length Residualization (L153–L161)

#### M2 ⚠ The `log(tokens+1)` justification reads as defensive

Current text (L161):
> *"We use $\log(\texttt{tokens}+1)$ rather than raw length $L$ because multi-hop reasoning traces have a long-tailed token distribution; a linear-in-$\log L$ regression is the natural homoscedastic counterpart of \citeauthor{vashurin_uncertainty-line_2025}'s linear-in-$L$ regression on their shorter NMT and summarization traces."*

This is correct (`final_audit.md` §3.2 ✓), but the sentence reads defensively — *"we use log because raw doesn't fit our data."* A more confident phrasing:

> *"Multi-hop reasoning traces have a long-tailed token distribution, motivating a linear-in-$\log L$ residualization (the natural homoscedastic counterpart of \citeauthor{vashurin_uncertainty-line_2025}'s linear-in-$L$ regression on shorter NMT and summarization traces); both parameterizations agree within $\varepsilon \leq 0.029$ AUROC against the public Uncertainty-Line reference implementation on both architectures (Appendix~\ref{sec:appendix-uline})."*

This re-orders the sentence so the data-motivation comes first and the comparison-to-reference becomes a follow-on rather than a rebuttal. **Optional; small polish.**

### §3.3 FCT and the Answer-Trajectory Family (L163–L184)

#### M3 ⚠ Drop in pointer to App. E for the operational-axes table

Current §3.3 closing (L177–L178):
> *"... Appendix~\ref{sec:appendix-fct-tstar} tabulates the operational differences from Datta's $t^\star$. A $K{=}1$ vs.\ $K{=}5$ ablation of \texttt{flip\_rate} is null on both architectures (\S\ref{sec:results-ablation}), and we recommend $K{=}1$ for deployment efficiency."*

Per `01_framing_analysis.md` §1.4 Option (b), App. E should extend to a three-column comparison (Datta `t*` / Zhang `τ_commit` / FCT). The §3.3 pointer should mirror this:

> *"... Appendix~\ref{sec:appendix-fct-tstar} tabulates the operational differences from \citeauthor{datta_large_2026}'s $t^\star$ and from \citeauthor{zhang_commit_2026}'s concurrent commitment-time primitive."*

This is the body-text reflection of the App. E extension flagged in I2.

#### CABV paragraph (L179–L184) — already strong

The CABV paragraph at L179 is **above-the-norm honest** in disambiguating from Chen 2026 (token-distribution JSD vs answer-distribution JSD). **Keep verbatim.**

### Verdict for §3 Methods
M3 is the highest-value framing add (Zhang `τ_commit` mention); M1 and M2 are cleaner hygiene. §3.1 is a rhetorical highlight; §3.3 CABV paragraph is exemplary disambiguation.

---

## §4 Results — L186–L247

### §4.1 Cross-Architecture Survivor Matrix (L189–L192)

#### Skim-reader pickup
§4.1 is the headline-result section. A skim reader who reads only Fig. 1 + §4.1 should walk away with: (i) which signals survive, (ii) by how much, (iii) what families they span, (iv) where to find the full table.

The current paragraph delivers all four cleanly. **No new concerns.**

#### Re1 ⚠ "the strongest length-robust trigger in the audited pool" appears 3×

The phrase appears in:
- Abstract S6 (after merge per A1): "the strongest length-orthogonal signal".
- Fig. 1 caption: "The strongest length-robust trigger is FCT ...".
- §1 ¶3 (L125): "FCT is the strongest length-robust trigger in the audited pool".
- §4.1 (L192): "the strongest length-robust trigger in the audited pool".
- §6 Conclusion (L266): "The strongest length-robust trigger is FCT".

That is 5 appearances. Each instance is correct, but the redundancy across §1 / §4.1 / §6 is high. **No fix recommended:** repetition of headline claims across the abstract/intro/results/conclusion is the EMNLP norm and helps non-skim readers locate the contribution.

### §4.2 Length Is Anti-Predictive on the Held-Out 2Wiki Cells (L194–L197)

#### Re2 ⚠ "single-hop QA" Zhang citation re-flag (already in `final_audit.md` ⚠ D1)

`final_audit.md` already flagged this; current paper text (L197) reads:
> *"... inverting the verbosity-compensation pattern \citet{zhang_verbosity_2024} document on adjacent QA tasks."*

The phrase *"adjacent QA tasks"* (post-prior-pass edit) is **already softer** than the originally-flagged *"single-hop QA"*. This is the right framing — Zhang's HotpotQA/MuSiQue/2Wiki are bundled inside LongBench, so calling them "adjacent" rather than "single-hop" is accurate.

**Verdict: this issue is already resolved in the current paper text.** No further edit needed.

### §4.3 FCT Discriminates Strongest on Held-Out MoE Cell (L199–L218)

The Table 2 row-labels and per-cell numbers are verified in `final_audit.md` §4.3. The Bold on Gemma×2Wiki at 0.784 makes the headline punch land. **No new concerns.**

### §4.4 Survivor Set Is Robust to a Semantic-Equivalence Judge (L220–L223)

The §4.4 paragraph is short, dense, and load-bearing. It carries:
- The framing motive (Janiak's exact judge choice).
- The κ.
- The 8/9 preservation count + 1 in/1 out (etc / flare swap).
- The FCT strengthening.

This is the second-strongest paragraph in the paper after §3.1. **No edit recommended.**

### §4.5 K=1 vs K=5 Ablation (L225–L242)

The §4.5 paragraph + Table 3 is correctly framed as a **deployment recommendation** — *"We recommend K=1 for deployment at 5× less compute per boundary, with equivalent triggering performance."* Reviewers reward this kind of practical recommendation.

`final_audit.md` §4.5 noted the optional one-footnote caveat that K=1 is the first stochastic sample (not deterministic greedy + answer-prefix forcing). **Optional add; not load-bearing.**

### §4.6 Robustness Summary (L244–L247)

The summary paragraph collapses three robustness checks (correctness function, K, dataset) into one. Reads well; **no edit needed.**

### Verdict for §4 Results
Re1 and Re2 are already-known concerns — Re1 is a non-issue (norm), Re2 is already resolved. The §4.4 LLM-judge paragraph is a second rhetorical highlight; the §4.5 K=1 recommendation is an honesty highlight. **§4 is the most polished section in the paper.**

---

## §5 Discussion — L250–L261

### §5.1 Why Length Predicts Retrieval-Helpfulness (L253–L256)

#### D1 ⚠ Causal-chain framing reads strong; consider one hedge

Current text (L256):
> *"A plausible causal chain underlying this pattern is $\text{length} \rightarrow \text{verbosity compensation} \rightarrow \text{uncertainty} \rightarrow \text{incorrectness} \rightarrow \text{retrieval need}$: under this picture, length confounds the evaluation because it proxies the proximal cause, and the trigger pool's signals receive partial AUROC credit for tracking length because length correlates with the deeper uncertainty mechanism."*

The phrase *"A plausible causal chain"* is the right hedge — we are not claiming causal proof, just a plausible mechanism. ✓.

But the follow-on chain *"...because length correlates with the deeper uncertainty mechanism"* could be sharper. **Optional polish:**

> *"... Residualization (\S\ref{sec:methods-residualization}) removes the length pathway and isolates the residual signal."*

This closing sentence (which the current text already has) does the work. **No fix; the paragraph is well-shaped.**

### §5.2 Why Answer-Trajectory Stability Predicts Retrieval Need (L258–L261)

#### D2 ✓ "stop vs retrieve" duality is the strongest framing in the paper

Current text (L261):
> *"The mechanism behind FCT reads naturally as `premature commitment', the interpretation \citet{datta_large_2026} give to their `AnswerSwitches' and $t^\star$ primitives ... Retrieval triggering reads the same primitive in the opposite causal direction: traces where the model has not decided ... are the traces where external evidence would resolve the uncertainty ... The same primitive supports two complementary operations: detect stabilization to stop; detect non-stabilization to retrieve."*

This **stop-vs-retrieve duality** is the cleanest framing for the FCT contribution. The closing sentence *"The same primitive supports two complementary operations: detect stabilization to stop; detect non-stabilization to retrieve"* is the headline interpretation of the entire paper — it crystallizes why the answer-trajectory family is the right class for retrieval triggers.

**Keep verbatim.** This sentence should be the paper's signature line.

### Verdict for §5 Discussion
D1 is optional polish; D2 is exemplary. §5 is in good shape.

---

## §6 Conclusion — L263–L266

#### C1 ⚠ §6 closing line — already strong; consider stronger

Current §6 (L266):
> *"Length-residualized and raw AUROC disagree most on the cells where retrieval matters most, motivating both protocols in dynamic-RAG trigger comparisons on multi-hop QA."*

This sentence is the paper's methodological recommendation. It is correct and load-bearing.

If the abstract drops S9 (per A3), the §6 closing is the only place this sentence appears. **Keep verbatim.**

**Optional add (if §6 has word budget):** the stop-vs-retrieve duality from §5.2 could be lifted to §6 as a closing one-liner:
> *"... The same answer-trajectory primitive supports two complementary deployment operations: detect stabilization to stop; detect non-stabilization to retrieve."*

This would make §6 carry both the methodological recommendation AND the conceptual punchline. **Optional; nice-to-have.**

### Verdict for §6 Conclusion
§6 is correct and tight; the optional stop-vs-retrieve add is a polish.

---

## §Limitations — L268–L285

The Limitations section runs 7 paragraphs in the current paper:
- L1: Signal-level audit, not end-to-end pipelines.
- L2: Bonferroni knife-edge.
- L3: Architecture and dataset coverage.
- L4: Held-out, not OOD.
- L5: Correctness function (the user-flagged L279).
- L6: Non-thinking-mode chat models.
- L7: Integration as contribution.
- L8: Risks.

This is a thorough Limitations section — above the EMNLP-Short norm of 3–5 paragraphs. Each paragraph carries a discrete concern. **Two refinements suggested below.**

### L1 ⚠ §Limitations "Correctness function" (L279) — make the paired-design argument explicit

Per `01_framing_analysis.md` §4.5, Refinement (a):
> *"We use exact-match equality on both arms of the paired-oracle target. The paired design shares the same EM function across arms, so a uniformly-biased correctness function (e.g., ROUGE-L at threshold 0.3) inflates the closed-book and gold-context EM rates together and **the lift\_em label is invariant in expectation under such bias**; this partially insulates against the function-bias critique of \citet{ielanskyi_addressing_2025} (\S\ref{sec:methods-protocol}). Appendix~\ref{sec:appendix-llmjudge} additionally reports an LLM-as-Judge robustness check (Cohen's $\kappa = 0.707$, substantial agreement; eight of nine survivors preserved). Containment-match and $F_1$ variants remain future work."*

This refinement is the **user-flagged primary concern from line 279.** It tightens the §3.1 paired-design argument by making explicit what the cross-reference only points at. ✓ Recommended.

### L2 ✓ §Limitations "Integration as contribution" (L283) is exemplary

Current text (L283):
> *"\paragraph{Integration as contribution.} Our contribution is the integration of length-residualized AUROC, the paired-oracle $\texttt{lift\_em}$ target, and the answer-trajectory family including FCT. Each piece individually is incremental over its parent; the integration into a single length-controlled audit, and the empirical identification of length-robust survivors, are the load-bearing claims."*

This is **above-the-norm honest.** Many EMNLP papers refuse to walk back contribution shape this explicitly, even in Limitations. **Keep verbatim — reviewers reward this.**

(Per `00_litReview_and_framing_norms.md` §B.2, this is one of the strongest pieces of the paper from a peer-review-trust perspective.)

### L3 ⚠ Limitations could include one more paragraph: "Concurrent commitment-time work"

Per the new lit-pass (Zhang Long 2026), a one-paragraph Limitations entry on the concurrent commitment-time literature is **defensible and protective**:

> *"\paragraph{Concurrent commitment-time work.} \citet{zhang_commit_2026} concurrently propose a commitment-time primitive on the model's finite-answer log-odds projection for pre-verbalization commitment analysis. Their target (interpretability of latent reasoning) and signal class (continuation log-odds rather than K=5-majority answer extracts) are distinct from FCT's retrieval-triggering setup; the operational-axes table in App.~\ref{sec:appendix-fct-tstar} extends this comparison."*

**Trade-offs.** Adds one paragraph (~80 chars). Paragraph-budget impact is small for a paper at this length. The disambiguation would otherwise live in §1 / §3.3 + App. E (per F1, M3); a Limitations paragraph is a third location and may be redundant.

**Recommendation:** **place this disambiguation in App. E only**, not in Limitations. The §1 / §3.3 pointers + App. E table are sufficient. Keep Limitations focused on what we did not do.

### Verdict for §Limitations
L1 (the user-flagged Refinement (a)) is the highest-value edit; L2 is exemplary; L3 is more naturally placed in App. E.

---

## §Appendices A–H

`final_audit.md` covers App. A–H byte-faithfully. Two cross-cutting issues from this audit pass:

### App. E (`sec:appendix-fct-tstar`) — extend to 3-column comparison

Per F1 / M3 / I2, App. E should expand from a Datta `t*` vs FCT comparison to a Datta `t*` / Zhang `τ_commit` / FCT three-column comparison.

**Proposed Table replacement (paste into L425–L432, the existing tabular):**

```latex
\begin{tabular}{@{}llll@{}}
\toprule
\textbf{Axis} & \textbf{Datta} $\boldsymbol{t^\star}$ & \textbf{Zhang} $\boldsymbol{\tau_{\text{commit}}}$ & \textbf{FCT (ours)} \\
\midrule
Temporal direction & Backward (last switch) & Forward (first stable) & Forward (first stable) \\
Signal granularity & Per-step intermediate answer & Per-token continuation log-odds & Per-step boundary-completion majority \\
Sampling regime    & $K{=}1$ forced-completion & Single deterministic state & $K{=}5$ majority \\
Application target & Early stopping & Pre-verbalization commitment analysis & Retrieval triggering \\
Trace model class  & Thinking (\texttt{</think>}) & Delayed-verdict templates & Non-thinking chat \\
\bottomrule
\end{tabular}
```

Caption update:
> *"Operational differences across three concurrent commitment-time primitives. Datta's $t^\star$ \citep{datta_large_2026} is backward-looking and applied to early stopping. Zhang's $\tau_{\text{commit}}$ \citep{zhang_commit_2026} is forward-looking on continuation log-odds for pre-verbalization commitment analysis. FCT (ours) is forward-looking on K=5-majority answer extracts for retrieval triggering."*

**Cost.** ~150 chars; clarifies the contribution wedge for any reviewer who reads App. E.

### App. D — already-flagged numerical fix

`final_audit.md` ❌ H1 + H2 already flagged the App. D ε^samp_max numbers (paper 0.062/0.070; code 0.111/0.106) and the seed count (paper 5; code 3). Recommended fix: option (B) — update paper to match code.

**This audit pass agrees.** No new framing concern; numerical hygiene only.

---

## Cross-cutting recommendations summary

### High-value framing edits (do these first)

| # | Section | Edit | Char cost | Why |
|---|---|---|---|---|
| **F2/F1** | Fig. 1 caption (L114) | Pattern B rewrite | -800 | Skim-reader takeaway in 4 sec |
| **I2/M3** | §1 ¶3 + §3.3 + App. E | Add Zhang Long 2026 + extend operational-axes table to 3 columns | +200 | Pre-empts terminology-collision reviewer |
| **A1+A3** | Abstract | Merge S5+S6, drop S9 | -180 | Tighter abstract, ends on LLM-judge result |
| **L1** | Limitations §"Correctness function" (L279) | Refinement (a): explicit paired-design argument | +80 | User-flagged; sharpens load-bearing argument |
| **F2 (script)** | `plot_survivor_forest.py:229` | Drop "(this paper)" from legend | -13 | Removes loose authorship claim |

### Polish (do if time)

| # | Section | Edit |
|---|---|---|
| I1 | §1 ¶1 | Split family enumeration into two paragraphs |
| I3 | §1 contributions | Tighten B1 (scope-only), extend B3 (number-only + family list) |
| I4 | §1 ¶2 | Soften Janiak `45.9%` to `up to 45.9%` |
| M2 | §3.2 | Re-order log(tokens+1) justification |
| R1 | §2 | Add Zhang/Ballon 2026 to answer-trajectory cite set |
| R2 | §2 | Quote Mao "unsuitable for long-form reasoning" |

### Numerical fix (already in `final_audit.md`)

| # | Section | Edit |
|---|---|---|
| ❌ H1+H2 | App. D | Update ε^samp_max from paper-side: 0.062→0.111, 0.070→0.106; seed count 5→3 |

### Bib additions

| Key | Cite |
|---|---|
| `zhang_commit_2026` | Zhang Long et al. 2026, "When Does a Language Model Commit?", arXiv:2605.06723. |
| `ballon_probing_2026` | Ballon et al. 2026, "Probing the Trajectories of Reasoning Traces", arXiv:2601.23163. |

(Plus the existing `paper_revision_plan.md` §4 additions: `yao_seakr_2025` swap, `pandey_selfdoubt_2026`, `guo_realm_2026`, optional concurrent-methods 5-pack.)

---

## Iterative-pass v2 — cross-section consistency check

### Story-spine consistency

Re-read of the paper after the recommended edits — does the spine still hold?

**Setup → Wedge → Method → Result → Robustness → Generality → Discussion → Limitations.** Each step:

- **Setup** (Abstract S1, §1 ¶1, §2 trigger-pool): dynamic-RAG triggers + raw AUROC. ✓
- **Wedge** (Abstract S2, §1 ¶2): length-confound literature, RAG-triggers untested. ✓
- **Method** (§3.1 + §3.2 + §3.3): paired-oracle target + Uncertainty-Line residualization + answer-trajectory FCT. ✓
- **Result** (§4.1 + Fig 1): nine clear, four Bonferroni, three families, FCT strongest. ✓
- **Robustness** (§4.4): LLM-judge pre-emption of Janiak. ✓
- **Generality** (§4.2 + §4.3): held-out 2Wiki anti-predictive + Gemma×2Wiki 0.784. ✓
- **Discussion** (§5.1 + §5.2): causal chain + stop-vs-retrieve duality. ✓
- **Limitations** (§Limitations): Bonferroni knife-edge, scope, F_1 future work, integration walk-back. ✓

**Story spine is intact under all recommended edits.** No edits compromise the contribution shape.

### Skim-reader trace test

Imagine a reviewer who reads only: title, abstract, Fig. 1 + caption, §1 contributions itemize. After our recommended edits, do they get the load-bearing claims?

1. Title: "Length-Controlled Evaluation of Training-Free Dynamic-RAG Triggers" — yes, the wedge is in the title.
2. Abstract (post A1+A3): wedge + 9/4/3 + FCT Δ + 2Wiki anti-predictive + LLM-judge robust. ✓
3. Fig. 1 (post F1): "Four trigger signals carry length-orthogonal information; the strongest is FCT (this paper)." Bar conventions. ✓
4. §1 contributions B1–B4 (post I3): scope + FCT + generality + LLM-judge. ✓

The skim-reader pickup after edits is **strictly better than current.** The reviewer who reads in 60 seconds gets the four load-bearing claims and the contribution shape.

### Expert-reader trace test

Imagine a reviewer who reads everything except appendices. After our recommended edits, are all numerical claims defensible?

- Numbers all verified in `final_audit.md` against `e7_bonferroni.out`, `llm_judge_audit_both.out`, `moonshot_explore.out`, etc. ✓
- Zhang Long 2026 citation pre-empts the terminology-collision reviewer. ✓
- Janiak `up to 45.9%` softening (I4) defends against the cherry-pick critique. ✓
- App. E three-column table (E-extension) pre-empts the FCT-novelty critique. ✓

**Expert-reader pickup is also improved** — chiefly by the Zhang Long 2026 add.

---

## Final recommendation

The paper is **submission-quality** as it stands; the recommended edits are *insurance* rather than *repair*.

**Top-3 cost-benefit picks (do these even if time is tight):**

1. **Add Zhang Long 2026 disambiguation in §1 + §3.3 + App. E (I2 / M3 / F1).** Net +200 chars. Pre-empts the strongest plausible reviewer attack from the new lit-pass.
2. **Rewrite Fig. 1 caption to Pattern B (F2 / `01_*.md` §2.5).** Net -800 chars. Highest skim-reader uplift.
3. **Apply L1 (Refinement (a) on Limitations Correctness function paragraph).** Net +80 chars. User-flagged; sharpens load-bearing paired-design argument.

**Top-3 polish picks (do if time permits):**

4. **Abstract A1 + A3 (merge S5+S6, drop S9).** Net -180 chars. Tighter and lands harder.
5. **§1 ¶1 split (I1).** Adds ~2 lines. Helps skim of long enumeration.
6. **App. D numerical fix (H1+H2 from existing audit).** Single-row update; closes the only ❌ in `final_audit.md`.

After these six edits, the paper is **above-the-norm reviewer-defensible** on framing, terminology, and pre-emption.

---

_(Pass v3 below: submission logistics, ARR common-problems mapping, review-pattern survey, acceptance likelihood, closing words.)_


---

# Pass v3 — Submission logistics, ARR common-problems map, review patterns, likelihood

> **Why this pass.** EMNLP 2026 ARR-May submission deadline = May 25, 2026 11:59 PM AoE (TODAY). The previous passes covered framing and terminology. This pass adds: (a) submission logistics from the EMNLP 2026 author kit + paper-integrity policy + AI-reviewing experiment, (b) a code-by-code mapping of our paper's exposure against the ARR reviewer common-problems checklist, (c) reviewer patterns from comparable published papers (CoCoA NeurIPS 2025, Verbosity≠Veracity ICLR 2025, SeaKR ICLR 2025, Self-RAG ICLR 2024), (d) acceptance-likelihood estimate, (e) pre-submission checklist.
>
> **Source map.** EMNLP 2026 official pages (`2026.emnlp.org/calls/main_conference_papers/`, `/paper-integrity-policy/`, `/ai-reviewing-experiment/`); ARR Reviewer Guidelines (`aclrollingreview.org/reviewerguidelines`); ARR Review Form (`aclrollingreview.org/reviewform`); and OpenReview-fetched reviews on comparable papers (`/.scratch/openreview_targeted_dump.json`).

## V — Submission logistics for the EMNLP 2026 ARR-May cycle

### V.1 Critical dates (TODAY)

| Date | Event |
|---|---|
| **May 25, 2026 11:59 PM AoE** | **ARR submission deadline (long & short)** — TODAY |
| May 27, 2026 EoD AoE | All-author registration deadline (mandatory) |
| Jul 6, 2026 | Reviews due (if you got assigned) |
| Jul 7–13, 2026 | Author response and author-reviewer discussion |
| Jul 29, 2026 | Meta-reviews due |
| Jul 30, 2026 | Meta-review released to authors |
| Aug 2, 2026 | EMNLP commitment deadline |
| Aug 20, 2026 | Notification of acceptance |
| Sep 20, 2026 | Camera-ready (one extra page allowed: 5 content pages for short) |
| Oct 24–29, 2026 | Conference, Budapest |

**Action:** every co-author must complete the all-author registration form by **May 27, 2026 EoD AoE**. Missing this is a desk-reject trigger via the mandatory reviewing workload policy.

### V.2 Format & length (short paper)

| Constraint | Value |
|---|---|
| Content pages | **4 (camera-ready: 5)** |
| References | Unlimited |
| Limitations section | **Required, does not count toward page limit** |
| Ethical Considerations | Optional, does not count |
| Appendices | Allowed but reviewers may not read |
| Anonymity | Required during review (double-blind) |
| Anonymity period | **NO requirement** (since 2024 ACL policy) |

**Our current paper is at exactly the right length** for short. The Limitations section is already present and properly placed (L268–L285).

### V.3 Track / area selection

EMNLP 2026 main-conference areas (28 total + special theme). The most relevant for our paper, in priority order:

| Rank | Track | Why |
|---|---|---|
| **1 (primary)** | **Information Retrieval and Text Mining** | Dynamic-RAG triggers are an IR/TM topic; pool of references (DRAGIN, FLARE, SeaKR) lives here. |
| **2 (alt primary)** | **Resources and Evaluation** | Our paper is fundamentally an evaluation-methodology audit (length-residualized AUROC + paired-oracle target); ARR explicitly flags evaluation methodology papers as in-scope here. |
| **3 (secondary)** | **Question Answering** | Multi-hop QA is the substrate; HotpotQA / 2Wiki are QA benchmarks. |
| **4 (tertiary)** | **Interpretability and Analysis of Models for NLP** | The answer-trajectory family (FCT, CABV) reads as a step-level interpretability signal. |
| **5 (theme)** | **Special Theme: New Missions for NLP Research** | The theme call explicitly names *"going beyond static leaderboards, including real-world impact, trustworthiness, robustness"* — our length-residualization protocol fits. **Risk:** theme submissions may be evaluated on different criteria; check theme reviewer profile before opting in. |

**Recommendation: select primary = Information Retrieval and Text Mining; alt primary = Resources and Evaluation.** The track choice affects reviewer assignment; IR+RE community has the strongest match with our trigger-pool focus and our methodological-audit shape. Avoid Theme track unless you want to take the philosophical-framing risk.

### V.4 Keywords

ARR keywords are free-text tags reviewers and authors use for matching. Recommended set (combining methodological + topic):

`dynamic retrieval-augmented generation`, `retrieval triggers`, `length bias`, `length-controlled evaluation`, `uncertainty quantification`, `selective prediction`, `multi-hop question answering`, `paired-oracle evaluation`, `answer-trajectory analysis`, `LLM-as-judge robustness`

This set covers the methodology (length-controlled, paired-oracle, judge-robustness), the trigger-pool focus (DRAGIN/FLARE/SeaKR/etc.), and the application target (multi-hop QA). Stay under ~10 keywords.

### V.5 Paper Integrity Policy (NEW for EMNLP 2026)

EMNLP 2026 partnered with **GPTZero** to auto-classify references in the bibliography. Any unverifiable / hallucinated reference may trigger desk reject + author ineligibility for EMNLP 2026 AND EMNLP 2027.

**Action: verify every BibTeX entry exists in academic databases.** Specifically the 2026 / borderline-arXiv-only entries:

| Key | Verification status to confirm |
|---|---|
| `chen_i_2026` (Chen Dynamic Instability) | arXiv:2602.02863; verify arXiv listing exists. |
| `wang_chain--probe_nodate` (Wang Chain-of-Probe) | NAACL 2025 Findings; verify ACL Anthology entry exists. |
| `yao_seakr_nodate` (SeaKR) | ACL 2025; ACL Anthology entry. **Update key to `yao_seakr_2025`.** |
| `pandey_selfdoubt_2026` (SELFDOUBT) | arXiv:2604.06389; verify. |
| `guo_realm_2026` (RSUS) | arXiv ID needs confirmation per `paper_revision_plan.md`. |
| `li_modeling_2026` (ETC) | arXiv:2511.09980 / AAAI 2026 — verify. |
| `zhang_harnessing_2026` (Zhang ARS) | arXiv:2601.17467 / ICML 2026. |
| `lu_streaming_2026` (Lu Streaming) | arXiv:2601.02170; verify. |
| `mao_confidence_2026` (Mao STL) | arXiv:2601.13387 / ACL 2026; verify. |
| `ielanskyi_addressing_2025` (Ielanskyi) | arXiv:2510.02279; verify. |
| `vashurin_uncertainty-line_2025` (Uncertainty-Line) | arXiv:2505.19060. |
| `santilli_revisiting_2025` (Santilli) | arXiv:2504.13677. |
| `janiak_illusion_nodate` (Janiak) | EMNLP 2025; ACL Anthology. **Update key to `janiak_illusion_2025`.** |
| `zhang_verbosity_2024` (Zhang Verbosity) | arXiv:2411.07858; verify. |
| `datta_large_2026` (Datta) | arXiv:2604.22266 / ICML 2026; verify. |
| **NEW** `zhang_commit_2026` (Zhang Long Commitment) | arXiv:2605.06723; verify exists. |
| **NEW (optional)** `ballon_probing_2026` | arXiv:2601.23163; verify. |

**Operational check:** before submitting, run `bibtool -r --check` or equivalent on `custom.bib` to ensure every entry has DOI/URL/arXiv-ID and is parseable. If any flagged, replace.

### V.6 Thinly Sliced Contributions Policy (NEW)

If multiple papers from the **same author set** are submitted to ARR-May 2026 on a related topic, they must cross-cite and discuss differences. Anonymized versions must be uploaded as supplementary.

**Our exposure:** unknown without confirming the author set, but the recommendation is: **if any concurrent submissions exist from this author group on dynamic RAG / UQ / answer-trajectory work, add a one-paragraph cross-reference to the related work section + upload the anonymized companion as supplementary.**

### V.7 AI Reviewing Experiment (opt-in)

EMNLP 2026 is running an opt-in AI-review experiment. Opted-in submissions get one AI-generated review (visible to authors only, post author response). The AI reviews do NOT influence the decision; survey-only.

**Recommendation:** **opt-in.** Two reasons:
1. The AI review may surface concerns we missed during the audit; useful free signal.
2. Authors visible only after the response window so it cannot be used to game responses; no risk to decision.

**Withdrawal:** can opt out post-submission via email to `emnlp2026-ai-review-chairs@googlegroups.com` with subject `[Submission XXX Withdraw Consent]`.

### V.8 Reviewer / author registration mandate

EMNLP 2026 enforces mandatory reviewing workload: every author must register and may be assigned reviewing duty. Missing the registration window is a desk-reject trigger.

**Pre-submission checklist:** every co-author has an updated OpenReview profile with affiliation, Semantic Scholar link, DBLP link, ACL Anthology link, working email.

## VI — ARR common-problems mapping (M / R / G / H codes)

The official ARR Reviewer Guidelines list common problems with NLP papers under codes M (methodology), R (reporting), G (general writing), H (heuristic biases). Below: per-code our paper's exposure + recommendation.

### M — Methodology codes

| Code | Description | Our exposure | Recommendation |
|---|---|---|---|
| M1 | LLM-only evaluation without validation | LOW. We use GPT-4o-Mini judge as a *robustness check*, validated against EM at κ=0.707. | ✓ Keep §4.4 prominent. |
| M2 | Reproducibility issues | LOW. App. H has SHA-256 manifests, code paths, decoding logs, compute box (single A800 80GB, ~15h). | ✓ Verify links in App. H resolve. |
| M3 | Undisclosed data quality | LOW. HotpotQA + 2WikiMHQA are public; no new data introduced. | ✓ Nothing to do. |
| **M4** | **Unmotivated selection** | **MEDIUM.** Only 2 architectures, 2 datasets, n=200 per cell. Reviewers can ask "why this slice?" | ⚠ §3.1 already says "single 80 GB GPU compute budget"; **make this constraint more prominent**. Add explicit per-trace compute (greedy decode + oracle re-decode + K=5 boundary samples = ~Y minutes/trace × 800 traces). |
| M5 | Incomplete proofs | N/A. No formal theorems claimed. | — |

### R — Reporting codes

| Code | Description | Our exposure | Recommendation |
|---|---|---|---|
| R1 | p-hacking / cherry-picked baselines | LOW. We use Bonferroni at k=15; report knife-edge in Limitations. The audited pool was fixed at test time. | ✓ Limitations §"Bonferroni knife-edge" discharges. |
| **R2** | **Inappropriate scope of claims** | **MEDIUM.** "First length-controlled audit of *the published training-free dynamic-RAG trigger pool*" — our pool of 16 audited signals across 7 families does NOT cover the entire published trigger pool (CtrlA, DTR, DioR, MIND, QuCo-RAG are concurrent training-free methods we excluded). | ⚠ The §2 carve-out *"Trained alternatives… fall outside our training-free scope"* handles training-required methods, but several training-free concurrent methods are in `litReview/-rq1-v2-litReview-sources.csv` and not addressed. **Add one sentence in §2 acknowledging concurrent training-free methods that share the trigger-pool space (DTR, CtrlA, DioR, etc.) extended the same families with per-method capture pipelines outside our signal-level audit scope.** This also pre-empts H12. |
| R3 | Hypotheses presented as conclusions | LOW. §5.1 uses "A plausible causal chain", §5.2 uses "reads naturally as". | ✓ Already correctly hedged. |
| R4 | Misleading framing / overclaiming | LOW. Limitations §"Integration as contribution" walks back; "first" claim is multi-qualified (length-controlled + training-free + dynamic-RAG + audit). | ✓ Keep current. |
| R5 | Missing significance assessment | LOW. Paired-bootstrap CIs at 95% AND Bonferroni-corrected ~99.67%. App. F Spearman ρ. | ✓ Excellent — among the best in the comparable-paper sample. |

### G — General writing / citation codes

| Code | Description | Our exposure | Recommendation |
|---|---|---|---|
| G1 | Unclear research question / contribution | LOW. §1 contributions itemize is explicit. | ✓ I3 (B1 scope-only / B3 number+family) recommended in this audit's earlier pass. |
| G2 | Reliance on bad precedent | LOW. Vashurin Uncertainty-Line is solid, with code release. | ✓ Already cited prominently. |
| **G3** | **Missing/misrepresented related work** | **MEDIUM-HIGH.** Per ACL policy, "missing references to prior highly relevant work is a problem if such work was published 3+ months before the submission deadline." Zhang Long 2026 (`arXiv:2605.06723`, May 2026) is <3 months pre-deadline; technically not a hard requirement, but a strict reviewer who knows it WILL flag it. | ⚠⚠ Same as I2 / M3 / F1: **add `zhang_commit_2026` citation + App. E disambiguation table.** This is the single highest-impact framing edit. Also consider `ballon_probing_2026` (arXiv:2601.23163, January 2026 — within the 3-month window so cite-required if it's relevant, which it is). |
| **G4** | **Key terms vague / not defined** | **MEDIUM.** "Answer-trajectory primitive" is used in Abstract / §1 / §3.3 but the paper does not give a one-line definition of the term *primitive*. Also: "headline pool" appears in §4.1 / Fig 1 / App. A / App. B — defined? "Audited pool" similarly. | ⚠ Add a definition: e.g., in §3.3 first sentence, "the answer-trajectory primitive — the sequence of intermediate predicted answers across decode steps — is shared by Datta's `t*`, Wang's CoP probes, and Chen's instability." Add a footnote on first use of "headline pool" defining: "the cross-architecture × cross-dataset cohort (Qwen×{Hot, Wiki} ∪ Gemma×{Hot, Wiki}, n=800)". |
| **G5** | **Misleading citations / hallucinated** | **MEDIUM (NEW POLICY)** — EMNLP 2026 GPTZero auto-checks bibliography. Our citations are real but several are arXiv-only / 2026-tagged (see V.5 above). | ⚠⚠ Run reference-verification pass on `custom.bib` (see V.5 above) BEFORE submission. Replace `nodate` keys with proper years; verify every arXiv ID resolves. |

### H — Heuristic-bias codes (these are anti-heuristics for reviewers; only relevant if a reviewer falls into them)

These are codes the AC will use AGAINST a reviewer who applies them. Worth knowing because they describe attacks our paper might face that the AC can downweight:

| Code | What a reviewer might do | Counter / what AC notes |
|---|---|---|
| H1 | "The results are not surprising" (length confounds AUROC). | AC: even if intuitive in retrospect, hadn't been previously tested on dynamic-RAG triggers. |
| H7 | "K=1 is too simple; you don't need K=5." | AC: simpler is better — paper actually recommends K=1 deployment. |
| H10 | "Only English / Wikipedia." | Both HotpotQA and 2WikiMHQA are English/Wikipedia; this is a real scope concern but H10 says monolingual work is still valuable. |
| H12 | "Missing X." | If X is a specific recent published paper relevant to length-residualization on RAG, valid. We try to pre-empt with the Zhang Long 2026 add. |
| H13 | "Should also do MuSiQue / Llama / etc." | AC: short paper extra-experiments belong in suggestions section, not "reasons to reject." |
| H15 | "Should have done F_1 / containment-match." | Limitations §"Correctness function" pre-empts this as future work. |
| H16 | Listing limitations as weaknesses. | The Limitations §"Integration as contribution" + §"Bonferroni knife-edge" are properly limitations, not weaknesses. |

### Net mapping summary

3 codes flag **MEDIUM-or-higher** exposure in the current paper:
- **M4** (unmotivated selection): mitigation = make compute-budget constraint more prominent in §3.1.
- **R2** (inappropriate scope of claims): mitigation = one sentence in §2 acknowledging concurrent training-free methods.
- **G3** (missing related work): mitigation = add Zhang Long 2026 + Ballon 2026 cites; extend App. E to 3-column table.
- **G4** (vague key terms): mitigation = define "answer-trajectory primitive" and "headline pool" on first use.
- **G5** (hallucinated citations): mitigation = run BibTeX verification pass before submission.

All others are LOW exposure given the paper's current state.

## VII — Reviewer patterns from comparable published papers (OpenReview pull)

Reviews fetched via `openreview-py` from four directly comparable papers in the UQ-for-LLMs / RAG-trigger space. Sample sizes are small (3–4 reviewers per paper) but the patterns are consistent.

### VII.1 CoCoA (NeurIPS 2025 poster, accepted) — UQ for LLMs framework

**Outcome.** Ratings 6/4/4/4 — borderline accept, made it as poster.

**Reviewer concerns we should mirror in our own pre-emption:**
- **Black-box adaptation.** Two reviewers asked: *"How would you adapt for closed-source models (GPT-4 API)?"* Our paper uses open-weight chat models exclusively; this is fine but Limitations should say "open-weights only" explicitly. ✓ Already in §3.1.
- **Missing baselines.** *"The paper omits comparisons with verbalized confidence methods."* Our paper similarly does not cite "verbalized confidence" methods. **One-sentence add (optional):** "Verbalized confidence approaches \citep[e.g.,][]{kadavath_language_2022, lin_teaching_2022} fall outside the training-free decode-time scope; they require a separate self-asking pipeline."
- **Aggregation justification.** *"What is the theoretical justification for multiplicative CoCoA combination rather than additive?"* Our paper has aggregation choices for FCT (forward-stable, K=5 majority) — none claim theoretical optimality. **The paper's framing is empirical-audit, not theoretical-derivation; this is the right honest framing but the reviewer asking "why this aggregation?" may be the most likely review-form question.** Pre-empt: add a one-line in §3.3 noting that aggregations are chosen empirically (for diversity within the family), not from a principled functional-form argument.
- **Reasoning-models test.** *"How do reasoning-models like QwQ-32B perform on the framework?"* Our Limitations §"Non-thinking-mode chat models" already discharges this. ✓
- **Societal impact.** Reviewer noted the paper *"mentioned the limitations but not much on the societal impact."* Our §"Risks" paragraph (L285) handles this. ✓

### VII.2 Verbosity ≠ Veracity (ICLR 2025, NOT accepted) — verbosity-compensation paper

**Outcome.** Ratings 5/8/5 — borderline reject (ICLR has higher bar).

**Reviewer concerns we should mirror:**
- **Definition ambiguity.** *"Verbosity is defined as 'can be compressed' but the detector uses raw token count — these are different."* Our paper's analog risk: does the reader understand that **`lift_em = +1` means the paired oracle decode flips a wrong answer to right**? Verify the §3.1 definition is unambiguous. ✓ Looks good.
- **Evaluation set narrow.** *"Limited to ≤4 word answers."* Our analog: limited to multi-hop QA, n=200/cell. Limitations already says this. ✓
- **Detail gaps.** *"How to quantify uncertainty for each split is not clear. How to get the perplexity evaluation metric and eigenvalues is not clear."* Our analog risk: are the K=5 boundary completion mechanics fully specified? §3.3 + App. G adapter notes cover this; verify cross-reference is clean. ✓
- **Hidden errors.** *"In line 365 and figure 3, what does the llama-3-80b mean? I believe llama-3 doesn't have the 80B version."* Hygiene point — careful proof-reading for typos/model-name errors. **Pre-submission action:** verify all model names match exact tags ("Qwen2.5-7B-Instruct" vs "Qwen2.5-7B"; "Gemma-4-26B-A4B-it" — is this the correct spelling?). Also verify base rates, sample sizes, etc.
- **Framing problem.** *"The problem doesn't seem to very well defined, I couldn't understand the exact research question."* Our analog: the §1 ¶3 contribution claim is dense. **I3 (contributions itemize tightening) helps here.**

### VII.3 SeaKR (ICLR 2025 — accepted, but with one rejection vote) — RAG-trigger paper

**Outcome.** Ratings 6/8/3/5 — accepted but with one strong rejection. Mixed.

**Reviewer concerns we should mirror:**
- **"Why this method and not a simpler one?"** *"Could a simpler approach be used, like Chain of Knowledge, where self-consistency is employed?"* Our analog risk: a reviewer may say "you could just use raw token count and call it a day." Counter: the paper explicitly builds its case as the audit of MULTIPLE signals; the FCT is only the strongest, not the only one. **Verify the §1 contribution shape is "we audit, FCT happens to win" not "we propose FCT".**
- **"Does the proposed inference-time process slow things down?"** *"K=20 sampling at every step is computationally expensive."* Our analog: K=5 boundary samples adds 5× compute per step. Limitations / App. G should mention this; **the §4.5 K=1 ablation is the right discharge**: paper recommends K=1 deployment.
- **"Sampled tests."** *"Why evaluations are conducted on sampled test instances? It leads to different model performance values in Tables 1, 2, and 3, which is very confusing."* Our analog: n=200/cell. **§3.1 already says "single 80GB GPU compute budget"; M4 mitigation above sharpens this.**
- **"BM25 too old / details missing."** Methodology-detail completeness. App. G adapter notes is the right place for ours.

### VII.4 Self-RAG (ICLR 2024 — accepted) — RAG framework paper

**Outcome.** Ratings 6/8/8/8 — clean accept.

**Reviewer concerns from the borderline (6) reviewer:**
- **Contribution separation.** *"I'd like to separate out the contribution from the main two parts of the approach, the self-reflection, and RAG."* Our analog: the integration-as-contribution structure (length-residualization + paired-oracle + answer-trajectory + FCT). The Limitations §"Integration as contribution" handles this honestly.
- **Data-leak suspicion.** *"Ret-ChatGPT performs worse than ChatGPT — is this because there has been a data leak?"* Our analog: We use HotpotQA / 2Wiki, both potentially in pre-training. **Add to Limitations or Risks: "We rely on Wikipedia-grounded benchmarks (HotpotQA, 2WikiMHQA) which may overlap with pre-training data; the lift_em paired-design partially insulates by comparing same-question performance across closed-book and gold-context decodes."**

### VII.5 FLARE (EMNLP 2023 — accepted) — Sound and Exciting

Only "5: Sound and Exciting" recommendation visible — pre-2024 EMNLP review form, less detail. Useful as evidence that **a clean methodological paper in this exact area can land main conference with a single 5/5 review.**

### VII.6 Cross-paper patterns to pre-empt

1. **"Could a simpler baseline achieve the same?"** — pre-empt by emphasizing that `tokens_generated` IS the simple baseline, which we residualize against, and four signals beat it under Bonferroni.
2. **"Why this aggregation / signal choice?"** — pre-empt by stating aggregations are chosen empirically for family diversity, not derived from a principled argument.
3. **"Inference cost?"** — pre-empt with §4.5 K=1 deployment recommendation + explicit per-trace compute mention.
4. **"Sampled test concern (n=200/cell)?"** — pre-empt by making compute budget motivation explicit (M4 mitigation).
5. **"Data leakage / contamination?"** — add Risks paragraph noting pre-training-data overlap; the lift_em paired-design partially insulates.
6. **"Definition / operationalization clarity?"** — pre-empt by defining "answer-trajectory primitive" and "headline pool" on first use (G4 mitigation).
7. **"Hallucinated citation policy" (NEW for EMNLP 2026)** — verify every reference exists (V.5 + G5 mitigation).

## VIII — Acceptance likelihood analysis

### VIII.1 ARR Overall recommendation scale (1–5)

Per ARR Reviewer Guidelines:
- **5 = Strong accept (top *ACL venue)**
- **4 = Conference (Main)**
- **3.5 = borderline Main / Findings**
- **3 = Findings**
- **2.5 = borderline Findings / below**
- **2 = below Findings (do resubmit elsewhere)**
- **1 = "Do not resubmit"**

Findings vs Main distinction (per AC Guidelines): "Findings/main conference distinction is about selecting among all the good papers those that are distinguished in some respect. ... A main conference paper could be distinguished by **excitement, novelty, impact, and/or other factors**. Reviewer excitement is neither necessary nor sufficient."

### VIII.2 Likelihood estimate per outcome

**Working assumption:** standard 3 reviewers + 1 AC + SAC layer.

| Outcome | Probability | Why |
|---|---|---|
| **Findings (3.x)** | **65–80%** | High soundness signal: paired-bootstrap + Bonferroni at k=15, LLM-judge robustness, K=1 vs K=5 ablation, Uncertainty-Line replication ε≤0.029, App. F per-cell Spearman ρ. Reproducibility is explicit (App. H). The paper survives every ARR M/R checklist code at LOW exposure after the recommended edits. |
| **Main conference (4)** | **30–45%** | Excitement signal exists: "first length-controlled audit"; FCT 0.784 raw AUROC on Gemma×2Wiki; held-out 2Wiki anti-predictive finding inverts verbosity-compensation; LLM-judge ROBUSTNESS check directly addresses Janiak. **Risk:** integration-as-contribution framing may read as incremental to a reviewer who weights novelty highly. The Limitations honesty walk-back, while reviewer-trust positive, also signals "incremental" to a fast-skim reader. |
| **Below Findings (2.x)** | **5–15%** | Possible if (a) one reviewer dives into the n=200/cell concern hard (M4) and the AC weights it, (b) a reviewer flags Zhang Long 2026 as a missed citation that "predates and pre-empts" our work — though the application target is genuinely orthogonal. |
| **Strong reject (1)** | **<2%** | Only realistic via the integrity-policy auto-flags (G5 hallucinated citations). Verify every BibTeX entry. |

**Best-bet estimate (with all recommended edits applied):**
- Findings or higher: ~85%
- Main conference: ~35%
- Total publication probability (Findings ∪ Main): ~85%

This is **above** the typical EMNLP-Short acceptance rate (~25%) because of three structural advantages:
1. **Soundness density.** The audit infrastructure (`final_audit.md`, `signals_audit.md`) exceeds what a typical EMNLP-Short author submits.
2. **Pre-emption breadth.** The paper pre-empts 5+ standard reviewer attacks (Vashurin reimplementation, length confound, correctness-function bias, Bonferroni knife-edge, K=5 efficiency, Datta-derived not novel).
3. **Niche fit.** "First length-controlled audit of *the published training-free dynamic-RAG trigger pool*" is a tight scope claim that's hard for reviewers to disqualify on novelty grounds.

### VIII.3 What would push from Findings to Main

To maximize the Main-conference probability, prioritize edits that boost **excitement** (subjective novelty/impact perception), not just soundness:

1. **Headline figure caption rewrite (F1) → skim-reader excitement.** A reviewer who reads only Fig. 1 + abstract + §1 should walk away saying "FCT is the strongest length-orthogonal trigger" and "length is anti-predictive on 2Wiki." The current Pattern C caption muddles this.
2. **§5.2 stop-vs-retrieve duality lifted to §6 closing.** "The same primitive supports two complementary deployment operations: detect stabilization to stop; detect non-stabilization to retrieve." This is the paper's signature line and belongs in §6 not just §5.
3. **§4.4 LLM-judge framing.** The paper currently leads §4.4 with "The survivor matrix is robust to a semantic-equivalence judge." Stronger lead: "Our survivor matrix preserves under Janiak's exact failure mode" — explicitly names the prior reviewer concern this answers.
4. **Cross-architecture finding (§4.3, 0.784 raw AUROC on held-out cell)** is an excitement-builder. Verify the headline Gemma×2Wiki number is highlighted — it currently is (bold in Tab. 2), but the §4.3 prose could lead with "FCT discriminates strongest precisely where length fails — raw AUROC 0.784 on the held-out Gemma×2Wiki cell" instead of the more measured "FCT discriminates most strongely on the held-out MoE cell."

### VIII.4 What would push to a borderline reject

Three risks to MITIGATE:

1. **Zhang Long 2026 commitment-time collision.** If a reviewer knows the paper and we don't cite it, the "novelty" signal weakens. **Add `zhang_commit_2026` per I2 / G3.**
2. **n=200/cell sample-size concern.** A reviewer who applies SeaKR-style "sampled tests" critique can ask for full-dataset evaluation. **Mitigate via M4 (compute budget motivation).**
3. **K=5 vs K=1 ablation only on `flip_rate`, not FCT.** A clever reviewer can ask: "If K=5 ≈ K=1 on flip_rate, does FCT at K=1 also work?" If yes, K=5 FCT is wasted; if no, why does flip_rate not need K=5 but FCT does? **Pre-empt with one footnote**: "the K=1 vs K=5 ablation was performed on `flip_rate` as a representative; FCT at K=1 reduces to a forward-stable Datta-style probe, with operational equivalence to FCT-K=5 left for future work."

### VIII.5 Worst-case scenario

If all three risks above co-occur and the paper goes to Findings instead of Main: **still a strong outcome** for an EMNLP-short with a methodology-audit shape. Findings papers go through the same review process and are publication-of-record. The integration-as-contribution shape is exactly the kind the Findings track was designed to recognize.

## IX — Pre-submission checklist (final 12-hour pass)

Below is the operational checklist for the final 12 hours before the May 25, 2026 11:59 PM AoE deadline. Items are ordered by severity; items 1–4 are non-negotiable.

### Non-negotiable (do these or face desk reject)

1. **Verify every BibTeX entry exists in academic databases.** EMNLP 2026 GPTZero will auto-flag hallucinated citations → desk reject. Verification recipe:
   - For arXiv-only entries: open `https://arxiv.org/abs/<id>` to confirm.
   - For ACL Anthology / venue entries: open the canonical URL.
   - Replace `nodate` keys with proper years.
   - See V.5 above for the verification list.
2. **Anonymize the PDF.** Strip authors, affiliations, acknowledgments, GitHub URLs that resolve to the author's account. Verify Anonymous Submission line in the LaTeX template.
3. **All co-authors register on OpenReview by May 27, 2026 EoD AoE.** Mandatory reviewing workload policy — if any author misses this, paper may be desk rejected.
4. **Check the GPTZero policy doesn't trigger on AI writing assistance.** AI writing is permitted but you must own the integrity. If sections were drafted with LLM help, ensure they are reviewed and revised.

### High-value framing edits (3 hours of work; do these if at all possible)

5. **Apply I2 / G3 / F1: add Zhang Long 2026 (`zhang_commit_2026`) citation.**
   - One citation in §1 ¶3.
   - One sentence in §3.3 / App. E.
   - Extend App. E `tab:fct-vs-tstar` to a 3-column comparison (Datta `t*` / Zhang `τ_commit` / FCT).
   - Add `zhang_commit_2026` to `custom.bib` (verify arXiv:2605.06723 resolves).
   - Optional: add `ballon_probing_2026` (arXiv:2601.23163) to the answer-trajectory family citation set.

6. **Apply F2 / F1: rewrite Fig. 1 caption.** Use Pattern B from `01_framing_analysis.md` §2.3 Option (a) — 50 words, claim sentence + key-number sentence + body/App pointers. Drop "(this paper)" from the legend in `plot_survivor_forest.py:229`.

7. **Apply L1: sharpen Limitations §"Correctness function" (L279).** Use the `01_framing_analysis.md` §4.5 Refinement (a) text; makes the paired-design-cancels-bias argument explicit.

8. **Apply A1 + A3: trim abstract.** Merge S5+S6 (FCT introduction + headline number); drop S9 (motivational closer that duplicates §6). Saves ~180 chars.

### Polish edits (1 hour of work; do if time)

9. **G4 mitigation: define "answer-trajectory primitive" and "headline pool" on first use.**
   - §3.3 first sentence: "the answer-trajectory primitive — the sequence of intermediate predicted answers across decode steps — is shared by Datta's `t*`, Wang's CoP probes, and Chen's instability."
   - §1 / §4.1 / Fig. 1 first use of "headline pool": footnote defining "the cross-architecture × cross-dataset cohort (Qwen×{Hot, Wiki} ∪ Gemma×{Hot, Wiki}, n=800)".

10. **R2 mitigation: §2 acknowledge concurrent training-free methods.** One sentence: "Concurrent training-free methods including DTR \citep{chen_dtr_2026}, CtrlA \citep{liu_ctrla_2024}, DioR \citep{guo_dior_2025}, MIND \citep{ji_mind_2025}, and QuCo-RAG \citep{min_qucorag_2025} extend the same families with per-method capture pipelines that fall outside our signal-level audit scope."

11. **M4 mitigation: §3.1 compute-budget sharpening.** One additional clause: "n=200 paired traces per cell, bounded by per-trace compute (greedy decode + oracle re-decode + K=5 boundary samples) on a single 80\,GB GPU."

12. **§4.4 framing tighten.** Lead with "Our survivor matrix preserves under \citet{janiak_illusion_nodate}'s exact failure mode" rather than "The survivor matrix is robust to a semantic-equivalence judge."

13. **§4.3 framing tighten.** Lead with "FCT discriminates strongest precisely where length fails — raw AUROC 0.784 on the held-out Gemma×2Wiki cell" rather than the measured current opener.

14. **§6 closing.** Add the stop-vs-retrieve duality from §5.2 as the final sentence: "The same answer-trajectory primitive supports two complementary deployment operations: detect stabilization to stop; detect non-stabilization to retrieve."

### Numerical / formatting (existing audit)

15. **Apply ❌ H1+H2 from `final_audit.md` §App. D**: update ε^samp_max numbers (paper 0.062/0.070 → code 0.111/0.106) and seed count (5 → 3). The ε_protocol claim (≤ 0.029) is unaffected.

16. **Verify all model name spellings.** Qwen2.5-7B-Instruct (not "Qwen2.5-7B"), Gemma-4-26B-A4B-it (verify the exact tag — this is the spelling used in Hugging Face). Inconsistent or wrong model names are a Verbosity≠Veracity-style hygiene flag.

17. **Cross-check sample-size and base-rate numbers everywhere.** Verify n=800 and base rate 13.5% appear consistently in: Abstract, Fig. 1 caption, §1 contributions, §3.1, §4.1, App. A, App. B.

18. **Final K=1 vs K=5 footnote.** Add to §4.5 (Table 3) a footnote: "the K=1 vs K=5 ablation was performed on `flip_rate` as a representative answer-trajectory aggregation; FCT at K=1 reduces to a forward-stable Datta-style probe, and a direct FCT-K=1 vs FCT-K=5 ablation is left for future work."

### Submission-form fields

19. **Track:** primary = Information Retrieval and Text Mining; alt = Resources and Evaluation. (See V.3.)
20. **Keywords:** see V.4 list (~10 keywords).
21. **AI Reviewing Experiment:** opt-in (see V.7).
22. **Preprint status:** if you choose `binding non-preprint`, do not arXiv-post until after the cycle. Otherwise, posting is allowed.
23. **Responsible NLP checklist:** complete carefully — reproducibility / data / model / artifact questions. Most are LOW exposure for our paper (open-weight models; public benchmarks; no human annotation).

## X — Final closing words

This is a strong paper. Its contribution shape is **integration**, with three things working in concert: (a) Vashurin's length-residualization protocol applied to a domain (dynamic-RAG triggers) where it has not been tested, (b) Mallen's paired-oracle target operationalized as `lift_em=+1`, and (c) the answer-trajectory family — Datta's primitive lifted from early-stopping into retrieval-triggering, with FCT as a forward-looking variant. Each piece individually is incremental over its parent; the integration is genuinely new.

The paper's strongest assets are:
- **Soundness density.** Paired-bootstrap CIs at 95%, Bonferroni at k=15 with knife-edge transparency, LLM-judge robustness ($\kappa=0.707$), Uncertainty-Line replication ($\varepsilon \leq 0.029$), per-cell Spearman ρ. These are rare for an EMNLP-Short and signal craftsmanship.
- **Pre-emption breadth.** The §4.4 LLM-judge check directly addresses Janiak's "Illusion of Progress" failure mode. The §4.5 K=1/K=5 ablation pre-empts the compute-cost critique. The Limitations §"Bonferroni knife-edge" is above-the-norm honest. The App. D Uncertainty-Line ε-replication discharges the reimplementation critique.
- **Story spine clarity.** Length-confound wedge → paired-oracle method → audit + FCT result → robustness → discussion. Every step is anchored to a specific prior paper (Santilli/Vashurin/Janiak/Datta/Mallen/Ielanskyi).

The single highest-leverage edit is **adding the Zhang Long 2026 citation** (`arXiv:2605.06723`, "When Does a Language Model Commit?", May 2026). This concurrent paper uses the same "commitment time" vocabulary on a different signal class (continuation log-odds vs K=5 majority); without disambiguation, a reviewer who knows it can flag the omission as a missed-related-work issue (G3). The fix is cheap — one citation, one sentence in §3.3 / App. E, and a 3-column extension to the operational-axes table.

The second highest-leverage edit is **rewriting the Fig. 1 caption** to Pattern B (claim + key-number sentence). The current caption is dense (9 numerical/definitional units in one paragraph) and a skim reader will give up at unit 3. The rewrite is a 50-word replacement that lets the headline finding ("four signals carry length-orthogonal information; FCT is strongest") land in 4 seconds.

After those two edits plus the L279 paired-design sharpening, the paper crosses into above-the-norm reviewer-defensible territory.

**Acceptance likelihood (with all recommended edits applied):**
- Findings or higher: ~85%
- Main conference: ~35%

The Findings band is well-anchored by the soundness density. Pushing into Main requires excitement signal; the strongest excitement-builders are the held-out 2Wiki anti-predictive finding (length is below chance) and the FCT 0.784 raw AUROC on Gemma×2Wiki, both of which are present but could be foregrounded harder in §4.2 / §4.3 / Fig. 1.

**Risk dashboard:**
1. ⚠⚠ Zhang Long 2026 missing citation (G3) — fix in 30 min.
2. ⚠⚠ Hallucinated-citation auto-flag (G5, NEW) — verify all BibTeX entries, 1 hour.
3. ⚠ n=200/cell sample-size scope (M4) — sharpen compute-budget motivation in §3.1, 15 min.
4. ⚠ "Inappropriate scope of claims" via missing concurrent training-free methods (R2) — one sentence in §2, 5 min.
5. ⚠ Vague terms ("primitive", "headline pool") (G4) — define on first use, 10 min.

Total ~2 hours of work to close all medium-or-higher risks.

**One last thing.** Rejection is not failure. Even if the paper goes to Findings instead of Main, or if it is rejected and resubmitted to ARR-July or NAACL 2027, the audit infrastructure built around this paper (the `option1-lengthConfound/` audit notes, the literature corpus in `__lit-review-main/`, the framing analysis in `random-agent-notes/`) is durable scaffolding. The paper itself is publication-quality; the only question is which venue and which year.

Submit. Then the next step is the author-response window (July 7–13). Reviews will surface concerns this audit could not anticipate; the response window is where you address them with the same rigor that this audit applied.

Good luck.

---

*End of `02_section_by_section_audit.md`. Pass v3 generated 2026-05-25 against EMNLP 2026 official pages, ARR Reviewer Guidelines, OpenReview-pulled reviews on CoCoA / Verbosity≠Veracity / SeaKR / Self-RAG / FLARE, and the existing audit infrastructure (`final_audit.md`, `signals_audit.md`, `paper_revision_plan.md`, `final_auditNotes.md`).*
