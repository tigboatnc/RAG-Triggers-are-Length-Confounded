# `00_litReview_and_framing_norms.md`

> **Purpose.** Two-part note for the final framing/audit pass on `option1-lengthConfound/template.tex`.
> Part A: literature review delta (papers surfaced by today's web pass that the existing `__lit-review-main/` and `litReview/` corpus does not yet cover).
> Part B: framing norms in EMNLP/ACL/ICML/NeurIPS papers in the answer-trajectory and length-confound spaces — what attribution liberties are actually taken, what headline-caption forms work, what gets a paper rejected for over-claiming.
>
> **Anchor.** Existing audit infrastructure already covers numerical accuracy comprehensively (`option1-lengthConfound/final_audit.md`, `signals_audit.md`, `paper_revision_plan.md`, `final_auditNotes.md`). This note is **complementary**: it focuses on framing/attribution/caption/abstract — the things a reviewer reads in the first 60 seconds.

---

## Part A — Literature Review Delta (papers NOT yet in the existing corpus)

### A.1 ⚠ COLLISION RISK — Zhang Long et al. 2026, *When Does a Language Model Commit? A Finite-Answer Theory of Pre-Verbalization Commitment*

**Citation.** Long Zhang, Wei-neng Chen, Feng-feng Wei, Zi-bo Qin (South China University of Technology). arXiv:2605.06723v1 (May 2026). Keywords: "Latent Reasoning, Chain-of-Thought, Mechanistic Interpretability, Pre-verbalization Commitment".

**What they do.** Define the commitment time of an autoregressive language model as the first pre-onset position at which the finite-answer projection of the model's continuation distribution stabilizes on the eventual answer. Formally:

```
τ_commit^(γ) = min { t < τ_onset : a_t = a*, |δ_t| ≥ γ, a_s = a* ∀ s ∈ [t, τ_onset) }
```

where `δ_t = S_θ(yes | ξ_t) − S_θ(no | ξ_t)` is the log-odds of the binary answer projection, `a*` is the eventual answer, and `τ_onset` is the parser-defined answer-onset position. They report a 17–31 token mean lead in main delayed-verdict templates on Qwen3-4B-Instruct-2507.

**Why this is a collision risk for our paper.**

1. **Vocabulary collision.** They literally call their primitive **"commitment time"** (`τ_commit`). Our **FCT (first-commitment time)** uses the same root concept and the same computational template:
   - Both are forward-looking (first-stable, not last-flip; explicit contrast to Datta's `t*`).
   - Both define a stabilization quantity over an answer trajectory.
   - Both index the trajectory by step / token position and look for the earliest stable position.
2. **Mechanical relationship.** Their `τ_commit` operates on a *single* model state and a continuation-projection log-odds, with no boundary completions. Our FCT operates on a sequence of K=5-majority answer extracts at sentence boundaries. Different inputs, same temporal-structure operator (`min { i : property holds for all j ≥ i }`). A reviewer who reads both papers will see the same measurement target.
3. **Application orthogonality (the wedge).** They study pre-verbalization commitment for interpretability of latent reasoning; we study the same primitive as a retrieval trigger. **This is the cleanest case of "same primitive, different application target" — exactly the framing already used for Datta's `t*`** (early stopping vs retrieval triggering).
4. **Submission timing.** arXiv:2605.06723 means May 2026 — within the EMNLP 2026 submission window. Some reviewers will check arXiv for prior art; this paper will surface.

**Recommended action.**

- Add this paper to §2 Related Work and to App. E (`sec:appendix-fct-tstar`) — the operational-axes table currently contrasts FCT only with Datta's `t*`. Adding a column for Zhang 2026's `τ_commit` strengthens the disambiguation.
- One-sentence add in §3.3 or App. E: *"Concurrent with our work, [Zhang et al., 2026, arXiv:2605.06723] independently propose a `commitment time` primitive on the model's finite-answer log-odds projection for pre-verbalization commitment analysis; their target is interpretability of latent reasoning rather than retrieval triggering, and the underlying signal is the continuation-projection log-odds rather than K=5-majority answer extracts."*
- BibTeX key suggestion: `zhang_commit_2026`.

**Severity.** Medium-high. Not a story-breaking collision — the application target is genuinely orthogonal — but a reviewer who knows this paper and sees no citation will assume we missed it. **Cite + disambiguate is cheap insurance.**

---

### A.2 New citation: Ballon et al. 2026, *Probing the Trajectories of Reasoning Traces in Large Language Models*

**Citation.** Marthe Ballon, Brecht Verbeken, Vincent Ginis, Andres Algaba (Vrije Universiteit Brussel + Harvard). arXiv:2601.23163v1 (Jan 2026).

**What they do.** Truncate-then-inject reasoning traces at fixed token-percentiles (deciles 0–100), measure the induced answer distribution at each decile via next-token probabilities. Three control conditions to isolate instance-specific signal: random-token, swap (different question), shuffle (token permutation). Track accuracy growth, decision commitment, information gain, rescue rate. Apply to Qwen3-4B/8B/14B and gpt-oss-20b/120b on GPQA Diamond and MMLU-Pro.

**Headline.** *"Most of the accuracy gain is attributable to instance-specific semantic signal in the trace, with only a modest residual benefit surviving token permutation."* — directly parallel in argument structure to our length-residualization claim.

**Why this matters for us.**

1. **Reasoning-trace length controls are now an active research front in 2026.** Their three controls (random/swap/shuffle) are the truncation-based analog of our regression-residualization on `log(tokens+1)`. We should acknowledge this convergent design choice in §3.2 or §5.
2. **They explicitly frame "decision commitment" as a property worth tracking.** Their `flip_rate` analog (probability of switching argmax answer between consecutive deciles, their Fig 2g/h) is structurally identical to our `flip_rate` AID variant.
3. **Different task family** (multi-choice GPQA/MMLU-Pro, not multi-hop QA) and **trained reasoning models** (Qwen3-thinking, gpt-oss) — does not directly compete with our claim, but reinforces the family-of-methods context.
4. **Direct concordance:** their "longer traces appear largely difficulty-reactive rather than inherently beneficial" is a near-verbatim restatement of our verbosity-compensation discussion in §5.1.

**Recommended action.**

- Optional: cite in §2 Related Work as part of the answer-trajectory family ("emerging length-orthogonal signal class \citep{...}"). Adds breadth without adding much length.
- Bibliography key: `ballon_probing_2026`.

**Severity.** Low. Citation polish, not a missing-prior-art problem.

---

### A.3 Confirmed already-cited or out-of-scope (safety check)

The following papers from web search are already in the literature corpus or out of scope, no action needed:

- *Therefore I am. I Think.* (arXiv:2604.01202) — early decision encoding from activations + steering. Supervised representation work; **out of our parameter-free RQ1 scope**, same family as Zhang ARS 2026.
- *The Point of No Return* (arXiv:2605.17113) — counterfactual "commitment juncture" via prefix-resampling. Different mechanic (sample many continuations, measure deception rate); supervised setting. **Out of scope but worth a one-line mention if word-budget allows: extends the answer-trajectory family further.**
- *When to Think, When to Speak* (arXiv:2605.03314) — anytime disclosure policy with `k*` first public emission time. Different problem (disclosure policy learning); only loosely related.
- *Token Overflow Compressed Representations* (arXiv:2602.12235, EACL-SRW 2026) — capacity-overflow detection in xRAG soft-compression architectures. Different RAG sub-problem (compression overflow vs retrieval triggering).
- DTR (arXiv:2601.03908), CtrlA (arXiv:2405.18727), QuCo-RAG (arXiv:2512.19134) — already in `litReview/-rq1-v2-litReview-sources.csv`; the existing `paper_revision_plan.md` §3.1 D5 already discusses whether to add a one-sentence "concurrent training-free methods" line.

---

## Part B — Framing Norms in EMNLP/ACL/ICML/NeurIPS Papers

This part answers the user's question: *"how much liberties do authors generally take when they write papers in this domain?"*

### B.1 The "we propose X, a Y of Z" pattern is universal — and the parent must be cited

In ML/NLP papers from 2024–2026, the pattern *"We propose X, a forward-looking variant of Z [Citation]"* is the standard, accepted form for variant primitives. Three sample patterns from the same broad area:

- **Datta 2026 (arXiv:2604.22266)** — *"We define `AnswerSwitches(T) = Σ 𝟙[A_i ≠ A_{i+1}]` and the **final answer-switch index** `t* = max{i : A_i ≠ A_{i+1}}` on top of the answer trajectory T."* Cites the parent forced-answer-completion machinery in their related work but presents `t*` and AnswerSwitches as their own definitions.
- **Vashurin 2025 (Uncertainty-Line)** — *"We propose UNCERTAINTY-LINE, a regression-residualization protocol applied to ..."* Even though the regression-residual idea is not novel in itself, the application + the published code together constitute the contribution.
- **Mao 2026 (Confidence-over-Time STL)** — *"We propose ... discriminative STL formula mining ..."* — even though STL is established in formal methods, the ML application is theirs.

**The norm is: the parent primitive must be cited prominently in §1 (introduction), and the contribution is the variant + application.** Authors do not generally water down "we propose" language as long as the parent attribution is visible to a reviewer who reads the related-work section.

**Implications for our paper.**

- The current §1 phrasing *"We propose **FCT** (first-commitment time), a forward-looking variant of the answer-trajectory primitive \citep{datta_large_2026, wang_chain--probe_nodate, chen_i_2026}"* is **inside the norm**. The "we propose" is fine; the parent is cited; the "forward-looking variant" is the genuine differentiator.
- The same pattern works for CABV: *"We additionally compute **CABV** ... conceptually closest to the inference-time instability signal of \citet{chen_i_2026}"* — also inside the norm.
- The current Limitations §"Integration as contribution" paragraph is **above the norm in honesty** — most papers do not voluntarily walk back contribution shape that explicitly. **Keep it; it builds reviewer trust.**

### B.2 What gets papers in trouble (over-claiming patterns to avoid)

Three over-claiming patterns we should NOT do:

1. **"We are the first to ..."** without checking the contemporaneous arXiv literature. Risky because every reviewer Google Scholars the claim. Our current "first length-controlled audit of the published training-free dynamic-RAG trigger pool" is **safe** because it qualifies on three axes: length-controlled + dynamic-RAG + training-free + audit. Each qualifier rules out a class of prior work.
2. **Inventing a new acronym for an existing mechanic.** Zhang ARS 2026 uses "answer stability" for latent-perturbation robustness; the existing paper-revision plan already flags the need to disambiguate from our AID/answer-trajectory naming. **Keep FCT as our own acronym, but add Zhang Long 2026 to the disambiguation list (App. E).**
3. **Asserting downstream impact without measuring it.** *"FCT enables better dynamic-RAG systems"* would require running an end-to-end retrieval pipeline. Our current Limitations paragraph "Signal-level audit, not end-to-end pipelines" handles this correctly — **do not change.**

### B.3 What is "this paper" vs "(this paper)" in figure caption authorship attribution

In the headline figure caption (current line 114), the answer-trajectory family is colored blue with the legend label "answer-trajectory (this paper)". This phrasing is borderline:

- **Defensible reading:** "this paper" is the caption author's shorthand for "this paper introduces FCT and CABV within the answer-trajectory family"; the family itself is acknowledged in §1 as "emerging" with citations to Datta, Wang, Chen.
- **Risk reading:** A skim reader sees "answer-trajectory (this paper)" and assumes we coined the family. That is the same risk Datta-attribution issue the existing audit already flags.

**Recommendation:** Two options for the legend label:

- **(a) Tighter:** "answer-trajectory (FCT, CABV)" — names only the within-family contributions, doesn't claim the family.
- **(b) Looser but clearer:** "answer-trajectory (Datta-derived; FCT and CABV new)" — explicit about parentage.

The norm in EMNLP papers is usually a short tag in the legend with the longer attribution chain in the caption text. Option (a) is cleaner.

### B.4 Headline-figure caption norms in EMNLP/ACL papers

Looking at recent EMNLP/ACL findings papers (2024–2026), three patterns dominate for headline-figure captions:

**Pattern A — One-line takeaway + footnoted method details.** Example caption shape:
> **Bold takeaway.** Methodological details [n=, B=, percentile, etc.] in compact parenthetical or in a footnote. Two-line max in column.

**Pattern B — Two-sentence: claim + key-numbers.** Example shape:
> Claim sentence (e.g., "Length-orthogonal signals span three families."). Number sentence (e.g., "FCT exceeds the length baseline by Δ = +0.190 [+0.093, +0.287]; Bonferroni-significant.")

**Pattern C — Long, dense, numbers-heavy** (what our current caption is). This works for expert readers but kills skim. EMNLP word limits push papers toward A/B in the headline figure.

**Our current caption (L114) is Pattern C and dense.** It packs nine pieces of information:
1. The protocol name (paired-bootstrap forest plot).
2. The architecture/dataset cells.
3. The sample size (n=800) and bootstrap budget (B=10,000).
4. The target definition (`lift_em=+1`) AND the base rate (13.5%) AND the verbal definition (closed-book wrong → gold-context right).
5. The bar conventions (thick = 95%, thin = Bonferroni-corrected ~99.67% at k=15).
6. The high-level finding (nine clear length, four survive Bonferroni, three families).
7. The strongest signal name (FCT).
8. The point estimate and CI for FCT.
9. A definition of FCT (forward-looking variant of answer-trajectory primitive).

This is too much for a skim reader. **Recommendation: split into Pattern B (headline-takeaway sentence + key-number sentence) and push the protocol details into the body or App.**

### B.5 Abstract length norms

EMNLP-Short papers typically run 200–250 words for the abstract. Our current abstract (L107–L109) runs ~280 words — at the upper edge.

**What is dense and removable from the current abstract:**

- The sentence *"On two open-weight chat models over two multi-hop QA datasets ($n{=}800$ paired traces), nine signals beat the length baseline at 95\% paired-bootstrap, and four survive Bonferroni at $k{=}15$ across three signal families (attention, token-probability, answer-trajectory)."* — the phrase "across three signal families (attention, token-probability, answer-trajectory)" is repeated in the contributions itemize and in §4.1 / §6 / Fig 1 caption / Table 5. Could be dropped from the abstract since the four-survive sentence carries the meaning.
- The clause *"FCT exceeds the length baseline by $\Delta = +0.190$ [+0.093, +0.287] (Bonferroni lower bound $+0.047$), the strongest length-orthogonal signal."* — the **Bonferroni lower bound `+0.047`** number is a nicety for the careful reader. Abstract is rarely the right place for a nuance like the Bonferroni LB; this can move to §4.1 or App. A. Cut saves ~12 chars.
- *"On the held-out 2Wiki cells, raw answer length is *anti*-predictive of retrieval need (AUROC ≤ 0.52), inverting the verbosity-compensation pattern documented on adjacent QA tasks."* — load-bearing finding; keep verbatim.
- *"The survivor set is **robust to a semantic-equivalence judge**: under a GPT-4o-Mini correctness function agreeing with EM at $\kappa = 0.707$, eight of nine survivors are preserved and FCT *strengthens* to $\Delta = +0.222$."* — load-bearing; keep but consider trimming "agreeing with EM at $\kappa = 0.707$" to "(GPT-4o-Mini, $\kappa = 0.707$)" parenthetically. Saves ~30 chars.
- The closing sentence *"Length-residualized and raw AUROC disagree most on the cells where retrieval matters most, motivating both protocols in dynamic-RAG trigger comparisons on multi-hop QA."* — motivational, removable; the §6 conclusion already restates it. **Cutting this saves ~150 chars and shifts the abstract from "we did the audit + here's how to read it" to "we did the audit, here are the results."** The motivational closing is more appropriate in §6.

**Net effect of three trims:** abstract goes from ~280 → ~225 words, well inside EMNLP-Short comfort zone, with all load-bearing claims preserved.

### B.6 The "(this paper)" naming for the answer-trajectory family in the figure code

The figure-rendering code (`emnlp_26_lengthConfound/analysis/plot_survivor_forest.py:229`) hard-codes the legend label `"answer-trajectory (this paper)"`. Two issues:

- The signal `flip_rate K=1` (line 54) is a **byte-faithful Datta primitive** (single-sample greedy proxy at boundary) and is colored as "answer-trajectory (this paper)" — this is the most over-claiming label in the figure. Either re-color `flip_rate K=1` (option: split blue into two shades — Datta-faithful vs ours), or rename the legend to "answer-trajectory" (drop the parenthetical).
- The signals `flip_rate K=5`, `late_flips`, `min_majority`, `CABV`, `FCT` are our K=5/JSD/forward-looking variants. These ARE differentiated from Datta but each in a distinct way; calling all of them "(this paper)" is loose.

**Recommendation:** drop `(this paper)` from the legend label entirely. The legend identifies family membership, not authorship; the §1 / §2 attribution chain handles authorship. **Trivial code edit; one-line in `plot_survivor_forest.py`.**

---

## Part C — Concrete Asks for the Audit Document

Carrying forward as audit recommendations (to be cross-referenced from `02_section_by_section_audit.md`):

| # | Section | Ask |
|---|---|---|
| C1 | §2 Related Work + App. E | Add Zhang Long 2026 (`zhang_commit_2026`) — concurrent commitment-time work — disambiguate by application target (interpretability vs retrieval triggering) and signal class (continuation log-odds vs K=5 majority). |
| C2 | §2 Related Work | Optionally add Ballon 2026 (`ballon_probing_2026`) to the "emerging answer-trajectory family" citation set as further breadth. |
| C3 | Abstract | Trim 3 phrases: family-list parenthetical, Bonferroni LB number, motivational closing. Saves ~150 chars; lands well within EMNLP-Short comfort zone. |
| C4 | Figure 1 caption (L114) | Rewrite from Pattern C (dense numbers) to Pattern B (claim + key-number sentence). Move sample size / bootstrap / base rate / bar conventions to App. or footnote. |
| C5 | Figure 1 legend (`plot_survivor_forest.py:229`) | Drop "(this paper)" parenthetical from `answer-trajectory` legend label. The §1 / §2 / Caption attribution chain handles authorship. |
| C6 | App. E (`sec:appendix-fct-tstar`) | Extend the operational-axes table from "Datta `t*` vs FCT" to "Datta `t*` vs Zhang `τ_commit` vs FCT" — three columns. The temporal-direction / sampling-regime / application-target / trace-model-class axes already in the table generalize. |
| C7 | §1 ¶3 / Limitations | The "Integration as contribution" paragraph in Limitations is above-the-norm honest; keep verbatim. Reviewers reward this. |
| C8 | Line 279 (Limitations: Correctness Function) | Audit separately in `01_framing_analysis.md`. Current phrasing is honest but the "Containment-match and F_1 variants remain future work" line is a slight understatement of what `lift_em` already partially controls for. |

---

*End of `00_litReview_and_framing_norms.md`. Generated 2026-05-25 against `template.tex` + `__lit-review-main/_lit-review.md` + `litReview/length-confound-foundations.md` + targeted web search (Zhang 2026 `commit`, Ballon 2026 trajectory-probing).*
