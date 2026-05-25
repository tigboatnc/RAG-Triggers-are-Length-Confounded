# `01_framing_analysis.md` — Framing Pass

> **Purpose.** Targeted framing analysis for the four user-asked questions:
> 1. FCT attribution: how much liberty are we taking by introducing FCT, and is it inside the field's norm?
> 2. Headline figure caption: rewrite for skim-reader / expert-reader balance.
> 3. Abstract compression: where to trim without losing load-bearing content.
> 4. Line 279 (Correctness function in Limitations): is it correctly phrased; can it be framed as our contribution?
>
> Anchored against `00_litReview_and_framing_norms.md` (norms) and `final_audit.md` (numerical/code accuracy already verified).

---

## §1. FCT attribution audit

### §1.1 What we currently say (verbatim, three locations)

**Abstract (L108):**
> *"We propose **FCT** (first-commitment time), a forward-looking variant of the answer-trajectory primitive: high on traces where the model commits to its final answer only late in the decode."*

**Introduction (L125):**
> *"We propose **FCT** (first-commitment time), a forward-looking variant of the answer-trajectory primitive \citep{datta_large_2026, wang_chain--probe_nodate, chen_i_2026}: FCT is the strongest length-robust trigger in the audited pool ..."*

**Methods (§3.3, L171–L177):**
> *"`FCT`, a forward-looking first-commitment time `[Eq. 3]`. High FCT denotes extended deliberation: the trace has not converged on its final answer until late in the decode. These aggregations adapt the `AnswerSwitches` primitive of \citet{datta_large_2026} to non-thinking-mode chat models; Appendix~\ref{sec:appendix-fct-tstar} tabulates the operational differences from Datta's $t^\star$."*

### §1.2 Verdict — the FCT framing is inside the field norm but **needs one more disambiguation**

**What we are doing right.** Three things:

1. **Parent is cited prominently in §1 / §2 / §3.3.** Datta 2026 is the explicit parent for the `AnswerSwitches` / `t*` primitives, and Wang 2024 / Chen 2026 are cited as the parallel answer-trajectory work. App. E (`sec:appendix-fct-tstar`) carries the operational-axes table — exactly the right place for the precise relationship.
2. **The "forward-looking variant" language is the contribution wedge that survives scrutiny.** Datta's `t* = max{...}` is backward (last switch); our `t_commit = min{...}` is forward (first stable). This is a real mathematical difference — they are different functions of the same trajectory, even when computed on the same sequence. The contribution is not "we invented commitment-time analysis" but "we read commitment-time forward, for triggering".
3. **The Limitations §"Integration as contribution" paragraph (L283) walks back any over-claim explicitly:** *"Each piece individually is incremental over its parent; the integration into a single length-controlled audit, and the empirical identification of length-robust survivors, are the load-bearing claims."* This is **above-the-norm** honest. Reviewers reward this.

**What is missing — the Zhang Long 2026 disambiguation.**

The web pass surfaced **Zhang Long et al. 2026, *When Does a Language Model Commit?* (arXiv:2605.06723, May 2026)** — a contemporaneous paper that **independently uses the term "commitment time"** for a structurally similar primitive. They define `τ_commit^(γ) = min{ t < τ_onset : a_t = a*, |δ_t| ≥ γ, a_s = a* ∀ s ∈ [t, τ_onset) }`. This is forward-looking, first-stable, on a finite-answer log-odds projection of the model's continuation distribution — same temporal-structure operator as our FCT, different inputs (continuation log-odds vs K=5 majority).

The application target is genuinely orthogonal (their target: pre-verbalization commitment for interpretability; ours: retrieval triggering), but the vocabulary collision is real. **Without disambiguation, a reviewer who has read both papers will assume we missed Zhang.**

### §1.3 Is FCT genuinely "ours" — what is the load-bearing differentiator?

Three differentiators that cumulatively make FCT a defensible "we propose":

| Axis | Datta `t*` | Zhang `τ_commit` | **FCT (ours)** |
|---|---|---|---|
| Temporal direction | Backward (last switch) | Forward (first stable) | **Forward (first stable)** |
| Signal granularity | Per-step intermediate answer | Per-token continuation log-odds | Per-step boundary-completion majority |
| Sampling regime | K=1 forced-completion | Single deterministic state | K=5 boundary samples |
| Target task | Early stopping | Pre-verbalization commitment / interpretability | **Retrieval triggering** |
| Trace model class | Thinking-mode (`</think>`) | Delayed-verdict templates (parser-controlled) | **Non-thinking-mode chat** |

**FCT is uniquely defined by the (forward + K=5 + retrieval-triggering + non-thinking) combination.** Each axis individually appears in some prior paper; the combination is ours. The honest framing in §1 / §2 needs to surface **two** parents (Datta backward; Zhang forward, concurrent), not one.

### §1.4 Recommended action

**Option (a) — Minimum viable.** One sentence in §3.3 or App. E acknowledging Zhang Long 2026:

> *"Concurrent with our work, \citet{zhang_commit_2026} propose a `commitment time` primitive on the model's finite-answer log-odds projection for pre-verbalization commitment analysis. Their target (interpretability of latent reasoning) and signal class (continuation log-odds) are distinct from FCT's K=5-majority retrieval-triggering setup; the operational-axes table (Tab.~\ref{tab:fct-vs-tstar}) extends to a three-way comparison."*

**Option (b) — More thorough.** Extend the operational-axes table in App. E (`tab:fct-vs-tstar`) from a 4-row × 2-column comparison to a 4-row × 3-column comparison (Datta / Zhang / FCT). Then drop a sentence in §3.3 pointing to the extended table.

**Recommendation: Option (b).** Cleanest; reviewers love a single-table comparison.

### §1.5 What NOT to do

- **Do not** rename FCT. The acronym is now load-bearing in the abstract / Fig 1 / contributions / §4.1 / §6 / App. A. Renaming would force a six-location edit and lose the headline-figure punch.
- **Do not** soften "we propose" to "we adapt" or "we extend". The forward-looking + K=5 + retrieval-triggering combination IS a genuine new variant. The norm allows "we propose" for a sufficiently differentiated variant; we are inside the norm. Softening would weaken the contribution shape unnecessarily.
- **Do not** drop the Datta citation chain to make room for Zhang. Both parents need to be cited.

---

## §2. Headline figure caption rewrite

### §2.1 Current caption (L114, ~155 words / ~870 chars)

```
Length-orthogonal trigger signals on the headline pool.
Paired-bootstrap forest plot of Δ = AUROC(s_resid) − AUROC(tokens_generated)
on Qwen2.5-7B + Gemma-4-26B-A4B-it × {HotpotQA, 2WikiMHQA}
(n=800 paired traces, B=10,000; lift_em=+1 flags traces with closed-book wrong → gold-context right, base rate 13.5%).
Thick bars: nominal 95% CIs; thin extensions: Bonferroni-corrected ~99.67% CIs at k=15.
Nine signals clear length; four survive Bonferroni and span three signal families.
The strongest length-robust trigger is FCT (first-commitment time; Δ = +0.190 [+0.093, +0.287]),
a forward-looking variant of the answer-trajectory primitive.
```

### §2.2 What is wrong (skim-reader / expert-reader trade-off)

Per `00_litReview_and_framing_norms.md` §B.4, this is Pattern C (long, dense, numbers-heavy). The skim reader who lands on this caption sees:
- A definition of Δ involving subscripts.
- An architecture × dataset cross-product expression.
- Two sample-size numbers (n, B).
- A target definition (lift_em=+1 with arrow).
- A base rate.
- Two CI conventions (95% thick, 99.67% thin).
- A multiple-comparison parameter (k).
- A survivor count (nine, four, three).
- A contribution name (FCT) + an acronym expansion + a Δ + a 95% CI + a description.

That is **9 numerical / definitional units in one paragraph.** A skim reader will give up at unit 3.

The expert reader needs every one of those units, but spread across the body and appendices, not piled into the caption.

### §2.3 Proposed rewrite (Pattern B — claim + key-number sentence)

**Option (a) — Tightest, ~50 words.** Recommended:

```latex
\caption{\textbf{Four trigger signals carry length-orthogonal information beyond the
length baseline; the strongest is \textsc{FCT} (first-commitment time, this paper).}
Forest plot of $\Delta = \mathrm{AUROC}(s_{\text{resid}}) - \mathrm{AUROC}(\texttt{tokens\_generated})$
on $n{=}800$ paired traces; thick bars are 95\% paired-bootstrap CIs, thin extensions are Bonferroni-corrected
99.67\% CIs at $k{=}15$. Setup details: \S\ref{sec:methods}; full numerical tables: App.~\ref{sec:appendix-bonferroni}, App.~\ref{sec:appendix-hero}.}
```

**Why this works:**

- Bold opening sentence is the takeaway a skim reader extracts in 4 seconds. "Four trigger signals carry length-orthogonal information; the strongest is FCT (this paper)."
- Bar conventions stay (necessary for the visual to be readable).
- All other quantitative details (n=800 unpacked, B=10,000, base rate, target definition, FCT Δ point estimate) move to the body — they are repeated in §4.1 and Fig. 1's call-out points.
- "(this paper)" inside the caption disambiguates FCT clearly without claiming the family.
- Pointer to §3 methods + App. A and App. B for the curious reader.

**Option (b) — Slightly longer, ~80 words, keeps the FCT Δ in the caption.** Use this if user prefers an in-caption number to anchor the headline:

```latex
\caption{\textbf{Length-orthogonal trigger signals on the headline pool.}
Paired-bootstrap forest plot of $\Delta = \mathrm{AUROC}(s_{\text{resid}}) - \mathrm{AUROC}(\texttt{tokens\_generated})$
on Qwen2.5-7B + Gemma-4-26B-A4B-it $\times$ \{HotpotQA, 2WikiMHQA\} ($n{=}800$ paired traces).
Thick bars: 95\% CIs; thin extensions: Bonferroni-corrected $\sim$99.67\% CIs at $k{=}15$.
Nine signals clear length, four survive Bonferroni across three families;
the strongest is \textsc{FCT} (first-commitment time, this paper; $\Delta = +0.190$ [+0.093, +0.287]),
a forward-looking variant of the answer-trajectory primitive (\S\ref{sec:methods-fct}).}
```

**Trade-offs:** (a) is the cleanest skim-friendly version; (b) carries one anchor number for the cite-and-skim reader. Both are scientifically accurate.

### §2.4 Scientific accuracy check

The current caption claims:
- *"the strongest length-robust trigger"* — correct (FCT has the largest Δ in the audited pool, +0.190; verified in `final_audit.md` §1.1 against `e7_bonferroni.out`).
- *"a forward-looking variant of the answer-trajectory primitive"* — correct **but** the "primitive" is singular; the answer-trajectory family already has multiple primitives (Datta's `AnswerSwitches`, Wang's CoP, Chen's instability JSD, etc.). Strictly accurate: "a forward-looking variant of the answer-trajectory family's commitment-time primitives." But this is pedantic — the singular "primitive" reads cleanly to a skim reader and the §1 / §3.3 chain spells out the parent. **No fix needed.**

The current caption does NOT claim FCT is novel (only "the strongest"). **The current caption is scientifically accurate.** The issue is purely density.

### §2.5 Recommended caption replacement (final)

Use **Option (a)** — the 50-word version. Move the n=800, B=10,000, base rate, lift_em=+1 definition, and FCT Δ details to body §4.1 (which already has them) or to a Fig. 1 footnote.

---

## §3. Abstract compression

### §3.1 Current abstract (L107–L109, ~285 words / ~1845 chars)

Per `00_litReview_and_framing_norms.md` §B.5, three trim-candidates identified:

### §3.2 Trim 1 — drop the family-list parenthetical (saves ~50 chars)

**Current:**
> *"... and four survive Bonferroni at $k{=}15$ across three signal families (attention, token-probability, answer-trajectory)."*

**Proposed:**
> *"... and four survive Bonferroni at $k{=}15$ across three signal families."*

**Why this is safe.** The three family names appear verbatim in:
- §1 contribution bullet 1 (L129)
- §4.1 first sentence (L185)
- §6 conclusion (L266)
- Fig. 1 caption (rewritten in §2.3 above)
- App. A Table 1 implicitly via the family-row labels

Five other paper locations carry the names; the abstract can survive without it. The skim reader who needs the names will find them in 5 seconds.

### §3.3 Trim 2 — pull the Bonferroni LB number out of the abstract (saves ~25 chars)

**Current:**
> *"FCT exceeds the length baseline by $\Delta = +0.190$ [+0.093, +0.287] (Bonferroni lower bound $+0.047$), the strongest length-orthogonal signal."*

**Proposed:**
> *"FCT exceeds the length baseline by $\Delta = +0.190$ [+0.093, +0.287], the strongest length-orthogonal signal."*

**Why this is safe.** The Bonferroni LB +0.047 is a careful-reader nuance, repeated in:
- §1 contribution bullet 2 (L130)
- §4.1 (L192) implicitly via the table reference
- App. A Table 1 (L306)
- Limitations §"Bonferroni knife-edge" (L273)

Four other locations carry it; abstract is a 1-second-attention zone where the LB is fenceposted noise, not signal.

### §3.4 Trim 3 — drop the motivational closing sentence (saves ~150 chars)

**Current closing:**
> *"Length-residualized and raw AUROC disagree most on the cells where retrieval matters most, motivating both protocols in dynamic-RAG trigger comparisons on multi-hop QA."*

**Proposed:** (delete entirely)

**Why this is safe.** The §6 Conclusion (L266) restates this exact sentence verbatim — *"Length-residualized and raw AUROC disagree most on the cells where retrieval matters most, motivating both protocols in dynamic-RAG trigger comparisons on multi-hop QA."* — so the closing is duplicated. A skim reader who reads only the abstract gets the methodological-recommendation closer too late (after the FCT and Janiak-robustness punchlines have already landed). Dropping it lets the abstract end on the strongest scientific finding (the LLM-judge robustness), which is a more memorable last sentence.

### §3.5 Optional Trim 4 — kappa parenthetical (saves ~25 chars)

**Current:**
> *"under a GPT-4o-Mini correctness function agreeing with EM at $\kappa = 0.707$, eight of nine survivors are preserved and FCT *strengthens* to $\Delta = +0.222$."*

**Proposed:**
> *"under a GPT-4o-Mini correctness function (vs.\ EM, $\kappa = 0.707$), eight of nine survivors are preserved and FCT *strengthens* to $\Delta = +0.222$."*

**Why this is safe.** The "agreeing with EM" verb-phrase is unnecessary; the parenthetical "vs. EM, κ=0.707" reads cleaner and is more typical of EMNLP abstract phrasing.

### §3.6 Net effect

Three trims (T1 + T2 + T3) save ~225 chars; with optional T4 ~250 chars. New abstract length: ~1600 chars / ~245 words. Comfortably inside EMNLP-Short comfort zone with all load-bearing claims preserved:

- "first length-controlled audit" + scope ✓
- 9 signals beat length, 4 survive Bonferroni, 3 families ✓
- FCT introduced + Δ point estimate + 95% CI ✓
- Anti-predictive 2Wiki finding ✓
- LLM-judge robustness with κ + survivor count + FCT strengthening ✓

**What is dropped:** family-name parenthetical (5 other locations carry it), Bonferroni LB (4 other locations), motivational closing (§6 carries it verbatim).

---

## §4. Line 279 — Correctness function paragraph audit

### §4.1 Current text (L279, Limitations)

> *"\paragraph{Correctness function.} We use exact-match equality on both arms of the paired-oracle target; the paired design shares the same EM function across arms (\S\ref{sec:methods-protocol}), and Appendix~\ref{sec:appendix-llmjudge} reports an LLM-as-Judge robustness check ($\kappa = 0.707$, eight of nine survivors preserved). Containment-match and $F_1$ variants remain future work."*

### §4.2 Three things this paragraph is doing

1. **Acknowledging EM is the chosen correctness function.** ✓
2. **Defending EM via the paired-design argument.** ✓ — paired-EM-equality across both arms partially insulates against Ielanskyi 2025's correctness-function-disagreement critique.
3. **Pointing to App. C for the LLM-judge robustness check.** ✓
4. **Naming two non-attempted alternatives (containment-match, F_1) as future work.** ✓ — but is this the right framing? See §4.4 below.

### §4.3 Is this correctly phrased? Yes — three minor refinements to consider

**Refinement (a) — Make the paired-design argument more visible.**

The current paragraph buries the load-bearing argument behind a `\S\ref{sec:methods-protocol}` pointer. The argument is:

> *"In the paired oracle decode `(closed_em, oracle_em)`, both arms apply the same EM function. The label `lift_em = +1 ⟺ EM(oracle) = 1 ∧ EM(closed) = 0` therefore depends only on the **rank-equivalence of EM** within each arm, not on EM's absolute correctness. A correctness function that systematically rates wrong-but-similar answers as correct (e.g., ROUGE-L threshold 0.3 per Janiak 2025) inflates `EM(closed)` and `EM(oracle)` together; the lift label is unchanged in expectation."*

This is the cleanest defense against Ielanskyi 2025's pitfall #2. The current Limitations paragraph alludes to it; making it more explicit is one short sentence.

**Proposed refinement:**

> *"\paragraph{Correctness function.} We use exact-match equality on both arms of the paired-oracle target. The paired design shares the same EM function across arms, so a uniformly-biased correctness function (e.g., ROUGE-L at threshold 0.3) inflates the closed-book and gold-context EM rates together and **the lift\_em label is invariant in expectation under such bias**; this partially insulates against the function-bias critique of \citet{ielanskyi_addressing_2025} (\S\ref{sec:methods-protocol}). Appendix~\ref{sec:appendix-llmjudge} additionally reports an LLM-as-Judge robustness check ($\kappa = 0.707$, eight of nine survivors preserved). Containment-match and $F_1$ variants remain future work."*

This makes the **paired-design-cancels-bias argument explicit**, which is one of the load-bearing pieces of the methodology. **Adds ~80 chars; recommended.**

**Refinement (b) — "Containment-match and F_1 variants remain future work" understates the design.**

This phrasing reads as if we did not consider F_1; in fact, the paired-EM design is **stronger than any single-arm F_1** because it cancels function bias across arms. F_1 is a robustness check, not a contribution gap. **Optional refinement:**

> *"... Containment-match and $F_1$ variants are reported as robustness checks in App.~\ref{sec:appendix-llmjudge}; we did not run them as headline targets because the paired-design argument transfers across any non-pathological correctness function."*

⚠ **Caveat:** this assumes containment-match and F_1 robustness checks already exist somewhere in the codebase — verify before committing. If they do not exist, the current "future work" framing is the right one.

**Refinement (c) — Add the kappa interpretation.**

Current: *"$\kappa = 0.707$, eight of nine survivors preserved"*. Adding "substantial agreement" via Landis-Koch is one phrase that the existing audit (§B in `final_auditNotes.md`) confirms is the right framing:

> *"Appendix~\ref{sec:appendix-llmjudge} additionally reports an LLM-as-Judge robustness check (Cohen's $\kappa = 0.707$, substantial agreement; eight of nine survivors preserved)."*

**Adds ~25 chars; optional but reads more confidently.**

### §4.4 Can this paragraph be re-framed as our contribution rather than a limitation?

**No, and we should not try.** Three reasons:

1. **Limitations is the right place for "what we did not do."** EM is the chosen function; F_1 / containment-match alternatives are future work. The reviewer expects this in Limitations, not in the contribution itemize.
2. **The paired-design-cancels-bias argument IS the contribution lift, but it is already credited in §3.1 (paired-oracle protocol section, L151) — *"The paired design shares the same exact-match correctness function across both arms, which partially insulates against the function-bias critique of \citet{ielanskyi_addressing_2025}."*** That is the right place for the contribution-flavor framing. The Limitations paragraph adds ones-removed risk acknowledgement (containment-match, F_1 not run).
3. **The §1 Contributions itemize bullet 4 already credits the LLM-judge robustness check as a contribution:**
   > *"A direct robustness check against the LLM-as-judge critique of \citet{janiak_illusion_nodate}: under a GPT-4o-Mini semantic-equivalence judge agreeing with EM at Cohen's $\kappa = 0.707$, eight of nine paired-significant survivors and all four Bonferroni-robust signals are preserved; FCT strengthens to $\Delta = +0.222$."*
   
   The contribution shape is fully captured here. The Limitations paragraph plays the role of "and yet there are alternative correctness functions we did not test as headlines" — which is honest and correct framing.

**Verdict on §4.4:** keep as a Limitations paragraph; do not promote to a contribution. The contribution shape is correctly placed in §1 bullet 4 and §3.1.

### §4.5 Final recommendation for line 279

**Adopt Refinement (a) — make the paired-design-cancels-bias argument explicit.** Adds ~80 chars; replaces a `\S\ref{}` pointer with a one-sentence statement of the argument. This is the single highest-value change to L279.

**Optionally Refinement (c) — add "substantial agreement" Landis-Koch hedge.** Trivial; reads more confidently.

**Skip Refinement (b)** unless containment-match / F_1 robustness numbers exist in the codebase. Verify with `e7_bonferroni.out` or `analysis/` outputs before claiming.

---

## §5. Cross-references for `02_section_by_section_audit.md`

The framing items above feed forward to the section-by-section audit:

| # | Source | Target audit section | Recommendation |
|---|---|---|---|
| F1 | §1.4 above | §3.3 + App. E | Add Zhang 2026 disambiguation; extend operational-axes table (Datta / Zhang / FCT). |
| F2 | §2.5 above | Fig. 1 caption (L114) | Rewrite to Pattern B (50 words, claim + key-number). |
| F3 | §3.6 above | Abstract (L107–L109) | Three trims: family parenthetical, Bonferroni LB number, motivational closing. |
| F4 | §4.5 above | Limitations §"Correctness function" (L279) | Make paired-design-cancels-bias argument explicit. |
| F5 | §1.5 above | (constraint) | Do not rename FCT, do not soften "we propose", do not drop Datta citation. |

---

*End of `01_framing_analysis.md`. Generated 2026-05-25 against `template.tex` + `00_litReview_and_framing_norms.md`.*
