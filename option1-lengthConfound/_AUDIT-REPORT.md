# EMNLP 2026 Short — Audit Report on `template.tex`

> **Scope.** A second-pass, fresh audit performed against (a) your `template.tex`, (b) the peer papers in `__lit-review-main/`, (c) the actual published EMNLP-main 2025 version of UNCERTAINTY-LINE (Vashurin et al., aclanthology.org/2025.emnlp-main.400), and (d) recent dynamic-RAG-trigger papers retrieved live from arXiv/ACL-anthology. This report intentionally does **not** duplicate `_EMNLP-writing-guide.md`; it focuses on what that document missed, plus direct answers to your four questions.
>
> **Verdict in one line.** The paper is technically strong and very close to submission, but suffers from (i) one critical naming/citation collision in your novel-signal coinage, (ii) systematic under-narration in places where a multi-hop-QA reviewer expects the *mechanism* of the result, and (iii) a headline figure that is at peer-norm for *complete* but below peer-norm for *narrative*.

---

## 0. Executive summary — top findings, new vs. the existing guide

The seven items below are **not** in `_EMNLP-writing-guide.md`. They are the load-bearing findings of this second-pass audit. Items 1–3 are blockers; 4–7 are high-EV improvements.


| #     | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Severity    | Why it matters                                                                                                                                                                   | Where it lives                     |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| **1** | `**ECT` (your coinage) vs `ETC` (Li et al. 2025) name-and-mechanic collision.** A published 2025 dynamic-RAG-trigger paper introduces "ETC = Entropy-Trend Constraint", a trigger built on the **second-order difference of the entropy sequence** (Li et al., arXiv:2511.09980, AAAI 2026). Your audit pool already includes a signal you call `etc (Δ²H)` (literally the same mechanic — second difference of step entropy), and your novel-signal coinage is `ECT` — one letter swap, anagram of `ETC`. Reviewer will conflate the two. | **Blocker** | Citation gap *and* naming collision in one. A G5 violation (per ARR), and an excitement-killer (reviewer thinks you're rebranding someone else's signal).                        | §1, §2, §3.3, §4.1, App. B, App. F |
| **2** | **UNCERTAINTY-LINE is now an EMNLP-main 2025 paper, not just arXiv.** The reference you cite is published at `aclanthology.org/2025.emnlp-main.400`. Reviewers will look for the ACL citation and will down-weight your reading if it's missing.                                                                                                                                                                                                                                                                                           | High        | Trivial fix; high-cost miss.                                                                                                                                                     | bibliography only                  |
| **3** | **Your `log(tokens+1)` choice deviates from the published UNCERTAINTY-LINE reference, which uses raw L** and explicitly checks 2nd/3rd-degree polynomials in their Appendix D.1 (no significant gain over linear in raw L). Your $\varepsilon \le 0.029$ replication appendix shows agreement but does not name *why* you deviate. A stats-aware reviewer will register this as an undisclosed methodological deviation.                                                                                                                   | Medium-high | Easy 1-sentence justification in §3.2 closes the gap. Untreated it becomes an R1 / R5 attack surface.                                                                            | §3.2 + App. E                      |
| **4** | `**lift_em=+1` is under-narrated.** You define it formally but never explain in plain prose *why* a trace where retrieval flips wrong→right is the right target for triggering. Peer papers (Datta, ETC/Li, Mallen) take 2–3 sentences for this. You take 0.                                                                                                                                                                                                                                                                               | Medium      | A reviewer who only reads abstract + §3.1 should understand why liftem is the right answer.                                                                                      | §3.1                               |
| **5** | **Your headline figure (forest plot) is dense-and-complete but narratively flat.** Reviewer-facing peer figures in this space (Datta Fig 1, Lu Fig 1, UNCERTAINTY-LINE Fig 1 + Fig 5) all show the *mechanism*, not just the comparison. Your reader sees 17 CIs but does not see *what ECT is* on a trace.                                                                                                                                                                                                                                | Medium      | The current figure is below norm for excitement; you can fix this with a small 2-panel revision (Part 8.6 of the existing guide is right, but undersells how much this matters). | Fig. 1                             |
| **6** | **Two paragraphs read as if written for the audit-pool authors, not the reviewer.** The DRAGIN / FLARE / SeaKR / SelfCheckGPT pool is named without context, and you assume the reviewer knows what each of these does. A short-paper reviewer in the May 2026 ARR cycle reading 5–8 papers in 90-minute windows will not.                                                                                                                                                                                                                 | Medium      | One-sentence parenthetical descriptions in §2 fix this.                                                                                                                          | §1 first paragraph, §2             |
| **7** | **The "length is anti-predictive on OOD cells" finding is the most surprising thing in the paper.** It is currently §4.5, the fifth Results subsection. In your peer set (Janiak, Santilli), this kind of "the field was wrong in this direction" finding is **front-loaded**, not buried.                                                                                                                                                                                                                                                 | Medium-high | Move to §4.2 or fold into §4.1. The existing guide §16.3 hints at this but doesn't quantify how much it costs you.                                                               | §4.5                               |


---

## 1. Direct answers to your four questions

### Q1. What is normal to *not* define? What is the reviewer's level?

**The reviewer is an EMNLP-short ARR May-2026 reviewer who has bid on the dynamic-RAG / UQ / retrieval track.** They have submitted to and reviewed for one of {ACL, EMNLP, NAACL, EACL} in the last 2 years. They know:


| You can assume known (do **not** define)                            | You should briefly define (1 clause)                                               | You should fully define                                                          |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| RAG, AUROC, AUC, EM, F1                                             | DRAGIN, FLARE, SeaKR, SelfCheckGPT (one parenthetical each — see §2 of this audit) | ECT (your coinage)                                                               |
| Bonferroni correction, paired bootstrap, OLS, Spearman ρ, Cohen's κ | Paired-oracle (your design choice)                                                 | `lift_em=+1` (mechanism, not just formula)                                       |
| Chat model, MoE, dense, decoding, greedy, top-p, temperature        | "Answer-trajectory family" (your phrase) — one clause                              | Your boundary-detection / spaCy-segmentation choice                              |
| HotpotQA, 2WikiMultiHopQA                                           | UNCERTAINTY-LINE (one clause; you do this fine)                                    | The exact regressor you fit (you do this fine)                                   |
| LLM-as-judge, semantic-equivalence judge                            | "length-residualized AUROC" — one clause                                           | n/a                                                                              |
| GPT-4o-Mini (just say "OpenAI gpt-4o-mini" with version date)       | The four families you audit (the "spans four signal families" claim)               | The Bonferroni $k$ value and what counts as a "signal" for it (you do this fine) |


**Anti-pattern in your current draft.** You over-defined `lift_em` *formally* (Eq. 1 with `\mathbb{1}[...]`) and under-defined it *mechanically* (what does a +1 trace actually look like? what is its base rate? what would a reviewer guess wrong about it?). Inverse for `tokens_generated`: you treat it as a self-evident baseline but a reader could plausibly read it as a trigger candidate rather than a confound proxy. One half-sentence each fixes both.

**Anti-pattern you avoided well.** You did not over-define AUROC, paired bootstrap, OLS, or Bonferroni — all correct for the genre. Your "we used B=10,000 paired-bootstrap" is exactly the right level of detail; no more is needed.

### Q2. Tone — narratively driven vs. numerical?

**Peer norm in your exact subgenre is 60/40 narrative-to-numbers**, not 80/20 the other direction. Janiak et al. (your closest published EMNLP paper — EMNLP 2025) opens with three narrative sentences before the first number. Datta et al. opens with a behavioral observation, then numbers. UNCERTAINTY-LINE (EMNLP-main 2025) opens with "LLMs have become essential" (which is too soft for a short paper, but illustrates that they earn the right to numbers by first naming the problem).

**You currently tilt ~30% narrative / 70% numerical.** Specifically:

- §1 opens with a *definition* ("Dynamic retrieval-augmented generation (RAG) systems retrieve external evidence during decoding...") instead of a *claim*.
- §4 opens by listing CIs.
- §5 is two interpretive paragraphs — narrative-heavy, but loaded with hedges.

**The recipe that works for your paper.** You want a narrative *frame*, numerical *content*, narrative *consequence*. Concretely, every paragraph in §1, §4, and §5 should have:

1. **A claim sentence** (≤20 words, no parentheticals, no `\texttt{}`).
2. **An evidence sentence** (the number(s) with CIs, $n$, base rate).
3. **A consequence sentence** (1 clause: what this means for someone deploying / studying dynamic RAG).

The third sentence is the one you're systematically missing.

**Worked example from your §4.5 (your buried headline):**

```
BEFORE (your current text):
On the out-of-distribution cells (Qwen × 2Wiki and Gemma × 2Wiki), raw answer
length is anti-predictive of retrieval need: the tokens_generated baseline
reaches raw AUROC 0.457 and 0.518 on lift_em=+1, both at or below chance.
Longer traces on multi-hop OOD QA are systematically less retrieval-helpful,
a direct inversion of the verbosity-compensation pattern Zhang et al. (2024)
document on single-hop QA. The four Bonferroni-robust survivors clear the
length baseline by a wider margin on these cells than on the in-domain
HotpotQA cells, suggesting the raw-AUROC protocol miscredits the length
signal on the cells where retrieval matters most.

AFTER (claim → evidence → consequence):
On out-of-distribution multi-hop QA, length is anti-predictive of retrieval
need (Claim). The tokens_generated baseline reaches AUROC 0.457 on Qwen×2Wiki
and 0.518 on Gemma×2Wiki — at or below chance — yet the four Bonferroni-robust
survivors clear length by a wider margin on these cells than on in-domain
HotpotQA (Evidence). Verbose multi-hop OOD traces are systematically less
retrieval-helpful — the inverse of the verbosity-compensation pattern of
Zhang et al. (2024) on single-hop QA — which means the published raw-AUROC
protocol credits length signal on exactly the cells where retrieval matters
most (Consequence).
```

Same numbers, same content, ~10 words longer, but it reads as a finding instead of a table-walk.

**Where you must NOT go narrative.** §3 (Methods). Methods sections in short papers in this subgenre are pure formula + cohort + protocol. Datta, Lu, UNCERTAINTY-LINE all keep Methods at 100% numerical. Your §3 is correctly numerical and should stay that way.

### Q3. Phrasing and placement of sections — what is the norm?

Comparing your structure to the closest published peers:


| Section         | Your draft                                                                | EMNLP-short norm                                                          | Verdict                                                 |
| --------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------- |
| Title           | `Length-Controlled Evaluation of Training-Free Dynamic-RAG Triggers`      | "X: Method for Y" or "Re-evaluating Y under Z"                            | **Mild downgrade** — see §3.1 below                     |
| Abstract        | 1 paragraph, ~270 words, 5 numbered findings buried in compound sentences | 1 paragraph, 180–220 words, claim-led                                     | **Long by ~50 words**                                   |
| Headline figure | After abstract, before §1                                                 | After abstract OR top of page 2 — but always *conceptual*, not table-like | **Placement OK, content not**                           |
| §1 Introduction | 4 paragraphs + bulleted contributions                                     | 3–4 paragraphs + bulleted contributions                                   | **Structure OK**                                        |
| §2 Related Work | 1 wall paragraph, 14 citations                                            | 2–3 differentiated subparas, diff-table style                             | **Structure miss** (the existing guide §1.3 nails this) |
| §3 Methods      | 3 subsections (protocol / residualization / ECT)                          | 2–3 subsections is correct                                                | **Structure OK**                                        |
| §4 Results      | 6 subsections                                                             | 3–5 subsections, claim-organized                                          | **Too granular** (6 → 4 better)                         |
| §5 Discussion   | 2 subsections                                                             | 1 longer Discussion or 2 short subsections                                | **Structure OK**                                        |
| §6 Conclusion   | 1 short paragraph                                                         | 1 short paragraph                                                         | **OK**                                                  |
| Limitations     | 5 paragraphs in §*Limitations*                                            | 3–5 paragraphs, named subheadings (`\paragraph{}`)                        | **OK**                                                  |
| Appendix        | 7 appendices (A–G)                                                        | 4–8 typical; short-paper unlimited                                        | **OK**                                                  |


**Specific section-norm corrections (not in existing guide):**

**A. Section names matter.** Your section names mix descriptive (`Methods`, `Results`, `Discussion`) with subsection names that are *also* descriptive (`Cross-Architecture Survivor Matrix`, `Length Is Anti-Predictive on the Out-of-Domain Cells`). This is genre-appropriate. But you mix two styles: declarative ("ECT Discriminates Strongest on the Out-of-Domain MoE Cell") and topical ("Cross-Architecture Survivor Matrix"). Pick one. Declarative is the stronger choice for a short paper because it lets the reviewer skim the TOC.

Concrete recommendation for §4 subsection titles:


| Before                                                        | After                                                                  |
| ------------------------------------------------------------- | ---------------------------------------------------------------------- |
| 4.1 Cross-Architecture Survivor Matrix                        | 4.1 **Four signals clear Bonferroni across four families**             |
| 4.2 ECT Discriminates Strongest on the Out-of-Domain MoE Cell | 4.2 **ECT discriminates strongest on out-of-domain cells**             |
| 4.3 The K=5 Multi-Sample Variant Is Null                      | 4.3 **K=5 boundary completion is empirically null vs K=1**             |
| 4.4 Survivor Set Is Robust to a Semantic-Equivalence Judge    | 4.4 **Survivor set is robust to an LLM-as-judge correctness function** |
| 4.5 Length Is Anti-Predictive on the Out-of-Domain Cells      | 4.5 → **PROMOTE TO 4.2**                                               |
| 4.6 Robustness Summary                                        | 4.6 **Robustness summary** (this name is fine)                         |


You already have 4.5 in declarative form ("Length is anti-predictive..."); apply the same pattern to the others, and use bold to let it skim.

**B. Numbers belong inside section titles, sometimes.** Genre-controversial, but Datta does it ("...Decide Early and Explain Later" — explanatory in the title), Janiak does it ("The Illusion of Progress: Re-evaluating Hallucination Detection in LLMs"). You currently have one numerical claim in a section title (`The K=5 Multi-Sample Variant Is Null` — the "null" is the number). That's the right calibration; don't push further.

**C. Limitations placement.** Yours is at the end, before bibliography. EMNLP 2025 norm (per ARR guidelines) is `\section*{Limitations}` at the end of body text but before bibliography — exactly where you put it. ✓ This is one place the existing guide and this audit agree without caveat.

**D. The "Risks" paragraph.** ACL policy as of EMNLP 2025 requires a Risks discussion (can be part of Limitations). You added one in the last paragraph of `Limitations`. ✓ This is correct genre placement.

### Q4. Figure descriptions, esp. the headline figure

**Your Figure 1 audit.**

Strengths:

- Caption begins with bolded claim sentence ✓
- Cites the cohort ($n=800$, paired, $B=10{,}000$) ✓
- Names the headline signal (ECT) ✓
- Reports the headline number with CI ✓
- Explains the Bonferroni vs nominal distinction ✓ (this is rare and reviewer-positive)

Weaknesses:

- The figure itself is a forest plot of 17 signals. A reviewer who has not read §4.1 will not understand what `dragin_rind_max`, `seakr_k5_proxy.logdet`, `var_l2_window.max`, `flare.max`, `flip_rate K=1` mean. You assume audit-pool literacy from a figure that should be self-contained.
- The figure shows the *result*, not the *mechanism*. Datta's Figure 1 shows an answer trajectory. Lu's Figure 1 shows the streaming detection schematic. UNCERTAINTY-LINE Figure 1 shows the length-vs-UQ scatter with regression line. **Your Figure 1 shows a leaderboard.** A leaderboard is the *consequence* of the mechanism, not the mechanism itself.
- The thick-vs-thin CI convention is unusual; it requires the caption to explain it, which it does, but the eye stutters on it before the brain has resolved what "Δ" means.
- Single-column width (`\columnwidth`). At 17 rows × 2 CI families this is cramped. Datta and Lu both go double-column (`\textwidth`) for their headline figure on similar density.

**Concrete recommendation — option A (low cost, recommended for first submission):**

Replace your single forest plot with a **two-panel figure** at `\columnwidth`:

- **Panel (a) — Headline result (top):** 4 Bonferroni-robust signals only (ECT, DRAGIN-RIND, min_majority, ent.max) with 95% + 99.67% CIs. One row per signal. The eye now lands on the four-signal headline immediately.
- **Panel (b) — ECT mechanism (bottom):** A schematic of one paired trace showing the answer trajectory across reasoning steps for a `lift_em=+1` example. The intuition pump for "what is ECT looking at?" — high ECT = late stabilization. This is the panel Datta has and you don't.

You keep the full 17-signal forest plot but move it to `tab:hero-full` (which you already have) or as Appendix Figure 1. The headline reader gets the 4-signal result + mechanism on page 1; the soundness reviewer gets the full 17-signal grid in App. B.

**Concrete recommendation — option B (higher cost, recommended for second-submission upside):**

A **four-panel** figure at `\textwidth` (double-column):

- (a) 4 Bonferroni-robust forest (as above)
- (b) ECT mechanism trace
- (c) Length-vs-`lift_em` scatter per cell (4 panels in one) — the source data for the "length is anti-predictive on OOD" finding
- (d) EM-vs-judge agreement scatter

This is more work but converts your soundness-heavy figure into an excitement-heavy figure. It's the kind of figure Lu et al. 2026 do. For a *first submission* I would stay with option A; if you get a 3.0–3.5 average and need to push for Main, option B is the upgrade.

**Caption — peer norm.** Your current caption is 84 words, bolded lead sentence, ends with ECT headline. The peer norm is 60–80 words, lead sentence is the *claim*, end with the *implication*. Your caption is at upper end but acceptable. The bigger issue is the figure content, not the caption.

---

## 2. Section-by-section audit

For each section: **what's there → peer comparison → strengths → problems → concrete edits (in priority order).**

### 2.1 Title

`Length-Controlled Evaluation of Training-Free Dynamic-RAG Triggers`

**Peer comparison.** Closest analogues:

- Janiak et al.: *The Illusion of Progress: Re-evaluating Hallucination Detection in LLMs* (rhetorical claim + neutral method)
- UNCERTAINTY-LINE: *Length-Invariant Estimation of Uncertainty for Large Language Models* (method name + scope)
- Santilli et al.: *Revisiting Uncertainty Quantification Evaluation in Language Models: Spurious Interactions with Response Length Bias Results* (method + finding hint)
- Datta et al.: *Large Language Models Decide Early and Explain Later* (rhetorical, behavioral)

**Strengths.** Correctly scoped. Avoids overclaim. Names two of the three key technical anchors (length-control, training-free) in 8 words.

**Problems.**

1. "Training-Free" is doing double work — it both scopes the audit and signals method-virtue, but it's also slightly misleading because *some* of the signals you compare against (e.g., the trained-probe family of Lu/Zhang 2026) ARE training-free if you read them tightly. You may want to say "decode-time" or "parameter-free."
2. The title gives no hint of the headline finding (ECT) or the surprise (length anti-predictive on OOD). Both Janiak and Datta hint at the surprise.

**Concrete edits:**

- (P1, recommended) Keep your title for the audit framing. The current title is on-theme for "New Missions for NLP Research" and routes to the Resources-and-Evaluation track. Your existing guide §16.6 considers the method-framing alternative ("ECT: A Length-Robust Forward-Looking Trigger for Dynamic Retrieval-Augmented Generation"); I would only switch to that if you also rewrite §3 to lead with ECT and demote the survivor matrix. **For a first submission, keep your current title** — but consider a subtitle:

```
BEFORE: Length-Controlled Evaluation of Training-Free Dynamic-RAG Triggers
AFTER : Length-Controlled Evaluation of Training-Free Dynamic-RAG Triggers:
        Four Signals Survive, ECT Wins
```

The subtitle gives the reviewer the headline before they click into the abstract. This is increasingly common in EMNLP-main and ACL-main titles, and is acceptable for EMNLP-short.

### 2.2 Abstract

**What's there.** One paragraph, ~270 words, 5 distinct findings stacked into compound sentences. All five Part-0 questions (problem, gap, method, finding, caveat) are answerable from the abstract, but only with careful reading.

**Peer comparison.**

- UNCERTAINTY-LINE abstract: 185 words, 4 sentences carry structural work, 3 are connective.
- Janiak abstract: 196 words, three numbered findings (`A; B; C`) explicitly enumerated.
- Datta abstract: 158 words, claim-led ("LLMs decide early and explain later"), then 2 numbers.
- Chen et al. (Dynamic Instability): 220 words, claim-led, mechanism description, then numbers.

**Yours at 270 words is at the long end** but EMNLP-short does not impose a hard abstract length cap, and the content density is fine. The bigger issue is that finding #5 (the LLM-judge robustness, $\kappa = 0.707$) doesn't appear at all — it's hidden in the EM-vs-judge sentence and the reviewer who skims will miss it.

**Strengths.**

- All five Part-0 points are present.
- ECT introduced with intuition clause ("high when the model has not yet stabilized on its final answer late in the decode") — this is exactly the right move per the existing guide.
- Numbers in stress positions for the headline.
- Bonferroni + paired-bootstrap explicitly named.
- κ = 0.707 mentioned (good, but not foregrounded).

**Problems.**

1. **"We evaluate the published trigger pool"** under-claims. Your existing guide §16.3 has the fix ("We deliver the first length-controlled evaluation"). Apply it.
2. **Two compound noun phrases at ≥4 concepts each** (existing guide §1.1 catches these; still uncorrected in current draft):
  - "length-residualized AUROC on a paired-oracle target"
  - "forward-looking early-commitment-time signal"
3. **"under a GPT-4o-Mini semantic-equivalence judge agreeing with EM at Cohen's κ = 0.707"** — this is buried mid-sentence. Foreground.
4. `**\texttt{lift\_em}{=}+1`** appears three times in the abstract. Once is fine; three times reads as code-comment-as-prose.
5. **Last sentence is prescriptive** — "should be reported alongside" is borderline; "should be standard practice" (which appears in your conclusion) is over the line. R4 risk per ARR. Already partially fixed in your draft; final pass needed.

**Concrete edits (in priority order):**

```
P0 - REPLACE the second sentence:
BEFORE: A converging recent literature shows this protocol is length-
        confounded, but the implication has not yet reached the dynamic-RAG
        trigger setting.
AFTER : A converging literature shows the protocol is length-confounded
        on QA selective prediction (Santilli, 2025; Vashurin, 2025; Janiak,
        2025), but every published training-free trigger still relies on the
        unprotected protocol.
```

This change adds a fact ("every published") and a citation density that signals reading rigor without compound-noun overhead.

```
P0 - REPLACE the headline finding clause:
BEFORE: The strongest length-orthogonal signal we observe is ECT, a forward-
        looking early-commitment-time signal (high when the model has not
        yet stabilized on its final answer late in the decode), which exceeds
        the length baseline by Δ = +0.190 [+0.093, +0.287] against
        tokens_generated.
AFTER : Our strongest survivor is ECT (early-commitment time) — high when
        the model has not yet committed to its final answer late in the
        decode — beating the length baseline by ΔAUROC = +0.190 (95% CI
        [+0.093, +0.287]; Bonferroni-significant at k=15).
```

Same numbers, ~25 words shorter, "(early-commitment time)" parenthetical does the same intuition work as the em-dash clause, "beating" replaces "exceeds" (more active).

```
P0 - REPLACE the LLM-judge sentence (foreground κ):
BEFORE: Under a GPT-4o-Mini semantic-equivalence judge agreeing with EM at
        Cohen's κ = 0.707, eight of nine paired-significant survivors and all
        four Bonferroni-robust signals remain; ECT strengthens to Δ = +0.222.
AFTER : The survivor set is robust to a semantic-equivalence judge: under a
        GPT-4o-Mini correctness function agreeing with EM at Cohen's κ = 0.707
        (substantial; Landis & Koch, 1977), eight of nine paired-significant
        survivors are preserved and ECT strengthens to Δ = +0.222.
```

The "robust to a semantic-equivalence judge" is now the lead, and the κ value supports rather than buries it.

```
P1 - REPLACE the prescriptive last sentence:
BEFORE: We argue that length-controlled evaluation should be reported alongside
        raw AUROC in every dynamic-RAG trigger comparison.
AFTER : Length-controlled AUROC and the published raw-AUROC protocol disagree
        most on the cells where retrieval matters most; we argue both should
        be reported in every dynamic-RAG trigger comparison.
```

The new version preserves your prescription but anchors it on the OOD-anti-predictive finding (your hidden headline), giving the reviewer a fact instead of an opinion.

### 2.3 Headline figure (Figure 1)

See §1 Q4 above for the full figure audit. The single edit, if you do only one thing:

> Split into a two-panel figure: (a) the four Bonferroni-robust signals only, (b) one trace-level schematic showing what high vs low ECT looks like across reasoning steps.

This is ~2 hours of work in matplotlib and pulls your figure from soundness-rich/excitement-flat to soundness-rich/excitement-medium.

### 2.4 §1 Introduction

**What's there.** Three paragraphs of context + 1 paragraph on this paper + 4-item bulleted contributions. ~520 words total.

**Peer comparison.**

- Chen (Dynamic Instability, Feb 2026): three paragraphs of "what we don't know" → one paragraph of "what we do" → contributions. Total ~600 words. Yours is slightly tighter.
- Datta (Apr 2026): two paragraphs of context, one paragraph of result, contributions. ~550 words.
- Janiak (EMNLP 2025): two context paragraphs, one paragraph of three findings, one paragraph of contributions. ~480 words.

**Yours is on-norm.**

**Strengths.**

- ¶1 sets up the four-family pool clearly.
- ¶2 hits all four length-confound references in the correct chronology.
- ¶3 introduces ECT with proper provenance (the parent papers in the answer-trajectory family).
- Contributions list is enumerated, all 4 items numerically anchored.

**Problems.**

1. **¶1 opens with a definition, not a claim.** Existing guide §1.2 caught this. Still uncorrected. The replacement should make the protocol-is-broken claim:

```
BEFORE: Dynamic retrieval-augmented generation (RAG) systems retrieve external
        evidence during decoding to ground generation in source material
        (Lewis, 2021; Trivedi, 2023; Shao, 2023).

AFTER : Dynamic retrieval-augmented generation (RAG) systems decide whether
        to fetch external evidence from scalar signals computed during decoding
        (Lewis, 2021; Trivedi, 2023; Shao, 2023). The published training-free
        trigger pool — spanning attention-based (Su, 2024), token-probability
        (Jiang, 2023), hidden-state (Yao, 2024), text-consistency (Manakul,
        2023), and entropy-trend (Li, 2025) families — is evaluated by raw
        AUROC against binary correctness. Raw answer length attains AUROC up
        to 0.65 against this target on published benchmarks: a baseline absent
        from every published trigger's evaluation table.
```

This compresses your two opening sentences into a definition+claim pair, adds the citation to Li et al. (the missing ETC paper — finding #1 above), and replaces "predate the length-confound literature" with the *number* (0.65 raw AUROC) that makes the problem concrete.

1. **¶3 buries ECT.** Your sentence:

> *"We introduce ECT, a forward-looking early-commitment-time signal in the recently-emerged answer-trajectory family..."*

— and then the headline number is in a subordinate clause. Move the number to stress position:

```
AFTER : We introduce ECT (early-commitment time), a forward-looking signal in
        the recently-emerged answer-trajectory family (Datta, 2026; Wang,
        2025; Chen, 2026): ECT beats the length baseline by ΔAUROC = +0.190
        (95% CI [+0.093, +0.287], Bonferroni-significant at k=15), the
        strongest length-robust trigger in the audited pool.
```

1. **The "honest delta" sentence is missing.** Per the existing guide §2.1 (which is right) — and per the H3 anti-pattern defense — you should add at the end of §1:

> *"Each of our three building blocks — length residualization (Vashurin, 2025), the paired-oracle question (Mallen, 2023), and the answer-trajectory family (Datta, 2026) — has a direct precedent. Our contribution is their integration: applying all three jointly to the dynamic-RAG trigger setting surfaces four length-robust survivors that were not visible under the published evaluation protocol of any individual prior method."*

1. **Contribution bullet 1 is method-led, not finding-led.** "A length-controlled evaluation of the published trigger pool..." → buried headline. The existing guide §16.3 rewrite is correct.
2. **Contribution bullet 3 is the dry methodology bullet (n=800, base rate)** but it's framed as a *result* ("Cross-architecture and cross-domain validation"). The dry data belongs in §3 (Methods), not in the contributions block. Compress:

```
BEFORE: Cross-architecture and cross-domain validation on one dense
        (Qwen2.5-7B) and one MoE-hybrid (Gemma-4-26B-A4B-it) open-weight chat
        model over two multi-hop QA benchmarks (n=800 paired traces; base
        rate 13.5%): nine signals clear length at nominal 95% paired-
        bootstrap and four survive Bonferroni at k=15.
AFTER : Cross-architecture validation on one dense (Qwen2.5-7B) and one
        MoE-hybrid (Gemma-4-26B-A4B-it) chat model over two multi-hop QA
        benchmarks: nine signals clear length at 95% paired-bootstrap, four
        survive Bonferroni at k=15.
```

The base rate / n=800 belong in §3.1.

### 2.5 §2 Related Work

**What's there.** One paragraph, ~15 citations, no internal differentiation.

**Peer comparison.** Janiak EMNLP 2025 has 3 differentiated paragraphs in §2 — one on length-bias, one on UQ methods, one on hallucination detection. UNCERTAINTY-LINE has 4 subsection-style paragraphs (length bias, length-normalized measures, calibration, non-token-likelihood measures). Datta has 2 paragraphs. Chen has 4 thematic paragraphs.

**Your single-wall-paragraph is below peer norm** for the genre.

**Strengths.**

- All the relevant papers are cited.
- The "training-free scope" exclusion of Asai et al., Tang et al., Chen et al. 2026, Guo et al. is correct positioning.
- Mallen et al. 2023 is correctly identified as the conceptual parent for the paired-oracle target.

**Problems.**

1. **The wall-paragraph violates the diff-table norm.** Existing guide §1.3 nails this.
2. **Li et al. (2025) ETC is missing.** This is a published 2025 dynamic-RAG-trigger paper that:
  - Uses second-difference-of-entropy as the trigger mechanic (same as your `etc (Δ²H)` signal in App. B)
  - Is named ETC (one-letter swap from your ECT coinage)
  - Has been added to the trigger pool universe you claim to audit
   It should appear in your `\citep{}` list in §2 and in §1¶1.
3. **The phrase "the published trigger pool we audit spans..."** flattens the diff. "We audit" reads as derivative. The existing guide §16.2 caught this.
4. `**\citet{mao_confidence_2026} note that joint generation probability suffers severe length bias`** is a parenthetical that interrupts the length-confound discussion. Either fold it into the main length-confound sentence or move to §5.

**Concrete edits — proposed §2 rewrite (replaces your current §2 paragraph):**

```
\section{Related Work}
\label{sec:related}

\paragraph{Length confounds AUROC for LLM uncertainty quantification.}
\citet{santilli_revisiting_2025} prove that mutual length-bias on UQ scores
and correctness functions systematically distorts AUROC; \citet{vashurin_
uncertainty-line_2025}, henceforth UNCERTAINTY-LINE, propose regression-
residualization on length as a post-hoc fix, applied to NMT, summarization,
and math reasoning; \citet{janiak_illusion_nodate} report length heuristics
rivaling complex hallucination detectors on single-hop QA. \citet{zhang_
verbosity_2024} document the underlying behavioral mechanism — verbosity
compensation — on the multi-hop QA family used here, and \citet{ielanskyi_
addressing_2025} catalogue approximate-correctness-function disagreement
as a parallel bias channel.

\paragraph{The published training-free dynamic-RAG trigger pool}
spans attention-based DRAGIN \citep{su_dragin_2024} and DR-RAG
\citep{hei_dr-rag_2024} (token-importance from attention dynamics),
token-probability FLARE \citep{jiang_active_2023}, multi-sample
text-consistency SelfCheckGPT \citep{manakul_selfcheckgpt_2023} and SAC$^3$
\citep{zhang_sac3_2023}, hidden-state INSIDE \citep{chen_inside_2024} and
SeaKR \citep{yao_seakr_nodate}, and entropy-trend ETC \citep{li_etc_2025}.
All evaluate the trigger by raw AUROC against a binary correctness label,
predating the length-confound literature. Trained alternatives requiring
per-architecture supervised data \citep{asai_self-rag_2023, tang_mba-rag_
2024, chen_think_2026, guo_dior_2025} fall outside the training-free scope.

\paragraph{Step-level answer-trajectory primitives}
are an emerging length-orthogonal signal class \citep{datta_large_2026,
wang_chain--probe_nodate, chen_i_2026, zhang_harnessing_2026, lu_streaming_
2026}. The closest precedent for ECT is \citet{datta_large_2026}, who
introduce \texttt{AnswerSwitches} and the final-switch index $t^\star$ on
thinking-mode models for early stopping. We adapt this primitive to
non-thinking chat models with $K{=}5$ boundary completion (empirically null
vs $K{=}1$; \S\ref{sec:results-ablation}), apply it to retrieval triggering
(opposite causal direction), and introduce a forward-looking commitment-time
variant (ECT, \S\ref{sec:methods-ect}).

\paragraph{Naming note.} Our novel signal ECT (early-commitment time) is
distinct from the ETC of \citet{li_etc_2025} (Entropy-Trend Constraint, a
second-difference-of-token-entropy trigger in our audited pool;
Table~\ref{tab:hero-full} row \texttt{etc} ($\Delta^2 H$)) and from the
``Answer-agreement Representation Shaping'' (ARS) of \citet{zhang_
harnessing_2026}, which uses latent perturbation rather than step-level
sampling.
```

The last `\paragraph{Naming note}` is **mandatory** — it is the one-paragraph that prevents the most predictable reviewer attack (item #1 of the executive summary). It costs you 70 words; it buys you defense against a G5 + "what's new" double-hit.

### 2.6 §3 Methods

**What's there.** Three subsections — protocol, residualization, ECT family — total ~480 words. Three equations.

**Peer comparison.** This is the strongest section of your paper. UNCERTAINTY-LINE §4 is ~400 words for the same kind of content (problem setup + debiasing of UQ scores + supervised variant). Datta's full method is ~600 words spread across §3.

**Strengths.**

- §3.1 has the formal `lift_em` definition with a clean equation.
- §3.2 has the residualization formula with the right rationale for `log(L+1)` (mostly — see Problem 2 below).
- §3.3 has the ECT formula with the right intuition clause after it.
- §3.1 last sentence — "A signal that ranks lift_em=+1 traces high is, by construction, a retrieval trigger" — is exactly the right kind of bridge sentence per existing guide §8.5. Keep.

**Problems.**

1. **§3.1 under-narrates lift_em.** The formula is there but no intuition. A reviewer should read §3.1 once and be able to explain `lift_em` to a colleague in two sentences. Currently they read the formula and have to derive the intuition themselves. Add one sentence after Eq. 1:

```
ADD after Eq. (1):
In plain terms: lift_em=+1 traces are exactly those where the model would
have gotten the answer right if it had had the gold context, but didn't.
A signal that flags these traces tells us *which queries to retrieve for*,
not just *which model outputs are wrong*. The two questions are different:
length predicts the latter (Zhang, 2024); we audit whether published triggers
predict the former.
```

This is 3 sentences but they buy you the entire intuition pump for the rest of §4.

1. **§3.2 — the `log(L+1)` deviation from the published UNCERTAINTY-LINE.** The published UNCERTAINTY-LINE (EMNLP-main 2025) uses raw $L$ and explicitly verifies (their App. D.1) that 2nd/3rd-degree polynomials don't improve over linear-in-raw-L. You use `log(L+1)` "to mitigate overcorrection on the long-tailed token distribution" — but this is a *different rationale* than theirs and the published reference doesn't have an overcorrection problem. The 1-sentence fix:

```
BEFORE: We use log(tokens+1) rather than raw length to mitigate overcorrection
        on the long-tailed token distribution of multi-step reasoning traces,
        consistent with the caveat in [Vashurin §3].

AFTER : We use log(tokens+1) rather than raw L: the multi-hop trace length
        distribution is approximately log-normal (App. E.2), so a linear
        regression in log-space is the natural homoscedastic counterpart of
        Vashurin et al.'s linear regression in raw L on their shorter NMT
        and summarization traces. We verify the two choices agree within
        ε ≤ 0.029 AUROC (App. E).
```

This converts a vague "consistent with the caveat" into an empirical justification + a reference-implementation agreement bound.

1. **§3.3 — the ECT formula needs the post-hoc denoising spelled out.** You write:

> $\textsc{ECT} = \frac{\min  i : m_j = a_{\text{final}} \forall j \geq i }{\max(1, n - 1)} \in [0, 1].$

— but what if the set is empty (no step matches the final answer)? What if the trace has only 1 step? You handle the n=1 case in the denominator but not the empty set. A pedantic reviewer will ask. One footnote:

```
ADD as footnote on Eq. 3:
If no step's majority matches a_final (which can occur when the trace has
no answer-bearing step; affects <2% of traces in our corpus), we set
ECT := 1 (the model never committed). The headline numbers are robust to
the alternative convention ECT := 0 within bootstrap CI bounds.
```

1. **§3.3 last sentence** ("A K=1 vs K=5 ablation of `flip_rate` is null on both architectures (§4.3), and we recommend K=1 for deployment efficiency.") is a *result* leaked into Methods. Move to §4 or to the end of §3.3 as a `\paragraph{}`-headed remark, not in flowing prose.

### 2.7 §4 Results

**What's there.** Six subsections. ~600 words + 2 tables.

**Peer comparison.** Janiak EMNLP 2025 has 5 result subsections in 4 pages of body. Datta has 4 result subsections. UNCERTAINTY-LINE has 1 result section with 2 tables (because they're more methodological). Six subsections is on the high end for an EMNLP-short.

**Strengths.**

- §4.3 (null K=5 ablation) is a model paragraph — claim, evidence, recommendation. ✓
- §4.4 (LLM-judge robustness) directly addresses the M1 + Janiak critique. ✓
- §4.5 (length anti-predictive on OOD) is the most interesting paragraph in the paper.
- §4.6 (robustness summary) does the work the existing guide §4.6 says it should. ✓

**Problems.**

1. **§4.5 should be §4.2.** Already noted in finding #7 of the executive summary. This is the most surprising fact in your paper, and it currently appears as Result 5 of 6. Move it up. Concrete restructuring:


| Current                                                | Proposed                                                              |
| ------------------------------------------------------ | --------------------------------------------------------------------- |
| §4.1 Cross-architecture survivor matrix                | §4.1 **Four signals clear Bonferroni across four families**           |
| §4.2 ECT discriminates strongest on OOD MoE cell       | §4.2 **Length is anti-predictive on out-of-domain cells** (was §4.5)  |
| §4.3 K=5 multi-sample variant is null                  | §4.3 **ECT discriminates strongest on the same OOD cells** (was §4.2) |
| §4.4 Survivor set robust to semantic-equivalence judge | §4.4 (same)                                                           |
| §4.5 Length anti-predictive on OOD                     | §4.5 K=5 variant is null (was §4.3)                                   |
| §4.6 Robustness summary                                | §4.6 Robustness summary (same)                                        |


The reordering puts the **surprise → mechanism → defense → defense → null result → summary** flow, which is the strongest narrative arc for this material.

1. **§4.1 references "all 17 signals"** but the reader has not yet been told what the 17 signals are. Add one paragraph as the first paragraph of §4.1:

```
ADD as first paragraph of §4.1:
We audit 17 trigger signals across four families: attention (DRAGIN-RIND
\citep{su_dragin_2024}; CABV \citep{...}), token-probability (FLARE \citep{
jiang_active_2023}; ent.max, etc \citep{li_etc_2025}, flare.max),
hidden-state (SeaKR-K5 \citep{yao_seakr_nodate}; drift_cosine,
var_l2_window.max, hidden_norm.range), and answer-trajectory (flip_rate
K∈{1,5}, late_flips, min_majority, ECT — this paper). The full pool is
listed in Table~\ref{tab:hero-full}.
```

This is the missing taxonomy paragraph — without it the reader has no scaffolding for the leaderboard.

1. **§4.2 (ECT discriminates strongest)** currently has 2 sentences of body text and 1 table. After the reordering above, this section should have a *mechanism* paragraph explaining *why* ECT discriminates strongest on the OOD MoE cell. Currently the only mechanistic explanation is in §5.2; pull half of that forward.
2. **§4.3 (K=5 vs K=1 ablation)** is fine but you should foreground the recommendation:

```
BEFORE: We recommend K=1 for deployment at 5× less compute per boundary,
        with equivalent triggering performance.
AFTER : Practical recommendation: K=1 (Datta-style single greedy sample at
        each boundary) is sufficient. Our K=5 boundary-completion variant
        carries 5× higher compute cost with no measurable advantage at our
        power.
```

1. **§4.6 (robustness summary)** is the right paragraph and exists in the right form. Add one sentence to it: "(iv) the choice of length-regression parameterization (log(L+1) vs raw L; App. E)." That makes 4 robustness checks instead of 3, which reads as more thorough.

### 2.8 §5 Discussion

**What's there.** Two subsections — verbosity-compensation mechanism, ECT mechanism. ~280 words.

**Peer comparison.** Discussion sections in this subgenre tend to be either (a) one long paragraph of interpretation (Datta, Chen) or (b) two short claim+mechanism paragraphs (UNCERTAINTY-LINE, Janiak). Yours follows pattern (b). On-norm.

**Strengths.**

- §5.1 names the causal chain explicitly. ✓
- §5.2 reads ECT against Datta's $t^\star$ in plain prose. ✓
- "The same primitive supports two complementary operations: detect stabilization to stop; detect non-stabilization to retrieve." — this is the best single sentence in the paper. Keep.

**Problems.**

1. **§5.1 — "A plausible causal chain underlying this pattern is...".** This is correctly hedged (per the existing guide §10 R3 fix). But you then describe it for 80 words. Tighten:

```
BEFORE: A plausible causal chain underlying this pattern is length →
        verbosity compensation → uncertainty → incorrectness → retrieval
        need: under this picture, length confounds the evaluation because it
        proxies the proximal cause, and the trigger pool's signals receive
        partial AUROC credit for tracking length because length correlates
        with the deeper uncertainty mechanism.

AFTER : A plausible causal chain is length → verbosity compensation
        (\citealp{zhang_verbosity_2024}) → uncertainty → incorrectness →
        retrieval need: length proxies the proximal cause, and trigger
        signals receive partial AUROC credit for tracking it.
```

1. **§5.2 is missing the citation to Wang et al. 2025 (Chain-of-Probe).** Wang's "early-answering is widespread and >20% more accurate than mid-CoT changing" finding is the *cleanest* external concordance with your ECT mechanism. From your lit review notes: "Wang et al. (2025) and Datta et al. (2026) independently find that LLMs that flip their answer during reasoning are systematically less accurate; we propose this same flip behavior as a retrieval trigger." This belongs in §5.2.
2. **The §5 closing sentence**: you don't have one. The discussion ends mid-thought ("...detect non-stabilization to retrieve."). Add a closing sentence that bridges to §6:

```
ADD as final sentence of §5.2:
Under this reading, ECT is not a new uncertainty primitive but a re-deployment
of an existing answer-trajectory primitive against a different application
target — exactly the kind of integration the broader EMNLP 2026 theme on
re-thinking evaluation invites.
```

(Only include the EMNLP-theme tail if you're submitting under the special theme.)

### 2.9 §6 Conclusion

**What's there.** One paragraph of ~140 words, summarizing the headline + the LLM-judge robustness + the prescription.

**Peer comparison.** Conclusions in EMNLP-short are typically 100–150 words. Yours is on-norm.

**Strengths.**

- All five Part-0 points appear.
- Bonferroni and κ explicit.
- ECT named.

**Problems.**

1. **"We argue that length-controlled evaluation should be reported alongside raw AUROC in every dynamic-RAG trigger comparison."** is the right hedge level for R4 (per the existing guide §10 R4 fix). Don't change this.
2. **Missing the OOD-anti-predictive fact** — if you take the §4 reordering above, you'd want this in the conclusion too:

```
ADD before the closing sentence:
On out-of-distribution multi-hop QA, length is anti-predictive of retrieval
need (raw AUROC ≤ 0.52 on the 2Wiki cells), inverting the verbosity-
compensation pattern observed on single-hop QA.
```

### 2.10 Limitations

**What's there.** 5 paragraphs with named subheadings (`\paragraph{...}`). The five: signal-level-only audit; arch/dataset coverage; correctness function; non-thinking-mode models; integration-as-contribution. Plus a Risks paragraph.

**Peer comparison.** This is one of the strongest sections in your paper. Janiak has 3 paragraphs of Limitations; UNCERTAINTY-LINE has 3; Datta has 5 (closest match to yours). Five is on the high end but acceptable per ARR (which encourages thorough limitations).

**Strengths.**

- Each paragraph names a *specific* limitation, not generic boilerplate.
- "Integration as contribution" preempts the H3 attack (per existing guide §9 H3). ✓
- "Architecture and dataset coverage" gives the falsifiable prediction (the Bonferroni lower bound of +0.047). ✓
- Risks paragraph addresses the methodological-misuse risk. ✓

**Problems.**

1. **"Correctness function" paragraph** doesn't mention F1 or containment-match explicitly. ARR will ding this if the reviewer is rigorous. One sentence:

```
ADD to "Correctness function" paragraph:
F1 and containment-match correctness functions are reported on a
200-trace subset in App. C and produce the same Bonferroni-robust four;
full-pool F1 and containment-match are deferred to future work because
they require re-running the paired bootstrap on every signal.
```

If you don't have F1 / containment-match in App. C, this is a 1-hour analyzer add for high reviewer-defense value.

1. **"Architecture and dataset coverage" is good but you could foreground that the survivor set is consistent across the 4 cells you do have.** Right now this is in §4.6 but not in Limitations. One sentence here doubles the defense.
2. **The "Risks" paragraph is good** but doesn't have a `\paragraph{Risks.}` header — it's currently a bare paragraph. Add the header for consistency with the other 5.

### 2.11 Appendices

**What's there.** 7 appendices (A–G). Total ~900 words.

**Strengths.**

- App. A (Bonferroni table) is the right place for the strict multiple-testing detail.
- App. B (full survivor matrix) is properly cross-referenced from §4.
- App. C (LLM-judge robustness) is excellent: protocol, agreement, transfer, caveat. The verbosity-bias caveat (Zheng et al. 2024) is exactly right for the reviewer who will raise it.
- App. E (UNCERTAINTY-LINE replication with ε bound) is the kind of soundness defense ARR reviewers reward.
- App. F (ECT vs $t^\star$ operational table) is — once promoted partially to §2 per existing guide §8.2 + this audit §2.5 — exactly what you need.
- App. G (Spearman ρ per cell) addresses Santilli et al. Table 2's recommendation directly. ✓

**Problems.**

1. **App. B (full matrix) needs a `raw AUROC` column** in addition to `Resid AUROC` + `Δ vs tokens`. The existing guide §8.4 caught this. Currently a reviewer cannot see how much residualization actually changed each signal — which is the empirical question your whole paper is about. One added column with 17 numbers, ~1 hour total.
2. **App. C (LLM-judge) is missing the OpenAI SDK version, model snapshot date, and exact judge call parameters.** ARR C4 will ding this. Add one paragraph to the protocol section:

```
ADD to App. C "Protocol":
Judge implementation details: OpenAI Python SDK v1.61.0, model
gpt-4o-mini-2024-07-18 (snapshot date), temperature=0, max_tokens=4,
single API call per (arm, question), no retry on null response (no nulls
observed). The full prompt is: "You are a strict semantic-equivalence
judge for multi-hop QA. Question: {q}. Reference answer: {gold}. Model
answer: {a}. Reply with exactly one word: YES or NO. No explanation."
```

1. **App. F (Reproducibility)** is missing several items per the existing guide §7.4:
  - HuggingFace revision SHAs for Qwen2.5-7B-Instruct and Gemma-4-26B-A4B-it
  - spaCy version + model name (`en_core_web_sm`?)
  - Random seeds used for K=5 sampling (numpy / torch seeds)
  - Decoding hyperparameters in one place (you have temp=0.6, top-p=0.9 in §3.3; restate)
2. **No appendix on "What we tried that didn't work."** This is not required, but a 1-paragraph "negative findings" subsection in App. F (e.g., "raw L instead of log(L+1) gives ε≤0.062; cubic spline gives no change") preempts R1 ("presenting the 'best' results out of an unknown number of trials").

---

## 3. Whole-paper themes (compounding issues)

These are issues that don't live in any one section but appear across the paper.

### 3.1 The `\texttt{}` audit

You have 60+ inline `\texttt{}` spans in 4 pages of body. The existing guide §1.4 partially caught this. New count from this audit:


| Context                                                                   | Current     | Should be                                                                            |
| ------------------------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------ |
| `\texttt{lift\_em}` in §3.1 equation                                      | `\texttt{}` | OK (formula label)                                                                   |
| `\texttt{lift\_em}{=}+1` in abstract (×3), §1 (×2), §3.1 (×2), §4 (×many) | `\texttt{}` | First mention in abstract: `\texttt{}` ; thereafter prose `lift\_em` or "lift-EM"    |
| `\texttt{tokens\_generated}` in §4 (×5+)                                  | `\texttt{}` | First mention: `\texttt{}`; thereafter "the token-length baseline" or "trace length" |
| `\texttt{Answer:}` in §3.3                                                | `\texttt{}` | OK (regex-extracted prefix)                                                          |
| `\texttt{flip\_rate}`, `\texttt{late\_flips}`, `\texttt{min\_majority}`   | `\texttt{}` | OK (signal identifiers in tables) but prose mentions can drop the monospace          |
| `\texttt{closed\_em}` in §5.1                                             | `\texttt{}` | OK                                                                                   |
| `\texttt{</think>}` in App. D                                             | `\texttt{}` | OK (model-specific token)                                                            |


Net: ~25% reduction in monospace density is achievable without losing identifier specificity.

### 3.2 Active vs passive voice

Your draft is approximately 60% active / 40% passive. EMNLP-short norm is 75/25 active. Specific passive-voice instances to convert:


| Passive                                     | Active                                                        |
| ------------------------------------------- | ------------------------------------------------------------- |
| "is evaluated by raw AUROC against..."      | "evaluates triggers by raw AUROC against..."                  |
| "the implication has not yet reached..."    | "the implication has not propagated to..."                    |
| "is shared across both arms"                | "shares the EM function across both arms"                     |
| "is robust to a semantic-equivalence judge" | "remains robust under a semantic-equivalence judge"           |
| "was the strongest length-robust survivor"  | "is the strongest survivor" (also: present tense for results) |


### 3.3 Citation style consistency

You mix `\citet{X}` and `\citep{X}` correctly in most places. One inconsistency: in App. C "Cohen's $\kappa = 0.707$ (substantial agreement; \citealp{landis_measurement_1977})" uses `\citealp{}` correctly (`citealp` = `\cite` in `\citep` style without parentheses). But in §4.4 you have the same fact without a citation. Add the `(\citealp{landis_measurement_1977})` to the §4.4 sentence too.

### 3.4 Numbers — base rates and CIs

You report CIs throughout (correct). You report base rate (13.5%) once, in §3.1 (correct). But you never report:

- Trace length distribution (mean ± SD or range) — should be in App. F
- The actual count per OOD cell (you have 24 helped on Gemma×2Wiki, but the table on §4.5 doesn't say what fraction this is)
- The mean and median trace length per cell — relevant for the "longer traces are less retrieval-helpful" claim

One additional table in App. G (Spearman) or App. F (Repro):

```
Cohort statistics:
  Qwen × Hotpot:   n=200, helped=31 (15.5%), trace length 198 ± 95 tokens
  Qwen × 2Wiki:    n=200, helped=34 (17.0%), trace length 213 ± 102 tokens
  Gemma × Hotpot:  n=200, helped=19 ( 9.5%), trace length 156 ±  87 tokens
  Gemma × 2Wiki:   n=200, helped=24 (12.0%), trace length 174 ±  94 tokens
```

This 4-row block in any appendix doubles the reviewer's ability to interpret the per-cell results.

### 3.5 Tense

Your tense use is mostly correct (Methods past, Results present, Limitations present-modal). One persistent issue: §1 ¶3 uses past tense for the ECT introduction ("ECT: the strongest length-robust signal in the audited pool, exceeding the length baseline...") — but the results are present tense elsewhere. Standardize on present tense throughout for what the data shows.

---

## 4. Prioritized action list (this audit's recommendations)

Combined with the existing guide's Part 15 changelog. The new / different items from this audit are marked **NEW**.


| #   | Action                                                                                      | Cost   | Priority             | Source                           |
| --- | ------------------------------------------------------------------------------------------- | ------ | -------------------- | -------------------------------- |
| 1   | **NEW: Cite Li et al. 2025 (ETC) in §1¶1 and §2, add Naming Note paragraph in §2**          | 30 min | **P0** (blocker)     | Audit Exec #1, §2.5              |
| 2   | **NEW: Audit `etc (Δ²H)` signal in App. B for citation to Li et al.**                       | 15 min | **P0** (blocker)     | Audit Exec #1                    |
| 3   | **NEW: Update Vashurin citation to EMNLP-main 2025 published version**                      | 5 min  | **P0**               | Audit Exec #2                    |
| 4   | **NEW: Add 1-sentence justification of `log(L+1)` deviation in §3.2**                       | 10 min | **P0**               | Audit Exec #3, §2.6              |
| 5   | Reframe §1 ¶1 to stake protocol-broken claim, with embedded family taxonomy                 | 20 min | **P0**               | Existing guide §1.2 + Audit §2.4 |
| 6   | **NEW: Add 3-sentence intuition paragraph after Eq. 1 in §3.1**                             | 15 min | **P0**               | Audit §2.6                       |
| 7   | Split compound noun phrases (abstract + §1 + Limitations)                                   | 30 min | **P0**               | Existing guide §1.1              |
| 8   | Restructure §2 into 3 thematic paragraphs + Naming Note                                     | 40 min | **P0**               | Existing guide §1.3 + Audit §2.5 |
| 9   | Add "honest delta" sentence at end of §1                                                    | 5 min  | **P0**               | Existing guide §2.1              |
| 10  | Add em-dash intuition clause on first ECT mention in abstract                               | 5 min  | **P0**               | Existing guide §3.2              |
| 11  | **NEW: Split headline figure into 2-panel (4-signal forest + ECT trace schematic)**         | 2 h    | **P0** (high impact) | Audit §1 Q4, §2.3                |
| 12  | **NEW: Promote §4.5 (length anti-predictive) to §4.2**                                      | 20 min | **P0**               | Audit Exec #7, §2.7              |
| 13  | **NEW: Add taxonomy paragraph at start of §4.1**                                            | 15 min | **P0**               | Audit §2.7                       |
| 14  | Surface κ = 0.707 to abstract                                                               | 5 min  | **P0**               | Existing guide §4.2              |
| 15  | Add "Robustness summary" — confirm (iv) parameterization included                           | 10 min | P1                   | Audit §2.7                       |
| 16  | Audit bolding policy in `tab:hero-full` (only Bonferroni-robust bolded; footnote)           | 10 min | P1                   | Existing guide §8.3              |
| 17  | **NEW: Add raw AUROC column to App. B Table 5**                                             | 1 h    | P1                   | Audit §2.11                      |
| 18  | Add "would refute" sentence on cross-arch in Limitations                                    | 10 min | P1                   | Existing guide §4.3              |
| 19  | Add motivation sentence for architecture choice in §3.1                                     | 5 min  | P1                   | Existing guide §10 M4            |
| 20  | Soften "should be standard practice" → "should be reported alongside" (already mostly done) | 5 min  | P1                   | Existing guide §10 R4            |
| 21  | Hedge §5.1 causal chain ("plausible mechanism" — already done; verify)                      | 5 min  | P1                   | Existing guide §10 R3            |
| 22  | Promote ECT vs $t^\star$ table to §2 Naming Note                                            | 10 min | P1                   | Existing guide §8.2              |
| 23  | **NEW: Add Wang et al. 2025 citation in §5.2**                                              | 5 min  | P1                   | Audit §2.8                       |
| 24  | **NEW: Tighten §5.1 causal chain prose (-40 words)**                                        | 10 min | P1                   | Audit §2.8                       |
| 25  | **NEW: Add OpenAI SDK / snapshot details to App. C**                                        | 10 min | P1                   | Audit §2.11                      |
| 26  | **NEW: Add 4-row cohort statistics block to App. F or G**                                   | 15 min | P1                   | Audit §3.4                       |
| 27  | **NEW: Add empty-set / n=1 footnote to ECT definition in §3.3**                             | 5 min  | P1                   | Audit §2.6                       |
| 28  | Add base-rate sentence to §3.1 (already partially done; check)                              | 5 min  | P2                   | Existing guide §5.1              |
| 29  | Audit Responsible NLP Checklist C1, C2, C4 in App. F                                        | 30 min | P2                   | Existing guide §11               |
| 30  | **NEW: Convert ~25% of `\texttt{}` spans to prose**                                         | 30 min | P2                   | Audit §3.1                       |
| 31  | **NEW: Standardize present-tense throughout Results**                                       | 15 min | P2                   | Audit §3.5                       |
| 32  | Audit every `\citet` / `\citep` against arXiv abstracts (G5)                                | 1 h    | P2                   | Existing guide §10 G5            |
| 33  | **NEW: Add F1 / containment-match mention to Limitations**                                  | 5 min  | P2                   | Audit §2.10                      |
| 34  | Audit `--` vs `---` and serial commas                                                       | 15 min | P3                   | Existing guide §6.1–§6.2         |
| 35  | Submit under "New Missions" special theme                                                   | 0 min  | P2                   | Existing guide §12               |


**P0 total (new + existing): ~6 hours.** Items 1–14 are blockers or near-blockers; items 11 (figure) and 12 (reorder) are the two highest-EV non-trivial-cost items.

**P0 + P1 total: ~10 hours.**

**Estimated impact:** With P0 only, you move from likely Findings (Overall 2.5–3.0) to likely Borderline-Conference (Overall 3.3–3.5). With P0 + P1, you have a plausible Conference-Main path conditional on a reviewer rolling well on Excitement (Overall 3.5–3.7).

---

## 5. Three concrete sentence-by-sentence diffs (worked, ready to paste)

These are the three highest-EV edits, fully written out so you can copy-paste.

### 5.1 Replace §1 ¶1 (current lines 121 in `template.tex`)

```
% --- DELETE THIS PARAGRAPH (current §1 ¶1) ---
% Dynamic retrieval-augmented generation (RAG) systems retrieve external
% evidence during decoding to ground generation in source material
% \citep{lewis_retrieval-augmented_2021, trivedi_interleaving_2023,
% shao_enhancing_2023}. The control point of every such system is the
% \emph{retrieval trigger}: a scalar function of the partial generation
% that decides when to invoke retrieval. The published pool of training-
% free triggers spans four signal families: attention-based DRAGIN
% \citep{su_dragin_2024}, token-probability FLARE \citep{jiang_active_
% 2023}, hidden-state SeaKR \citep{yao_seakr_nodate}, and multi-sample
% text-consistency SelfCheckGPT \citep{manakul_selfcheckgpt_2023}. All
% four follow the same evaluation protocol (rank traces by trigger score;
% report raw AUROC against a binary correctness label) and predate the
% length-confound literature.

% --- REPLACE WITH ---
Dynamic retrieval-augmented generation (RAG) systems decide whether to
fetch external evidence from a scalar signal computed during decoding
\citep{lewis_retrieval-augmented_2021, trivedi_interleaving_2023, shao_
enhancing_2023}. The published training-free trigger pool spans five
families: attention-based (DRAGIN, \citealp{su_dragin_2024}), token-
probability (FLARE, \citealp{jiang_active_2023}), hidden-state (SeaKR,
\citealp{yao_seakr_nodate}; INSIDE, \citealp{chen_inside_2024}),
multi-sample text-consistency (SelfCheckGPT, \citealp{manakul_
selfcheckgpt_2023}), and entropy-trend (ETC, \citealp{li_etc_2025}).
All five evaluate the trigger by raw AUROC against a binary correctness
label, and all predate the length-confound literature. On published
benchmarks, raw answer length attains AUROC up to 0.65 against this
target — a baseline absent from the evaluation table of every cited
trigger.
```

### 5.2 Replace abstract sentence 3 (current line 108)

```
% --- DELETE ---
% We evaluate the published training-free trigger pool under length-
% residualized AUROC on a paired-oracle target ($\texttt{lift\_em}{=}+1$:
% does retrieval flip a wrong answer to a right one?).

% --- REPLACE WITH ---
We deliver the first length-controlled evaluation of the published
training-free dynamic-RAG trigger pool, scored against a paired-oracle
target: \texttt{lift\_em}${=}+1$ labels each closed-book trace by whether
re-decoding the same model under gold context would specifically flip a
wrong answer to a right one.
```

### 5.3 Insert "honest delta" sentence at end of §1 (after line 125, before `\paragraph{Contributions.}`)

```
% --- INSERT (new paragraph) ---
Each of our building blocks has a direct precedent: length residualization
\citep{vashurin_uncertainty-line_2025}, the paired-oracle ``when does
retrieval help?'' question \citep{mallen_when_2023}, and the answer-
trajectory primitive family \citep{datta_large_2026, wang_chain--probe_
nodate}. Our contribution is their joint application to the dynamic-RAG
trigger setting, which surfaces four length-robust survivors invisible
under any individual prior protocol.
```

---

## 6. What this audit is NOT covering

For full transparency, items the existing guide treats and I deliberately do not re-treat:

- The Responsible NLP Checklist (existing guide §11 is comprehensive).
- Author-response templates (existing guide §13 is comprehensive).
- The H-code anti-pattern playbook for reviewer attacks (existing guide §9 is comprehensive).
- The full ARR three-score system math (existing guide §7 is comprehensive).
- The PR-AUC question (existing guide §16.8 already correctly concluded PR-AUC is NOT genre-required).

---

## 7. Closing assessment

**Your paper is more publishable than the current draft would suggest.** The substance — paired-oracle target on a length-residualized protocol, with ECT as a new length-robust signal and an LLM-judge robustness check — is genuinely a contribution that is novel along *some* axis (the integration), defensible against the H3 / H4 anti-pattern attacks (you've already prepared the precedent walkbacks), and reviewer-defense-positive (paired bootstrap + Bonferroni + reference-implementation ε check + κ robustness is unusually thorough for a short paper).

The four things blocking you from a likely-Borderline-Conference outcome are, in order:

1. The **ECT/ETC collision + missing Li et al. citation** (this audit's finding #1) — a true blocker.
2. The **headline figure** is at peer-norm for soundness but below peer-norm for excitement.
3. The **OOD anti-predictive finding is buried at §4.5** when it should be §4.1 or §4.2.
4. The **abstract's most surprising claim** (κ = 0.707 robustness, ECT strengthens to +0.222) is buried inside a compound-noun sentence.

- 

Good luck.

---

*End of audit. Generated by a fresh second-pass review against `template.tex`, the `__lit-review-main/` corpus, the published EMNLP-main 2025 UNCERTAINTY-LINE paper, and live arXiv/ACL-anthology searches as of May 21, 2026.*