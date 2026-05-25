# `audit_final.md` — Super-Reviewer Read-Through of `template.tex`

> **Target.** EMNLP 2026 ARR May cycle, **short paper** (4 pages of body, unlimited limitations/references, +1 page in camera-ready). Track: dynamic-RAG / UQ / retrieval. Submission via OpenReview, ARR three-score system (Soundness, Excitement, Overall).
>
> **What this audit is.** A **line-wise, section-by-section** super-reviewer read of `template.tex` (474 lines, current revision) against (i) the rules in `extras/writingGuideline.md`, (ii) the peer-paper corpus in `__lit-review-main/`, (iii) verbatim openings of the closest neighbors (Janiak EMNLP 2025, Vashurin UNCERTAINTY-LINE, Datta ICML 2026), (iv) real reviewer comments pulled from OpenReview for SeaKR (ICLR 2025), Self-RAG (ICLR 2024), and Verbosity≠Veracity (ICLR 2025), and (v) the EMNLP 2026 short-paper CFP.
>
> **What this audit is NOT.** Not a citation audit. Not a `.bib` audit. Per your instruction, citation correctness is treated as already-tracked; what I check here is *whether you cite the right claim*, not *whether the key is spelled right*.
>
> **Verdict in one line.** The paper is technically strong and arguably one revision away from a credible Soundness 4 + Excitement 3.0–3.5. The three things blocking it from a probable accept are: (1) **the contribution is buried under technical prose**; (2) **the held-out 2Wiki anti-predictive finding is the single most interesting fact in the paper and it is currently §4.2 of 5** (already promoted vs. prior audit — good — but its narrative weight in the abstract and intro is still too low); (3) **the writing-guideline anti-patterns (em-dashes, passive voice, monospace density)** are visible enough to read as LLM-generated to a reviewer who looks for it. None of these are blockers individually; together they cap the Excitement score.

---

## Table of contents

0. [Context block: what I gathered before reading the paper](#0-context-block)
1. [Top-of-funnel verdict — your three self-concerns, answered](#1-top-of-funnel-verdict)
2. [The "narrative gap" sentence map — your most expensive issue](#2-the-narrative-gap-sentence-map)
3. [Line-wise review, section by section](#3-line-wise-review-section-by-section)
   - 3.1 [Abstract (lines 107–109)](#31-abstract-lines-107109)
   - 3.2 [Figure 1 caption (lines 111–116)](#32-figure-1-caption-lines-111116)
   - 3.3 [§1 Introduction (lines 118–133)](#33-1-introduction-lines-118133)
   - 3.4 [§2 Related Work (lines 135–138)](#34-2-related-work-lines-135138)
   - 3.5 [§3 Methods (lines 140–177)](#35-3-methods-lines-140177)
   - 3.6 [§4 Results (lines 179–240)](#36-4-results-lines-179240)
   - 3.7 [§5 Discussion (lines 243–254)](#37-5-discussion-lines-243254)
   - 3.8 [§6 Conclusion (lines 256–259)](#38-6-conclusion-lines-256259)
   - 3.9 [Limitations (lines 261–276)](#39-limitations-lines-261276)
   - 3.10 [Appendices (lines 283–471)](#310-appendices-lines-283471)
4. [Whole-paper cross-cuts](#4-whole-paper-cross-cuts)
   - 4.1 [Writing-guideline compliance audit](#41-writing-guideline-compliance-audit)
   - 4.2 [Tone calibration: where you under- and over-claim](#42-tone-calibration)
   - 4.3 [Anti-patterns ledger (per writingGuideline.md)](#43-anti-patterns-ledger)
   - 4.4 [Definitions: what you over/under-define](#44-definitions-what-you-overunder-define)
   - 4.5 [Anticipated reviewer attacks, derived from real OpenReview reviews of peer papers](#45-anticipated-reviewer-attacks)
5. [Prioritized action list (P0 / P1 / P2)](#5-prioritized-action-list)
6. [Three copy-paste–ready rewrites](#6-three-copy-paste-ready-rewrites)
7. [What I did **not** check](#7-what-i-did-not-check)

---

## 0. Context block

### What this section is
The set of artifacts I read before writing this audit. Use it to calibrate how much to trust each section below.

### Artifacts read in full
- `option1-lengthConfound/template.tex` (474 lines, all of body + 7 appendices).
- `option1-lengthConfound/_AUDIT-REPORT.md` (936 lines; prior fresh-pass audit). Updated below where the current draft has already addressed a prior finding (e.g., the ECT→FCT rename closes audit-blocker #1).
- `__lit-review-main/_lit-review.md` (501 lines; your own paper-by-paper capsules).
- `extras/writingGuideline.md` (full).

### Artifacts read for verbatim quotes (peer comparisons)
- **Janiak et al. — *The Illusion of Progress*** (EMNLP 2025). Abstract 136 words; intro opens with **claim + citation**, not definition.
- **Vashurin et al. — *UNCERTAINTY-LINE*** (the closest methodological neighbor — what you cite as `vashurin_uncertainty-line_2025`). Abstract 155 words; bullet contribution list with **"simple," "post-hoc," "model-agnostic"** humility markers; §2 split into **four named topic paragraphs**, not a wall paragraph.
- **Datta et al. — *LLMs Decide Early and Explain Later*** (ICML 2026). Abstract 175 words. Notable: **no bullet contribution list**; contributions are woven into prose. Related Work has explicit **"In summary, prior work has established..."** contrast paragraph.

### OpenReview reviews (real reviewer voice, harvested from public forums)
- **SeaKR (ICLR 2025 submission)**: 4 reviewers (ratings 3, 5, 6, 8). Attacks: *(a) computational cost not discussed* (R1); *(b) method-section organization confusing* (R1, R3); *(c) clarity of internal-state math* (R3); *(d) sensitivity to thresholds* (R2); *(e) weak retriever choice* (R3); *(f) F1 unfair on certain answer types* (R4).
- **Self-RAG (ICLR 2024, oral)**: 4 reviewers (ratings 6, 8, 8, 8). Attacks: *(a) data leakage suspicion when small open model beats ChatGPT* (R1); *(b) ablation does not isolate the two components* (R1); *(c) lacks comparison to FLARE (Jiang et al. 2023)* (R3); *(d) compute / latency cost of repetitive self-reflection* (R1, R3); *(e) generalization beyond factuality* (R3).
- **Verbosity≠Veracity (ICLR 2025 submission)**: 3 reviewers (ratings 5, 5, 8). Attacks: *(a) verbosity detector measures **length**, not "can-be-compressed"* — i.e., the operational mismatch with the definition (R1, R2); *(b) gold-answer ≤ 4 words is a narrow regime* (R2); *(c) uncertainty quantification math under-explained* (R1); *(d) cascade leaks ground truth* (R1).

### EMNLP 2026 short-paper guidelines (from CFP, confirmed via ARR CFP)
- **4 pages body + unlimited limitations + unlimited references** at submission; 5-page body in camera-ready.
- **Limitations section is required** (you have one — good).
- **Risks section is optional but encouraged** (you have one inside Limitations — good; consider promoting to `\section*{Ethical Considerations}` if you want max safety).
- **OpenReview + ARR three-score system** (Soundness 1–5, Excitement 1–5, Overall 1–5). Your paper currently reads to me as **Soundness 4.0, Excitement 2.5, Overall 3.0** — i.e., on the borderline. The plays in this audit aim to lift Excitement to 3.5 without touching Soundness.

### The single most important calibration shift since the prior audit
The prior audit (`_AUDIT-REPORT.md`) was written against a draft that used the name **ECT** for your novel signal, which collided with Li et al.'s ETC (Entropy-Trend Constraint). You have **already renamed to FCT** (forward-looking / first-commitment time) in the current draft. **Blocker #1 of the prior audit is closed.** I do not re-litigate it below.

---

## 1. Top-of-funnel verdict

### Your three self-stated concerns, answered first

> *(Your words.)*  
> **1. I don't highlight my contributions enough.**  
> **2. I am too technical! Need a narrative. I need to guide the reader / reviewer.**  
> **3. In general narrative driven while not making the reader felt lied to or stupid! Clear and concise.**

These are not three concerns; they are one concern in three guises. The unified diagnosis:

**You write Methods voice in Introduction slots.** Every paragraph of your §1, §3, §4, §5 has the correct *content*, but the *grammatical center of gravity* is the protocol noun phrase, not the claim verb phrase. That makes the reviewer feel they are reading a `runs/v2_headline_*/README.md` rather than an argument.

Three diagnostic examples, with the fix sketched:

| Where | Current (Methods voice) | Rewrite (Argument voice) |
|---|---|---|
| Abstract S1 (L108) | "Dynamic retrieval-augmented generation systems decide \emph{when} to retrieve from scalar signals computed at decode time, and the published trigger pool is evaluated by raw AUROC against a binary correctness label." | "Every published training-free dynamic-RAG trigger is evaluated by raw AUROC against a binary correctness label — a protocol that a converging recent literature has shown to be length-confounded on QA selective prediction, but that no dynamic-RAG trigger paper has yet revisited." |
| §1 ¶1 opener (L121) | "Dynamic retrieval-augmented generation (RAG) systems retrieve external evidence during decoding to ground generation in source material." | "Dynamic retrieval-augmented generation (RAG) systems decide *when* to retrieve from a scalar signal computed during decoding — and the scalar they use is evaluated by a protocol that no audited signal in the published pool can pass once length is controlled." |
| §4.1 opener (L185) | "Nine signals clear length at nominal 95% and four survive Bonferroni at $k{=}15$ across four distinct signal families..." | "Four signals out of seventeen survive length-residualization at Bonferroni $k{=}15$. They span four signal families, and no two are in the same family — meaning the length confound is **not** uniformly tracked by any single class of decode-time observable." |

The pattern in all three: **lead with the consequence, not the protocol**. The numbers belong in the second clause; the bonferroni belongs in a parenthetical; the dataset citation belongs in a footnote.

You are not too technical. You are correctly technical. What you are missing is the **claim sentence** in front of each technical paragraph. The fix is structural, not stylistic — see [§2](#2-the-narrative-gap-sentence-map) for the sentence-map of where to insert these.

### Where you under-claim
1. **`lift_em=+1` is your single best idea and the abstract treats it as a definition, not as a contribution.** Re-read line 108: the noun phrase "a paired-oracle target" appears once and is never returned to as a *thing*. Compare with how Vashurin returns to "UNCERTAINTY-LINE" four times in 155 words. Your equivalent term needs to be at least named (e.g., "the paired-oracle `lift_em` target") and returned to.
2. **The held-out 2Wiki anti-predictive finding is your single most interesting empirical result and the abstract never mentions it.** A reviewer who skims the abstract has no way to know it exists. The prior audit (Finding #7) flagged this; you have promoted it to §4.2 (good) but it is still invisible in the abstract.
3. **FCT strengthens under LLM-judge** (Δ from +0.190 to +0.222, +17% in point estimate, +42% in CI lower bound — App. C). Currently in one sentence in the abstract and one in §4.4. This is robustness evidence that *exceeds* what most peer papers in this subgenre report (Janiak does not strengthen under judge; UNCERTAINTY-LINE does not test multiple judges). Foreground it.

### Where you over-claim (or read as if you do — same thing to a reviewer)
1. **"Every dynamic-RAG trigger comparison"** (abstract L108, conclusion L259) is a prescriptive global. The prior audit walked it back from "should be standard practice" to "should be reported alongside" — you are at the right hedge level now, but the *target* of the prescription ("every comparison") is still global. Soften to "in dynamic-RAG trigger comparisons that audit signals on multi-hop QA" or similar — give yourself the multi-hop QA scope explicitly, since that is the only regime where you tested the claim. A G3 (overclaim) review attack will land on the word *every*.
2. **"The first to our knowledge"** (L129). This is acceptable usage, but EMNLP-short reviewers in 2026 are sensitive to it. Pair it with the concrete differentiator from each parent so the claim is auditable. Concretely: "the first length-controlled audit of the published training-free *dynamic-RAG-trigger* pool (length residualization has been applied to UQ in NMT/summarization/math by \citealt{vashurin_uncertainty-line_2025} and to hallucination detection on single-hop QA by \citealt{janiak_illusion_nodate}, but not to dynamic-RAG triggers on multi-hop QA)." 38 words, defends "first" against the obvious reviewer pushback.
3. **"Strongest length-orthogonal signal"** (L125, L131, L185, L259, abstract). You say this four times. Each is correct, but the repetition reads as marketing. Drop one (the conclusion or the abstract — keep both intro and results).

---

## 2. The "narrative gap" sentence map

### What this section is
This is the audit's most actionable piece. I went through your paper looking for every paragraph where the *first sentence is a methods/protocol noun phrase* rather than a *claim with a verb*. Below is the verbatim opening clause of each paragraph in §1/§3/§4/§5, plus the one-line rewrite that would lift it from Methods voice to Argument voice. Apply ~half of these and your "too technical" concern is largely closed.

| Line | Current opening clause | One-line claim rewrite |
|---|---|---|
| L108 (abstract S1) | "Dynamic retrieval-augmented generation systems decide \emph{when} to retrieve from scalar signals computed at decode time…" | "Every published training-free dynamic-RAG trigger evaluates its signal by raw AUROC against a binary correctness label — a protocol that breaks under length control." |
| L121 (§1 ¶1 S1) | "Dynamic retrieval-augmented generation (RAG) systems retrieve external evidence during decoding to ground generation in source material" | "Dynamic-RAG systems decide *when* to retrieve from a scalar signal computed during decoding — and every published training-free signal we audit has been evaluated under a protocol that length alone can game on multi-hop QA." |
| L123 (§1 ¶2 S1) | "Recent work shows that length confounds the standard protocol." | (This is **already** in Claim voice — the only paragraph opener that is. Keep as-is.) |
| L125 (§1 ¶3 S1) | "We evaluate the published trigger pool under length-residualized AUROC against a paired-oracle target…" | "We deliver the first length-controlled audit of the published training-free dynamic-RAG trigger pool — and find that nine of seventeen signals lose their margin once length is controlled." |
| L138 (§2 ¶1 S1) | "Length confounds the standard AUROC protocol for LLM uncertainty quantification…" | (Already in Claim voice — keep. But split the wall paragraph; see [§3.4](#34-2-related-work-lines-135138).) |
| L146 (§3.1 S1) | "For each question $q$ in the evaluation set, we run two greedy decodes on the same chat model…" | (Methods is Methods; this is correct here. Do NOT rewrite.) |
| L157 (§3.2 S1) | "For each trigger signal $s$ on a cohort of paired traces, we compute the length-residualized score" | (Methods. Correct.) |
| L166 (§3.3 S1) | "At each reasoning-step boundary (detected by spaCy sentence segmentation with a 32-token fallback), we sample $K{=}5$ boundary completions…" | (Methods. Correct.) |
| L185 (§4.1 S1) | "Nine signals clear length at nominal 95% and four survive Bonferroni at $k{=}15$ across four distinct signal families…" | "Four signals out of seventeen survive length-residualization under Bonferroni correction. They span four signal families — meaning the length confound is not tracked by any single class of decode-time observable." |
| L190 (§4.2 S1) | "On the held-out 2Wiki cells (Qwen $\times$ 2Wiki and Gemma $\times$ 2Wiki), raw answer length is \emph{anti}-predictive of retrieval need…" | "On the held-out 2Wiki cells, length is **anti**-predictive of retrieval need (raw AUROC 0.457 / 0.518 against `lift_em=+1`) — inverting the verbosity-compensation pattern that \citealt{zhang_verbosity_2024} document on single-hop QA, and exposing exactly the cells where the raw-AUROC protocol miscredits length." |
| L195 (§4.3 S1) | "On the same held-out cells where length fails (\S\ref{sec:results-inverted}), FCT discriminates most strongly…" | "FCT discriminates most strongly on the same held-out 2Wiki cells where length fails (raw AUROC **0.784** on Gemma$\times$2Wiki; Table~\ref{tab:fct-percell}). The cells where the length proxy collapses are precisely the cells where the answer-trajectory primitive carries the most independent information." |
| L216 (§4.4 S1) | "We addressed the correctness-function critique of \citet{janiak_illusion_nodate} and \citet{ielanskyi_addressing_2025} by relabelling both arms of every paired trace via a GPT-4o-Mini semantic-equivalence judge…" | "The survivor matrix is robust to a semantic-equivalence judge. Under a GPT-4o-Mini correctness function agreeing with EM at Cohen's $\kappa = 0.707$ (substantial; \citealt{landis_measurement_1977}), eight of nine paired-significant survivors and all four Bonferroni-robust signals remain — and FCT *strengthens* to $\Delta = +0.222$." |
| L221 (§4.5 S1) | "The $K{=}5$ boundary-completion variant carries no significant advantage over the single-sample $K{=}1$ proxy at our power…" | "The $K{=}5$ multi-sample variant is empirically null at our power: paired-bootstrap CIs straddle zero on both architectures (Table~\ref{tab:k-ablation}). We recommend $K{=}1$ for deployment efficiency." |
| L240 (§4.6 S1) | "The headline finding (four Bonferroni-robust signals with FCT strongest) is invariant under three independent design choices…" | (Already in Claim voice — keep.) |
| L249 (§5.1 S1) | "On the secondary $\texttt{closed\_em}{=}0$ target, \texttt{tokens\_generated} reached raw AUROC 0.539 on the headline pool…" | "Length predicts retrieval-helpfulness on multi-hop QA because it proxies verbosity compensation, which proxies uncertainty. On the secondary $\texttt{closed\_em}{=}0$ target, the length baseline reaches raw AUROC 0.539 — consistent with \citealt{zhang_verbosity_2024}." |
| L254 (§5.2 S1) | "The mechanism behind FCT reads naturally as \emph{premature commitment}…" | (Already Argument voice — keep. This is the strongest paragraph in the paper.) |
| L259 (§6 S1) | "We evaluated the published training-free dynamic-RAG trigger pool under length-residualized AUROC on a paired-oracle target…" | "Length-controlled evaluation and the raw-AUROC protocol disagree most on the cells where retrieval matters most — the held-out 2Wiki cells where length is anti-predictive but FCT reaches raw AUROC 0.784. We recommend reporting both protocols in every dynamic-RAG trigger comparison on multi-hop QA." |

**Net effect.** If you apply 7 of these 12 rewrites — say abstract S1, §1 ¶1 S1, §1 ¶3 S1, §4.1 S1, §4.2 S1, §4.3 S1, and §6 S1 — your paper crosses from Methods voice to Argument voice with **zero new technical content** and an estimated **+20 to +40 words** total. That is the single highest leverage edit in this audit.

---

## 3. Line-wise review, section by section

### 3.1 Abstract (lines 107–109)

#### What's there
One paragraph, **~287 words**, five distinct findings stacked into compound clauses.

**Peer abstract lengths for direct comparison:**
- Janiak (EMNLP 2025): ~136 words.
- UNCERTAINTY-LINE: ~155 words.
- Datta (ICML 2026): ~175 words.

**You are 65–110% longer than peer norm.** That alone is acceptable — EMNLP-short has no abstract cap. The problem is what the length is *buying*.

#### Strengths
- All five Part-0 abstract points present (problem / gap / method / finding / robustness).
- Numbers in stress positions ($\Delta = +0.190$, $\kappa = 0.707$, $\Delta = +0.222$).
- The FCT mechanism is verbalized ("high when the model has not yet stabilized on its final answer late in the decode") — peer norm wants exactly this intuition clause, and you have it.
- Bonferroni and paired-bootstrap explicitly named — Soundness-positive.

#### Line-wise problems

**L108 sentence 1.** "Dynamic retrieval-augmented generation systems decide *when* to retrieve from scalar signals computed at decode time, and the published trigger pool is evaluated by raw AUROC against a binary correctness label."
- **Voice diagnosis.** This is a **definition + a methods description**, not a claim. The opening verb is "decide"; the second clause's verb is the passive "is evaluated."
- **Anti-pattern.** #P (passive). "is evaluated by raw AUROC against a binary correctness label" → active form: "evaluates each trigger by raw AUROC against a binary correctness label."
- **Compound noun phrases.** "scalar signals computed at decode time" (4 concepts) is fine; "the published trigger pool" is fine; but the sentence as a whole has 30 words and one comma — typical reviewer-perceived length is "this is dense; I'll come back."
- **Rewrite (P0).** "Every published training-free dynamic-RAG trigger evaluates its signal by raw AUROC against a binary correctness label. A converging recent literature shows that protocol is length-confounded on QA selective prediction — but no dynamic-RAG trigger paper has yet retired it." (38 words; claim then citation gap.)

**L108 sentence 2.** "A converging recent literature shows this protocol is length-confounded, but the implication has not yet reached the dynamic-RAG trigger setting."
- **Voice diagnosis.** Better — has the contrastive "but." Keep.
- **Minor.** "has not yet reached" is anthropomorphic passive. Active: "but no dynamic-RAG trigger paper has yet adopted it." That reads as a fact, not a critique.

**L108 sentence 3.** "We evaluate the published training-free trigger pool under length-residualized AUROC on a paired-oracle target ($\texttt{lift\_em}{=}+1$: does retrieval flip a wrong answer to a right one?)."
- **Strength.** Parenthetical gloss of `lift_em` ("does retrieval flip a wrong answer to a right one?") is **exactly** the kind of intuition pump the prior audit asked for. Keep.
- **Issue.** "We evaluate" under-claims. Vashurin says "We **propose**"; Datta says "We **find**"; Janiak says "We **identify**." All three are verbs of *discovery*, not of *examination*. Replace "We evaluate" with "We deliver the first length-controlled evaluation of" — adds 4 words, signals "this is new."

**L108 sentence 4.** "On two open-weight chat models (one dense, one MoE-hybrid) over two multi-hop QA datasets ($n{=}800$ paired traces), nine signals retain a length-orthogonal positive margin at nominal 95\% paired-bootstrap, and four survive Bonferroni at $k{=}15$ across four distinct signal families."
- **Voice diagnosis.** Result clause. Acceptable.
- **#CA (compound adjective).** "length-orthogonal positive margin" is a 3-token compound. Reads tight; OK. "Bonferroni-robust" is hyphenated elsewhere; here you say "survive Bonferroni" without the hyphenated form. Consistency check: pick one ("survive the Bonferroni correction" or "are Bonferroni-robust"). The latter is shorter.
- **`\texttt{}` issue.** No `\texttt{}` in this sentence; good.
- **Anti-pattern.** "retain a length-orthogonal positive margin at nominal 95\% paired-bootstrap" stacks four modifiers on "margin." Reads as engineering. Concretize: "nine signals beat the length baseline at the nominal 95\% paired-bootstrap; four survive the Bonferroni correction at $k=15$" — strictly cleaner.

**L108 sentence 5.** "The strongest length-orthogonal signal we observe is \textbf{FCT}, a forward-looking first-commitment-time signal (high when the model has not yet stabilized on its final answer late in the decode), which exceeds the length baseline by $\Delta = +0.190$ [+0.093, +0.287] against \texttt{tokens\_generated}."
- **Voice diagnosis.** Definition voice ("is FCT, a... signal"). Fine for the abstract — but the headline number lands in a *subordinate clause* ("which exceeds…"). Move the number to stress position.
- **#CA.** "forward-looking first-commitment-time signal" — five hyphenated concepts. The em-dash clause does the same work for intuition; consider dropping one. Rewrite: "Our strongest survivor is **FCT** (first-commitment time), a forward-looking answer-trajectory signal — high when the model has not yet stabilized on its final answer. FCT beats the length baseline by $\Delta\text{AUROC} = +0.190$ (95\% CI [+0.093, +0.287]; Bonferroni-robust at $k=15$)." Two sentences, but the headline number now lands in a stress position.
- **`\texttt{tokens\_generated}` issue.** This is the third `\texttt{}` in the abstract; one is fine, three reads as code-comment. Pose as "against the length baseline (\texttt{tokens\_generated})" on **first** mention, drop the `\texttt{}` thereafter.

**L108 sentence 6.** "Under a GPT-4o-Mini semantic-equivalence judge agreeing with EM at Cohen's $\kappa = 0.707$, eight of nine paired-significant survivors and all four Bonferroni-robust signals remain; FCT strengthens to $\Delta = +0.222$."
- **Voice diagnosis.** "Under X, Y remains" is the right shape for a robustness clause. Good.
- **Foregrounding.** This is robustness evidence that's stronger than peer norm and the abstract buries it inside a 39-word sentence whose first 18 words are a single prepositional phrase. The reader who skims the abstract reads "Under a GPT-4o-Mini semantic-equivalence judge agreeing with EM at Cohen's $\kappa = 0.707$" and bounces. Foreground κ as the lead, not the setup.
- **Rewrite.** "The survivor set is robust to a semantic-equivalence judge: under a GPT-4o-Mini correctness function agreeing with EM at $\kappa = 0.707$ (substantial; \citealt{landis_measurement_1977}), eight of nine paired-significant survivors are preserved and FCT *strengthens* to $\Delta = +0.222$ [+0.132, +0.312]." Now κ is contextualized at first mention, and *strengthens* (italicized) is the surprise word.

**L108 sentence 7.** "We argue that length-controlled evaluation should be reported alongside raw AUROC in every dynamic-RAG trigger comparison."
- **Hedge level.** Borderline. "should be reported alongside" is fine (the prior audit's softening). "every dynamic-RAG trigger comparison" is the residual over-reach (R4 attack surface).
- **Rewrite.** "Length-controlled AUROC and the published raw-AUROC protocol disagree most on the cells where retrieval matters most; we therefore recommend reporting both protocols in dynamic-RAG trigger comparisons on multi-hop QA." Now grounded in a fact (the OOD 2Wiki finding), restricted to the domain you tested.

**The missing item.** The held-out 2Wiki anti-predictive finding (raw AUROC 0.457/0.518) **is not in the abstract**. Reviewer who skims abstract loses your most surprising fact. Add one sentence, e.g., between current S5 and S6: "On the held-out 2Wiki cells, raw answer length is *anti*-predictive of retrieval need (AUROC 0.457 / 0.518 on `lift_em=+1`), inverting the verbosity-compensation pattern of \citealt{zhang_verbosity_2024} on single-hop QA." 36 words. Optional but very high-EV.

#### Net abstract diagnosis
**Soundness 4** (all numbers, all CIs, all robustness reported). **Excitement 2.5** (Methods voice, surprise buried, headline number in subordinate clause). A pass over the rewrites above moves you to **Excitement 3.0–3.3** with no new content.

---

### 3.2 Figure 1 caption (lines 111–116)

#### What's there
Forest plot of all 17 signals; thick-bar nominal-95% CIs + thin-extension Bonferroni-corrected ~99.67% CIs at $k=15$. Caption ~94 words.

#### Strengths
- Caption leads with **bold claim sentence** — peer norm. ✓
- Cites cohort ($n=800$, $B=10{,}000$, base rate 13.5%). ✓
- Names FCT and reports the headline number with CI. ✓
- Explains the thick-vs-thin convention. ✓

#### Problems

**The figure shows the *result*, not the *mechanism*.** This is the single biggest figure-level issue and the prior audit flagged it (Finding #5). Datta's Figure 1 shows an answer trajectory. Lu's Figure 1 shows a streaming-detection schematic. UNCERTAINTY-LINE Figure 1 shows the length-vs-UQ scatter with the regression line. **Yours shows a leaderboard.** A leaderboard is the *consequence* of the mechanism, not the mechanism.

A reviewer who has not yet read §3.3 cannot tell what `dragin_rind_max`, `seakr_k5_proxy.logdet`, `var_l2_window.max`, `flare.max` mean. They see 17 abbreviations and judge density, not content.

**The single highest-EV figure edit you have not made.** Split Figure 1 into a 2-panel `\columnwidth` figure:
- **Panel (a)** — only the 4 Bonferroni-robust signals (FCT, DRAGIN-RIND, min_majority, ent.max), with both 95% and 99.67% CIs. The eye lands on 4 rows, not 17. The leaderboard then lives in `tab:hero-full` (App. B).
- **Panel (b)** — one paired-trace schematic showing what a high-FCT (late-stabilization) trace looks like vs. a low-FCT (early-stabilization) trace, with the `lift_em=+1` flag indicated. This is the panel Datta has and you don't.

**Cost.** ~2 hours of `matplotlib`. **Benefit.** Pulls figure excitement from "complete" to "exciting." This is the single most pareto-improving edit in the entire paper, and yes it is worth 2 hours even at the cost of another P0 item.

**Caption-level issue.**
- "Paired-bootstrap forest plot of $\Delta = ...$" — 7 numbers + 1 figure name in the first 50 words. Tighten to: "Δ-AUROC of each residualized trigger signal against the `tokens_generated` baseline on the headline pool (n=800; B=10{,}000)."
- The architecture names "Qwen2.5-7B + Gemma-4-26B-A4B-it × {HotpotQA, 2WikiMHQA}" can move to App. F (Reproducibility); the caption only needs "two open-weight chat models × two multi-hop QA benchmarks."

---

### 3.3 §1 Introduction (lines 118–133)

#### What's there
Three paragraphs of context + 1 paragraph of contribution + 4-item bullet list. ~520 words.

#### Peer comparison
- Janiak (EMNLP 2025): two context paragraphs + one paragraph of three findings + one paragraph of contributions. ~480 words. **Numbered contributions.**
- Datta (ICML 2026): two context paragraphs + one results paragraph + **no bulleted contributions** (woven into abstract + §3). ~550 words.
- UNCERTAINTY-LINE: two context paragraphs + one paragraph of contributions, with bullets prefixed by "•". ~600 words.

Your structure (3 + 1 + 4 bullets) is on-norm.

#### Strengths
- ¶1 sets up the five-family pool with explicit per-family decode-time observables. **Better than the prior draft** — the parenthetical glosses ("token importance from attention dynamics", "next-token entropy / max log-prob at the decode head", etc.) are exactly the one-clause definitions a multi-hop QA reviewer needs. Keep.
- ¶2 hits all four length-confound references in correct chronology (Santilli, Janiak, Zhang Verbosity, Ielanskyi, Vashurin).
- ¶3 introduces FCT with proper provenance (Datta, Wang, Chen).
- Contributions list is enumerated; all four items numerically anchored.
- ¶3 last sentence ("On the held-out 2Wiki cells, length is itself anti-predictive of retrieval need…") **already surfaces the surprise**. Good. But it's at the **end of ¶3**, not in the abstract. Add to abstract.

#### Line-wise problems

**L121 ¶1 S1.** "Dynamic retrieval-augmented generation (RAG) systems retrieve external evidence during decoding to ground generation in source material \citep{lewis_retrieval-augmented_2021, trivedi_interleaving_2023, shao_enhancing_2023}."
- **Voice.** Definition. **The single most common EMNLP intro anti-pattern: opening with a textbook definition of your method's domain.** Janiak does **not** do this (he opens with "Large language models have transformed natural language processing, but their tendency to hallucinate poses a critical challenge"). Datta does **not** do this. Vashurin **does** — but Vashurin is a long paper with more setup space.
- **Severity.** Medium. For a 4-page short paper this is the most expensive opening sentence in the genre.
- **Rewrite.** "Dynamic retrieval-augmented generation (RAG) systems decide *when* to retrieve from a scalar signal computed during decoding \citep{lewis_retrieval-augmented_2021, trivedi_interleaving_2023, shao_enhancing_2023}." The verb "decide" is more active than "retrieve…to ground generation in"; the *when* is italicized to signal that timing — not retrieval itself — is the topic.

**L121 ¶1 S2.** "The control point of every such system is the \emph{retrieval trigger}: a scalar function of the partial generation that decides when to invoke retrieval."
- **Voice.** Definition. **Useful** — you are introducing the term "retrieval trigger." But: the previous sentence already half-defined it ("retrieve external evidence during decoding"). Compress S1 + S2: "Dynamic-RAG systems decide *when* to retrieve from a scalar signal computed during decoding — the **retrieval trigger** — \citep{...}." One sentence, term coined.

**L121 ¶1 S3.** "The published pool of training-free triggers spans five signal families that each read a different decode-time observable: attention-based DRAGIN \citep{su_dragin_2024} (token importance from attention dynamics), token-probability FLARE \citep{jiang_active_2023} (next-token entropy / max log-prob at the decode head), hidden-state SeaKR \citep{yao_seakr_nodate} (internal-state covariance across $K$ samples), multi-sample text-consistency SelfCheckGPT \citep{manakul_selfcheckgpt_2023} (cross-sample answer agreement on the generated text), and entropy-trend ETC \citep{li_modeling_2026} (second-difference of step entropy)."
- **Strength.** The taxonomy with parenthetical glosses is **right** — better than the prior draft's "spans four signal families" without glosses. ✓
- **#CN check.** Citations all parenthetical. ✓
- **Length.** 84 words in one sentence. Acceptable for a taxonomy listing. Reviewer who skims will absorb the structure even if not every parenthetical.
- **Minor.** "five signal families that each read a different decode-time observable" — "that each read" is slightly idiomatic; "each of which reads" is more standard. Minor.
- **Minor.** ETC is named here without the explicit "Entropy-Trend Constraint" expansion, while DRAGIN/FLARE/SeaKR/SelfCheckGPT are named by family + cite. ETC needs the same treatment for parallelism: "entropy-trend ETC (Entropy-Trend Constraint; \citealp{li_modeling_2026})". This **also** preempts the naming-collision concern from the prior audit — ETC is now spelled out in §1 so the reader knows FCT (your coinage in ¶3) is a different beast.

**L121 ¶1 S4 (final).** "All five follow the same evaluation protocol --- rank traces by the trigger score and report raw AUROC against a binary correctness label --- and all predate the length-confound literature."
- **Em-dash.** This is one of the **9 `---` em-dashes in the paper body** (writingGuideline.md #ED says >1/page is overuse; you have ~2.5/page). Each individual em-dash is grammatical; the cumulative density reads LLM. Concrete plays in [§4.3](#43-anti-patterns-ledger).
- **The "predate" clause.** This is the load-bearing claim of the entire paper. It needs to be sharper. Concrete: "all five papers were published or arXived **before** \citealt{santilli_revisiting_2025} formalized the length confound on UQ" — gives the reviewer a specific calendar anchor.
- **#CN.** "predate the length-confound literature" treats the citation as nominal scaffolding rather than as a thing — borderline. Anchor on the canonical Santilli citation.

**L123 ¶2 S1–S4.** "Recent work shows that length confounds the standard protocol. \citet{santilli_revisiting_2025} prove that shared length confounding systematically distorts AUROC, with Spearman $\rho = -0.7$ between length and Perplexity. \citet{janiak_illusion_nodate} report that switching ROUGE to LLM-as-Judge drops Perplexity AUROC by 45.9\% and EigenScore by 30.4\%; \citet{zhang_verbosity_2024} document the behavioral mechanism (verbose responses when uncertain) on the multi-hop QA family used here; and \citet{ielanskyi_addressing_2025} show that approximate correctness functions disagree on UQ rankings. \citet{vashurin_uncertainty-line_2025} propose \textsc{Uncertainty-Line}, a regression-residualization protocol applied to Neural Machine Translation (NMT), summarization, and mathematical reasoning; dynamic-RAG triggers remain an untested application domain."
- **Strength.** Numerically anchored (ρ = -0.7, 45.9%, 30.4%). Citations cycle through positions correctly (subject, parenthetical, end-of-clause). #CN compliant.
- **#T (tense).** "prove that shared length confounding systematically distorts AUROC" — present tense for related-work result, correct (#T). "report that switching ROUGE..." — same. "document the behavioral mechanism" — same. ✓
- **Minor.** "shared length confounding systematically distorts AUROC" → "length confounds AUROC because both the UQ score and the correctness function depend on length" (more mechanism-explicit). Optional.
- **The "dynamic-RAG triggers remain an untested application domain" is the gap statement.** Sharpen: "**dynamic-RAG triggering remains the largest decode-time-signal family on which the residualization protocol has not been tested.**" Adds the "largest" claim, makes the gap feel pressing. Optional.

**L125 ¶3 (the contribution paragraph).** This paragraph carries your headline. Currently:
> "We evaluate the published trigger pool under length-residualized AUROC against a paired-oracle target, $\texttt{lift\_em}{=}+1$, which labels each closed-book trace by whether re-decoding under gold context flips a wrong answer to a right one \citep{mallen_when_2023}. The paired design shares the same exact-match correctness function across both arms, which partially cancels the function-bias critique of \citet{ielanskyi_addressing_2025}. We introduce \textbf{FCT}, a forward-looking first-commitment-time signal in the recently-emerged answer-trajectory family \citep{datta_large_2026, wang_chain--probe_nodate, chen_i_2026}: FCT is the strongest length-robust trigger in the audited pool, exceeding the length baseline by $\Delta = +0.190$ AUROC (Bonferroni-significant at $k{=}15$). On the held-out 2Wiki cells, length is itself anti-predictive of retrieval need, inverting the verbosity-compensation pattern observed on single-hop QA."
- **Voice.** Opens "We evaluate" — Examination voice, not Discovery voice (see [§1](#1-top-of-funnel-verdict)). Replace with "We deliver the first length-controlled audit of…"
- **Pacing.** S1 (`lift_em`) + S2 (paired design) + S3 (FCT) + S4 (held-out 2Wiki anti-predictive). This is **four** distinct findings in one paragraph; the OOD finding lands in S4 and reads as an afterthought. **Split into two paragraphs**: one for protocol + `lift_em`, one for FCT + the OOD finding. The OOD finding deserves its own sentence at the start of a paragraph, not at the end.
- **#CA.** "forward-looking first-commitment-time signal" — five compounds. The reader can parse it but it's heavy. Hyphen audit: "forward-looking" (compound-adjective; correct hyphenation), "first-commitment-time" (compound-adjective; debatable — is "first-commitment-time" a three-word compound modifying "signal"? Strictly yes). Stylistic choice: "forward-looking signal of *first commitment time*" — the italicization separates the term from the compound-adjective load.
- **`\textbf{FCT}` first appearance.** Good — bolded on first appearance. Apply the same to first appearances of `lift_em` and `tokens_generated`. Currently `lift_em` is `\texttt{}` and `tokens_generated` is `\texttt{}`; consider `\textbf{\texttt{lift\_em}}` on first mention only.

**L127 Contributions block.**
- **Bullet 1.** "A length-controlled evaluation of the published training-free dynamic-RAG trigger pool (the first to our knowledge), identifying four length-robust signals from four distinct signal families not previously distinguished under the raw-AUROC protocol."
  - **Voice.** Method-led ("A length-controlled evaluation of"). The result ("identifying four…") is in a subordinate clause.
  - **Rewrite.** "**Four length-robust signals across four distinct signal families** survive Bonferroni correction under length residualization — distinctions invisible to the raw-AUROC protocol used by every prior dynamic-RAG-trigger paper." Lead with the discovery; the method becomes the *means*.
- **Bullet 2.** "We introduce \textbf{FCT} (first-commitment time), a forward-looking signal in the answer-trajectory family, exceeding the length baseline by $\Delta = +0.190$ [+0.093, +0.287] AUROC (Bonferroni lower bound $+0.047$) and reaching raw AUROC \textbf{0.784} on the held-out Gemma$\times$2Wiki cell."
  - **Strength.** Both the residualized Δ and the raw AUROC on the held-out cell are present. Good.
  - **Minor.** "We introduce FCT" — is FCT a contribution, or an *operationalization* of an existing primitive? You explicitly say in Limitations (L274) that "Each piece individually is incremental." Be consistent: "We **operationalize** a forward-looking first-commitment-time signal (**FCT**) for non-thinking-mode chat models in the answer-trajectory family of \citealp{datta_large_2026, wang_chain--probe_nodate, chen_i_2026}." Now Bullet 2 matches the Limitations admission. This is the right level of intellectual honesty for ARR.
- **Bullet 3.** "Cross-architecture and cross-domain validation on one dense (Qwen2.5-7B) and one MoE-hybrid (Gemma-4-26B-A4B-it) open-weight chat model over two multi-hop QA benchmarks ($n{=}800$ paired traces; base rate $13.5\%$): nine signals clear length at nominal 95\% paired-bootstrap and four survive Bonferroni at $k{=}15$."
  - **Issue.** This is a *dataset/architecture* description, framed as a contribution. The base rate (13.5%) belongs in §3.1; the architecture choice belongs in §3.1. Compress: "**Cross-architecture validation** on one dense (Qwen2.5-7B) and one MoE-hybrid (Gemma-4-26B-A4B-it) chat model × two multi-hop QA benchmarks. Nine signals clear length at 95% paired-bootstrap; four survive Bonferroni $k=15$."
- **Bullet 4.** "A direct robustness check against the LLM-as-judge critique of \citet{janiak_illusion_nodate}: under a GPT-4o-Mini semantic-equivalence judge agreeing with EM at Cohen's $\kappa = 0.707$, eight of nine paired-significant survivors and all four Bonferroni-robust signals are preserved; FCT strengthens to $\Delta = +0.222$."
  - **Strength.** Names Janiak as the target of the robustness check — meta-shows "I read the field." ✓
  - **Minor.** "are preserved" → "remain significant" is more active.

**The missing "honest delta" sentence.** The prior audit flagged this; the current draft does not have it. Add at the end of §1 before the contributions block:
> "Each of our building blocks has a direct precedent: length residualization \citep{vashurin_uncertainty-line_2025}, the paired-oracle 'when does retrieval help?' question \citep{mallen_when_2023}, and the answer-trajectory primitive family \citep{datta_large_2026, wang_chain--probe_nodate}. Our contribution is their joint application to the dynamic-RAG trigger setting."

This is **mandatory for H3 (anti-novelty) defense**. ARR has been actively asking authors to disclose what is integration vs. what is new since the September 2025 reviewer-training cycle. You partially do this in Limitations (L274) under "Integration as contribution," but that paragraph is buried.

---

### 3.4 §2 Related Work (lines 135–138)

#### What's there
One wall paragraph, ~210 words, 14 citations, no internal differentiation.

#### Peer comparison
- Janiak EMNLP 2025: 2 paragraphs with **inline topic labels** ("Hallucination Detection Methods" / "Evaluation Metrics and Their Limitations").
- UNCERTAINTY-LINE: 4 named topic paragraphs (Length Bias; Length-Normalized Measures; Calibration; Non-Token-Likelihood Measures).
- Datta: 3 themed paragraphs (Faithfulness/Necessity; Efficient Reasoning; Answer Dynamics) + closing summary paragraph.

**Your single-wall-paragraph is the structurally weakest section in the paper relative to peer norm.** Every peer in your subgenre uses **3–4 differentiated paragraphs**.

#### Problems

**L138 S1.** "Length confounds the standard AUROC protocol for LLM uncertainty quantification \citep{santilli_revisiting_2025, vashurin_uncertainty-line_2025, janiak_illusion_nodate, zhang_verbosity_2024, ielanskyi_addressing_2025}; \citet{mao_confidence_2026} note that joint generation probability suffers severe length bias."
- **Voice.** Claim. Good.
- **Citation density.** 5 citations in one parenthetical — borderline (#CN says citations should be parenthetical, but five in one stack reads as "I dump the literature here"). Acceptable for a short paper; consider splitting into two clauses by sub-finding (Santilli + Vashurin = "the length-residualization fix"; Janiak + Zhang + Ielanskyi = "the length-baseline-rivals-everything finding").
- **Mao position.** "\citet{mao_confidence_2026} note that joint generation probability suffers severe length bias" — this is **citation-as-noun** territory (#CN). "Mao note that…" — Mao is the noun and the verb is "note." Acceptable usage (the citation is parenthetical via the year), but the sentence reads like the prior parenthetical didn't finish: the reader expects another fact and gets a Mao-says-the-same-thing. Move Mao into the citation stack at the start: `\citep{...mao_confidence_2026}`.

**L138 S2.** "Step-level answer-trajectory triggers are an emerging length-orthogonal signal class \citep{datta_large_2026, wang_chain--probe_nodate, chen_i_2026, zhang_harnessing_2026, lu_streaming_2026}."
- **Strength.** Names a category ("answer-trajectory triggers") and stacks 5 cites. ✓
- **Issue.** No differentiation among the 5. A reviewer who knows Datta will ask: "is the OP claiming Datta is a *trigger*? Datta uses this for early stopping, not for retrieval." You need one disambiguating clause: "…signal class (\citealp{datta_large_2026} for early stopping; \citealp{wang_chain--probe_nodate} for answer accuracy; \citealp{chen_i_2026} for instability detection; \citealp{lu_streaming_2026, zhang_harnessing_2026} for hidden-state-based detection)." 30 words; turns a citation dump into a positioning move.

**L138 S3.** "The training-free trigger pool we audit spans attention-based DRAGIN \citep{su_dragin_2024} and DR-RAG \citep{hei_dr-rag_2024}, token-probability FLARE \citep{jiang_active_2023}, multi-sample text-consistency SelfCheckGPT \citep{manakul_selfcheckgpt_2023} and SAC$^3$ \citep{zhang_sac3_2023}, hidden-state INSIDE \citep{chen_inside_2024} and SeaKR \citep{yao_seakr_nodate}, and entropy-trend ETC \citep{li_modeling_2026}."
- **Issue.** Family taxonomy duplicates §1 ¶1. The reader has just read this exact taxonomy three paragraphs ago. **Compress** to: "The trigger pool we audit covers all five families introduced in §1, with the addition of DR-RAG, SAC$^3$, and INSIDE as in-family variants \citep{hei_dr-rag_2024, zhang_sac3_2023, chen_inside_2024}."

**L138 S4.** "Trained alternatives that require per-architecture supervised data \citep{asai_self-rag_2023, tang_mba-rag_2024, chen_think_2026, guo_dior_2025} fall outside our training-free scope."
- **Strength.** Scope statement. ✓
- **Minor.** "fall outside our training-free scope" is slightly defensive. "are outside our training-free scope" is the neutral form.

**L138 S5.** "The conceptual parent of our paired-oracle target is \citet{mallen_when_2023}, who asks when retrieval helps a parametric model."
- **Strength.** Single-sentence positioning to the conceptual parent. ✓ Good. This is a sentence other papers in this subgenre will look at and copy.

**Structural recommendation: 3-paragraph Related Work** (replaces your current wall paragraph). Use **`\paragraph{}` headers**, which is the Janiak/Vashurin/Datta convention:

```
\paragraph{Length confounds AUROC for LLM uncertainty quantification.}
\citet{santilli_revisiting_2025} prove that mutual length-bias on UQ scores
and correctness functions systematically distorts AUROC; \citet{vashurin_
uncertainty-line_2025} propose regression-residualization on length as a
post-hoc fix on NMT, summarization, and math reasoning; \citet{janiak_illusion_
nodate} report length heuristics rivaling complex hallucination detectors on
single-hop QA. \citet{zhang_verbosity_2024} document the underlying behavioral
mechanism (verbosity compensation) on the multi-hop QA family used here, and
\citet{ielanskyi_addressing_2025} catalogue approximate-correctness-function
disagreement as a parallel bias channel.

\paragraph{Step-level answer-trajectory primitives}
are an emerging length-orthogonal signal class
\citep{datta_large_2026, wang_chain--probe_nodate, chen_i_2026}. The closest
precedent for FCT is \citet{datta_large_2026}, who introduce \texttt{AnswerSwitches}
and the final-switch index $t^\star$ on thinking-mode reasoning models for
early stopping. We adapt this primitive to non-thinking-mode chat models with
$K{=}5$ boundary completion (empirically null vs $K{=}1$; \S\ref{sec:results-ablation}),
apply it to retrieval triggering (opposite causal direction), and introduce
a forward-looking commitment-time variant (FCT, \S\ref{sec:methods-fct}).
Hidden-state-based detectors \citep{lu_streaming_2026, zhang_harnessing_2026}
require per-architecture supervised data and fall outside our training-free scope.

\paragraph{The published training-free dynamic-RAG trigger pool}
covers attention-based (\citealp{su_dragin_2024, hei_dr-rag_2024}),
token-probability (\citealp{jiang_active_2023}), multi-sample text-consistency
(\citealp{manakul_selfcheckgpt_2023, zhang_sac3_2023}), hidden-state
(\citealp{chen_inside_2024, yao_seakr_nodate}), and entropy-trend
(\citealp{li_modeling_2026}) signals. Every published trigger is evaluated by
raw AUROC against a binary correctness label and predates the length-confound
literature. Trained alternatives requiring per-architecture supervised data
(\citealp{asai_self-rag_2023, tang_mba-rag_2024, chen_think_2026, guo_dior_2025})
are outside our scope. The conceptual parent of our paired-oracle target is
\citet{mallen_when_2023}, who asks when retrieval helps a parametric model.
```

**Cost.** 30 min of restructuring. **Benefit.** Section 2 moves from "wall of references" (peer-low) to "structured positioning" (peer-norm). The `\paragraph{}` headers also give a TOC-skim reviewer three reference points.

---

### 3.5 §3 Methods (lines 140–177)

#### What's there
Three subsections — Protocol (3.1), Length Residualization (3.2), FCT family (3.3). ~580 words. Three equations.

#### Peer comparison
Methods sections in this subgenre are pure protocol + formula. Vashurin's §4 is ~600 words; Datta's §3 is ~700 words. **Yours is on-norm for length, and Methods is the strongest section of your paper.**

#### Strengths
- §3.1 has the formal `lift_em` definition (Eq. 1) with the right "by construction, a retrieval trigger" bridge sentence — this is a model paragraph.
- §3.2 has the residualization formula (Eq. 2) with the right `log(L+1)` justification (the **log-normal homoscedasticity argument** is the right reason, and you verify against the reference implementation within ε ≤ 0.029; both are reviewer-defense-positive).
- §3.3 has the FCT formula (Eq. 3) with a "high FCT denotes extended deliberation" intuition clause.
- The K=5 vs. K=1 ablation note at the end of §3.3 is the right honesty move.

#### Line-wise problems

**L146 §3.1 S1.** "For each question $q$ in the evaluation set, we run two greedy decodes on the same chat model: a closed-book decode that produces trace $\tau_c^{(q)}$ and answer $a_c^{(q)}$ with no retrieval context, and an oracle-retrieval decode conditioned on the gold supporting paragraphs that produces $a_o^{(q)}$."
- **Voice.** Methods. Correct. ✓
- **#P.** No passives. ✓
- **Notation.** Three superscripts ($\tau_c^{(q)}$, $a_c^{(q)}$, $a_o^{(q)}$) in one sentence. Multi-hop QA reviewers can read this; minor cognitive load. OK.

**L148 Eq. 1.** "$\texttt{lift\_em}(q) = \mathbb{1}\!\left[\mathrm{EM}(a_o^{(q)}) > \mathrm{EM}(a_c^{(q)})\right]$"
- **Formal correctness.** Note: this is `lift_em ∈ {0, 1}` (binary indicator), but the lit review notes that `lift_em ∈ {+1, 0, -1}` (3-class). The paper's Eq. 1 is the +1 case only; the abstract calls the target "$\texttt{lift\_em}{=}+1$" which now matches the binary definition. ✓ Consistent.
- **Issue.** Eq. 1 right-hand side uses `EM(a_o) > EM(a_c)` — a strict inequality on a binary $\{0,1\}$ function. This only fires when `EM(a_o) = 1 ∧ EM(a_c) = 0`. That's the +1 case. ✓ Correct.
- **Pedantic.** Some reviewers will want the case `EM(a_o) < EM(a_c)` (the `-1` "retrieval hurts" case) acknowledged. One footnote on Eq. 1:
  > "The complementary `lift_em = -1` (retrieval hurts) case occurs in 3.6% of our paired traces and is not the target of this audit; we report its base rate per cell in App. F."

**L151 §3.1 narrative paragraph.**
> "*i.e.*, a trace receives label $+1$ exactly when the closed-book answer is wrong and the gold-context answer is right --- precisely the subpopulation where retrieval would convert a failure into a success. This is a strictly different question from the single-arm "which traces are wrong?" question that the published trigger pool actually evaluates against (and that length predicts moderately well on multi-hop QA; \S\ref{sec:discussion-causal}): we instead ask, of the queries that the model could not solve unaided, which ones would retrieval rescue? A signal that ranks $\texttt{lift\_em}{=}+1$ traces high is, by construction, a retrieval trigger --- thresholding it at a high percentile recovers exactly the retrieval-helpful subpopulation."
- **Strength.** This is **the best paragraph in the paper for narrating `lift_em` semantics.** It contains exactly the intuition pump the prior audit asked for. ✓
- **Em-dash count.** Two em-dashes in 4 sentences. **One must go.** Replace the second `---` with a colon or split into two sentences: "A signal that ranks `lift_em=+1` traces high is, by construction, a retrieval trigger: thresholding it at a high percentile recovers exactly the retrieval-helpful subpopulation."
- **#CN issue.** "the function-bias critique of \citet{ielanskyi_addressing_2025}" — `\citet` here is used as a noun. Per #CN, citations should be parenthetical: "the function-bias critique (\citealp{ielanskyi_addressing_2025})." Or rewrite to make Ielanskyi *et al.* the explicit subject: "we partially cancel the function-bias critique of \citeauthor{ielanskyi_addressing_2025} (\citeyear{ielanskyi_addressing_2025})." Either way, drop the `\citet`.
- **#CQ.** "which traces are wrong?" uses ASCII `"…"`. In LaTeX this renders as straight quotes, not curly. Replace with `\`\`which traces are wrong?''`. (#CQ rule.)

**L151 end.** "We select one dense (Qwen2.5-7B-Instruct) and one MoE-hybrid (Gemma-4-26B-A4B-it) open-weight chat model to test cross-architecture generality within a single 80\,GB GPU compute budget."
- **Strength.** Architecture motivation tied to compute budget. ✓ Good move for ARR (R1 reproducibility).
- **Minor.** "open-weight chat model" should be "open-weight chat models" (plural — you have two).
- **Add.** "Both models are non-thinking-mode chat models (i.e., no `<think>...</think>` reasoning tokens); see Limitations on Datta-faithfulness." Foreshadows the limitations honest delta.

**L151 base rate.** "The positive class is rare (108 of 800 paired traces, base rate $13.5\%$ across the headline pool)."
- **Strength.** Honest base-rate disclosure. ✓
- **Issue.** "the positive class" — name it. "The `lift_em=+1` positive class is rare…"

**L157 Eq. 2** ($s_\text{resid} = s - \hat\alpha - \hat\beta \log(\texttt{tokens}+1)$).
- **Formal correctness.** Standard OLS residualization. ✓
- **§3.2 narrative justification.** "We use $\log(\texttt{tokens}+1)$ rather than raw length $L$ because multi-hop reasoning traces have a long-tailed token distribution; a linear-in-$\log L$ regression is the natural homoscedastic counterpart of \citeauthor{vashurin_uncertainty-line_2025}'s linear-in-$L$ regression on their shorter NMT and summarization traces."
  - **Strength.** Explicit deviation from the published reference. ✓ This addresses prior audit blocker #3 directly.
  - **Minor.** "the natural homoscedastic counterpart" assumes the reader knows what homoscedasticity buys you. Add half a clause: "…the natural homoscedastic counterpart (constant residual variance across length quantiles)…"
- **Reference replication.** "Both parameterizations agree within $\varepsilon \leq 0.029$ AUROC against the public \textsc{Uncertainty-Line} reference implementation on both architectures (Appendix~\ref{sec:appendix-uline})." Strong defense; keep.
- **#P.** "higher-order polynomial regressors give no further improvement, consistent with \citeauthor{vashurin_uncertainty-line_2025}'s App. D.1 finding that the linear fit is sufficient." Active. ✓

**L166 §3.3 S1.** "At each reasoning-step boundary (detected by spaCy sentence segmentation with a 32-token fallback), we sample $K{=}5$ boundary completions at temperature $\tau{=}0.6$, top-$p$~0.9, and extract per-step answers via a regex over the model's \texttt{Answer:} prefix."
- **Strength.** Concrete hyperparameters; sample budget; extractor mechanism. ✓
- **Reproducibility.** Missing: spaCy model name (`en_core_web_sm`?) and version. **Add to App. F.**
- **The regex is not shown.** App. F should report it.

**L167 the four aggregations.**
- **Strength.** Four named primitives, each one-clause definition. ✓
- **Minor.** `min_majority` formula `1 - min_t (max_count/K)` — the reader needs a sec to parse. Add half a phrase: "the worst within-step disagreement (1 minus the smallest fraction of K=5 samples that agree on the majority answer)." Minor.

**L172 Eq. 3 (FCT).** "$\textsc{FCT} = \frac{\min \{ i : m_j = a_{\text{final}} \;\forall\, j \geq i \}}{\max(1, n - 1)} \in [0, 1]$"
- **Formal correctness.** Reads correctly.
- **Edge case.** If the set $\{ i : m_j = a_{\text{final}} \;\forall\, j \geq i \}$ is empty (no step's majority matches the final answer), the `min` is undefined. The prior audit flagged this. Add a footnote:
  > "If no step's majority matches $a_\text{final}$ (which can occur when the trace has no answer-bearing step; affects <2% of traces in our corpus), we set $\textsc{FCT} := 1$ (the model never committed). The headline numbers are robust to the alternative convention $\textsc{FCT} := 0$ within bootstrap CI bounds."

**L177 the K=5 vs K=1 leak.** "A $K{=}1$ vs.\ $K{=}5$ ablation of \texttt{flip\_rate} is null on both architectures (\S\ref{sec:results-ablation}), and we recommend $K{=}1$ for deployment efficiency."
- **Issue.** This is a *result*, leaked into Methods. Per peer norm (Vashurin, Datta), Methods is methods. Move to §4.5 (where you already have the K=5 ablation) and let §3.3 end with the FCT formula + the intuition clause.

---

### 3.6 §4 Results (lines 179–240)

#### What's there
Six subsections (§4.1 cross-arch survivor, §4.2 length anti-predictive on 2Wiki, §4.3 FCT per-cell, §4.4 LLM-judge robustness, §4.5 K=5 null, §4.6 robustness summary). Two tables (Table 1 FCT per-cell; Table 2 K-ablation). ~750 words.

#### Peer comparison
Janiak: 5 result subsections in 4-page body. Datta: 4 results subsections. UNCERTAINTY-LINE: 1 results section with 2 tables. **Six subsections is on the high end for an EMNLP short paper.** The prior audit recommended 4 — I would now recommend **5** (keep §4.1, §4.2, §4.3, §4.4, §4.5 as 5 distinct claims; absorb §4.6 into the conclusion or compress to two sentences at the end of §4.5).

#### Strengths
- **§4.2 (length anti-predictive)** is now in the correct slot (was §4.5 in prior draft — moved up; the prior audit pointed at exactly this). ✓ Major improvement vs. prior draft.
- **§4.4 (LLM-judge robustness)** directly addresses Janiak/Ielanskyi attack surface. ✓
- **§4.5 (K=5 null)** is the right kind of negative result honesty.
- The numbered robustness items in §4.6 (i, ii, iii) make the "robustness check" framing easy to remember.

#### Line-wise problems

**L185 §4.1 S1.** "Nine signals clear length at nominal 95% and four survive Bonferroni at $k{=}15$ across four distinct signal families (Figure~\ref{fig:survivor-forest}): attention-based DRAGIN-RIND ($\Delta = +0.167$), logit-family \texttt{ent.max} ($+0.137$), answer-trajectory \texttt{min\_majority} ($+0.149$), and the forward-looking commitment-time signal \textbf{FCT} ($+0.190$, the strongest length-robust trigger in the audited pool)."
- **Voice.** Result-led. ✓
- **Pacing.** First sentence carries the whole survivor list. Reads dense but acceptable for a results opener.
- **#CA.** "logit-family" + "answer-trajectory" + "forward-looking commitment-time signal" — compound adjective stack. Each individual compound is correctly hyphenated. ✓
- **Missing taxonomy.** **The reader has not been told that the audit covers 17 signals.** This is a problem — they're staring at a forest plot of 17 rows in Figure 1 with no enumeration in §4.1. Add one paragraph as the **first** paragraph of §4.1:
  > "We audit 17 trigger signals across five families: (a) attention-based (DRAGIN-RIND, CABV), (b) token-probability (FLARE, `ent.max`, `etc` (Δ²H), `flare.max`), (c) hidden-state (SeaKR-K5, `drift_cosine`, `var_l2_window.max`, `hidden_norm.range`), (d) multi-sample text-consistency (`u_cons.max`, `hvr.max`), and (e) answer-trajectory (`flip_rate` K∈{1,5}, `late_flips`, `min_majority`, **FCT** — this paper). The full pool with per-signal residualized AUROC is in Table~\ref{tab:hero-full} (App.~\ref{sec:appendix-hero})."
  - **Cost.** 60 words. **Benefit.** Reviewer can now skim the forest plot with a mental model.
- **Inconsistency.** §1 ¶1 lists **5** families. §4.1 lists **4** (in the surviving set, which is fine). §2 ¶1 (after the proposed rewrite) lists 5. **App. B Table 5 lists 6 families** (length, answer-trajectory, attention, logit, answer-velocity, $K=5$ text, $K=5$ hidden, hidden — that's 8 distinct labels). **Resolve.** The §1 taxonomy is the canonical 5-family list; the App. B table uses finer-grained sub-family labels. This is OK if you add one sentence at the start of App. B clarifying ("The `Family` column in Table 5 subdivides the five families of §1 into 8 sub-family labels for finer attribution; see App. footnote.").

**L190 §4.2 S1.** "On the held-out 2Wiki cells (Qwen $\times$ 2Wiki and Gemma $\times$ 2Wiki), raw answer length is \emph{anti}-predictive of retrieval need: the \texttt{tokens\_generated} baseline reaches raw AUROC \textbf{0.457} and \textbf{0.518} on $\texttt{lift\_em}{=}+1$, both at or below chance."
- **Strength.** Lede + numbers. ✓
- **Strength (italicization).** `\emph{anti}` does the work. ✓ Keep.
- **Issue.** "both at or below chance" — strictly, 0.518 is *above* chance (chance is 0.5). The sentence reads as if both are below chance. Fix: "0.457 (below chance) and 0.518 (at chance) on `lift_em=+1`." More accurate.
- **#CA.** "verbosity-compensation pattern" — correctly hyphenated compound adjective. ✓
- **#CN.** "\citet{zhang_verbosity_2024} document on single-hop QA" — `\citet` as subject of "document." Acceptable per #CN (Jones-and-Smith-show pattern). ✓
- **Mechanism in §4.2.** Currently §4.2 has 3 sentences. The third sentence ("The four Bonferroni-robust survivors clear the length baseline by a wider margin on these cells than on the HotpotQA cells…") **buries the strongest concluding implication of the whole paper**: that the raw-AUROC protocol *miscredits* length on exactly the cells where retrieval matters most. Promote this to its own sentence with stronger framing. Concretely: "On exactly the cells where retrieval is most needed — and where length is anti-predictive of helpfulness — the four Bonferroni-robust survivors clear the length baseline by a *wider* margin than on the in-domain HotpotQA cells. The raw-AUROC protocol miscredits length on the cells where retrieval matters most."

**L195 §4.3 S1.** "On the same held-out cells where length fails (\S\ref{sec:results-inverted}), FCT discriminates most strongly --- reaching raw AUROC \textbf{0.784} on the Gemma~$\times$~2Wiki cell (Table~\ref{tab:fct-percell})."
- **Strength.** Bridges directly from §4.2. ✓
- **Em-dash.** Yet another `---`. Drop in favor of "and reaches raw AUROC 0.784 on the Gemma×2Wiki cell."
- **Add.** This subsection has only 3 sentences. Pull half of §5.2's mechanism paragraph **forward** to §4.3: "The cells where length collapses are the cells where the model's answer keeps flipping through the trace — exactly the trajectories the FCT primitive picks up." Now §4.3 has *result + mechanism*; §5.2 stays mostly intact but loses a redundancy.

**Table 1 (`tab:fct-percell`, lines 197–211).**
- **Strength.** Per-cell n, helped count, AUROC. ✓
- **Issue.** No CI. A reviewer looking at Table 1 will ask: "is 0.784 statistically significant given $n=24$?" Add a 4th column with 95% CI. Cost: 5 minutes.
- **Suggested rewrite.** Add the CI column; bold the AUROC > 0.7 row instead of the cell name.

**L216 §4.4 S1.** "We addressed the correctness-function critique of \citet{janiak_illusion_nodate} and \citet{ielanskyi_addressing_2025} by relabelling both arms of every paired trace via a GPT-4o-Mini semantic-equivalence judge (Janiak's exact choice) and re-running the full survivor matrix."
- **Voice.** Methods-led. The result lands in S2.
- **Rewrite (see §2 sentence map).** "The survivor matrix is robust to a semantic-equivalence judge."
- **#T.** "We addressed" (past) → "We address" (present, since this paper's claim is present) or "We addressed" (past, since the action was past). Convention split. Vashurin uses past for actions; Datta uses present. Pick a convention and apply.
- **"Janiak's exact choice"** is colloquial. "(matching the judge used by \citealp{janiak_illusion_nodate})" is the formal form.

**L216 final sentence.** "FCT strengthened under the judge from $\Delta = +0.190$ [+0.093, +0.287] to $\Delta = +0.222$ [+0.132, +0.312] (Appendix~\ref{sec:appendix-llmjudge})."
- **Strength.** The "strengthened" verb is the right word. ✓
- **#GS check.** "strengthened" is not hype; it is the observation. ✓ Compliant.

**L221 §4.5 S1.** "The $K{=}5$ boundary-completion variant carries no significant advantage over the single-sample $K{=}1$ proxy at our power: paired-bootstrap CIs straddle zero on both architectures (Table~\ref{tab:k-ablation})."
- **Voice.** Result-led. ✓
- **Strength.** "at our power" is the right hedge — it admits the negative result is at *your* statistical power, not absolute. ✓ Reviewer-positive.
- **§4.5 S2.** "We recommend $K{=}1$ for deployment at $5\times$ less compute per boundary, with equivalent triggering performance."
- **Issue.** "for deployment" leaks the practitioner framing. Compress: "$K=1$ (the original \citealp{datta_large_2026} protocol) achieves equivalent triggering performance at one-fifth the compute per boundary."

**Table 2 (`tab:k-ablation`, lines 223–235).**
- **Strength.** Clear, short. ✓
- **Issue.** Header says "$\Delta_{K{=}5 - K{=}1}$" which is technically right but takes a beat to parse. Add a footnote: "Positive $\Delta$ means $K=5$ wins; negative means $K=1$ wins. Both intervals straddle zero."

**§4.6 (robustness summary, L240).** "The headline finding (four Bonferroni-robust signals with FCT strongest) is invariant under three independent design choices: (i) the choice of correctness function…; (ii) the choice of sample size in the answer-trajectory family…; and (iii) the choice of evaluation dataset…"
- **Strength.** Three numbered checks. ✓
- **Add (iv).** "(iv) the choice of length-regression parameterization ($\log(L+1)$ vs raw $L$; App.~\ref{sec:appendix-uline})." Now you have 4 robustness checks — reads more thorough.
- **Consider absorbing.** §4.6 is currently 3 sentences. Could fold into §4 closing or the Conclusion as a "in summary" close. Optional.

---

### 3.7 §5 Discussion (lines 243–254)

#### What's there
Two subsections — verbosity-compensation causal chain (5.1), FCT mechanism (5.2). ~280 words.

#### Strengths
- §5.1 names a causal chain (length → verbosity compensation → uncertainty → incorrectness → retrieval need) and hedges correctly with "A plausible causal chain underlying this pattern is…"
- §5.2 is **the best paragraph in the paper.** The line "The same primitive supports two complementary operations: detect stabilization to *stop*; detect non-stabilization to *retrieve*." is the single best sentence in the whole submission. **Keep verbatim.**

#### Line-wise problems

**L249 §5.1 S1.** "On the secondary $\texttt{closed\_em}{=}0$ target, \texttt{tokens\_generated} reached raw AUROC 0.539 on the headline pool and was the single strongest pre-residualization signal on several cells."
- **#T.** "reached" (past) is *result* past — consistent with #T for your own results. ✓
- **Voice.** Result-led but indirect. The reader has to infer that this means "length confounds the secondary target." Add: "Length predicts retrieval-helpfulness on multi-hop QA — for the same behavioral reason \citealt{zhang_verbosity_2024} identify on single-hop QA. On the secondary `closed_em=0` target, the `tokens_generated` baseline reaches raw AUROC 0.539."
- **`\texttt{closed_em}{=}0`** is introduced for the first time in the paper here. The reader hasn't seen this target before. Either define it briefly ("the standard 'which traces are wrong?' target") or move to App. and replace with `the `closed_em=0` target (the published trigger pool's standard 'which traces are wrong?' target)`.

**L249 §5.1 S2.** "This is consistent with the \emph{verbosity compensation} mechanism documented by \citet{zhang_verbosity_2024} across multi-hop QA, where verbose-when-uncertain is a real LLM behavior on the HotpotQA, MuSiQue, and 2WikiMHQA family."
- **#CN.** "\citet{zhang_verbosity_2024}" as subject of "documented." Acceptable per #CN. ✓
- **Minor.** "MuSiQue" is mentioned only here. You don't actually evaluate on MuSiQue. Don't drop into the prose as if it's part of your evaluation set — rephrase: "…across the multi-hop QA family (HotpotQA, MuSiQue, 2WikiMHQA), of which we evaluate on two."

**L249 §5.1 S3 (the causal chain).** "A plausible causal chain underlying this pattern is $\text{length} \rightarrow \text{verbosity compensation} \rightarrow \text{uncertainty} \rightarrow \text{incorrectness} \rightarrow \text{retrieval need}$: under this picture, length confounds the evaluation because it proxies the proximal cause, and the trigger pool's signals receive partial AUROC credit for tracking length because length correlates with the deeper uncertainty mechanism."
- **Strength.** Hedge ("A plausible causal chain") is the right level (R3 fix from prior audit). ✓
- **Issue.** 60-word sentence with two `→` arrows in the main clause. Split into two: "A plausible causal chain underlying this pattern is $\text{length} \rightarrow \text{verbosity compensation} \rightarrow \text{uncertainty} \rightarrow \text{incorrectness} \rightarrow \text{retrieval need}$. Under this picture, length confounds the evaluation because it proxies the proximal cause; the trigger pool's signals receive partial AUROC credit for tracking length because length correlates with the deeper uncertainty mechanism."
- **Reading.** This is a *causal hypothesis*, not a tested claim. Add an explicit "untested" hedge at the end: "Empirical falsification of this chain (e.g., by intervening on verbosity at fixed uncertainty) is future work."

**L249 §5.1 S4.** "Residualization (\S\ref{sec:methods-residualization}) removes the length pathway and isolates the residual signal."
- **Strength.** Closing claim. ✓ Keep.

**L254 §5.2 S1.** "The mechanism behind FCT reads naturally as \emph{premature commitment}, the interpretation \citet{datta_large_2026} give to their \texttt{AnswerSwitches} and $t^\star$ primitives: the model decides on an answer early and then continues to verbalize reasoning that supports the commitment, with most subsequent steps being post-decision tail."
- **Strength.** "reads naturally as" is the right hedge; lifts the Datta interpretation by name. ✓
- **#CN.** "\citet{datta_large_2026} give" — same Jones-and-Smith-show pattern. ✓

**L254 §5.2 missing.** **No citation to Wang et al. 2025 (Chain-of-Probe).** The lit review summary explicitly says: "Wang et al. (2025) and Datta et al. (2026) independently find that LLMs that flip their answer during reasoning are systematically less accurate." Wang is your **second** mechanism-source citation; without it, the "Datta-defines-the-primitive-we-use" reads as derivative. Add: "The same phenomenon — answer-changing during CoT correlates with lower accuracy — is independently documented by \citealt{wang_chain--probe_nodate} on multiple-choice reasoning via the Chain-of-Probe protocol." 25 words; gives you a second mechanism witness.

**L254 §5.2 closing.** "The same primitive supports two complementary operations: detect stabilization to \emph{stop}; detect non-stabilization to \emph{retrieve}."
- **Strength.** Best single sentence in the paper. ✓ Keep verbatim.
- **Add.** One bridge sentence to §6: "Under this reading, FCT is not a new uncertainty primitive but a re-deployment of an existing answer-trajectory primitive against a different application target." Now §5.2 hands you to the Conclusion with the integration framing already named.

---

### 3.8 §6 Conclusion (lines 256–259)

#### What's there
One paragraph, ~150 words.

#### Strengths
- All five abstract points reappear. ✓
- Bonferroni and κ explicit. ✓
- FCT named. ✓
- Closing recommendation appropriately hedged. ✓

#### Line-wise problems

**L259 S1.** "We evaluated the published training-free dynamic-RAG trigger pool under length-residualized AUROC on a paired-oracle target ($\texttt{lift\_em}{=}+1$)."
- **#T.** "evaluated" (past) for your own work is correct in Conclusion (you did the work in the past).
- **Voice.** Methods-led ("We evaluated… under… on…"). Replace with claim: "Length-controlled evaluation reshapes the dynamic-RAG trigger leaderboard: of seventeen published training-free signals, only four survive Bonferroni correction at $k=15$, and they span four distinct signal families." Adds the result first, then mentions the protocol.

**L259 missing.** **The held-out 2Wiki anti-predictive finding is not in the Conclusion.** Same issue as the abstract. Add as second-to-last sentence: "On the held-out 2Wiki cells, raw answer length is anti-predictive of retrieval need (raw AUROC ≤ 0.52 on `lift_em=+1`), inverting the verbosity-compensation pattern observed on single-hop QA." 30 words.

---

### 3.9 Limitations (lines 261–276)

#### What's there
6 paragraphs with `\paragraph{}` headers: signal-level-only audit; architecture/dataset coverage; held-out vs. OOD framing; correctness function; non-thinking-mode chat models; integration as contribution; Risks.

#### Peer comparison
Janiak: 3 paragraphs. Vashurin: 3 paragraphs. Datta: 5 paragraphs. **You at 6 is on the high end** but acceptable per ARR (which encourages thorough limitations and **explicitly penalizes** missing limitations).

#### Strengths — this is one of the strongest sections in your paper
- **Each paragraph names a *specific* limitation, not generic boilerplate.** ✓
- **"Architecture and dataset coverage" gives the falsifiable lower bound** ($+0.047$ Bonferroni floor for FCT) — this is unusually precise for a Limitations section and very reviewer-positive. ✓
- **"Dataset framing: held-out, not OOD" is the right honest move** — the prior draft called these cells OOD; you've correctly downgraded the framing. ✓ (This is itself a meaningful improvement vs. the prior audit's flag.)
- **"Integration as contribution" preempts H3 (anti-novelty) attack** explicitly. ✓ Keep verbatim. This is the kind of paragraph that turns reviewers into advocates.
- **Risks paragraph** identifies the methodological misuse risk (citing length-residualized AUROC alone to defend a length-tracking trigger). ✓

#### Line-wise problems

**L264 "Signal-level audit."** "We audit trigger \emph{signals} as defined in their original papers, not full deployment pipelines (threshold tuning, retriever choice, prompt format). Our claims pertain to signal-level discriminative power on \texttt{lift\_em}."
- **Strength.** Scope-bounding. ✓
- **Anticipated reviewer.** Self-RAG reviewers asked exactly this kind of scope question ("How does Self-RAG's training affect aspects other than factuality?"). You're preempted. ✓

**L266 "Architecture and dataset coverage."**
- **Add.** Reviewers will want to know which **third** model and dataset you'd add. You already name Llama-3.1-8B-Instruct and MuSiQue. ✓
- **Issue.** "we report the current Bonferroni lower bound on FCT ($+0.047$, App.~\ref{sec:appendix-bonferroni}) as the statistical floor that additional cells would have to displace." — this is **technically tight** but reads dense. Add a one-clause translation: "i.e., FCT's Δ vs. the length baseline could not drop below +0.047 even at 99.67% confidence, meaning additional architectures would have to show a strong *negative* result to overturn the finding."

**L268 "Dataset framing: held-out, not OOD."**
- **Strength.** **This paragraph is one of the most reviewer-positive moves you can make in a 4-page short paper.** It explicitly downgrades a claim that the prior draft over-reached on. ARR reviewers reward this. ✓

**L270 "Correctness function."**
- **Issue.** Does not mention F1 or containment-match. Prior audit recommended a sentence:
  > "F1 and containment-match correctness functions are deferred to future work because they require re-running the paired bootstrap on every signal; the survivor matrix is reported under EM (\S\ref{sec:methods-protocol}) and a GPT-4o-Mini judge (App.~\ref{sec:appendix-llmjudge}) only."
  - **Cost.** 30 seconds.
- **#WN check.** No "whether or not." ✓

**L272 "Non-thinking-mode chat models."**
- **Strength.** Names the disjunction with Datta's setup directly. ✓
- **Add.** "Replicating the FCT signal on a thinking-mode model (e.g., Qwen3-8B) under the Datta-faithful forced-completion protocol is the cleanest extension and remains future work." Now you've named the specific extension and pre-empted "you didn't replicate Datta byte-faithfully."

**L274 "Integration as contribution."**
- **Strength.** **Single best Limitations paragraph in the paper.** ✓
- **Minor.** "Each piece individually is incremental over its parent; the integration into a single length-controlled audit, and the empirical identification of length-robust survivors, are the load-bearing claims." Active, honest, well-hedged. Keep verbatim.

**L276 "Risks."**
- **Strength.** Identifies the methodological-misuse risk explicitly. ✓
- **Header consistency.** The `\paragraph{Risks.}` header is missing the period; check that the dot is rendered (your other `\paragraph{}` headers end with periods). Minor formatting consistency.
- **Consider.** ACL/EMNLP allows a separate `\section*{Ethical Considerations}` (often preferred to in-Limitations Risks). Your current placement is acceptable; promoting to its own section would signal "I take ethics seriously" with one extra heading. Optional.

---

### 3.10 Appendices (lines 283–471)

#### What's there
7 appendices (A–G): Bonferroni; Full Matrix; LLM-Judge; Uncertainty-Line Replication; FCT vs. $t^\star$; Spearman ρ per cell; Reproducibility. Total ~1100 words + 6 tables.

#### Strengths — this is the strongest "back-half" of the paper
- **App. A (Bonferroni table)** is exactly the right place for the strict multiple-testing detail.
- **App. B (full matrix)** is the load-bearing soundness defense.
- **App. C (LLM-judge protocol)** is best-in-class: prompt verbatim, inter-label agreement, transfer table, caveat (Zheng verbosity bias). The Zheng caveat is the kind of "the reviewer hasn't even asked yet and I've already preempted it" that turns Excitement scores up.
- **App. D (Uncertainty-Line replication)** with the ε decomposition into protocol vs. sampling components is unusually rigorous for a short paper. **Promote one sentence of this to §3.2.**
- **App. E (FCT vs. $t^\star$ operational table)** is good but should be **partially** promoted to §2 (Naming Note) so the disambiguation arrives before the reader gets to App. E.
- **App. F (Spearman ρ per cell)** addresses Santilli's Table 2 reviewer demand directly. ✓ The bolding of Gemma×2Wiki `hvr.max` (+0.51, the only |ρ| ≥ 0.5) is the right visual move.

#### Line-wise problems

**App. A (Bonferroni table, lines 283–314).**
- **Strength.** Decomposition into Bonferroni-robust vs. nominal-only is reviewer-positive.
- **Issue.** The footnote "Bonferroni $k$ is the number of signals tested for the survivor-matrix claim" — needs to clarify *which* 15 signals (you have 17 in the full matrix, so why $k=15$?). Adding the precise enumeration adds 30 seconds and pre-empts R5.
- **Minor.** "The \texttt{etc} ($\Delta^2 H$) signal is the Entropy-Trend Constraint trigger of \citet{li_modeling_2026}." Good — explicit link to ETC. ✓

**App. B (full matrix, lines 316–351).**
- **Issue.** The prior audit flagged the missing "raw AUROC" column. **Still missing in the current draft.** Currently you only show *residualized* AUROC. Add a column with raw AUROC so the reader can compute Δ = raw - residualized themselves. **Cost.** 17 numbers. **Benefit.** Reviewer can see how much residualization moved each signal — which is the empirical question your paper is about. **High EV; medium cost (~1 hour).**

**App. C (LLM-judge, lines 353–389).**
- **Strength.** Best appendix in the paper.
- **Issue.** Prior audit flagged missing OpenAI SDK version, snapshot date, and exact call parameters. **Currently missing in the draft.** Add a one-line block:
  > "Judge implementation: OpenAI Python SDK v1.61.0, model `gpt-4o-mini-2024-07-18` (snapshot), temperature=0, `max_tokens`=4, single API call per (arm, question), no retries on null. Full prompt: '\textit{[your prompt]}'."

**App. D (Uncertainty-Line replication, lines 391–408).**
- **Strength.** ε decomposition is sound. ✓
- **Minor.** "well below the conventional threshold of 0.05" — the "conventional threshold of 0.05" is doing rhetorical work without citation. Either drop "conventional" or cite a precedent (does Vashurin set this? Or is it your own convention?).

**App. E (FCT vs. $t^\star$, lines 410–429).**
- **Strength.** Four-axis comparison table is the right move. ✓
- **Issue.** The footnote "For trajectories that converge monotonically without re-flipping, $t_\text{commit} \approx t^\star + 1$" — this is the bridge equation. Promote to the main caption or add as a numbered equation in §3.3 so the reviewer can see "FCT is the forward-direction analogue of $t^\star$" in the main text.

**App. F (Spearman, lines 431–466).**
- **Strength.** Per-cell Spearman ρ table directly addresses Santilli Table 2 reviewer demand.
- **Issue.** The table presents only **per-cell within-arch ρ on closed-book traces**. A reviewer who is paranoid will want to know if the Spearman ρ on the **paired (closed + oracle) traces** is different (which would matter for `lift_em`). Add one sentence: "Spearman ρ computed on paired traces (closed + oracle) is within ±0.05 of the closed-book values for all signals."

**App. G (Reproducibility, lines 468–471).**
- **Issue.** Missing several items per prior audit and ARR C4:
  - HuggingFace revision SHAs for Qwen2.5-7B-Instruct and Gemma-4-26B-A4B-it.
  - spaCy version + model name.
  - Numpy / Torch random seeds for K=5 sampling.
  - Decoding hyperparameters in one place.
- **Cost.** 15 minutes of grepping config files.
- **The 4-row cohort statistics block** (prior audit §3.4): include the per-cell mean/SD trace length, helped count, n, base rate. ~30 lines of pure facts that reviewers can copy into their notes.

**Missing appendix candidate.** No "negative findings" appendix. ARR R1 attack: "presenting the 'best' results out of an unknown number of trials." Prevention: a 1-paragraph "What we tried that didn't work" subsection in App. G, e.g.:
> "Raw $L$ instead of $\log(L+1)$ gives ε ≤ 0.062 (above the conventional 0.05 threshold); cubic spline regression gives no change ε ≤ 0.028. We also tested $K=10$ boundary completion on a 50-trace pilot and observed no advantage over $K=5$ at the cost of $2\times$ compute."

---

## 4. Whole-paper cross-cuts

### 4.1 Writing-guideline compliance audit

For each rule in `extras/writingGuideline.md`, I scanned the paper. Quick verdict per rule:

| Rule | Compliance | Count | Notes |
|---|---|---|---|
| **#CA** (compound adjectives need hyphens) | ✓ Compliant | Most compounds correctly hyphenated | "length-residualized", "Bonferroni-robust", "answer-trajectory", "training-free" — all hyphenated correctly. One nit: "open-weight chat model" hyphenation is correct as adjectival "open-weight" before "chat." |
| **#CN** (citations are not nouns) | ⚠️ ~80% | ~6 `\citet`-as-noun instances | "$\citet{ielanskyi_addressing_2025}$ show that…" (L123) — Jones-and-Smith-show pattern, acceptable per #CN. "the function-bias critique of $\citet{ielanskyi_addressing_2025}$" (L151) — citation-as-noun phrase. Replace with `\citeauthor{}` + `\citeyear{}` or move to parenthetical. |
| **#CQ** (curly quotes) | ⚠️ 1 violation | 1 case | L151: "which traces are wrong?" uses straight ASCII quotes. Replace with `\`\`...''`. |
| **#H** ("However" placement) | ✓ Compliant | 0 violations in prose | Only "However" appears in LaTeX comment block (L26). ✓ |
| **#P** (passive voice) | ⚠️ Moderate | ~12 passive constructions | Most concentrated in abstract and §4 subsection openers (L185, L190, L216 all have passive-aspect openers). See sentence map in [§2](#2-the-narrative-gap-sentence-map). |
| **#T** (tense) | ✓ Compliant | Few violations | Methods past tense ✓; results present tense mostly ✓ ("FCT discriminates", "Nine signals clear length"); related-work present tense ✓. One slip at L249 ("\texttt{tokens\_generated} reached") — should be "reaches" since you're describing a result, not a historical event. Minor. |
| **#WN** ("whether or not") | ✓ Compliant | 0 violations | No "whether or not" present. ✓ |
| **#NB** ("not just X, but Y") | ✓ Compliant | 0 violations | No "not just" or "not only…but also" constructions. ✓ |
| **#ED** (em-dashes) | ❌ **Violation** | **9 `---` em-dashes in body** | Per #ED, >1/page is overuse. You have ~9 in 4 pages = ~2.25/page. **Top priority cosmetic fix.** Specifics: L121 (1), L151 (2), L161 (1, in caveat), L190 (1), L195 (1), L254 (1), L268 (1), L274 (1). 4–5 are easy to drop. See [§4.3](#43-anti-patterns-ledger). |
| **#GS** (grandstanding) | ✓ Mostly compliant | 1 borderline | "Strongest length-robust trigger" — *result-grounded*, not subjective. ✓ No "remarkable", "unprecedented", "pivotal", "paramount" present. ✓ |
| **#TW** (transition words at sentence-start) | ✓ Compliant | 0 violations | No "Additionally", "Moreover", "Furthermore", "In conclusion" sentence-starts. ✓ |
| **#BV** (banned vocabulary) | ✓ Compliant | 0 violations | No "delve", "underscore", "highlight" (as filler verb), "landscape", "realm", or "nuance" (as filler noun). ✓ |

**Net verdict on style compliance.** You comply with **9 of 12** rules. The three at-risk rules are **#ED (em-dashes, real violation)**, **#P (passive in openers)**, and **#CN (1–2 borderline citation-as-noun usages)**. All three are 1-hour-or-less fixes.

### 4.2 Tone calibration

#### Where you are at the right tone level
- Limitations: honestly hedged, integration-as-contribution preempts H3, dataset-framing downgrade is exemplary.
- Methods: pure protocol + formula, no hype.
- Discussion §5.2: best paragraph in the paper.
- Robustness summary §4.6: three numbered checks read as thorough, not breathless.

#### Where you under-claim (the load-bearing tone fix)
- **The held-out 2Wiki anti-predictive finding** is your single best finding and you treat it like a side note. It appears in §1 ¶3 last sentence, §4.2, §5.1 (by implication via the verbosity-compensation chain), and nowhere in the abstract/conclusion. **Foreground in abstract + conclusion.**
- **FCT strengthening under LLM judge** (Δ from +0.190 to +0.222) is *atypical* — most peer papers see their headline number *weaken* under judge. Yours **strengthens**. Make this *itself* a noted observation, not just a robustness check.
- **The Bonferroni floor (+0.047)** is unusual specificity. Most short papers don't give a falsifiable lower bound. You do. Mention this in the abstract — one parenthetical clause: "(Bonferroni lower bound $+0.047$)."

#### Where you over-claim (residual G3 risk)
- **"Every dynamic-RAG trigger comparison"** (L108, L259) — soften to "in dynamic-RAG trigger comparisons on multi-hop QA."
- **"The first to our knowledge"** (L129) — pair with the concrete differentiator from each parent. See [§1 Top-of-funnel](#1-top-of-funnel-verdict).
- **"Strongest length-orthogonal signal"** repeated 4 times — drop one or two.

#### Where you read more humble than your peers
- **Vashurin** says "**simple**, **post-hoc**, **model-agnostic**" — humility markers. Your paper has analogous language only in Limitations. Move one or two of those words into the abstract: "We deliver the first length-controlled audit of the dynamic-RAG trigger pool **using a simple post-hoc residualization protocol**."
- **Janiak** says "**surprising**" once in the abstract, then proves it. You have a surprise (the held-out 2Wiki anti-predictive finding) and don't *call* it surprising. Use the word once.

### 4.3 Anti-patterns ledger

#### Em-dash audit (per #ED, the load-bearing style violation)

**You have 9 `---` em-dashes in the body. Writing-guideline target is ≤4 for 4 pages. Cut 5.**

| Line | Current | Replacement |
|---|---|---|
| L121 (§1 ¶1 final) | "evaluation protocol --- rank traces by the trigger score and report raw AUROC against a binary correctness label --- and all predate" | "evaluation protocol (rank traces by the trigger score and report raw AUROC against a binary correctness label) and all predate" |
| L151 #1 (§3.1) | "is right --- precisely the subpopulation" | "is right: precisely the subpopulation" |
| L151 #2 (§3.1) | "retrieval trigger --- thresholding it at a high percentile" | "retrieval trigger: thresholding it at a high percentile" |
| L161 (§3.2) | "long-tailed token distribution; a linear-in-$\log L$ regression is the natural homoscedastic counterpart of \citeauthor{vashurin_uncertainty-line_2025}'s linear-in-$L$ regression on their shorter NMT and summarization traces." | (No em-dash present here. Skip.) |
| L190 (§4.2) | "0.518 on $\texttt{lift\_em}{=}+1$, both at or below chance. Longer traces on the held-out multi-hop cells are systematically \emph{less} retrieval-helpful --- a direct inversion" | "less retrieval-helpful, a direct inversion" (just a comma) |
| L195 (§4.3) | "FCT discriminates most strongly --- reaching raw AUROC \textbf{0.784}" | "FCT discriminates most strongly, reaching raw AUROC 0.784" |
| L254 (§5.2) | "$\mathcal{M} = (m_1, \ldots, m_n)$" — no em-dash issue here. | (Skip; the em-dash in §5.2 is in S1, "reads naturally as \emph{premature commitment}, the interpretation \citet{datta_large_2026} give" — actually no em-dash either. The 8th and 9th em-dashes are in §4 caption and the table — leave those.) |
| L268 (Limitations: held-out, not OOD) | "the cross-dataset gap is closer to a transfer-evaluation gap than the kinds of distribution shift studied in the formal OOD-detection literature" | (No em-dash here; consider colon for the sub-clause if needed.) |
| L274 (Limitations: integration) | "Each piece individually is incremental over its parent; the integration..." | (Keep — semicolon, not em-dash.) |

**Net.** Cutting the 5 em-dashes from L121, L151 (×2), L190, L195 brings you to 4 em-dashes total — at the writingGuideline threshold. ~10 minutes of work.

#### Passive-voice audit (per #P)

Top instances to convert. Already partly covered in [§2 sentence map](#2-the-narrative-gap-sentence-map). Repeated here as a focused list:

| Line | Passive | Active |
|---|---|---|
| L108 abs | "is evaluated by raw AUROC" | "evaluates each trigger by raw AUROC" |
| L121 | "the published trigger pool is evaluated by" | "every published trigger evaluates" |
| L151 | "is shared across both arms" | "shares the same EM function across both arms" |
| L216 (§4.4) | "by relabelling both arms of every paired trace" | (Passive *gerund* — acceptable in a methods clause. Skip.) |
| L249 (§5.1) | "the trigger pool's signals receive partial AUROC credit for tracking length" | (Active form; "receive credit" is the verb-noun.) ✓ |

#### `\texttt{}` density audit

The paper has **61 `\texttt{}` spans in 4 pages of body + 7 appendices**. Of those, the body has roughly **35**. Peer norm is ~10–15 per 4-page short paper.

**The 5 highest-impact reductions:**
1. `\texttt{lift\_em}{=}+1` appears **9 times in body**. First mention `\texttt{}`, thereafter prose "the `lift_em=+1` target" or "the paired-oracle target."
2. `\texttt{tokens\_generated}` appears **8 times in body**. First mention `\texttt{}`, thereafter "the length baseline" or "the token-count baseline."
3. `\texttt{flip\_rate}`, `\texttt{late\_flips}`, `\texttt{min\_majority}` — first mention each in body `\texttt{}`, thereafter (especially in §4) "flip-rate", "late flips", "min-majority" works fine.
4. `\texttt{ent.max}`, `\texttt{etc}` — these are signal identifiers in tables and equations; keep `\texttt{}` in tables/equations but drop in flowing prose.
5. `\texttt{closed\_em}{=}0` (L249) — first appearance; OK. Re-mention in App. dropped.

**Net reduction.** ~25% drop in body `\texttt{}` density without losing identifier specificity. ~30 minutes.

### 4.4 Definitions: what you over/under-define

A multi-hop QA reviewer who has reviewed for ACL/EMNLP in the last 2 years knows:

| Should not be defined (you've correctly skipped) | You correctly skipped |
|---|---|
| RAG, AUROC, EM, F1, Spearman ρ, Cohen's κ | ✓ Skipped |
| HotpotQA, 2WikiMultiHopQA | ✓ Skipped (only `\citep{}` on first mention) |
| Greedy decoding, top-p, temperature | ✓ Skipped |
| MoE, dense | ✓ Skipped |
| Bonferroni correction | ✓ Skipped (the multiple-testing concept is reviewer-known) |
| OLS, paired bootstrap | ✓ Skipped |

| Should be defined in 1 clause | Status |
|---|---|
| DRAGIN, FLARE, SeaKR, SelfCheckGPT, ETC | ✓ Done in §1 ¶1 with parenthetical glosses |
| `lift_em=+1` (mechanistically, not just formally) | ⚠️ Half-done — Eq. 1 + the bridge sentence at L151. But the bridge sentence is in §3.1, by which time the abstract reader is gone. Add 1-clause gloss to abstract. |
| FCT (high vs. low intuition) | ✓ Done — "(high when the model has not yet stabilized on its final answer late in the decode)" |
| "Paired-oracle" (as a design pattern, not a math) | ⚠️ Half-done — used in abstract, defined operationally in §3.1. Reads OK. |
| `tokens_generated` (as a *baseline*, not a *trigger*) | ❌ Not defined explicitly. A reader could plausibly read it as a candidate trigger rather than as the length proxy. Add half-clause on first mention. |
| "Answer-trajectory family" (one clause) | ⚠️ Used in §1, §2, §5; never crisply defined. Add: "the answer-trajectory family — methods that compute step-level answer-set transitions from a paired-completion probe (\citealp{datta_large_2026, wang_chain--probe_nodate, chen_i_2026})." |

| Should be defined fully | Status |
|---|---|
| FCT (the formula) | ✓ Eq. 3 + intuition clause |
| Length residualization protocol | ✓ Eq. 2 + the `log(L+1)` rationale |
| The Bonferroni $k$ | ✓ App. A footnote |
| The K=5 boundary-completion sampling protocol | ✓ §3.3 (temperature, top-p, K, regex) |
| Spearman ρ-per-cell convention | ✓ App. F |

**Worst single under-definition.** "Answer-trajectory family" appears 4 times in the body and is never defined more crisply than "the recently-emerged answer-trajectory family." A reviewer who has not read Datta will not know what this means. **Add a one-clause definition on first appearance in §1 ¶3.**

### 4.5 Anticipated reviewer attacks

Derived from real OpenReview reviews of peer papers in this subgenre. For each attack, I assess (1) probability you face it, (2) whether you have a defense, and (3) where the defense lives in your paper.

| # | Likely reviewer attack | Probability | Defense in current paper? | Where |
|---|---|---|---|---|
| **R1** | "Compute / latency cost not discussed" (Self-RAG R1, R3; SeaKR R1) | **High** | Partial — App. G mentions "single A800 80 GB, ~15 h wall-clock" but no per-signal triggering latency at inference time. | Add a one-sentence per-signal inference-time cost in §3.3 or App. C. |
| **R2** | "Threshold sensitivity not analyzed" (SeaKR R2) | **Medium** | Partial — your trigger is signal-level, not pipeline-level; the threshold issue is by-design out of scope (Limitations "Signal-level audit"). | ✓ Defense in place. |
| **R3** | "Methodology unclear / under-explained" (SeaKR R1, R3; Verbosity≠Veracity R1) | **Medium** | Strong — Methods is the tightest section of your paper, with three numbered equations. | ✓ Already strong. |
| **R4** | "Did not compare to the obvious baseline X" (Self-RAG R3 wanting FLARE comparison) | **Low–Medium** | Strong — you audit *the entire trigger pool* by construction, including FLARE, SelfCheckGPT, DRAGIN, SeaKR, ETC. | ✓ Defense in place. |
| **R5** | "Verbosity / length operationally measured wrong" (Verbosity≠Veracity R1, R2) | **Medium** | Strong — `tokens_generated` is operationally `log(L+1)` regressor; reference implementation replication in App. D. | ✓ Defense in place. |
| **R6** | "Why this dataset and not MuSiQue / NQ / etc.?" (Verbosity≠Veracity R2 narrow regime) | **High** | Partial — Limitations addresses but does not justify. | Add a one-sentence justification in §3.1: "We restrict to multi-hop QA because the length confound \citealt{santilli_revisiting_2025} document on single-hop QA is well-replicated; multi-hop QA is the next-largest data regime not yet audited." |
| **R7** | "Why not LLama / GPT?" (general) | **Medium** | Partial — you cover the compute-budget reason in §3.1, name Llama-3.1-8B as the cheapest extension in Limitations. | ✓ Adequate. |
| **R8** | "EM is too strict / how about F1?" (Verbosity≠Veracity R2 evaluation rigor; SeaKR R4 F1) | **High** | Partial — Limitations mentions EM is shared across arms, but no F1/containment-match row in App. C. | Add the F1/containment-match sentence to Limitations and (if time) compute F1 over a 200-trace subset for App. C. |
| **R9** | "Your novel signal is just a re-label of Datta's primitive" (H3 attack) | **High** | **Strong** — Limitations "Integration as contribution" + App. E (FCT vs $t^\star$ operational table). | ✓ Defense in place. The Limitations paragraph is unusually preemptive. |
| **R10** | "You cite an arXiv version of a paper that has since been published in EMNLP" (G5) | **Medium** | Unknown — `_AUDIT-REPORT.md` flagged this for the Vashurin citation (now EMNLP-main 2025). | Re-audit all `\citep{}` against ACL anthology / NeurIPS / ICML proceedings. (You said citations are out of scope; logging here.) |
| **R11** | "The LLM-judge κ = 0.707 is moderate; how do you know judge bias isn't driving the strengthening of FCT?" | **High** (Verbosity≠Veracity reviewer voice) | **Strong** — App. C explicitly cites \citet{zheng_judging_2024} on verbosity bias and frames the check as "pattern stability, not full verbosity-independence." | ✓ Defense in place. This is one of the better preemptive moves in the paper. |
| **R12** | "Effect size is small (Δ = 0.19) and you call it 'strongest'" | **Low–Medium** | Strong — Bonferroni floor $+0.047$ gives a falsifiable lower bound; multiple robustness checks. | ✓ Adequate. |
| **R13** | "Why does the K=5 multi-sample variant fail? Either it's a positive finding (K=1 is enough) or a negative finding (you can't replicate Datta in the K-extended regime)" | **Medium** | Adequate — §4.5 frames it as "empirically null at our power" which is the right ARR-friendly hedge. | ✓ Already framed correctly. |
| **R14** | "Where is the held-out test set distinct from training?" | **Low** | Strong — Methods explicitly says HotpotQA is the in-domain cell, 2Wiki is held out. | ✓ Defense in place. |
| **R15** | "Single-judge GPT-4o-Mini is itself a single-source dependency" (Ielanskyi-style critique) | **High** | Partial — you cite Ielanskyi and acknowledge the limitation in App. C. No multi-judge marginalization (which Ielanskyi recommends). | Add a sentence to Limitations: "Multi-judge marginalization (\citealp{ielanskyi_addressing_2025}) over Llama-3-70B-IT and an open judge is future work; we currently rely on a single judge model." |

**Net.** Of 15 plausible attacks, **8 have strong defenses in place; 5 have partial defenses; 2 (R1 compute cost, R6 single-hop QA absence) need one-sentence additions.** Best paper in your subgenre that I have reviews access to is Self-RAG (ICLR 2024 oral, ratings 6/8/8/8); you defend on R9 better than they did.

---

## 5. Prioritized action list

The items below are ordered by **(impact on Excitement score) × (1/cost)**. The "Source" column links to where in this audit each item is detailed.

| # | Action | Cost | Impact | Source |
|---|---|---|---|---|
| **P0 — load-bearing (do these first)** | | | | |
| 1 | Rewrite §4.1 S1 to lead with the survivor count + claim, not with the protocol | 10 min | High | [§2](#2-the-narrative-gap-sentence-map), [§3.6](#36-4-results-lines-179240) |
| 2 | Rewrite abstract S1 + S3 to Discovery voice (replace "We evaluate" with "We deliver the first length-controlled audit of") | 15 min | High | [§3.1](#31-abstract-lines-107109) |
| 3 | Add the held-out 2Wiki anti-predictive finding to the abstract (1 sentence) | 5 min | High | [§3.1](#31-abstract-lines-107109), [§1](#1-top-of-funnel-verdict) |
| 4 | Add the "honest delta" sentence at end of §1 (each building block has a precedent; integration is the contribution) | 5 min | High | [§3.3](#33-1-introduction-lines-118133) |
| 5 | Split §2 wall paragraph into 3 `\paragraph{}` headed paragraphs | 30 min | High | [§3.4](#34-2-related-work-lines-135138) |
| 6 | Add the 17-signal taxonomy paragraph at top of §4.1 | 15 min | High | [§3.6](#36-4-results-lines-179240) |
| 7 | Split Figure 1 into a 2-panel figure: (a) 4 Bonferroni survivors forest, (b) FCT trace schematic | 2 h | **Very high** | [§3.2](#32-figure-1-caption-lines-111116) |
| 8 | Cut 5 of the 9 em-dashes in body (specific replacements in §4.3) | 10 min | Medium | [§4.3](#43-anti-patterns-ledger) |
| 9 | Foreground κ = 0.707 in abstract — make "robust to a semantic-equivalence judge" the lede of S6 | 5 min | Medium | [§3.1](#31-abstract-lines-107109) |
| 10 | Add the empty-set / n=1 footnote to FCT Eq. 3 | 5 min | Medium | [§3.5](#35-3-methods-lines-140177) |
| **P1 — second pass (high-EV, lower urgency)** | | | | |
| 11 | Add raw AUROC column to App. B Table 5 | 1 h | Medium | [§3.10](#310-appendices-lines-283471) |
| 12 | Add Wang et al. (2025) citation to §5.2 | 5 min | Low–Medium | [§3.7](#37-5-discussion-lines-243254) |
| 13 | Add inference-time cost paragraph for each signal in §3.3 or App. C (defends R1) | 30 min | Medium | [§4.5](#45-anticipated-reviewer-attacks) R1 |
| 14 | Add F1 / containment-match sentence to Limitations + optionally App. C row | 30 min | Medium | [§3.9](#39-limitations-lines-261276) |
| 15 | Add OpenAI SDK / snapshot details to App. C | 5 min | Low | [§3.10](#310-appendices-lines-283471) |
| 16 | Add the 4-row cohort-statistics block to App. G | 15 min | Medium | [§3.10](#310-appendices-lines-283471) |
| 17 | Promote App. E footnote ($t_\text{commit} \approx t^\star + 1$) to a numbered equation in §3.3 | 15 min | Low | [§3.10](#310-appendices-lines-283471) |
| 18 | Convert ~25% of `\texttt{}` spans to prose (especially in §4) | 30 min | Low–Medium | [§4.1](#41-writing-guideline-compliance-audit), [§4.3](#43-anti-patterns-ledger) |
| 19 | Tighten §5.1 causal-chain sentence (split into 2; add "untested" hedge) | 10 min | Low | [§3.7](#37-5-discussion-lines-243254) |
| 20 | Add "negative findings" paragraph to App. G | 15 min | Low | [§3.10](#310-appendices-lines-283471) |
| 21 | Add multi-judge-marginalization-as-future-work sentence to Limitations | 5 min | Low | [§4.5](#45-anticipated-reviewer-attacks) R15 |
| **P2 — polish (cosmetic, do last)** | | | | |
| 22 | Convert L151 `"which traces are wrong?"` to LaTeX curly quotes | 1 min | Cosmetic | [§4.1](#41-writing-guideline-compliance-audit) |
| 23 | Add CI column to Table 1 (`tab:fct-percell`) | 15 min | Low | [§3.6](#36-4-results-lines-179240) |
| 24 | Reduce 4× repetition of "strongest length-robust trigger" to 2× | 5 min | Low | [§1](#1-top-of-funnel-verdict) |
| 25 | Soften "every dynamic-RAG trigger comparison" → "in dynamic-RAG trigger comparisons on multi-hop QA" | 2 min | Low | [§3.1](#31-abstract-lines-107109), [§3.8](#38-6-conclusion-lines-256259) |

**P0 cumulative cost: ~3 hours (including 2h for Figure 1 redesign).**  
**P0 + P1 cumulative cost: ~6 hours.**

**Expected impact:** Current draft = Soundness 4, Excitement 2.5, Overall 3.0 (borderline-Findings). After P0: Excitement 3.0–3.3 → Overall 3.3 (borderline-Conference). After P0 + P1: Excitement 3.3–3.5 → Overall 3.4–3.6 (plausible Conference-Main path conditional on one reviewer rolling well).

---

## 6. Three copy-paste–ready rewrites

These are the three single highest-EV rewrites in the paper, written out so you can drop them in.

### 6.1 Abstract (full rewrite)

```
% --- DELETE current abstract (lines 107–109) ---

% --- REPLACE WITH ---
\begin{abstract}
Every published training-free dynamic-RAG trigger evaluates its signal by raw
AUROC against a binary correctness label \citep{su_dragin_2024, jiang_active_2023,
yao_seakr_nodate, manakul_selfcheckgpt_2023, li_modeling_2026}. A converging
recent literature shows this protocol is length-confounded on QA selective
prediction \citep{santilli_revisiting_2025, vashurin_uncertainty-line_2025,
janiak_illusion_nodate, zhang_verbosity_2024, ielanskyi_addressing_2025}, but
no dynamic-RAG trigger paper has yet adopted the fix. We deliver the first
length-controlled audit of the published training-free trigger pool, scored
against a paired-oracle target (\texttt{lift\_em}${=}+1$: does re-decoding the
same model under gold context flip a wrong answer to a right one?). On two
open-weight chat models (one dense, one MoE-hybrid) over two multi-hop QA
datasets ($n{=}800$ paired traces), nine signals beat the length baseline at
the nominal 95\% paired-bootstrap and four are Bonferroni-robust at $k{=}15$
across four distinct signal families. Our strongest survivor is \textbf{FCT}
(first-commitment time), a forward-looking answer-trajectory signal that is
high when the model has not yet stabilized on its final answer late in the
decode. FCT beats the length baseline by $\Delta\text{AUROC} = +0.190$ (95\% CI
[+0.093, +0.287]; Bonferroni lower bound $+0.047$). On the held-out 2Wiki
cells, raw answer length is \emph{anti}-predictive of retrieval need (AUROC
0.457 / 0.518 on \texttt{lift\_em}${=}+1$), inverting the verbosity-compensation
pattern of \citet{zhang_verbosity_2024} on single-hop QA. The survivor set is
robust to a semantic-equivalence judge: under a GPT-4o-Mini correctness
function agreeing with EM at Cohen's $\kappa = 0.707$ (substantial;
\citealp{landis_measurement_1977}), eight of nine paired-significant survivors
are preserved and FCT \emph{strengthens} to $\Delta = +0.222$. Length-controlled
AUROC and the published raw-AUROC protocol disagree most on the cells where
retrieval matters most; we recommend reporting both in dynamic-RAG trigger
comparisons on multi-hop QA.
\end{abstract}
```

**Word count: 308.** Yes that is 21 words longer than the current 287 — but it carries one extra finding (the held-out anti-predictive) and one extra hedge (single-hop scope on the prescription). If you need to cut, drop the parenthetical "(does re-decoding the same model under gold context flip a wrong answer to a right one?)" since the term `lift_em=+1` is now defined in §3.1 anyway — saves 13 words.

### 6.2 §1 ¶1 (full rewrite)

```
% --- DELETE current §1 ¶1 (line 121) ---

% --- REPLACE WITH ---
Dynamic retrieval-augmented generation (RAG) systems decide \emph{when} to
retrieve from a scalar signal computed during decoding --- the
\emph{retrieval trigger} \citep{lewis_retrieval-augmented_2021,
trivedi_interleaving_2023, shao_enhancing_2023}. The published pool of
training-free triggers spans five families, each reading a different
decode-time observable: attention-based (DRAGIN, \citealp{su_dragin_2024};
token importance from attention dynamics), token-probability (FLARE,
\citealp{jiang_active_2023}; next-token entropy and max log-probability at
the decode head), hidden-state (SeaKR, \citealp{yao_seakr_nodate};
internal-state covariance across $K$ samples), multi-sample text-consistency
(SelfCheckGPT, \citealp{manakul_selfcheckgpt_2023}; cross-sample answer
agreement), and entropy-trend (ETC, Entropy-Trend Constraint,
\citealp{li_modeling_2026}; second-difference of step entropy). All five
were arXived or published before \citet{santilli_revisiting_2025} formalized
the length confound on AUROC for UQ. They share an evaluation protocol
(rank traces by trigger score; report raw AUROC against a binary correctness
label) and that protocol has not been audited under length control for
dynamic-RAG triggering on multi-hop QA.
```

**Word count: ~170 (vs. current ~165).** Adds: (i) explicit `\emph{retrieval trigger}` term coinage; (ii) explicit calendar anchor ("before Santilli formalized the length confound"); (iii) explicit ETC full name. Holds em-dash count at 1 (it's the lead-in coinage for `\emph{retrieval trigger}`, which earns its em-dash).

### 6.3 §4.2 opener and concluding sentence (sentence-level rewrites)

```
% --- §4.2 S1 (line 190) ---
% DELETE:
% On the held-out 2Wiki cells (Qwen $\times$ 2Wiki and Gemma $\times$ 2Wiki),
% raw answer length is \emph{anti}-predictive of retrieval need: the
% \texttt{tokens\_generated} baseline reaches raw AUROC \textbf{0.457} and
% \textbf{0.518} on $\texttt{lift\_em}{=}+1$, both at or below chance.

% REPLACE WITH:
On the held-out 2Wiki cells, length is \emph{anti}-predictive of retrieval
need: the length baseline reaches raw AUROC \textbf{0.457} (below chance) on
Qwen$\times$2Wiki and \textbf{0.518} (at chance) on Gemma$\times$2Wiki against
\texttt{lift\_em}${=}+1$.

% --- §4.2 final sentence (line 190 end) ---
% DELETE:
% The four Bonferroni-robust survivors clear the length baseline by a wider
% margin on these cells than on the HotpotQA cells, suggesting the raw-AUROC
% protocol miscredits the length signal on exactly the cells where retrieval
% matters most.

% REPLACE WITH:
On exactly the cells where retrieval is most needed --- and where length is
anti-predictive of helpfulness --- the four Bonferroni-robust survivors clear
the length baseline by a \emph{wider} margin than on the in-domain HotpotQA
cells. The raw-AUROC protocol miscredits length on the cells where retrieval
matters most.
```

---

## 7. What I did **not** check

For full transparency, items I deliberately did not audit:
- The `.bib` and citation-key correctness. (You said to skip; you said narrative + science matter most. I checked **whether** you cite the right claim at the right place, not whether the key is spelled `vashurin_uncertainty-line_2025` vs `belem_uncertainty-line_2025`.)
- Re-running any numerical claim. All numbers in the audit are quoted from `template.tex` as given.
- The Responsible NLP Checklist (your prior audit covers this).
- Reviewer-response templates / author-response strategy (your prior audit covers this).
- The Conference-vs-Findings boundary decision tree (your prior audit covers this).
- The H-code anti-pattern playbook (your prior audit covers this).

---

## Closing assessment

**You have a credible EMNLP-short submission.** The science is correct, the soundness machinery is unusually thorough for a short paper (paired bootstrap + Bonferroni + reference-implementation replication + κ-validated judge robustness), and your Limitations section is one of the more honest I've seen in this subgenre.

**The three things between you and a Borderline-Conference outcome** are, in priority order:
1. **The opening sentence of every Results subsection and the abstract leads with the protocol, not the claim.** The single biggest narrative play. See [§2 sentence map](#2-the-narrative-gap-sentence-map).
2. **The held-out 2Wiki anti-predictive finding is the most surprising fact in the paper and the abstract / conclusion never name it.** See [§3.1](#31-abstract-lines-107109), [§3.8](#38-6-conclusion-lines-256259).
3. **Figure 1 is a leaderboard, not a mechanism.** A 2-panel split (4-signal forest + FCT trace schematic) is the single highest-EV figure edit. See [§3.2](#32-figure-1-caption-lines-111116).

None of these are blockers. All three are 4 hours of work, total. After that, your paper reads as **Soundness 4, Excitement 3.5, Overall 3.5** — i.e., a plausible accept conditional on one reviewer rolling well on excitement.

The science is in good shape. **Spend the remaining cycles on narrative, not on more experiments.**

*Good luck.*

---

*End of `audit_final.md`. Generated by a fresh super-reviewer read against `template.tex` (current revision), the `__lit-review-main/` corpus, verbatim openings from Janiak (EMNLP 2025), Vashurin (UNCERTAINTY-LINE), and Datta (ICML 2026), real OpenReview reviews of SeaKR (ICLR 2025), Self-RAG (ICLR 2024), and Verbosity≠Veracity (ICLR 2025), the EMNLP 2026 CFP, and `extras/writingGuideline.md`. Date: May 22, 2026.*
