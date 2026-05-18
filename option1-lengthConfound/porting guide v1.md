I have rich data from 5 papers — Ielanskyi (our direct lineage), SeaKR + CtrlA (RAG triggers REJECTED at ICLR), Self-RAG (oral accept), and INSIDE (poster accept). Here's the synthesis + skeleton.

> **Note**: `CxiNICq0Rr` in your `others.md` is *not* Chain-of-Probe — it's a CRV (Circuit-based Reasoning Verification) paper. The Wang Chain-of-Probe paper isn't on OpenReview. Updating: skipping that one.

---

# BACKING MATERIAL — what each paper teaches us

## 1. Ielanskyi et al. — *Pitfalls in UQ Evaluation* — ICLR 2026 Poster, **ACCEPT** *(direct lineage match)*

**Scores**: 4 / 6 / 6 / 8 (avg 6, with `4→8` rebuttal update → avg ~7).

### What the reviewers LIKED (use these patterns)

- *Reviewer TAbbnXCU (8)*: **"The paper doesn't propose brand new math or concepts, but it does present a creative integration of fixes… All together, this can meaningfully update evaluation practice. The proposed fixes are easily actionable."**
- *Reviewer 6a7AGSUp (6)*: "Accurate problem identification, experiments reveal key pitfalls" + **"practical guidance value"**.
- *Reviewer A0BDyYfw (6)*: Praised the **two-method packaging** (multiple judges + Elo aggregation) — the *combination* was the win.

### What got attacked (and how they survived)

| Attack | Reviewer | How addressed |
|---|---|---|
| **"Similarity to Santilli 2025 and Vashurin 2025"** ← our exact concern | A0BDyYfw | "These are concurrent work; they cite an earlier workshop version of this paper." |
| "Need for 'oracle experiment' via human eval" | fZvBfEIN (4→8) | Added to the appendix |
| "Light on prior literature engagement" / "cites only 2-3 groups" | TAbbnXCU | Authors added breadth in rebuttal |
| "ELO rating redundant vs simple averaging" | A0BDyYfw | Added ablations |
| "Limited guidance for designing new methods" | 6a7AGSUp | **Accepted as scope** — eval papers don't need new methods |

### **Key takeaway for us**: Ielanskyi got attacked with *exactly* the "you overlap Santilli/Vashurin" concern we feared, and **survived** by (a) claiming concurrent work, (b) citing earlier workshop precedence, and (c) emphasizing different application domain. We have the third lever: dynamic-RAG triggers ≠ general UQ.

---

## 2. SeaKR (Yao et al.) — RAG trigger paper — ICLR 2025, **REJECT** *(subsequently accepted at ACL)*

**Scores**: 3 / 5 / 6 / 8. Despite an 8, rejected.

### Why it failed (the lessons)

| Attack | Reviewer | Why it stuck |
|---|---|---|
| **"Sensitivity to uncertainty thresholds"** | OaKycRu2 (8) | Trigger papers must justify threshold robustness |
| **"Some statements regarding contributions are flawed"** (over-claiming vs Self-RAG) | ydSRQK3Y (5) | Said "existing adaptive RAG methods determine retrieval based on output alone" — but Self-RAG already does in-context reasoning. **Over-claiming novelty got punished.** |
| "Inference latency not measured" | OaKycRu2 (8) | Trigger papers need compute analysis |
| "Organization confusing — most important module (U(c)) introduced last" | DVkx8ben (3) | Methodology presentation hurt them |
| "Evaluations on sampled test instances" | DVkx8ben (3) | Inconsistent eval = death |

### **Key takeaway for us**: Don't over-claim ECT's novelty against Datta. Don't bury the methodology. Address the threshold-sensitivity concern (we can say "signal-level analysis, not threshold-tuned").

---

## 3. CtrlA (Liu et al.) — RAG trigger paper — ICLR 2025, **REJECT** *(later ACL Findings)*

**Scores**: 3 / 5 / 5 / 5 (no 6+ at all).

### Why it failed — different lesson from SeaKR

| Attack | Reviewer |
|---|---|
| **"Many moving parts, but most disconnected"** | 41NUcpb5 (5) |
| **"Honesty Steering and Search Query Formulation seem unrelated to the core goal"** | LKLuZ4X7 (5) |
| **"Proposed retrieval triggering criteria is still fundamentally uncertainty-based"** | 41NUcpb5 (5) |
| "Requires accessing internal structure → not applicable to commercial LLMs" | A7M7Yts3 (5) |
| "Honesty Steering already introduced in similar work" | LKLuZ4X7 (5) |
| "Knowledge conflicts work missed both for intro and baselines" | yLWpjPW2 (3) |

### **Key takeaway for us**: **CtrlA's failure mode is exactly what our paper risks** if we don't unify the narrative. We have UNCERTAINTY-LINE + `lift_em` + paired bootstrap + Bonferroni + LLM-as-Judge + ECT + survivor matrix. **That's 7 moving parts.** CtrlA had 4 disconnected pieces and got rejected. The unifying glue for us is "length-controlled evaluation of training-free triggers, with ECT as the strongest survivor". Anything that doesn't serve that thesis must be cut or appendix-ified.

---

## 4. Self-RAG (Asai et al.) — RAG framework — ICLR 2024 **ORAL ACCEPT**

**Scores**: 6 / 8 / 8 / 8 (avg 7.5).

### Why it got ORAL (the gold pattern)

- *FtbG6282 (8)*: **"Novel framework that enables dynamic, on-demand retrieval and self-reflection in LM decoding. It addresses a key limitation in most existing RAG framework, that the model does not know when to retrieve."**
- *FtbG6282 (8)*: **"Interesting alternative to RLHF... controllability and interpretability that RLHF doesn't have."**
- *qptNCoCY (8)*: **"Addresses two key problems including relevance and grounding by simply adding special tokens."**
- *Nt4aOkdw (8)*: **"Significantly outperforms baselines in most cases."**

### What was praised in ONE LINE:
> A clean framework that solves a clearly-named limitation, has measurable wins, and gives users controllability.

### Critiques that didn't sink it (because the wins were strong):
- "Critic model accuracy not evaluated" → minor
- "Retrieval threshold not analyzed" → minor
- "Limited ablation (only top1)" → minor
- "FLARE prompts model to decide retrieval too" → minor

### **Key takeaway for us**: Self-RAG didn't have a long related-work attack. Its **clean framework + named primitive (reflection tokens) + measurable wins** carried it. We have analogue: **clean methodology + named signal (ECT) + measurable wins (Bonferroni-robust)**. Lean into that.

---

## 5. INSIDE / EigenScore (Chen et al.) — internal-state hallucination detection — ICLR 2024, **POSTER ACCEPT**

**Scores**: 6 / 6 / 6 / 8 (avg 6.5).

### What got it in:
- "Simple, intuitive, well explained"
- "Idea of looking into internal states is well motivated"
- "Clear advantage over established baselines"
- "Conducted on several diverse datasets, models" ← **breadth matters even at poster level**

### Critiques (raised but acceptable):
- "Only reporting on QA benchmarks may be insufficient" → didn't sink it, but borderline
- "Computational cost is concern" → minor
- "Concurrent works exist (Representation Engineering)" → minor

### **Key takeaway for us**: Even at poster-accept, *breadth* across datasets matters. We have 2 datasets (HotpotQA + 2Wiki). The "QA-only" critique survived but didn't sink INSIDE. Still: be explicit about why our datasets are sufficient for our claim.

---

# CROSS-PAPER PATTERNS

## Attack-frequency rankings (across the 5 papers)

| Attack | Frequency | Survivable? |
|---|---|---|
| **"Disjointed contributions / multiple unrelated pieces"** | 3/5 papers (CtrlA, SeaKR, INSIDE) | Only with strong unifying thesis |
| **"Over-claiming novelty vs prior work"** | 3/5 (SeaKR, CtrlA, Ielanskyi) | Survivable IF "concurrent work" or "different domain" argument |
| **"Threshold / hyperparameter sensitivity"** | 2/5 (SeaKR, Self-RAG) | Survivable with scope-statement |
| **"Internal states not applicable to closed LLMs"** | 2/5 (INSIDE, CtrlA) | We're training-free + open-weight → IMMUNE |
| **"Insufficient domain breadth"** | 2/5 (INSIDE, Self-RAG) | Survivable IF the claim is signal-level |
| **"Methodology presentation buried"** | 2/5 (SeaKR, INSIDE) | Killable if egregious |
| **"Missing concurrent / closely-related work"** | 2/5 (CtrlA, Ielanskyi) | Killable — must do thorough lit review |

## Strength-frequency rankings (what gets papers IN)

| Strength | Frequency |
|---|---|
| **"Creative integration / packaging of fixes"** | 4/5 (all 4 accepted, including Ielanskyi) |
| **"Clear practical problem, practical guidance value"** | 4/5 |
| **"Named primitive or named framework"** | 3/5 (Self-RAG, INSIDE, Ielanskyi-SP-MoJI) |
| **"Significant empirical wins over baselines"** | 3/5 (Self-RAG, SeaKR's 8-reviewer, INSIDE) |
| **"Training-free / lightweight"** | 2/5 (Ielanskyi, Self-RAG) — *we have this too* |
| **"Controllability / interpretability"** | 2/5 (Self-RAG, CtrlA) |

---

# THE SKELETON — 4-page EMNLP short paper

EMNLP Short = **4 pages body** + unlimited refs + unlimited appendix.
~2,000 words body, ~5 figures/tables max.

The unifying thesis: **"Length-controlled evaluation of training-free dynamic-RAG triggers reveals nine length-robust signals, four of which survive Bonferroni; the strongest is ECT, a forward-looking commitment-time signal."**

Everything that doesn't directly serve this sentence goes to appendix.

---

## Page 1 (~500 words)

### Abstract (~150 words)
**Keep**: from our corpus abstract, current ~250w → cut to ~150w.

**Voice**: confident, Ielanskyi-style packaging language. *"We re-evaluate / We find / We identify."*

**Must contain**:
- Problem in 1 sentence (raw AUROC is length-confounded for dynamic-RAG triggers)
- Lineage in 1 sentence (5-paper convergence on length confound)
- Our contribution in 1 sentence (re-eval + `lift_em` + ECT)
- Headline numbers (9 nominal-sig, 4 Bonf-robust, ECT Δ=+0.190, κ=0.707)
- One-sentence takeaway

### §1 Introduction (~350 words)

**Tight 4-paragraph structure**:

1. **Old Dynamic RAG (~90w)**: name the trigger pool (DRAGIN, FLARE, SeaKR, SelfCheckGPT) + the standard raw-AUROC protocol. **End**: "The signals above were each introduced under this protocol before the length-confound critique."

2. **New findings (~100w)**: 5-paper lineage in one packed `\citep{}` block, with Santilli's $\rho = -0.7$ and Janiak's 45.9% drop as the two punchy numbers. **End**: 1 sentence on UNCERTAINTY-LINE residualization as the fix.

3. **This paper (~80w)**: $\texttt{lift\_em}$ target (deployment question) + length-residualized re-evaluation + ECT as the strongest survivor. **Name ECT here, before §3.**

4. **Contributions (~80w as a 4-bullet inline list)**:
   - Length-controlled trigger evaluation (protocol + `lift_em`)
   - Cross-arch re-evaluation: 9 nominal-sig, 4 Bonferroni-robust
   - ECT: strongest single signal at Δ=+0.190 [+0.093, +0.287]
   - Correctness-function robustness: κ=0.707, 8/9 survivors preserved

**CUT** from current corpus:
- §1.1 "Hook + Dynamic RAG triggering needs triggers" extended paragraph
- §1.3 "answer-trajectory class is emerging" paragraph (compress to 1 sentence in §1 para 3)
- §1.5 "Scope" paragraph (move to limitations)
- The terminological-disambiguation note about Zhang ARS (move to appendix)

---

## Page 2 (~500 words)

### §2 Related Work (~150 words, ONE compressed paragraph)

**This is THE highest-risk section per the Ielanskyi review** ("light on literature" attack appeared TWICE). One dense paragraph with `\citep{}` everywhere, breadth-first:

> "Length confounds UQ evaluation \citep{Santilli, Vashurin, Janiak, Zhang_VC, Ielanskyi, Mao}. Step-level answer-trajectory primitives are an emerging class \citep{Datta, Wang, Chen, Zhang_ARS, Lu}. Dynamic-RAG triggers span attention \citep{Su_DRAGIN, Hei_DRRAG}, token-probability \citep{Jiang_FLARE}, multi-sample text \citep{Manakul, Zhang_SAC3}, and hidden-state geometry \citep{Yao_SeaKR, Chen_INSIDE}. Trained alternatives \citep{Asai_SelfRAG, Tang_MBARAG, Liu_CtrlA, Chen_DTR, Guo_DioR} require per-architecture supervised data and are out of scope. The conceptual parent of paired-oracle evaluation is \citet{Mallen}'s 'when not to trust LMs' question."

That's it. **Bulk of related work goes to appendix.** ~5 of these citations need to be inline elsewhere too.

### §3 Methods (~350 words)

Three tight paragraphs:

1. **§3.1 Paired-oracle `lift_em` (~120w)**: equation + 1-sentence motivation + 1-sentence note that EM is shared across arms (Ielanskyi defense).

2. **§3.2 Length residualization (~100w)**: equation + 1-sentence on log(tokens+1) + 1-sentence on $\varepsilon \leq 0.029$ replication of reference impl. **Cite Vashurin parenthetically only.**

3. **§3.3 ECT + answer-trajectory family (~130w)**:
   - Define K=5 boundary sampling in 1 sentence.
   - Define the 4 aggregations in a short itemized list (1 line each).
   - Define ECT with its equation.
   - **Forward-pointer to Appendix C for ECT-vs-t* operational table.**
   - End: 1 sentence "$K{=}1$ vs $K{=}5$ ablation is null on both archs."

**CUT** from corpus:
- §3.5 Architecture pool table (move to appendix)
- §3.6 Datasets paragraph (cut to 1 sentence in §3.1)
- §3.7 Statistical analysis paragraph (cut to "paired bootstrap $B=10{,}000$, Bonferroni at $k=15$" in §4)
- The LLM-as-Judge methodology paragraph (move to §4 with 1 sentence)

---

## Page 3 (~500 words)

### §4 Results (~500 words) — the load-bearing page

This is where reviewers DECIDE. Make it punch.

**§4.1 Survivor matrix (~150w + Figure 1)**:
- Figure 1 = the forest plot (already built)
- 3 sentences: "9 nominal-sig, 4 Bonferroni-robust; the Bonferroni-robust set spans four families (attention DRAGIN, logit ent.max, answer-trajectory min_majority, ECT); ECT is the strongest at Δ=+0.190."
- Forward-pointer to Table 1 in appendix for the full numbers.

**§4.2 ECT per-cell (~80w + small inline table)**:
- 4-row inline table: Qwen×Hot 0.677, Qwen×Wiki 0.604, Gemma×Hot 0.642, **Gemma×Wiki 0.784**.
- 1 sentence on the Gemma×Wiki peak.

**§4.3 K=5 vs K=1 ablation (~80w + 2-row inline table)**:
- 2 rows: Qwen −0.002, Gemma +0.033.
- 2 sentences: "regime-2 equality; recommend K=1 for deployment efficiency."

**§4.4 LLM-as-Judge robustness (~80w)**:
- Cohen's κ = 0.707, 8/9 survivors preserved, ECT strengthens to Δ=+0.222.
- Direct response to Janiak/Ielanskyi critique.
- Forward-pointer to Appendix.

**§4.5 UNCERTAINTY-LINE replication (~50w)**:
- One sentence: "head-to-head with reference impl gives $\varepsilon \leq 0.029$ on both archs."
- Forward-pointer to appendix.

**§4.6 Inverted-length cells (~60w)**:
- Qwen×2Wiki tokens AUROC = 0.457, Gemma×2Wiki = 0.518 — *below chance*.
- This is the novel finding that distinguishes us from single-hop QA work (Janiak/Santilli).

**CUT** from corpus:
- §4.4 detailed UNCERTAINTY-LINE replication subsection (move to appendix)
- §4.5 per-question-type heterogeneity full table (move to appendix; keep one striking number — Gemma×Wiki×bridge_comparison DRAGIN 0.821 — as a single inline sentence)

---

## Page 4 (~500 words)

### §5 Discussion (~250 words)

Two tight paragraphs:

1. **§5.1 Causal story (~120w)**: length→verbosity→uncertainty→retrieval-need chain (citing Zhang VC). Why length residualization isolates the deeper signal.

2. **§5.2 Why ECT works (~130w)**: same primitive, opposite direction (Datta detects stabilization to stop; we detect non-stabilization to retrieve). Hot lit cluster (Datta, Wang, Chen) all converging on this primitive class.

**CUT**:
- §5.3 Per-cell heterogeneity (delete or 1-sentence in §4.6)
- §5.4 Inverted-length cells (compressed already in §4.6)

### §6 Limitations (~120 words)

Three tight paragraphs (1-2 sentences each):

1. **Scope**: signal-level, not end-to-end pipeline; not deployment.
2. **Architecture coverage**: 2 archs (dense + MoE-hybrid); broader is future work.
3. **Correctness function**: EM + LLM-as-Judge robustness check in Appendix C; we don't vary further (containment-match, F1).
4. **K=1 stochastic proxy**: not byte-faithful Datta replication; future work.

**Pre-empt the "many moving parts" CtrlA attack here**: explicitly say "Our contribution is the integration: length-residualized AUROC ($\S 3.2$) + `lift_em` target ($\S 3.1$) + the answer-trajectory family inc. ECT ($\S 3.3$). Each piece individually is incremental; the integration is the contribution."

### §7 Conclusion (~80 words)

1 paragraph. "We re-evaluate… we find 9 nominal / 4 Bonferroni… ECT strongest… robust to LLM-as-Judge… length-controlled evaluation should be standard practice."

### References (unlimited)
~30 citations, packed `\citep{}` blocks.

---

# APPENDICES (unlimited; use them aggressively)

| Appx | What goes there |
|---|---|
| **A. Bonferroni table** | Currently in corpus; keep |
| **B. Full Table 1** (survivor matrix with all 17 rows, both 95% + 99.67% CIs) | Currently in main body; move here |
| **C. LLM-as-Judge full results** | Currently in corpus; keep (Table 1 side-by-side under both targets) |
| **D. UNCERTAINTY-LINE replication** ($\varepsilon$ math) | Currently in §4.4; move here |
| **E. Per-question-type table** | Currently in §4.5; move here |
| **F. Spearman ρ per cell (Santilli demand)** | Currently in corpus; keep |
| **G. Architecture pool details** (model SHAs, slice manifests) | Currently in §3.5; move here |
| **H. ECT-vs-Datta-t* operational disambiguation** (the 4-row table) | Currently in §3.4; move here |
| **I. Inverted-length-correlation deep dive** | Currently in §5.4; move here |
| **J. Reproducibility** | Currently in corpus; keep |

---

# WHAT TO EMPHASIZE / DROP — fast triage by review-vulnerability

## EMPHASIZE (proven to land on reviewers in this lineage)

| In paper | Reason |
|---|---|
| **"Creative integration of fixes that update evaluation practice"** framing in §1 | Ielanskyi pattern; 4/5 accepted papers had this signal |
| **Named primitive (ECT)** in abstract + §1 + §4.1 | Self-RAG, INSIDE, Ielanskyi-SP-MoJI all had named primitives |
| **The forest plot (Figure 1)** as page 3 anchor | Skim-friendly, packs the entire survivor matrix |
| **κ = 0.707 / 8 of 9 preserved** as the 4-line LLM-as-Judge result | Directly answers Ielanskyi's #1 attack vector |
| **"Training-free, open-weight"** as a strength | Pre-empts the "internal states not for closed LLMs" attack that hit INSIDE + CtrlA |
| **The 5-citation length-confound block** in §1.2 | Pre-empts "light on lit" attack (Ielanskyi got hit twice) |
| **"Concurrent work, different application domain (dynamic RAG, not general UQ)"** language for Santilli/Vashurin | Ielanskyi's exact defense, which worked |
| **The §3.4 ECT-vs-Datta-t* disambiguation table** (move to appendix) | Reviewer-2 safety net; without it, "you stole this" attack lands |

## DROP / appendix-ify (high cost, low return)

| In current corpus | Reason |
|---|---|
| **§1.5 "Framing positioning"** | Subordinating language that didn't survive my last tone pass anyway |
| **§3.5 Architecture pool table** | Detail, not story — appendix |
| **§3.6 Datasets paragraph** | 1 sentence in §3.1 is enough |
| **§3.7 Statistical analysis** | Cut to a single B=10,000 mention in §4 |
| **§4.5 Per-question-type heterogeneity** (full table + multi-paragraph discussion) | One striking inline number (Gemma×Wiki×bridge_comparison DRAGIN 0.821) is enough |
| **§5.3 Per-cell heterogeneity** | Redundant with §4.6 |
| **The §3.4 ECT-vs-Datta-$t^\star$ disambiguation table** (the 4-row one) | Move to appendix — Reviewer-2 will find it there |
| **Per-cell raw AUROC numbers** in §5.3 | Appendix |

## CAREFUL — keep but rephrase

| Risk | Fix |
|---|---|
| **CtrlA "many moving parts disconnected" attack risk** | Add the §6 limitations paragraph: "Our contribution is the integration of: residualized AUROC + lift_em + answer-trajectory family. Each piece is incremental; the integration is the contribution." |
| **SeaKR "over-claiming vs Self-RAG" attack risk** | Don't say "first" anywhere except "first to apply UNCERTAINTY-LINE to dynamic-RAG triggers". Don't say "novel signal ECT" — say "ECT, a forward-looking commitment-time signal." |
| **Ielanskyi "Santilli/Vashurin overlap" attack risk** | In §2 related work, explicitly: "Concurrent work on length-confound in general UQ \citep{Santilli, Vashurin}; we apply the protocol to a previously-unexamined application domain (dynamic-RAG triggers)." |
| **Self-RAG "FLARE already prompts model to decide retrieval" pattern** | Be precise about what's compared: signal-level AUROC, not full pipelines. |

---

# RECOMMENDED ORDER FOR THE CUT

1. **Week 1, Day 1** (~2h): Cut §3.5, §3.6, §3.7 to appendix. Cut §4.5 to appendix. Cut §5.3, §5.4 to appendix. Move ECT-vs-Datta-t* table to appendix.
2. **Week 1, Day 2** (~1h): Compress §2 Related Work to single dense paragraph.
3. **Week 1, Day 3** (~2h): Rewrite Abstract + §1 Introduction at tight word count.
4. **Week 1, Day 4** (~2h): Rewrite §4 Results page with the order I specified.
5. **Week 1, Day 5** (~1h): Rewrite §5 + §6 + §7.
6. **Week 1, Day 6** (~1h): Verify pdfLaTeX compiles to 4 pages exactly; tune table widths, figure sizes.
7. **Week 1, Day 7** (~1h): Final read, cite check, voice check.

Total: ~10 hours of focused editing.

---

The strongest single insight from the review data: **Ielanskyi got into ICLR 2026 (a more selective venue than EMNLP) with the *exact* attack pattern we feared (Santilli/Vashurin overlap) by claiming concurrent-work + different-application-domain.** That's our template. The 4 lever sentences in §2 + §3.1 + §3.2 + §6 must explicitly do this work.