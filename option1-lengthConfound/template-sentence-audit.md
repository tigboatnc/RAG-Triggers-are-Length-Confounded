# Sentence-by-Sentence Audit — `template.tex`

> Working doc. Each sentence is one row. Tick `[x]` as we go. Reference convention: `<section-tag> S<n>` (e.g. `intro.p2 S3`, `methods.protocol S5`).

## Legend used in nested proof bullets

- **✓** verified (citation real & claim accurate; number matches data; code path exists).
- **⚠** partial / nuance (real paper but year/author/claim drift, or empirical match within rounding).
- **✗** unverified or wrong (citation key inconsistent w/ subject paper, no supporting code/output, or claim doesn't reproduce).
- **?** could not verify in available time / needs human eye.

## ⚠ CRITICAL infrastructure issues found during audit (independent of sentence content)

- **No `option1-lengthConfound/custom.bib`** — `template.tex` line 288 has `\bibliography{custom}` but there is no `custom.bib` anywhere in the repo (checked recursively). This will cause `bibtex`/`biber` to produce empty `[?, ?, ?]` placeholders for every `\citep{...}` / `\citet{...}` / `\citeauthor{...}` call in the PDF. **Must be authored before submission.** Every citation key audited below is a candidate entry.
- **No `option1-lengthConfound/figs/` directory** — `template.tex` line 113 has `\includegraphics{figs/survivor_forest.png}` and the figure file lives at `emnlp_26_lengthConfound/figs/survivor_forest.{png,pdf}`. The figure must be copied / symlinked / re-pathed before `pdflatex` will compile.
- **References-vs-citations match table** (one-line sketch, full per-sentence detail below):
  - `lewis_retrieval-augmented_2021` → Lewis et al. NeurIPS 2020 (arXiv 2005.11401). Year drift OK (NeurIPS 2020 / proceedings 2021). ✓
  - `trivedi_interleaving_2023` → IRCoT (Trivedi et al. ACL 2023). PDF present in `references-recollections-v2/`. ✓
  - `shao_enhancing_2023` → Iter-RetGen (Shao et al. EMNLP-Findings 2023, arXiv 2305.15294). ✓
  - `su_dragin_2024` → DRAGIN (Su et al. ACL 2024). PDF present. ✓
  - `jiang_active_2023` → FLARE (Jiang et al. EMNLP 2023). PDF present. ✓
  - `li_modeling_2026` → ETC / Li et al. "Modeling Uncertainty Trends for Timely Retrieval in Dynamic RAG" (**AAAI 2026 Oral**, arXiv 2511.09980 Nov 2025). PDF in `references-recollections-v2/` labeled 2025 = arXiv year. **Cite key 2026 = AAAI venue year. Both valid; no year drift.** ✓ Authors confirmed: Bo Li, Tian Tian, Zhenghua Xu, Hao Cheng, Shikun Zhang, Wei Ye. ETC = "Entropy-Trend Constraint" verified via official `WisdomShell/ETC` and `pkuserc/ETC` repos.
  - `yao_seakr_nodate` → SeaKR (Yao et al. ACL 2025, arXiv 2406.19215). ⚠ `_nodate` suffix should be replaced with `_2025`.
  - `chen_inside_2024` → INSIDE (Chen et al. ICLR 2024). PDF present. ✓
  - `guo_realm_2026` → **NAMING ISSUE.** PDF in `references-recollections-v2/` is "Guo et al. 2026 - When to Retrieve During Reasoning: Adaptive Retrieval for Large Reasoning Models" (arXiv 2604.26649). The signal introduced there is **RSUS** (Reasoning Step Uncertainty Score), NOT REALM. Cite key `guo_realm_2026` is **misleading** — should be `guo_when_2026` or `guo_rsus_2026`. Content claim ("multi-sample text-consistency, cross-sample step-boundary continuation agreement", K=3) is byte-faithful with the paper's `U_cons` component. ⚠
  - `pandey_selfdoubt_2026` → SELFDOUBT (Pandey et al.). PDF present. ✓
  - `datta_large_2026` → Datta et al. "Decide Early and Explain Later" (arXiv 2604.22266). PDF present. ✓
  - `wang_chain--probe_nodate` → Chain-of-Probe (Wang et al. NAACL-Findings 2025). PDF present. ⚠ `_nodate` should be `_2025`; also note **double-hyphen** in key (`chain--probe`) likely a Zotero-export artifact — fine for BibTeX but ugly.
  - `chen_i_2026` → Chen et al. "I May Not Have Articulated Myself Clearly" / Dynamic Instability (PDF present). ✓
  - `santilli_revisiting_2025` → Santilli et al. arXiv 2504.13677. PDF present. ✓
  - `janiak_illusion_nodate` → Janiak et al. "Illusion of Progress" EMNLP 2025. PDF present. ⚠ should be `_2025`.
  - `zhang_verbosity_2024` → Y. Zhang et al. arXiv 2411.07858 (ACL 2025 UncertainLP). PDF present. ✓
  - `ielanskyi_addressing_2025` → Ielanskyi et al. arXiv 2510.02279. PDF present. ✓
  - `vashurin_uncertainty-line_2025` → Vashurin et al. arXiv 2505.19060. PDF present. ✓
  - `mallen_when_2023` → Mallen et al. ACL 2023 "When Not to Trust Language Models" (arXiv 2212.10511). Verified via web search. ✓
  - `zhang_commit_2026` → "When Does a Language Model Commit? A Finite-Answer Theory of Pre-Verbalization Commitment" (arXiv 2605.06723). Verified via web search; matches description in `App.~\ref{sec:appendix-fct-tstar}`. ✓
  - `mao_confidence_2026` → Mao et al. arXiv 2601.13387 (ACL 2026). PDF present. ✓
  - `zhang_harnessing_2026` → J. Zhang et al. "Harnessing Reasoning Trajectories" (ICML 2026). PDF present. ✓
  - `lu_streaming_2026` → Lu et al. arXiv 2601.02170. PDF present. ✓
  - `hei_dr-rag_2024` → DR-RAG (Hei et al. arXiv 2406.07348). Verified via web search. ✓
  - `manakul_selfcheckgpt_2023` → SelfCheckGPT (Manakul et al. EMNLP 2023). PDF present. ✓
  - `zhang_sac3_2023` → SAC³ (J. Zhang et al. EMNLP-Findings 2023). Verified via web search. ✓ **NOTE distinct from Y. Zhang verbosity-2024 and J. Zhang ARS-2026 — three different Zhang first authors. Make sure BibTeX entries disambiguate fully.**
  - `asai_self-rag_2023` → Self-RAG (Asai et al. arXiv 2310.11511). PDF present. ✓
  - `tang_mba-rag_2024` → MBA-RAG (Tang et al. COLING 2025, arXiv 2412.01572). Verified via web search. ⚠ Year drift: arXiv 2024 / venue 2025; either is defensible.
  - `chen_think_2026` → Chen et al. "Think Deep, Not Just Long". PDF present. ✓
  - `guo_dior_2025` → DioR (H. Guo et al. ACL 2025, arXiv 2504.10198). Verified via web search. ✓ (Note: distinct from `guo_realm_2026` / `guo_when_2026` — different first authors named Guo: Hanghui Guo vs the When-to-Retrieve Guo. Confirm both BibTeX entries are right.)
  - `yang_hotpotqa_2018` → HotpotQA (Yang et al. EMNLP 2018). Verified via web search. ✓
  - `ho_constructing_2020` → 2WikiMultiHopQA (Ho et al. COLING 2020). Verified via web search. ✓
  - `landis_measurement_1977` → Landis & Koch Biometrics 33(1):159–174. Verified via web search. ✓
  - `zheng_judging_2024` → Zheng et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (NeurIPS Datasets & Benchmarks 2023). Verbosity-bias claim in paper §B.2 — byte-faithful w/ the audit sentence. ⚠ Year drift: NeurIPS 2023 / proceedings publication 2024 — either is defensible.
  - `trivedi_musique_2022` → MuSiQue (Trivedi et al. TACL 2022, arXiv 2108.00573). Verified via web search. ✓

## Bottom-line takeaway from infrastructure pass

- **Citation pool is real** (every key audited maps to a paper whose subject matter is consistent with how it is invoked in `template.tex`); the dominant risk is the **missing `custom.bib`**, not the citations themselves.
- **Pre-existing PDFs cover ~80%** of the citation pool already (`__lit-review-main/` + `references-recollections-v2/`); the remaining ~20% (Lewis-RAG, IRCoT, Mallen, HotpotQA, 2Wiki, MuSiQue, Landis-Koch, Zheng-MTBench, DR-RAG, MBA-RAG, DioR, SAC3, Self-RAG-bib, Zhang-commit) are publicly available and have been verified by web search.
- **Two cite-key cosmetic issues** to fix before submission: rename `guo_realm_2026` → `guo_when_2026` (it's the RSUS paper, not REALM); resolve all `_nodate` suffixes (Janiak, Wang, SeaKR). Neither affects correctness, both are cosmetic.

---

## ⚠ Triage: ranked list of substantive issues uncovered by per-sentence audit

> **All numerical headline claims (n=800, Δ=+0.190, FCT=0.784, κ=0.707, Δ=+0.222, ε≤0.029, Bonferroni LB=+0.047) reproduce exactly against `analysis/*.out` outputs.** The empirical backbone of the paper is solid. The issues below are about claim *framing*, missing infrastructure files, and small specification drifts between paper and code.

### P0 — must-fix before submission (compile-blocking or factually wrong)

1. **No `custom.bib`** anywhere in repo. `\bibliography{custom}` on `template.tex` L288 will fail to resolve any citation. **Must author `option1-lengthConfound/custom.bib`** with the ~35 entries enumerated in the infrastructure pass above.
2. **No `option1-lengthConfound/figs/`**. Figure 1 (`\includegraphics{figs/survivor_forest.png}` on `template.tex` L113) will fail to compile. **Must copy / symlink `emnlp_26_lengthConfound/figs/survivor_forest.png` to `option1-lengthConfound/figs/`.**
3. **App. H S5 — "800 per-call JSONs":** actual cache contains **1,600** JSONs (`llm_judge_cache/{arch}/{cell}/*.json`). **Fix:** change "800" → "1,600" or rephrase as "the per-call JSON cache (oracle and closed-arm judgements for each of 800 paired traces)."

### P1 — substantive but defensible (reviewer-risk medium)

4. **§3.2 S5 — polynomial-regressor claim:** "higher-order polynomial regressors give no further improvement, consistent with Vashurin's App. D.1" reads as if **we** tested poly-2/3 on our data. Our `analysis/uncertainty_line_audit.py` only fits a single `LinearRegression`. **Either:** (a) re-frame as inheriting Vashurin's finding ("we adopt the linear fit, consistent with…"), or (b) add a poly-2/3 sweep on our data (cheap, ~30 min).
5. **§3.3 I2 `late_flips`:** paper says "any change in the **multiset** of sampled answers" but code uses `set(ans)` (drops multiplicity). **Fix:** either change paper claim to "any change in the **set** of sampled answers", or fix `signal_lib.py::late_flips_count` to compare `Counter(ans)`.
6. **App. C `\paragraph{Protocol}` — judge prompt quote drops the middle sentence:** code has 3 sentences ("…Your task is to decide whether a model-predicted answer is semantically equivalent to a gold answer.…"), paper quotes only 2. **Fix:** quote the full 3-sentence prompt for byte-faithfulness.
7. **App. G G.seakr S1 — "K=20 in the paper":** SeaKR paper's published best-k interval is [10, 25], not specifically K=20. **Fix:** change "K=20 in the paper" → "K ∈ [10, 25] in the paper, best-k interval" (the K interval *is* mentioned in the same sentence already; just drop the unsupported "K=20" anchor).
8. **§2 S2 — categorization of `lu_streaming_2026` / `zhang_harnessing_2026` as "step-level answer-trajectory triggers":** Both are *supervised* hidden-state-probe / contrastive-shaping methods, not training-free trajectory primitives. **Fix:** split S2 into two sub-clauses: "training-free trajectory primitives (Datta/Wang/Chen) + supervised hidden-state probes (Zhang ARS/Lu)" — disambiguating the family while preserving the broader "emerging length-orthogonal signal class" framing.
9. **§2 S3 — DR-RAG mentioned in the "training-free trigger pool we audit":** we do NOT audit DR-RAG (not in `signal_lib.py` SIGNALS). **Fix:** remove DR-RAG cite from S3, or move it to a separate sentence ("related dynamic-RAG frameworks such as DR-RAG…").
10. **App. A footnote F3 — k-sensitivity claim** ("Re-running at k=14 or k=16 shifts Bonferroni LBs by less than ±0.005 AUROC and does not change the four-survivor count"): not empirically tested. **Fix:** add `--bonferroni-k` flag to `e7_bonferroni_and_spearman.py` and run sensitivity (cheap, ~10 min), or hedge to "we estimate this shift to be < ±0.005."

### P2 — cosmetic / hedge tightenings (reviewer-risk low)

11. **§5.1 S1 — `tokens_generated` "single strongest pre-residualization signal on several cells":** on the Qwen+Gemma headline pool, internal signals (DRAGIN-RIND, ent.max) often *beat* `tokens_generated` raw. Hedge to "competitive baseline" or "competitive length-proxy on several cells".
12. **§4.2 S2 — "Longer traces … are systematically *less* retrieval-helpful":** Gemma×2Wiki tokens 0.518 is slightly *above* chance (just not predictively useful). The "systematically less" wording works for Qwen×2Wiki (0.457) but is mild overclaim for Gemma×2Wiki. Hedge to "no longer reliably retrieval-helpful, with raw AUROC at or below chance."
13. **§7.2 S2 — "three across two":** if `min_majority` drops, the remaining 3 signals span 3 families (attention/DRAGIN-RIND + token-prob/ent.max + answer-traj/FCT), not 2. Hedge or specify "the `ent.max`-drop case reduces to three signals across two families."
14. **App. F C4 — answer-trajectory block "≈ 0.1 lower in |ρ|":** computed difference is ~0.16, not 0.1. Either tighten to "≈ 0.15" or keep round number with explicit "≈" hedge.
15. **App. G G.etc S2 — "the second-difference operator is byte-faithful":** Li 2026's Δ²H is signed; our `etc_score` uses `max|Δ²H|` (absolute). Hedge to "we evaluate `max|Δ²H|` rather than the signed second-difference; magnitude-faithful but not sign-faithful."
16. **Intro p1 S5 — "the six published peer families predate the length-confound literature":** ETC (2026), SeaKR (2025), RSUS (2026), SELFDOUBT-T1 (2026) are contemporaneous, not predating. Only DRAGIN (2024) and FLARE (2023) clearly predate. Hedge to "the earliest published peer families predate" or "most of the peer-pool families predate".
17. **App. H S4 — script enumeration:** add `analysis/e7_bonferroni_and_spearman.py` to the list (it produces the headline Table 1 / Bonferroni and App. F / Spearman).
18. **Intro p3 S1 — `\citep{mallen_when_2023}` placement:** Mallen does not formalize `lift_em`; he asks the *conceptual* question. Either keep as a conceptual-credit cite (defensible) or strengthen to "conceptually motivated by `\citet{mallen_when_2023}`".
19. **§2 S4 — categorization of `chen_think_2026` ("Think Deep") as a trained alternative:** verify by reading the PDF; if not trained, move out of S4.
20. **App. C `\paragraph{Caveat}` — `\citet{zheng_judging_2024}` verbosity-bias cite:** the literature has moved on (Judging-the-Judges 2406.07791 argues it's primarily position-bias-driven). Keep as is, or add a one-clause hedge.

### Citation-key cosmetic fixes (P3, cosmetic)

- `guo_realm_2026` → `guo_when_2026` or `guo_rsus_2026` (it's the When-to-Retrieve paper, not REALM).
- `yao_seakr_nodate` → `yao_seakr_2025`.
- `janiak_illusion_nodate` → `janiak_illusion_2025`.
- `wang_chain--probe_nodate` → `wang_chain-of-probe_2025` (also fixes the double-dash).
- `tang_mba-rag_2024` → `tang_mba-rag_2025` (COLING 2025 venue) — or keep 2024 as arXiv year; both defensible.

---

# Title (`title`)

- [ ] **T1.** Length-Controlled Evaluation of Training-Free Dynamic-RAG Triggers
  - ✓ **Title vs. claims.** Title makes four implicit promises, each cashed elsewhere: (1) **length-controlled** = the residualization protocol of §3.2 (Eq. 2). (2) **evaluation** = the survivor-matrix protocol of §4.1 + appendices. (3) **training-free** = explicit RQ1 scope; trained alternatives Self-RAG/MBA-RAG/CtrlA/DioR are explicitly dropped in `\S 2` (Related Work S4). (4) **dynamic-RAG triggers** = the published pool DRAGIN/FLARE/SeaKR/ETC/RSUS/SELFDOUBT-T1 + answer-trajectory family enumerated in `\S 1` intro p1 S3-S4. All four promises are visibly cashed.
  - ✓ **Spelling.** "Dynamic-RAG" is hyphenated consistently throughout the rest of the paper.
  - ⚠ **Hedge against title overclaim.** "Length-Controlled Evaluation" is a stronger phrasing than "Length-Residualized Evaluation"; the protocol is regression-residualization (Eq. 2), which controls *linearly* for `log(tokens+1)`, not for arbitrary length-correlated factors. The `\S 3.2 S5` sentence ("higher-order polynomial regressors give no further improvement") and `App. uline` evidence (`ε_protocol ≤ 0.029`) defend the linear choice. Defensible as is; minor reviewer-prep angle.

---

# Abstract

- [ ] **S1.** Dynamic retrieval-augmented generation systems decide *when* to retrieve from a scalar trigger signal, scored by raw AUROC against binary correctness.
  - ✓ **Framing claim — uncited prose.** This is descriptive of standard practice; the supporting citations are deferred to `intro.p1 S5` ("All seven follow the same evaluation protocol... rank traces by the trigger score; report raw AUROC..."). No citation required at the abstract level; matches standard practice in DRAGIN, FLARE, SeaKR, INSIDE, ETC, RSUS papers.
- [ ] **S2.** A converging recent literature has shown this protocol is length-confounded on adjacent QA selective-prediction and UQ tasks, yet most dynamic-RAG trigger evaluations still use it.
  - ✓ **Claim has 5 supporting citations** in §1 p2 (`santilli_revisiting_2025`, `janiak_illusion_nodate`, `zhang_verbosity_2024`, `ielanskyi_addressing_2025`, `vashurin_uncertainty-line_2025`) and §2 S1; all 5 PDFs verified. The "converging" hedge is well-supported.
  - ✓ **"adjacent QA selective-prediction and UQ tasks"** — accurate: Santilli/Janiak/Vashurin all use single-hop QA + NMT/summarization/math, NOT multi-hop QA-with-paired-oracle. The "adjacent" hedge is precisely calibrated.
- [ ] **S3.** We present the first length-controlled audit of the published training-free dynamic-RAG trigger pool under length-residualized AUROC on a paired-oracle target (`lift_em`=+1: retrieval rescue).
  - ✓ **"first" claim.** No prior length-residualized audit of the DRAGIN/FLARE/SeaKR/ETC/RSUS/SELFDOUBT-T1 dynamic-RAG trigger pool exists in `__lit-review-main` or `references-recollections-v2`. Vashurin 2025 applies the protocol to NMT/summarization/math UQ; Janiak 2025 / Santilli 2025 don't apply residualization at all (they identify the problem); our paper is the first to combine residualization + the dynamic-RAG-trigger pool + paired-oracle target. Defensible.
  - ✓ **`lift_em=+1` definition** — cashed in §3.1 Eq. 1 and code at `emnlp_26_lengthConfound/analysis/r1_paired_audit.py` line 66 (`lift = +1 if o_em > c_em else (-1 if o_em < c_em else 0)`). Code is byte-faithful.
- [ ] **S4.** On two open-weight chat models over two multi-hop QA datasets (n=800 paired traces), nine signals beat the length baseline at 95% paired-bootstrap, and four survive Bonferroni at k=15 across three signal families.
  - ✓ **n=800.** `e7_bonferroni.out` line 2: `n=800 pop=108 base=0.135 target=lift_em=+1 B=10000`. Matches.
  - ✓ **9 signals at 95%.** `e7_bonferroni.out`: signals with `✓` at 95% column (counting): DRAGIN-RIND, ent.max, etc (Δ²H max), u_cons.max, flip_rate K=5, flip_rate K=1, min_majority, CABV, FCT = **exactly 9**. Matches.
  - ✓ **4 survive Bonferroni.** Same file, `✓✓` column: DRAGIN-RIND, ent.max, min_majority, FCT = **exactly 4**. Matches.
  - ✓ **k=15.** `e7_bonferroni.out` line 3: `Bonferroni at k=15: α=0.0033`. Matches the Eq.-of-α derivation 0.05/15 = 0.00333.
  - ✓ **"three signal families."** Among the 4 Bonferroni-robust: attention (DRAGIN-RIND), token-probability (ent.max), answer-trajectory (min_majority + FCT). Counts 3 distinct families. Matches.
  - ✓ **"two open-weight chat models over two multi-hop QA datasets."** Cells loaded in `runs/v2_headline_qwen2.5-7b-instruct/` and `runs/v2_headline_gemma-4-26b-a4b-it/`, each with HotpotQA (D1/E1) and 2WikiMHQA (D2/E2) jsonl files. n=200×4 = 800 paired. Matches.
- [ ] **S5.** We propose **FCT** (first-commitment time), a forward-looking variant of the answer-trajectory primitive that flags traces where the model commits to its final answer only late in the decode; FCT exceeds the length baseline by Δ = +0.190 [+0.093, +0.287], the strongest length-orthogonal signal.
  - ✓ **FCT definition.** `moonshot_explore.py` lines 141–169 (`fct_score`): `t_commit = smallest i such that for every j >= i with answers populated, the K=5 majority answer at step j equals final_ans`. Returned value `t_commit / max(1, n_populated_steps - 1)`. Byte-faithful with §3.3 Eq. 3.
  - ✓ **Δ = +0.190 [+0.093, +0.287].** `e7_bonferroni.out` line 26: `FCT 0.691 +0.190 [+0.093,+0.287]`. Exact match.
  - ✓ **"strongest length-orthogonal signal."** `e7_bonferroni.out`: among the 4 Bonferroni survivors, FCT's Δ (+0.190) is largest (next: DRAGIN-RIND +0.167). Matches.
  - ⚠ **"forward-looking variant of the answer-trajectory primitive."** True — FCT is forward-looking (find smallest i such that majorities stabilize *after*), distinct from Datta's backward-looking t* (largest i such that A_i ≠ A_{i+1}). The same forward/backward distinction is tabulated in App. fct-tstar Table 5 Row 1. The "primitive" being extended is Datta's `AnswerSwitches`/t*. Defensible.
- [ ] **S6.** On the held-out 2Wiki cells, raw answer length is *anti*-predictive of retrieval need (AUROC ≤ 0.52), inverting the verbosity-compensation pattern documented on adjacent QA tasks.
  - ✓ **"AUROC ≤ 0.52" on 2Wiki cells.** `moonshot_explore.out` line 47 (Gemma×2Wiki HotpotQA pooled n=400): `tokens_generated 0.558 0.439` — that's pooled, not per-cell. Per-cell numbers in §4.2 S1 say "raw AUROC 0.457 and 0.518" for Qwen×2Wiki and Gemma×2Wiki. Both ≤ 0.52. ✓. 0.457 < 0.5 = anti-predictive on Qwen × 2Wiki. ✓
  - ⚠ **Per-cell raw AUROC for `tokens_generated` on 2Wiki cells.** Need to confirm by reading per-cell breakdown more carefully. From `uncertainty_line_audit_qwen2.5-7b-instruct.out` line 42: `tokens_generated (sanity) 0.457` (Qwen × 2Wiki); from `uncertainty_line_audit_gemma-4-26b-a4b-it.out` line 42: `tokens_generated (sanity) 0.518` (Gemma × 2Wiki). Both ≤ 0.52. ✓ **EXACT MATCH.**
  - ✓ **"inverting the verbosity-compensation pattern."** Zhang_verbosity_2024 documents verbose-when-uncertain on LongBench (which bundles HotpotQA/MuSiQue/2WikiMHQA), so the *expected* direction is positive AUROC on `closed_em=0`. On the held-out 2Wiki `lift_em=+1` cells we observe negative (anti-predictive). The inversion is real and correctly attributed.
- [ ] **S7.** The survivor set is **robust to a semantic-equivalence judge**: under a GPT-4o-Mini correctness function (vs. EM, κ = 0.707), eight of nine survivors are preserved and FCT *strengthens* to Δ = +0.222.
  - ✓ **κ = 0.707.** `llm_judge_audit_both.out` line 7: `Cohen κ: 0.707`. Exact match.
  - ✓ **"eight of nine survivors preserved."** Count from `llm_judge_audit_both.out`: under `lift_em_judge=+1` (top block), signals with `✓` at 95%: DRAGIN-RIND, ent.max, flare.max, u_cons.max, flip_rate K=5, flip_rate K=1, min_majority, CABV, FCT = 9 signals. Cross-checked against the 9 EM survivors (DRAGIN-RIND, ent.max, etc, u_cons.max, flip_rate K=5, flip_rate K=1, min_majority, CABV, FCT): **`etc` is lost, `flare.max` is gained** — net survivors = 9 (same count, one swap). Of the original 9 EM survivors, 8 are still significant under judge (etc dropped). ✓ Matches the audit sentence.
  - ✓ **"FCT strengthens to Δ = +0.222."** `llm_judge_audit_both.out` line 32: `FCT 0.722 +0.222 [+0.132,+0.312]`. Exact match.
  - ✓ **"GPT-4o-Mini correctness function."** `llm_judge_audit_both.out` line 2: `model: gpt-4o-mini`. Matches.

---

# Figure 1 caption (`fig:survivor-forest`)

- [ ] **S1.** **Four trigger signals carry length-orthogonal information beyond the length baseline; the strongest is FCT (first-commitment time, this paper).**
  - ✓ **Four-survivors claim.** `e7_bonferroni.out` `✓✓` column: DRAGIN-RIND, ent.max, min_majority, FCT. Match.
  - ✓ **"strongest is FCT."** Δ = +0.190 for FCT > +0.167 DRAGIN-RIND > +0.149 min_majority > +0.137 ent.max. Match.
  - ✓ **Figure file.** Plotting code at `emnlp_26_lengthConfound/analysis/plot_survivor_forest.py` (260 lines, opened recently). Output exists at `emnlp_26_lengthConfound/figs/survivor_forest.{png,pdf}`. **⚠ deployment-blocker reminder**: file lives in `emnlp_26_lengthConfound/figs/`, paper expects `figs/` relative to `option1-lengthConfound/template.tex`. Must copy/symlink.
- [ ] **S2.** Paired-bootstrap forest plot of Δ = AUROC(s_resid) − AUROC(`tokens_generated`) on n=800 paired traces (Qwen2.5-7B + Gemma-4-26B-A4B-it × {HotpotQA, 2WikiMHQA}, B=10,000).
  - ✓ **Formula.** `signal_lib.py` lines 266–292 (`bootstrap_auroc_diff`): paired bootstrap of `Δ = AUROC(key_a) − AUROC(key_b)`, applied at call site `e7_bonferroni_and_spearman.py` line 225 with `(f"__res_{key}__", "tokens", is_pos)` — i.e. residualized signal vs. raw tokens, on `lift_em == +1`. Byte-faithful.
  - ✓ **n=800, B=10,000.** Confirmed in `e7_bonferroni.out`.
  - ✓ **Architecture × dataset cells.** All 4 cells (Qwen×Hot, Qwen×Wiki, Gemma×Hot, Gemma×Wiki) confirmed loaded in `e7_bonferroni_and_spearman.py` `load_arch_cells` function, called for both Qwen and Gemma. Pooled at line 179: `pool = qwen_rows + gemma_rows`.
- [ ] **S3.** Thick bars: 95% paired-bootstrap CIs; thin extensions: Bonferroni-corrected ~99.67% CIs at k=15.
  - ✓ **Bonferroni 99.67%.** `e7_bonferroni_and_spearman.py` line 226: `alphas=(0.05, 0.05 / N_SIGNALS_FOR_BONFERRONI)`, where `N_SIGNALS_FOR_BONFERRONI = 15` (line 72). 0.05/15 = 0.0033 → 99.67% CI. Matches.
  - ⚠ **Plotting code visual choice.** Need to read `plot_survivor_forest.py` to confirm the thick/thin distinction is exactly that. (Have seen the file is 260 lines and PNG/PDF exist.) Caption claim is the standard convention; plot is unverified visually but the underlying statistic is correct.
- [ ] **S4.** Setup: §methods; full numerical tables: App. bonferroni, App. hero.
  - ✓ Cross-references resolve: `\S\ref{sec:methods}` line 141, `App.~\ref{sec:appendix-bonferroni}` line 293, `App.~\ref{sec:appendix-hero}` line 326.

---

# 1. Introduction

## intro.p1 — peer-pool framing

- [ ] **S1.** Dynamic retrieval-augmented generation (RAG) systems retrieve external evidence during decoding to ground generation in source material `\citep{lewis_retrieval-augmented_2021, trivedi_interleaving_2023, shao_enhancing_2023}`.
  - ✓ **`lewis_retrieval-augmented_2021`** → Lewis et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (NeurIPS 2020, arXiv 2005.11401). The seminal RAG paper. ⚠ Year drift: paper is NeurIPS 2020; cite key is 2021 (defensible since proceedings are dated 2020 but vol 33 published Dec 2020, some citation indexers say 2021).
  - ✓ **`trivedi_interleaving_2023`** → Trivedi et al. "Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions" (IRCoT, ACL 2023). PDF present at `references-recollections-v2/Trivedi et al. - 2023 - Interleaving Retrieval with Chain-of-Thought Reasoning…`. ✓
  - ✓ **`shao_enhancing_2023`** → Shao et al. "Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy" (Iter-RetGen, EMNLP-Findings 2023, arXiv 2305.15294). Verified via web search. ✓
  - ⚠ **Cite-context note.** All three are RAG foundations, not specifically *dynamic-RAG-trigger* papers (Lewis = the original static-RAG paper; IRCoT = trigger-on-every-step; Iter-RetGen = trigger-every-iteration). The sentence frames them as "retrieve external evidence during decoding to ground generation in source material" — generic enough that all three fit. Reasonable.
- [ ] **S2.** The control point of every such system is the *retrieval trigger*: a scalar function of the partial generation that decides when to invoke retrieval.
  - ✓ **Uncited prose.** Definitional setup; framing is consistent with DRAGIN/FLARE/SeaKR papers (which all define a "retrieval trigger" or equivalent gating decision). No citation needed.
- [ ] **S3.** The published peer pool of training-free triggers spans six signal families that each read a different decode-time observable: attention-based DRAGIN `\citep{su_dragin_2024}` (token importance from attention dynamics), token-probability FLARE `\citep{jiang_active_2023}` (next-token entropy / max log-prob at the decode head), entropy-trend ETC `\citep{li_modeling_2026}` (second-difference of step entropy), hidden-state SeaKR `\citep{yao_seakr_nodate}` (regularized Gram determinant of K sample EOS hidden states) with INSIDE `\citep{chen_inside_2024}` as the EigenScore variant, multi-sample text-consistency RSUS `\citep{guo_realm_2026}` (cross-sample step-boundary continuation agreement), and hedging-marker SELFDOUBT-T1 `\citep{pandey_selfdoubt_2026}` (hedge-to-verify ratio over the reasoning trace text).
  - ✓ **`su_dragin_2024`** → DRAGIN (Su et al. ACL 2024). PDF present. RIND signal description ("token importance from attention dynamics") byte-faithful with the paper's RIND = uncertainty × influence × semantic-contribution score (Su §3.1, Eq. 1–3).
  - ✓ **`jiang_active_2023`** → FLARE (Jiang et al. EMNLP 2023). PDF present. "next-token entropy / max log-prob at the decode head" matches FLARE's online trigger `max_t p_t < τ`.
  - ⚠ **`li_modeling_2026`** → "Modeling Uncertainty Trends for Timely Retrieval in Dynamic RAG" (Li et al.). PDF dated **2025** in `references-recollections-v2/`. **Year drift: 2025 vs cite key 2026.** Resolve before submission. Content claim ("second-difference of step entropy") matches Li's Δ²H trend operator (their §3). ✓ for content.
  - ⚠ **`yao_seakr_nodate`** → SeaKR (Yao et al. ACL 2025, arXiv 2406.19215). Verified via web search; description ("regularized Gram determinant of K sample EOS hidden states") matches Yao §3 *exactly*. ⚠ cite-key `_nodate` should be `_2025`.
  - ✓ **`chen_inside_2024`** → INSIDE / EigenScore (Chen et al. ICLR 2024). PDF present. "EigenScore variant" of SeaKR is correct: SeaKR uses uncentered Gram determinant, INSIDE uses centered covariance EigenScore; the K nonzero eigenvalues are rank-equivalent up to centering (confirmed at `App. seakr` G.seakr S2 of this paper).
  - ⚠ **`guo_realm_2026`** → **NAMING ISSUE (already flagged).** Paper is Guo et al. 2026 "When to Retrieve During Reasoning" (arXiv 2604.26649). RSUS is byte-faithful: paper §3 defines RSUS = α·U_verb + β·U_ent + γ·U_cons with U_cons = step-boundary K=3-sample agreement. The sentence cites only the cross-sample-consistency component (`U_cons`), accurate. **But cite key labels it `realm` which is a different paper** (Guu et al. REALM); rename to `guo_when_2026` or `guo_rsus_2026` before submission.
  - ⚠ **K=3 vs K=5.** Paper uses K=3 for U_cons; our audit uses K=5 for `u_cons.max` (the trace-level adapter). The §3.3 S1 / `methods.fct` S1 sentence says "we sample K=5 boundary completions". So we audit RSUS *primitive* at K=5 rather than K=3. This is an adapter relative to the originating paper; flagged in App. adapters G.opening. Defensible.
  - ✓ **`pandey_selfdoubt_2026`** → SELFDOUBT (Pandey et al.). PDF present. "Hedge-to-verify ratio" matches paper's HVR = h/(v+1). ✓
  - ✓ **"six signal families"** count: attention(DRAGIN) + token-probability(FLARE) + entropy-trend(ETC) + hidden-state(SeaKR/INSIDE) + multi-sample text-consistency(RSUS) + hedging-marker(SELFDOUBT-T1) = **6**. Matches.
- [ ] **S4.** The contemporaneous answer-trajectory family `\citep{datta_large_2026, wang_chain--probe_nodate, chen_i_2026}` (step-level dynamics of the model's predicted answer over the decode) extends this pool with step-level signals.
  - ✓ **`datta_large_2026`** → Datta et al. "Large Language Models Decide Early and Explain Later" (arXiv 2604.22266). PDF present. Their AnswerSwitches / t* primitives are step-level dynamics on the model's predicted answer trajectory — exactly the family described.
  - ⚠ **`wang_chain--probe_nodate`** → Wang et al. "Chain-of-Probe" (NAACL-Findings 2025). PDF present. ⚠ cite-key has double-dash AND `_nodate` — fix both: `wang_chain-of-probe_2025` is the proper form.
  - ✓ **`chen_i_2026`** → Chen et al. "I May Not Have Articulated Myself Clearly: Diagnosing Dynamic Instability in LLM Reasoning at Inference Time" (arXiv 2602.02863). PDF present.
  - ⚠ **"Contemporaneous"** — accurate hedge: Wang NAACL 2025 / Chen Feb 2026 / Datta Apr 2026 all post-date the original DRAGIN/FLARE/SeaKR papers (2023–2024), making them "contemporaneous" with each other and with our work. Defensible.
- [ ] **S5.** All seven follow the same evaluation protocol (rank traces by the trigger score; report raw AUROC against a binary correctness label); the six published peer families predate the length-confound literature.
  - ⚠ **"All seven"** — counts as: 6 peer (S3) + 1 answer-trajectory (S4 combined as one family). The "seven" includes the answer-trajectory family as the 7th. Defensible.
  - ✓ **"Raw AUROC against a binary correctness label"** — true for DRAGIN (eval on HotpotQA EM), FLARE (eval on long-form generation correctness), SeaKR (eval on QA EM), INSIDE (TruthfulQA accuracy), ETC (multi-hop QA EM), RSUS (`When to Retrieve…` paper uses correlation with downstream retrieval benefit, not pure AUROC — but the principle is the same: rank-then-score), SELFDOUBT-T1 (QA correctness).
  - ⚠ **"The six published peer families predate the length-confound literature"** — temporal claim: length-confound papers are Santilli 2025, Vashurin 2025, Janiak 2025, Ielanskyi 2025, Zhang Verbosity 2024. The peer-pool papers are DRAGIN 2024, FLARE 2023, ETC 2025 (or 2026 per cite key), SeaKR 2025, INSIDE 2024, RSUS 2026, SELFDOUBT-T1 2026. ETC, SeaKR, RSUS, SELFDOUBT-T1 are 2025–2026 = contemporary with length-confound papers, not predating. **DRAGIN 2024 and FLARE 2023 do predate.** Defensible-but-slightly-overclaim; consider hedging to "many of the published peer families predate" or "the earliest published peer families predate". ⚠ **Reviewer-risk: weak hedge.**

## intro.p2 — length-confound literature

- [ ] **S1.** A converging recent literature has shown the standard AUROC protocol is length-confounded on adjacent QA selective-prediction and UQ tasks.
  - ✓ Setup sentence; supporting citations follow in S2–S4. No standalone citation needed.
- [ ] **S2.** `\citet{santilli_revisiting_2025}` prove that shared length confounding systematically distorts AUROC, with Spearman ρ = −0.7 between length and Perplexity.
  - ✓ **`santilli_revisiting_2025`** → Santilli et al. arXiv 2504.13677. PDF present. The "shared length confounding systematically distorts AUROC" claim is their Appendix C Scenario (ii) formal result (confirmed in `__lit-review-main/_lit-review.md` lines 215–219).
  - ✓ **"Spearman ρ = −0.7 between length and Perplexity."** Per `_lit-review.md` line 224, Santilli's Figure 4 reports `token-length × Perplexity correlation: −0.7`. **Exact match.**
- [ ] **S3.** `\citet{janiak_illusion_nodate}` report Perplexity AUROC drops of up to 45.9% and EigenScore drops of up to 30.4% when switching ROUGE to LLM-as-Judge on certain Mistral cells; `\citet{zhang_verbosity_2024}` document the behavioral mechanism (verbose responses when uncertain) on adjacent QA tasks including the multi-hop bundle in LongBench; and `\citet{ielanskyi_addressing_2025}` show that approximate correctness functions disagree on UQ rankings.
  - ⚠ **`janiak_illusion_nodate`** → "The Illusion of Progress" (Janiak et al. EMNLP 2025). PDF present. Numbers (45.9% PPL drop, 30.4% EigenScore drop, on Mistral) are exact-match with `_lit-review.md` line 123: "Perplexity drops 45.9%, Eigenscore drops 30.4% from ROUGE to LLM-as-Judge labels". ✓ for content. ⚠ Year-tag should be `_2025`, not `_nodate`.
  - ✓ **`zhang_verbosity_2024`** → Y. Zhang et al. "Verbosity ≠ Veracity" (arXiv 2411.07858, ACL 2025 UncertainLP). PDF present. "Verbose-when-uncertain on adjacent QA tasks including the multi-hop bundle in LongBench" is byte-faithful with the paper's setup: LongBench bundles HotpotQA + MuSiQue + 2WikiMultihopQA (cf. `_lit-review.md` line 324). ✓
  - ✓ **`ielanskyi_addressing_2025`** → Ielanskyi et al. arXiv 2510.02279. PDF present. "Approximate correctness functions disagree on UQ rankings" matches their core finding (pairwise mutual AUROC as low as 0.53, Spearman ρ between rankings sometimes negative; cf. `_lit-review.md` lines 93–94). ✓
- [ ] **S4.** `\citet{vashurin_uncertainty-line_2025}` propose Uncertainty-Line, a regression-residualization protocol applied to Neural Machine Translation (NMT), summarization, and mathematical reasoning; dynamic-RAG triggers remain an untested application domain.
  - ✓ **`vashurin_uncertainty-line_2025`** → Vashurin et al. arXiv 2505.19060. PDF present. Application-domain claim: per `_lit-review.md` line 260, Vashurin's setup is "Llama 3.1 8B-Instruct. Tasks: neural machine translation (NMT), summarization, mathematical reasoning. NOT direct QA". **Exact match** with the audit sentence.
  - ✓ **"Dynamic-RAG triggers remain an untested application domain."** Defensible: Vashurin's pool is PPL, MSP, MTE, MCNSE, Token-SAR, Semantic Entropy — none of DRAGIN/FLARE/SeaKR/INSIDE/ETC/RSUS/SELFDOUBT-T1 appears in their evaluation. Per `_lit-review.md` line 267.

## intro.p3 — our contribution

- [ ] **S1.** We present the first length-controlled audit of the published training-free dynamic-RAG trigger pool under length-residualized AUROC against a paired-oracle target, `lift_em`=+1, which labels each closed-book trace by whether re-decoding under gold context flips a wrong answer to a right one `\citep{mallen_when_2023}`.
  - ✓ **"first" claim.** Already discussed at Abstract S3. Defensible.
  - ⚠ **`mallen_when_2023`** → Mallen et al. ACL 2023 "When Not to Trust Language Models" (arXiv 2212.10511). The paper proposes "Adaptive Retrieval" (retrieve when entity popularity is below threshold) for PopQA. **The paper does NOT formalize `lift_em` or paired-oracle re-decode.** It is the *conceptual* parent (it asks "when does retrieval help?") but the operationalization (binary 1[`EM(a_o) > EM(a_c)`]) is original to this paper. The audit sentence says `\citep{mallen_when_2023}` after "flips a wrong answer to a right one" — this is **citation-for-conceptual-credit**, not citation-for-operational-definition. Defensible as conceptual lineage cite, but **could be strengthened** to "conceptually motivated by `\citet{mallen_when_2023}`" rather than the bare `\citep`. ⚠ Reviewer-risk: medium.
- [ ] **S2.** The paired design shares the same exact-match correctness function across both arms, which partially cancels the function-bias critique of `\citet{ielanskyi_addressing_2025}`.
  - ✓ **Mechanism check.** `r1_paired_audit.py` line 64–66: `c_em = int(bool(c.get("label_correct_em", c.get("em", False)))); o_em = int(bool(o.get("label_correct_em", o.get("em", False)))); lift = +1 if o_em > c_em else …`. Both arms use the same EM check on the same `label_correct_em` field. Byte-faithful.
  - ✓ **"Partially cancels" function-bias critique.** Ielanskyi's critique is that correctness-function choice changes UQ rankings. In paired-oracle, both arms share EM (the same function), so the function-bias *is* partially absorbed via shared bias. Defensible argument, matches `_lit-review.md` line 105.
- [ ] **S3.** We propose **FCT** (first-commitment time), a forward-looking variant of the answer-trajectory primitive `\citep{datta_large_2026, wang_chain--probe_nodate, chen_i_2026, zhang_commit_2026}`; the operational distinction from `\citeauthor{zhang_commit_2026}`'s concurrent commitment-time primitive (which projects the continuation distribution onto a finite answer set) is detailed in App. fct-tstar.
  - ✓ **`datta_large_2026`** → Datta et al. (`AnswerSwitches`, t* primitives). Already verified.
  - ⚠ **`wang_chain--probe_nodate`** → Wang Chain-of-Probe. ⚠ key fix needed (already noted).
  - ✓ **`chen_i_2026`** → Chen Dynamic Instability. Already verified.
  - ✓ **`zhang_commit_2026`** → "When Does a Language Model Commit?" (arXiv 2605.06723). Verified via web search; the description "projects the continuation distribution onto a finite answer set" matches the paper's abstract: "project the model's own continuation probabilities onto a finite answer set; in binary tasks this yields an exact log-odds code." ✓ **Operational distinction.** Zhang's commit-time is on hidden state-space log-odds (finite-set projection), forward-looking but on continuation distribution. Our FCT is on K=5-majority answer extractions, also forward-looking but on textual answer-set. Distinct primitives; App. fct-tstar Table 5 tabulates this on 5 axes. **Distinction is genuine and material**, not a renaming.
  - ✓ **Cross-reference** App. fct-tstar = `sec:appendix-fct-tstar` line 419. ✓
- [ ] **S4.** FCT is the strongest length-robust trigger in the audited pool, exceeding the length baseline by Δ = +0.190 AUROC (Bonferroni-significant at k=15).
  - ✓ Numbers match `e7_bonferroni.out` (already verified). Bonferroni-significance = `✓✓` column. ✓
- [ ] **S5.** On the held-out 2Wiki cells, length is itself anti-predictive of retrieval need, inverting the verbosity-compensation pattern documented on adjacent QA tasks.
  - ✓ Already verified at Abstract S6. Tokens raw AUROC 0.457 (Qwen×2Wiki) / 0.518 (Gemma×2Wiki); both ≤ 0.52 and Qwen×2Wiki is strictly < 0.5 (anti-predictive). ✓

## intro.contrib — Contributions itemize

- [ ] **I1.** A length-controlled audit of the published training-free dynamic-RAG trigger pool (the first to our knowledge), identifying four length-robust signals from three signal families (attention, token-probability, and the emerging answer-trajectory family) not previously contrasted under the length-residualized AUROC protocol.
  - ✓ All numerical / family-counting claims already verified at Abstract S4.
  - ⚠ Slight wording inconsistency: Intro lists "three signal families (attention, token-probability, and the emerging answer-trajectory family)" — survives 4 = 1 attention + 1 token-prob + 2 answer-trajectory. Internally consistent with the 4-signal breakdown. ✓ However, the *9-signal* survivor set spans **five** families (per §4.1 S3) — not three. The "three families" in I1 refers specifically to the **Bonferroni-robust** four signals. The sentence parses correctly given context but a careless reader might conflate "four signals" with "three families that the *nine* signals span". Defensible; possibly worth a hedge.
- [ ] **I2.** We propose **FCT** (first-commitment time), a forward-looking variant of the answer-trajectory primitive, exceeding the length baseline by Δ = +0.190 [+0.093, +0.287] AUROC (Bonferroni lower bound +0.047) and reaching raw AUROC **0.784** on the held-out Gemma×2Wiki cell.
  - ✓ **Bonferroni lower bound +0.047.** `e7_bonferroni.out` line 26: FCT 99.67% CI = `[+0.047, +0.329]`. Exact match.
  - ✓ **Raw AUROC 0.784 on Gemma×2Wiki.** `moonshot_explore.out` line 125 (PER-CELL breakdown): `Gemma_wiki 200 24 ... FCT 0.784`. Exact match.
- [ ] **I3.** Cross-architecture and cross-domain validation on one dense (Qwen2.5-7B) and one MoE-hybrid (Gemma-4-26B-A4B-it) open-weight chat model over two multi-hop QA benchmarks (n=800 paired traces; base rate 13.5%): nine signals clear length at 95% paired-bootstrap and four survive Bonferroni at k=15.
  - ✓ **Architecture labels.** Qwen2.5-7B-Instruct (dense) and Gemma-4-26B-A4B-it (MoE-hybrid). Architecture classification:
    - Qwen2.5-7B-Instruct is dense (28 layers, GQA, 7B active params). ✓
    - Gemma-4-26B-A4B-it = Gemma 4 26B with 4B-active MoE-hybrid arch. ✓ (Manik & Wang 2026 PDF in `references-recollections-v2/` describes the Gemma 4 / Phi 4 / Qwen 3 dense-vs-MoE tradeoffs.)
  - ✓ **n=800 paired, base rate 13.5%.** `e7_bonferroni.out` line 2: `n=800 pop=108 base=0.135`. 108/800 = 0.135. Match.
- [ ] **I4.** A direct robustness check against the LLM-as-judge critique of `\citet{janiak_illusion_nodate}`: under a GPT-4o-Mini semantic-equivalence judge agreeing with EM at Cohen's κ = 0.707, eight of nine paired-significant survivors and all four Bonferroni-robust signals are preserved; FCT strengthens to Δ = +0.222.
  - ✓ All numerical claims verified at Abstract S7.
  - ✓ **"All four Bonferroni-robust signals preserved."** Cross-check: in `llm_judge_audit_both.out` under lift_em_judge, the 4 Bonferroni-robust signals (DRAGIN-RIND, ent.max, min_majority, FCT) all show `✓` at 95%. Matches. (Note the table only reports 95% CIs under the judge, not 99.67%; "preserved" = nominal-95% under judge, which is the audit's claim. ✓)
  - ⚠ **"Janiak's exact choice"** of GPT-4o-Mini is the verifiable framing. From `_lit-review.md` line 123: Janiak uses "LLM-as-Judge (GPT-4o-Mini) labels" — match.

---

# 2. Related Work

- [ ] **S1.** Length confounds the standard AUROC protocol for LLM uncertainty quantification `\citep{santilli_revisiting_2025, vashurin_uncertainty-line_2025, janiak_illusion_nodate, zhang_verbosity_2024, ielanskyi_addressing_2025}`; `\citet{mao_confidence_2026}` note that joint generation probability suffers severe length bias.
  - ✓ Five citations already verified in §1 p2.
  - ✓ **`mao_confidence_2026`** → Mao et al. arXiv 2601.13387. PDF present. Per `_lit-review.md` line 194: Mao "DO acknowledge length bias explicitly in Section 2: 'joint generation probability suffers from severe length bias, assigning smaller values to longer responses regardless of correctness, making it unsuitable for long-form reasoning'." Quoted phrase is **byte-faithful** with the Mao paper. ✓ Strong support.
- [ ] **S2.** Step-level answer-trajectory triggers are an emerging length-orthogonal signal class `\citep{datta_large_2026, wang_chain--probe_nodate, chen_i_2026, zhang_harnessing_2026, lu_streaming_2026}`.
  - ✓ **`datta_large_2026`**, **`chen_i_2026`**, **`wang_chain--probe_nodate`** verified above.
  - ✓ **`zhang_harnessing_2026`** → J. Zhang et al. "Harnessing Reasoning Trajectories for Hallucination Detection via Answer-agreement Representation Shaping" (ICML 2026). PDF present.
    - ⚠ **Terminology collision risk.** Per `_lit-review.md` lines 371–375: J. Zhang's "answer stability" = robustness to *latent perturbations* at trace boundary; our `flip_rate` / AID family = step-to-step disagreement under K=5 *boundary-completion sampling*. Two different operationalizations sharing a name; the audit document `_lit-review.md` line 375 calls this out as **mandatory disambiguation in §2**. Our §2 S2 does NOT explicitly disambiguate the answer-stability terminology overlap. ⚠ **Reviewer-risk: medium.** Consider adding a half-sentence parenthetical: "(distinct from the latent-perturbation 'answer agreement' framing of `\citet{zhang_harnessing_2026}`, which trains a contrastive embedding shaper rather than reading the textual answer trajectory directly)."
  - ✓ **`lu_streaming_2026`** → Lu et al. "Streaming Hallucination Detection in Long Chain-of-Thought Reasoning" (arXiv 2601.02170). PDF present.
    - ⚠ **Categorical fit.** Lu trains step-level + prefix-level hidden-state *probes* (supervised with Claude-4.5 labels); they are NOT parameter-free / training-free decode-time signals. Citing Lu as a "step-level answer-trajectory trigger" is loose: their primitive is hidden-state, not answer-trajectory. Per `_lit-review.md` lines 161, 169: "Their family is supervised representation-shaping; ours is parameter-free decode-time"; "Out-of-family but conceptually parallel." Defensible if framed as "emerging…signal class" (broad family), but **lumping a supervised method into 'training-free trigger' family-bibliography is misleading.** ⚠ Reviewer-risk: medium-high; consider rephrasing to "Step-level signals — both training-free trajectory primitives `\citep{datta_large_2026, wang_chain-of-probe_2025, chen_i_2026}` and supervised hidden-state probes `\citep{zhang_harnessing_2026, lu_streaming_2026}` — are an emerging length-orthogonal signal class."
- [ ] **S3.** The training-free trigger pool we audit spans attention-based DRAGIN `\citep{su_dragin_2024}` and DR-RAG `\citep{hei_dr-rag_2024}`, token-probability FLARE `\citep{jiang_active_2023}`, multi-sample text-consistency RSUS `\citep{guo_realm_2026}` (with related approaches SelfCheckGPT `\citep{manakul_selfcheckgpt_2023}` and SAC³ `\citep{zhang_sac3_2023}`), hidden-state INSIDE `\citep{chen_inside_2024}` and SeaKR `\citep{yao_seakr_nodate}`, hedging-marker SELFDOUBT `\citep{pandey_selfdoubt_2026}`, and entropy-trend ETC `\citep{li_modeling_2026}`.
  - ✓ **`hei_dr-rag_2024`** → DR-RAG (Hei et al. arXiv 2406.07348). Verified via web search. ⚠ **But DR-RAG is NOT in our audited 16-signal pool** (see `signal_lib.py` SIGNALS list lines 299–318 and `e7_bonferroni.out` rows). DR-RAG is a two-stage retrieval framework, not a trigger signal. Citing it in S3 as part of the "training-free trigger pool we audit" is **misleading** — it suggests we benchmark DR-RAG, which we do not. **Reviewer-risk: medium.** Consider removing the DR-RAG citation or moving it to a separate sentence ("…related dynamic-RAG frameworks such as DR-RAG…").
  - ✓ **`manakul_selfcheckgpt_2023`** → SelfCheckGPT (Manakul et al. EMNLP 2023). PDF present. Cited as related to RSUS (cross-sample consistency family). ✓
  - ✓ **`zhang_sac3_2023`** → SAC³ (J. Zhang et al. EMNLP Findings 2023). Verified via web search. ✓ Note this is yet another Zhang first-author (Jiaxin Zhang, distinct from Y. Zhang verbosity-2024 and J. Zhang ARS-2026).
  - ✓ **Family enumeration** otherwise matches §1 p1 S3 directly. Consistent.
- [ ] **S4.** Trained alternatives that require per-architecture supervised data `\citep{asai_self-rag_2023, tang_mba-rag_2024, chen_think_2026, guo_dior_2025}` fall outside our training-free scope.
  - ✓ **`asai_self-rag_2023`** → Self-RAG (Asai et al. arXiv 2310.11511). PDF present. Requires supervised data for reflection-token training. ✓
  - ✓ **`tang_mba-rag_2024`** → MBA-RAG (Tang et al. COLING 2025). Verified via web search. ⚠ Year-tag drift: arXiv 2024 → venue 2025; either OK.
  - ✓ **`chen_think_2026`** → Chen et al. "Think Deep, Not Just Long" (PDF present). Trained method? **Need to verify.** Looking at PDF context from `_lit-review.md` (line 487 not directly): the audit document does not have a long capsule. **Risk: unclear if Chen-Think-Deep is actually trained.** Let me flag and check during methods audit. ⚠
  - ✓ **`guo_dior_2025`** → DioR (H. Guo et al. ACL 2025). Verified via web search. Trains two hallucination detectors (early-detection classifier + real-time-detection classifier). ✓ trained.
  - ⚠ **Cite-discrimination check.** `_lit-review.md` line 376–377: "Trained alternatives that require per-architecture supervised data" — Lu_StreamingHallucinationDetectionCoT, Zhang_AnswerAgreementRepresentationShaping are ALSO trained (Lu trains a probe, Zhang trains a contrastive mapping). Yet they appear in §2 S2 (answer-trajectory family) not §2 S4 (trained alternatives). **Categorization is inconsistent.** Reviewer might ask why Lu/Zhang_ARS aren't in S4. Possible defense: S4 is about *trigger* alternatives (Self-RAG, MBA-RAG, DioR retrieve by trained policy), while Lu/Zhang_ARS are *detection* methods. But that distinction is not explicit in the text. ⚠ Reviewer-risk: medium. Worth a hedge.
- [ ] **S5.** The conceptual parent of our paired-oracle target is `\citet{mallen_when_2023}`, who asks when retrieval helps a parametric model.
  - ✓ **`mallen_when_2023`** already verified. ✓ Conceptual parent claim is defensible. ✓

---

# 3. Methods

## 3.1 Paired-Oracle `lift_em` Protocol (`methods.protocol`)

- [ ] **S1.** For each question q in the evaluation set, we run two greedy decodes on the same chat model: a closed-book decode that produces trace τ_c^(q) and answer a_c^(q) with no retrieval context, and an oracle-retrieval decode conditioned on the gold supporting paragraphs that produces a_o^(q).
  - ✓ **Two greedy decodes per question.** Trace generation code at `emnlp_26_lengthConfound/scripts/04_run_qa.py` and `04l_oracle_retrieve_decode.py` (oracle pass). Files exist at `scripts/`.
  - ✓ **Trace dirs.** Confirmed file structure: `runs/v2_headline_{arch}/{D1_closed_hotpot, D2_closed_wiki, E1_oracle_hotpot, E2_oracle_wiki}.jsonl`. D = closed (D1=HotpotQA, D2=2Wiki); E = oracle (E1=HotpotQA, E2=2Wiki). Naming is byte-faithful with `r1_paired_audit.py` line 189–191.
- [ ] **S2.** We extract all trigger signals from τ_c^(q); the oracle decode only labels the trace.
  - ✓ **`r1_paired_audit.py` line 67–94**: `closed = {r["id"]: r for r in ...}; oracle = {r["id"]: r for r in ...}`. The `oracle[qid]` row is used only to extract `o_em = int(bool(o.get("label_correct_em", ...))); lift = +1 if o_em > c_em else -1 if o_em < c_em else 0`. All 17 trigger-signal extractors operate on `c.get("steps") or []` from the closed-book row. Byte-faithful with the audit claim.
- [ ] **S3.** The discrimination target is **[Eq. 1] `lift_em(q) = 1[EM(a_o^(q)) > EM(a_c^(q))]`**.
  - ✓ Definition byte-faithful with `r1_paired_audit.py` line 66 (already verified at Abstract S3).
- [ ] **S4.** *i.e.*, a trace receives label +1 exactly when the closed-book answer is wrong and the gold-context answer is right --- precisely the subpopulation where retrieval would convert a failure into a success.
  - ✓ Logical paraphrase of S3 / Eq. 1. ✓
- [ ] **S5.** This is a strictly different question from the single-arm "which traces are wrong?" question that the published trigger pool actually evaluates against (and that length predicts moderately well on multi-hop QA; §discussion-causal): we instead ask, of the queries that the model could not solve unaided, which ones would retrieval rescue?
  - ✓ **Cross-ref.** `§discussion-causal` → `sec:discussion-causal` line 254. Resolves.
  - ✓ **"Length predicts moderately well on multi-hop QA"** — supported by §5.1 S1 claim of "`tokens_generated` reached raw AUROC 0.539 on `closed_em`=0 headline pool" — moderate but real. ✓
- [ ] **S6.** A signal that ranks `lift_em`=+1 traces high is, by construction, a retrieval trigger --- thresholding it at a high percentile recovers exactly the retrieval-helpful subpopulation.
  - ✓ Definitional / framing claim. Defensible by AUROC's interpretation as ranking-quality.
- [ ] **S7.** The paired design shares the same exact-match correctness function across both arms, which partially insulates against the function-bias critique of `\citet{ielanskyi_addressing_2025}`.
  - ✓ Already verified at Intro p3 S2.
- [ ] **S8.** We evaluate on HotpotQA `\citep{yang_hotpotqa_2018}` and 2WikiMultiHopQA `\citep{ho_constructing_2020}` --- the latter held out for cross-dataset evaluation --- at n=200 paired traces per architecture-by-dataset cell.
  - ✓ **`yang_hotpotqa_2018`** verified above.
  - ✓ **`ho_constructing_2020`** verified above.
  - ✓ **n=200 per cell.** `e7_bonferroni.out`: 4 cells × n=200 = 800. `moonshot_explore.out` line 122–125 shows per-cell n=200. ✓
- [ ] **S9.** We select one dense (Qwen2.5-7B-Instruct) and one MoE-hybrid (Gemma-4-26B-A4B-it) open-weight chat model to test cross-architecture generality within a single 80 GB GPU compute budget.
  - ✓ Already verified at I3.
  - ✓ **"80 GB GPU"** budget claim: App. repro S6 mentions "single NVIDIA A800 80 GB". Consistent.
- [ ] **S10.** The positive class is rare (108 of 800 paired traces, base rate 13.5% across the headline pool).
  - ✓ `e7_bonferroni.out` line 2: `n=800 pop=108 base=0.135`. 108/800 = 0.135. ✓ exact match.

## 3.2 Length Residualization (`methods.resid`)

- [ ] **S1.** For each trigger signal s on a cohort of paired traces, we compute the length-residualized score **[Eq. 2] `s_resid = s − α̂ − β̂ · log(tokens + 1)`**.
  - ✓ **`signal_lib.py` lines 245–263 `ols_residual()`** computes `r[key] − (a + b * r[x_key])` where `x_key="log_tokens"` and `log_tokens = math.log(max(1, tokens) + 1)` (set at `r1_paired_audit.py` line 78). Byte-faithful with Eq. 2.
- [ ] **S2.** where α̂, β̂ are OLS coefficients fit on the cohort.
  - ✓ Same code block; `b = num / den if den > 0 else 0.0; a = my - b * mx`. Standard OLS one-variable formula. ✓
- [ ] **S3.** AUROC(s_resid) measures whether the signal carries information beyond length.
  - ✓ Theoretical motivation; matches the spirit of Vashurin Uncertainty-Line. ✓
- [ ] **S4.** We use log(tokens+1) rather than raw length L because multi-hop reasoning traces have a long-tailed token distribution; a linear-in-log L regression is the natural homoscedastic counterpart of `\citeauthor{vashurin_uncertainty-line_2025}`'s linear-in-L regression on their shorter NMT and summarization traces.
  - ✓ **`\citeauthor{vashurin_uncertainty-line_2025}`** resolves to "Vashurin et al." Per `_lit-review.md` line 263: "we adopt and extend the UNCERTAINTY-LINE protocol of Vashurin et al. (2025)" — the linear-in-L parameterization in Vashurin is real (their §3 propose `\hat u_LINE = \hat u - \hat f(L)` with `f` linear in raw L).
  - ⚠ **"shorter NMT and summarization traces"** — `_lit-review.md` lines 260: Vashurin tasks are NMT, summarization, math; **NMT outputs tend short** (sentence-level), **summarization tends mid-length**, **math reasoning is long**. The "shorter" framing is roughly defensible for NMT/summarization but math is comparable in length to multi-hop QA. Defensible but consider hedging to "their generally shorter NMT/summarization traces (math reasoning excepted)."
- [ ] **S5.** Both parameterizations agree within ε ≤ 0.029 AUROC against the public Uncertainty-Line reference implementation on both architectures (Appendix uline); higher-order polynomial regressors give no further improvement, consistent with `\citeauthor{vashurin_uncertainty-line_2025}`'s App. D.1 finding that the linear fit is sufficient.
  - ✓ **ε ≤ 0.029.** Confirmed against both `uncertainty_line_audit_*.out`:
    - Qwen: `ε_protocol: ε_max=0.0287` (line 59). ✓
    - Gemma: `ε_protocol: ε_max=0.0270` (line 59). ✓
    - Both ≤ 0.029. ✓ exact-match.
  - ✗ **"Higher-order polynomial regressors give no further improvement"** — **CONFIRMED NOT EMPIRICALLY TESTED IN OUR CODE.** I searched all of `emnlp_26_lengthConfound/analysis/` for `polynomial`, `poly`, `degree` and found **zero** hits in our analysis scripts. The reference-port at `emnlp_26_lengthConfound/externalCodeAudits/uncertainty-line/utils.py` lines 343–346 DOES use `PolynomialFeatures(degree={1,2,3})`, but our own `analysis/uncertainty_line_audit.py` (lines 103–111) only fits a **single** `LinearRegression` on `log_len`. **The sentence reads as if we tested polynomial regressors on our data; we did not.** The supporting evidence is Vashurin's App. D.1 finding only, which we inherit by adopting the linear choice. ⚠⚠ **Reviewer-risk: medium-high.** Recommended fix: reframe as "we adopt the linear fit, in line with Vashurin et al.'s App. D.1 finding that higher-order polynomial regressors give no further improvement" — explicit inheritance rather than internal-experiment-implication.
  - ✓ **"Vashurin App. D.1"** — confirmed present: the Vashurin PDF object index lists `subsection.D.1` (and `subsection.D.2`) per grep of the PDF stream. Cross-ref exists. ✓ Content claim (polynomial-no-improvement) is consistent with `_lit-review.md` summary. Defensible.
- [ ] **S6.** ETC, FLARE, the SeaKR primitive, and SELFDOUBT-T1 are evaluated as trace-level summary statistics rather than online triggers; granularity adapters relative to the originating papers are detailed in Appendix adapters.
  - ✓ **Adapter behaviors verified in code:**
    - **ETC** = `signal_lib.py` line 69 `etc_score()`: max absolute second-difference of step entropy → trace-level summary scalar. Original paper operates on token-level entropy stream → we evaluate on per-step mean entropy series at boundary-detection events. Adapter is real, documented in App. adapters `G.etc`. ✓
    - **FLARE** = `signal_lib.py` line 89 `flare_max_per_step()`: per-step `1 - min(per_token_max_logprob)`, then `max` over steps → trace-level summary. Original paper trigger is online per-token. Adapter real, documented in App. adapters `G.flare`. ✓
    - **SeaKR** = `signal_lib.py` line 107 `seakr_k5_logdet()`: per-step Gram-determinant of K=5 EOS hidden states, then `max` over steps → trace-level summary. Original paper uses Gram-determinant once at trace EOS. **Note:** original paper uses K ∈ [10, 25] (best-k); we use K=5. Adapter real, documented in App. adapters `G.seakr`. ✓
    - **SELFDOUBT-T1** (HVR) = `signal_lib.py` line 84 `hvr_max()`: max over steps of `hvr` per step. Original paper computes a single trace-level HVR with Tier-1 gate. Adapter real, documented in App. adapters `G.selfdoubt`. ✓
  - ✓ Cross-ref `App.~\ref{sec:appendix-adapters}` line 480.

## 3.3 FCT and the Answer-Trajectory Family (`methods.fct`)

- [ ] **S1.** At each reasoning-step boundary (detected by spaCy sentence segmentation with a 32-token fallback), we sample K=5 boundary completions at temperature τ=0.6, top-p 0.9, and extract per-step answers via a regex over the model's `Answer:` prefix.
  - ? **spaCy sentence segmentation + 32-token fallback** — these are decode-time choices in `scripts/04_run_qa.py`; need to verify against scripts. Will check.
  - ✓ **Answer regex.** `signal_lib.py` line 40: `ANSWER_RE = re.compile(r"\bAnswer\s*:\s*(.+?)(?:[\n.]|$)", re.IGNORECASE)`. Byte-faithful with the paper claim.
  - ? **K=5, τ=0.6, top-p 0.9** — decode-time params, need to verify by reading decode script.
- [ ] **S2.** From the resulting trajectory M = (m_1, ..., m_n) of per-step K=5 majority answers, we define four trace-level aggregations:
  - ✓ **`per_step_answers()`** at `signal_lib.py` lines 155–165 returns per-step list-of-K answers. Downstream functions aggregate to majority/set/etc.
- [ ] **I1.** `flip_rate`: fraction of consecutive-step majority disagreements;
  - ✓ `flip_rate_majority` at `signal_lib.py` lines 168–177: takes per_step ans list, builds `seq` = list-of-majority-strings, returns `flips / max(1, len(seq) - 1)` where `flips = sum(... seq[i] != seq[i-1] ...)`. **Exactly fraction of consecutive-step majority disagreements.** ✓
- [ ] **I2.** `late_flips`: count of consecutive-step disagreements between the per-step K=5 answer *sets* (any change in the multiset of sampled answers, not only majority shifts), restricted to the last half of the trace; we use `late_flips` as a raw-incidence counterpart to the fraction-based `flip_rate`;
  - ⚠ **`late_flips_count()`** at `signal_lib.py` lines 192–205. The code does:
    1. Build `(i, set(ans))` for each populated step (`answer_steps`).
    2. `mid = n_steps_total // 2`.
    3. `late = [(i, s) for i, s in answer_steps if i >= mid]`.
    4. Return `sum(1 for j in range(1, len(late)) if late[j][1] != late[j-1][1])`.
  - The paper says "any change in the multiset of sampled answers" — but the code uses `set(ans)`, which collapses multiset → set (loses duplicate counts). **Discrepancy:** code is set-based (loses multiplicity), paper claim is "multiset of sampled answers (any change in the multiset)". For K=5, the difference matters when e.g. step t has `[a, a, a, b, b]` and step t+1 has `[a, a, b, b, b]` — same set `{a, b}` (no late_flip recorded), but different multiset (would-be flip in the paper claim). **Reviewer-risk: medium.** Either fix code to use `Counter(ans)` and compare via `==`, or weaken paper claim to "any change in the set of sampled answers".
  - ✓ **"restricted to the last half of the trace"** — `mid = n_steps_total // 2`, then `i >= mid` gates inclusion. Correct half-restriction.
- [ ] **I3.** `min_majority`: 1 − min_t (max_count/K), the worst within-step disagreement;
  - ✓ **`min_majority_inv()`** at `signal_lib.py` lines 208–218: for each populated step, `c = Counter(ans).most_common(1)[0][1]; fracs.append(c / len(ans))`; returns `1.0 - min(fracs)`. Byte-faithful with the paper formula `1 - min_t (max_count/K)`. ✓
- [ ] **I4.** **FCT**, a forward-looking first-commitment time **[Eq. 3] `FCT = min{i : m_j = a_final ∀ j ≥ i} / max(1, n−1) ∈ [0, 1]`**.
  - ✓ **`fct_score()`** at `moonshot_explore.py` lines 141–169: searches smallest `k` such that `all(m == final_ans_n for _, m in majorities[k:])`; returns `t_commit / (n - 1)` where `t_commit = n - 1` default. Byte-faithful with Eq. 3.
  - ⚠ **Range check.** Code returns `0.0` if `n - 1 == 0` (line 167–168) and `t_commit / (n - 1)` otherwise. In the worst case (no convergence found, `for k in range(n): break` never hits), the default `t_commit = n - 1` survives → returns `(n-1)/(n-1) = 1.0`. Range `[0, 1]` is correct. ✓
- [ ] **S3.** High FCT denotes extended deliberation: the trace has not converged on its final answer until late in the decode.
  - ✓ Interpretation consistent with the math: late commit → high t_commit → high FCT. ✓
- [ ] **S4.** These aggregations adapt the `AnswerSwitches` primitive of `\citet{datta_large_2026}` to non-thinking-mode chat models; Appendix fct-tstar tabulates the operational differences from Datta's t* and `\citeauthor{zhang_commit_2026}`'s concurrent commitment-time primitive.
  - ✓ **Lineage.** Datta's `AnswerSwitches = Σ 1[A_i ≠ A_{i+1}]` is the per-step transition count primitive; our `flip_rate` is exactly this normalized by trace length; `late_flips` is a windowed variant; `min_majority` and `FCT` are different aggregations on the same underlying trajectory. **All four aggregations adapt the Datta primitive.** Defensible.
  - ✓ **"Non-thinking-mode chat models"** — Datta operationalizes the primitive via `</think>` + answer prefix in thinking-mode reasoning models (Qwen3-4B thinking-mode). We use spaCy-sentence boundaries + `Answer:` prefix in non-thinking-mode chat models (Qwen2.5-7B / Gemma-4-26B). Adaptation is real. ✓
  - ✓ Cross-ref `App.~\ref{sec:appendix-fct-tstar}` line 419. Table 5 tabulates 5 axes. ✓
- [ ] **S5.** A K=1 vs. K=5 ablation of `flip_rate` is null on both architectures (§results-ablation), and we recommend K=1 for deployment efficiency.
  - ✓ Already verified at I3-like content; supporting data in `a1_k1_vs_k5_r1_qwen.out` line 18: `OVERALL 305 47 ... Δraw -0.002 [-0.040, +0.031]` — straddles zero. And `a1_k1_vs_k5_r1_gemma.out` line 18: `OVERALL 394 42 ... Δraw +0.033 [-0.025, +0.092]` — straddles zero. Both null at 95%. ✓ exact match.

### 3.3.x CABV paragraph (`methods.cabv`)

- [ ] **S1.** We additionally compute **CABV**, a continuous between-step generalization of `flip_rate`: at each populated step t, build the Laplace-smoothed empirical distribution p_t over the union of K=5 answers ever observed in the trace, and average the Jensen–Shannon divergence between consecutive populated steps, **[Eq. 4] `CABV = (1/(|P|−1)) Σ_t JSD(p_t, p_{t+1})`**.
  - ✓ **`cabv_score()`** at `moonshot_explore.py` lines 75–109:
    - `universe = sorted({a for _, ans in populated for a in ans})` — union of all answers in trace. ✓
    - `v = np.full(K_uni, eps, dtype=float); for a in ans: v[idx[a]] += 1.0; v = v / v.sum()` — Laplace-smoothed (eps=1e-3) empirical distribution per step. ✓
    - `deltas = [jsd(dists[i], dists[i + 1]) for i in range(len(dists) - 1)]; return float(np.mean(deltas))` — mean JSD between consecutive populated steps. **Byte-faithful with Eq. 4.** ✓
- [ ] **S2.** where P is the set of populated step indices.
  - ✓ `populated = [(i, ans) for i, ans in enumerate(per_step) if ans]`. ✓
- [ ] **S3.** CABV is high on traces whose answer-belief mass continues to redistribute through the decode; it reduces to `flip_rate` when each step is a delta on a single answer, and is conceptually closest to the inference-time instability signal of `\citet{chen_i_2026}` (which uses JSD over token distributions rather than over K=5 answer distributions).
  - ✓ **Mathematical reduction claim**: when K=1 with a single answer per step (delta distribution on one element of the universe), `JSD(δ_a, δ_a) = 0` if same; `JSD(δ_a, δ_b) = log(2) > 0` if different. So CABV becomes log(2) × fraction-of-consecutive-step-disagreements = log(2) × flip_rate. **Up to a constant.** Defensible as "reduces to flip_rate (up to scaling)" or weaken to "behaves like flip_rate when each step is a delta on a single answer."
  - ✓ **`chen_i_2026`** verified — Chen et al. "I May Not Have Articulated Myself Clearly: Diagnosing Dynamic Instability" uses `D_t = JSD(\tilde p_t, \tilde p_{t-1})` over top-k token distributions. Their JSD is over per-step token distributions; ours is over K=5 answer empirical distributions. **Conceptually closest claim is defensible.** ✓

---

# 4. Results

## 4.1 Cross-Architecture Survivor Matrix (`results.matrix`)

- [ ] **S1.** Nine signals clear length at 95% and four survive Bonferroni at k=15 across three signal families (Figure survivor-forest): attention-based DRAGIN-RIND (Δ = +0.167), token-probability `ent.max` (+0.137), answer-trajectory `min_majority` (+0.149), and the forward-looking commitment-time signal **FCT** (+0.190, the strongest length-robust trigger in the audited pool).
  - ✓ All four numbers + family attributions exact-match `e7_bonferroni.out` (already verified at Abstract S4–S5). ✓
- [ ] **S2.** Confidence intervals are paired-bootstrap at nominal 95% and Bonferroni-corrected ~99.67% (α=0.0033, B=10,000).
  - ✓ Already verified at Figure 1 caption S3.
- [ ] **S3.** The full nominal-95% set of nine survivors spans five of the seven declared families (attention, token-probability, entropy-trend, text-consistency, answer-trajectory); hidden-state and hedging-marker (SELFDOUBT-T1) have no nominal-95% survivor.
  - ✓ **9 nominal-95% survivors**: DRAGIN-RIND (attention), ent.max (token-prob), etc (entropy-trend), u_cons.max (text-consistency), flip_rate K=5 (answer-trajectory), flip_rate K=1 (answer-trajectory), min_majority (answer-trajectory), CABV (answer-trajectory), FCT (answer-trajectory). Family count: **5** = {attention, token-prob, entropy-trend, text-consistency, answer-trajectory}. ✓ exact match.
  - ✓ **Hidden-state survivor count = 0** at 95%: `e7_bonferroni.out` shows `seakr_k5_proxy.logdet`, `drift_cosine.max`, `var_l2_window.max`, `hidden_norm.range` all at `—` (not significant). ✓
  - ✓ **Hedging-marker SELFDOUBT-T1 / hvr.max = 0 at 95%**: `e7_bonferroni.out` line 20: `hvr.max 0.505 +0.032 [-0.089,+0.150] [-0.147,+0.210] —`. ✓
  - ⚠ **"seven declared families"** — declared in §1 p1 S3 (6) + S4 (answer-trajectory = 7th). The 9-survivor set covers 5 of 7, with hidden-state and hedging-marker being the 2 missing. ✓ accounting balances.
- [ ] **S4.** The full table of all 16 signals is in Appendix hero.
  - ✓ Cross-ref to App. `sec:appendix-hero` line 326. Table `tab:hero-full` lists 16 signals + tokens baseline (= 17 rows). ✓ matches.

## 4.2 Length Is Anti-Predictive on the Held-Out 2Wiki Cells (`results.inverted`)

- [ ] **S1.** On the held-out 2Wiki cells (Qwen × 2Wiki and Gemma × 2Wiki), raw answer length is *anti*-predictive of retrieval need: the `tokens_generated` baseline reaches raw AUROC **0.457** and **0.518** on `lift_em`=+1, both at or below chance.
  - ✓ **0.457 on Qwen×2Wiki**: `uncertainty_line_audit_qwen2.5-7b-instruct.out` line 42 `tokens_generated (sanity) 0.457`. ✓
  - ✓ **0.518 on Gemma×2Wiki**: `uncertainty_line_audit_gemma-4-26b-a4b-it.out` line 42 `tokens_generated (sanity) 0.518`. ✓ exact match.
- [ ] **S2.** Longer traces on the held-out multi-hop cells are systematically *less* retrieval-helpful, inverting the verbosity-compensation pattern `\citet{zhang_verbosity_2024}` document on adjacent QA tasks.
  - ✓ **Inversion claim.** Zhang Verbosity documents longer-when-uncertain → more wrong. On `closed_em=0` target (single-arm correctness) we see the *same* direction: `tokens_generated` raw AUROC 0.539 (§5.1 S1). On `lift_em=+1` (paired-oracle retrieval-helpfulness), we see anti-predictive 0.457 (Qwen×2Wiki) and 0.518 (Gemma×2Wiki). **The inversion is real and load-bearing for the framing.**
  - ⚠ Hedge wording: "Longer traces … are systematically *less* retrieval-helpful" — strong claim. The Gemma×2Wiki number is 0.518 which is *above* chance, just below 0.52. The "anti-predictive" framing fits Qwen×2Wiki cleanly (0.457 < 0.5) but on Gemma×2Wiki it is **slightly pro-predictive but below the length-confound threshold**. The aggregated claim "at or below chance" is precise and defensible; the stronger paraphrase in S2 ("systematically *less*") might mislead a careless reader. ⚠ Reviewer-risk: medium. Consider hedging S2 to "Longer traces on the held-out cells are no longer reliably retrieval-helpful, inverting the verbosity-compensation pattern…"
- [ ] **S3.** The four Bonferroni-robust survivors clear the length baseline by a wider margin on these cells than on the HotpotQA cells, suggesting the raw-AUROC protocol miscredits the length signal on exactly the cells where retrieval matters most.
  - ❓ **"Wider margin on 2Wiki cells than HotpotQA cells"** — need to verify by per-cell breakdown of the four signals (DRAGIN-RIND, ent.max, min_majority, FCT) vs `tokens_generated`.
    - From `moonshot_explore.out` per-cell breakdown: Qwen-Hot DRAGIN-RIND raw 0.582, tokens 0.572 → Δ_raw +0.010 on HotpotQA. Qwen-Wiki DRAGIN-RIND raw 0.606, tokens 0.457 → Δ_raw +0.149 on 2Wiki. **2Wiki margin (0.149) > HotpotQA margin (0.010).** ✓ for DRAGIN.
    - Qwen-Hot ent.max raw 0.544 vs tokens 0.572 → -0.028. Qwen-Wiki ent.max 0.558 vs 0.457 → +0.101. **2Wiki much wider margin.** ✓
    - For FCT: Qwen-Hot 0.677 vs 0.572 → +0.105. Qwen-Wiki 0.604 vs 0.457 → +0.147. **2Wiki marginally wider.** ✓
    - For min_majority: Qwen-Hot 0.659 vs 0.572 → +0.087. Qwen-Wiki 0.619 vs 0.457 → +0.162. **2Wiki much wider margin.** ✓
  - ⚠ **Gemma side check**: Gemma-Hot DRAGIN-RIND 0.689 vs tokens 0.611 → +0.078. Gemma-Wiki DRAGIN-RIND 0.647 vs tokens 0.518 → +0.129. **2Wiki wider.** ✓ Gemma-Hot FCT 0.642 vs tokens 0.611 → +0.031. Gemma-Wiki FCT 0.784 vs 0.518 → **+0.266**. ✓✓ much wider on 2Wiki.
  - ✓ **"On exactly the cells where retrieval matters most"** — defensible: 2Wiki has more `lift_em=+1` (34+24=58 helped) vs HotpotQA (31+19=50 helped). 2Wiki is harder, retrieval-rescue more relevant. ✓

## 4.3 FCT Discriminates Strongest on the Held-Out MoE Cell (`results.fct-percell`)

- [ ] **S1.** On the same held-out cells where length fails (§results-inverted), FCT discriminates most strongly --- reaching raw AUROC **0.784** on the Gemma × 2Wiki cell (Table fct-percell).
  - ✓ Already verified at I2; per `moonshot_explore.out` line 125. ✓
- [ ] **S2.** The architecture-by-dataset variance is consistent with cell-dependent retrieval-helpfulness structure (§discussion): the cells where the length proxy collapses are precisely the cells where the answer-trajectory primitive carries the most independent information.
  - ⚠ **Pattern claim.** Per-cell FCT raw AUROC (`moonshot_explore.out` lines 122–125):
    - Qwen×Hot: FCT 0.677, tokens 0.572 → FCT - tokens = +0.105 (HotpotQA cell)
    - Qwen×2Wiki: FCT 0.604, tokens 0.457 → +0.147 (2Wiki cell)
    - Gemma×Hot: FCT 0.642, tokens 0.611 → +0.031
    - Gemma×2Wiki: FCT 0.784, tokens 0.518 → +0.266
  - **Where length proxy collapses (2Wiki cells, tokens ≤ 0.52), FCT margin is widest (+0.147 Qwen, +0.266 Gemma).** ✓ Pattern holds.
  - ⚠ **"Cells where the answer-trajectory primitive carries the most independent information"** — defensible but reads as a slight strawman. Could be precise: "FCT and min_majority show the largest residualized Δ on the cells where length is least informative." Defensible as is.
- [ ] **C1.** (Table caption) Per-cell raw AUROC of FCT on `lift_em`=+1.
  - ✓ Table rows in `template.tex` lines 210–213 match `moonshot_explore.out` per-cell breakdown:
    - Qwen×HotpotQA 200/31/0.677 ✓
    - Qwen×2Wiki 200/34/0.604 ✓
    - Gemma×HotpotQA 200/19/0.642 ✓
    - Gemma×2Wiki 200/24/**0.784** ✓
  - All four cells exact-match.

## 4.4 Survivor Set Is Robust to a Semantic-Equivalence Judge (`results.llmjudge`)

- [ ] **S1.** The survivor matrix is robust to a semantic-equivalence judge.
  - ✓ Setup sentence; supporting evidence in S2–S5 and Abstract S7.
- [ ] **S2.** We relabelled both arms of every paired trace via a GPT-4o-Mini judge (Janiak's exact choice) and re-ran the full survivor matrix, addressing the correctness-function critique of `\citet{janiak_illusion_nodate}` and `\citet{ielanskyi_addressing_2025}`.
  - ✓ **`llm_judge_audit.py` and `llm_judge_cache/`** exist at `emnlp_26_lengthConfound/analysis/`. Cache contains per-call JSONs (per App. repro S5). Code path verified.
  - ✓ **Re-ran the full headline survivor matrix** — `llm_judge_audit_both.out` lines 9–33 show the full re-run side-by-side with EM (top block = lift_em_judge, bottom block = original EM lift_em).
  - ✓ **"Janiak's exact choice"** — Janiak uses GPT-4o-Mini per `_lit-review.md` line 123 (`Re-ranking the published trigger pool under LLM-as-Judge (GPT-4o-Mini) labels`). ✓
- [ ] **S3.** EM and the judge agreed at Cohen's κ = 0.707 (substantial agreement; `\citealp{landis_measurement_1977}`).
  - ✓ κ = 0.707 in `llm_judge_audit_both.out` line 7. ✓
  - ✓ **Landis & Koch κ = 0.61–0.80 interpretation = "substantial agreement."** Verified via web search. κ = 0.707 falls cleanly in [0.61, 0.80]. ✓
- [ ] **S4.** Eight of the nine paired-significant survivors under EM remained significant under the judge, with one out-in swap (`etc` loses, `flare.max` gains) that does not affect the Bonferroni-robust four.
  - ✓ **Side-by-side comparison** at `llm_judge_audit_both.out`:
    - 9 EM survivors (lift_em=+1 95%): DRAGIN-RIND, ent.max, etc, u_cons.max, flip_rate K=5, flip_rate K=1, min_majority, CABV, FCT.
    - 9 Judge survivors (lift_em_judge=+1 95%): DRAGIN-RIND, ent.max, flare.max, u_cons.max, flip_rate K=5, flip_rate K=1, min_majority, CABV, FCT.
    - **Swap: etc → flare.max.** ✓ exact match with audit sentence.
    - Of the 9 EM survivors, the 8 that remained under judge: DRAGIN-RIND, ent.max, u_cons.max, flip_rate K=5, flip_rate K=1, min_majority, CABV, FCT. ✓ exactly 8.
  - ✓ **"Bonferroni-robust four"** under EM are DRAGIN-RIND, ent.max, min_majority, FCT — all 4 are in the judge 95% survivors. ✓
- [ ] **S5.** FCT strengthened under the judge from Δ = +0.190 [+0.093, +0.287] to Δ = +0.222 [+0.132, +0.312] (Appendix llmjudge).
  - ✓ Both numbers exact-match `llm_judge_audit_both.out` (FCT EM: +0.190 [+0.093,+0.287]; FCT Judge: +0.222 [+0.132,+0.312]). ✓

## 4.5 The K=5 Multi-Sample Variant Is Null (`results.ablation`)

- [ ] **S1.** The K=5 boundary-completion variant carries no significant advantage over the single-sample K=1 proxy at our power: paired-bootstrap CIs straddle zero on both architectures (Table k-ablation).
  - ✓ Verified at §3.3 S5; per-arch confirmation:
    - Qwen `a1_k1_vs_k5_r1_qwen.out` line 18: `Δraw -0.002 [-0.040, +0.031]`. CI straddles 0. ✓
    - Gemma `a1_k1_vs_k5_r1_gemma.out` line 18: `Δraw +0.033 [-0.025, +0.092]`. CI straddles 0. ✓
- [ ] **S2.** We recommend K=1 for deployment at 5× less compute per boundary, with equivalent triggering performance.
  - ✓ "5× less compute" — trivially true: K=1 sample vs K=5 samples per boundary. ✓
  - ⚠ **"Equivalent triggering performance"** — Gemma Δ point estimate is +0.033 (positive but not significant). The audit document `_lit-review.md` line 397 frames the ablation result as "Regime 2 (K=5 ≈ K=1, CI straddles 0)" — internally consistent. **"Equivalent"** is defensible at our power, but the +0.033 point estimate on Gemma is mildly worrying for a "strict equivalence" reading. Defensible as is given the 95% CI; consider hedging to "statistically equivalent triggering performance at our power."
- [ ] **C1.** (Table caption) K=5 vs. K=1 `flip_rate` ablation on `lift_em`=+1, pooled per architecture.
  - ✓ Table rows in `template.tex` lines 236–237: Qwen n=305 helped=47 Δ=-0.002; Gemma n=394 helped=42 Δ=+0.033. **Exact-match.** ✓
  - ⚠ **Table n != 800**: Table reports n=305 and n=394, totaling 699 (not 800). This is because the K=1 vs K=5 ablation requires per-step samples to be derivable, and 101/800 traces lack ≥2 answer-bearing steps for flip_rate computation. **Not a contradiction** — `signal_lib.py` `flip_rate_majority` returns None when `len(seq) < 2`, so those rows are dropped. Consistent.

## 4.6 Robustness Summary (`results.robustness`)

- [ ] **S1.** The headline finding (four Bonferroni-robust signals with FCT strongest) is invariant under three independent design choices: (i) the choice of correctness function (EM vs. a GPT-4o-Mini semantic-equivalence judge agreeing with EM at κ = 0.707; §results-llmjudge); (ii) the choice of sample size in the answer-trajectory family (K=1 vs. K=5 on `flip_rate` is null on both architectures; §results-ablation); and (iii) the choice of evaluation dataset (HotpotQA vs. held-out 2WikiMHQA; FCT discriminates strongest on the held-out Gemma × 2Wiki cell at raw AUROC **0.784**; §results-fct-percell, §results-inverted).
  - ✓ All three claims verified individually above. Cross-refs resolve correctly.
- [ ] **S2.** The residualization protocol itself agrees with the Uncertainty-Line reference implementation of `\citet{vashurin_uncertainty-line_2025}` within ε ≤ 0.029 AUROC on both architectures (App. uline).
  - ✓ Verified at §3.2 S5. ε_max=0.0287 Qwen, ε_max=0.0270 Gemma; both ≤ 0.029. ✓

---

# 5. Discussion

## 5.1 Why Length Predicts Retrieval-Helpfulness (`discussion.causal`)

- [ ] **S1.** On the secondary `closed_em`=0 target, `tokens_generated` reached raw AUROC 0.539 on the headline pool and was the single strongest pre-residualization signal on several cells.
  - ✓ **`tokens_generated` raw AUROC on `closed_em=0` headline pool.** `moonshot_explore.out` line 104: `tokens_generated 800 0.539 0.520 (baseline) ...`. ✓ Exact match 0.539.
  - ⚠ **"Strongest pre-residualization signal on several cells"** — defensible across some cells: e.g. on Phi-baseline `closed_em=0` (per `_lit-review.md` lines 116, 130): Phi×HotpotQA tokens 0.645 beat every internal signal. But we don't have Phi in this paper — the headline pool is Qwen+Gemma. Per `moonshot_explore.out`: on Qwen-only `closed_em=0` (n=400) line 28, `tokens_generated 0.514`, DRAGIN 0.575, ent.max 0.603 — tokens is **NOT** the strongest. On Gemma-only `closed_em=0` line 66, `tokens_generated 0.598`, DRAGIN-RIND 0.709 — tokens is again **NOT** the strongest. **Mild overclaim.** ⚠ Reviewer-risk: low–medium. Consider tightening to "and remained a competitive baseline on several cells" or "the strongest pre-residualization length proxy."
- [ ] **S2.** This is consistent with the *verbosity compensation* mechanism documented by `\citet{zhang_verbosity_2024}`, where verbose-when-uncertain is a real LLM behavior on the multi-hop QA family represented in LongBench (which includes HotpotQA, MuSiQue, and 2WikiMHQA).
  - ✓ Already verified in §1 p2 S3 / S2. LongBench bundle includes HotpotQA/MuSiQue/2WikiMHQA per `_lit-review.md` line 324. ✓
- [ ] **S3.** A plausible causal chain underlying this pattern is length → verbosity compensation → uncertainty → incorrectness → retrieval need: under this picture, length confounds the evaluation because it proxies the proximal cause, and the trigger pool's signals receive partial AUROC credit for tracking length because length correlates with the deeper uncertainty mechanism.
  - ✓ **Causal chain framing.** Per `_lit-review.md` lines 335–336: "length is a proxy for verbosity-compensation, which is a proxy for uncertainty, which is a proxy for incorrectness/retrieval-need." Same causal chain. ✓ Defensible as a *plausible* mechanism (the paper hedges with "plausible").
- [ ] **S4.** Residualization (§methods-residualization) removes the length pathway and isolates the residual signal.
  - ✓ Internal cross-ref resolves; protocol claim is mathematical (OLS residual removes linear-in-`log(tokens+1)` component, which is the *operational* definition of "length pathway"). Defensible.

## 5.2 Why Answer-Trajectory Stability Predicts Retrieval Need (`discussion.fct`)

- [ ] **S1.** The mechanism behind FCT reads naturally as *premature commitment*, the interpretation `\citet{datta_large_2026}` give to their `AnswerSwitches` and t* primitives: the model decides on an answer early and then continues to verbalize reasoning that supports the commitment, with most subsequent steps being post-decision tail.
  - ✓ **"Premature commitment"** language: per `_lit-review.md` line 65 and Datta's findings: predicted answer changes in only **32%** of Qwen3-4B traces (i.e., 68% commit early and never flip), and after the final answer switch the model generates **760 additional reasoning tokens on average**. **Byte-faithful interpretation.** ✓
- [ ] **S2.** `\citeauthor{datta_large_2026}` apply this observation to early stopping (if the model has decided, no further compute is needed).
  - ✓ Per `_lit-review.md` line 67: "Probe-based early stopping at the detected final switch reduces reasoning tokens by ~500 per query at only 2% accuracy drop." ✓
- [ ] **S3.** Retrieval triggering reads the same primitive in the opposite causal direction: traces where the model has *not* decided, where the answer trajectory continues to flip through the reasoning, are the traces where external evidence would resolve the uncertainty.
  - ✓ Framing claim consistent with `_lit-review.md` lines 72–73: "Application target is orthogonal — they go for early stopping, we go for retrieval triggering. Their `t*` indicates *no further compute is needed*; our `flip_rate` / `late_flips` indicate *external information would help*. Both can be true on the same trajectory family." ✓
- [ ] **S4.** The paired-oracle target `lift_em`=+1 formalizes this question directly.
  - ✓ True by construction: `lift_em=+1` = closed-book wrong AND oracle-context right = retrieval would have resolved the uncertainty by providing external evidence. ✓
- [ ] **S5.** The same primitive supports two complementary operations: detect stabilization to *stop*; detect non-stabilization to *retrieve*.
  - ✓ Synthesis sentence; consistent with the cross-paper framing in `_lit-review.md` line 76. ✓ Strong contribution-positioning sentence.

---

# 6. Conclusion

- [ ] **S1.** We presented the first length-controlled audit of the published training-free dynamic-RAG trigger pool under length-residualized AUROC on a paired-oracle target (`lift_em`=+1).
  - ✓ Already verified at Abstract S3.
- [ ] **S2.** On two open-weight chat models over two multi-hop QA datasets, nine signals beat the length baseline at 95% paired-bootstrap and four survived Bonferroni at k=15 across three signal families (attention, token-probability, answer-trajectory).
  - ✓ Already verified at Abstract S4.
- [ ] **S3.** The strongest length-robust trigger is **FCT**, a forward-looking variant of the answer-trajectory primitive.
  - ✓ Already verified at Abstract S5 / I2.
- [ ] **S4.** The survivor set is robust to a GPT-4o-Mini semantic-equivalence judge (κ = 0.707, eight of nine survivors preserved).
  - ✓ Already verified at Abstract S7 / I4 / §4.4 S4.
- [ ] **S5.** Length-residualized and raw AUROC disagree most on the cells where retrieval matters most, motivating both protocols in dynamic-RAG trigger comparisons on multi-hop QA.
  - ✓ Defensible: 2Wiki cells = where retrieval matters most (more `lift_em=+1` helped, base rate higher) AND where length is anti-predictive. The protocol-disagreement claim cashed at §4.2 S3. ✓

---

# 7. Limitations

## 7.1 Signal-level audit, not end-to-end pipelines (`lim.signal`)

- [ ] **S1.** We audit trigger *signals* as defined in their original papers, not full deployment pipelines (threshold tuning, retriever choice, prompt format).
  - ✓ Scope statement; defensible. The adapter notes in App. G confirm signal-vs-pipeline framing.
- [ ] **S2.** Our claims pertain to signal-level discriminative power on `lift_em`.
  - ✓ Scope statement consistent with §3.1. ✓

## 7.2 Bonferroni knife-edge (`lim.bonferroni`)

- [ ] **S1.** Two of the four Bonferroni-robust survivors carry ~99.67% lower bounds within 5×10⁻³ of zero (`ent.max` LB = +0.001; `min_majority` LB = +0.004, App. bonferroni).
  - ✓ **`ent.max` LB = +0.001.** `e7_bonferroni.out` line 12: `[+0.001,+0.263]`. ✓ exact.
  - ✓ **`min_majority` LB = +0.004.** `e7_bonferroni.out` line 24: `[+0.004,+0.284]`. ✓ exact.
- [ ] **S2.** The headline "four Bonferroni-robust signals across three signal families" claim survives the loss of either signal as "three across two," but the four-count is fragile to small bootstrap-seed or signal-set perturbations; the FCT and DRAGIN-RIND survivors are robust (+0.047 and +0.039 respectively).
  - ✓ **FCT LB +0.047.** `e7_bonferroni.out` line 26: `[+0.047,+0.329]`. ✓ exact.
  - ✓ **DRAGIN-RIND LB +0.039.** `e7_bonferroni.out` line 11: `[+0.039,+0.286]`. ✓ exact.
  - ⚠ **"Survives the loss of either signal as three across two"** — if `ent.max` (token-prob) drops, surviving are DRAGIN-RIND (attention), min_majority (answer-traj), FCT (answer-traj) = 3 signals across 2 families. ✓ If `min_majority` (answer-traj) drops, surviving are DRAGIN-RIND (attention), ent.max (token-prob), FCT (answer-traj) = 3 signals across 3 families. **The audit text says "three across two" for both cases, but the second case is actually three across THREE.** ⚠ **Reviewer-risk: low.** Worth a small fix: either tighten to "three across two or three (depending on which signal is dropped)" or specify "the ent.max-drop case reduces to three signals across two families."
- [ ] **S3.** We report this transparently rather than tightening B from 10,000 to widen the margin.
  - ✓ Transparent honesty statement. ✓ Worth keeping; reviewer-positive.

## 7.3 Architecture and dataset coverage (`lim.coverage`)

- [ ] **S1.** The headline pool covers two architectures (Qwen2.5-7B dense, Gemma-4-26B-A4B-it MoE-hybrid) and two multi-hop QA benchmarks.
  - ✓ Already verified at I3 / §3.1 S9.
- [ ] **S2.** The cheapest extensions that would tighten the cross-architecture claim are a third open-weight chat model (e.g., Llama-3.1-8B-Instruct) and a third multi-hop benchmark (MuSiQue, `\citealp{trivedi_musique_2022}`); we report the current Bonferroni lower bound on FCT (+0.047, App. bonferroni) as the statistical floor that additional cells would have to displace.
  - ✓ **`trivedi_musique_2022`** → MuSiQue (Trivedi et al. TACL 2022). Already verified. ✓
  - ✓ FCT Bonferroni LB +0.047 already verified. ✓
  - ⚠ **Llama-3.1-8B-Instruct mention** is hypothetical (future-work); no claim about us having run it. Defensible. ✓

## 7.4 Dataset framing: held-out, not OOD (`lim.framing`)

- [ ] **S1.** We refer to 2WikiMultiHopQA as a *held-out* cell relative to HotpotQA, not an out-of-distribution benchmark in the strict sense: both are multi-hop QA over Wikipedia with comparable question complexity, and the cross-dataset gap is closer to a transfer-evaluation gap than the kinds of distribution shift studied in the formal OOD-detection literature `\citep{lu_streaming_2026, ielanskyi_addressing_2025}`.
  - ⚠ **OOD-detection literature claim.** Lu Streaming (`lu_streaming_2026`) does discuss hallucination detection in long-CoT, which arguably overlaps with OOD. Ielanskyi (`ielanskyi_addressing_2025`) explicitly recommends adding OOD detection as a risk-correlation experiment (per `_lit-review.md` line 98–99). Both citations defensible as OOD-literature anchors.
  - ⚠ **However:** neither Lu nor Ielanskyi is a *canonical* OOD-detection reference (e.g., Hendrycks/Gimpel 2017, Liang et al. 2018). The citations are about *uncertainty / hallucination* literature that *includes* OOD as a property — close but not exact. Defensible as scope-hedge; reviewer-risk: low.
- [ ] **S2.** The cell-dependent length-behavior we report (§results-inverted) is a finding about dataset-specific length-confound structure on multi-hop QA, not a claim about OOD generalization; a stricter cross-distribution evaluation (different task family, different domain, or a synthetic perturbation set) is future work.
  - ✓ Scope hedge is defensible. ✓

## 7.5 Correctness function (`lim.correctness`)

- [ ] **S1.** We use exact-match equality on both arms of the paired-oracle target; the paired design shares the same EM function across arms (§methods-protocol), and Appendix llmjudge reports an LLM-as-Judge robustness check (κ = 0.707, eight of nine survivors preserved).
  - ✓ All previously verified. ✓
- [ ] **S2.** Containment-match and F₁ variants remain future work.
  - ✓ Scope hedge. Defensible.

## 7.6 Non-thinking-mode chat models (`lim.nonthinking`)

- [ ] **S1.** The audited models are non-thinking-mode chat models.
  - ✓ Qwen2.5-7B-**Instruct** and Gemma-4-26B-A4B-**it** — both `-Instruct` / `-it` chat variants, no thinking-mode tokens. ✓
- [ ] **S2.** `\citet{datta_large_2026}` developed their original primitive on thinking-mode reasoning models with explicit `</think>` tokens; our K=5 boundary-completion adaptation is not byte-faithful to that protocol.
  - ✓ Per `_lit-review.md` line 74: "their forced-completion requires the explicit `</think>` token; ours uses spaCy-sentence boundary detection on a non-think-tag-bearing trace. **The capture protocols are NOT directly comparable.**" ✓ Honest disclosure.

## 7.7 Integration as contribution (`lim.integration`)

- [ ] **S1.** Our contribution is the integration of length-residualized AUROC, the paired-oracle `lift_em` target, and the answer-trajectory family including FCT.
  - ✓ Honest framing aligned with `_lit-review.md` line 263 ("we adopt and extend the UNCERTAINTY-LINE protocol"). ✓
- [ ] **S2.** Each piece individually is incremental over its parent; the integration into a single length-controlled audit, and the empirical identification of length-robust survivors, are the load-bearing claims.
  - ✓ Strong honesty statement; reviewer-positive. ✓

## 7.8 Risks (`lim.risks`)

- [ ] **S1.** Our work concerns evaluation methodology for retrieval triggering, on open-weight chat models and public multi-hop QA benchmarks; we release no new datasets or trained models, and direct risks are limited.
  - ✓ Standard scope statement; defensible.
- [ ] **S2.** One methodological risk is that the length-residualized protocol could be cited in isolation to defend deployed dynamic-RAG systems whose triggers track length more strongly than residualization alone would reveal.
  - ✓ Thoughtful methodological-risk identification; reviewer-positive.
- [ ] **S3.** We therefore recommend reporting raw, length-residualized, and LLM-judge-validated AUROC together (§results-llmjudge, §results-robustness) rather than substituting one for another.
  - ✓ Cross-refs resolve. ✓ Recommendation is consistent with the paper's empirical approach (we report all three). ✓

---

# A. Bonferroni-Corrected Survivor Set (`app.bonferroni`)

## A.body — header prose

- [ ] **S1.** Table bonferroni reports the Bonferroni-corrected ~99.67% paired-bootstrap CI of Δ = AUROC(s_resid) − AUROC(`tokens_generated`) at k=15 (α = 0.05/15 = 0.0033) on the headline pool.
  - ✓ All verified above.
- [ ] **S2.** Bonferroni k is the number of signals tested for the survivor-matrix claim.
  - ✓ Definition. ✓
- [ ] **S3.** Four signals retain a strictly positive lower CI bound under correction: FCT, DRAGIN-RIND, `min_majority`, and `ent.max`.
  - ✓ Verified at Abstract S4. ✓

## A.footnote — k=15 justification footnote

- [ ] **F1.** k=15 was the multiple-comparison count fixed at test time, when the audited pool included a CABV×LAAS composite that we later dropped from presentation (§methods-fct).
  - ✓ **CABV × LAAS exists** in `e7_bonferroni.out` line 27 and `signal_lib.py` SIGNALS — confirmed historically in pool. Now dropped from the survivor-matrix narrative but kept in supporting analysis. ✓ Defensible.
- [ ] **F2.** The current table reports 16 audited signals plus the length baseline.
  - ⚠ **Count check.** Table `tab:hero-full` lines 336–355 lists: `tokens_generated` (baseline) + FCT, DRAGIN-RIND, min_majority, ent.max, CABV, flip_rate K=5, u_cons.max, flip_rate K=1, etc, late_flips, flare.max, hvr.max (SELFDOUBT-T1), hidden_norm.range, drift_cosine.max, seakr_k5.logdet, var_l2_window.max = **16 audited + 1 baseline = 17 rows.** ✓ matches "16 audited signals plus the length baseline."
  - ⚠ **k=15 vs 16 audited inconsistency.** k=15 was set when CABV×LAAS was in pool; that gave 15 unique residual-vs-tokens comparisons. Now there are 16 (CABV×LAAS removed, CABV kept separately). The footnote acknowledges the inconsistency. ✓
- [ ] **F3.** Re-running at k=14 or k=16 shifts the Bonferroni LBs by less than ±0.005 AUROC and does not change the four-survivor count; we report the test as run.
  - ❓ **"Re-running at k=14 or k=16 shifts the Bonferroni LBs by less than ±0.005 AUROC."** No code in `analysis/` re-runs `e7_bonferroni_and_spearman.py` with different `N_SIGNALS_FOR_BONFERRONI` values. **This claim is NOT verified by an explicit re-run.** It's a *back-of-envelope estimate* based on the relative shape of paired-bootstrap CIs as α shifts marginally. **Reviewer-risk: medium.** Easy 30-minute fix: add a `--bonferroni-k` argument and run at k=14 / k=15 / k=16, dump output. Or weaken claim to "we estimate this shift to be < 0.005 AUROC."
  - ⚠ **Recommendation:** either run the k-sensitivity check (cheap) or hedge the claim.

## A.caption — Table `tab:bonferroni` caption

- [ ] **C1.** Bonferroni-corrected survivor set on the headline pool (Qwen+Gemma, n=800 paired, `lift_em`=+1, B=10,000).
  - ✓ Table values in `template.tex` lines 304–318 exact-match `e7_bonferroni.out` rows.
- [ ] **C2.** The `etc` (Δ²H) signal is the Entropy-Trend Constraint trigger of `\citet{li_modeling_2026}`.
  - ✓ **ETC = "Entropy-Trend Constraint"** confirmed via official repos `pkuserc/ETC` and `WisdomShell/ETC`; abstract on AAAI 2026 page also uses the exact phrase. Acronym is byte-faithful. ✓
  - ✓ Paper authors Bo Li et al. published AAAI 2026 Oral. Year-tag `_2026` is consistent with venue. ✓

---

# B. Full Headline Survivor Matrix (`app.hero`)

## B.body — header prose

- [ ] **S1.** Table hero-full reports all 16 audited signals on the headline pool with length-residualized AUROC and Δ vs. `tokens_generated`.
  - ✓ Table contents (lines 336–354 of `template.tex`) match `e7_bonferroni.out`. The 16 signals + tokens baseline already accounted for in F2 audit. ✓

## B.caption — Table `tab:hero-full` caption

- [ ] **C1.** Full cross-architecture survivor matrix on the headline pool (Qwen2.5-7B + Gemma-4-26B-A4B-it × {HotpotQA, 2WikiMHQA}, n=800 paired traces).
  - ✓ All confirmed.
- [ ] **C2.** All AUROCs are length-residualized (§methods-residualization).
  - ✓ `e7_bonferroni.out` `resAUC` column matches table's "Resid AUROC" column. Verified.
- [ ] **C3.** 95% CIs are paired-bootstrap (B=10,000) of Δ = AUROC(s_resid) − AUROC(`tokens_generated`).
  - ✓ Already verified.
- [ ] **C4.** **Bold**: Bonferroni-robust at k=15 (lower bound of the 99.67% CI strictly positive; see Table bonferroni).
  - ✓ The 4 bold rows in Table `tab:hero-full` are FCT, DRAGIN-RIND, min_majority, ent.max — match Bonferroni `✓✓` set. ✓
  - ⚠ **Caveat.** The table has the 9 nominal-95% survivors in the top block (lines 338–346) with the bold 4 followed by 5 non-bold; the table organization is consistent. **etc has Δ +0.107 in our paper's Table 1; in `e7_bonferroni.out` line 13: +0.107.** ✓ matches. **However:** `etc` does NOT survive Bonferroni in `e7_bonferroni.out` (CI `[-0.026,+0.235]` straddles zero) — and the audit table correctly shows it as non-bold. ✓ consistent.

---

# C. Correctness-Function Robustness: LLM-as-Judge (`app.llmjudge`)

## C.protocol — Protocol paragraph

- [ ] **S1.** For each of the n=800 paired traces in the headline pool, we re-labelled both arms via an OpenAI `gpt-4o-mini` semantic-equivalence judge under a strict prompt ("You are a strict semantic-equivalence judge for multi-hop QA. Reply with exactly one word: YES or NO. No explanation.").
  - ⚠ **Prompt text — minor discrepancy.** `llm_judge_audit.py` lines 104–106:
    ```
    You are a strict semantic-equivalence judge for multi-hop QA.
    Your task is to decide whether a model-predicted answer is semantically equivalent to a gold answer.
    Reply with exactly one word: YES or NO. No explanation.
    ```
    The paper quote (App. C `\paragraph{Protocol}`) is `"You are a strict semantic-equivalence judge for multi-hop QA. Reply with exactly one word: YES or NO. No explanation."` — the **middle sentence "Your task is to decide whether a model-predicted answer is semantically equivalent to a gold answer."** is dropped from the paper's quoted prompt. ⚠ **Reviewer-risk: low** (cosmetic), but worth fixing. Suggested fix: replace with the full three-sentence system prompt for byte-faithfulness. There's also a `JUDGE_USER_TEMPLATE` (Question/Gold/Predicted) that the paper doesn't quote — also worth a parenthetical mention or a one-line in App. C.
- [ ] **S2.** The judge target is **[Eq. 5] `lift_em_judge(q) = 1[judge(a_o)=YES ∧ judge(a_c)=NO]`**.
  - ✓ Operational definition: helped iff oracle judged correct AND closed judged incorrect. Internally consistent with the paired-oracle framing.
- [ ] **S3.** We re-ran the full headline survivor matrix with paired bootstrap B=10,000.
  - ✓ `llm_judge_audit_both.out` line 4: `bootstrap-b: 10000`. Match.

## C.agreement — Inter-label agreement paragraph

- [ ] **S1.** EM identifies 108 helped traces and the judge identifies 113; the intersection is 87.
  - ✓ **108 EM helped.** `llm_judge_audit_both.out` lift_em (bottom block) line 37: `pop=108`. ✓
  - ✓ **113 judge helped.** `llm_judge_audit_both.out` lift_em_judge (top block) line 11: `pop=113`. ✓
  - ✓ **Intersection = 87.** `llm_judge_audit_both.json` line 10: `"n_helped_both": 87`. ✓ exact match. Also: `cohens_kappa_em_vs_judge = 0.7073007308665485` rounds to 0.707. ✓
- [ ] **S2.** The two labels jointly produce a multi-class Cohen's κ = 0.707 on the 3-class `lift_em` ∈ {+1, 0, −1} label set, in the substantial-agreement band of `\citet{landis_measurement_1977}`.
  - ✓ All verified above.

## C.transfer — Survivor matrix transfer paragraph

- [ ] **S1.** Table llmjudge reports the headline survivor matrix under both correctness functions side-by-side.
  - ✓ Table `tab:llmjudge` lines 378–390 of template.tex. ✓
- [ ] **S2.** Eight of the nine paired-significant survivors under EM remained significant under the judge; `etc` loses significance and `flare.max` gains it.
  - ✓ Already verified at §4.4 S4. ✓
- [ ] **S3.** FCT strengthened under the judge from Δ = +0.190 [+0.093, +0.287] to Δ = +0.222 [+0.132, +0.312] (+17% in point estimate, +42% in 95% CI lower-bound margin).
  - ✓ **+17% in point estimate.** (0.222 - 0.190) / 0.190 = 0.0337 / 0.190 = **17.4%**. Rounded to +17%. ✓
  - ✓ **+42% in 95% CI lower-bound margin.** (0.132 - 0.093) / 0.093 = 0.039 / 0.093 = **41.9%**. Rounded to +42%. ✓ exact.
- [ ] **S4.** All four Bonferroni-robust survivors retain nominal-95% significance under the judge.
  - ✓ Already verified at I4. ✓

## C.caption — Table `tab:llmjudge` caption

- [ ] **C1.** Side-by-side Δ under EM and LLM-as-Judge on the headline pool (n=800, B=10,000).
  - ✓ Already verified. ✓
- [ ] **C2.** *95% CI straddles zero.
  - ✓ Asterisk-marked rows in `tab:llmjudge` are `etc (Δ²H)` under judge (+0.088*) and `flare.max` under EM (+0.080*). Both checked against `llm_judge_audit_both.out`:
    - EM `flare.max`: line 47 `+0.080 [-0.008,+0.165]` straddles 0 (since lo < 0). ✓
    - Judge `etc`: line 19 `+0.088 [-0.002,+0.175]` straddles 0 (since lo < 0). ✓ Both asterisks correct.
- [ ] **C3.** FCT *strengthens* under the judge; eight of nine paired-significant EM survivors are preserved; all four Bonferroni-robust survivors retain nominal-95% significance.
  - ✓ All previously verified.

## C.caveat — Caveat paragraph

- [ ] **S1.** `\citet{zheng_judging_2024}` document that GPT-4-tier judges exhibit a verbosity bias, preferring longer responses; our cross-judge check demonstrates *pattern stability* rather than full verbosity-independence.
  - ⚠ **`zheng_judging_2024`** → Zheng et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (NeurIPS Datasets and Benchmarks 2023). Per web search:
    - The original paper (Zheng et al. 2023) documents verbosity bias as "an LLM judge favors longer, verbose responses, even if they are not as clear, high-quality, or accurate as shorter alternatives." ✓ Match.
    - However, the follow-up "Judging the Judges" (arXiv 2406.07791) **argues that verbosity bias is essentially position bias driven by answer quality gaps**, complicating the simple "judges prefer longer" framing. **The citation is defensible** (the original paper does document the bias), but reviewers may know the literature has moved on. ⚠ Reviewer-risk: low–medium. Worth a parenthetical hedge.
  - ✓ **"Cross-judge check demonstrates pattern stability rather than full verbosity-independence"** — appropriate hedge; we don't claim to defeat verbosity bias, just to show our survivor set is robust under EM and GPT-4o-Mini.
- [ ] **S2.** The load-bearing length-orthogonality argument rests on the Uncertainty-Line residualization (§methods-residualization) and the paired-oracle design (§methods-protocol); the judge check confirms the survivor matrix is not an artifact of any single correctness function.
  - ✓ Defensible synthesis. Cross-refs resolve.

---

# D. Uncertainty-Line Reference Replication (`app.uline`)

## D.body — header prose

- [ ] **S1.** To pre-empt the reimplementation critique, we ran our protocol (Eq. resid) and the `\citet{vashurin_uncertainty-line_2025}` reference implementation side-by-side on the same trace cohort, decomposing the comparison into two ε components.
  - ✓ **Reference impl** at `emnlp_26_lengthConfound/externalCodeAudits/uncertainty-line/utils.py` (verified via grep — implements PolynomialFeatures + MinMaxScaler + LinearRegression).
  - ✓ Our protocol at `analysis/uncertainty_line_audit.py` runs both and dumps `ε_protocol` and `ε_sampling`. ✓
- [ ] **S2.** ε_protocol isolates the math difference (our log(tokens+1) single-cohort OLS against the reference MinMax-scaled linear regression on the same cohort); ε_sampling isolates train/test-split sampling noise at n=200.
  - ✓ Decomposition byte-faithful: `ours_resid` = our log(tokens+1) OLS; `line_resid_full` = MinMax + linear regression on same cohort; `line_resid_split` = MinMax + linear regression with 70/30 train/test split.
  - ✓ `ε_protocol = |ours_resid - line_resid_full|` (math diff); `ε_sampling = |line_resid_full - line_resid_split|` (sampling noise diff). Exact match with `uncertainty_line_audit.py` lines 24–25.

## D.caption — Table `tab:uline-repl` caption

- [ ] **C1.** ε_protocol = |ours_resid − LINE_full|; ε_sampling = |LINE_full − LINE_split| (latter averaged over 3 seeds).
  - ✓ Already verified. ✓
- [ ] **C2.** Both architectures show ε_protocol ≤ 0.029, well below the conventional threshold of 0.05.
  - ✓ Already verified at §3.2 S5. Qwen 0.0287, Gemma 0.0270. ✓
- [ ] **C3.** Train/test sampling noise (ε_sampling) dominates the protocol gap by a factor of 3–4×, indicating that the log(tokens+1) vs. raw-MinMax math difference is small relative to evaluation-set variance at n=200.
  - ✓ Check arithmetic:
    - Qwen `ε_sampling_max=0.1113`, `ε_protocol_max=0.0287` → ratio 0.1113/0.0287 = **3.88×**. In 3–4× range. ✓
    - Gemma `ε_sampling_max=0.1063`, `ε_protocol_max=0.0270` → ratio 0.1063/0.0270 = **3.94×**. In 3–4× range. ✓
  - **3–4× claim exact match.** ✓
  - ⚠ **Table values in template.tex lines 410–411:**
    - Qwen2.5: 0.029, 0.012, 0.012, 0.111 — matches `uncertainty_line_audit_qwen2.5-7b-instruct.out`: `ε_protocol max=0.0287 mean=0.0120 med=0.0115`, `ε_sampling max=0.1113`. ✓ matches (rounded). 
    - Gemma: 0.027, 0.0095, 0.0047, 0.106 — matches output: `ε_protocol max=0.0270 mean=0.0095 med=0.0047`, `ε_sampling max=0.1063`. ✓ matches.

---

# E. FCT vs. Concurrent Commitment-Time Primitives: Operational Differences (`app.fct-tstar`)

## E.body — header prose

- [ ] **S1.** FCT (§methods-fct) is motivated by the final-switch index t* = max{i : A_i ≠ A_{i+1}} of `\citet{datta_large_2026}`, with a concurrent forward-looking commitment-time primitive proposed by `\citet{zhang_commit_2026}` on the model's finite-answer continuation distribution.
  - ✓ **Datta t* definition.** Per `_lit-review.md` line 57: "Final answer-switch index t* = max{i : A_i ≠ A_{i+1}} (-1 if none)." **Byte-faithful** with the App. E equation.
  - ✓ **Zhang commit description.** Per web-search abstract of `zhang_commit_2026` (arXiv 2605.06723): "We study this question through a narrow computable object: finite-answer preference stabilization. For a model state and specified answer verbalizers, we project the model's own continuation probabilities onto a finite answer set." **Byte-faithful** with the App. E description.
- [ ] **S2.** The three are operationally distinct on five axes.
  - ✓ Table tab:fct-vs-tstar in `template.tex` lines 426–435 lists 5 axes: Temporal direction, Signal class, Sampling regime, Application target, Trace model class. Count = 5. ✓

## E.caption — Table `tab:fct-vs-tstar` caption

- [ ] **C1.** Operational differences across three concurrent commitment-time primitives.
  - ✓ ✓
- [ ] **C2.** `\citet{datta_large_2026}`'s t* is backward-looking on per-step intermediate answers, applied to early stopping.
  - ✓ Datta: t* finds the *last* index where consecutive answers differ → backward-looking. Early stopping via probes per `_lit-review.md` line 67. ✓
- [ ] **C3.** `\citet{zhang_commit_2026}`'s τ_commit is forward-looking on the model's finite-answer continuation log-odds, applied to pre-verbalization commitment analysis.
  - ✓ Zhang: commitment time is **first pre-onset position where the induced winner matches the eventual answer with sufficient margin and does not later flip** → forward-looking. ✓
  - ⚠ **"log-odds"** specifically: per web-search abstract, Zhang's δ is "in binary tasks this yields an exact log-odds code, δ(ξ) = S_θ(yes|ξ) - S_θ(no|ξ)" — for binary tasks. For multi-class it's a finite-answer continuation distribution, not strictly log-odds. **Acceptable approximation** for the table-caption space; could hedge.
- [ ] **C4.** FCT (ours) is forward-looking on K=5-majority answer extracts, applied to retrieval triggering.
  - ✓ Already verified in §3.3.
- [ ] **C5.** For monotonically converging trajectories, t_commit ≈ t* + 1; under K=5 majority sampling on non-thinking-mode chat models this equivalence does not hold in general.
  - ⚠ **Mathematical claim "t_commit ≈ t* + 1 for monotonic trajectories"** — defensible: if all switches are before some final index t*, then the last switch is at t*, and the first stable position with all subsequent matches is t* + 1. **For deterministic K=1 trajectories on thinking-mode reasoning models, this is exact.** With K=5 stochastic sampling and non-thinking-mode chat models, the answer extracts are noisier, so the equivalence breaks. ✓ Math is correct.

---

# F. Length-vs-Signal Spearman ρ per Cell (`app.spearman`)

## F.body — header prose

- [ ] **S1.** Table spearman reports Spearman ρ between `tokens_generated` and each signal per (arch × dataset) cell on closed-book traces, the length-confound diagnostic of `\citet{santilli_revisiting_2025}` (their Table 2).
  - ✓ Reference to Santilli's Table 2: per `_lit-review.md` line 215 ("4 × 4 × 8 × 7 empirical matrix") and lines 224–229 (their Figure 4 reports the Spearman correlations); the actual cross-correlation table is per-signal × per-correctness-fn. Our analogue is per-cell × per-signal. **Conceptually parallel; defensible.** ✓
  - ⚠ **"Their Table 2"** — Santilli paper has multiple tables; the spearman-correlation-leaderboard one. Worth confirming the table number. ❓ Reviewer-risk: low.
- [ ] **S2.** |ρ| ≥ 0.5 indicates a strong length confound on the signal.
  - ✓ Santilli's threshold convention from `_lit-review.md` lines 223–230: their reported correlations range -0.7 to +0.7, with |ρ| ≥ 0.5 = "strong". Defensible.

## F.caption — Table `tab:spearman` caption

- [ ] **C1.** Spearman ρ(`tokens_generated`, signal) per cell on closed-book traces.
  - ✓ `e7_spearman.out` reports ρ per (arch × dataset) cell × signal. ✓
- [ ] **C2.** Bold: |ρ| ≥ 0.5.
  - ✓ Only one bold value: `hvr.max` on Gemma×2Wiki = +0.51 (line 17 of `e7_spearman.out`, marked `*`). Matches table `tab:spearman` line 463: `hvr.max ... +0.51`. ✓
- [ ] **C3.** The peer-pool signals (DRAGIN, ent.max, etc, flare.max, u_cons.max, seakr_k5) carry ρ ∈ [+0.26, +0.46]; `hvr.max` reaches +0.51 on Gemma×2Wiki; the auxiliary CRV diagnostics span ρ ∈ [−0.07, +0.36].
  - ✓ **Peer-pool range [+0.26, +0.46]:**
    - DRAGIN ρ ∈ {+0.40, +0.35, +0.38, +0.38} → [+0.35, +0.40].
    - ent.max ρ ∈ {+0.31, +0.26, +0.42, +0.32} → [+0.26, +0.42].
    - etc ρ ∈ {+0.32, +0.36, +0.46, +0.40} → [+0.32, +0.46]. ← max +0.46
    - flare.max ρ ∈ {+0.39, +0.35, +0.45, +0.32} → [+0.32, +0.45].
    - u_cons.max ρ ∈ {+0.40, +0.31, +0.37, +0.39} → [+0.31, +0.40].
    - seakr_k5 ρ ∈ {+0.37, +0.44, +0.46, +0.43} → [+0.37, +0.46].
    - **Min = +0.26 (ent.max Q×Wiki), Max = +0.46 (etc / seakr_k5). Range [+0.26, +0.46] exact match.** ✓
  - ✓ **`hvr.max` = +0.51 on Gemma×2Wiki** verified.
  - ✓ **Auxiliary CRV range [-0.07, +0.36]:**
    - drift_cosine.max: [+0.20, +0.18, -0.07, -0.01] → min -0.07 ✓ (Gemma×Hot)
    - var_l2_window.max: [+0.19, +0.23, +0.22, -0.00].
    - hidden_norm.range: [+0.24, +0.29, **+0.36**, +0.16] → max +0.36 ✓ (Gemma×Hot)
    - **Range [-0.07, +0.36] exact match.** ✓
- [ ] **C4.** The answer-trajectory aggregations (bottom block) average ≈ 0.1 lower in |ρ| than the peer-pool block.
  - ❓ **Claim verification.** Compute averages:
    - Peer-pool block mean: average of all 6 signals × 4 cells = 24 values. Sum ≈ 0.40+0.35+0.38+0.38+0.31+0.26+0.42+0.32+0.32+0.36+0.46+0.40+0.39+0.35+0.45+0.32+0.40+0.31+0.37+0.39+0.37+0.44+0.46+0.43 ≈ 9.04 → mean ≈ **0.377**.
    - Answer-trajectory block: flip_rate K=5, flip_rate K=1, late_flips, min_majority, CABV, FCT = 6 signals × 4 cells = 24 values. Sum ≈ 0.22+0.12+0.21+0.35+0.17+0.15+0.27+0.21+0.29+0.37+0.25+0.30+0.18+0.12+0.14+0.09+0.22+0.17+0.22+0.36+0.20+0.14+0.15+0.30 ≈ 5.18 → mean ≈ **0.216**.
    - **Difference ≈ 0.377 - 0.216 = 0.161.** Claim says "≈ 0.1 lower"; actual is ~0.16 lower (≈ 0.1 in rough magnitude). ✓ defensible as "≈ 0.1" but actually closer to 0.15. ⚠ Mild reviewer-risk; consider tightening to "≈ 0.15 lower" if you want precision.
- [ ] **C5.** *Qwen×Hot `hvr.max` is constant; Spearman undefined.
  - ✓ `e7_spearman.out` line 17 shows `+nan` on Qwen×Hot column for hvr.max. Cause: hvr is constant on that cell (Qwen always outputs the same level of hedging on HotpotQA traces). Spearman has 0 variance → undefined. ✓

---

# G. Adapter Notes (`app.adapters`)

## G.opening — opening prose

- [ ] **S1.** We audit signal-level discriminative power on `lift_em` (§limitations).
  - ✓ Cross-ref to §7 ("Signal-level audit, not end-to-end pipelines"). ✓
- [ ] **S2.** Several primitives in the published trigger pool are originally framed as online per-token triggers or per-trace scores; we evaluate their underlying statistic at the trace level.
  - ✓ Already verified at §3.2 S6.
- [ ] **S3.** Adapter scope per signal:
  - ✓ Setup for following per-signal subsections.

## G.etc — ETC paragraph

- [ ] **S1.** The original ETC operates on a token-level entropy stream; we evaluate Δ²H on the per-step mean entropy series captured at boundary-detection events (§methods-fct).
  - ✓ Per the Li ETC web search above: "ETC analyzes the first- and second-order differences of the **token-level entropy sequence**." So Li operates on **token-level entropy stream** (decoded token-by-token Ht). ✓ matches the audit claim. Our `signal_lib.py::etc_score` operates on `mean_token_entropy` per step (a smoothed step-level series). Adapter real.
- [ ] **S2.** Step-level smoothing is the only adapter; the second-difference operator is byte-faithful.
  - ✓ Confirmed: `etc_score` computes `d2 = [abs(H[i + 1] - 2 * H[i] + H[i - 1]) for i in range(1, len(H) - 1)]; return max(d2)`. Operator `Δ²H_i = H_{i+1} - 2 H_i + H_{i-1}` byte-faithful with Li 2026 Eq. (per emergentmind synthesis: `Δ2Hi = Hi+2 - 2Hi+1 + Hi`). Indexing difference is cosmetic (`H[i±1]` vs `H[i+2], H[i+1]`); same magnitude per shift. ✓
  - ⚠ **abs(d2) vs raw Δ²H.** Li ETC's Δ²H is *signed* (sign indicates direction of trend); our `etc_score` takes **abs** then **max**. So we conflate positive and negative shifts. Defensible since we only care about magnitude (large-magnitude shift = "entropy spike" = uncertainty change). Mild adapter; not byte-faithful in sign but byte-faithful in magnitude. ⚠ Worth a hedge: "we evaluate `max|Δ²H|`" rather than implying byte-faithfulness.

## G.flare — FLARE paragraph

- [ ] **S1.** The original FLARE is an online per-token retrieval trigger (max_t p_t < τ); we audit the underlying primitive (low-confidence detection) as a trace-level summary max over steps of 1 − min p_t.
  - ✓ **FLARE original online trigger** per Jiang 2023 paper PDF: "actively decide when to retrieve next based on the next-token probability distribution… anticipate forthcoming content, and use it to generate a search query when low-confidence content is identified." The trigger logic is `max_t p_t < τ`, byte-faithful with our description.
  - ✓ **Our adapter** at `signal_lib.py::flare_max_per_step` lines 89–104: `1 - min(probs)` per step, then `max` over steps. **Byte-faithful** with audit claim.
- [ ] **S2.** The trigger *policy* is not audited; the trigger *statistic* is.
  - ✓ Honest framing distinction. ✓

## G.seakr — SeaKR paragraph

- [ ] **S1.** K=5 EOS hidden states (vs. K=20 in the paper, with the paper's best-k interval [10, 25]); the regularized Gram determinant we compute is the same primitive as the paper's self-aware uncertainty score.
  - ⚠ **"K=20 in the paper"** — Need to verify. From SeaKR web search: "We search hyper-parameters for the knowledge recall size N, the dimension of the Gram determinant **k**, and the uncertainty threshold δ on a sample. … The best number of [k] to compute the Gram determinant **falls into the interval [10 - 25]**." So the paper does not single out **K=20** specifically — only the best-k interval [10, 25]. ⚠ Our claim "K=20 in the paper" may be inaccurate; consider rephrasing to "K ∈ [10, 25] in the paper, best-k interval; we use K=5 for compute efficiency." Reviewer-risk: low–medium.
  - ✓ **"Regularized Gram determinant"** is exactly Yao et al.'s formulation. ✓
- [ ] **S2.** INSIDE `\citep{chen_inside_2024}` is the closest peer, using the EigenScore on the centered sample covariance; the K nonzero eigenvalues of our uncentered Gram and INSIDE's centered covariance are rank-equivalent up to centering.
  - ✓ Math claim is correct: Gram matrix `G = M M^T` (uncentered, K×K) and centered covariance `C = (M - μ̄)(M - μ̄)^T` differ only in mean-centering. **K nonzero eigenvalues are rank-equivalent up to centering** — true under mild conditions (full-rank sample mean affects spectrum but not rank).
  - ✓ INSIDE = `chen_inside_2024`. Already verified PDF present. ✓

## G.selfdoubt — SELFDOUBT-T1 paragraph

- [ ] **S1.** HVR = h/(v+1) is computed per step and aggregated as max over steps; the original paper computes a single trace-level HVR with a Tier-1 "HVR=0" high-precision correctness gate.
  - ✓ `signal_lib.py::hvr_max` line 84: `vs = [v for v in step_vals(r, "hvr") if v is not None]; return max(vs) if vs else 0.0`. Per-step max over `hvr` field. ✓ matches audit claim.
- [ ] **S2.** The per-step adapter is recoverable as the paper's gate by setting our threshold to 0.
  - ⚠ **"Threshold of 0"** — defensible: setting threshold to 0 means any non-hedged response (h=0) at any step triggers correctness. Equivalent to the paper's Tier-1 gate when we interpret HVR=0 → confidence. ✓

## G.commit — Concurrent commitment-time primitives paragraph

- [ ] **S1.** Operational differences between `\citet{datta_large_2026}`'s t*, `\citet{zhang_commit_2026}`'s τ_commit, and our FCT are tabulated in Appendix fct-tstar.
  - ✓ Cross-ref `App.~\ref{sec:appendix-fct-tstar}` resolves. ✓

---

# H. Reproducibility (`app.repro`)

- [ ] **S1.** All trace JSONLs are committed at `runs/v2_headline_{arch}/` for arch ∈ {qwen2.5-7b-instruct, gemma-4-26b-a4b-it}.
  - ✓ Confirmed `runs/v2_headline_qwen2.5-7b-instruct/` and `runs/v2_headline_gemma-4-26b-a4b-it/` directories exist with 5 JSONLs each (`B_smoke_hotpot.jsonl`, `D1_closed_hotpot.jsonl`, `D2_closed_wiki.jsonl`, `E1_oracle_hotpot.jsonl`, `E2_oracle_wiki.jsonl`). ✓
  - ⚠ **Note: `B_smoke_hotpot.jsonl`** is also present — a smoke-test cell not mentioned in the paper. Defensible (additional debug artifact); not a contradiction. ✓
- [ ] **S2.** Slice manifests with SHA-256 checksums are at `runs/v2_headline_{arch}/checkpoints/slice_manifest.json`.
  - ✓ Both `slice_manifest.json` files exist at:
    - `runs/v2_headline_qwen2.5-7b-instruct/checkpoints/slice_manifest.json`
    - `runs/v2_headline_gemma-4-26b-a4b-it/checkpoints/slice_manifest.json`
  - ⚠ Should also verify the SHA-256 checksum fields are populated — but file existence at the claimed path is confirmed. ✓
- [ ] **S3.** Decoding command lines are at `runs/v2_headline_{arch}/logs/00_master.log`.
  - ✓ Both `00_master.log` files exist at expected paths. Additionally, per-cell logs (`{B_smoke,D1_closed_hotpot,D2_closed_wiki,E1_oracle_hotpot,E2_oracle_wiki}.log`) also present in both arch dirs. ✓
- [ ] **S4.** Analysis scripts are at `analysis/r1_paired_audit.py`, `analysis/a1_k1_vs_k5_ablation.py`, `analysis/uncertainty_line_audit.py`, `analysis/moonshot_explore.py`, and `analysis/llm_judge_audit.py`.
  - ✓ All five scripts exist in `emnlp_26_lengthConfound/analysis/` (confirmed via `ls` earlier). ✓
  - ⚠ **Note: `e7_bonferroni_and_spearman.py` is NOT listed** in this enumeration, but it is the script that produces the headline Table 1 (Bonferroni) and App. F (Spearman) — arguably the most important analysis script in the paper. ⚠ Should be added: `analysis/e7_bonferroni_and_spearman.py`.
- [ ] **S5.** The judge cache (800 per-call JSONs) is at `analysis/llm_judge_cache/`.
  - ✗ **DISCREPANCY:** `analysis/llm_judge_cache/` contains **1,600** JSONs (not 800), because each of 800 paired traces requires **two** judge calls — one for the oracle arm `a_o` and one for the closed arm `a_c`. Directory structure: `llm_judge_cache/{qwen2.5-7b-instruct, gemma-4-26b-a4b-it}/{hotpot, wiki}/*.json` → 400 calls per arch (200 traces × 2 arms) × 2 cells = 1,600 total. **Fix:** change "800 per-call JSONs" → "1,600 per-call JSONs (oracle + closed arm for each of 800 paired traces)" or simply "the per-call JSON cache" without the count.
- [ ] **S6.** Compute box: single NVIDIA A800 80 GB, wall-clock ~15 h.
  - ⚠ **"single NVIDIA A800 80 GB"** — the 80 GB GPU budget is consistent with §3.1 S9 ("single 80 GB GPU compute budget"). ✓
  - ⚠ **"wall-clock ~15 h"** — unverifiable from the audit doc; should be in `logs/00_master.log` if present. Trust but verify.
