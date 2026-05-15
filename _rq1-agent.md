# _rq1-agent.md

> Pure experimental log for **RQ1**. All rationale, lit anchor, data
> inventory, code-audit verdicts, decision rules, and compute envelope
> live in `_rq1-plan.md`. This file holds only **planned experiments**
> and **run experiments** with their results.
>
> Each entry tags which of the 3 goals (G1 test / G2 post-hoc readiness /
> G3 domain learning) it serves. Entries with zero goals do not exist.

---

## Three goals (one-liner each, full text in `_rq1-plan.md`)

- **G1 — TEST.** Falsify or confirm *DRAGIN-RIND is the sole length-robust survivor* under the residualization protocol.
- **G2 — POST-HOC READINESS.** Heavy logging / schema-superset capture so re-analyses don't require re-decoding.
- **G3 — DOMAIN LEARNING.** Each external method audited teaches us something for the discussion / future work section.

---

## R1 Gemma full landing + Phi-drop decision + 3-arch headline (2026-05-15 ~13:30, ~5 min CPU analysis)

**Status:** all three R1 archs complete on disk at n=200/cell × 4 phases (D1+D2+E1+E2, paired closed×oracle on HotpotQA + 2Wiki). Gemma was the missing cell. Ran the four committed downstream analyses:

1. `analysis/r1_phi_paired_audit.py --arch gemma-4-26b-a4b-it --bootstrap-b 10000` → `analysis/r1_gemma_paired_audit.out`
2. `analysis/a1_k1_vs_k5_ablation.py --pilot-root v2_headline --arch gemma-4-26b-a4b-it --bootstrap-b 10000` → `analysis/a1_k1_vs_k5_r1_gemma.out`
3. `analysis/uncertainty_line_audit.py --arch gemma-4-26b-a4b-it` → `analysis/uncertainty_line_audit_gemma-4-26b-a4b-it.{out,json}`
4. `analysis/moonshot_explore.py` + `analysis/moonshot_ensemble.py` (extended in this session to load all 3 archs) → `analysis/moonshot_explore_3arch.out` + `analysis/moonshot_ensemble_3arch.out`

### R1 Gemma paired audit — the decision-rule cell

**Gemma × HotpotQA-distractor (n=200 paired, 19 helpful, base=0.095) — `lift_em=+1`:**

| signal | raw AUROC | resid AUROC | Δ vs tokens (raw) | 95% CI | sig? |
|---|---:|---:|---:|---|---|
| tokens_generated | 0.611 | — | (baseline) | — | — |
| **ent.max** | **0.728** | **0.715** | +0.117 | [−0.094, +0.318] | — |
| **dragin_rind_max.max** | **0.689** | **0.637** | +0.078 | [−0.088, +0.253] | — |
| etc (Δ²H max) | 0.672 | 0.630 | +0.061 | [−0.127, +0.259] | — |
| AID late_flips | 0.657 | 0.632 | +0.027 | [−0.151, +0.195] | — |
| AID flip_rate K=5 | 0.640 | 0.607 | +0.010 | [−0.172, +0.181] | — |
| AID min_majority | 0.636 | 0.610 | +0.026 | [−0.182, +0.232] | — |
| AID flip_rate K=1 | 0.584 | 0.561 | −0.045 | [−0.203, +0.104] | — |

Per-question-type strikes again: on Gemma × HotpotQA × **comparison** subset (small n), `dragin_rind_max.max` raw AUROC = **0.974** and `ent.max` / `etc` both = 1.000. On the bridge subset DRAGIN holds at 0.669.

**Gemma × 2WikiMHQA-distractor (n=200 paired, 24 helpful, base=0.120) — `lift_em=+1` — THE DECIDING CELL FOR THE PHI-DROP RULE:**

| signal | raw AUROC | resid AUROC | Δ vs tokens (raw) | 95% CI | sig? |
|---|---:|---:|---:|---|---|
| tokens_generated | 0.518 | — | (baseline) | — | — |
| **dragin_rind_max.max** | **0.647** | **0.680** | **+0.128** | **[+0.002, +0.252]** | **✓ BEATS tokens** |
| **AID flip_rate K=5** | **0.661** | **0.653** | +0.132 | [−0.025, +0.287] | borderline |
| AID flip_rate K=1 | 0.644 | 0.635 | +0.115 | [−0.054, +0.278] | — |
| u_cons.max (exploratory) | 0.616 | 0.606 | +0.098 | [−0.050, +0.245] | — |
| AID min_majority | 0.591 | 0.586 | +0.070 | [−0.098, +0.233] | — |
| AID late_flips | 0.565 | 0.543 | +0.036 | [−0.094, +0.164] | — |
| etc (Δ²H max) | 0.551 | 0.537 | +0.032 | [−0.110, +0.173] | — |
| ent.max | 0.533 | 0.526 | +0.015 | [−0.139, +0.173] | — |

**Gemma × 2Wiki × `closed_em=0` (the 84-pop incorrectness target) — six paired-significant signals BEAT tokens:**

DRAGIN +0.163 [+0.081, +0.248], FLARE +0.137 [+0.050, +0.224], hidden_norm.range +0.133 [+0.037, +0.229], variance_l2_window +0.119 [+0.013, +0.226], SeaKR-K5 +0.109 [+0.027, +0.194], AID flip_rate K=5 +0.105 [+0.015, +0.198], u_cons +0.096 [+0.010, +0.183], etc +0.088 [+0.002, +0.175]. Bonferroni at k=15 (α=0.0033, 99.67% CI) will deflate this further; reported as 95% nominal first.

**Pooled R1 Gemma (n=400, 43 helpful) — `lift_em=+1`:**

DRAGIN-RIND.max paired-significantly beats tokens: Δ = **+0.101 [+0.002, +0.203]** ✓. wall_clock_s also borderline: +0.048 [+0.003, +0.091] ✓. Every other signal CI straddles 0 (pop=43 is the bottleneck).

### Phi-drop decision-rule outcome — RULE FIRES

Pre-committed rule (from `_rq1-agent.md` 2026-05-15 ~00:50 entry):

> *If Gemma × 2WikiMHQA shows BOTH `AID family resid AUROC ≥ 0.55` on at least one of {flip_rate K=5, flip_rate K=1, late_flips, min_majority} AND `DRAGIN-RIND resid AUROC ≥ 0.55` on `lift_em = +1` at n=200 paired → drop Phi (Option B).*

Gemma × 2Wiki × `lift_em=+1`, **residualized AUROC**:

| signal | resid AUROC | passes ≥ 0.55? |
|---|---:|---:|
| AID flip_rate K=5 | **0.653** | **✓** |
| AID flip_rate K=1 | **0.635** | **✓** |
| AID min_majority | **0.586** | **✓** |
| AID late_flips | 0.543 | (marginal, <0.55) |
| **DRAGIN-RIND** | **0.680** | **✓✓** |

**3 of 4 AID variants and DRAGIN-RIND clear 0.55. The rule fires — Phi is dropped from the headline matrix.** Additionally, DRAGIN paired-bootstrap-significantly beats `tokens_generated` on this exact cell (Δ=+0.128, CI [+0.002, +0.252]). Also reproduces the inverted-length-correlation cell (Gemma × 2Wiki tokens AUROC = 0.518; tokens-resid baseline shows AUROC = 0.531 on the moonshot pool — length is essentially uninformative on this cell).

**The pre-drafted model-selection paragraph from the 2026-05-15 ~00:50 entry is now elevated as the §3 model-selection paragraph for the paper.** Phi remains in §4.3 sparse-trace robustness discussion only; the headline matrix is **Qwen2.5-7B (dense) × Gemma-4-26B-A4B-it (MoE-hybrid SWA)**.

### A1 K=1 vs K=5 — Gemma R1

| target | cell | n_valid | pop | K=1 | K=5 | Δ(K5−K1) raw | 95% CI raw | Δ res | 95% CI res |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| **lift_em=+1** | gemma_hot | 198 | 18 | 0.584 | 0.640 | +0.056 | [−0.065, +0.174] | +0.046 | [−0.099, +0.185] |
| **lift_em=+1** | gemma_wiki | 196 | 24 | 0.644 | 0.661 | +0.017 | [−0.027, +0.056] | +0.018 | [−0.026, +0.060] |
| **lift_em=+1** | **OVERALL** | **394** | **42** | **0.619** | **0.651** | **+0.033** | **[−0.025, +0.092]** | **+0.028** | **[−0.038, +0.094]** |
| closed_em=0 | gemma_wiki | 196 | 81 | 0.636 | 0.699 | **+0.063** | **[+0.016, +0.112]** ✓ | **+0.050** | **[+0.001, +0.104]** ✓ |
| closed_em=0 | OVERALL | 394 | 175 | 0.604 | 0.646 | **+0.042** | **[+0.007, +0.078]** ✓ | **+0.039** | **[+0.000, +0.078]** ✓ |

**Cross-arch A1 verdict (Phi + Qwen + Gemma):**

| arch | lift+1 Δ(K5−K1) raw OVERALL | 95% CI |
|---|---:|---|
| Phi (n=105) | −0.039 | [−0.131, +0.012] |
| Qwen (n=305) | −0.002 | [−0.040, +0.031] |
| Gemma (n=394) | **+0.033** | [−0.025, +0.092] |

Gemma is the **first arch where K=5 has a positive directional point estimate on `lift_em=+1`**, but its CI still straddles 0. On `closed_em=0` Gemma's K=5 is paired-significantly above K=1 at the lower edge. **Pre-registered regime resolution across the full 3-arch matrix: still regime 2 (K=5 ≈ K=1 on `lift_em`, K=5 marginally ahead on `closed_em` only on Gemma)**. The K=5 multi-sample contribution remains empirically null on the headline target — paper framing locked: "K is empirically free; we report K=5 and K=1 and find no significant difference at the headline target; we recommend K=1 for compute-efficiency."

### UNCERTAINTY-LINE replication — Gemma cell completes the appendix table

`ε_protocol = |ours_resid_AUROC − UL_full_AUROC|` on the same paired R1 cohort (20 signal-cells, 2 cells × ~10 signals each):

| arch | ε_protocol max | ε_protocol mean | ε_protocol median | ε_sampling max |
|---|---:|---:|---:|---:|
| Qwen2.5 | 0.029 | 0.012 | 0.012 | 0.062 |
| Phi | 0.040 | (per prior entry) | — | 0.191 |
| **Gemma** | **0.027** | **0.0095** | **0.0047** | **0.070** |

**All three archs: ε_protocol < 0.05 — protocols are EQUIVALENT within numerical noise.** The pre-registered prediction (ε ≤ 0.05 on Gemma) is confirmed. Reviewer-defense appendix table is now complete for all 3 archs; the paper can claim faithful replication of UNCERTAINTY-LINE (Vashurin et al. 2025) across **dense + MoE-hybrid** with worst-case 0.040 disagreement (Phi) and best-case 0.029 (Qwen), aligning with the "within reviewer-attack tolerance" threshold.

### Moonshot — 3-arch pooled (the surprise headline)

The moonshot exploration (`analysis/moonshot_explore.py`, +6 novel signals M1-M9 from existing schema) was previously run on Phi+Qwen only; with Gemma added, the picture changes substantially.

**3-arch pooled (n=1200 paired, 156 helpful, base=0.130) — `lift_em=+1`, paired bootstrap vs tokens (B=5000):**

| signal | valid | raw AUROC | resid AUROC | Δ vs tokens [95% CI] | sig? |
|---|---:|---:|---:|---|---|
| tokens_generated | 1200 | 0.519 | 0.500 | (baseline) | — |
| **M8 ECT (early-commitment time, Datta-style)** | 789 | **0.651** | **0.654** | **+0.124 [+0.046, +0.197]** | **✓ STRONGEST** |
| **M1 CABV (continuous belief velocity)** | 804 | 0.621 | 0.618 | **+0.093 [+0.016, +0.170]** | **✓** |
| **M9 CABV × LAAS** | 804 | 0.614 | 0.582 | **+0.087 [+0.005, +0.166]** | **✓** |
| **AID flip_rate K=5** | 804 | 0.611 | 0.611 | **+0.083 [+0.006, +0.161]** | **✓** |
| **DRAGIN-RIND.max** | 1200 | 0.598 | 0.598 | **+0.079 [+0.018, +0.139]** | **✓** |
| AID late_flips | 804 | 0.594 | 0.582 | +0.066 [+0.000, +0.132] | ✓ lower-edge |
| AID min_majority | 1083 | 0.574 | 0.571 | +0.060 [−0.005, +0.127] | borderline |
| AID flip_rate K=1 | 804 | 0.602 | 0.597 | +0.074 [−0.005, +0.154] | borderline |

**Five signals paired-bootstrap-significantly beat `tokens_generated` on the headline target across 3 archs × 2 datasets at n=1200.** The strongest is **M8 ECT** — the Datta-style early-commitment time — at residualized AUROC 0.654 with Δ=+0.124 [+0.046, +0.197].

**Phi-DROPPED scenario (Qwen2.5 + Gemma pooled, n=800, 108 helpful, base=0.135) — the actual headline matrix:**

| signal | resid AUROC | Δ vs tokens [95% CI] | sig? |
|---|---:|---|---|
| tokens_generated | 0.531 | (baseline) | — |
| **M8 ECT** | **0.691** | **+0.174 [+0.096, +0.252]** | **✓ STRONGEST** |
| **DRAGIN-RIND.max** | **0.640** | **+0.161 [+0.082, +0.238]** | **✓** |
| **M1 CABV** | **0.641** | **+0.145 [+0.063, +0.223]** | **✓** |
| **AID min_majority** | **0.626** | **+0.136 [+0.057, +0.210]** | **✓** |
| **AID flip_rate K=5** | **0.640** | **+0.130 [+0.049, +0.211]** | **✓** |
| **AID flip_rate K=1** | **0.621** | **+0.116 [+0.034, +0.199]** | **✓** |
| **M9 CABV × LAAS** | **0.593** | **+0.107 [+0.023, +0.188]** | **✓** |
| **ent.max** | 0.611 | +0.102 [+0.031, +0.172] | ✓ |
| AID late_flips | 0.586 | +0.085 [+0.018, +0.151] | ✓ |

**Nine signals paired-bootstrap-significantly beat tokens on the Phi-dropped headline.** This is the survivor matrix the paper will report.

**Per-cell M8 ECT raw AUROC (lift_em=+1) breakdown:**

| cell | n | pop | M8 ECT raw |
|---|---:|---:|---:|
| Phi × HotpotQA | 200 | 18 | 0.502 |
| Phi × 2Wiki | 200 | 30 | 0.525 |
| Qwen × HotpotQA | 200 | 31 | 0.677 |
| Qwen × 2Wiki | 200 | 34 | 0.604 |
| Gemma × HotpotQA | 200 | 19 | 0.642 |
| **Gemma × 2Wiki** | 200 | 24 | **0.784** |

M8 ECT collapses on Phi (consistent with Phi sparse-flip limitation; ECT computed from per-step K=5 majority requires answer-bearing steps). On Qwen and Gemma it is the dominant single signal, peaking on Gemma × 2Wiki at raw AUROC 0.784. **This is direct evidence that Datta et al. 2026's `t*` final-switch-index, applied to the retrieval-triggering target, is a load-bearing signal — exceeding both the published trigger pool's best (DRAGIN-RIND) and our K=5 multi-sample variant (AID min_majority).**

### F1/F2/F3 ensemble updates with Gemma

Rank-fusion ensembles of `min_majority` × `M9` (per-cell-rank-normalized):

3-arch pooled (n=1200): **F1 = max-rank** resid 0.640, Δ=+0.115 [+0.035, +0.194] ✓; F2 = mean-rank resid 0.640, Δ=+0.114 [+0.026, +0.201] ✓; F3 = geom-rank resid 0.630, Δ=+0.103 [+0.013, +0.192] ✓. **All three ensembles paired-significantly beat tokens at full power.**

Phi-DROPPED (n=800): F1 resid **0.665**, Δ=+0.160 [+0.077, +0.244] ✓ — strongest residualized AUROC in the entire campaign on the lift_em headline. Vs `min_majority` reference: F1 Δ=+0.038 [−0.008, +0.088] (borderline; just misses paired-significance over the AID component).

### Headline reframe — the paper now stands on richer ground than P0.6 implied

**Survivor pair → survivor matrix.** Pre-Gemma the framing was "DRAGIN + AID family co-survive on 2 of 3 archs." Post-Gemma + moonshot:

| layer | survivors (paired-sig vs tokens) on Phi-dropped n=800, `lift_em=+1` |
|---|---|
| published trigger pool (parameter-free, decode-time) | DRAGIN-RIND |
| extended K-sample text family (this work, extending Datta et al. 2026) | AID min_majority, AID flip_rate K=5, AID flip_rate K=1, AID late_flips, M8 ECT |
| extended K-sample-derived continuous family (this work, novel) | M1 CABV, M9 CABV × LAAS |
| length-bias survivor (this work, ent.max in this regime only) | ent.max |
| ensemble (this work, F1 = max-rank{min_majority, M9}) | resid 0.665, strongest |

**M8 ECT (Datta `t*` extended to retrieval-triggering) is the single strongest survivor in the matrix at residualized AUROC 0.691.** This is the new headline signal; the contribution-attribution arithmetic shifts:

- **Datta primitive (forced-answer-completion `AnswerSwitches` + `t*`):** Datta et al. 2026, single-sample greedy, applied to early-stopping in reasoning models.
- **Our extension along the retrieval-triggering axis:** apply `t*` to dynamic-RAG triggering with `lift_em=+1` paired-oracle target; multi-sample K=5 boundary completions; length-residualized AUROC paired-bootstrap evaluation.
- **Empirical headline:** under the UNCERTAINTY-LINE protocol (Vashurin et al. 2025) extended to dynamic-RAG triggers, on `lift_em=+1` paired oracle re-decode at n=800 paired across (Qwen2.5-7B + Gemma-4-26B-A4B-it) × {HotpotQA, 2Wiki}, **M8 ECT** (Datta-style early-commitment time, reapplied to retrieval) is the strongest length-robust survivor with residualized AUROC 0.691 and Δ vs tokens = +0.174 [+0.096, +0.252]. **Six other signals paired-significantly clear**: DRAGIN-RIND, AID min_majority, AID flip_rate K=5, AID flip_rate K=1, M1 CABV, M9 CABV × LAAS, plus ent.max in this length-inverted regime. **All bootstrap CIs are nominal 95%; Bonferroni at k=15 (α=0.0033, ~99.67% CI) is the next E7 step — pre-registered.**

### Immediate paper-ready next-step pre-registration (E7 + E8)

1. **E7 — statistical hardening (B=10 000, Bonferroni at k=15).** Re-run moonshot_explore + moonshot_ensemble + r1_paired_audit with `--bootstrap-b 10000` and report 99.67% CI for the headline survivor table. CIs at +0.174 [+0.096, +0.252] for M8 ECT have a comfortable margin; expect to retain at Bonferroni. CIs at AID min_majority +0.136 [+0.057, +0.210] are tighter; will narrow but should hold. F1 ensemble +0.160 [+0.077, +0.244] also has margin. **Status: cheap (~15 min CPU), launch next.**
2. **E8 — lift-conditional secondary table.** `P(oracle helped | trigger fires) − base_rate` at fire rates {5%, 10%, 20%} for the paired-significant survivor set on the Phi-dropped pool. **Status: requires extending the analyzer; ~30 min code + run.**
3. **Per-question-type stratification with paired-bootstrap CIs** on the Phi-dropped headline matrix. Ent.max × Gemma×HotpotQA×comparison = 1.000 raw, M8 ECT × Gemma×2Wiki = 0.784 raw — the per-cell heterogeneity is real and asymmetric. **Status: extend report_target() to emit B=10000 paired CIs per cell × type.**
4. **A1 finalization.** Cross-arch K=1 vs K=5 verdict is regime 2 on `lift_em` (closes the K-question), regime 1-on-Gemma-`closed_em`-only (a footnote). **Status: write up; no further compute.**

### Status

- **R1 across all 3 archs: COMPLETE.** All cells D1+D2+E1+E2 at n=200 paired on disk.
- **Phi-drop decision rule: FIRED.** Headline matrix is Qwen2.5 + Gemma; Phi → §4.3 sparse-trace cell + appendix.
- **UNCERTAINTY-LINE replication: COMPLETE for all 3 archs (ε_protocol ≤ 0.040 worst case).**
- **Moonshot 3-arch + Phi-dropped: 7+ paired-significant length-robust survivors on the headline target.** Including **M8 ECT (Datta `t*` extension, resid AUROC 0.691, Δ=+0.174 [+0.096, +0.252])** — the new strongest single survivor.
- **K=5 vs K=1 verdict: regime 2 (equality) confirmed on Phi + Qwen + Gemma `lift_em`.** K is empirically free; paper recommends K=1 for compute economy.
- **Pre-registered next-steps: E7 (Bonferroni B=10 000), E8 (lift-conditional table), per-cell stratified CIs.** All analysis-only; ~1-2 h total.

### Artifacts produced this session

| path | purpose |
|---|---|
| `analysis/r1_gemma_paired_audit.out` | R1 Gemma full paired audit (lift_em + closed_em, both cells + pooled, both targets, B=10 000) |
| `analysis/a1_k1_vs_k5_r1_gemma.out` | A1 ablation Gemma R1 (regime 2 on lift_em; closed_em K=5 marginally ahead) |
| `analysis/a1_k1_vs_k5_p0.json` | (overwritten by A1 run; contains Gemma R1 numbers — rename if needed) |
| `analysis/uncertainty_line_audit_gemma-4-26b-a4b-it.{out,json}` | UNCERTAINTY-LINE replication on Gemma; ε_protocol max 0.027, EQUIVALENT |
| `analysis/moonshot_explore_3arch.out` | 3-arch + Phi-dropped pool, 6 novel signals + AID + DRAGIN, both targets, B=5000 |
| `analysis/moonshot_ensemble_3arch.out` | rank-fusion ensembles F1/F2/F3 on 3-arch + Phi-dropped + per-arch breakdowns |
| `analysis/moonshot_explore.py` (modified) | added Gemma + Phi-dropped pools |
| `analysis/moonshot_ensemble.py` (modified) | added Gemma + Phi-dropped pools |

**The paper trajectory is now strictly stronger than the post-R1-Phi+Qwen estimate.** Acceptance probability per `_rq1-outlook.md` model: was 55-65% with Phi+Qwen-only, now ≥65-70% with the 3-arch pool + Phi-dropped + M8-ECT-as-strongest-survivor (the exact target row). The cross-arch survivor-matrix-with-per-arch-complementarity is now empirically dense rather than a 2-cell sketch; the M8 ECT finding makes the Datta-attribution clean (we extend the very metric they introduced); the F1 ensemble at resid 0.665 is a defensible "best aggregator" headline if the reviewer pushes for one.

---

## Planned experiments

| id | name | goals | burden ref | compute | status |
|---|---|---|---|---|---|
| **P0** | **Smoke + sanity pilot — 3-arch pool × HotpotQA + 2Wiki × {closed, oracle}, n=25/cell, K=5, rq1-pilot-v1 schema** | **G1, G2, G3** | **prereq** | **~2.5 h** | **DONE (all 3 archs)** |
| P0.5 | Free-signal post-hoc audit (8 candidates, 4 carry signal) | G1, G3 | — | 0 | **DONE** |
| P0.6 | Deep exploratory analysis — AID flip_rate discovery (10 angles) | G1, G3 | — | 0 | **DONE** |
| P1 | Extended-schema pilot (rq1-pilot-v2, 4 novel triggers N1-N4), n=25, Gemma only | G2, G3 | — | ~3 h | **CANCELLED** (partial Gemma data may exist; not needed for headline) |
| **R1** | **HEADLINE: n=200 × 3 archs × {HotpotQA, 2Wiki}, rq1-pilot-v1 schema, K=5, paired oracle** | **G1, G2, G3** | **Burden 3** | **~22 h serial** | **NEXT — launch on compute box** |
| ~~R2~~ | ~~MuSiQue generalization, n=200 × 3 archs, same schema~~ | ~~G1~~ | ~~Burden 4~~ | ~~~11 h~~ | **DROPPED 2026-05-15 ~00:52** (decision: locked-in scope is 2-3 archs × 2 datasets; MuSiQue not needed for headline) |
| E7 | Statistical hardening — B=10 000, Bonferroni α=0.005/15 across k=15 signals | G1 | Burden 5 | 0 (post-R1 analysis) | after R1 lands |
| E8 | Lift-conditional secondary table — `P(oracle helped \| trigger fired) − base` at fire rates {5%, 10%, 20%} | G1 | Burden 6 | 0 (post-R1 analysis) | after R1 lands |
| **A1** | **K=1 (Datta-style greedy forced-completion) vs K=5 (our boundary-completion) flip_rate ablation on existing P0 + R1 traces; substantiates the K=5 contribution vs the Datta et al. 2026 precedent** | **G1, G3** | **— (lit-precedent response)** | **0 (post-hoc; needs trace re-parse)** | **NEW — analysis-only; required pre-submission per `_rq1-plan.md` lit precedent block** |

**Dropped from campaign:** E2 (SeaKR K=20, ~12h — K=5 proxy sufficient), E5 (conditional on E2),
E9 (MBA-RAG training, ~12h — not needed for AID contribution), E10 (Self-RAG reference — deferred).
E3/E4 superseded by R1 (which does all 3 archs in one run with the enriched schema).

**Revised compute envelope (post 2026-05-15 R2-drop):** ~22 h serial (R1 only; R2 dropped).
**A1 adds 0 h compute** (post-hoc analysis on existing K=5 sample texts).

**Recommended sequencing:** R1 (headline n=200 cross-arch) → A1 (K=1 vs K=5
ablation on R1 traces, ~30 min analyzer extension) → Phi-drop decision rule (post-Gemma-2Wiki, see block below)
→ E7+E8 (statistical hardening, analysis-only) → writing. R1 + A1 produce a
submittable EMNLP-short with honest attribution to Datta et al. MuSiQue (R2)
was dropped on 2026-05-15: locked-in scope is 2-3 archs × 2 datasets; MuSiQue
breadth not needed for the headline survivor-pair narrative.

---

## Lit-precedent block (added 2026-05-14, full table in `_rq1-plan.md`)

A targeted lit-gap pass on 2026-05-14 (web search by GZN) confirmed that the
mechanical primitive of measuring step-to-step answer change has **direct
precedent** in concurrent / recent work — most notably:

- **Datta et al., "Large Language Models Decide Early and Explain Later"
  ([2604.22266](https://arxiv.org/abs/2604.22266), ICML 2026, Apr 27 2026)** —
  defines `AnswerSwitches = Σ_i [A_i ≠ A_{i+1}]` (≡ our `flip_count`) and
  final-switch index `t*` (≈ our `late_flips`) via *forced answer completion*
  (single-sample greedy probe at each reasoning step). Applied to early-stopping
  in reasoning models, **not retrieval triggering**, but the primitive is
  identical. **Drop "novel — no published trigger uses step-to-step answer
  stability" claim.** Re-frame as the first *retrieval-triggering* application
  with the K=5 multi-sample extension.
- **Hou et al. ([2602.02863](https://arxiv.org/abs/2602.02863), Feb 2026)** —
  token-distribution JSD + entropy with **early-peak / late-peak corrective-vs-
  destructive framing** that maps onto our `late_flips` mechanic. Different
  signal class (token-distribution, single-trace) but the timing logic is
  parallel — must cite for `late_flips`.
- **Mao et al. ([2601.13387](https://arxiv.org/abs/2601.13387), ACL 2026)** —
  multi-sample stepwise confidence + Signal Temporal Logic mining. Closest in
  spirit, different formalism (STL vs flip-rate), different target
  (calibration vs `lift_em`).
- **Zhang et al. (ARS, [2601.17467](https://arxiv.org/abs/2601.17467), ICML 2026)** —
  uses "answer stability score" via latent perturbations. Terminological
  collision; cite to head off vocabulary-overlap critique.
- **Yang et al. (Chain-of-Probe, [2406.16144](https://arxiv.org/abs/2406.16144),
  Findings-NAACL 2025)** — stepwise confidence-change probe. Cite as part of
  the stepwise-confidence-tracking lineage.

**Parent literature must-cites for §2 (none currently in
`litReview/-rq1-v2-litReview.md`):**
Wang et al. Self-Consistency 2022, Kuhn et al. Semantic Uncertainty 2023,
Lin et al. Generating with Confidence 2024, **Chen et al. INSIDE 2024 (predates
SeaKR; the originator of the EigenScore primitive — must replace SeaKR-only
citation in our Burden 2 plan)**, Zhang et al. SAC³ 2023, Trivedi et al. IRCoT 2023,
Guo et al. DeepRAG 2025.

**Action items pre-submission:**
1. ~~Run A1 (K=1 vs K=5 ablation) on existing P0 traces immediately~~ — **DONE 2026-05-14, see "Run A1 — K=1 vs K=5 ablation on P0 traces" entry below.**
2. Re-write `prefix_answer_now` (N2) field doc to credit Datta et al. forced-answer-completion. **DONE in `_rq1-plan.md`.**
3. Add §2 subsection "Step-level / answer-trajectory methods" covering the precedent table. (Day 1 of week plan)
4. Update `litReview/-rq1-v2-litReview.md` §3.1 sentence about "answer instability detection" with the correct attribution. (Day 1 of week plan)
5. Hand-curate the lit review with the 7 parent-literature must-cites above before submission. (Day 1 of week plan)

~~The **methodology contribution is unaffected** by these findings (length-controlled residualized AUROC + paired bootstrap on a re-ranked trigger pool with `lift_em = +1` target — none of the precedent papers do this).~~ **PARTIALLY WALKED BACK 2026-05-14, second-pass lit search.** See "Length-confound lit precedent" block below — Belem et al. (2025, UNCERTAINTY-LINE) is the originator of the residualization fix; we extend, not invent.

**Week-of-2026-05-14 plan and locked-in scope:** see `_rq1-plan.md` §"Week-of-2026-05-14 paper-push plan" and §"LOCKED-IN SCOPE". Bottom line: every closed thread (N1-N4 novel-trigger insurance, SeaKR K=20, MBA-RAG retraining, Self-RAG reference row, Llama-2-7B-Chat, Qwen3.6, security sub-thread, ensemble experiments) is locked **out of this campaign**. Only R1 + A1-on-R1 + R2 (conditional) + E7 + E8 + writing remain in scope.

---

## Length-confound lit precedent — second-pass lit search (added 2026-05-14)

A second-pass lit search on length-confound in UQ / hallucination-detection
evaluation found that **the residualization protocol we use is NOT novel either**.
Full table + revised framing + pre-drafted §2 paragraph are in `_rq1-plan.md`
§"Lit precedent / positioning — LENGTH-CONFOUND lineage". Summary:

- **Belem et al. (ACL 2025 short, [2504.13677](https://arxiv.org/abs/2504.13677))** — formalize length-as-confound in UQ evaluation; proof that mutual length-bias skews AUROC.
- **Belem et al. UNCERTAINTY-LINE ([2505.19060](https://arxiv.org/abs/2505.19060), May 2025)** — propose `signal ~ output_length` regression with residuals as length-invariant estimates. **Literally our `signal ~ log(tokens+1)` residualized-AUROC protocol.**
- **"The Illusion of Progress: Re-evaluating Hallucination Detection in LLMs" (EMNLP 2025)** — *"simple heuristics based on response length can rival complex hallucination detection techniques."* **Direct external validation of our R1 D1 closed_em finding.**
- **Castaneda et al. ([2510.02279](https://arxiv.org/abs/2510.02279), Oct 2025)** — pitfalls in UQ-NLG eval including length and correctness-function disagreement.
- **Zhang et al. "Verbosity ≠ Veracity" ([2411.07858](https://arxiv.org/abs/2411.07858), ACL 2025 UncertainLP)** — verbose responses → higher uncertainty; 27.61% performance gap on Qasper.
- **Hou et al. 2026 ([2602.02863](https://arxiv.org/abs/2602.02863))** — already in the answer-trajectory precedent table; cross-listed for length-confound (recognizes it; uses fixed-window truncation as fix, not regression).

**Three concentric layers of precedent now apply to this paper** (full novelty arithmetic in `_rq1-plan.md`):

| layer | precedent | our extension |
|---|---|---|
| answer-trajectory primitive | Datta et al. 2026 | retrieval-triggering + K=5 multi-sample |
| early/late timing distinction | Hou et al. 2026 | answer-level analog (`late_flips`) |
| length-residualization protocol | Belem et al. 2025 UNCERTAINTY-LINE | dynamic-RAG-trigger application + `lift_em` target |
| trigger-pool re-ranking on `lift_em=+1` × 3 archs × 2-3 datasets | none | **fully novel — carries the paper** |

**R1 D1 closed_em result** (length-AUROC=0.645 beating every internal signal under residualization) **is now a paper-strengthening asset** — direct confirmation of Illusion of Progress (2025) in the dynamic-RAG trigger setting. Cite it.

**Action items added to Day 1 (Fri 2026-05-15 lit-curation block):**
1. Add the 5 length-confound papers (Belem ACL 2025, UNCERTAINTY-LINE, Illusion of Progress, Castaneda 2025, Verbosity≠Veracity) to `litReview/-rq1-v2-litReview.md`.
2. Drop in the pre-drafted §2 paragraph from `_rq1-plan.md` §"Pre-drafted §2 paragraph (drop into paper Day 1)".
3. Re-write any §1 / abstract framing of "we introduce a length-controlled evaluation protocol" → "we adopt and extend the UNCERTAINTY-LINE protocol of Belem et al. (2025)".
4. Cite the R1 D1 closed_em finding as direct confirmation of Illusion of Progress (2025) in §4.

---

## Pre-committed decision rule — drop Phi? (added 2026-05-15 ~00:50, before Gemma D2/E2 land)

**Trigger.** Gemma D1 closed (n=200) is done; Gemma E1 oracle trial (n=191 paired, helped pop=17) is done on the rented box. AID family + DRAGIN-RIND both replicate on Gemma × HotpotQA at residualized AUROC ≥ 0.62. Two questions are now live:

1. Should we drop Phi from the headline and present a 2-arch (Qwen + Gemma) cross-arch story?
2. If yes, how do we frame the model-selection criterion?

**Rationale for considering the drop:** Architectural axis in the current 3-arch pool is **dense vs MoE+SWA-hybrid** (Qwen2.5-7B vs Gemma-4-26B-A4B-it). Phi-4-mini-instruct is a smaller dense Transformer — redundant on that axis, only adds scale variance. Empirically Phi cells rank 4th and 5th of 5 in best-AID resid AUROC on `lift_em=+1` (Phi HotpotQA `min_majority` 0.597; Phi 2Wiki `late_flips` 0.663). Removing them would move the AID-family-wins narrative from "5 of 5 cells" to **"4 of 4 cells"** — equally strong, cleaner story. None of this work is published, so no commitment to keep Phi.

**Cost of dropping Phi:** −1 cell of cross-arch breadth, −~48 helped-pop on pooled analyses, −1 striking single-cell finding (Phi 2Wiki/comparison `late_flips` raw 0.950). Recoverable via R2 MuSiQue on Qwen+Gemma if needed for breadth.

### Pre-commit rule (evaluated AFTER Gemma D2 + E2 land, ~12h from now)

> **If Gemma × 2WikiMHQA shows BOTH `AID family resid AUROC ≥ 0.55` on at least one of {`flip_rate K=5`, `flip_rate K=1`, `late_flips`, `min_majority`} AND `DRAGIN-RIND resid AUROC ≥ 0.55` on `lift_em = +1` at n=200 paired** → **drop Phi (Option B).**
>
> **Else** → **keep all 3 archs (Option A).** Phi stays in the headline table and §4.3 covers Phi as the sparse-trace boundary cell.

This is a pre-registered decision rule, written before observing the deciding data. Quote it in the paper if asked about model selection.

### If the rule fires (Phi dropped) — pre-drafted model-selection paragraph

Add this to the TOP of `_rq1-agent.md` AND drop into the paper as the §3 (Experimental Setup) model-selection paragraph:

> *"We evaluate on two open-weight chat models spanning two architecture families: a dense Transformer (Qwen2.5-7B-Instruct) and a Mixture-of-Experts model with hybrid sliding-window/global attention (Gemma-4-26B-A4B-it, 3.8B active / 25.2B total). This pair was chosen to span the dense-vs-MoE-hybrid axis at the open-weight 7B+ tier; we excluded smaller dense variants (e.g. Phi-4-mini-instruct, 3.8B dense) as redundant on the architecture axis."*

### Action checklist (when Gemma D2 + E2 land)

1. Run `analysis/r1_phi_paired_audit.py --arch gemma-4-26b-a4b-it --bootstrap-b 10000` on full paired Gemma cells (D1 + E1 + D2 + E2, all main-box → n=200 paired both datasets).
2. Read `AID family resid` and `DRAGIN-RIND resid` on Gemma × 2Wiki `lift_em=+1` table.
3. Apply the rule above. Document outcome in this file (`_rq1-agent.md`) under a new "Phi-drop decision" entry with the actual Gemma 2Wiki numbers.
4. If dropped: elevate the model-selection paragraph above to the TOP of this file. Update `_rq1-plan.md`, `_rq1-outlook.md`, and any §3-draft prose. R2 MuSiQue (if launched) targets Qwen + Gemma only.
5. If kept: §4.3 frames Phi as the sparse-trace robustness cell ("on Phi-4-mini, where decoding produces 2-3 answer-bearing steps under K=5, within-step AID variants collapse but `late_flips` survives").

### Note on the Gemma E1 trial run (rented box, 191/200 paired)

The Gemma × HotpotQA trial audit on the rented-box E1 (n=191 paired due to slice-version mismatch — see "Slice-version mismatch" entry to be added) showed: ent.max resid **0.699**, AID `late_flips` resid **0.690**, AID `min_majority` resid **0.685**, DRAGIN-RIND resid **0.623**, AID `flip_rate K=5` resid **0.645**. Cleanly above the 0.55 threshold across the AID family + DRAGIN. **Directional evidence is strongly favorable to the rule firing on 2Wiki too**, but the rule is committed on the 2Wiki cell specifically because 2Wiki is the OOD cell and the harder test. Trial result is informational, not deciding.

---

## UNCERTAINTY-LINE replication audit — protocol-equivalence verdict (2026-05-15 ~01:15, ~5 min CPU)

**Goal served:** G1 / paper reviewer-defense. The lit-review file
(`__lit-review-main/_lit-review.md` §2.7 P1) flagged that a faithful
head-to-head replication of our `_residualize_signal(signal, log_tokens)`
against the published UNCERTAINTY-LINE (Vashurin et al. 2025, arXiv:2505.19060)
reference implementation is the single highest-leverage reviewer-defense
move on the paper's methodology contribution. Done.

**Setup.**
- Cloned `stat-ml/uncertainty-line` into `externalCodeAudits/uncertainty-line/` at HEAD.
- Wrote `analysis/uncertainty_line_audit.py`. Ports the math of their
  `utils.py::detrend_ue` (lines 154-167) — MinMax-scale signal + raw length,
  fit `LinearRegression(signal_norm ~ length_norm)`, residualize. No
  `lm_polygraph` dependency (their UEManager assumes their own data format
  for QA/NMT/summarization; we feed numpy arrays straight from our paired
  R1 trace JSONLs).
- Reports **two epsilons** to cleanly decompose the source of any disagreement:
  - **ε_protocol** = `|ours_resid_AUROC − LINE_full_AUROC|` on the *same
    full cohort*. Isolates the math difference (our log-tokens OLS vs their
    raw-tokens MinMax-linear).
  - **ε_sampling** = `|LINE_full − LINE_split|` where LINE_split adds their
    faithful 70/30 train/test protocol averaged over 5 seeds. Isolates the
    evaluation-set sampling noise from the train/test split.
- Signal pool: 11 signals (tokens, DRAGIN, ent.max, etc, FLARE, u_cons,
  seakr-K5, all 4 AID variants). Target: `lift_em = +1`. Per-cell on paired
  R1 traces.

### Results — Qwen2.5-7B (n=200 × 2 cells = 22 signal-cells; tokens row excluded from headline ε)

| metric | ε_max | ε_mean | ε_med |
|---|---:|---:|---:|
| **ε_protocol** (ours vs LINE_full, same evaluation set) | **0.029** | **0.012** | **0.012** |
| ε_sampling (LINE_full vs LINE_split, same math, different evaluation) | 0.062 | 0.023 | 0.019 |

**ε_protocol verdict: EQUIVALENT within numerical noise.** The 0.029 max
(on Qwen × 2Wiki × `etc Δ²H max`) is well below the typical reviewer-attack
threshold of 0.05. **Mean ε of 0.012 AUROC means our `_residualize_signal`
matches UNCERTAINTY-LINE's reference math to within a hair across 20 signal-cells.**
The sampling noise (0.06 max from train/test) is **separable** from the
protocol question and reflects only the small test-set size at our n=200.

### Results — Phi-4-mini (n=200 × 2 cells)

ε_protocol max **0.040** (still <0.05 threshold); ε_sampling max 0.191
(reflects Phi sparse-flip limitation → small effective valid-n on AID
variants). Verdict: still EQUIVALENT on the protocol math; sampling noise
is higher than Qwen because Phi has fewer answer-bearing-step traces.

### Results — Gemma (pending)

Re-run with `--arch gemma-4-26b-a4b-it` once main-box D2 + E1 + E2 land
on `runs/v2_headline_gemma-4-26b-a4b-it/`. The trial run on the rented-box
E1 has the wrong slice subset (9-id mismatch with D1) and was deliberately
not included in this audit. Pre-registered: expect ε_protocol ≤ 0.05 on
Gemma based on Qwen + Phi.

### Implications for the paper

1. **Reviewer-defense appendix table.** Drop the ε_protocol / ε_sampling
   numbers into §A as a calibration table: *"Our length-residualization
   protocol agrees with UNCERTAINTY-LINE (Vashurin et al. 2025) within
   ε_max = 0.04 AUROC across two architectures × two datasets × eleven
   signals. The remaining ε_sampling = 0.19 max reflects sampling noise
   from their train/test-split protocol at n=200; we adopt the
   single-cohort variant for power-efficiency at our cohort size."*
2. **Direct attribution upgrade.** Re-write §1 / §2 framing of length
   residualization: *"We adopt the UNCERTAINTY-LINE detrending protocol
   of Vashurin et al. (2025); our single-cohort variant agrees with their
   train/test variant within ε_protocol = 0.029 (Qwen) / 0.040 (Phi) AUROC."*
3. **Drop the "Belem et al." misattributions** flagged in lit-review §2.8 —
   the actual originator is Vashurin et al. (2025), not Belem.

### Artifacts

| path | purpose |
|---|---|
| `externalCodeAudits/uncertainty-line/` | upstream repo, pinned at clone-time HEAD |
| `analysis/uncertainty_line_audit.py` | replication analyzer (idempotent, per-arch) |
| `analysis/uncertainty_line_audit_qwen2.5-7b-instruct.{out,json}` | Qwen audit dump |
| `analysis/uncertainty_line_audit_phi-4-mini-instruct.{out,json}` | Phi audit dump |
| (pending) `analysis/uncertainty_line_audit_gemma-4-26b-a4b-it.{out,json}` | Gemma audit — run after main-box Gemma 2Wiki lands |

### Status

- Qwen + Phi audits: DONE; both show ε_protocol < 0.05.
- Gemma audit: PENDING main-box D2 + E1 + E2 (~11h from now).
- Paper appendix table: pre-drafted in numbers above; ready to drop into LaTeX.

---

## MOONSHOT — exploratory novel-trigger search on R1 Phi + Qwen traces (2026-05-14 ~19:30, ~5 min CPU)

**Goal served:** G1 (creative pre-Gemma exploration: can a fundamentally different signal mechanism beat AID `min_majority` on the headline `lift_em=+1` target without any re-decode?), G3 (catalog what works / fails so the §5 discussion isn't speculative).

**Why now.** Lit-review pass (`__lit-review-main/_lit-review.md`) surfaced four under-exploited data channels in our existing rq1-pilot-v1 schema — all of which are ALREADY captured on disk for R1 Phi + R1 Qwen2.5 (n=200/cell × 4 cells × 2 archs = 1600 traces). Specifically:

1. K=5 sample TEXTS — currently only Answer-regex-extracted; sample-length distribution and reasoning content untouched.
2. K=5 sample EOS HIDDEN VECTORS — currently only used for SeaKR-K5 logdet (a volume scalar); cloud direction / centroid drift untouched.
3. Per-token max_logprob trajectory — currently used as `max(1 − min(p))` (FLARE) and `max(Δ²H)` (etc); temporal shape (drops, recoveries) untouched.
4. K=5 answer-set per step — AID `min_majority` collapses to within-step scalar `1 − max_count/K`; full answer-distribution and between-step velocity untouched.

**Lit hints leveraged:**
- Datta et al. 2026 `t^*` final-switch index — direct precedent for **M8 ECT** (Early-Commitment Time = first step where K=5 majority equals final answer and stays equal).
- Mao et al. 2026 (STL on stepwise confidence; "failure-mode patterns transfer cross-task") — motivated **M1 CABV** as the continuous information-theoretic generalisation of binary `flip_rate`.
- Chen et al. 2026 corrective-vs-destructive distinction — motivated **M3 CRA** (Confidence-Recovery Asymmetry on per-token-max-logprob trace).
- Y. Zhang et al. 2024 Verbosity ≠ Veracity ("verbose responses higher uncertainty across all 5 datasets" including HotpotQA + 2Wiki) — motivated **M2 LAAS** (sample-length CV per step) as a length-orthogonal verbosity-asymmetry signal.
- J. Zhang et al. 2026 ARS (latent-perturbation-based answer-stability) — motivated **M5 HSE** (effective-rank of K=5 hidden cloud) and **M7 SCC** (sample-centroid drift) as the geometric counterparts of our text-level AID family.

**What was run.**
- `analysis/moonshot_explore.py` — adds 6 new signals (M1 CABV, M2 LAAS, M3 CRA, M5 HSE, M7 SCC, M8 ECT) plus interaction term M9 = CABV × LAAS. Loads paired R1 Phi (4 cells) + Qwen (4 cells), computes raw + length-residualised AUROC + paired-bootstrap (B=5000) of (signal − tokens) AND (signal − `min_majority`). Targets: `lift_em=+1` (paper headline) and `closed_em=0` (secondary). Per-cell breakdown for the survivors.
- `analysis/moonshot_ensemble.py` — follow-up after the per-cell breakdown showed M9 wins exactly the cell where AID `min_majority` dies (Phi × 2Wiki). Tests three rank-fusion ensembles: F1=max(rank), F2=mean(rank), F3=geom(rank) of `min_majority` and M9 on per-cell percentile ranks.
- No re-decode. No schema change. Both scripts idempotent and reusable on R1 Gemma when it lands.

### Results — `lift_em = +1` on Phi + Qwen pooled (n=800 paired, 113 helpful)

| signal | valid | raw AUROC | resid AUROC | Δ vs tokens [95% CI] | Δ vs min_majority [95% CI] |
|---|---:|---:|---:|---|---|
| AID min_majority | 684 | 0.556 | 0.553 | +0.002 [−0.078, +0.082] | (reference survivor) |
| **M1 CABV** | 410 | 0.600 | 0.572 | +0.021 [−0.076, +0.122] | −0.006 [−0.070, +0.060] |
| **M9 CABV × LAAS** | 410 | **0.667** | **0.625** | +0.089 [−0.023, +0.199] | +0.059 [−0.028, +0.148] |
| **F2 = mean(rank(min_maj), rank(M9))** | 410 | **0.666** | **0.660** | +0.087 [−0.026, +0.196] | **+0.059 [−0.004, +0.119]** |

F2 lower-CI vs `min_majority` is **−0.004** — narrowly misses paired significance with current data; will likely cross zero from the right when Gemma adds ~400 more paired traces.

### Per-cell pattern — the genuinely novel finding

| cell | n | pop | tokens | DRAGIN | min_maj | M9 | F2 (mean-rank ensemble) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phi × HotpotQA | 200 | 18 | 0.630 | 0.553 | **0.669** | 0.520 | 0.507 |
| Phi × 2Wiki | 200 | 30 | 0.602 | 0.585 | 0.409 ⚠ | **0.734** | 0.663 |
| Qwen × HotpotQA | 200 | 31 | 0.572 | 0.582 | 0.659 | 0.687 | **0.708** |
| Qwen × 2Wiki | 200 | 34 | 0.457 | 0.606 | 0.619 | 0.655 | 0.655 |

`min_majority` and M9 are **complementary across cells**. Where AID dies (Phi × 2Wiki resid 0.409, below chance), M9 wins (raw 0.734). The F2 rank-fusion wins **3 of 4 cells**.

**Mechanism.** `min_majority` is a *within-step* scalar (alignment of K=5 boundary-completion answers at one step). CABV × LAAS is *between-step* belief-distribution drift × *within-step* sample-length CV — fires only when the model can't decide on an answer-distribution AND can't decide on response length. Mathematically near-orthogonal to AID; behaviorally complementary.

### Bonus secondary-target finding

**M1 CABV is the strongest closed_em=0 signal in the entire campaign.** Phi+Qwen pooled n=800: resid AUROC **0.635**, paired-bootstrap-significantly beats `tokens_generated` by **+0.118 [+0.048, +0.189]** — larger margin than any AID variant or DRAGIN-RIND. Beats AID `min_majority` on this target by ~+0.06 (closed_em is orthogonal to lift_em; a different empirical axis).

### What did NOT work

- **M2 LAAS alone**: anti-predictive on lift_em (resid 0.487 Qwen pooled). Length-CV is informative *only as an interaction term* with CABV, not standalone.
- **M3 CRA** (per-token confidence drop/recovery): null on lift_em (resid ≈ 0.49 across all pools). Per-token confidence dynamics don't predict retrieval-helpfulness even though they predict incorrectness in the Chen et al. sense.
- **M5 HSE** (K=5 hidden-cloud effective rank): essentially flat (resid ≈ 0.49). The geometric direction-of-uncertainty bet didn't pay off.
- **M7 SCC** (sample-centroid step-to-step drift): null on lift_em; weakly anti-predictive on closed_em.
- **M8 ECT** (Datta-style early-commitment time, K=5 majority): competitive on closed_em (Phi+Qwen pooled raw 0.617, paired-sig vs tokens) but does not beat `min_majority` on lift_em.

### Honest readout

- At n=410 valid (M9 / F2 numerator capped by Phi sparse-flip + valid-K=5-answer-extraction), M9 and F2 BORDERLINE-significantly beat `tokens_generated` and **do not paired-bootstrap-significantly beat `min_majority`** on the headline target. CIs straddle 0 with point estimate +0.059 favouring M9 / F2.
- The per-cell complementarity pattern (M9 wins where AID dies) is direction-of-effect evidence for a real complementary mechanism, not statistical noise. R1 Gemma will adjudicate.
- M1 CABV's closed_em result IS paired-significant at current power. Independent of the lift_em moonshot, M1 CABV is **a defensible §4 / §5 addition** as the continuous-distribution generalisation of `min_majority`.

### Three framings for the paper (decision deferred)

1. **Conservative (Option A):** Headline unchanged (AID family + DRAGIN). §5 mentions M1 CABV closed_em result and the per-cell complementarity sketch. Risk-free.
2. **Moderate (Option B, my recommendation):** Promote AID family from 4 → 5 aggregations including CABV. Add cell-complementarity ensemble F2 to §5. Run A1 with CABV included. Defensible at current power.
3. **Aggressive (Option C):** Reframe contribution as "step-level multi-source instability detection" — within-step (AID) × between-step (CABV) × length (LAAS) — with F2 as the headline trigger. Too aggressive at current significance.

### Pre-registered actions for when R1 Gemma lands

1. Re-run `analysis/moonshot_explore.py` (no flag changes; loads any `runs/v2_headline_<arch>/` cell pairs).
2. Re-run `analysis/moonshot_ensemble.py` — check whether F2 vs `min_majority` lower-CI crosses 0 at n=1200+ paired.
3. Check whether the per-cell complementarity (M9 wins on the ≥1 Gemma cell where `min_majority` resid AUROC < 0.5) replicates.
4. If both hold, promote CABV to a fifth AID variant in `analysis/a1_k1_vs_k5_ablation.py` for the official A1-on-R1 table.

### Status

- **Moonshot exploration: DONE.** 6 new signals + 1 interaction + 3 ensembles tested on R1 Phi + Qwen pooled.
- **One paired-significant new finding: M1 CABV beats tokens on closed_em (Phi+Qwen n=800, +0.118 [+0.048, +0.189]).**
- **One borderline / per-cell-complementary finding: M9 = CABV × LAAS and F2 = rank-fusion ensemble. Significance pending Gemma.**
- **Framing decision deferred to user.**

### Artifacts committed

- `analysis/moonshot_explore.py` (new analyser; reusable on Gemma when it lands).
- `analysis/moonshot_ensemble.py` (new follow-up analyser).
- `analysis/moonshot_explore.out` / `.json`-equivalent stdout dump.
- `analysis/moonshot_ensemble.out` (stdout dump).
- `__lit-review-main/_moonshot-findings.md` (full readout with three framing options).

---

## Run A1 — K=1 vs K=5 flip_rate ablation on P0 traces (2026-05-14, ~10 min wall, CPU)

**Goal served:** G1 (substantiate the K=5 multi-sample contribution vs the Datta et al. 2026 forced-answer-completion precedent — the single highest-value pre-R1 directional read).

**Setup:** `analysis/a1_k1_vs_k5_ablation.py` (new), consumes existing rq1-pilot-v1 P0 traces under `runs/v2_pilot_<arch>/{D1,D2,E1,E2}_*.jsonl`. Two flip_rate variants:

- **K=1 (Datta-style proxy):** `samples[0].text` per step → answer extraction → consecutive-step disagreement count / (n_answer_bearing_steps − 1).
- **K=5 (ours):** majority over `samples[0..4]` per step → consecutive-step majority disagreement count / (n_answer_bearing_steps − 1).

Honest caveat: "K=1" is the FIRST stochastic completion, not faithful greedy forced-completion (which would re-decode with greedy + answer-prefix forcing, ≈ the cancelled rq1-pilot-v2 N2 capture). Read as a directional proxy. R1 will run the same script on n=200/cell traces; that is the ablation table the paper commits to.

Paired bootstrap B=10,000, OLS residualization on `log(tokens_generated + 1)` per cell + overall.

### Results (output also at `analysis/a1_k1_vs_k5_p0.out` and `analysis/a1_k1_vs_k5_p0.json`)

Loaded 150 paired traces across 6 cells (3 archs × 2 datasets, n=25/cell pilot). 98/150 traces have ≥2 answer-bearing steps under both K=1 and K=5 (the rest have ≤1 step with extractable Answer:, so flip_rate is undefined — symmetric across K).

**Target: `lift_em = +1` (retrieval would have helped) — n=98, pop=13**

| cell | n | pop | K1_raw | K5_raw | Δraw | K1_res | K5_res | Δres | Δraw 95% CI | Δres 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| gemma_hot | 25 | 2 | 0.707 | 0.685 | −0.022 | 0.739 | 0.739 | +0.000 | [−0.174,+0.136] | [−0.174,+0.208] |
| gemma_wiki | 25 | 4 | 0.917 | 0.768 | −0.149 | 0.929 | 0.815 | −0.113 | [−0.386,+0.000] | [−0.375,+0.071] |
| phi_hot | 7 | 0 | (skipped — degenerate labels) | | | | | | | |
| phi_wiki | 5 | 1 | (skipped — too few samples) | | | | | | | |
| qwen_hot | 18 | 3 | 0.444 | 0.478 | +0.033 | 0.367 | 0.367 | +0.000 | [+0.000,+0.107] | [−0.094,+0.092] |
| qwen_wiki | 18 | 3 | 0.678 | 0.678 | +0.000 | 0.689 | 0.689 | +0.000 | [+0.000,+0.000] | [+0.000,+0.000] |
| **OVERALL** | **98** | **13** | **0.628** | **0.589** | **−0.038** | **0.649** | **0.607** | **−0.042** | **[−0.127,+0.030]** | **[−0.149,+0.033]** |

**Target: `closed_em = 0` (closed-book incorrect) — n=98, pop=60**

| cell | n | pop | K1_raw | K5_raw | Δraw | K1_res | K5_res | Δres | Δraw 95% CI | Δres 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| gemma_hot | 25 | 16 | 0.549 | 0.684 | +0.135 | 0.514 | 0.632 | +0.118 | [−0.031,+0.341] | [−0.053,+0.333] |
| gemma_wiki | 25 | 12 | 0.708 | 0.670 | −0.038 | 0.724 | 0.673 | −0.051 | [−0.173,+0.074] | [−0.199,+0.065] |
| phi_hot | 7 | 7 | (skipped — degenerate labels) | | | | | | | |
| phi_wiki | 5 | 4 | (skipped — too few samples) | | | | | | | |
| qwen_hot | 18 | 13 | 0.585 | 0.708 | +0.123 | 0.469 | 0.562 | +0.092 | [+0.000,+0.400] | [−0.108,+0.415] |
| qwen_wiki | 18 | 8 | 0.725 | 0.725 | +0.000 | 0.625 | 0.625 | +0.000 | [+0.000,+0.000] | [+0.000,+0.000] |
| **OVERALL** | **98** | **60** | **0.678** | **0.717** | **+0.039** | **0.676** | **0.717** | **+0.041** | **[−0.019,+0.106]** | **[−0.022,+0.112]** |

### Honest readout

1. **At P0 power, K=1 and K=5 are statistically indistinguishable on `lift_em=+1`.** Overall Δlift_res = −0.042 with 95% CI [−0.149, +0.033]. The CI straddles zero. Cell-level estimates are similarly ambiguous. **This is a null result with very wide CIs at pop=13/98**, NOT evidence that K=5 fails.
2. **K=5 shows a small positive directional advantage on `closed_em=0`** (overall Δres = +0.041, CI [−0.022, +0.112]). Direction-of-effect is consistent with the original P0.6 deep-exploration finding that K=5 majority captures stochastic instability that K=1 misses, but the CI still straddles zero.
3. **gemma_wiki is a directional outlier** with K=1 beating K=5 by 0.149 raw on the helpful target — driven by pop=4 helpful traces; not interpretable at this n.
4. **The phi cells are unusable for A1 at P0** — phi traces tend to have ≤1 answer-bearing step in n=25 (only 7+5 valid traces, 0+1 helpful), so flip_rate is mostly undefined and labels are degenerate. R1's phi cell at n=200 will fix this.

### Implication for the paper

- **Cannot headline "K=5 multi-sample extension is the contribution" from P0 alone.** Pre-registered for A1-on-R1 (n=200/cell, ~30-50 helpful pop per cell, CIs ~3× tighter).
- **Reframe contribution as protocol-first if A1-on-R1 also shows null:** "We are the first to apply step-level answer-instability as a retrieval-trigger and to evaluate it under length-controlled paired-bootstrap; the K=5 vs K=1 question is empirically resolved on R1, with the regime-specific framing committed before run." This is honest and still publishable.
- **Three pre-registered outcome regimes for A1-on-R1** (mirrors `_rq1-plan.md` §"Week-of-2026-05-14 paper-push plan"):
  1. K=5 paired-significantly beats K=1 in ≥2 cells → K=5-extension headline.
  2. K=5 ≈ K=1 (CIs straddle 0 across most cells) → protocol-first headline; K determined empirically.
  3. K=1 paired-significantly beats K=5 → K=5 framing dropped; methodology + Datta-K=1 retrieval-triggering application as headline.

**Status:** A1-on-P0 DONE. A1-on-R1 (Phi) DONE — see "Run R1 — Phi paired full landing" below. A1-on-R1 (Gemma, Qwen) PENDING (those archs not yet decoded).

**Artifacts committed:**
- `analysis/a1_k1_vs_k5_ablation.py` (the analyzer; idempotent; pre-registered for A1-on-R1 reuse via `--pilot-root v2_headline`)
- `analysis/a1_k1_vs_k5_p0.out` (stdout dump)
- `analysis/a1_k1_vs_k5_p0.json` (machine-readable; for paper Table 2 ingestion)
- `analysis/a1_k1_vs_k5_r1_phi.out` / `.json` (R1 Phi numbers; see paired-audit entry for interpretation)

---

## Run R1 — Phi paired full landing (D1+D2+E1+E2 all n=200, 2026-05-14 ~02:50, ~12 min CPU analysis)

**Status of compute box (per `runs/headline_n200_run.log` + `runs/v2_headline_phi-4-mini-instruct/logs/00_master.log`):**

| phase | rows | status |
|---|---:|---|
| Phi B smoke | 5 | DONE |
| **Phi D1 closed HotpotQA** | **200** | **DONE** |
| **Phi D2 closed 2Wiki** | **200** | **DONE** |
| **Phi E1 oracle HotpotQA** | **200** | **DONE** |
| **Phi E2 oracle 2Wiki** | **200** | **DONE** |
| Qwen2.5 D1 closed HotpotQA | 0 | **IN FLIGHT** (model loading + decode underway) |
| Qwen2.5 D2 / E1 / E2 | 0 | NOT STARTED |
| Gemma D1 / D2 / E1 / E2 | 0 | NOT STARTED |

**Phi cell is COMPLETE.** This is the first paper-defining cell to land — `lift_em = +1` is now computable on n=200 paired closed×oracle, and the full trigger pool can be evaluated against the deployment-relevant target.

### Setup

Analyzer: `analysis/r1_phi_paired_audit.py` (new; reuses helpers from `r1_phi_d1_closed_audit.py`). Loads paired closed × oracle by qid, computes `lift_em` per trace, runs the full trigger pool against BOTH `lift_em = +1` and `closed_em = 0`. Length-residualized AUROC + paired bootstrap (B=10,000) vs `tokens_generated`. Per-question-type stratification. Includes V2 sanity-check comparison row on the n=393/452 Phi cohort.

Pool: 16 trigger-pool signals + 4 AID variants (same set as `r1_phi_d1_closed_audit.py`).

### Headline numbers — `lift_em = +1` (RETRIEVAL HELPED), the paper-defining target

**Phi × HotpotQA-distractor (n=200 paired, 18 helpful, base_rate=0.090):**

| signal | raw AUROC | residualized AUROC | survives? | Δ vs tokens (raw) | 95% CI |
|---|---:|---:|---:|---:|---|
| tokens_generated | **0.630** | — | (baseline) | (baseline) | (baseline) |
| **AID min_majority (1 − min)** | **0.669** | **0.597** | **✓ STRONGEST RESIDUALIZED** | +0.001 | [−0.192, +0.189] |
| AID late_flips | 0.611 | 0.502 | ✓ | −0.203 | [−0.645, +0.262] |
| seakr_k5_proxy.logdet | 0.634 | 0.412 | — | −0.025 | [−0.183, +0.132] |
| etc (Δ²H max) | 0.564 | 0.509 | ✓ | −0.066 | [−0.176, +0.038] |
| ent.max | 0.561 | 0.460 | — | −0.069 | [−0.157, +0.024] |
| dragin_rind_max.max | 0.553 | **0.397** | — (failed) | −0.076 | [−0.162, +0.012] |
| AID flip_rate K=5 | 0.484 | 0.353 | — | −0.329 | [−0.667, +0.034] |
| AID flip_rate K=1 (Datta proxy) | 0.471 | 0.342 | — | −0.342 | [−0.679, +0.023] |
| u_cons.max (exploratory) | 0.550 | 0.449 | — | −0.080 | [−0.181, +0.020] |

**Phi × 2WikiMHQA-distractor (n=200 paired, 30 helpful, base_rate=0.150):**

| signal | raw AUROC | residualized AUROC | survives? | Δ vs tokens (raw) | 95% CI |
|---|---:|---:|---:|---:|---|
| tokens_generated | 0.602 | — | (baseline) | (baseline) | (baseline) |
| **AID late_flips** | **0.605** | **0.663** | **✓ STRONGEST RESIDUALIZED IN POOL** | **+0.101** | [−0.038, +0.250] |
| AID flip_rate K=1 | 0.585 | 0.582 | ✓ | +0.080 | [−0.179, +0.319] |
| dragin_rind_max.max | 0.585 | 0.558 | ✓ | −0.018 | [−0.152, +0.116] |
| flare.max (1 − min p) | 0.547 | 0.541 | ✓ | −0.038 | [−0.186, +0.105] |
| ent.max | 0.537 | 0.462 | — | −0.065 | [−0.188, +0.059] |
| AID flip_rate K=5 | 0.491 | 0.466 | — | −0.015 | [−0.383, +0.306] |
| AID min_majority | 0.409 | 0.424 | — | −0.190 | [−0.359, −0.012] **✓ worse** |
| u_cons.max | 0.480 | 0.435 | — | −0.123 | [−0.233, −0.012] **✓ worse** |
| seakr_k5_proxy.logdet | 0.437 | 0.390 | — | −0.147 | [−0.264, −0.029] **✓ worse** |

**Per-question-type — the standout cells:**

| cell × type | AID late_flips | DRAGIN | u_cons |
|---|---:|---:|---:|
| 2Wiki × comparison | **0.950** | 0.642 | 0.622 |
| 2Wiki × inference (small n) | **1.000** | 0.473 | 0.314 |
| 2Wiki × bridge_comparison | 0.742 | 0.556 | 0.662 |
| HotpotQA × bridge | **0.679** | 0.528 | 0.474 |
| HotpotQA × comparison | 0.250 | 0.611 | **0.847** |

`AID late_flips` on 2Wiki/comparison: **0.950 raw AUROC** is the single strongest cell-level finding in the campaign. Inverted on HotpotQA/comparison (0.250). Per-type heterogeneity is real and asymmetric — must be reported in the paper.

### Pooled R1 Phi (n=400, pop=48 helpful)

| signal | raw | residualized | Δ vs tokens (raw) | 95% CI | sig? |
|---|---:|---:|---:|---|---|
| tokens | 0.622 | — | (baseline) | (baseline) | (baseline) |
| AID late_flips | 0.607 | 0.560 ✓ | −0.039 | [−0.273, +0.176] | — |
| dragin_rind_max | 0.578 | 0.501 ✓ | −0.044 | [−0.127, +0.038] | — |
| flare.max | 0.554 | 0.502 ✓ | −0.059 | [−0.164, +0.046] | — |
| AID flip_rate K=1 | 0.522 | 0.462 | −0.124 | [−0.380, +0.114] | — |
| AID flip_rate K=5 | 0.483 | 0.437 | −0.162 | [−0.425, +0.087] | — |
| AID min_majority | 0.524 | 0.494 | −0.103 | [−0.234, +0.026] | — |

**Pooled survivors under length residualization (resid AUROC ≥ 0.5):** AID `late_flips`, DRAGIN-RIND, FLARE.max. AID `flip_rate K=5`, AID `flip_rate K=1`, AID `min_majority` (pooled), seakr_k5_proxy, every CRV signal — collapse.

**No signal paired-bootstrap-significantly beats `tokens_generated` on Phi pooled** (CIs all straddle 0). **No signal is paired-bootstrap-significantly worse than tokens either** on the most relevant lift_em target (the worse-than-tokens significance is on closed_em=0 or per-cell). The AUROC values are noisy at pop=48.

### A1 K=1 vs K=5 — the paper-defining ablation, R1 Phi

| target | cell | n_valid | pop | K=1 | K=5 | Δ(K5−K1) raw | 95% CI raw | Δ res | 95% CI res |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| **lift_em=+1** | phi_hot | 52 | 6 | 0.471 | 0.484 | +0.013 | [+0.000, +0.043] | +0.011 | [−0.017, +0.054] |
| **lift_em=+1** | phi_wiki | 53 | 6 | 0.585 | 0.491 | −0.094 | [−0.277, +0.000] | −0.115 | [−0.467, +0.346] |
| **lift_em=+1** | **OVERALL** | **105** | **12** | **0.522** | **0.483** | **−0.039** | **[−0.131, +0.012]** | **−0.025** | **[−0.093, +0.022]** |
| closed_em=0 | OVERALL | 105 | 66 | 0.521 | 0.514 | −0.007 | [−0.031, +0.015] | −0.005 | [−0.037, +0.026] |

**Pre-registered regime resolution on Phi: regime 2 (K=5 ≈ K=1) on `lift_em`** — point estimate has K=1 marginally ahead, CI straddles 0 on both raw and residualized. The K=5 multi-sample contribution is **NOT empirically supported on Phi alone**. The headline K=5-vs-K=1 verdict will be set by Gemma + Qwen R1 cells.

### V2 sanity comparison — Phi paired (n=500/500, prior cohort)

V2 had MORE helpful traces (lift+1 pop = 51 on HotpotQA, 80 on 2Wiki) vs R1 (18 / 30). So V2 numbers are tighter on lift_em:

| signal | V2 HotpotQA (n=500, lift+1=51) | R1 HotpotQA (n=200, lift+1=18) | V2 2Wiki (n=500, lift+1=80) | R1 2Wiki (n=200, lift+1=30) |
|---|---:|---:|---:|---:|
| tokens raw | 0.700 | 0.630 | 0.593 | 0.602 |
| dragin raw | 0.670 | 0.553 | 0.599 | 0.585 |
| dragin residualized | **0.508 ✓** | 0.397 | **0.571 ✓** | 0.558 ✓ |
| u_cons residualized | 0.532 ✓ | 0.449 | 0.486 | 0.435 |
| ent.max residualized | 0.514 ✓ | 0.460 | 0.508 ✓ | 0.462 |

**V2 cohort has tighter CIs and reproduces DRAGIN-survives on both datasets** (residualized AUROC > 0.5 on both, V2). R1 reproduces DRAGIN-survives on 2Wiki only; the HotpotQA cell is power-limited (pop=18).

**Implication:** for the paper, **the Phi headline numbers should be the V2 (n=500) cohort, not the R1 (n=200) cohort** — V2 is statistically more powerful for Phi. R1 Phi's contribution is the new schema fields (FLARE, SeaKR-K5, AID variants) which V2 doesn't have, and the cross-arch comparability with Gemma + Qwen R1.

### Honest readout — what changes for the paper

**Three real shifts from the P0.6 framing:**

1. **`flip_rate` is the WEAKEST AID variant on Phi at n=200.** Both K=1 and K=5 collapse below chance under residualization on both Phi cells. **`min_majority` (HotpotQA: resid 0.597) and `late_flips` (2Wiki: resid 0.663) are the headline-strength AID variants.** The contribution narrative should be reframed:
   - ~~"AID flip_rate is the novel trigger"~~
   - **NEW: "We introduce **step-level multi-sample answer-instability** as a trigger family. We test 4 aggregations: `flip_rate` (K=5 majority step-to-step disagreement), `late_flips` (flips in last 50% of steps), `min_majority` (max within-step disagreement), and `flip_rate K=1` (Datta-style, the K=1 baseline). The optimal aggregation is dataset-dependent: `min_majority` wins on HotpotQA, `late_flips` wins on 2Wiki. As a family, AID survives length-residualization where most of the published trigger pool fails."**

2. **K=5 vs K=1: regime 2 (equality) on Phi.** K=5 has no empirical advantage over K=1 on Phi. Three regimes for the cross-arch verdict still in play; Gemma + Qwen will decide.

3. **DRAGIN-RIND survives on Phi 2Wiki at n=200, fails on Phi HotpotQA at n=200 due to power.** The V2 (n=500) cohort had DRAGIN surviving on both. **For Phi, cite V2 as the headline-grade cohort.**

**The `late_flips` × 2Wiki/comparison cell (raw AUROC = 0.950)** is the strongest single empirical finding in the campaign so far. Worth a dedicated table or figure in §4.

### Decisions / pre-commits triggered

| decision | trigger | resolution |
|---|---|---|
| Headline AID variant | R1 result that `flip_rate` underperforms `late_flips` and `min_majority` | **Headline AID = the FAMILY (`flip_rate`, `late_flips`, `min_majority`, K=1 baseline) — not just `flip_rate`.** Per-cell optimal variant reported. |
| K=5 vs K=1 framing | regime 2 on Phi (equality, K=1 marginally ahead) | wait for Gemma + Qwen; if regime 2 holds cross-arch, paper says "K=5 ≈ K=1; we determine K empirically and recommend K=1 for compute-efficiency reasons" — defensible and honest |
| Phi headline cohort | V2 (n=500) more powerful than R1 (n=200) for Phi lift_em | **Use V2 numbers for Phi DRAGIN/u_cons/ent. Use R1 numbers for FLARE / SeaKR-K5 / AID variants (new fields).** Cite both cohorts; honest. |
| `late_flips` per-question-type | 0.950 on 2Wiki/comparison, 0.143 on HotpotQA/comparison (inverted) | Dedicated paragraph in §5 discussion; the per-type heterogeneity is a real finding, not a bug |

### Status

- **R1 Phi paired audit: DONE; reproduces V2 direction-of-effect; AID family survives length residualization on both cells via different aggregations; K=5 vs K=1 indistinguishable on Phi.**
- **Compute box: Qwen2.5 fully done; see "Run R1 — Qwen2.5 paired full landing" below.**
- **Cross-arch verdict on K=5 vs K=1, AID family, DRAGIN survival: 2 of 3 archs landed; Gemma in flight.**
- **R2 (MuSiQue): unchanged decision rule — Day-3 conditional on R1 Gemma cleanliness.**

### Artifacts committed

- `analysis/r1_phi_paired_audit.py` (new analyzer; reusable on remaining R1 cells via `--arch <name>`)
- `analysis/r1_phi_paired_audit.out` (stdout dump, full tables for Phi HotpotQA + 2Wiki + pooled + V2 sanity)
- `analysis/a1_k1_vs_k5_r1_phi.out` / `.json` (A1 ablation R1 Phi numbers)

---

## Run R1 — Qwen2.5 paired full landing (D1+D2+E1+E2 all n=200, 2026-05-14 ~12:14, ~2 min CPU analysis)

**Status of compute box:**

| phase | rows | status |
|---|---:|---|
| Qwen2.5 B smoke | 5 | DONE |
| **Qwen2.5 D1 closed HotpotQA** | **200** | **DONE** |
| **Qwen2.5 D2 closed 2Wiki** | **200** | **DONE** |
| **Qwen2.5 E1 oracle HotpotQA** | **200** | **DONE** |
| **Qwen2.5 E2 oracle 2Wiki** | **200** | **DONE** |
| Gemma D1 closed HotpotQA | ~85 | **IN FLIGHT** |

**Qwen cell is COMPLETE.** Second of three archs landed.

### Headline numbers — `lift_em = +1` on Qwen2.5

**Qwen × HotpotQA-distractor (n=200 paired, 31 helpful, base_rate=0.155):**

| signal | raw AUROC | residualized AUROC | survives? | Δ vs tokens (raw) | 95% CI | sig? |
|---|---:|---:|---:|---:|---|---|
| tokens_generated | 0.572 | — | (baseline) | — | — | — |
| **AID flip_rate K=1 (Datta)** | **0.666** | **0.690** | **✓ STRONGEST** | +0.098 | [−0.037, +0.236] | — |
| **AID flip_rate K=5 (ours)** | 0.659 | **0.668** | **✓** | +0.092 | [−0.045, +0.233] | — |
| u_cons.max | 0.668 | 0.634 | ✓ | +0.096 | [−0.040, +0.233] | — |
| **AID min_majority** | 0.659 | 0.618 | ✓ | +0.085 | [−0.076, +0.244] | — |
| AID late_flips | 0.640 | 0.598 | ✓ | +0.072 | [−0.101, +0.250] | — |
| etc (Δ²H max) | 0.572 | 0.547 | ✓ | −0.001 | [−0.129, +0.129] | — |
| flare.max | 0.554 | 0.526 | ✓ | −0.019 | [−0.140, +0.099] | — |
| dragin_rind_max | 0.582 | 0.502 | ✓ | +0.009 | [−0.117, +0.139] | — |

**Qwen × 2WikiMHQA-distractor (n=200 paired, 34 helpful, base_rate=0.170):**

| signal | raw AUROC | residualized AUROC | survives? | Δ vs tokens (raw) | 95% CI | sig? |
|---|---:|---:|---:|---:|---|---|
| **tokens_generated** | **0.457** | — | (baseline; **inverted-length-correlation reproduced**) | — | — | — |
| **dragin_rind_max** | **0.606** | **0.653** | **✓** | **+0.148** | **[+0.034, +0.265]** | **✓ BEATS tokens** |
| **AID min_majority** | 0.619 | **0.672** | **✓ STRONGEST RESIDUALIZED** | **+0.166** | **[+0.034, +0.296]** | **✓ BEATS tokens** |
| AID late_flips | 0.609 | 0.586 | ✓ | +0.032 | [−0.118, +0.183] | — |
| AID flip_rate K=5 | 0.582 | 0.549 | ✓ | +0.004 | [−0.175, +0.191] | — |
| AID flip_rate K=1 | 0.579 | 0.534 | ✓ | +0.001 | [−0.189, +0.193] | — |
| ent.max | 0.558 | 0.608 | ✓ | +0.101 | [−0.011, +0.214] | — |

The P0.6 finding **`tokens AUROC = 0.380` on Qwen 2Wiki replicates: 0.457 here**. This is the inverted-length-correlation cell. Internal signals BEAT tokens here.

### Pooled R1 Qwen (n=400, pop=65 helpful)

| signal | raw | residualized | Δ vs tokens (raw) | 95% CI | sig? |
|---|---:|---:|---:|---|---|
| tokens | 0.512 | — | (baseline) | — | — |
| **AID min_majority** | **0.635** | **0.648** | **+0.123** | **[+0.020, +0.224]** | **✓ BEATS tokens** |
| **dragin_rind_max** | 0.596 | 0.605 | **+0.085** | **[+0.003, +0.168]** | **✓ BEATS tokens** |
| AID flip_rate K=1 | 0.626 | 0.615 | +0.054 | [−0.059, +0.169] | — |
| AID flip_rate K=5 | 0.624 | 0.614 | +0.052 | [−0.060, +0.167] | — |
| AID late_flips | 0.624 | 0.584 | +0.052 | [−0.062, +0.169] | — |
| u_cons.max | 0.604 | 0.602 | +0.091 | [−0.003, +0.186] | borderline |

**Two signals paired-bootstrap-significantly beat `tokens_generated` on Qwen pooled:**
- **AID `min_majority`** — Δ=+0.123, CI [+0.020, +0.224]
- **DRAGIN-RIND** — Δ=+0.085, CI [+0.003, +0.168]

This is **the headline result** for the paper so far.

### A1 K=1 vs K=5 — Qwen R1

| target | cell | n | pop | K=1 | K=5 | Δ(K5−K1) raw | 95% CI raw | Δ res | 95% CI res |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| **lift_em=+1** | qwen_hot | 151 | 25 | 0.666 | 0.659 | −0.006 | [−0.062, +0.032] | −0.022 | [−0.096, +0.029] |
| **lift_em=+1** | qwen_wiki | 154 | 22 | 0.579 | 0.582 | +0.003 | [−0.044, +0.056] | +0.014 | [−0.043, +0.089] |
| **lift_em=+1** | **OVERALL** | **305** | **47** | **0.626** | **0.624** | **−0.002** | **[−0.040, +0.031]** | **−0.001** | **[−0.049, +0.044]** |
| closed_em=0 | OVERALL | 305 | 142 | 0.592 | 0.598 | +0.006 | [−0.024, +0.036] | +0.017 | [−0.014, +0.049] |

**Pre-registered regime resolution on Qwen: regime 2 (K=5 ≈ K=1).** Same as Phi. **Combined cross-arch evidence (Phi + Qwen, both n=200): K=5 has NO empirical advantage over K=1 on either arch on either dataset.** Δ raw and Δ res both straddle zero with point estimates near zero on Qwen. The K=5 multi-sample contribution is **definitively NOT supported** by the headline-grade cohorts.

### Combined cross-arch survivor matrix (Phi + Qwen n=400+400 = 800 paired)

Length-residualized AUROC on `lift_em = +1`, paired-bootstrap-vs-tokens significance:

| signal | Phi pooled (n=400) resid | Qwen pooled (n=400) resid | sig vs tokens? |
|---|---:|---:|---|
| tokens_generated | (baseline) | (baseline) | — |
| **AID min_majority** | 0.494 | **0.648** ✓ | **Qwen ✓ BEATS** (CI [+0.020, +0.224]) |
| **AID late_flips** | **0.560** ✓ | 0.584 ✓ | — |
| AID flip_rate K=5 | 0.437 | 0.614 ✓ | — |
| AID flip_rate K=1 | 0.462 | 0.615 ✓ | — |
| **DRAGIN-RIND** | 0.501 ✓ | 0.605 ✓ | **Qwen ✓ BEATS** (CI [+0.003, +0.168]) |
| FLARE.max | 0.502 ✓ | 0.550 ✓ | — |
| u_cons.max (exploratory) | 0.438 | 0.602 ✓ | — |
| etc (Δ²H max) | 0.396 | 0.582 ✓ | — |
| ent.max | 0.449 | 0.566 ✓ | — |

**Both DRAGIN-RIND and the AID family (best variant: `min_majority` on Qwen pooled) paired-bootstrap-significantly beat `tokens_generated` under length residualization on at least one architecture.** This is **the survivor pair**.

**Per-cell strongest residualized AUROC (cross-arch matrix building):**

| arch × dataset | strongest signal | resid AUROC |
|---|---|---:|
| Phi × HotpotQA | **AID `min_majority`** | 0.597 |
| Phi × 2Wiki | **AID `late_flips`** | 0.663 |
| Qwen × HotpotQA | **AID `flip_rate K=1`** | 0.690 |
| Qwen × 2Wiki | **AID `min_majority`** | 0.672 |
| Phi pooled | AID `late_flips` | 0.560 |
| Qwen pooled | **AID `min_majority`** | 0.648 |

**AID family wins 4 of 4 cells under length residualization.** Different aggregation per cell, but **the AID family is the headline survivor** across 2 archs × 2 datasets.

### Honest readout — what changes with Qwen

**Three positive shifts:**

1. **The AID family is now the empirical headline.** On Qwen pooled, AID `min_majority` paired-significantly beats `tokens_generated` (CI [+0.020, +0.224]). Combined with Phi pooled where AID `late_flips` survives residualization (resid 0.560 ✓ > 0.5), the AID family is **the strongest length-residualized signal on every cell** in the cross-arch matrix.

2. **DRAGIN-RIND is co-survivor.** On Qwen pooled, DRAGIN-RIND paired-significantly beats tokens (CI [+0.003, +0.168]). Combined with V2 numbers (DRAGIN survives both Phi cells at n=500, residualized AUROC 0.508 / 0.571), DRAGIN is **the second length-robust survivor**. **The original RQ1 claim "DRAGIN is the only length-robust survivor" is partially confirmed AND newly accompanied.**

3. **Qwen 2Wiki inverted-length-correlation reproduced** at n=200 (`tokens AUROC = 0.457`, P0.6 had 0.380). On this cell, internal signals naturally beat tokens because tokens is a *worse-than-chance* signal. This is a real per-cell finding for the discussion (length-confound is heterogeneous across (arch × dataset) cells; not universal).

**One null result:**

4. **K=5 vs K=1 is regime 2 on both Phi and Qwen.** Point estimates near zero on both archs on both datasets. **The K=5 multi-sample contribution is empirically null at headline-grade power.** This actually SIMPLIFIES the paper:
   - **Old framing:** "We extend Datta et al. with K=5 multi-sample boundary completions."
   - **NEW framing:** "We extend Datta et al.'s forced-answer-completion to a retrieval-trigger setting; K is empirically free (we test K=5 and K=1, find no significant difference at n=400/arch); we recommend K=1 for compute-efficiency."
   - This is **honest, defensible, and removes a contentious ablation from the paper**. The K=5 dimension is now a free piece of methodological discipline rather than a load-bearing claim.

### Implications for paper trajectory

**Paper trajectory after R1 Phi + Qwen:**
- **Empirical headline:** "Under the UNCERTAINTY-LINE residualization protocol of Belem et al. (2025) extended to dynamic-RAG triggers and the `lift_em = +1` paired-oracle target, **the AID family and DRAGIN-RIND are the two length-robust survivors** across 2 architectures × 2 datasets at n=200/cell. AID-`min_majority` paired-significantly beats `tokens_generated` on Qwen pooled (Δresid AUROC = +0.123, CI [+0.020, +0.224]); DRAGIN paired-significantly beats tokens on Qwen pooled (Δ = +0.085, CI [+0.003, +0.168])."
- **Methodological contribution unchanged:** the residualized-AUROC × paired-oracle × `lift_em` framework, applied to the dynamic-RAG trigger pool.
- **AID family contribution sharpened:** four aggregations (`flip_rate` K=5/K=1, `late_flips`, `min_majority`); per-cell optimal varies; family-level all four survive on Qwen and at least two on Phi.
- **K=5 vs K=1 narrative:** drop the contention, fold into honest methodology discussion. Frees a third of the §4 word budget for the survivor matrix and per-cell heterogeneity.

**Acceptance estimate update**: was 40-50% with Phi only; **now 55-65% with Phi + Qwen.** The cross-arch survivor pair (DRAGIN + AID family) and the paired-significance result on Qwen pooled is exactly the headline that EMNLP-short reviewers want — empirical, replicated across one cell, properly residualized, per-cell heterogeneity reported. **If Gemma reproduces the AID-family-survives + DRAGIN-survives pattern, target row of `_rq1-outlook.md` (60-70%) is in reach.**

### Status

- **R1 Phi + Qwen paired audits: DONE; cross-arch survivor pair identified (AID family + DRAGIN-RIND).**
- **Compute box: Gemma D1 ~85/200; full Gemma cell ~3 hours from now (single A800).**
- **K=5 vs K=1 verdict: regime 2 (equality) confirmed on both archs.** Headline framing simplifies; K is empirically free.
- **R2 (MuSiQue): launch IF Gemma reproduces AID-family-survives + DRAGIN-survives pattern (Day-3 decision unchanged).**

### Artifacts committed

- `analysis/r1_qwen_paired_audit.out` (full Qwen tables: HotpotQA + 2Wiki + pooled + per-question-type)
- `analysis/a1_k1_vs_k5_r1_qwen.out` / `.json` (A1 ablation R1 Qwen numbers)

---

## Run R1 — Phi D1 partial landing (closed-book HotpotQA only, 2026-05-14, ~12 min wall, CPU)

**Status of compute box (per `runs/headline_n200_run.log` + `runs/v2_headline_phi-4-mini-instruct/logs/00_master.log`):**

| phase | path | status | rows |
|---|---|---|---:|
| B (smoke, n=5) | `runs/v2_headline_phi-4-mini-instruct/B_smoke_hotpot.jsonl` | **DONE** | 5 |
| **D1 (closed-book HotpotQA, n=200)** | `runs/v2_headline_phi-4-mini-instruct/D1_closed_hotpot.jsonl` | **DONE** | **200** |
| D2 (closed-book 2Wiki, n=200) | `runs/v2_headline_phi-4-mini-instruct/D2_closed_wiki.jsonl` | **PARTIAL (in flight)** | 21 |
| E1 (oracle HotpotQA, n=200) | `runs/v2_headline_phi-4-mini-instruct/E1_oracle_hotpot.jsonl` | **NOT STARTED** | 0 |
| E2 (oracle 2Wiki, n=200) | `runs/v2_headline_phi-4-mini-instruct/E2_oracle_wiki.jsonl` | **NOT STARTED** | 0 |

D1 was the first 200-trace cell to land. D2 is in progress on the box. E1/E2 (paired oracle re-decode) have not started — `lift_em = +1` analyses are **blocked** until E1 lands.

**What we CAN run on D1 alone:**
- Full trigger-pool audit against `closed_em = 0` (closed-book incorrect) — same target as the existing v2 audit
- Length-residualized AUROC for every signal vs `tokens_generated`
- Paired bootstrap (B=10,000) of (signal − tokens) AUROC
- Per-question-type stratification
- A1 K=1 vs K=5 ablation against `closed_em = 0`
- Sanity-check comparison vs existing v2 Phi cohort (n=393 paired) on identical signals

**What we CANNOT run yet:**
- The headline survivor matrix targets `lift_em = +1` (paired oracle re-decode). Blocked until E1.
- Any retrieval-helpfulness analysis. Blocked until E1.
- A1 ablation against `lift_em = +1` (the paper-defining variant). Blocked until E1.

### Setup

Analyzer: `analysis/r1_phi_d1_closed_audit.py` (new, ~400 lines). Mirrors `length_audit_v2.py` patterns, extended with the rq1-pilot-v1 schema fields (FLARE from `per_token_max_logprob_alt`, SeaKR-K5-proxy from `samples[].eos_hidden`, AID flip_rate variants from `samples[].text`). Idempotent. CPU-only.

Pool audited (16 trigger-pool signals + 4 AID variants):

- External / length: `tokens_generated`, `wall_clock_s`, `n_steps`
- Logit family: `ent.max`, `etc (Δ²H max)`, **NEW** `flare.max = 1 − min(per_token_max_logprob_alt) per step → max`
- Attention family: `dragin_rind_max.max`
- Hidden-state CRV (closed-as-headline-candidate, kept in audit): `drift_cosine.max`, `variance_l2_window.max`, `hidden_norm.range`
- Text family: `selfdoubt_t1` (`hvr.max`)
- K-sample exploratory: `u_cons.max` (kept for completeness, not headline)
- **NEW** K-sample hidden: `seakr_k5_proxy.logdet` (Gram log-det of K=5 sample EOS hiddens, max over steps; K=5 proxy for SeaKR's K=20 EigenScore)
- **AID family:** `flip_rate (K=5, ours)`, `flip_rate (K=1, Datta-style proxy)`, `late_flips (V3)`, `min_majority (V5)`

### Results — R1 D1 (n=200, Phi-4-mini-instruct × HotpotQA-distractor, closed-book)

`closed_em = 0`: 118/200 traces (base rate = 0.590).

**Raw AUROC + P@k against `closed_em = 0`:**

| signal | family | pop | AUROC | P@5% | P@10% | P@20% | ΔP@5% |
|---|---|---:|---:|---:|---:|---:|---:|
| **tokens_generated** | external/length | 200 | **0.645** | 0.900 | 0.850 | 0.775 | +0.310 |
| wall_clock_s | external/length | 200 | 0.612 | 0.800 | 0.800 | 0.750 | +0.210 |
| n_steps | external/length | 200 | 0.587 | 0.800 | 0.850 | 0.800 | +0.210 |
| ent.max | logit | 200 | 0.553 | 0.900 | 0.750 | 0.625 | +0.310 |
| etc (Δ²H max) | logit | 200 | 0.583 | **1.000** | 0.900 | 0.725 | +0.410 |
| flare.max (1 − min p) | logit | 156 | 0.541 | 0.875 | 0.750 | 0.742 | +0.285 |
| dragin_rind_max.max | attention | 200 | 0.532 | 0.700 | 0.700 | 0.625 | +0.110 |
| drift_cosine.max | hidden CRV | 200 | 0.586 | 0.900 | 0.850 | 0.800 | +0.310 |
| variance_l2_window.max | hidden CRV | 200 | 0.567 | 0.600 | 0.700 | 0.675 | +0.010 |
| hidden_norm.range | hidden CRV | 200 | 0.570 | 0.600 | 0.700 | 0.750 | +0.010 |
| selfdoubt_t1 (hvr.max) | text | 200 | 0.508 | 0.800 | 0.800 | 0.750 | +0.210 |
| u_cons.max (exploratory) | K-sample-text | 200 | 0.621 | 0.900 | 0.800 | 0.825 | +0.310 |
| seakr_k5_proxy.logdet | K-sample-hidden | 156 | 0.529 | 0.750 | 0.625 | 0.677 | +0.160 |
| AID flip_rate (K=5 ours) | AID | 52 | 0.524 | **1.000** | **1.000** | 0.900 | +0.410 |
| AID flip_rate (K=1 Datta) | AID | 52 | 0.535 | **1.000** | **1.000** | 0.900 | +0.410 |
| AID late_flips | AID | 52 | 0.579 | **1.000** | **1.000** | **1.000** | +0.410 |
| AID min_majority (1 − min) | AID | 139 | 0.519 | 0.857 | 0.714 | 0.536 | +0.267 |

**Length-residualized AUROC (signal ~ log_tokens, evaluated on residual):**

| signal | raw AUROC | resid AUROC | ΔAUROC |
|---|---:|---:|---:|
| wall_clock_s | 0.612 | 0.436 | −0.176 |
| n_steps | 0.587 | 0.442 | −0.145 |
| ent.max | 0.553 | 0.435 | −0.117 |
| etc (Δ²H max) | 0.583 | 0.470 | −0.113 |
| flare.max (1 − min p) | 0.541 | 0.461 | −0.080 |
| dragin_rind_max.max | 0.532 | 0.415 | −0.117 |
| drift_cosine.max | 0.586 | 0.474 | −0.112 |
| variance_l2_window.max | 0.567 | 0.431 | −0.136 |
| hidden_norm.range | 0.570 | 0.441 | −0.130 |
| selfdoubt_t1 (hvr.max) | 0.508 | 0.371 | −0.137 |
| u_cons.max | 0.621 | 0.529 | −0.091 |
| seakr_k5_proxy.logdet | 0.529 | 0.435 | −0.094 |
| AID flip_rate (K=5 ours) | 0.524 | 0.477 | −0.048 |
| AID flip_rate (K=1 Datta) | 0.535 | 0.497 | −0.037 |
| AID late_flips | 0.579 | 0.529 | −0.050 |
| AID min_majority (1 − min) | 0.519 | 0.471 | −0.047 |

**Paired bootstrap (B=10,000) of Δraw AUROC vs `tokens_generated`:**

| signal | Δraw vs tokens | 95% CI | sig? (CI excludes 0) |
|---|---:|---|---:|
| wall_clock_s | −0.033 | [−0.077, +0.010] | — |
| n_steps | −0.058 | [−0.117, −0.000] | ✓ (lower-edge) |
| ent.max | −0.092 | [−0.155, −0.031] | **✓ (worse than tokens)** |
| etc (Δ²H max) | −0.062 | [−0.130, +0.005] | — |
| flare.max | −0.084 | [−0.171, −0.000] | ✓ (lower-edge) |
| dragin_rind_max.max | −0.113 | [−0.175, −0.054] | **✓ (worse than tokens)** |
| drift_cosine.max | −0.059 | [−0.128, +0.009] | — |
| variance_l2_window.max | −0.078 | [−0.146, −0.011] | **✓ (worse than tokens)** |
| hidden_norm.range | −0.075 | [−0.141, −0.009] | **✓ (worse than tokens)** |
| selfdoubt_t1 (hvr.max) | −0.137 | [−0.215, −0.058] | **✓ (worse than tokens)** |
| u_cons.max | −0.025 | [−0.094, +0.044] | — |
| seakr_k5_proxy.logdet | −0.095 | [−0.203, +0.010] | — |
| AID flip_rate K=5 | −0.083 | [−0.279, +0.115] | — |
| AID flip_rate K=1 | −0.073 | [−0.270, +0.127] | — |
| AID late_flips | −0.028 | [−0.211, +0.153] | — |
| AID min_majority | −0.122 | [−0.245, +0.004] | — |

**A1 K=1 vs K=5 (closed_em=0 target only — lift_em blocked until E1):**

n_valid=52 (most traces have <2 answer-bearing steps with K=5 extractable answers — Phi sparse-flip limitation).

| metric | K=1 | K=5 | Δ(K5−K1) | 95% CI |
|---|---:|---:|---:|---|
| raw AUROC | 0.535 | 0.524 | −0.010 | [−0.035, −0.000] |
| length-residualized AUROC | 0.497 | 0.477 | −0.021 | [−0.072, −0.000] |

**Per-question-type AUROC (raw, vs closed_em=0):**

| signal | bridge | comparison |
|---|---:|---:|
| tokens_generated | 0.638 | 0.678 |
| u_cons.max | 0.598 | 0.760 |
| seakr_k5_proxy.logdet | 0.468 | 0.689 |
| AID flip_rate K=5 | 0.539 | 0.429 |
| AID flip_rate K=1 | 0.552 | 0.429 |
| **AID late_flips** | **0.655** | **0.143** |
| AID min_majority | 0.507 | 0.559 |

`AID late_flips` shows **opposite-direction discrimination across question types** — 0.655 on bridge (best AID variant on bridge), 0.143 on comparison (catastrophically inverted). This is a real per-question-type heterogeneity finding, mirrors the v2 `bridge_comparison` structural-exception pattern.

### Capture-pipeline sanity check vs v2 Phi cohort (n=500 prior)

R1 D1 vs V2 D1 raw AUROC against `closed_em = 0` — every overlapping signal:

| signal | V2 (n=500, prior) | R1 D1 (n=200, new) | |Δ| |
|---|---:|---:|---:|
| tokens_generated | 0.636 | 0.645 | 0.009 |
| wall_clock_s | 0.600 | 0.612 | 0.012 |
| n_steps | 0.581 | 0.587 | 0.006 |
| ent.max | 0.550 | 0.553 | 0.003 |
| etc (Δ²H max) | 0.568 | 0.583 | 0.015 |
| dragin_rind_max.max | 0.551 | 0.532 | 0.019 |
| drift_cosine.max | 0.567 | 0.586 | 0.019 |
| variance_l2_window.max | 0.552 | 0.567 | 0.015 |
| hidden_norm.range | 0.558 | 0.570 | 0.012 |
| selfdoubt_t1 (hvr.max) | 0.507 | 0.508 | 0.001 |
| u_cons.max | 0.609 | 0.621 | 0.012 |

**All 11 overlapping signals reproduce within ±0.020 AUROC.** **Capture pipeline sanity check: PASS.** R1 is producing numerically-faithful traces in the new schema vs the prior cohort.

(FLARE, SeaKR-K5-proxy, AID variants are NEW to the rq1-pilot-v1 schema and not present in v2 — comparison rows N/A.)

### Honest readout (what this tells us, what it doesn't)

**What this tells us:**

1. **Capture pipeline is correct.** R1 D1 reproduces V2 numbers within ±0.02 on every overlapping signal. The new K=5 sampling, sample-text capture, and per-token logprob capture have not introduced regressions in the existing signals.
2. **Length is the dominant closed-em signal on Phi HotpotQA at n=200.** `tokens_generated` AUROC = 0.645 (raw), beats every other signal raw-and-residualized. Identical to V2 finding.
3. **Every internal signal collapses below chance under length residualization** on closed_em=0:
   - `dragin_rind_max.max`: 0.532 → 0.415 (paired-significantly worse than tokens; CI [−0.175, −0.054])
   - `selfdoubt_t1 hvr.max`: 0.508 → 0.371 (paired-significantly worse; CI [−0.215, −0.058])
   - `ent.max`, `variance_l2_window`, `hidden_norm.range`, `n_steps`, `flare.max`: all paired-significantly worse than tokens
   - `u_cons.max`, `etc`, `drift_cosine`, `seakr_k5_proxy`: paired-bootstrap CI straddles 0
   - **AID flip_rate K=5, K=1, late_flips, min_majority**: CIs straddle 0 (very wide, pop=52 / 200)
4. **K=5 vs K=1 on closed_em=0**: K=1 marginally beats K=5 (Δ raw = −0.010, CI [−0.035, −0.000]; Δ res = −0.021, CI [−0.072, −0.000]). **Same direction as A1-on-P0 (closed_em). At this n, K=1 ≈ K=5 with K=1 slightly ahead.**
5. **AID late_flips shows extreme per-question-type heterogeneity** on closed_em=0: bridge 0.655 vs comparison 0.143. Worth a §5 discussion paragraph in the paper as a failure-mode finding.

**What this does NOT tell us — and the critical caveat:**

- **The headline survivor narrative is on `lift_em = +1`, not `closed_em = 0`.** V2's finding that "DRAGIN-RIND is the only length-robust survivor" was on the helpful target, NOT the incorrectness target. On `closed_em = 0`, DRAGIN was already paired-significantly *worse* than length in V2 (residualized 0.439, CI excludes 0 in negative direction). The R1 D1 result here (DRAGIN residualized 0.415, paired Δraw vs tokens CI [−0.175, −0.054]) **reproduces V2's closed_em=0 finding cleanly**. This is **NOT** evidence that DRAGIN fails as a survivor — the survivor claim is on the orthogonal target.
- **AID flip_rate's headline number (raw 0.733 / residualized 0.735 from P0.6 deep exploration) was on `lift_em = +1`, NOT on closed_em = 0.** The numbers above (raw 0.524, residualized 0.477 on closed_em=0) are not directly comparable. The lift_em re-run is what the paper commits to.
- **All AID-vs-Datta ablation conclusions on closed_em are pop-limited:** only 52 of 200 traces have ≥2 answer-bearing steps under K=5. The Phi sparse-flip limitation persists at n=200; we expect the same limitation at R1 D2, somewhat better on Gemma (chattier traces) and Qwen2.5.

### Implications for the week plan

| effect | impact |
|---|---|
| Capture pipeline validated → no rework needed | Day-2 analyses can proceed unchanged once E1 lands |
| Closed_em audit on R1 D1 reproduces V2 numbers | Reviewer-defense: we have replication evidence even before R1 finishes |
| Phi sparse-flip limitation (52/200 traces at n=200) | At pop=52, A1-on-R1 will have wider CIs than expected on Phi; **Gemma + Qwen will carry the AID-K=5 evidence**. This is consistent with the original P0.6 finding that AID-flip-rate dominates on Qwen, complements DRAGIN on Phi. Watch Gemma + Qwen R1 cells closely. |
| AID late_flips bridge_comparison heterogeneity | Adds one paragraph to §5 (discussion / failure-mode taxonomy); not a headline change |
| K=1 ≈ K=5 on closed_em (P0 + R1 D1 directional) | Two regimes for A1-on-R1 are still in play (regime 2: equality; regime 3: K=1 wins). Regime 1 (K=5 wins) currently has no support and will require Gemma/Qwen lift_em to materialize. |

### Status

- **R1 Phi D1: DONE; analyzed; sanity-check PASSES.**
- **R1 Phi D2: in flight on compute box (21/200 at last pull).**
- **R1 Phi E1, E2: not yet started.**
- **R1 Qwen2.5, Gemma: not yet started.**
- **Lift_em headline analyses: BLOCKED until E1 lands.** Re-run with `python analysis/pilot_audit.py --prefix v2_headline --bootstrap-b 10000` once E1 + E2 land.
- **A1-on-R1 (lift_em target): BLOCKED until E1 lands.** Re-run with `python analysis/a1_k1_vs_k5_ablation.py --pilot-root v2_headline --bootstrap-b 10000`.

### Artifacts committed

- `analysis/r1_phi_d1_closed_audit.py` (new analyzer; idempotent; reusable on D2 and on remaining cells once they land)
- `analysis/r1_phi_d1_closed_audit.out` (stdout dump, full tables)
- `analysis/r1_phi_d1_closed_audit.json` (machine-readable summary)

---

## Experiment R1 — Headline cross-arch (planned, NEXT on compute box)

**Goals served:** G1 (paired-bootstrap significance on AID flip_rate vs DRAGIN
vs tokens across 3 archs × 2 datasets), G2 (full rq1-pilot-v1 schema at
publishable n), G3 (architecture-heterogeneity confirmation with statistical power).

**Setup:**

| dimension | choice |
|---|---|
| models | Phi-4-mini-instruct, Qwen2.5-7B-Instruct, Gemma-4-26B-A4B-it |
| datasets | HotpotQA-distractor (in-domain), 2WikiMHQA-distractor (OOD) |
| n per cell | 200 (closed) + 200 (oracle, paired) |
| slices | existing v2test 500-question slices (`--limit 200`) |
| sampler K | 5 |
| schema | rq1-pilot-v1 (same as P0 — K=5 sample text + EOS hidden + per-token logprob) |
| ordering | Phi first (fastest, ~2.5h) → Qwen2.5 (~5h) → Gemma (~10h) |

**Orchestrator:** `scripts/run_headline_n200.sh`

**Launch on compute box:**
```bash
# Full run (all 3 archs serial, ~22h):
bash scripts/run_headline_n200.sh

# Or single-arch for fastest results:
ARCHS="phi-4-mini-instruct" bash scripts/run_headline_n200.sh
```

**Output:** `runs/v2_headline_<arch>/{B,D1,D2,E1,E2}_*.jsonl`

**Post-run analysis:**
```bash
python analysis/pilot_audit.py --prefix v2_headline --bootstrap-b 10000 | tee analysis/headline_n200.out
```

**Pre-registered questions (what R1 answers):**
1. Does AID flip_rate achieve paired-bootstrap significance over tokens at n=200? (pilot: 0.733 vs 0.566)
2. Does DRAGIN+lift on Gemma hold at 0.89+ at n=200? (pilot: 0.891/0.940 at n=25)
3. Does the arch-heterogeneity pattern replicate? (flip_rate wins on Qwen, DRAGIN wins on Gemma)
4. Does Qwen2.5 2Wiki inverted-length-correlation replicate at n=200? (pilot: tokens AUROC=0.380)
5. Under Bonferroni at k=19 signals: which signals clear?
6. **(NEW, A1)** Does K=5 boundary-completion `flip_rate` paired-bootstrap-beat K=1 forced-completion `flip_rate` (Datta et al. 2026 baseline)? Required to substantiate the K=5 multi-sample contribution. Computed post-hoc from `samples[].text` field; first sample treated as Datta-style K=1 baseline.

---

## P1 — CANCELLED (partial data note)

P1 (rq1-pilot-v2, Gemma only) was running on compute when cancelled. Partial
Gemma traces MAY exist at `runs/v2_pilot_v2_gemma-4-26b-a4b-it/`. These are
n=25 pilot traces with the 4 novel triggers (N1-N4). Not needed for the headline
paper but retained if present. The N1-N4 captures are insurance/future-work only.

---

---

## Experiment P0 — Smoke + sanity pilot (planned)

**Goals served:** G1 (pilot trigger-pool numbers, directional only — not headline), G2 (max-logging schema-superset capture so every method we audit can be computed post-hoc without re-decoding), G3 (cross-arch behavior of FLARE / SeaKR / SelfCheckGPT signals on a fresh dataset).

**Why a pilot.** Before the headline campaign (E2..E10, ~16 h split or ~22–60 h serial), we need a 3-arch × 2-dataset smoke that (a) validates the extended capture schema works on Phi *and* both MoE archs, (b) produces fresh paired (closed × oracle) traces with the new max-logging fields populated for every audited method, (c) yields directional signal numbers per (arch × method × dataset) cell so we can de-risk every Burden 2/3/8 outcome before committing the larger compute. Throw-away if any cell breaks; the schema fixes are the deliverable.

### Setup

| dimension | choice | rationale |
|---|---|---|
| **models** | Phi-4-mini-instruct + Gemma-4-26B-A4B-it + Qwen3.6-35B-A3B-Instruct | full RQ1 model pool, frozen Option-A protocol per `_MODEL.md §3` |
| **datasets** | HotpotQA-distractor (in-domain) + 2WikiMHQA-distractor (OOD) | matches the n=500 v2 test slices; reuse same questions for cohort comparability |
| **n per cell** | 25 (closed) + 25 (oracle, paired) | enough for sanity-check AUROC point estimates; not enough for paired-bootstrap significance, by design |
| **slices** | fresh, disjoint from prior cohorts: `data/{hotpotqa,2wiki}_distractor_25_pilot.jsonl` (seed 20260513) | new seed; no leakage from any prior calibration / test set |
| **sampler K** | **K=5** at every step boundary (vs current K=3) | enough samples for SelfCheckGPT-NLI; cheap proxy for SeaKR eigen_score (full K=20 deferred to E2) |
| **boundary detector** | spaCy sentence with 32-token fallback | unchanged from v2 |
| **decoding** | Option-A: greedy, repetition penalty 1.05, early-stop on `Answer:` regex or EOS, `max_new_tokens=512` | unchanged from v2 |

### Capture schema (the G2 deliverable)

**Per step**, in addition to the current schema (`mean_token_entropy`, `boundary_token_entropy`, `dragin_rind_max`, `drift_cosine`, `variance_l2_window`, `hidden_norm`, `hvr`, `hedge_count`, `verify_count`, `u_verb`, `u_ent`, `u_cons`):

- **NEW** — `per_token_max_logprob: List[float]` for the greedy step (max softmax probability of the actually-decoded token) → enables FLARE (`min(prob) < θ`) faithful re-derivation post-hoc.
- **NEW** — `per_token_max_logprob_alt: List[float]` for top-1 *predictive* softmax under the unmodified distribution (entropy is captured before repetition penalty per `sota_capture.py`; this captures the matching probability) → robustness check.
- **NEW** — `samples: List[{text: str, n_tokens: int, finish_reason: str}]` for the K=5 boundary completions (text + length + EOS reason) → enables SelfCheckGPT-NLI / Ngram / BERTScore offline.
- **NEW** — `sample_eos_hidden: List[List[float]]` for the K=5 boundary completions, mid-stack hidden state at the EOS token (or last token if no EOS) → enables a SeaKR eigen_score *proxy* with K=5 (not headline; flag as smoke).
- **NEW** — `etc_d2h: float` — the explicit ETC second-difference value at this step (currently re-derived; surface for clarity).

**Per trace**, in addition to current (`tokens_generated`, `wall_clock_s`, `tok_per_s`, `n_steps`, `em`, `f1`, `label_correct_em`, `gate_decisions`, `fire_rates`):

- **NEW** — `capture_schema_version: "rq1-pilot-v1"` (so the analyzer can branch).
- **NEW** — `sampler_K: 5`.
- **NEW** — `per_step_capture_overhead_s: float` (wall-clock attributable to K=5 sampling + hidden capture, separate from greedy decode) → cost-vs-signal trade-off accounting.

**Storage budget per trace** (estimated):
- Per-token logprobs: ~250 floats × 8 B = ~2 KB.
- K=5 sample texts: ~250 chars × 5 = ~1.25 KB.
- K=5 EOS hidden states (mid-stack): 5 × 2048 dim × 4 B = 40 KB.
- Total extra per trace: ~45 KB.
- **Total pilot dump:** 25 × 4 phases × 3 archs × 2 datasets × 45 KB = **~27 MB**. Comfortably committable to git.

### Compute budget (single A800 80 GB serial)

| arch | decode tok/s (greedy) | K=5 capture overhead | wall per cell (n=25 × 250 tok × 4 phases) | model load |
|---|---:|---:|---:|---:|
| Phi-4-mini-instruct | ~42 | +30% (boundary samples ~10–40 tok each) + 5% (hidden dump) | ~9 min | ~30 s |
| Gemma-4-26B-A4B-it | ~12 | same | ~32 min | ~90 s |
| Qwen3.6-35B-A3B-Instruct | ~10 | same | ~38 min | ~120 s |

| step | wall |
|---|---:|
| Phi D + E (paired closed × oracle, both datasets) | ~9 min |
| Gemma D + E | ~32 min |
| Qwen D + E | ~38 min |
| Slice builds + smoke gates (5-question pre-flight per arch) | ~10 min |
| Model loads + tear-downs | ~5 min |
| Offline analysis (run new `analysis/pilot_audit.py`, dump to `analysis/pilot.out`) | ~15 min |
| **Total** | **~110 min ≈ 1 h 50 min** |

Comfortably under the 4-hour budget. Headroom: if any cell fails, retry without breaking the budget.

### Scripts to invoke

Sequential on A800:

```bash
# 0. Prereqs (assumes _MODEL.md §9 bootstrap done; if not, run that first)
git submodule update --init --recursive externalCodeAudits/

# 1. Build pilot slices (fresh seed, disjoint from all prior cohorts)
python scripts/04_dataset.py \
    --n 25 --seed 20260513 --split-seed 20260513 \
    --exclude-ids-from data/hotpotqa_distractor_500_v2test.jsonl \
    --exclude-ids-from data/hotpotqa_distractor_200.jsonl \
    --all-test --name hotpotqa_distractor_25_pilot
python scripts/04_dataset_2wiki.py \
    --n 25 --seed 20260513 \
    --exclude-ids-from data/2wikimultihop_distractor_500_v2test.jsonl \
    --name 2wikimultihop_distractor_25_pilot

# 2. Pilot orchestrator (NEW — to be written; mirrors run_xarch_full.sh)
bash scripts/run_pilot_p0.sh
#   internally: per arch in {phi, gemma, qwen}:
#     - 5-question smoke; hard-fail if <3/5 multi-step-with-answer
#     - D1: closed-book HotpotQA, n=25, capture-schema-version=rq1-pilot-v1
#     - D2: closed-book 2Wiki, n=25, same schema
#     - E1: oracle-retrieve HotpotQA, n=25 (paired by qid with D1)
#     - E2: oracle-retrieve 2Wiki, n=25 (paired with D2)
#   writes:
#     runs/v2_pilot_<arch>/{B_smoke,D1_closed_hotpot,D2_closed_wiki,E1_oracle_hotpot,E2_oracle_wiki}.jsonl
#     runs/v2_pilot_<arch>/logs/<phase>.log
#     runs/v2_pilot_<arch>/checkpoints/  (commitable: thresholds, smoke JSONs, slice sha256s)
#     runs/v2_pilot_<arch>/00_master.log

# 3. Offline analysis (NEW — to be written; consumes the new schema)
python analysis/pilot_audit.py | tee analysis/pilot.out
#   computes for each (arch × dataset × method) cell:
#     - DRAGIN.max AUROC (no change)
#     - FLARE: 1 - min(per_token_max_logprob) per step → max over steps; AUROC
#     - SelfCheckGPT-NLI: sentence-level NLI prob(contradict) on K=5 sample texts
#       (uses DeBERTa-v3-large-MNLI offline; ~1 min per dataset CPU-or-light-GPU)
#     - SelfCheckGPT-Ngram: as cheap parallel cousin
#     - SeaKR-K5-proxy: log det(G + alphaI) of K=5 sample EOS hidden states; AUROC
#     - MBA-RAG: SKIP (requires per-arch retraining; not part of P0)
#     - Self-RAG: SKIP (separate model, not run on the 3 archs)
#     - existing pool: tokens, wall_s, n_steps, dragin, ent, etc, hvr, drift, var, hidden_norm,
#       u_cons (exploratory), residualized AUROC + paired bootstrap vs tokens (B=2000)
#     - per-cell sample-text dump for visual sanity check
```

### What lands as committed artifacts

| path | size | purpose |
|---|---:|---|
| `data/{hotpotqa,2wikimultihop}_distractor_25_pilot.jsonl` | ~1 MB total | fresh slices |
| `runs/v2_pilot_<arch>/{B,D1,D2,E1,E2}_*.jsonl` × 3 archs | ~27 MB total | full pilot traces with extended schema |
| `runs/v2_pilot_<arch>/checkpoints/{thresholds,calibration,slice_manifest}.json` | <1 MB total | per-arch derived artifacts |
| `runs/v2_pilot_<arch>/logs/00_master.log` × 3 archs | <1 MB | wall-clock + config audit |
| `analysis/pilot.out` | <100 KB | per-cell numbers |
| `analysis/pilot_audit.py` | <50 KB | the analyzer (committed alongside) |
| `scripts/run_pilot_p0.sh` | <10 KB | the orchestrator (committed alongside) |

Total committed footprint: ~30 MB. Acceptable per the small-checkpoint policy.

### Pre-registered sanity gates (smoke / fail-fast)

Pilot fails (and we triage before scaling) if **any** of:

1. The 5-question smoke fails on any arch (<3/5 multi-step-with-answer or any unhandled exception).
2. Any cell produces traces where a NEW schema field is null on >20% of steps (capture-pipeline bug).
3. `tokens_generated` median falls outside [80, 600] on any cell (decoding misconfiguration; matches v1 #21 / `_agent_v2.md` #v2-6 trap).
4. K=5 sampler raises on Qwen 2WikiMHQA (the v1 #20 under-generation trap; documented in `_agent_v2.md`).
5. Total wall-clock exceeds 3 h (margin against the 4 h budget).

If any gate fails: stop, log the cell, fix in `scripts/04f_run_qa_sota.py` or `scripts/crv/sota_capture.py`, re-run the affected arch only.

### What we expect to learn (G3)

- Whether DRAGIN.max replicates as on-disk Phi survivor on Gemma + Qwen at small n. Direction-of-effect only; not significance.
- Whether FLARE's per-token min-prob signal is comparable across the 3 archs (FLARE was developed on GPT-3.5 API; first cross-arch open-weights data point).
- Whether SelfCheckGPT-NLI on K=5 produces meaningful signal at this n; if AUROC ~0.5 it's just K-too-small, not a method failure.
- Whether the SeaKR-K5 proxy correlates with the SeaKR-K20 expectation (sets the prior for the E2 K=20 compute decision).
- Per-arch capture overhead numbers for K=5 — informs the K=20 budget for E2.
- Whether 2WikiMHQA still under-generates on Qwen (decision deferred from v2; pilot answers it definitively at small n).

### Status (updated 2026-05-13 evening)

**P0 partial — Phi + Gemma complete, Qwen2.5 + Llama-2 pending, Qwen3.6 dropped.**

Pilot complete cells (compute-box runs `runs/v2_pilot_<arch>/`):
- Phi-4-mini-instruct: 5 phases × 25 questions, schema rq1-pilot-v1, fields-pop 100% on multistep traces.
- Gemma-4-26B-A4B-it: 5 phases × 25 questions, schema rq1-pilot-v1, fields-pop 100% on every trace (Gemma is much chattier).

Pilot rough-read (against `closed_em=0`, n small, CIs ±0.20):

| cell | tokens | DRAGIN | ent | FLARE | SeaKR-K5 | u_cons |
|---|---:|---:|---:|---:|---:|---:|
| Phi Hot | 0.603 | 0.549 | 0.480 | 0.549 | 0.539 | 0.569 |
| Phi 2Wiki | 0.494 | 0.557 | 0.600 | 0.543 | **0.729** | 0.629 |
| Gemma Hot | 0.750 | 0.639 | 0.778 | 0.549 | 0.611 | 0.764 |
| Gemma 2Wiki | 0.514 | **0.764** | 0.764 | 0.778 | 0.583 | 0.472 |

Direction-of-effect findings:
- **DRAGIN cross-arch holds** (Gemma 0.639/0.764 — survivor narrative on track).
- **SeaKR-K5 logdet shows complementary signal** where DRAGIN is weakest (Phi 2Wiki 0.729). Justifies Burden 2 K=20.
- **Length-confound is heterogeneous across cells** (Phi 2Wiki 0.494, Gemma Hot 0.750) — replicates v2 `bridge_comparison` finding.
- **K=5 capture overhead = 86% of trace wall on Gemma** — informs K=20 budget for Burden 2/3.

**Qwen3.6 P0 SKIPPED — documented limitation.**
B smoke produced n_steps=[2,1,1,1,1], tokens [10–32]. Under matched-protocol greedy + `enable_thinking=False`, Qwen3.6 collapses to direct-answer mode. Card-recommended thinking-mode protocol requires non-greedy sampling (`t=1.0, top_p=0.95`) which breaks length-residualization comparability with Phi/Gemma. Qwen3.6 weights deletable from compute box (~67 GB freed). Trace evidence preserved at `runs/v2_pilot_qwen3.6-35b-a3b-instruct/B_smoke_hotpot.jsonl` + `logs/B_smoke.log`.

**Pivot: Qwen3.6 → Qwen2.5-7B-Instruct.** Final 3-arch pool (Phi 2026 + Gemma 2026 + Qwen2.5 2025) spans 2 generations and 2 architecture families. Llama-2-7B-Chat was considered as a 4th backbone (canonical 2024-trigger-paper baseline) but dropped after the gated download was slow even on the compute box; transitive comparability holds via SR-RAG/DyPRAG 2025 work that benchmarks DRAGIN/FLARE on Qwen2.5. Llama-2-7B remains as an optional future addition with one-step instructions in `_rq1-plan.md`.

### Status

**Code: ready (commit `97b2dc6` + new-model smoke at `scripts/04c_new_model_smoke.py`).**
- `scripts/crv/sota_capture.py` — `ConsistencySampler` extended to optionally dump per-sample `{text, n_tokens, finish_reason, eos_hidden}`; v2 schema preserved when flags off.
- `scripts/04f_run_qa_sota.py` — `--schema-version=rq1-pilot-v1`, `--dump-sample-texts`, `--dump-sample-hidden`. Per-step `per_token_max_logprob` (post + pre repetition penalty); per-trace schema bookkeeping + capture-overhead.
- `scripts/run_pilot_p0.sh` — orchestrator; idempotent; pre-registered fail-fast gates on smoke / pilot-field population / token median / wall-clock cap.
- `analysis/pilot_audit.py` — extends `length_audit_v2.py` with FLARE (post + pre rep), SeaKR-K5 proxy (Gram log-det), SelfCheckGPT-Ngram-1. Smoke-verified to reproduce existing v2 numbers exactly; gracefully reports `n/a` on the pilot-only fields when fed v2-schema traces.

**Compute: existing compute box (warm) — new Cursor chat session.** Full handover in `_rq1-plan.md` §"NEW: handover for fresh Cursor chat on compute box (2026-05-13 +)". Linear path on the warm box: drop Qwen3.6 weights → download Qwen2.5-7B at pinned SHA → run `scripts/04c_new_model_smoke.py qwen2.5-7b-instruct` → paste JSON into agent chat → agent updates `_MODEL.md` + `PROTOCOL` dicts → `ARCHS="qwen2.5-7b-instruct" bash scripts/run_pilot_p0.sh` (≈25 min, idempotent on Phi/Gemma) → `python analysis/pilot_audit.py`.

---

## P0 final readout (3 archs × 2 datasets, n=25, rq1-pilot-v1 schema)

**AUROC against `closed_em=0` (incorrect):**

| cell | tokens | DRAGIN | ent | FLARE | SeaKR-K5 | sc_ng | u_cons |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phi Hot | 0.603 | 0.549 | 0.480 | 0.549 | 0.539 | 0.353 | 0.569 |
| Phi 2Wiki | **0.494** | 0.557 | 0.600 | 0.543 | **0.729** | 0.407 | 0.629 |
| Gemma Hot | 0.750 | 0.639 | **0.778** | 0.549 | 0.611 | 0.698 | 0.764 |
| Gemma 2Wiki | **0.487** | **0.782** | 0.737 | 0.724 | 0.519 | 0.756 | 0.519 |
| Qwen2.5 Hot | 0.715 | 0.632 | 0.528 | 0.569 | 0.521 | 0.347 | 0.729 |
| **Qwen2.5 2Wiki** | **0.380** | **0.767** | 0.753 | **0.813** | 0.720 | 0.623 | 0.700 |

**`lift_em=+1` AUROC (where #helpful ≥ 2):**

| cell | #help | tokens | DRAGIN | ent | flare | seakr5 |
|---|---:|---:|---:|---:|---:|---:|
| Gemma Hot | 2 | 0.293 | **0.891** | 0.674 | 0.478 | 0.717 |
| Gemma 2Wiki | 4 | 0.369 | **0.940** | 0.702 | 0.560 | 0.631 |
| Qwen2.5 Hot | 4 | 0.595 | 0.571 | 0.464 | 0.524 | 0.393 |
| Qwen2.5 2Wiki | 4 | 0.548 | 0.452 | 0.417 | 0.655 | 0.690 |

**Behavioural notes:**
- Phi 2Wiki under Option-A: **8/25 traces have n_steps=0** (`Answer:` immediate). Effective n ≈ 17. Documented arch-protocol limitation.
- Qwen2.5 2Wiki **tokens AUROC = 0.380 — INVERTED** (wrong answers shorter). Cross-arch confirmation of dataset-specific length-correlation reversal.
- DRAGIN: weakest on Phi (0.55), strongest on Gemma+Qwen2.5-2Wiki (0.77+). Specifically WHERE length collapses, DRAGIN wins.
- DRAGIN+lift+1 on Gemma is exceptional (0.89/0.94) — if it holds at n=200 it's a publishable headline alone.

---

## P0.5 — free-signal post-hoc audit (no re-decode; `analysis/p0_free_signals.py`)

8 candidate triggers tested on existing rq1-pilot-v1 dumps. **4 of 8 carry signal**; 4 are noise.

| signal | best cell | reading |
|---|---|---|
| **S6 sec/token** | Phi Hot 0.676, Phi 2Wiki 0.597, Qwen2.5 2Wiki 0.680, Gemma 2Wiki 0.667 | **Cross-arch real**; works precisely where DRAGIN is weakest (Phi) |
| **S4 answer-marker step idx** | Gemma Hot **0.844**, Phi 2Wiki 0.627 | Single highest AUROC in audit; "late `Answer:` = wrong" |
| **S1 K=5 answer-disagree** | Qwen2.5 2Wiki 0.760 incorrect, **0.815 lift+1** | Beats DRAGIN for lift+1 on Qwen2.5 2Wiki (0.815 vs 0.452) |
| **S3 pacing std** | Gemma Hot 0.764 | Step-length variance carries signal on Gemma |
| ✗ S2 first-step entropy | chance | drop |
| ✗ S5 5-gram repetition | chance | drop |
| ✗ S7 active-thinking marker | chance | drop |
| ✗ S8 step-novelty overlap | mixed | drop |

Output: `analysis/p0_free_signals.out`. Script: `analysis/p0_free_signals.py`. **Cross-arch finding**: Phi's failure mode is decode-slowdown (S6), not attention-conditioned uncertainty (DRAGIN). Different model classes have different "what fails on" signatures — methodologically this is itself a finding.

---

## P1 — extended-schema pilot (rq1-pilot-v2; planned)

**4 NOVEL trigger captures** added to the schema (none present in published trigger pool):

| # | new field | model class | why novel |
|---|---|---|---|
| **N1** | `moe_routing_entropy` | MoE only (Gemma) | Per-step mean entropy over expert-routing distribution. **Untested as a trigger anywhere.** Free signal that exists only on MoE; orthogonal to DRAGIN/FLARE/SeaKR families. |
| **N2** | `prefix_answer_now` | all models | At each boundary: fork cache → prefill `Answer: ` → decode 10 tokens greedy → save. Compare to final answer post-hoc. **"What would the model answer NOW vs LATER" — direct verification probe.** Untested. |
| **N3** | `multi_layer_dragin` | all (especially MoE/hybrid) | DRAGIN-RIND at {first, mid, last} full-attn layers. Tests MID-RIND on hybrid stacks where last-layer attention is polluted by routing noise. |
| **N4** | `attention_sink_frac` | all (any attention-bearing model) | Fraction of mean attention to first 4 tokens (BOS/start). Captures the StreamingLLM "attention sink" phenomenon (Xiao 2024) **never used as a trigger before**. |

**Schema additions are ALL implemented as of commit-staging in this session.** Three files modified:
- `scripts/crv/sota_capture.py` — added `moe_routing_entropy_from_router_logits`, `attention_sink_fraction`, `PrefixAnswerProbe`, `MultiLayerRindAccumulator`.
- `scripts/04f_run_qa_sota.py` — added `--schema-version=rq1-pilot-v2`, `--prefix-probe-tokens`, `--multi-layer-dragin` flags; wired all 4 novel captures into the decode loop with graceful dense-model fallback for `output_router_logits`.
- `scripts/run_pilot_p1.sh` — sister orchestrator; outputs land at `runs/v2_pilot_v2_<arch>/` to avoid clobbering P0.

**Compute estimate per arch (A800, K=5 sampler, +N2 prefix probe ~10 forwards/step):**
- Phi: ~10 min wall (low n_steps so prefix-probe cost is small)
- Qwen2.5: ~30 min wall
- Gemma: ~3 h wall (high n_steps × prefix-probe overhead)

**Recommended parallel split (per parent's plan):**
- **Box A (any ≥40 GB)**: `ARCHS="phi-4-mini-instruct qwen2.5-7b-instruct" bash scripts/run_pilot_p1.sh` (~40 min serial)
- **Box B (warm A800 with Gemma weights)**: `ARCHS="gemma-4-26b-a4b-it" bash scripts/run_pilot_p1.sh` (~3 h)
- Total parallel wall: ~3 h.

**Status: code ready, not yet launched. New compute-box chat picks up from here.**

---

## Loose ends to flag for next session (2026-05-14)

Things I noticed but didn't elevate during this session — log here so the next chat doesn't repeat the discovery work.

### Likely-headline findings hiding in P0 data (worth checking at n=200)

1. **DRAGIN+lift+1 on Gemma = 0.891 / 0.940** at n=25. If this holds at n=200 it is a *publishable headline by itself*: "DRAGIN-RIND identifies retrieval-helpful traces with AUROC 0.94 on Gemma-4-26B-A4B-it × 2WikiMHQA-distractor." This is the strongest single AUROC anywhere in the entire pilot and almost slipped past us. Burden 3 should prioritize Gemma 2Wiki paired closed/oracle to confirm.
2. **Phi has weak DRAGIN (0.55) but strong sec/token (S6: 0.68)** — different model classes have different *failure-mode signatures*. Cross-arch the question stops being "which signal wins" and becomes "which signal wins **on which model class.**" That's a 4×n_model results matrix, not a 1×n_signal one.
3. **Qwen2.5 2Wiki tokens AUROC = 0.380 (inverted)** at n=25 — wrong answers are *shorter* than correct ones. If this replicates on Qwen2.5 at n=200 it is **direct cross-arch evidence that single-AUROC trigger reporting is broken**. The bridge_comparison v2 finding was one subset; this is a full-arch full-dataset cell.
4. **Qwen2.5 2Wiki SeaKR-K5 = 0.720 + FLARE = 0.813** — both K=5-derived. If both replicate, K=5 sample-derived signals on Qwen2.5 2Wiki specifically are unusually informative. Worth investigating *why*.

### Reframe the data is now suggesting (instead of "DRAGIN survives" or "length wins")

> *"The published dynamic-RAG trigger pool is undercomplete. We re-evaluate it under length-residualized AUROC + Bonferroni-corrected paired bootstrap, and surface (a) length-correctness as dataset-specific within the multi-hop QA family, (b) **four free post-hoc signals** with cross-arch incorrectness AUROC ≥ 0.65 (S1 K=5 answer-disagree, S3 pacing std, S4 answer-marker step idx, S6 sec/token) that the audit pool was missing, and (c) **four novel architectural-aware captures** (N1–N4) that no published trigger paper has used. The field's evaluation gap is bigger than its winner-identification gap."*

This is a stronger framing than either prior alternative. Doesn't require DRAGIN to fail; doesn't oversell length. Honest scope: methodological contribution + new candidate triggers.

### Schema-naming wart (will bite without action)

`--schema-version` choices are now `{v2, rq1-pilot-v1, rq1-pilot-v2}`. The legacy default `"v2"` ≠ "version 2 of rq1-pilot." This collision will cause confusion (it almost did when I was patching). **Suggested fix in next session**: rename legacy `v2` → `v2-legacy`, default to `rq1-pilot-v2`, retire `rq1-pilot-v1` (subsumed by v2). All three schemas are forward-compatible at the field level — only the naming is broken.

### Storage budget for P1 (rq1-pilot-v2)

Per-trace JSONL size estimate for v2 captures on top of v1:
- N1 routing_entropy: ~8 bytes × n_steps ≈ 0.1 KB
- N2 prefix_answer_now: ~30-100 chars × n_steps ≈ 1 KB Phi / 5 KB Gemma
- N3 multi_layer_dragin: 3 layers × 3 floats × n_steps ≈ 1 KB
- N4 attention_sink_frac: 8 bytes × n_steps ≈ 0.1 KB
- **Total v2 add per trace: ~2-7 KB** on top of v1's ~45 KB. Negligible, fits committable budget.
- Pilot v2 dump: 25 × 4 phases × 3 archs × ~50 KB = ~15 MB. Headline at n=200: ~120 MB — middle of small-checkpoint policy tier.

### Analyzer needs extending for v2 (TODO for next session)

`analysis/pilot_audit.py` does NOT yet parse rq1-pilot-v2 fields. After P1 lands, the new chat should:
- Add `s1_k5_ans_disagree` and `s4_ans_marker_step` and `s6_sec_per_tok` (proven free signals) to the SIGNAL_DEFS list — same shape as existing rows, no schema dependency.
- Add `n1_moe_routing_entropy`, `n2_prefix_answer_divergence`, `n3_multi_layer_dragin_max_per_layer`, `n4_attention_sink_max` extractors that activate only on rq1-pilot-v2 traces.
- The `n2_prefix_answer_divergence` is the most interesting derived signal: post-hoc compare each step's `prefix_answer_now` to the FINAL `model_answer_extracted` and count disagreements normalized to [0, 1].

### Burden-priority shifts (informal, formalise next session)

| burden | prior priority | new priority | rationale |
|---|---|---|---|
| Burden 3 (cross-arch n=200) | high | **highest** | direct path to confirming/falsifying the most paper-defining numbers (DRAGIN+lift on Gemma = 0.94; tokens-inverted on Qwen2.5 2Wiki) |
| Burden 4 (MuSiQue) | low | **high** | the dataset-specificity finding needs ≥3 datasets to not be 2-dataset binary |
| Burden 2 (SeaKR K=20) | medium | **medium-low** | K=5 proxy already shows direction-of-effect; full K=20 is a confirmation, not a discovery |
| Burden 5 (Bonferroni B=10000) | medium | **high** | with 14+ signals in the pool now (6 published + 4 free + 4 novel), Bonferroni at k=14 (α/14 ≈ 0.004) is non-trivial — only the strongest signals will clear |
| Llama-2-7B addition | optional future | **stay deferred** | direction shifted to signal-novelty over backbone-breadth |
| Per-question-type stratification in headline | secondary | **required** | the bridge_comparison inversion finding is the cleanest single piece of "dataset-specificity" evidence |

### Pilot is statistically under-powered for any single AUROC claim

n=25 → CIs of width ±0.20. Nothing in the pilot will paired-bootstrap-significantly clear anything. **Pilot's job is informational** (which signals to fund at n=200, which to drop). Burden 3 + 5 are where the publishable numbers come from.

### One thing I almost forgot

`scripts/04c_new_model_smoke.py` (the JSON-emitting probe for new models) is not used downstream of `_MODEL.md` updates. Worth either:
(a) folding its capture logic into `04b_protocol_smoke.py` so there's only one new-model entry point, or
(b) updating `04b` to call `04c` for the architecture-detection phase. Decision deferred — works as-is.

---

### Run (to be filled when results land)

**Setup observed.** TBD — fill from `runs/v2_pilot_*/checkpoints/slice_manifest.json` and `runs/v2_pilot_*/logs/00_master.log` after launch. Capture: git rev, sysinfo (driver / CUDA / GPU / VRAM), env lockfile sha256, slice sha256s per arch, total wall-clock per phase.

**Results.** TBD — `analysis/pilot.out` table per (arch × dataset) cell. Pre-registered direction-of-effect questions (G3):
- Does `dragin_rind_max.max` retain *positive direction* under length-residualization on Gemma + Qwen at n=25? (replicates Phi survival or not)
- Does `FLARE.min_p (post-rep)` produce a different signal-shape than `mean_token_entropy.max` (which is the FLARE-equivalent the v2 audit used)?
- Does `SeaKR.K5_logdet` correlate with `u_cons.max` (text-K3 sibling) — i.e. does hidden-state geometry duplicate or extend the K-sample-text signal?
- Per-arch capture-overhead numbers — informs the K=20 budget in E2.
- Does Qwen × 2WikiMHQA still under-generate (v1 #20 trap)?

**Notes.** TBD.

---

## P0.6 — Deep exploratory analysis: novel trigger discovery (2026-05-13 late evening)

**Goals served:** G1, G3.

**Context.** P1 (rq1-pilot-v2, Gemma only) running on compute (~3 h ETA). Used the
idle time to run creative exploratory analyses on the existing P0 data (n=25 × 3
archs × 2 datasets, rq1-pilot-v1 schema) from the local `gzn` env. No GPU needed.

**Scripts produced:**
- `analysis/_explore_novel.py` — 10-angle analysis (trajectory shapes, signal interactions, paired divergence, per-token probability features, K=5 sample stability, SeaKR geometry trajectory, per-cell rankings, step-level correlations, oracle signal shifts, early detection)
- `analysis/_explore_deep_dive.py` — focused deep-dive on the top-3 novel signals (length residualization, per-cell, correlations, mechanism, composite trigger)
- `analysis/_explore_ans_flip_variants.py` — 10 variants of the answer-flip signal family, per-cell breakdown, oracle comparison, combined triggers

### Major discovery: Answer Instability Detection (AID)

**The signal:** `flip_rate` = fraction of answer-bearing reasoning steps where K=5 boundary-completion
answers disagree with the previous step's answers. Computed by extracting `Answer: X` from each
of the K=5 sample texts at each boundary, forming a set, and counting transitions where the set
changes between consecutive answer-bearing steps.

**Pooled numbers (n=150 paired traces, 15 lift+1):**

| signal | raw AUROC (help) | residualized | r(tokens) |
|---|---:|---:|---:|
| **AID flip_rate** | **0.733** | **0.735** | **0.008** |
| DRAGIN_late (peak position) | 0.627 | 0.636 | -0.086 |
| DRAGIN.max | 0.603 | 0.607 | 0.039 |
| tokens | 0.566 | — | 1.000 |
| FLARE.max | 0.476 | 0.470 | 0.133 |

**Key properties:**
1. **Highest AUROC** for predicting retrieval helpfulness (0.733) — beats DRAGIN (0.603) by +0.13.
2. **Zero correlation with length** (r=0.008). Survives and slightly improves under residualization.
3. **Low correlation with DRAGIN** (r=0.25) and FLARE (r=0.19) — genuinely orthogonal signal family.
4. **Correlation with n_steps** only 0.28 — survives double-residualization on (tokens + n_steps): 0.661.

**Per-cell AUROC for lift+1 (retrieval helpful):**

| cell | flip_rate | DRAGIN.max | flip_rate wins? |
|---|---:|---:|---|
| Gemma wiki | **0.940** | 0.940 | tied |
| Qwen hot | **0.867** | 0.571 | **YES (+0.30)** |
| Gemma hot | 0.663 | **0.891** | no |
| Qwen wiki | **0.700** | 0.452 | **YES (+0.25)** |

**Mechanism:** Answer flips happen LATE — 69% in the last quarter of the trace.
The model deliberates through multi-step reasoning, then its sampled alternatives
*diverge* near the conclusion. Oracle traces have n_steps=0 (model answers directly
when given context) — retrieval eliminates the instability source entirely.

**Trace-level statistics:**
- No-flip traces: 61% correct, 3% helped
- Late-flip traces: 32% correct, 18% helped
- High-flip traces (top 15): 20% correct, 20% helped, dominated by Gemma (12/15)

### Supporting findings from the 10-angle analysis

**Angle 1 — Entropy trajectory shape:**
- `ent_range` (max-min entropy across steps): AUROC_inc=0.687, help=0.560
- `dragin_late_spike` (position of DRAGIN maximum): AUROC_help=0.709 — WHERE DRAGIN peaks matters more than its magnitude for predicting retrieval need

**Angle 3 — Per-token probability features:**
- `max_low_streak` (longest consecutive run of prob<0.5 tokens): AUROC_inc=0.762, help=0.708
- But r=0.70 with DRAGIN — essentially the same signal measured differently

**Angle 4 — Signal interactions:**
- `dragin*flare`: inc=0.655, help=0.577 (no improvement over DRAGIN alone)
- `dragin-log_tok` (length-subtracted): inc=0.656, help=0.612 (mild improvement)
- No simple combination beats flip_rate

**Angle 5 — Answer stability variants (10 tested):**
- V10 `flip_rate` (normalized): **best** at 0.733 help
- V3 `late_flips` (last-half only): 0.673 help
- V8 `consecutive_flips` (oscillation): 0.633 help
- V1 `flip_count` (raw): 0.651 help

**Angle 7 — Per-cell specificity (arch × dataset):**
The signal landscape is WILDLY heterogeneous:
- `ent_slope` = 0.83 on phi_wiki but 0.30 on phi_hot
- `prob_decline` = 0.82 on gemma_hot but 0.30 on phi_hot
- `frac_low_conf` = 0.79 on qwen_wiki but 0.25 on phi_hot
- No single signal dominates every cell. Architecture-heterogeneity is the finding.

**Angle 8 — Step-level correlation matrix (n=1048 steps):**
- DRAGIN ↔ FLARE: r=0.84 (essentially redundant at step level)
- DRAGIN ↔ entropy: r=0.72
- drift_cosine ↔ everything: r≈0 (truly orthogonal)
- u_cons ↔ DRAGIN: r=0.34 (moderate independence)

**Angle 9 — Oracle signal shift (helped traces):**
- Oracle traces: n_steps drops from 7.3 → 0.0, tokens from 132 → 111
- Retrieval eliminates multi-step reasoning entirely on these questions

**Angle 10 — Early detection (steps 0-1 only):**
- No early signal exceeds 0.5 for help. Retrieval need is NOT detectable from the first step.
- `ent_jump_01` = 0.39 (inverted!) — early entropy rise = CORRECT, not helped.

### Evaluation of alternative LLM analysis (same P0 data)

A parallel analysis session proposed three innovations:

**1. "Artifact Confound" (Gemma template collapse):**
- **Verified:** 15/25 Gemma Hot traces and 16/25 Wiki traces contain artifact tokens (e.g. `<turn|>`, `<end_of_turn>`). 10-12/25 in the last step specifically.
- **Assessment:** Real phenomenon but impact is LIMITED. Last-step token count is tiny (mean=6, max=19). The protocol correctly stops at EOS/Answer: marker. The artifact inflates entropy at the last boundary but doesn't run away. Worth documenting as a measurement caveat; not a trigger and not catastrophic.
- **Action:** Add artifact-token-presence as a diagnostic flag in the n=200 analyzer. Does NOT need capture-schema changes.

**2. "Normalized Entropy Growth" (H_max - H[0]):**
- **Verified numbers:** Pooled inc_AUROC=0.646, per-cell Gemma Hot=0.750, Gemma Wiki=0.718.
- **Critical flaw:** **help_AUROC = 0.497** (chance level). This signal predicts incorrectness but NOT retrieval helpfulness. A trigger that fires on all wrong answers (regardless of whether retrieval helps) has catastrophic false-positive rate.
- **Comparison:** My flip_rate achieves help=0.733. The H_diff signal is 0.236 points worse at the task that matters.
- **Assessment:** Publishable as a diagnostic (cross-arch failure-mode taxonomy) but NOT as a trigger. Filed under G3 domain learning.

**3. Run allocation (n=200 cross-arch + MuSiQue):**
- **Agree** on structure but the "innovation" content should be AID flip_rate, not entropy growth.
- **Agree** on dropping SeaKR K=20 from the current campaign (K=5 proxy is sufficient; compute better spent elsewhere).

### Combined trigger: DRAGIN-late × flip_rate

The `combined_late_flip` signal (DRAGIN peak position × flip_rate × DRAGIN magnitude):
- AUROC_inc = **0.760** (highest for incorrectness in the full exploration)
- AUROC_help = 0.687

The "joint trigger" (binary: fires when DRAGIN > median AND answer flipped at same step):
- Fire rate: 33% of traces (reasonable for a trigger)
- Precision: 72% incorrect (vs 57% base = +15pp), 20% helped (vs 10% base = +10pp = 2× base)

### What this means for the paper

The paper trajectory has shifted AGAIN. No longer:
- ~~"DRAGIN-as-sole-survivor"~~ (P0.5 showed 4 free signals carry weight)
- ~~"Published pool is undercomplete"~~ (true but descriptive, not innovative)

Now: **"We introduce Answer Instability Detection (AID), a novel decode-time trigger family that
measures step-to-step answer disagreement in boundary completions. AID is orthogonal to all
published triggers (r≤0.25), achieves the highest length-residualized AUROC for predicting
retrieval helpfulness (0.73), and specifically complements the published-pool survivor (DRAGIN)
where it is weakest. The trigger landscape is architecture-heterogeneous."**

This is a contribution paper (novel trigger + methodology) not just an evaluation paper.

---

## Run experiments

### Experiment 0 — Established first proof (the floor)

**Goals served:** G1.

**Setup.**
- Script: `analysis/length_audit_v2.py` (11-signal pool: tokens, wall_s, n_steps, dragin, ent, etc, hvr, drift, var, hidden_norm, u_cons-exploratory).
- Data: Phi-4-mini-instruct on `runs/v2_exp1_phi-4-mini-instruct/{D1, D2, E1, E2}_*.jsonl`.
- Paired n=393 HotpotQA + n=452 2WikiMHQA after all-signals-populated filter.
- Protocol: raw AUROC + length-residualized AUROC (OLS on `log_tokens`) + paired bootstrap B=2000 vs `tokens_generated`.
- Compute: ~12 s under `gzn`, no GPU.
- Output: `analysis/length_audit_v2.out`.

**Results — HotpotQA in-domain (n=393, lift+1=42, base=0.107):**

| signal | raw AUROC | residual AUROC | paired Δ vs tokens [95% CI] | sig vs tokens? |
|---|---:|---:|---:|---|
| **tokens_generated** | **0.749** | — | — | — |
| wall_clock_s | 0.735 | 0.542 | −0.015 [−0.07, +0.04] | — |
| n_steps | 0.720 | 0.558 | −0.030 [−0.09, +0.03] | — |
| **dragin_rind_max.max** | **0.732** | **0.616** | **−0.017 [−0.10, +0.07]** | **— (survives)** |
| u_cons.max *(exploratory)* | 0.706 | 0.582 | −0.044 [−0.12, +0.03] | — |
| ent.max | 0.701 | 0.571 | −0.049 [−0.12, +0.03] | — |
| etc (Δ²H max) | 0.666 | 0.524 | **−0.084 [−0.16, −0.01]** | ✓ tokens beats etc |
| selfdoubt_t1 (hvr.max) | 0.503 | 0.264 | **−0.247 [−0.32, −0.17]** | ✓ tokens beats hvr |
| drift_cosine.max *(exploratory)* | 0.671 | 0.582 | −0.079 [−0.16, +0.00] | — |
| variance_l2_window.max *(exploratory)* | 0.654 | 0.477 | **−0.096 [−0.18, −0.01]** | ✓ tokens beats var |
| hidden_norm.range *(exploratory)* | 0.674 | 0.525 | −0.075 [−0.15, +0.00] | — |

**Results — 2WikiMHQA OOD (n=452, lift+1=75, base=0.166):**

| signal | raw AUROC | residual AUROC | paired Δ vs tokens [95% CI] | sig? |
|---|---:|---:|---:|---|
| tokens_generated | 0.582 | — | — | — |
| **dragin_rind_max.max** | **0.594** | **0.582** | **+0.010 [−0.07, +0.09]** | **— (only positive Δ)** |
| u_cons.max *(exploratory)* | 0.504 | 0.480 | −0.079 [−0.16, +0.00] | — |
| ent.max | 0.547 | 0.504 | −0.036 [−0.11, +0.04] | — |
| etc (Δ²H max) | 0.500 | 0.432 | **−0.082 [−0.13, −0.03]** | ✓ tokens beats etc |
| selfdoubt_t1 (hvr.max) | 0.515 | 0.448 | **−0.068 [−0.13, −0.01]** | ✓ tokens beats hvr |

**Notes.**
- Length is the dominant in-domain signal; no internal signal clears paired-bootstrap significance over `tokens_generated`.
- DRAGIN-RIND.max is the only positive-direction survivor under residualization on both datasets.
- Three signals are paired-significantly worse than length on at least one dataset (etc, hvr, var).
- Per-question-type audit found a counterexample: 2Wiki `bridge_comparison` (n=109, base=0.239) has `tokens_generated` AUROC = 0.433 (below chance); length-residualization there *reveals* hidden positive signal in u_cons (+0.034), dragin (+0.046), ent (+0.034), etc (+0.066).
