## UPDATE 2026-05-15 ~13:45 — outlook revised after R1 complete + Phi-drop fired

**The estimates below are STALE.** They predate R1 (n=200 × 3 archs × 2 datasets), the moonshot 3-arch finding, and the Phi-drop decision. Current reality (see `_rq1-agent.md` "R1 Gemma full landing + Phi-drop decision + 3-arch headline" section):

| scenario | original estimate | **current estimate** |
|---|---:|---:|
| EMNLP-short acceptance, audit-flavored, Phi-dropped | n/a | **~50-60% combined (main + Findings)** |
| EMNLP-short main-track only | 30-35% with current-data-only | **~25-35%** |
| EMNLP-short Findings only | n/a (estimated path) | **~25-30%** |

**Why the upgrade.** Post-R1 + post-moonshot evidence dramatically strengthens the audit-paper:
1. **9 signals paired-bootstrap-significantly beat `tokens_generated`** on the Phi-dropped n=800 headline pool under length-residualization. Pre-R1 estimate was 2 (DRAGIN + 1 AID variant).
2. **M8 ECT** (Datta-`t*` extension, resid AUROC 0.691, Δ=+0.174 [+0.096, +0.252]) is a clean, citable, paired-significant strongest-survivor finding. **Provides the audit paper with a concrete name-able result.**
3. **UNCERTAINTY-LINE replication** at ε_protocol ≤ 0.040 across all 3 archs disarms the methodology-reimplementation reviewer attack.
4. **Cross-arch K=1 vs K=5 = regime 2** on `lift_em` across all 3 archs — the multi-sample walkback is no longer a paper risk; it's a clean ablation with a defensible recommendation (K=1 for compute).
5. **Phi sparse-flip becomes a feature, not bug** — moves to §4.3 as the "sparse-trace boundary cell" robustness finding. The Phi-dropped headline matrix is strictly cleaner.

**Why not higher.** The Datta-2026 precedent is still real; the contribution arithmetic is "audit + protocol + survivor matrix + M8-ECT-as-Datta-extension" — defensible but not breakthrough. **First paper, no in-domain advisor: realistic ceiling is Findings.** Main-track is plausible but not the default expectation.

**Submission decision.** Audit-flavored. Locked in `_rq1-agent.md` top-of-file. ARR submission deadline May 25, 2026 is 10 days out — recommend next-cycle submission for first-paper polish unless deadline pressure is real.

---

## EMNLP-short likelihood with current data only (HISTORICAL — see UPDATE above)

Honest, calibrated rating below. I'm using EMNLP-short base rate ~25-30% acceptance, then conditioning on what you actually have.

### What "current data only" means

| asset | what's on disk | strength |
|---|---|---|
| Phi v2 traces (n=393 paired Hot, n=452 paired 2Wiki) | `runs/v2_exp1_phi-4-mini-instruct/{D1,D2,E1,E2}_*.jsonl` | **publishable n** for one arch × two datasets |
| 11-signal length-residualized audit on Phi | `analysis/length_audit_v2.out` | **headline-quality** for Phi-only methodology paper |
| 3-arch P0 pilot (n=25/cell) | `runs/v2_pilot_<arch>/...` | **directional only**, not significance-testable |
| AID `flip_rate` discovery | `analysis/_explore_*.py` outputs | n=393/452 on Phi only; n=25/cell on cross-arch |
| Trigger pool re-ranking (DRAGIN survives, ETC/HVR/var-l2 fail) | derived from Phi v2 audit | **clean empirical finding** |
| Code-audit pool (DRAGIN, FLARE, SeaKR, SelfCheckGPT, MBA, Self-RAG) | `externalCodeAudits/` | **scope/positioning asset**, not yet executed |

### Probability of acceptance

**My calibrated point estimate: ~25-35%, dependent heavily on how you frame.**

Decomposed:

| framing | rough prior | conditional accept | joint |
|---|---:|---:|---:|
| "Length-controlled re-evaluation of dynamic-RAG triggers (Phi-only audit + 3-arch pilot)" — methodology paper | 0.50 | 0.40 | 0.20 |
| "AID-flip-rate as novel retrieval trigger (extension of Datta et al.)" — empirical paper | 0.30 | 0.20 | 0.06 |
| Both, with AID as one finding among many in a Phi-anchored audit | 0.20 | 0.45 | 0.09 |
| **expected acceptance probability** | | | **~0.30–0.35** |

That's marginal-but-feasible — slightly above EMNLP-short base rate due to genuine methodology novelty, dragged down by hostile-review playbook items and the Datta-precedent walkback.

### What's working FOR you

1. **Methodology contribution is intact and addresses a flagged open problem.** Brown SLR 2025 (in your own lit review!) explicitly says *"Evaluation remains overlap-heavy; cost/safety often neglected"* and *"Generalization under domain shift is brittle."* You operationalize the fix. **+10pp** above base rate.
2. **The trigger-pool re-ranking is empirically sharp on Phi at publishable n.** "DRAGIN-RIND is the only length-robust survivor on Phi (n=393/452, 11 audited signals); ETC, HVR, variance-l2 paired-bootstrap-significantly worse than length" is a clean empirical headline. **+5pp.**
3. **`lift_em = +1` paired closed-vs-oracle target is genuinely novel** for trigger evaluation and reads as the "right deployment question." Datta et al. don't do this. **+5pp.**
4. **The Datta precedent actually helps the framing** — it demonstrates the field cares about answer-trajectory primitives. You're now a Datta-extension, which is easier to position than a from-scratch claim. **+3pp.**
5. **EMNLP-short is the right venue size for this work** — too small for a long paper but enough for the methodology + audit. **+3pp.**

### What's working AGAINST you

1. **Single-architecture headline.** Reviewer 2 ("RQ1 is methodological; one model is insufficient" — you anticipated this in `_rq1-plan.md` Burden 3). With current data, the cross-arch claim is n=25/cell, which any reviewer will flag as too thin for paired-bootstrap statistics. **−15pp.**
2. **AID at n=25 cross-arch is suggestive, not significant.** If you headline AID, reviewers will ask for the n=200 confirmation. If you bury AID, you weaken the contribution. **−10pp.**
3. **The Datta-precedent walkback shrinks the AID contribution** to "K=5 multi-sample extension + retrieval-trigger application" — defensible but less exciting than what was previously a from-scratch novel signal. The K=5-vs-K=1 ablation must be in the paper or reviewers will ask. **−5pp.**
4. **The lit review (`-rq1-v2-litReview.md`) is currently shallow** — missing INSIDE, Self-Consistency, SAC³, Datta, Hou, Mao, Chain-of-Probe. Submitting without expanding §2 is a fast-rejection risk. **−5pp** (recoverable in 2-3 days of writing).
5. **2WikiMHQA inverted-length-correlation finding is intriguing but currently fragile** (one cell, one arch). **−3pp.**
6. **No comparison to FLARE / SeaKR-K20 / SelfCheckGPT-NLI head-to-head** — code audit done, but methods not actually run. Reviewer will ask. **−5pp.**

### Most-honest framing for the current-data-only short paper

> **Title:** *"A Length-Controlled Evaluation Protocol for Dynamic-RAG Triggers, with a Step-Level Answer-Instability Extension"*
>
> **§1 contribution claims:**
> 1. We introduce a length-controlled evaluation protocol (residualized AUROC + paired bootstrap, `lift_em = +1` target) for dynamic-RAG trigger signals on multi-hop QA paired with oracle re-decode.
> 2. Under this protocol, applied to 11 published trigger signals on Phi-4-mini-instruct (n=393/452 paired traces on HotpotQA-distractor / 2WikiMHQA-distractor), DRAGIN-RIND is the only length-robust survivor; ETC, SELFDOUBT-T1 HVR, and variance-l2-window are paired-bootstrap-significantly worse than length.
> 3. We extend the *forced-answer-completion* primitive of Datta et al. (2026) — originally proposed for early-stopping in reasoning models — into a retrieval-triggering signal via K=5 multi-sample boundary completions ("AID-flip-rate"). On Phi, AID-flip-rate matches DRAGIN-RIND under length residualization (raw 0.733, residualized 0.735, r(tokens)=0.008) and is orthogonal to it (r=0.25), suggesting per-architecture complementarity.
> 4. A pilot (n=25 × 3 archs × 2 datasets) provides directional evidence for cross-architecture survivor heterogeneity, motivating a larger-scale follow-up.

This framing is **defensible at ~30-35% acceptance probability** with current data.

### What would push you to 50%+

| action | impact | wall-clock |
|---|---|---|
| Run R1 (n=200 cross-arch) | **+15-20pp** | ~22h on one A800 |
| Run A1 K=1 vs K=5 ablation | **+5pp** (head-off Datta critique) | <1h |
| Hand-curate lit review with the 7 must-cites | **+5pp** (avoids fast-reject) | 2-3 days |
| Add bootstrap CIs to the n=25 pilot tables | **+3pp** (shows you know the limits) | <1h |
| Run FLARE / SelfCheckGPT-NLI on existing K=5 traces | **+3-5pp** (closes "incomplete pool" critique) | ~1-2h |
| Pre-submission anticipated-reviewer rebuttal pass | **+3pp** | half day |

### What pushes you to 60-70%

R1 + R2 (MuSiQue) + A1 + lit review hand-curation + FLARE/SelfCheckGPT-NLI integration on R1 traces — about 35h compute and 3-4 days of writing. That's the path the user's `_rq1-plan.md` already lays out.

### Bottom line

| scenario | current data only | + R1 | + R1 + R2 + A1 + lit fill |
|---|---:|---:|---:|
| EMNLP-short acceptance estimate | **30-35%** | **45-55%** | **60-70%** |
| risk of fast-reject for "single-arch / cross-arch unsupported" | high | low | very low |
| risk of fast-reject for "novelty walkback / Datta precedent" | medium-high | low (if A1 in paper) | low |

**Honest recommendation:** the current data is enough to *write* the paper now in the framing above and have something defensible at submission. But the cost-benefit of R1 (+22h compute, +15-20pp acceptance probability) is overwhelming — that's by far the highest-leverage single action available to you, and your plan already has it queued. **Don't submit current-data-only without R1 unless there's a hard external deadline.**

What's the actual EMNLP deadline you're aiming at? That changes the recommendation — if it's >3 weeks out, run R1; if it's <2 weeks and you can't get compute, write current-data-only with explicit "n=25 pilot — directional" flagging in the cross-arch table.