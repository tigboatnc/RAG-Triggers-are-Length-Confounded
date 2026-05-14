## EMNLP-short likelihood with current data only

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