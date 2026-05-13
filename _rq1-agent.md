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

## Planned experiments

| id | name | goals | burden ref | compute (single A800) | status |
|---|---|---|---|---|---|
| E1 | Re-apply audit extensions (SeaKR ln-entropy proxies + 6 aggregator-robustness rows; 11 → 19 signals) | G1 | Burden 1 | 0 (analysis) | not started |
| E2 | Phi K=20 SeaKR eigen_score + FLARE per-token logprob dump + SelfCheckGPT K=3 sample-text dump (single bundled re-decode) | G1, G2, G3 | Burden 2 + Burden 8-FLARE + Burden 8-SelfCheck | ~12 h | not started |
| E3 | Cross-arch baseline on Gemma-4-26B-A4B-it (n=200, D + E, full v2 capture + bundled FLARE/SelfCheck dumps) | G1, G3 | Burden 3 (Gemma) | ~4.5 h | not started |
| E4 | Cross-arch baseline on Qwen3.6-35B-A3B-Instruct (n=200, D + E, full v2 capture + bundled FLARE/SelfCheck dumps) | G1, G3 | Burden 3 (Qwen) | ~5.5 h | not started |
| E5 | Cross-arch SeaKR eigen_score on Gemma + Qwen (K=20, hidden-state-sample capture; conditional on E2 outcome) | G1 | Burden 2 + Burden 3 | ~13 h | conditional |
| E6 | MuSiQue benchmark-family generalization, n=200 × 3 archs (closed + oracle) | G1 | Burden 4 | ~11 h | not started |
| E7 | Statistical hardening — re-run audit at B=10 000, Bonferroni α=0.005 across k=10 published triggers | G1 | Burden 5 | 0 | not started |
| E8 | Lift-conditional secondary table — `P(oracle helped \| trigger fired) − base` at fire rates {5%, 10%, 20%} with bootstrap CIs | G1 | Burden 6 | 0 | not started |
| E9 | MBA-RAG per-arch BERT-base bandit retraining on calibration cohort, 3 archs × 2 datasets | G1, G3 | Burden 8-MBA | ~12 h | not started |
| E10 | Self-RAG-LLaMA-2-7B reference-model row on HotpotQA-500 / 2Wiki-500 | G1, G3 | Burden 8-SelfRAG | ~2 h | not started |

**Total compute envelope:** ~60 h serial (~30 h with two boxes).

**Recommended sequencing for highest-information-per-compute single-week campaign:** E1 → E2 → E3 → E4 → E7 → E8. Lands the headline cross-arch + SeaKR + FLARE + SelfCheckGPT integration with statistical hardening.

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
