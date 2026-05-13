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

| id | name | goals | burden ref | compute | status |
|---|---|---|---|---|---|
| **P0** | **Smoke + sanity pilot — fresh data, max-logging schema, 3-arch pool (Phi + Gemma + **Qwen2.5-7B**) × HotpotQA + 2Wiki × {closed, oracle}, n=25/cell, K=5 sampler with sample-text + per-sample-hidden-state dump + per-token max_logprob dump** | **G1, G2, G3** | **prereq for E2..E10** | **~2.5 h on A800 (Phi + Gemma done; Qwen2.5 pending; Qwen3.6 + Llama-2 dropped)** | **partial: Phi+Gemma done; Qwen2.5 pending on new compute-box session** |
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
