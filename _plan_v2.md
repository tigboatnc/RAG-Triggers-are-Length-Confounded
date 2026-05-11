# _plan_v2.md

> Forward-looking design document for the v2 evaluation campaign. Self-sufficient from this point — does not require reading `_PLAN.md` or `_AGENT.md` (v1). `_MODEL.md` is the source of truth for per-model protocol and is unchanged.
>
> Companion log: `_agent_v2.md` (experimental entries land there).
> v1 artifacts (`_PLAN.md`, `_AGENT.md`, `runs/phase4_*`, `data/hotpotqa_distractor_200.jsonl`) are retained for audit but **not load-bearing**: no claim in v2 derives statistical evidence from them.

---

## What the project is asking

Can a parameter-free, pre-decode signal from an LLM's mid-decoding hidden states predict, at the granularity of a reasoning step, whether the LLM should pause and retrieve external evidence? The signal of interest is the **Continuous Representation Vector (CRV)** — the mean-pooled mid-stack hidden state over the tokens of a reasoning step — and two scalar features derived from it: cosine drift between successive CRVs, and rolling L2-norm variance.

The honest contribution scope is **detection**, not intervention. The gate's quality is judged by whether, when it fires, oracle retrieval actually fixes incorrect traces (lift-conditional precision) — not by downstream EM lift under a deployed RAG policy.

## What v2 has to do that v1 did not

v1 produced pilots that are below the 2026 evaluation floor and are therefore non-publishable as significance claims. The audit (in `_AGENT.md` #26) confirmed three constraints v2 must satisfy:

1. **n ≥ 300 test per dataset**, with 500 as the working target (matches Adaptive-RAG / FLARE / ReARTeR / GainRAG / Semantic Entropy).
2. **≥ 2 datasets**, with at least one zero-shot OOD (matches ReaLM-Retrieve / Search-R1 / SEP).
3. **Multiple-comparison-corrected paired tests** (Bonferroni or equivalent) for any gate-vs-gate claim.

Calibration n=120 is within the community floor (SELFDOUBT n=90; ReARTeR n=200) and does not need to scale.

## Design constraints (non-negotiable)

| constraint | rationale |
|---|---|
| Zero-leakage: test split is opened **once**, at writeup time, after all model-selection knobs are frozen on calibration. | Standard hygiene; v1's L5 contract carries over. |
| Frozen Option-A decoding protocol per `_MODEL.md`: greedy, per-arch `enable_thinking` and `repetition_penalty`, early-stop on `Answer:` regex or EOS, `max_new_tokens=1024`. | Reproducibility; matches the protocol the pilot scripts already implement. |
| Baseline pool: **DRAGIN, ETC, SELFDOUBT-Tier 1, Random**. Hybrid simulators (ReaLM-sim, GRIP-sim) are retired per v1 #16 and do not return. | Reviewer credibility. |
| Statistical reporting: paired bootstrap B=10,000, paired-margin tests, **Bonferroni correction** across the gate pool. McNemar test as parallel non-parametric check. | 2026 dynamic-RAG standard. |
| Architecture-portable / dataset-portable claims require **at least one zero-shot OOD test**. | Otherwise reviewers can attribute findings to dataset-specific quirks. |

## What is portable from v1 (code + knowledge only, no results)

### Code (uncommitted but parsing-OK on the workspace)

| script | role | parameterised on |
|---|---|---|
| `scripts/04_run_qa.py` | canonical decode driver + EM/F1 grader; chat-template-aware | model, dataset, layer-mode |
| `scripts/04f_run_qa_sota.py` | sweep driver with DRAGIN attention capture + per-step entropy + boundary detection | model, dataset, layer-mode |
| `scripts/04_dataset.py` | HotpotQA-distractor slice builder (stratified by level × type, with `slice_sha256`) | n, seed, source mirror |
| `scripts/04_dataset_2wiki.py` | 2WikiMHQA-distractor slice builder, schema-normalized to match HotpotQA | n, seed, source mirror |
| `scripts/04g_sota_showdown.py` | Δfrac calibration + paired bootstrap analyzer | gate pool, target fire rate |
| `scripts/04l_oracle_retrieve_decode.py` | oracle-retrieve re-decode (filters distractors, re-decodes with supporting paragraphs only) | model, dataset |
| `scripts/04m_utility_scoring.py` | per-policy {EM, F1} with paired bootstrap | min_steps, bootstrap_json |
| `scripts/04n_lift_conditional.py` | lift-conditional gate diagnostics (recall, precision, Δlift) | min_steps |
| `scripts/qa_grading.py` | answer extraction, EM, F1 (canonical) | — |
| `scripts/crv/` | CRV tracker, boundary detector, baseline gate classes (DRAGIN, ETC, SELFDOUBT-T1, Random) | — |

None of these need methodological changes. They are dataset-size-agnostic; v2 only changes the slice sizes and dataset set.

### Knowledge (calibrated lessons, not statistical claims)

1. **`enable_thinking=False` (Qwen) works on HotpotQA, fails on 2WikiMHQA.** Qwen under-generated (~30 tokens, no `Answer:` line, no reasoning chain) on 2Wiki under the v1 protocol. Protocol amendment options for v2: enable thinking on both, or enable thinking only on 2Wiki, or accept under-generation on 2Wiki as a documented OOD finding. **Decision deferred.**
2. **Qwen on HotpotQA-distractor closed-book is ~62-80% EM** depending on split. Closed-book ceiling leaves little retrieval headroom for oracle-conditional lift signal. **Phi has lower closed-book EM (~44%)**, which leaves more lift room and may be a better single-architecture starting point for utility-conditional evaluation.
3. **Trace-level fire rate ≠ step-level fire rate.** Calibrating gates at step-level iso-fire-rate=0.30 produces trace-level fire rates of 80-90% on Qwen, which collapses gate selectivity. v2 should **calibrate at the trace level** (i.e., target trace-fire-rate ∈ {0.10, 0.30}) or analyze step-level explicitly.
4. **L1 (MoE expert-routing noise at last layer)** is real on Qwen and Gemma; mid-stack mitigation is real per Phase 2 (`figs/phase2_*_midstack.png` artifacts). Phi (dense) has no L1, so mid-stack vs last-layer is an empty distinction. This argues for mid-stack as the default capture site on MoE archs and either layer being fine on Phi.
5. **Δfrac (calibration metric) is fire-rate-confounded.** A gate firing at 90% will reach Δfrac > 0 on any dataset where short-trace ↔ correct-trace correlation exists. The defensible metric is **lift-conditional precision** (P(oracle helped | gate fired)) computed against a per-trace lift label. v2 should report both, with lift-conditional as the primary.
6. **HotpotQA-distractor supports oracle retrieval cheaply.** Gold supporting facts are already in the prompt mixed with distractors; oracle-retrieve is a paragraph filter, not a retriever invocation. 2WikiMHQA-distractor has the same structure.
7. **All four baseline gates fired at much higher trace-level rates than intended on v1** (DRAGIN 80-95%, ETC 55-70%, SELFDOUBT-T1 0-7%, Random matched). v2 should calibrate the **target fire rate at the granularity of the comparison**, not assume step-level calibration translates.

## What v2 explicitly does not commit to

- A specific submission venue or timeline.
- A specific model set (Phi-only, Qwen-only, or all-three).
- A specific second dataset (2WikiMHQA, MuSiQue, Bamboogle, IIRC).
- A specific thinking-mode policy for any model on any second dataset.
- A specific aggregation choice (drift / variance / OR) — must be cross-validated per arch on calibration.
- The choice between k-fold CV on calibration vs a held-out validation split. Both meet the zero-leakage constraint; the choice is operational.

These decisions land in `_agent_v2.md` entries as parent specifies them.

## Workflow contract for `_agent_v2.md`

Experiment-first, not pre-registration-first. Each experiment is logged in two parts:

- **Experiment N** — what was run. Model, dataset, n, knobs, scripts invoked, wall clock, logs paths.
- **Results N** — what came out. Numbers, tables, raw JSON paths, anything weird observed. Both positive and negative results land with the same weight.

No success/failure/inconclusive criteria are committed *before* a run; results are read on their merits. Heavy logging at every layer (per-trace, per-step, per-bootstrap-resample) is mandatory — post-hoc analysis is always valuable, especially when an experiment surfaces something the original framing didn't anticipate.

Pilots from v1 are referenced by id (e.g. v1 #20, v1 #26) but never cited as evidence for v2 claims.
