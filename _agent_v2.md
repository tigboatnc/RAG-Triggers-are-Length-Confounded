# _agent_v2.md

> Experimental log for the v2 campaign. Self-sufficient from this point — does not require reading `_AGENT.md` (v1). Forward-looking plan lives in `_plan_v2.md`. `_MODEL.md` is unchanged and authoritative for per-model protocol.
>
> v1 artifacts (`_AGENT.md`, `runs/phase4_*`, `data/hotpotqa_distractor_200.jsonl`) are retained as code-and-knowledge inheritance only. **No statistical claim in v2 derives from v1 numbers.**

---

## State at session start (2026-05-11)

- v1 produced pilots that are below the 2026 evaluation floor (n=40 test, single dataset). See `_AGENT.md` #26 for the full literature audit + retraction of v1 significance claims.
- The decision to restart cleanly in v2 was taken after that audit landed.
- Code and knowledge inheritance from v1 is enumerated in `_plan_v2.md` §"What is portable from v1". The carrying-forward summary is: **all scripts under `scripts/` are reusable; no v1 results are evidence in v2**.
- Hardware: parent has compute-box access; local PC has no GPU and is for analysis only.

---

## Carry-overs from v1 (what v2 inherits)

### Carry-over A — Code

All scripts under `scripts/` parse and import correctly on the workspace env. The pipeline is:

```
slice builder → decode driver → trace capture → calibration → bootstrap analyzer
                       ↓                                              ↑
              oracle-retrieve re-decode → utility / lift-conditional scoring
```

Reusable as-is. Dataset-size-agnostic. Parameterised on model, dataset path, layer-mode, fire-rate target.

### Carry-over B — Knowledge (operational lessons, no result transfer)

- **HotpotQA-distractor** is the natural in-domain anchor. Gold supporting facts are in the prompt mixed with distractors; oracle-retrieve is a paragraph-title filter.
- **2WikiMHQA-distractor** has the same prompt schema (after `04_dataset_2wiki.py` normalization) and is the most obvious OOD second dataset. Caveat: Qwen `enable_thinking=False` under-generates on 2Wiki; Phi (no thinking mode) was not observed to have this problem during v1 Phase 3.
- **Closed-book EM gradient across architectures** on HotpotQA-distractor (v1 pilot, directional only): Phi ~44%, Gemma ~50-58%, Qwen ~62-80% (split-dependent). For utility-conditional analyses, **lower closed-book EM leaves more retrieval headroom** and produces more lift-positive labels per test sample.
- **Baseline gate pool**: DRAGIN, ETC, SELFDOUBT-Tier 1, Random. Hybrid simulators (ReaLM-sim, GRIP-sim) are permanently retired. Source-of-truth implementations in `scripts/crv/baseline_gates.py`.
- **L1 landmine** (MoE expert-routing noise at last layer) is established on Qwen and Gemma in v1 Phase 2; mid-stack mitigation works. Phi is dense, no L1. Capture at mid-stack by default on MoE archs; either layer is acceptable on Phi.
- **Δfrac is fire-rate-confounded**; lift-conditional precision (P(oracle helped | gate fired)) is the more diagnostic primary metric.
- **Trace-level fire rate ≠ step-level fire rate** — calibrating at step-iso-0.30 produces trace-iso-0.85+ on Qwen. v2 should target the granularity of the actual comparison.

### Carry-over C — Statistical burden (the bar v2 must clear)

Per `_AGENT.md` #26 audit:

| dimension | v1 | v2 floor | v2 target |
|---|---|---|---|
| test n per dataset | 40 | 300 | 500 |
| dataset count | 1 | 2 (1 in-domain + 1 OOD) | 2 |
| paired-margin tests | bootstrap | bootstrap + Bonferroni | bootstrap + Bonferroni + McNemar |
| OOD protocol | none | frozen thresholds, no re-fit | frozen thresholds, no re-fit |
| multiple-comparison correction | none | yes (k≥4 gate comparisons) | yes |

---

## Open burdens (what v2 must do before any submission claim)

### Burden 1 — Choose the v2 evaluation cohort

- Decide model set: Phi-only / Qwen-only / one-then-other / all-three.
- Decide datasets: HotpotQA-distractor (anchor) + which OOD (2WikiMHQA / MuSiQue / Bamboogle / IIRC).
- Decide cohort sizes: aim ≥500 test per dataset, ≥120 calibration (reuse v1 calibration if dataset matches, else build fresh).

### Burden 2 — Lock the protocol amendments per (model × dataset)

- `enable_thinking` per Qwen × dataset pair. v1 finding: ✗ on 2Wiki under `enable_thinking=False`. Options: enable for both, enable only for 2Wiki, or accept under-generation as OOD finding.
- `repetition_penalty` per model (already pinned in `_MODEL.md`).
- Boundary detector ("sentence", n_tokens=32) — pinned; can be revisited.

### Burden 3 — Decide model-selection knobs (mode, fire rate) procedure

- Cross-validate per-arch latent mode (drift / variance / OR) and target fire rate on calibration: either k-fold CV on calibration only, or split calibration into a calib + val sub-split.
- Pick **target granularity**: step-level iso-fire-rate or trace-level iso-fire-rate.

### Burden 4 — Heavy logging at every layer

Per-trace decode metadata, per-step trace records, per-resample bootstrap arrays, slice manifests with seed + sha256. Each experiment dumps a structured JSON that is post-hoc-replayable. No pre-registration ceremony; experiments are read on their merits.

### Burden 5 — Build OOD slice + smoke

If OOD dataset is 2WikiMHQA, the slice builder already exists (`scripts/04_dataset_2wiki.py`). Smoke before any full sweep: 5-question run, check `tokens_generated >= 60` on ≥3/5, all `eos` or `answer_marker`, no errors. Same gate as v1 #21.

### Burden 6 — Last-layer ablation on at least one MoE arch (mechanism claim)

If the paper claims mid-stack is essential, last-layer must be tested on at least Qwen (or Gemma) at v2-scale. Otherwise drop the mechanism claim and frame mid-stack as a defensive choice motivated by prior work (Phase 2 internal results) without re-asserting it as a finding.

### Burden 7 — Compute audit

Whatever cohort/model set is chosen, the compute estimate must be logged before the sweep, and the actual wall-clock measured against it.

---

## Method suggestion — single-architecture, burden-meeting, fast-to-execute test on Phi-4-mini

**Suggested method**: a single-architecture lift-conditional gate-quality test on Phi-4-mini-instruct, two datasets (HotpotQA-distractor in-domain + 2WikiMHQA-distractor zero-shot OOD), 500 test questions each, calibrated on the existing 120-question HotpotQA calibration slice with k-fold CV.

### Why Phi (and not Qwen or Gemma)

- **Closed-book EM is in the right band.** Phi was the lowest-EM model on v1 HotpotQA (~44% closed-book). Oracle retrieval has room to flip ~30-50% of incorrect traces to correct, meaning the lift-positive class will be **populated** at 500 test samples — expected ~50-100 lift-positives per dataset, vs Qwen's ~12-25 at the same n. Lift-conditional analyses become statistically informative.
- **No L1 landmine.** Dense decoder, no MoE routing noise. Mid-stack vs last-layer is empirically empty on Phi, so the method is tested in isolation from the mechanism question. Cleaner causal story.
- **Fast.** Phi decodes at ~42 tok/s on A800 vs Qwen's ~12 tok/s. The full 4-decode-pass schedule (closed-book × 2 datasets + oracle-retrieve × 2 datasets, ~2000 total decodes at ~250 tokens each) lands in **~3-4 h wall clock** — feasible in one compute session.
- **Honest scope.** "On the smallest-active-param edge LLM, the variance-mode CRV gate is the first parameter-free pre-decode signal to selectively identify retrieval-helpful reasoning traces under zero-shot OOD transfer" is a single-arch claim that does not over-reach.

### Headline claim under test

> **H1 (in-domain):** On HotpotQA-distractor (n=500 test), the variance-mode CRV gate has **paired-significantly higher P(oracle_helped | fired)** than each of {DRAGIN, ETC, SELFDOUBT-Tier 1, Random}, with Bonferroni-corrected α = 0.05 / 4 = 0.0125.
>
> **H2 (OOD):** With thresholds frozen from HotpotQA calibration, the same gate retains paired-significant advantage on 2WikiMHQA-distractor (n=500 test).

### Experimental design

| step | what | input | output |
|---|---|---|---|
| 1. Slice build | reuse v1 HotpotQA-200 calibration rows (n=120 calibration); build HotpotQA-500-test (fresh seed, disjoint from v1 200); build 2WikiMHQA-500-test (fresh) | `scripts/04_dataset.py` rerun with new seed + size; `scripts/04_dataset_2wiki.py` rerun | three JSONLs in `data/` |
| 2. Smoke (×2) | 5-question smoke per dataset, criteria from v1 #21 | Phi + slice JSONL | smoke logs in `runs/v2_*` |
| 3. Calibration | k-fold CV on the 120-q HotpotQA calibration: pick mode (drift / variance / OR) and threshold targeting **trace-level fire rate 0.30** | existing v1 calibration traces (Phi mid_stack, already on disk) | frozen knobs JSON in `runs/v2_phi_calibration.json` |
| 4. In-domain decode | Phi × HotpotQA-500-test, mid-stack CRV + DRAGIN attention capture + per-step entropy | `scripts/04f_run_qa_sota.py` | `runs/v2_phi_hotpot_500.jsonl` |
| 5. OOD decode | Phi × 2WikiMHQA-500-test, same configuration | same | `runs/v2_phi_2wiki_500.jsonl` |
| 6. Oracle re-decode (×2) | Phi × {HotpotQA-500, 2Wiki-500} with supporting-only paragraphs | `scripts/04l_oracle_retrieve_decode.py` | `runs/v2_phi_*_oracle.jsonl` |
| 7. Bootstrap analysis | lift-conditional Δ_em + Δ_f1 with Bonferroni-corrected paired margins; McNemar parallel | `scripts/04n_lift_conditional.py` + small wrapper for Bonferroni | `runs/v2_phi_headline.json` |

Compute estimate:
- Decode (×4 = HotpotQA closed-book + 2Wiki closed-book + HotpotQA oracle + 2Wiki oracle): 4 × 500 × 250 tok / 42 tok-s ≈ 200 min = ~3.3 h.
- Trace-capture overhead (~10% from v1 Phase 3): +20 min.
- Analysis (local, no GPU): <30 min.
- **Total: ~4 h compute + ~1 h analysis.** Single compute session.

### What we report when it lands

- Per-gate trace-level fire rate (in-domain + OOD).
- Per-gate lift-conditional precision = P(oracle helped | gate fired). Paired bootstrap (B=10k) for the latent gate vs each baseline. Bonferroni-corrected (k=4 → α=0.0125) AND uncorrected p-values both reported.
- Per-gate recall on lift_em=+1 and lift_em=-1 (false-harm-positive rate).
- McNemar test on the binary lift_em=+1 label between latent and each baseline.
- Lift-positive class breakdown by question type (compositional / bridge / comparison) and closed-book confidence band.
- Wall clock, GPU memory, throughput per phase.
- All bootstrap arrays dumped to disk so any post-hoc statistic can be re-derived without re-decoding.

Read the results on their merits.

### What this method *deliberately does not* claim

- Architecture portability (one arch tested).
- Mid-stack vs last-layer mechanism (irrelevant on dense Phi; tested separately if at all).
- Cross-task generalization (two QA datasets, same task family).
- Production deployability (detection, not intervention).

### What this method *does* claim if it lands

- Burden satisfied: n=500 test per dataset (×2), Bonferroni-corrected paired tests, OOD zero-shot transfer with frozen thresholds.
- A single-architecture case study at the right statistical scale, with code released, traces released, and a parameter-free training-free baseline that competes against DRAGIN / ETC / SELFDOUBT-T1 on the metric these methods were designed against (oracle-retrieve lift, not Δfrac).
- A defensible "case study + future work on portability" framing for either EMNLP-short or a workshop venue, depending on outcome.

### Why this is the right starting point (not "do everything on three archs")

- One compute session decides whether the method has any legs. If it works on Phi, Qwen and Gemma are clean extensions at known compute cost (~12 h each). If it fails on Phi, the whole research direction terminates before consuming the rest of the week.
- Phi was treated as the "honest negative" on v1's Δfrac metric. Re-testing it on the metric it was always supposed to be evaluated on (lift-conditional, not Δfrac) is the cleanest single-experiment falsifier of the entire research line.
- Burden-meeting per the audit; falsifiable in either direction; the result is informative regardless of sign.

---

## Experiment 1 — Phi-4-mini on HotpotQA-500 + 2WikiMHQA-500 zero-shot OOD

### Setup

- **Model:** Phi-4-mini-instruct (`models/phi-4-mini-instruct`), dense, 32 layers, mid-stack layer.
- **Calibration cohort:** all 200 v1 HotpotQA rows (`data/hotpotqa_distractor_200.jsonl`,
  every split tag pooled). Re-decode if `runs/phase4_qa_phi-4-mini-instruct_mid_stack_dragin.jsonl`
  is missing.
- **Test slices (fresh, disjoint from calibration):**
  - In-domain: `data/hotpotqa_distractor_500_v2test.jsonl` — built with seed 20260601,
    `--exclude-ids-from data/hotpotqa_distractor_200.jsonl`, `--all-test`.
  - OOD: `data/2wikimultihop_distractor_500_v2test.jsonl` — built with seed 20260601;
    default `split="ood_test"` tag.
- **Calibration target:** trace-level iso-fire-rate 0.30 (v1 #25 lesson — step-level
  iso-0.30 collapses gate selectivity). Per-gate threshold picked from sorted per-step
  score quantiles to land closest to a trace-level rate of 0.30 on the calibration cohort.
  `selfdoubt_t1` uses the native Pandey rule (HVR > 0; no calibration knob).
- **Latent gate mode:** chosen by 5-fold CV over the calibration cohort, picking the
  mode (drift / variance / or) with the largest mean per-fold Δfrac (tie-break: largest
  minimum across folds).
- **Frozen knobs at test time:** thresholds + latent_mode + random_p = 0.30 — applied
  unchanged to both HotpotQA-500 and 2WikiMHQA-500 (zero-shot OOD per SEP /
  ReaLM-Retrieve conventions).
- **Oracle retrieve:** `04l_oracle_retrieve_decode.py` re-decodes with paragraphs filtered
  to supporting-facts-only.
- **Analysis:** `04o_exp_analyzer.py` reports, for each gate (latent, ETC, SELFDOUBT-T1,
  DRAGIN, random):
  - trace-level fire rate, recall on lift_em=+1, precision = P(lift=+1 | fired),
    FPR, harm rate (P(lift=−1 | fired)).
  - paired-bootstrap (B=10k) margin CIs for `latent − baseline` on three statistics:
    Δlift_em (fired − not fired), Δlift_f1, and precision.
  - 2-sided p, Bonferroni @ α=0.05/4=0.0125 over the 4 baselines, and uncorrected p.
  - McNemar test on the binary lift_em=+1 label (latent vs each baseline,
    continuity-corrected, exact p approx).
  - stratifications by split, by question type, by closed-book F1 band, by n_steps band.
  - per-trace dump + raw per-resample bootstrap arrays (with `--dump-raw`).
- **Orchestration:** `scripts/run_exp1_phi.sh` — single-command run, heavy logging to
  `runs/v2_exp1_phi-4-mini-instruct/logs/`. Phases A (slice build) → B (smoke decode,
  hard-fails if <3/5 traces usable) → C (calibration) → D (test decodes) → E (oracle
  decodes) → F (analysis) → G (wrap).
- **Wall-clock estimate:** ~90–120 min on a single H100 (decode dominates).

### Results

TBD — pending compute.

### Notes

- 2Wiki rows are tagged `split="ood_test"` by the existing 2Wiki builder; `04o_exp_analyzer`
  groups by `split` so this surfaces as its own stratum without any rename.
- Phi has no `enable_thinking` knob; OOD is purely a prompt-distribution shift, not a
  decoding-protocol shift. Avoids the v1 #20 Qwen / 2Wiki under-generation trap.
- If the in-domain run's lift-positive class is small (helpful count < 30), keep going
  — under-power is a result, not a failure mode. Just report the CI width and McNemar
  count instead of arguing about significance.
- Carry-over from v1 #26: no operational success/failure thresholds were pre-committed.
  Results read on their merits.
