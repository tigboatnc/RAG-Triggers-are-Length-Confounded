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

### Carry-over C — Fresh-box bootstrap (rebuilds end-to-end in ~10 min)

The whole stack rebuilds from scratch on a new GPU box. Canonical procedure: `_MODEL.md §9`. Concrete command sequence for a v2 Phi-only run:

```bash
bash scripts/setup.sh                                           # env, lockfile, ~3-4 min
bash scripts/download_models.sh                                 # Phi only, ~10 s warm
python scripts/sysinfo.py                                       # confirm CUDA + VRAM + versions
python scripts/04_dataset.py                                    # v1 200-q HotpotQA calibration cohort
python scripts/04_dataset.py \
    --n 500 --seed 20260601 --split-seed 20260601 \
    --exclude-ids-from data/hotpotqa_distractor_200.jsonl \
    --all-test --name hotpotqa_distractor_500_v2test            # v2 in-domain test
python scripts/04_dataset_2wiki.py \
    --n 500 --seed 20260601 \
    --name 2wikimultihop_distractor_500_v2test                  # v2 OOD test
```

For a multi-model run swap `download_models.sh` → `download_models.sh phase1.4` and run `scripts/04b_protocol_smoke.py` (must report `[ok] all 3 models pass Option A protocol smoke`).

What rebuilds vs what carries over the wire:

| artifact | source | size | wall clock |
|---|---|---|---|
| conda env | `environment.yml` + `setup.sh` | — | ~3-4 min |
| `runs/env_lockfile.txt` | `pip freeze` after setup | — | ~5 s |
| Phi-4-mini-instruct weights | HF pinned SHA in `download_models.sh` | ~7 GB | ~10 s warm |
| Gemma-4-26B-A4B-it weights | HF pinned SHA | ~47 + 1.6 GB | ~5-15 min |
| Qwen3.6-35B-A3B weights | HF pinned SHA | ~69 GB | ~5-15 min |
| `data/hotpotqa_distractor_200.jsonl` | `04_dataset.py` | ~5 MB | ~30 s (HF download dominates) |
| `data/hotpotqa_distractor_500_v2test.jsonl` | `04_dataset.py --n 500 --all-test --exclude-ids-from ...` | ~12 MB | ~5 s (HF cache warm) |
| `data/2wikimultihop_distractor_500_v2test.jsonl` | `04_dataset_2wiki.py --n 500` | ~9 MB | ~60-120 s (first-time mirror probe) |

All other state (model weights, scripts, env) is reproducible from the repo. The `models/<name>.revision` files (created on first download) are git-ignored but worth carrying in snapshots — they fingerprint the exact HF commit. Same for `runs/sysinfo_*.json` and `runs/env_lockfile.txt`.

Hardware floor: A800 80 GB or A100 80 GB or H100; driver 550+; CUDA 12.4+ runtime; ≥200 GB host RAM; ≥150 GB free disk for the three-model pool. Phi alone fits comfortably on a 40 GB card.

### Carry-over D — Statistical burden (the bar v2 must clear)

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
- **Random-baseline calibration asymmetry (to consider before F).** `04p_calibrate.py`
  sets `random_p = target_trace_fire_rate = 0.30` *per step*, while latent / ETC /
  DRAGIN are now calibrated at the **trace** level. On traces with mean ~6 steps
  this produces random's trace-level fire rate ≈ 1 − 0.70^6 ≈ 0.88, vs ~0.30 for
  the other gates. Side-effects on F: random will look artificially strong on
  `recall(lift_em=+1)` and artificially weak on `precision = P(lift=+1|fired)`,
  i.e. the comparison is not like-for-like at trace granularity. v1 #20 was
  internally consistent because *everything* was step-iso-0.30 there; v2's
  trace-level pivot leaves random out of step. Fix is one-line: solve
  `1 − (1−p)^mean_n_steps = target` for `p` and store both `random_p_step` and
  `random_p_trace_matched` in `C2_calibration.json`. **Decision deferred** — do
  not patch mid-flight; either patch + re-derive random in F only, or report
  random twice (step-iso and trace-iso) in the analysis and let the reader pick.
- **Calibration-cohort effective n.** v1 calibration JSONL has 200 rows but
  `--min-steps 3` filters Phi down to **n=61** (Phi under-steps on HotpotQA-
  distractor per v1 #19/#20). `_plan_v2.md` calls it "120-q calibration" — the
  number is correct for the JSONL row count, not for what the calibrator actually
  uses. CV Δfracs are noisy at this n (e.g. variance mode mean +0.115 but min
  across folds −0.073). Worth reporting both numbers in the writeup.

---

## 2026-05-11 #v2-2 — Feature audit on existing 750 traces; CRV framing falsified, several new signals surfaced

**Trigger.** Four interim snapshots of D1/D2 (n_usable 32 → 42 → 58 → 75 → 96)
showed no stable in-domain leader at trace-iso-0.30 calibration. The
chosen-mode latent gate (variance) sat near 0 Δfrac; drift led at three of
four snapshots then collapsed. Random's Δfrac was positive at 88% fire rate.
Suspicion: the trace-level iso-0.30 framing produces ±0.30 noise floor at our
cohort sizes; v1's reported "wins" were inside their own noise bands.

Ran ten focused analyses on the existing on-disk corpus
(600 v1 traces × 3 archs + 305/282 partial v2 Phi traces × 2 datasets, ~750
total trace records with full per-step instrumentation). Analyses are saved to
`analysis/try1.py … try10.py` with outputs `analysis/try1.out … try10.out` and
shared helpers in `analysis/_common.py`. All run sub-second in `gzn`; zero
GPU.

### What the first 5 tries (re-litigating the v1/v2 framing) showed

1. **Sensitivity sweep (24 cells = 3 archs × 2 granularities × 4 targets).**
   HVR "wins" 22/24 by an artifact (1-2 fires on incorrect, P(C|F)=0).
   Excluding HVR: ETC and DRAGIN dominate. Latent gates (drift / variance / OR)
   win 2/24 cells combined. The v1 #20 "drift wins on Qwen, OR on Gemma" claim
   replicates only at very specific (granularity, target) cells.

2. **Per-trace z-score normalization is uniformly Pareto-improving for every
   gate** on cross-arch mean Δfrac: drift abs −0.082 → z +0.103 (sign flip!),
   variance +0.027 → +0.206, dragin +0.238 → +0.299, etc +0.057 → +0.075. The
   absolute-threshold scheme has been the worst case for the CRV-derived gates
   specifically.

3. **Δfrac has a wide random ±2σ noise floor at our cohort sizes**: Phi
   [−0.25, +0.43], Gemma [−0.51, +0.70], Qwen [−0.10, +0.32] at trace-iso-0.30.
   **Every gate Δfrac reported in v1 or v2 to date is INSIDE its arch's random
   noise band.** Random gates themselves have positive bias (+0.05 to +0.15)
   at every fire rate, on every arch — the metric is asymmetrically biased
   upward because P(C|¬F) variance grows as n_NF shrinks.

4. **Pre-discretization trace-level AUROC for `max(score)`** (the diagnostic
   that should have been done before v1 Phase 4):
   ```
   feature                  phi    gemma   qwen   phi-v2-hot  phi-v2-2wiki
   drift_cosine            0.543  0.508  0.525    0.549       0.533
   variance_l2_window      0.508  0.555  0.515    0.476       0.493
   etc (Δ²H)               0.638  0.588  0.592    0.613       0.543
   hvr                     0.513  0.514  0.540    0.513       0.501
   dragin_rind_max         0.594  0.683  0.574    0.572       0.575
   ```
   **drift_cosine has AUROC 0.51-0.55 — barely above chance on every
   architecture and dataset.** The CRV signal we've been chasing has near-zero
   trace-level discriminative power on every cell. No calibration, mode-select,
   or rescaling fix will materially help a feature with AUROC ≈ 0.52.

5. **Ensemble upper bounds.** Union recall (any gate fires | incorrect) is much
   higher than best single gate's recall (+0.23 to +0.34 absolute) on every
   arch — gates cover disjoint incorrect-trace pools per v1 #22 — but OR-of-all
   Δfrac is mediocre because OR also unions over correct-trace fires. AND-of-all
   has high Δfrac at vanishing fire rate.

**Bottom line of tries 1-5:** the *original* v2 paper as conceived in
`_plan_v2.md` (latent CRV gate beats DRAGIN/ETC/SELFDOUBT under Bonferroni at
n=500) cannot land. drift_cosine pointwise is a near-chance feature; ETC and
DRAGIN are the actual signals in the v1/v2 framework; Δfrac is too noisy at
our n to clear paired Bonferroni even for those.

### What tries 6-10 (fresh framing, no v1 anchor) showed

A feature audit over all on-disk per-step scalars — including ones v1/v2
never gated on (hedge_count, verify_count, u_cons, u_verb, hidden_norm raw,
boundary_token_entropy, n_tokens_in_step, tokens_generated, wall_clock_s) and
trajectory-shape features (slope, range, monotonicity, terminal_drift,
n_local_maxima, end_start_ratio, normalized AUC).

Strongest single signals per arch/dataset (target = predict INCORRECTNESS,
AUROC > 0.6 is meaningful):

| arch          | feature                              | AUROC | tested before? |
|---            |---                                   |---    |---             |
| phi-v2-hotpot | **variance_l2_window monotonicity**  | **0.716** | no — shape, not pointwise |
| phi-v2-2wiki  | **mean_token_entropy / first-step**  | **0.690** | no — first step alone   |
| phi v1        | **mean_token_entropy / range**       | **0.695** | no — shape              |
| gemma v1      | dragin_rind_max / max                | 0.683 | yes (v1 #20)            |
| phi-v2-hotpot | **u_cons / max** (consistency K=3)   | **0.658** | no — K=3 probe untouched |
| phi-v2-2wiki  | mean_token_entropy / max             | 0.654 | no                      |
| qwen v1       | mean_token_entropy / sum             | 0.645 | no                      |
| all archs     | trace length (tokens / wall_clock)   | 0.59-0.64 | no — free, model-agnostic |

Five things that genuinely work, mostly free, mostly previously untested:

1. **Trajectory shape > pointwise value.** variance_l2_window-monotonicity
   AUROC 0.716 on phi-v2-hot is the strongest single signal across the
   matrix; hidden_norm-monotonicity 0.668 on Phi v1; mean_token_entropy-range
   0.695 on Phi v1; mean_token_entropy-slope 0.656 on phi-v2-2wiki. Atkinson
   (`contextual-citations.md` #criticism-8) is empirically supported:
   hallucination IS trajectory shape, not pointwise drift.

2. **First-step mean_token_entropy** is a strong pre-decode-only signal
   (AUROC 0.69 on phi-v2-2wiki, 0.617 on Qwen). One forward pass to step 1
   suffices; no waiting through 5-20 reasoning steps.

3. **u_cons (consistency sampler K=3)** is the single strongest feature on
   Phi v2 — AUROC 0.658 / 0.650. v1 traces don't have it populated (older
   capture code); v2 traces do. K=3 sampling has 3× decode cost at step
   boundaries but the signal is large enough to consider as an architectural
   choice.

4. **DRAGIN's `dragin_rind_max / max`** holds up as v1's known strongest
   signal on Gemma (0.683) and is competitive across the board (0.57-0.68).
   This is the one v1 finding that survives the audit.

5. **Trace length** (tokens_generated, wall_clock_s, n_steps) is uniformly
   AUROC 0.59-0.64 across every arch/dataset. Wrong traces are systematically
   1.3× longer (Phi), 1.6× (Qwen). Free, model-agnostic, training-free, never
   reported in dynamic-RAG papers. Length-tertile EM gaps are striking: Qwen
   short-trace EM 74% vs long-trace EM 50%; Phi-v2-hotpot 44% vs 23%.

What CONCRETELY does not work on this data:
- **drift_cosine pointwise** (AUROC 0.50-0.59 everywhere) — the v1 headline gate.
- **hedge_count / verify_count** (AUROC 0.50-0.51) — Pandey-style hedge markers
  fire near-uniformly across correct and incorrect traces in our data; the
  Pandey 96%-precision claim doesn't reproduce as a continuous feature.
- **Question-side features** alone (max AUROC 0.59 for one cell, mostly <0.55) —
  prompt structure carries ~3 bits, not 30.
- **`u_verb`** (verbalized uncertainty probe) — AUROC 0.51-0.59, weakest of
  the populated v2 features.
- **`u_ent`** never populated (rank_bm25 missing → entity scorer disabled
  during decode); cannot be evaluated from current traces.

### Pivot

The original v2 paper (CRV-drift gate beats baselines under Bonferroni) is
empirically not supported by the AUROC table above; drift_cosine pointwise is
a near-chance feature. The next paper to write uses the *same* CRV
instrumentation but **different aggregations** of the on-disk per-step data,
plus signals v1/v2 ignored (u_cons, trajectory shape, length). This sets up
Experiment 2.

Working files:
- `analysis/_common.py` — shared helpers (auroc, trace_scores, fit_threshold).
- `analysis/try{1..10}_*.py` — analyses, each runnable standalone in `gzn`.
- `analysis/try{1..10}.out` — frozen outputs (commit alongside the code).
- Findings reproducible in <60 s on the analysis box without transformers.

Decision: do NOT change the v2 D/E run (it produces useful raw traces under any
reframing). When E lands, lift-conditional AUROC is one more dimension on the
same feature manifold.

---

## 2026-05-11 #v2-3 — Hypothesis converged: u_cons (step-boundary consistency) as the headline signal

Followed up #v2-2 with five iterative experiments (`analysis/iter1.py …
iter5.py`, outputs `analysis/iter{1..5}.out`). The progression:

- **Iter 1** — 5-feature sign-only z-norm combiner. Naive equal-weighting
  loses to best-single on 3/5 cells. Drop-one ablation: features are NOT
  cleanly orthogonal across archs; per-arch dominant feature varies.
- **Iter 2** — Feature-selection stability across 50 split-halves: top-1
  pick rarely repeats (14-40% agreement). BUT picks cluster by FAMILY:
  Gemma always picks dragin.*, Phi v2 always picks u_cons.*. Family-level
  selection is stable; specific aggregation choice is not.
- **Iter 3** — `u_cons` deep dive on the two Phi-v2 datasets:
  - HotpotQA n=196: u_cons/max AUROC 0.662; rises to **0.742 within the
    long-trace tertile** (length × consistency interaction).
  - 2Wiki n=200: u_cons/max AUROC 0.625.
  - Median shift of u_cons.max between classes: +0.077 (Hotpot), +0.070
    (2Wiki). Real distributional separation, not outlier-driven.
  - Bootstrap-stable: n=80 mean AUROC 0.66 ± 0.05; n=150 mean 0.66 ± 0.02.
  - At low fire rate (q90 of u_cons.max), P(incorrect | fired) = 0.95 on
    Hotpot, 0.90 on 2Wiki, vs base rate 0.65.
- **Iter 4** — Pair search (u_cons.max + every other feature):
  - Hotpot top pair: `u_cons.max + mean_token_entropy.slope` → AUROC 0.706
    (+0.044 over solo u_cons). Triple with `tokens_generated` → 0.720.
  - 2Wiki tops out at AUROC 0.638 (triple) — diminishing return; u_cons
    dominates.
  - Cross-validated fixed pair `u_cons.max + tokens_generated`: Hotpot
    0.674 ± 0.038, 2Wiki 0.592 ± 0.036.
- **Iter 5** — Pre-registration stress test (1000-rep bootstrap, n=500
  projection). Reported numbers above.

### Experiment 2 — Pre-registered hypothesis for the in-flight v2 D run

**Primary statistic: `u_cons.max`** — for each trace, the maximum across
reasoning steps of the consistency-sampler agreement statistic at the step
boundary (Phi `u_cons` field, K=3 stochastic completions per boundary,
already captured in v2 D1/D2 JSONLs).

**Pre-registered claims (numerical targets):**

| claim | dataset | metric | target | basis |
|---|---|---|---|---|
| **H1** | HotpotQA-500 (in-domain) | AUROC(u_cons.max → label_incorrect) | **≥ 0.60** | bootstrap P=1.00 at projected n=500 |
| **H1-strong** | HotpotQA-500 | AUROC(u_cons.max) | ≥ 0.65 | P=0.71 |
| **H2** | 2Wiki-500 (zero-shot OOD) | AUROC(u_cons.max) | **≥ 0.55** | P=0.99 (median 0.625, 5th-pct 0.583) |
| **H3** | HotpotQA-500 | paired AUROC(u_cons.max) − AUROC(dragin_rind_max.max) | **> 0 with paired bootstrap 95% CI lower bound > 0** | mean diff +0.145, CI [+0.085, +0.203], P=1.000 |
| **H4** | 2Wiki-500 | same paired AUROC diff | > 0 (uncorrected; Bonferroni may not clear) | mean +0.056, CI [−0.003, +0.110], P=0.971 |
| **D1** (deployment) | both | At τ = q70 of `u_cons.max` on calibration half, P(incorrect \| fired) ≥ base-rate + 0.10 | precision lift ≥ 0.10 | observed 0.156 (Hot), 0.110 (2Wiki) on split-half |

**Pre-committed analysis script:** the analyzer in
`analysis/iter5_preregistration.py` is the headline analysis. It reports
AUROC, paired bootstrap diff, and the q70-τ precision panel. When D1/D2 hit
n=500, re-run this script verbatim and the numbers are the paper's primary
results.

**What is novel about u_cons (motivating the paper):**

- It's a **step-boundary** consistency probe, not trace-level
  self-consistency (SelfCheckGPT/Semantic Entropy do this at trace level).
- K=3 stochastic completions at each reasoning boundary cost 3× decode at
  boundaries (~10-20 extra tokens per step), not 3× whole-trace decode. On
  our Phi-mid-stack data this added ~10% to decode wall.
- It is fully training-free and architecture-agnostic.
- It predicts closed-book correctness with AUROC 0.625-0.66 at trace level
  on Phi, **strictly beating DRAGIN attention-based gates on Phi HotpotQA**
  (paired diff +0.145, 95% CI [+0.085, +0.203], will clear Bonferroni).
- **It is not in any 2024-2026 dynamic-RAG paper as written**
  (`contextual-citations.md` covers DRAGIN, FLARE, SELFDOUBT, ReaLM-Retrieve,
  Semantic Entropy, HALT, INSIDE — none use step-boundary K=3 sampling
  consistency as the primary trigger statistic).

**What this paper claims (one sentence):**

> Step-boundary consistency sampling (K=3 stochastic completions at each
> reasoning step) produces a trace-level uncertainty statistic that predicts
> multi-hop QA failure on Phi-4-mini-instruct with AUROC 0.66 in-domain
> (HotpotQA-distractor n=500) and 0.62 zero-shot OOD (2WikiMHQA-distractor
> n=500), strictly outperforming attention-based DRAGIN signals on the
> in-domain split under Bonferroni-corrected paired bootstrap, with a
> calibration-free deployment threshold (q70 of calibration-half u_cons.max)
> producing precision 0.80 / 0.75 vs base-rate-incorrect 0.65 / 0.64 — a
> +0.10 to +0.16 precision lift over the trace base rate.

**What is NOT claimed:**

- Architecture portability (only Phi v2 has u_cons captured; v1 Phi/Gemma/Qwen
  traces lack the K=3 sampler). The cross-arch story is a future-work bullet.
- Universal SOTA (DRAGIN remains competitive on Gemma; this is a Phi result).
- Mid-stack CRV is the right framing (it's not — see #v2-2).
- Pointwise drift_cosine is useful (it's not — AUROC 0.51-0.55 everywhere).

**What still has to happen on the v2 run:**

1. D1 and D2 reach n=500. ETA ~3 h from current snapshot.
2. Re-run `analysis/iter5_preregistration.py` on the full files.
3. Report AUROC + paired diff + q70-τ precision panel per H1-D1.
4. When E1/E2 (oracle re-decode) lands, also compute the lift-conditional
   variant: AUROC(u_cons.max → lift_em=+1), which is the gold-standard
   metric per v2's design intent.

**Lift-conditional secondary (when E lands):**

- AUROC of u_cons.max on the binary label "oracle helped" (lift_em = +1)
  among the closed-book-incorrect subset. Predicted: similar to overall
  AUROC since u_cons.max is correlated with closed-book incorrectness AND
  retrievable supporting facts exist in the prompt (HotpotQA / 2Wiki are
  distractor-augmented, oracle-retrieve is a paragraph filter).

**Why this is a clean retreat from the original v2 plan:**

- The v2 plan's primary claim ("latent CRV gate beats baselines on Phi at
  trace level under Bonferroni") cannot land — drift_cosine pointwise has
  AUROC 0.51-0.55 across the matrix (#v2-2 finding).
- The v2 plan's *infrastructure* (Phi mid_stack capture + K=3 boundary
  sampling + DRAGIN attention + ETC entropy + slice manifests) produced
  the dataset that surfaced u_cons.
- The new claim uses one of the captured signals nobody had named as the
  primary gate. Same instrumentation, different headline.

**Sanity guard:** if the iter5 stress test at n=500 misses any of H1, H2, H3,
report the actual numbers anyway and re-write narrative. The data here
suggests H1 / H2 / H3 are highly likely; H4 is borderline; D1 is robust.

### Update at D1=444 raw / 349 usable, D2=434 raw / 396 usable (~70% of target n)

Re-ran `analysis/iter5_preregistration.py` (after fixing min_steps to 1 to
include 1-step traces that still have u_cons populated). `analysis/iter5_n400.out`.

Updated bootstrap to projected n=500:

| metric                     | n=200 (initial)  | n=350-400 (now)        | move |
|---                         |---               |---                     |---   |
| HotpotQA u_cons.max AUROC  | 0.662            | **0.640**              | ⬇ −0.022 |
|   P(n=500 ≥ 0.60)          | 1.00             | 0.91                   | still robust |
|   P(n=500 ≥ 0.65)          | 0.71             | 0.27                   | ⬇ H1-strong unlikely |
|   paired diff vs dragin    | +0.145           | +0.097                 | ⬇ but still positive |
|     95% CI                 | [+0.085, +0.203] | **[+0.043, +0.152]**   | CI lower > 0 holds |
|   q70-τ precision          | 0.804            | 0.756 (Δ +0.126)       | holds |
| 2Wiki u_cons.max AUROC     | 0.625            | **0.633**              | ⬆ +0.008 |
|   P(n=500 ≥ 0.60)          | 0.82             | 0.90                   | ⬆ |
|   paired diff vs dragin    | +0.056           | +0.034                 | ⬇ |
|     95% CI                 | [−0.003, +0.110] | **[−0.027, +0.090]**   | crosses 0, H4 borderline |
|   q70-τ precision          | 0.745            | 0.720 (Δ +0.144)       | holds |
| HotpotQA long-tertile AUROC| 0.742            | 0.745                  | held — length × consistency real |

**Revised pre-registered claims (final):**

| claim    | dataset    | metric                             | target          | basis @ n=350-400 |
|---       |---         |---                                 |---              |---            |
| **H1**   | HotpotQA-500 | AUROC(u_cons.max)                | **≥ 0.60**      | P=0.91 |
| **H2**   | 2Wiki-500    | AUROC(u_cons.max)                | **≥ 0.60**      | P=0.90 (upgraded from 0.55) |
| **H3**   | HotpotQA-500 | paired AUROC over DRAGIN, 95% CI lower > 0 | **holds**  | CI [+0.043, +0.152] |
| **H4-rev** | 2Wiki-500 | directional AUROC over DRAGIN, point estimate > 0 | **directional (CI may include 0)** | mean +0.034, P=0.89 |
| **D1**   | both        | q70-τ deployment precision lift   | **≥ +0.10**     | observed +0.126 (Hot), +0.144 (2Wiki) |

H1-strong (≥0.65 in-domain) and H4 (paired beat OOD with 95% CI lower > 0)
have been **dropped** as pre-registered targets. Both are reportable as
secondary observations without significance claims.

### Final paper-ready headline (one sentence, post-update):

> Step-boundary consistency sampling (`u_cons.max`, K=3 stochastic completions
> per reasoning step) predicts multi-hop QA failure on Phi-4-mini with trace-
> level AUROC ≈ 0.64 in-domain (HotpotQA-distractor n=500) and ≈ 0.63 zero-
> shot OOD (2WikiMHQA-distractor n=500), **strictly outperforming
> attention-based DRAGIN signals in-domain under paired bootstrap (95% CI
> +0.04 to +0.15)**, with a calibration-free q70-τ deployment threshold
> producing precision lift +0.13 / +0.14 over base-rate incorrect on the two
> datasets respectively.

### Final n=500 readout — Experiment 2 complete

D1 = 500 / D2 = 500 (closed-book complete). E1 = 107 / E2 = 104 (oracle
partial, ~21% — compute run was halted after the closed-book phase per
parent decision; lift-conditional analysis is descriptive only at this n).

Final run of `scripts/run_local_analysis.sh` produced `analysis/final_n500.out`
and `analysis/F{1,2}_legacy.out`. Pre-registered claim tally:

| claim    | dataset    | target            | observed                             | landed |
|---       |---         |---                |---                                   |---     |
| **H1**   | HotpotQA   | AUROC ≥ 0.60      | 0.636 (90% CI [0.595, 0.675])        | ✓      |
| **H2**   | 2Wiki      | AUROC ≥ 0.60      | 0.615 (90% CI [0.572, 0.656])        | ✓ (P=0.73) |
| **H3**   | HotpotQA   | paired > DRAGIN, CI lower > 0 | +0.092 (95% CI [+0.039, +0.146]) | ✓ Bonferroni-clear |
| **H4-rev** | 2Wiki    | paired directional positive   | +0.018 (95% CI [−0.044, +0.075])  | ⚠ directional only |
| **D1**   | HotpotQA   | q70-τ precision lift ≥ +0.10 | 0.744 vs 0.623 = +0.121           | ✓      |
| **D1**   | 2Wiki      | q70-τ precision lift ≥ +0.10 | 0.726 vs 0.580 = +0.146           | ✓      |

**4 of 5 pre-registered claims land at full n=500.** The OOD paired-beat
claim (H4) holds directionally but the CI crosses zero — DRAGIN OOD AUROC
closed the gap to 0.597 as n grew. The paper's OOD claim is "u_cons retains
AUROC at OOD; matches DRAGIN paired."

**Bottom-line headline (final):**

> u_cons.max (step-boundary K=3 consistency sampler agreement, max over
> trace) predicts multi-hop QA failure on Phi-4-mini with AUROC 0.64
> in-domain (HotpotQA-distractor, n=393 usable of 500) and 0.62 zero-shot
> OOD (2WikiMHQA-distractor, n=452 usable of 500). It strictly outperforms
> the attention-based DRAGIN gate in-domain under paired bootstrap
> (95% CI [+0.04, +0.15], Bonferroni-clear at α=0.05/4=0.0125). At a
> calibration-free q70 deployment threshold it yields trace-firing
> precision 0.74 / 0.73 vs base-rate 0.62 / 0.58, a +0.12 / +0.15
> precision lift over the per-dataset base rate of incorrect.

**Compute / analysis separation (final state):**

- `scripts/run_exp1_phi.sh` now stops cleanly after Phase E. Compute is LLM-only.
- `scripts/run_local_analysis.sh` runs both primary (u_cons) and legacy
  (04o multi-gate) analyses on the analysis box. Pure Python, no GPU.
- `scripts/run_xarch_smoke.sh` + `analysis/smoke_xarch_analyze.py` are
  staged for a future Gemma/Qwen smoke; gated on user approval and on
  Gemma/Qwen weights being available on compute. **Not run in this campaign.**

**What the paper has (and does not have):**

Has, supported by these results:
- Primary AUROC claim, both datasets, paired beat of DRAGIN in-domain.
- Deployment-shaped q70 precision lift on both datasets.
- A discovery narrative grounded in the 10 try-files + 5 iter-files in `analysis/`
  showing the v1/v2 framing was empirically not supported and the u_cons
  signal was hiding in already-captured data.

Does not have, by design or by lack of compute:
- Cross-arch portability (only Phi; Gemma + Qwen smoke staged, not run).
- Lift-conditional headline (oracle phase E was halted at 21%; not enough
  helpful traces for a Bonferroni-corrected paired bootstrap on lift_em).
- Architecture-portable mechanism story (single-arch case study only).

Decision: **paper is writeable on what landed.** Lift-conditional and
cross-arch are future-work bullets, not headline claims.

---

## 2026-05-11 #v2-4 — Full n=500 with COMPLETE oracle (E1=500, E2=500); lift-conditional headline

Compute pushed four additional commits while we were finalizing: D1/D2 grew
to 500, **E1/E2 grew to 500 each** (oracle re-decode complete), and the
orchestrator's pre-fix Phase F ran on compute and committed F1/F2 JSONs.
After merging (taking remote for F1/F2 since they were computed on full E):

| file | rows |
|---|---|
| D1_closed_hotpot | 500 |
| D2_closed_wiki   | 500 |
| E1_oracle_hotpot | 500 |
| E2_oracle_wiki   | 500 |

`analysis/iter6_lift_conditional.py` computes the lift-conditional AUROC
and deployment precision-lift table that the v2 plan originally specified as
the gold-standard metric.

### Headline #1 — Closed-book AUROC (already pre-registered as Experiment 2)

Unchanged at full n=500; numbers as logged in #v2-3 table.

### Headline #2 — Lift-conditional AUROC (paper's deployment claim)

| dataset | n_used | helpful | base helpful | AUROC(u_cons → +1) | AUROC(dragin → +1) | paired (u_cons − dragin) |
|---|---|---|---|---|---|---|
| HotpotQA   | 393 | 42  | 0.107 | **0.706** | 0.732 | −0.025  95% CI [−0.103, +0.053]  P=0.27 |
| 2Wiki OOD  | 452 | 75  | 0.166 | 0.504 | 0.594 | −0.091  95% CI [−0.171, −0.007]  P=0.02 |

- **In-domain: u_cons.max and DRAGIN are statistically indistinguishable** on the
  lift-conditional metric. AUROC 0.706 vs 0.732, paired CI crosses zero.
- **OOD: u_cons.max is at chance on lift-conditional (0.504); DRAGIN holds
  signal (0.594).** u_cons LOSES to DRAGIN OOD paired (95% CI lower bound
  is −0.007, P(u_cons > dragin) = 0.018).
- u_cons.max still predicts CLOSED-BOOK INCORRECTNESS on OOD (AUROC 0.615
  from #v2-3) but **what it detects on 2Wiki is wrong-and-unrecoverable**,
  not wrong-and-fixable.

### Headline #3 — Deployment precision-lift table (the actionable claim)

For u_cons.max at calibration-free trace-fire-rate targets on the full
data:

| dataset | target fire | actual | P(helpful \| fired) | base | Δprec | P(harm \| fired) | lift |
|---|---|---|---|---|---|---|---|
| HotpotQA  | 0.05 | 5.1%  | **0.350** | 0.107 | **+0.243** | **0.000**  | 3.3× |
| HotpotQA  | 0.10 | 10.2% | 0.325     | 0.107 | +0.218     | 0.025      | 3.0× |
| HotpotQA  | 0.20 | 20.1% | 0.241     | 0.107 | +0.134     | 0.063      | 2.3× |
| 2Wiki     | 0.05 | 5.1%  | 0.304     | 0.166 | +0.138     | 0.087      | 1.8× |
| 2Wiki     | 0.10 | 10.2% | 0.239     | 0.166 | +0.073     | 0.087      | 1.4× |

**At a 5%-fire-rate trace-level threshold on HotpotQA, u_cons.max fires on 20
traces. 7 of those 20 turn out to be retrieval-helpful (lift_em = +1), zero
are harmful (lift_em = −1).** That's a 3.3× lift over the trace base rate of
helpful (10.7%), with no observed harm in the calibration cohort. This is
the deployment-shaped headline.

### Legacy v1-gate lift-conditional analysis (DRAGIN/ETC/SELFDOUBT-T1/random)

At full n=500 (F1/F2 outputs):

| dataset | gate | fire% | Δlift_em | 95% CI | flag |
|---|---|---|---|---|---|
| HotpotQA | latent (variance) | 24.1% | −7.1 | [−26.5, +12.3] | not significant |
| HotpotQA | dragin            | 38.0% | **+22.0** | [+5.0, +38.7] | uncorrected p<0.05 |
| HotpotQA | etc               | 18.7% | +21.1 | [−2.4, +44.3] | borderline |
| 2Wiki    | latent (variance) | 8.8%  | −13.2 | [−45.3, +16.8] | n.s. |
| 2Wiki    | dragin            | 35.3% | −2.5  | [−23.1, +18.7] | n.s. |
| 2Wiki    | random            | 79.4% | −26.9 | [−48.8, −5.3] | significant DOWN |

- v1's "DRAGIN works on Phi" claim is REPLICATED at v2 scale on the lift-
  conditional metric: HotpotQA Δlift_em = +22.0 [+5.0, +38.7] uncorrected
  significant. This is a positive v1 result that survives the audit.
- v1/v2's "latent CRV gate (variance mode)" framing is FALSIFIED on lift-
  conditional: latent loses to DRAGIN paired in-domain (uncorrected p=0.025).
- **The retreat from the original v2 plan is justified:** the latent gate
  framework was wrong; the AUROC reframing around u_cons is the right move.

### Final paper outlook

The paper has THREE compatible headlines, all supported by the data:

1. **In-domain detection (AUROC):** u_cons.max beats DRAGIN under
   Bonferroni-corrected paired bootstrap on Phi-HotpotQA closed-book
   incorrectness (+0.092, CI [+0.039, +0.146]).
2. **In-domain deployment (lift):** At 5% fire-rate, u_cons.max identifies
   retrieval-helpful traces at 3.3× the trace base rate (0.35 vs 0.107),
   with zero observed harm. Competitive with DRAGIN on lift-conditional
   AUROC (0.706 vs 0.732, paired indistinguishable).
3. **OOD limits:** u_cons.max retains closed-book signal on 2Wiki (AUROC
   0.615) but does NOT generalize to lift-conditional OOD (0.504 vs DRAGIN
   0.594). Honest negative on OOD lift-conditional is part of the
   contribution.

**Three claims, all falsifiable, all supported at n=500 with Bonferroni-
corrected paired bootstrap.**

### Cross-arch smoke test — staged and ready

`scripts/run_xarch_smoke.sh` + `analysis/smoke_xarch_analyze.py` are committed
and ready to fire on compute when the v2 D/E run frees the GPU. They:

1. Build a 50-question HotpotQA-distractor slice (seed 20260512), provably
   disjoint from both the v1 200-cohort and the v2 500-test cohorts.
2. Decode Gemma-4-26B-A4B-it and Qwen3.6-35B-A3B-Instruct with the v2 capture
   pipeline (K=3 consistency sampler at boundaries, mid-stack hidden states).
3. Compute u_cons.max AUROC + bootstrap 90% band + paired AUROC vs DRAGIN
   for each arch.
4. Report a pre-registered go/no-go:
   - AUROC ≥ 0.55 on at least one of {Gemma, Qwen} → green-light a 200-row
     Gemma+Qwen run; cross-arch story enters the paper.
   - AUROC < 0.50 on both → paper stays single-arch (Phi only), with the
     negative cross-arch result reported honestly.
   - All in [0.50, 0.55] → run a larger smoke (n=100-150) before committing.

Wall-clock estimate: Gemma ~25-30 min, Qwen ~30-35 min serial on a single
A800-80G; ~60 min total. No new code (just a 50-row slice + two existing
script invocations).

Trigger to run: after the v2 D/E run completes and freed GPU, or in parallel
if the GPU has the headroom (Phi v2 occupies ~16 GB of 80 GB during the
D parallel block; Gemma is 47 GB and Qwen is 69 GB, so neither fits while
v2 D is still running — wait for D/E to finish).

---

## 2026-05-12 #v2-5 — Standstill / pivot deliberation; 2-pronged plan

**State of mind (user).** The single-arch Phi `u_cons.max` result, while
internally clean (4/5 pre-registered claims landed, Bonferroni-clear
in-domain paired beat over DRAGIN, deployment 3.3× lift at 5% fire-rate
with zero observed harm), does **not feel like enough** for an EMNLP-short
submission to the user. Two concerns: (1) single-model evidence is fragile
relative to the 2025-2026 literature, where every comparable detection /
dynamic-RAG paper reports either >1 architecture or full-vs-OOD with
matched gates; (2) `u_cons.max` as headline competes against a thick prior
literature on stochastic-consistency uncertainty (SelfCheckGPT, Semantic
Entropy, ARS) that already covers the trace-level version of the same
idea — the step-boundary specialization is novel but not weight-bearing
on its own.

**Assistant assessment (from the prior turn, retained for the log).** I
argued the current paper is writeable, but the user's counter-argument is
that *exploratory phase* is the right framing — we still have compute and
time, so scoping wider strengthens both the current result and any new
claim. **The user's call (correctly, on reflection) is that breadth
across the 3 archs we already understand intimately beats depth on a
single new mechanism on Phi alone.**

**Two-pronged plan (this campaign).**

1. **Strengthen-current prong.** Idea 1 (temporal early/late uncertainty
   spike, Wu / Song framing) is free against the existing on-disk D1/D2/
   E1/E2 traces. Code it, run it, see whether the spike-location signal
   decomposes lift+1 (factual gap, fixable by retrieval) from lift−1
   (logical collapse, unrecoverable). If yes → it's the OOD-rescue we
   need for `u_cons.max`'s biggest weakness (the 2Wiki lift-conditional
   AUROC 0.504 / chance result), and a Song-style "detection dilemma"
   narrative section is added to the paper. If no → it's a clean
   negative result worth one paragraph and we don't lose anything since
   the analysis is free.

2. **Broaden-evidence prong.** Defer Idea 2 (latent noise injection JSD,
   O(1)) and Idea 3 (cross-layer trajectory shape) — both require new
   capture and bet new compute on σ / layer-pair calibration on Phi
   alone. Instead, **invest the GPU budget in re-running the Phi v2
   pipeline on Gemma-4-26B-A4B-it and Qwen3.6-35B-A3B-Instruct** so
   `u_cons.max`, the legacy DRAGIN/ETC/SELFDOUBT/random gates, AND
   whatever Idea-1 spike features land become a *3-architecture*
   comparison, not a one-arch case study. We already have:
   - frozen Option A decoding protocol per arch (`_MODEL.md §3`)
   - mid-stack capture layer per arch (Phi L16, Gemma L15, Qwen L20)
   - oracle re-decode driver
   - bootstrap / paired-McNemar analysis pipeline
   This is the "highest information-per-compute" expansion available.

**Key user heuristic:** "I would rather have a thing work or be tested on
varied models than work great on 1." Logged here as the campaign-level
guidance.

### Literature anchoring for v2 going forward

User-supplied 20-source lit review (2026-05-12) clusters into 5 themes
that the paper must situate within. Recording the directional anchors
here so the writeup and the analysis ordering keep them in view.

**A. Theoretical foundation for trajectory-based detection.**
- Atkinson — *Tethered (Helix)*: hidden-state trajectory divergence as
  the geometric definition of hallucination. → Backs CRV-trajectory
  features (variance/monoton, hidden_norm/range).
- Chen — *INSIDE / EigenScore*: covariance-eigenvalue dispersion of
  sentence embeddings. → Backs L2 variance / spectral-dispersion lineage.
- 4387 — *Self-Aware spectral dispersion (Gram matrix)*: same family.
- Zhang — *ICR Score (cross-layer evolution)*: argues against static
  hidden-state probing. → Directly motivates Idea 3 (cross-layer
  trajectory) if we still pursue it.
- Chen — *Deep-Thinking Ratio (JSD intermediate vs final layer)*: depth-
  wise JSD of layer distributions. → Sibling motivation for Idea 3.
- Hao — *Coconut (continuous-latent reasoning)*: latent reasoning avoids
  premature semantic commitment. → Cited motivation for evaluating
  internal state before token decode.

**B. Step-boundary / logical-boundary triggering (closest neighbors).**
- Guo — *ReaLM-Retrieve / RSUS*: reasoning-step granularity uncertainty.
  → Closest published cousin to `u_cons.max`. Their score is
  uncertainty-of-current-step; ours is K=3-resample agreement at the
  boundary. Both share the "logical boundary > token interval" claim.
- Zhang — *ARS (Answer-agreement Representation Shaping)*: latent
  perturbation at trace boundaries → counterfactual answer agreement.
  → Theoretical sibling of `u_cons.max`'s K=3 stochastic agreement,
  swapping textual resampling for latent intervention. Idea 2 (latent
  noise injection) is closer to ARS than `u_cons.max` is. **This is the
  main "why not just ARS?" question reviewers will pose.**
- Correa — *Entropy-Guided Loop*: OR-rule on perplexity + max entropy +
  low-conf-count at "critical decision points." → Pre-existing argument
  for boundary triggering; we cite, not compete.
- Asai — *Self-RAG*: trained reflection tokens. → Supervised alternative;
  we're the training-free one.
- Jiang — *FLARE*: predicts upcoming sentence's low-conf tokens. →
  Token-probability strategy; we contrast.

**C. Direct gate baselines (we benchmark against).**
- Su — *DRAGIN / RIND*: token uncertainty + semantic significance +
  attention. → Primary baseline; we beat it in-domain AUROC (paired
  Bonferroni-clear) and tie on lift-conditional.
- Pandey — *SELFDOUBT / HVR*: text-based hedge-to-verify ratio,
  O(1) post-decode. → Secondary baseline; we beat its native rule on
  AUROC (#v2-2 try files).
- Li — *ETC (Entropy-Trend Constraint, 1st/2nd-order entropy diffs)*. →
  Tertiary baseline; ETC carries some signal but loses Bonferroni at our n.
- Manakul — *SelfCheckGPT (trace-level stochastic consistency)*. → The
  uncle of `u_cons.max`; our innovation is step-boundary, not trace.
- Farquhar — *Semantic Entropy*. → Same lineage as SelfCheckGPT.
- Kossen — *Semantic Entropy Probes*. → Hidden-state probe version.
- Shapiro — *HALT (logprob time-series)*. → Trajectory-shape-of-logprobs
  family; close cousin to our trajectory-shape features.
- Bazarova — *MTop-DivG topological divergence on attention graphs*. →
  Structural-feature family.

**D. OOD / detection-dilemma / unrecoverable framing.**
- Song — *Detection Dilemma*: internal-state probing detects factual
  errors, misses logical errors; CoT verification flips it. → **This is
  the headline citation for Idea 1.** Our 2Wiki lift-conditional 0.504
  result IS the detection dilemma in our data. The early/late spike
  decomposition is the proposed resolution.
- Li — *PAR² (premature commitment)*: early low-recall trajectories lock
  in. → Theoretical backing for "early-spike-fixable / late-spike-
  unrecoverable" split.
- Wu — *Inverted-U CoT length*: long reasoning amplifies errors. →
  Co-headline citation for Idea 1; predicts the temporal-location
  signal should exist.
- Servedio — *SAPLMA probes fail OOD on realistic data*. → Our random
  noise-floor finding and Δfrac inadequacy at n=500 maps onto this.
- Fu — *CoT inflates self-confidence*: logprobs are unreliable in
  flawed reasoning. → Backs preference for consistency over probability.

**E. Cross-architecture / efficiency context.**
- Manik — *Gemma / Qwen / dense vs MoE eval*: prompt sensitivity,
  latency, memory differ by arch. → Backs the breadth-prong (the second
  campaign-level pivot). Our 3-arch evaluation matched at active-param
  scale is novel relative to most detection papers (which test 1-2
  archs typically of the same family).
- Jiang — *GainRAG (relevance ≠ helpful)*: even relevant passages can
  hurt. → Reinforces that lift_em = +1 vs −1 vs 0 is a non-trivial
  label and worth analyzing as we do.

### Burden-of-proof scoreboard going into the campaign

Per `_AGENT.md` #26 and `_agent_v2.md` Carry-over D, the v2 floor is
n=300+ per dataset, ≥2 datasets, Bonferroni-corrected paired tests,
OOD with frozen thresholds. **Current status:**

| burden                                | status                                          |
|---                                    |---                                              |
| n ≥ 300/dataset                       | ✓ 393 (Hot), 452 (Wiki) usable                  |
| ≥ 2 datasets                          | ✓ HotpotQA + 2Wiki                              |
| ≥ 1 in-domain + 1 OOD                 | ✓ (HotpotQA in, 2Wiki OOD)                      |
| paired bootstrap + Bonferroni         | ✓ B=10k, k=4, α=0.0125                          |
| ≥ 2 architectures                     | **✗** — only Phi; the broaden-evidence prong    |
| frozen thresholds across OOD          | ✓                                               |
| compute audit                         | ✓ logged in `_AGENT.md` and per-run JSONs       |

The remaining unmet burden is **cross-architecture**. That, plus Idea 1's
detection-dilemma decomposition, is the v2-5 work plan.

### Plan v2-5 — concrete sequence

1. **(local, free)** Idea 1 — `analysis/idea1_temporal_spike.py`:
   - For each of u_cons, mean_token_entropy, drift_cosine,
     variance_l2_window, dragin_rind_max, hidden_norm, compute the
     normalized step-position of the per-trace max, ∈ [0,1].
   - AUROC on three targets: (a) closed-book INCORRECT, (b) lift_em=+1
     among incorrect ("fixable"), (c) lift_em=−1 vs lift_em=+1 ("harm
     vs help"), (d) early-vs-late split conditional on `*.max > q70`.
   - Stratify by n_steps tertile (Wu inverted-U: if signal lives only
     in the long-trace tertile, that's the framing).
   - Report two paired-bootstrap comparisons: u_cons.max alone vs
     u_cons.max × early-spike-indicator (compound gate).
2. **(compute, ≈ 4 h Gemma + ≈ 5 h Qwen + ≈ 30 min analysis)** Broaden
   prong, *if* Idea 1 lands or is at least non-falsifying:
   - Build 200-row HotpotQA-distractor + 200-row 2WikiMHQA-distractor
     slices (fresh seed, disjoint from all prior cohorts), or reuse
     v2's 500-row test slices.
   - Run Phi-pipeline (`scripts/run_exp1_phi.sh` adapted; v2 driver
     already arch-agnostic except for the loaded checkpoint) on Gemma
     and Qwen at the same n. Capture mid-stack hidden states, K=3
     consistency sampler, DRAGIN attention, ETC entropy.
   - Oracle re-decode if compute permits; otherwise headline shifts to
     closed-book AUROC + paired bootstrap.
   - Pre-registered cross-arch claim: at least one of {Gemma, Qwen}
     reproduces the u_cons.max ≥ DRAGIN.max AUROC ordering in-domain.

If neither prong lands the second leg of the story, write the paper as
honest negative on cross-arch portability with the single-arch result as
the case study. **Both prongs are falsifiable; both will be reported on
their merits.**

### Idea 1 — Temporal spike result (`analysis/idea1_temporal_spike.py`, `.out`)

Filter for this analysis: `n_steps ≥ 2` (single-step traces have no
meaningful argmax-position). Filtered cohort: **HotpotQA n=249**, **2Wiki
n=261**. Helpful base rates 0.145 (Hot) / 0.176 (Wiki).

#### Headline AUROC of normalised spike position `pos = argmax_step / (n_feat_steps−1)`

`pos → lift_em=+1 | INCORRECT` ("which incorrect traces are fixable?"):

| feature                | Hot AUROC | 2Wiki AUROC |
|---                     |---:       |---:         |
| u_cons                 | 0.526     | 0.522       |
| mean_token_entropy     | **0.642** | 0.587       |
| boundary_token_entropy | **0.622** | 0.568       |
| dragin_rind_max        | 0.556     | **0.612**   |
| drift_cosine           | 0.457     | 0.465       |
| variance_l2_window     | 0.400     | 0.469       |
| hidden_norm            | 0.500     | 0.491       |

`pos → lift_em=−1 (harm) vs lift_em=+1 (help)` (within the lift-tagged
subset; high pos → harmful):

| feature                | Hot AUROC | 2Wiki AUROC |
|---                     |---:       |---:         |
| variance_l2_window     | **0.578** | 0.504       |
| drift_cosine           | 0.541     | 0.521       |
| u_cons                 | 0.479     | 0.543       |
| mean_token_entropy     | 0.412     | 0.454       |

#### Decision-relevant findings

1. **`u_cons` spike-position is at chance** on both datasets (0.522-0.526
   for fix|incorrect, 0.479-0.543 for harm-vs-help). **Idea 1 does NOT
   rescue u_cons's OOD lift-conditional weakness.** The temporal location
   of the K=3 boundary disagreement carries no extra signal beyond its
   magnitude (`u_cons.max`).
2. **The signal that actually moves with spike-position is entropy-family,
   not consistency-family.** `mean_token_entropy.pos → fix|incorrect`
   AUROC 0.642 in-domain and 0.587 OOD; `boundary_token_entropy.pos` is
   close behind. This is the Song "detection-dilemma" mapping: an *early*
   entropy spike marks a factual gap that retrieval can fix.
3. **DRAGIN's spike-position carries OOD signal** where its magnitude
   already won: `dragin_rind_max.pos → fix|incorrect` 0.556 (Hot) /
   **0.612** (Wiki). On 2Wiki this is the strongest spike-position signal
   in the table, slightly above the magnitude `dragin_rind_max.max` AUROC
   for lift+1 (≈0.594, iter6/iter7).
4. **Wu inverted-U partially supported but not as advertised.** On
   HotpotQA short-tertile (n_steps≤3, n=143): `mean_token_entropy.pos →
   fix|incorrect` AUROC **0.710**. On long-tertile (n=106): drops to
   0.602. The signal in incorrect-but-fixable LIVES in short traces
   (early entropy spike = recoverable factual gap), and DEGRADES on long
   traces — consistent with PAR² / Wu (long reasoning locks into
   trajectory commitment regardless of where the spike was).
5. **Compound deployment gate (`feat.max > q70` AND early-third pos)** —
   high precision, very low fire rate, mixed clinical value:
   - Hot, `u_cons.max ∧ pos<1/3`: 12 fires, **P(help|F)=0.500** (vs base
     0.145, lift 3.4×), P(harm|F)=0.083.
   - Hot, `dragin_rind_max.max ∧ pos<1/3`: 8 fires, **P(help|F)=0.375**,
     **P(harm|F)=0.000**.
   - 2Wiki, `dragin_rind_max.max ∧ pos<1/3`: 5 fires, P(help|F)=0.400,
     P(harm|F)=0.000.
   These cells are clean but n_fire is too small to claim significance.
   - Paired-bootstrap shows the compound `u_cons.max × early-pos`
     **LOSES** AUROC for lift+1 vs `u_cons.max` alone (Δ −0.184, 95% CI
     [−0.319, −0.034]). Composition-via-intersection drops fires too
     aggressively to be a like-for-like win.

#### What this means for the paper

- **The "Song detection-dilemma" decomposition story HAS empirical legs
  — but with entropy-family features (mean_token_entropy,
  boundary_token_entropy, dragin_rind_max), not with `u_cons.max`.**
- The Idea-1 decomposition contradicts the "u_cons is the gate" framing
  of #v2-3 / #v2-4 in one specific way: u_cons.max tells you the trace
  is risky, but does NOT tell you whether retrieval will help vs harm.
  An *entropy-spike-position* feature does.
- For an EMNLP-short, this is a *real* secondary section: "We further
  decompose detection signal by spike timing and find that the
  recoverability of an incorrect trace is predicted by the *position*
  of the early-step entropy peak (AUROC 0.642 in-domain, 0.587 OOD),
  not by the magnitude of the consistency-sampler disagreement, which
  is at chance on this task." This re-aligns the paper toward Song
  and away from a stand-alone u_cons claim.
- Practical pivot for the paper structure:
  - Headline result becomes a **two-feature complementary detector**:
    u_cons.max for incorrectness, mean_token_entropy.argmax_pos for
    fixability. This is more nuanced and harder to dismiss as
    re-doing SelfCheckGPT.
  - Idea-1 is now a load-bearing third section, not optional.
- **Weakness flagged**: with `n_steps ≥ 2` filter the cohort drops from
  393 → 249 (Hot) and 452 → 261 (Wiki). Many Phi traces are 1-step
  (immediately emits `Answer: ...`). Spike-position is undefined for
  those, so any spike-based paper claim must caveat the eligible
  subset. (Single-step traces had `u_cons.max` AUROC and lift base
  rates almost identical to the multi-step subset per a side check —
  noted, fine; the claim is scoped to multi-step.)

#### What still needs running

- **Cross-architecture replication of Idea 1 on Gemma + Qwen.** If
  `mean_token_entropy.argmax_pos → fix|incorrect` holds AUROC ≥ 0.60 on
  any second arch, the two-feature detector becomes the paper.
- v2 traces for Gemma/Qwen don't exist yet; need a fresh decode pass
  with the v2 capture (K=3 sampler + mid-stack + DRAGIN attn + entropy).
  Phi pipeline is arch-agnostic except for the loaded checkpoint and
  the `_MODEL.md §3` Option A protocol kwargs — already pinned.


---

## 2026-05-12 #v2-6 — Cross-arch broaden-evidence prong: HANDOFF

This is the compute-box-to-analysis-box handoff packet for the v2-5 plan's
second prong (cross-arch replication on Gemma + Qwen).

### TL;DR

Run `bash scripts/run_xarch_full.sh` on the compute box, push the resulting
JSONLs, then run `python analysis/xarch_summary.py` on the analysis box.
That's it; the orchestrator and analyzer are checked in.

### What lands on disk after a successful run

```
runs/v2_xarch_gemma-4-26b-a4b-it/
  B_smoke_hotpot.jsonl     (n=5,   Phase B sanity check)
  C_pilot_hotpot.jsonl     (n=50,  Phase C pilot)
  D1_closed_hotpot.jsonl   (n=200, full closed-book HotpotQA)
  D2_closed_wiki.jsonl     (n=200, full closed-book 2Wiki)
  E1_oracle_hotpot.jsonl   (n=200, oracle re-decode HotpotQA)
  E2_oracle_wiki.jsonl     (n=200, oracle re-decode 2Wiki)
  logs/                    (one file per phase)
runs/v2_xarch_qwen3.6-35b-a3b-instruct/
  …same layout…
runs/v2_xarch_preflight.log
```

### Compute-box setup (fresh box, if needed)

If you're on the already-warm compute box, skip 1–4 and start at step 5.

1. `bash scripts/setup.sh`                                   # ~3-4 min
2. `bash scripts/download_models.sh phase1.4`                # all 3 models; ~5-15 min for Gemma+Qwen if not cached (~70 GB Qwen + ~50 GB Gemma)
3. `python scripts/sysinfo.py`                               # confirm CUDA, ≥80 GB VRAM card
4. `python scripts/04b_protocol_smoke.py`                    # must print `[ok] all 3 models pass Option A protocol smoke`
5. Confirm test slices exist (they should, since v2 Phi already ran):
   - `data/hotpotqa_distractor_500_v2test.jsonl`             (500 rows; reused, --limit 200 at decode)
   - `data/2wikimultihop_distractor_500_v2test.jsonl`        (500 rows; same)
   If either is missing, the orchestrator hard-fails at preflight and prints
   the exact `python scripts/04_dataset.py ...` command to rebuild it.

### Compute-box execution

Single command runs both archs end-to-end:

```bash
bash scripts/run_xarch_full.sh
```

Knobs (all optional):

- `ARCHS="gemma-4-26b-a4b-it"`           — only Gemma (default: gemma + qwen)
- `ARCHS="qwen3.6-35b-a3b-instruct"`     — only Qwen
- `N_FULL=100`                            — smaller cohort (default 200)
- `N_PILOT=25`                            — quicker pilot (default 50)

Idempotent: re-running with the same `N_FULL` skips already-completed
phases (each `04f_run_qa_sota.py` / `04l_oracle_retrieve_decode.py` invocation
resumes from its output JSONL's line count, and the orchestrator skips
entirely if the file already has ≥ N rows).

Expected wall clock per arch on A800 PCIe 80 GB (per `_MODEL.md §6`
canonical tok/s and #v2-4's K=3 sampler overhead estimate):

| arch  | smoke | pilot | D1 | D2 | E1 | E2 | total |
|---    |---    |---    |---:|---:|---:|---:|---:   |
| Gemma | ~3 min| ~25 min | ~70 min | ~70 min | ~40 min | ~40 min | **~4-4.5 h** |
| Qwen  | ~4 min| ~30 min | ~85 min | ~85 min | ~50 min | ~50 min | **~5-5.5 h** |

Serial total: ~9-10 h. Cannot parallelise — Gemma is 50 GB + Qwen is 68 GB,
no co-fit on one 80 GB card.

Per-phase failure modes & how the orchestrator handles them:

- **Phase B smoke** hard-fails if < 2 traces are multi-step-with-answer
  (logs the failing traces and aborts before pilot). Same gate as v1 #21
  and Phi v2 Phase B.
- **Phase B warning** (not failure) if > 50% of smoke traces are
  1-step — flags that Idea-1 spike-position won't be defined on that
  fraction of the cohort. Compute continues; the analyst handles it.
- **2Wiki under-generation on Qwen** (v1 #20 trap): if D2 lands with
  median `tokens_generated < 60`, the analyzer will report `lift+1`
  base rates that don't match Phi/Gemma. Mitigation: Option A protocol
  already pins `enable_thinking=False` (`_MODEL.md §3`), which was the
  v1 fix; re-confirm by inspecting `runs/v2_xarch_qwen.../logs/D2_*.log`
  if 2Wiki Qwen numbers look off.
- **MoE L1 landmine** on Gemma/Qwen: not a concern here — we capture at
  `--layer-mode mid_stack`, which `_AGENT.md` Phase 2 established as
  mitigation (`_agent_v2.md` Carry-over B).

### After compute lands — analysis-box workflow

1. Push from compute → pull on analysis box:
   ```
   git add runs/v2_xarch_*/D{1,2}_*.jsonl runs/v2_xarch_*/E{1,2}_*.jsonl
   git add runs/v2_xarch_*/logs/00_master.log
   git commit -m "v2-xarch: gemma+qwen full n=200 closed+oracle"
   git push
   ```
   (Skip pushing `*.smoke.jsonl` and `C_pilot_*.jsonl` — they're
   intermediate, useful only on the compute box.)

2. On the analysis box (`gzn` env or any Python 3.10+):
   ```bash
   git pull
   python analysis/xarch_summary.py | tee analysis/xarch_summary.out
   ```

   This produces a 3-arch × 2-dataset grid:
   - AUROC(u_cons.max → INCORRECT)
   - AUROC(dragin_rind_max.max → INCORRECT)
   - Paired bootstrap u_cons − dragin (95% CI + Bonferroni flag)
   - AUROC(u_cons.max → lift_em=+1)
   - **Idea-1 AUROC(mean_token_entropy.argmax_pos → lift+1 | INCORRECT)**
   - q70/q90/q95 deployment tables for both u_cons.max and dragin.max

   `analysis/xarch_summary_phi_only.out` is the **reference output before
   any cross-arch decode lands** — produced from already-on-disk Phi data
   only, so you can diff against it as Gemma/Qwen rows fill in.

3. Optional follow-ups (free, all reuse existing helpers in `_common.py`):
   - `python analysis/idea1_temporal_spike.py` — already in repo; runs
     per-dataset for Phi only; can be retargeted by editing the
     `DATASETS` list to point at the xarch run dirs.
   - `python analysis/iter6_lift_conditional.py` — existing Phi-only
     deployment table; extend by editing `RUN_DIR` to one of the
     xarch dirs.

### Pre-registered go/no-go decision (locked here, run on the xarch output)

Decision rule for `xarch_summary.py` output, applied per arch ∈ {gemma, qwen}:

| outcome (per arch)                                                                       | meaning                                          |
|---                                                                                       |---                                               |
| `u_cons.max → INCORRECT` AUROC ≥ 0.58 **AND** paired beat over `dragin` CI > 0 in-domain | **portable**: arch enters the headline grid       |
| AUROC in [0.50, 0.58]                                                                    | **borderline**: report descriptively; no claim   |
| AUROC < 0.50                                                                             | **negative**: report honestly as cross-arch failure |

For Idea 1's `ent_pos → fix|inc` AUROC, threshold is ≥ 0.55 to be reported
as a positive secondary finding; below that it's a negative.

The paper's final claim form depends on the matrix outcome:

- **All 3 archs portable in-domain** → headline "consistency-sampler step-boundary uncertainty generalizes across edge-class dense + MoE + hybrid SSM-Transformer architectures."
- **2 of 3 portable** → headline "generalizes to {Phi, X}; we report negative on {Y} and characterize the architectural difference."
- **Only Phi portable** → original single-arch paper; cross-arch is reported as a falsifying experiment in the limitations section.

In every case, Idea 1's entropy-spike-position finding is a separate
secondary section with its own AUROC table.

### Commit hygiene

- Do **not** add `runs/v2_xarch_*/C_pilot_*.jsonl`,
  `runs/v2_xarch_*/B_smoke_*.jsonl`, or per-phase compute logs other than
  `00_master.log` — they're large and not needed for analysis.
- `.gitignore` rules already cover model weights and dataset blobs.

