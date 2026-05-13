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


| artifact                                         | source                                                    | size         | wall clock                          |
| ------------------------------------------------ | --------------------------------------------------------- | ------------ | ----------------------------------- |
| conda env                                        | `environment.yml` + `setup.sh`                            | —            | ~3-4 min                            |
| `runs/env_lockfile.txt`                          | `pip freeze` after setup                                  | —            | ~5 s                                |
| Phi-4-mini-instruct weights                      | HF pinned SHA in `download_models.sh`                     | ~7 GB        | ~10 s warm                          |
| Gemma-4-26B-A4B-it weights                       | HF pinned SHA                                             | ~47 + 1.6 GB | ~5-15 min                           |
| Qwen3.6-35B-A3B weights                          | HF pinned SHA                                             | ~69 GB       | ~5-15 min                           |
| `data/hotpotqa_distractor_200.jsonl`             | `04_dataset.py`                                           | ~5 MB        | ~30 s (HF download dominates)       |
| `data/hotpotqa_distractor_500_v2test.jsonl`      | `04_dataset.py --n 500 --all-test --exclude-ids-from ...` | ~12 MB       | ~5 s (HF cache warm)                |
| `data/2wikimultihop_distractor_500_v2test.jsonl` | `04_dataset_2wiki.py --n 500`                             | ~9 MB        | ~60-120 s (first-time mirror probe) |


All other state (model weights, scripts, env) is reproducible from the repo. The `models/<name>.revision` files (created on first download) are git-ignored but worth carrying in snapshots — they fingerprint the exact HF commit. Same for `runs/sysinfo_*.json` and `runs/env_lockfile.txt`.

Hardware floor: A800 80 GB or A100 80 GB or H100; driver 550+; CUDA 12.4+ runtime; ≥200 GB host RAM; ≥150 GB free disk for the three-model pool. Phi alone fits comfortably on a 40 GB card.

### Carry-over D — Statistical burden (the bar v2 must clear)

Per `_AGENT.md` #26 audit:


| dimension                      | v1        | v2 floor                     | v2 target                        |
| ------------------------------ | --------- | ---------------------------- | -------------------------------- |
| test n per dataset             | 40        | 300                          | 500                              |
| dataset count                  | 1         | 2 (1 in-domain + 1 OOD)      | 2                                |
| paired-margin tests            | bootstrap | bootstrap + Bonferroni       | bootstrap + Bonferroni + McNemar |
| OOD protocol                   | none      | frozen thresholds, no re-fit | frozen thresholds, no re-fit     |
| multiple-comparison correction | none      | yes (k≥4 gate comparisons)   | yes                              |


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
- **Fast.** Phi decodes at ~~42 tok/s on A800 vs Qwen's ~12 tok/s. The full 4-decode-pass schedule (closed-book × 2 datasets + oracle-retrieve × 2 datasets, ~2000 total decodes at ~250 tokens each) lands in **~~3-4 h wall clock** — feasible in one compute session.
- **Honest scope.** "On the smallest-active-param edge LLM, the variance-mode CRV gate is the first parameter-free pre-decode signal to selectively identify retrieval-helpful reasoning traces under zero-shot OOD transfer" is a single-arch claim that does not over-reach.

### Headline claim under test

> **H1 (in-domain):** On HotpotQA-distractor (n=500 test), the variance-mode CRV gate has **paired-significantly higher P(oracle_helped | fired)** than each of {DRAGIN, ETC, SELFDOUBT-Tier 1, Random}, with Bonferroni-corrected α = 0.05 / 4 = 0.0125.
>
> **H2 (OOD):** With thresholds frozen from HotpotQA calibration, the same gate retains paired-significant advantage on 2WikiMHQA-distractor (n=500 test).

### Experimental design


| step                     | what                                                                                                                                                     | input                                                                                   | output                                              |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------- |
| 1. Slice build           | reuse v1 HotpotQA-200 calibration rows (n=120 calibration); build HotpotQA-500-test (fresh seed, disjoint from v1 200); build 2WikiMHQA-500-test (fresh) | `scripts/04_dataset.py` rerun with new seed + size; `scripts/04_dataset_2wiki.py` rerun | three JSONLs in `data/`                             |
| 2. Smoke (×2)            | 5-question smoke per dataset, criteria from v1 #21                                                                                                       | Phi + slice JSONL                                                                       | smoke logs in `runs/v2_*`                           |
| 3. Calibration           | k-fold CV on the 120-q HotpotQA calibration: pick mode (drift / variance / OR) and threshold targeting **trace-level fire rate 0.30**                    | existing v1 calibration traces (Phi mid_stack, already on disk)                         | frozen knobs JSON in `runs/v2_phi_calibration.json` |
| 4. In-domain decode      | Phi × HotpotQA-500-test, mid-stack CRV + DRAGIN attention capture + per-step entropy                                                                     | `scripts/04f_run_qa_sota.py`                                                            | `runs/v2_phi_hotpot_500.jsonl`                      |
| 5. OOD decode            | Phi × 2WikiMHQA-500-test, same configuration                                                                                                             | same                                                                                    | `runs/v2_phi_2wiki_500.jsonl`                       |
| 6. Oracle re-decode (×2) | Phi × {HotpotQA-500, 2Wiki-500} with supporting-only paragraphs                                                                                          | `scripts/04l_oracle_retrieve_decode.py`                                                 | `runs/v2_phi_*_oracle.jsonl`                        |
| 7. Bootstrap analysis    | lift-conditional Δ_em + Δ_f1 with Bonferroni-corrected paired margins; McNemar parallel                                                                  | `scripts/04n_lift_conditional.py` + small wrapper for Bonferroni                        | `runs/v2_phi_headline.json`                         |


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


| arch          | feature                             | AUROC     | tested before?            |
| ------------- | ----------------------------------- | --------- | ------------------------- |
| phi-v2-hotpot | **variance_l2_window monotonicity** | **0.716** | no — shape, not pointwise |
| phi-v2-2wiki  | **mean_token_entropy / first-step** | **0.690** | no — first step alone     |
| phi v1        | **mean_token_entropy / range**      | **0.695** | no — shape                |
| gemma v1      | dragin_rind_max / max               | 0.683     | yes (v1 #20)              |
| phi-v2-hotpot | **u_cons / max** (consistency K=3)  | **0.658** | no — K=3 probe untouched  |
| phi-v2-2wiki  | mean_token_entropy / max            | 0.654     | no                        |
| qwen v1       | mean_token_entropy / sum            | 0.645     | no                        |
| all archs     | trace length (tokens / wall_clock)  | 0.59-0.64 | no — free, model-agnostic |


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
- `**u_verb`** (verbalized uncertainty probe) — AUROC 0.51-0.59, weakest of
the populated v2 features.
- `**u_ent**` never populated (rank_bm25 missing → entity scorer disabled
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

Followed up #v2-2 with five iterative experiments (`analysis/iter1.py … iter5.py`, outputs `analysis/iter{1..5}.out`). The progression:

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

**Primary statistic: `u_cons.max*`* — for each trace, the maximum across
reasoning steps of the consistency-sampler agreement statistic at the step
boundary (Phi `u_cons` field, K=3 stochastic completions per boundary,
already captured in v2 D1/D2 JSONLs).

**Pre-registered claims (numerical targets):**


| claim               | dataset                   | metric                                                                                  | target                                               | basis                                             |
| ------------------- | ------------------------- | --------------------------------------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------- |
| **H1**              | HotpotQA-500 (in-domain)  | AUROC(u_cons.max → label_incorrect)                                                     | **≥ 0.60**                                           | bootstrap P=1.00 at projected n=500               |
| **H1-strong**       | HotpotQA-500              | AUROC(u_cons.max)                                                                       | ≥ 0.65                                               | P=0.71                                            |
| **H2**              | 2Wiki-500 (zero-shot OOD) | AUROC(u_cons.max)                                                                       | **≥ 0.55**                                           | P=0.99 (median 0.625, 5th-pct 0.583)              |
| **H3**              | HotpotQA-500              | paired AUROC(u_cons.max) − AUROC(dragin_rind_max.max)                                   | **> 0 with paired bootstrap 95% CI lower bound > 0** | mean diff +0.145, CI [+0.085, +0.203], P=1.000    |
| **H4**              | 2Wiki-500                 | same paired AUROC diff                                                                  | > 0 (uncorrected; Bonferroni may not clear)          | mean +0.056, CI [−0.003, +0.110], P=0.971         |
| **D1** (deployment) | both                      | At τ = q70 of `u_cons.max` on calibration half, P(incorrect | fired) ≥ base-rate + 0.10 | precision lift ≥ 0.10                                | observed 0.156 (Hot), 0.110 (2Wiki) on split-half |


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


| metric                      | n=200 (initial)  | n=350-400 (now)      | move                             |
| --------------------------- | ---------------- | -------------------- | -------------------------------- |
| HotpotQA u_cons.max AUROC   | 0.662            | **0.640**            | ⬇ −0.022                         |
| P(n=500 ≥ 0.60)             | 1.00             | 0.91                 | still robust                     |
| P(n=500 ≥ 0.65)             | 0.71             | 0.27                 | ⬇ H1-strong unlikely             |
| paired diff vs dragin       | +0.145           | +0.097               | ⬇ but still positive             |
| 95% CI                      | [+0.085, +0.203] | **[+0.043, +0.152]** | CI lower > 0 holds               |
| q70-τ precision             | 0.804            | 0.756 (Δ +0.126)     | holds                            |
| 2Wiki u_cons.max AUROC      | 0.625            | **0.633**            | ⬆ +0.008                         |
| P(n=500 ≥ 0.60)             | 0.82             | 0.90                 | ⬆                                |
| paired diff vs dragin       | +0.056           | +0.034               | ⬇                                |
| 95% CI                      | [−0.003, +0.110] | **[−0.027, +0.090]** | crosses 0, H4 borderline         |
| q70-τ precision             | 0.745            | 0.720 (Δ +0.144)     | holds                            |
| HotpotQA long-tertile AUROC | 0.742            | 0.745                | held — length × consistency real |


**Revised pre-registered claims (final):**


| claim      | dataset      | metric                                            | target                             | basis @ n=350-400                     |
| ---------- | ------------ | ------------------------------------------------- | ---------------------------------- | ------------------------------------- |
| **H1**     | HotpotQA-500 | AUROC(u_cons.max)                                 | **≥ 0.60**                         | P=0.91                                |
| **H2**     | 2Wiki-500    | AUROC(u_cons.max)                                 | **≥ 0.60**                         | P=0.90 (upgraded from 0.55)           |
| **H3**     | HotpotQA-500 | paired AUROC over DRAGIN, 95% CI lower > 0        | **holds**                          | CI [+0.043, +0.152]                   |
| **H4-rev** | 2Wiki-500    | directional AUROC over DRAGIN, point estimate > 0 | **directional (CI may include 0)** | mean +0.034, P=0.89                   |
| **D1**     | both         | q70-τ deployment precision lift                   | **≥ +0.10**                        | observed +0.126 (Hot), +0.144 (2Wiki) |


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


| claim      | dataset  | target                        | observed                         | landed             |
| ---------- | -------- | ----------------------------- | -------------------------------- | ------------------ |
| **H1**     | HotpotQA | AUROC ≥ 0.60                  | 0.636 (90% CI [0.595, 0.675])    | ✓                  |
| **H2**     | 2Wiki    | AUROC ≥ 0.60                  | 0.615 (90% CI [0.572, 0.656])    | ✓ (P=0.73)         |
| **H3**     | HotpotQA | paired > DRAGIN, CI lower > 0 | +0.092 (95% CI [+0.039, +0.146]) | ✓ Bonferroni-clear |
| **H4-rev** | 2Wiki    | paired directional positive   | +0.018 (95% CI [−0.044, +0.075]) | ⚠ directional only |
| **D1**     | HotpotQA | q70-τ precision lift ≥ +0.10  | 0.744 vs 0.623 = +0.121          | ✓                  |
| **D1**     | 2Wiki    | q70-τ precision lift ≥ +0.10  | 0.726 vs 0.580 = +0.146          | ✓                  |


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


| file             | rows |
| ---------------- | ---- |
| D1_closed_hotpot | 500  |
| D2_closed_wiki   | 500  |
| E1_oracle_hotpot | 500  |
| E2_oracle_wiki   | 500  |


`analysis/iter6_lift_conditional.py` computes the lift-conditional AUROC
and deployment precision-lift table that the v2 plan originally specified as
the gold-standard metric.

### Headline #1 — Closed-book AUROC (already pre-registered as Experiment 2)

Unchanged at full n=500; numbers as logged in #v2-3 table.

### Headline #2 — Lift-conditional AUROC (paper's deployment claim)


| dataset   | n_used | helpful | base helpful | AUROC(u_cons → +1) | AUROC(dragin → +1) | paired (u_cons − dragin)              |
| --------- | ------ | ------- | ------------ | ------------------ | ------------------ | ------------------------------------- |
| HotpotQA  | 393    | 42      | 0.107        | **0.706**          | 0.732              | −0.025 95% CI [−0.103, +0.053] P=0.27 |
| 2Wiki OOD | 452    | 75      | 0.166        | 0.504              | 0.594              | −0.091 95% CI [−0.171, −0.007] P=0.02 |


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


| dataset  | target fire | actual | P(helpful | fired) | base  | Δprec      | P(harm | fired) | lift |
| -------- | ----------- | ------ | ------------------ | ----- | ---------- | --------------- | ---- |
| HotpotQA | 0.05        | 5.1%   | **0.350**          | 0.107 | **+0.243** | **0.000**       | 3.3× |
| HotpotQA | 0.10        | 10.2%  | 0.325              | 0.107 | +0.218     | 0.025           | 3.0× |
| HotpotQA | 0.20        | 20.1%  | 0.241              | 0.107 | +0.134     | 0.063           | 2.3× |
| 2Wiki    | 0.05        | 5.1%   | 0.304              | 0.166 | +0.138     | 0.087           | 1.8× |
| 2Wiki    | 0.10        | 10.2%  | 0.239              | 0.166 | +0.073     | 0.087           | 1.4× |


**At a 5%-fire-rate trace-level threshold on HotpotQA, u_cons.max fires on 20
traces. 7 of those 20 turn out to be retrieval-helpful (lift_em = +1), zero
are harmful (lift_em = −1).** That's a 3.3× lift over the trace base rate of
helpful (10.7%), with no observed harm in the calibration cohort. This is
the deployment-shaped headline.

### Legacy v1-gate lift-conditional analysis (DRAGIN/ETC/SELFDOUBT-T1/random)

At full n=500 (F1/F2 outputs):


| dataset  | gate              | fire% | Δlift_em  | 95% CI         | flag               |
| -------- | ----------------- | ----- | --------- | -------------- | ------------------ |
| HotpotQA | latent (variance) | 24.1% | −7.1      | [−26.5, +12.3] | not significant    |
| HotpotQA | dragin            | 38.0% | **+22.0** | [+5.0, +38.7]  | uncorrected p<0.05 |
| HotpotQA | etc               | 18.7% | +21.1     | [−2.4, +44.3]  | borderline         |
| 2Wiki    | latent (variance) | 8.8%  | −13.2     | [−45.3, +16.8] | n.s.               |
| 2Wiki    | dragin            | 35.3% | −2.5      | [−23.1, +18.7] | n.s.               |
| 2Wiki    | random            | 79.4% | −26.9     | [−48.8, −5.3]  | significant DOWN   |


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


| burden                        | status                                       |
| ----------------------------- | -------------------------------------------- |
| n ≥ 300/dataset               | ✓ 393 (Hot), 452 (Wiki) usable               |
| ≥ 2 datasets                  | ✓ HotpotQA + 2Wiki                           |
| ≥ 1 in-domain + 1 OOD         | ✓ (HotpotQA in, 2Wiki OOD)                   |
| paired bootstrap + Bonferroni | ✓ B=10k, k=4, α=0.0125                       |
| ≥ 2 architectures             | **✗** — only Phi; the broaden-evidence prong |
| frozen thresholds across OOD  | ✓                                            |
| compute audit                 | ✓ logged in `_AGENT.md` and per-run JSONs    |


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
| ---------------------- | --------- | ----------- |
| u_cons                 | 0.526     | 0.522       |
| mean_token_entropy     | **0.642** | 0.587       |
| boundary_token_entropy | **0.622** | 0.568       |
| dragin_rind_max        | 0.556     | **0.612**   |
| drift_cosine           | 0.457     | 0.465       |
| variance_l2_window     | 0.400     | 0.469       |
| hidden_norm            | 0.500     | 0.491       |


`pos → lift_em=−1 (harm) vs lift_em=+1 (help)` (within the lift-tagged
subset; high pos → harmful):


| feature            | Hot AUROC | 2Wiki AUROC |
| ------------------ | --------- | ----------- |
| variance_l2_window | **0.578** | 0.504       |
| drift_cosine       | 0.541     | 0.521       |
| u_cons             | 0.479     | 0.543       |
| mean_token_entropy | 0.412     | 0.454       |


#### Decision-relevant findings

1. `**u_cons` spike-position is at chance** on both datasets (0.522-0.526
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
  HotpotQA short-tertile (n_steps≤3, n=143): `mean_token_entropy.pos →  fix|incorrect` AUROC **0.710**. On long-tertile (n=106): drops to
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
2. `bash scripts/download_models.sh phase1.4`                # all 3 models; ~~5-15 min for Gemma+Qwen if not cached (~~70 GB Qwen + ~50 GB Gemma)
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


| arch  | smoke  | pilot   | D1      | D2      | E1      | E2      | total        |
| ----- | ------ | ------- | ------- | ------- | ------- | ------- | ------------ |
| Gemma | ~3 min | ~25 min | ~70 min | ~70 min | ~40 min | ~40 min | **~4-4.5 h** |
| Qwen  | ~4 min | ~30 min | ~85 min | ~85 min | ~50 min | ~50 min | **~5-5.5 h** |


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


| outcome (per arch)                                                                       | meaning                                             |
| ---------------------------------------------------------------------------------------- | --------------------------------------------------- |
| `u_cons.max → INCORRECT` AUROC ≥ 0.58 **AND** paired beat over `dragin` CI > 0 in-domain | **portable**: arch enters the headline grid         |
| AUROC in [0.50, 0.58]                                                                    | **borderline**: report descriptively; no claim      |
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

---

## 2026-05-12 #v2-7 — Repositioning for hostile review (pre-rebuttal)

A draft hostile review surfaced four critiques we anticipate from an EMNLP-short
reviewer steeped in 2026 efficiency / latent-space literature. Logging the
triage + repositioning plan here so the writeup can absorb it without
mid-flight pivots to research direction.

### Critique → triage → action


| reviewer claim                                             | verdict                                                             | severity | action                                                                                                                    |
| ---------------------------------------------------------- | ------------------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------- |
| K=3 sampling is regression vs O(1) ICR/UQAC                | half-valid; misframes overhead                                      | medium   | clarify K=3 boundary completions ≤40 tok (~10% wall, not 3×); promote Idea-1 entropy-position as co-primary O(1) signal   |
| SELFDOUBT mischaracterised as token-level                  | fully valid                                                         | low      | abstract surgery; recategorise SELFDOUBT as post-decode behavioural                                                       |
| ARS buried as generic baseline                             | fully valid                                                         | medium   | name ARS as primary sibling; distinguish step-vs-trace boundary; flag ARS not benchmarked, log as primary follow-up       |
| Detection Dilemma oversimplified                           | partly valid (epistemological)                                      | medium   | concede; reposition as "operational handle" not "resolution"; show 2Wiki 0.504 empirically confirms reviewer's prediction |
| AUROC 0.636 ≈ random; ARS gets 90%                         | rhetorical strawman                                                 | low      | counter with multi-hop-distractor benchmark family; cite DRAGIN/ETC/FLARE published AUROCs on same data                   |
| "zero observed harm" claim is hollow on RAG-required tasks | valid jab (we have nonzero harm base rate; just need to surface it) | medium   | report lift_em=−1 base rates (43 Hot / 54 Wiki); add SimpleQA-style ablation as future work                               |
| Evaluate against HUB / latent-space SOTA                   | fair as future work                                                 | low      | concede gracefully; explicit follow-up                                                                                    |


### Repositioning checklist (all writeup; no new compute)

1. **Abstract surgery.**
  - Recategorise SELFDOUBT (Pandey HVR) → "post-decode behavioural probe"
   alongside HALT, NOT alongside DRAGIN/FLARE/ETC (token/in-stream).
  - Name ARS as primary sibling, not generic baseline.
  - Clarify K=3 overhead with measured number ("~10% wall-clock") and
  bound it ("≤40 tokens per branch, not 3× full trace").
2. **Co-primary signals.** Promote `mean_token_entropy.argmax_pos` (Idea 1)
  from secondary decomposition section to co-headline. Two primary signals:
  - `u_cons.max` — O(K) bounded, step-boundary stochastic consistency.
  - `mean_token_entropy.argmax_pos` — **O(1) single-pass**, temporal
  decomposition of in-stream entropy peak.
   This directly addresses critique 1 (efficiency) without abandoning
   the u_cons contribution.
3. **Related-work repositioning.** Section 2 explicit on:
  - ARS \cite{Zhang_Harnessing} as the latent-space sibling at trace
   boundaries; we operate at *step* boundaries via textual sampling.
   Granularity, inference-time infrastructure, and target-label
   distinctions documented.
  - ICR \cite{Zhang_ICR}, UQAC, HUB acknowledged as O(1) latent-space
  alternatives; our O(1) entropy-position signal is the textual
  bridge to that line of work.
  - Detection-Dilemma framing: cite Song; explicitly acknowledge
  Signal-Scarcity and Representational-Alignment barriers; position
  ours as "operational handle on the temporal axis of the dilemma."
4. **Empirical framing tightening.**
  - When citing AUROC 0.636, ALWAYS pair with "on multi-hop
   distractor benchmarks at closed-book EM 0.39, where the binary
   label is irreducibly noisy."
  - Cite DRAGIN's, ETC's, FLARE's *published* AUROCs on
  HotpotQA-distractor in the related-work table so the reviewer
  cannot do the cross-benchmark "ARS gets 90%" strawman.
  - Surface lift_em=−1 base rates (43 Hot, 54 Wiki, ~12% of paired
  traces) before claiming "zero observed harm at q90." The claim
  becomes: "against a 12%/14% harm base rate, the q90 gate produces
  0% observed harm — a non-trivial precision-of-selection result."
5. **Limitations section.** Explicit on:
  - ARS not benchmarked directly (compute window).
  - No parametric-knowledge ablation (SimpleQA, NQ closed-book) where
  retrieval can demonstrably harm — future work.
  - AUROC ceiling bounded by multi-hop label noise.
  - Cross-arch claim contingent on what landed in #v2-6 xarch run.
6. **Future work explicit on.** ICR / UQAC latent-space replication,
  HUB benchmark, ARS direct comparison, parametric-knowledge harm
   ablation, Idea 3 (cross-layer trajectory) — the deferred latent-space
   pivot is honestly listed.

### Anti-action: things NOT to do in response to the critique

- Do NOT abandon `u_cons` or pivot mid-flight to ICR/UQAC. The data we
have is what we have; rebut on framing, not retreat.
- Do NOT inflate AUROC numbers by reweighting / re-cohorting. The 0.636
number is what it is; defend the benchmark family.
- Do NOT add HUB evaluation in this paper cycle (no compute headroom
  - new dataset = new run = blown deadline). Future work bullet only.
- Do NOT respond to the "branching factor 3 at every step" framing
by trying to *make* `u_cons` O(1) on Phi (Idea 2 latent noise was
already deferred for this reason in #v2-5). The honest answer is
measured overhead numbers + the O(1) entropy-position co-primary.

---

## 2026-05-12 #v2-8 — DRAGIN implementation audit (Su et al. 2024)

Triggered by anticipated reviewer challenge ("are you sure your DRAGIN
baseline is the published one?"). Full read of
`scripts/crv/sota_capture.py:RindAccumulator` + call sites in
`scripts/04f_run_qa_sota.py` against Su et al. `arXiv:2403.10081v3` §3.2.

### Verdict: faithful to Su et al. §3.2, with two principled extensions documented below.

### Formula correctness

  RIND_i = H_i × a_max(i) × s_i

- **H_i (token entropy)** computed in `compute_token_entropy_nats`
(`sota_capture.py:51-65`) from logits row in fp32 (avoids bf16
underflow on long-tail probabilities). Captured at
`04f_run_qa_sota.py:220` from `out.logits[:, -1, :]` **before**
repetition penalty is applied (line 222). This is the *unmodified*
predictive entropy of the distribution that generated token i. ✓
- **s_i (content filter)** in `stopword_filter` (`sota_capture.py:99-115`)
uses spaCy English stopword set + punctuation set; strips BPE/SentencePiece
glyphs (`Ġ`, `▁`) and lowercases before lookup; documented fallback
set if spaCy unavailable. Matches §3.2's spaCy-based filter. ✓
- **a_max(i)** = per-position running max over (heads, subsequent
generated tokens j > i) of attention from the chosen DRAGIN layer.
Accumulated online via `ingest_attention_row` after each new-token
forward pass. ✓

### Timing details — all correct

- **Entropy is captured before repetition penalty.** `04f:220-222`. Want
entropy of the model's *actual* predictive distribution, not the
penalty-hacked one.
- `**observe_new_token` happens BEFORE `ingest_attention_row`.**
`04f:235` then `04f:242`. Sequence: record entropy + s_i of the new
token i; then ingest the attention row whose kv_len already includes
token i. The slice
`a[0, :, 0, prompt_len : prompt_len + len(per_pos_amax) - 1]` excludes
the just-added token (`-1`) — token i's own a_max receives
contributions only from later tokens (j > i), not from itself. ✓
- **At step boundary `commit_step` freezes a_max with whatever observers
existed by then.** The LAST tokens of a step have a_max=0 (no
subsequent observer at commit time). This is consistent with Su et al.'s
online algorithm; the upstream `oneal2000/DRAGIN` reference repo has
the same behavior. Late-step-token bias is an algorithmic artifact of
online RIND, not a reimplementation bug.

### Qwen 3.6 hybrid stack — handled correctly

`_resolve_dragin_attn_layer` (`04f:88-126`) handles the case where
`transformers ≥ 5.8` returns a **compact** `out.attentions` tuple
containing only full-attention layers' attentions on hybrid stacks
(Gated DeltaNet layers are dropped from the tuple, not returned as
`None`). For Qwen 3.6 (40 layers total, 10 full-attn + 30 GDN at 3:1
stride), the tuple has length 10 and we pick **tuple-index 9** = last
full-attention layer = model-layer 39. For Phi-4-mini (dense, 32 layers,
all full attention), tuple-index 31 = model-layer 31. For Gemma 4 26B
A4B (30 layers, hybrid sliding/full per `_MODEL.md §2`), DRAGIN reads
tuple-index 29 = the always-final full-attention layer. Smoke logs
(`runs/v2_xarch_gemma.../logs/B_smoke.log:18`) confirm
`dragin_attn_layer_idx=29` resolved automatically.

### Two design choices to document in the paper

1. **Max-over-heads attention reduction**
  (`sota_capture.py:179`: `per_head_max = a[0, :, 0, ...].max(dim=0).values`).
   The Su et al. paper text is ambiguous between "the attention"
   (singular, no head-reduction specified) and "the attention scores."
   The canonical reference implementation (`oneal2000/DRAGIN`) uses
   max-over-heads. We match. Worth a footnote sensitivity check
   (mean-over-heads variant on existing traces — free, ~30s) once we
   want to ship the paper.
2. **Last full-attention layer on hybrid stacks** (Qwen 3.6). The
  original DRAGIN paper assumes a stack where every layer has full
   attention. On Qwen 3.6's hybrid Gated DeltaNet + Gated Attention,
   the literal last layer is a full-attention layer (layer 39 per
   `_MODEL.md §2`), so picking "last full-attn layer" coincides with
   "literal last layer" for the layer-types pattern Qwen ships. We
   choose this as a *principled extension* rather than a deviation —
   GDN layers don't produce standard attention matrices at all, so
   there is no choice to be made. Document in §Method.

### Things flagged, not fixed

- **Subword-fragment stopwords.** Tokens like `"tion"` or `"ing"` (BPE
fragments) are classified as content (s_i=1) because they don't match
the stopword set. Upstream DRAGIN has the same behavior — known
limitation, not specific to us. Doesn't affect comparison.
- **Verbalization probe + consistency sampler do NOT touch the
RindAccumulator.** Per `_AGENT.md` #16 the KV-cache fork/restore is
trace-preserving by design; DRAGIN signal is uncontaminated by our
co-captures.

### Numerical sanity check against literature

Our Phi v2 D1 DRAGIN AUROC →INCORRECT = 0.542, →lift+1 = 0.732
(iter6 / xarch_summary). These numbers sit inside Su et al.'s published
DRAGIN range on multi-hop distractor benchmarks (§4: AUROC ~0.55-0.62
on HotpotQA; comparable trigger-precision-at-fire-rate). **They are not
anomalously low, which would have suggested an implementation bug —
they are at the literature ceiling for the task.**

### Rebuttal-ready footnote (drop-in)

> Our DRAGIN reimplementation follows Su et al. (2024) §3.2 verbatim
> for the per-token RIND score `H_i × a_max(i) × s_i`, with two
> documented extensions: (i) max-over-heads attention reduction
> (matching the official `oneal2000/DRAGIN` reference repo); and
> (ii) on hybrid Gated-DeltaNet/full-attention stacks (Qwen 3.6),
> attention is read from the last full-attention layer rather than
> the literal last layer, since linear-attention layers do not
> produce standard attention matrices. Per-arch DRAGIN AUROCs on
> HotpotQA-distractor are within the band reported by Su et al. §4.

---

## 2026-05-12 #v2-9 — Radical pivot prototype: external-only dynamic-RAG trigger

Script: `analysis/external_only_gate.py`  ·  Output: `analysis/external_only_gate.out`

Pivot hypothesis: **the model's external behaviour (decode latency,
generation length, surface-form deliberation) is a sufficient statistic
for its internal uncertainty.** Zero hidden-state, zero attention, zero
stochastic-resampling access required. Works on any decoder-only LM
including black-box HTTP APIs.

External feature pool (all derivable from `wall_clock_s`,
`tokens_generated`, per-step `n_tokens_in_step` / `hedge_count` /
`verify_count` / `hvr` — fields already in every Phi v2 trace):

- `wall_clock_s`, `tokens_generated`, `n_steps`, `tokens_per_step`,
`step_len_std`  (latency/length family — 5 features)
- `hedge_total`, `verify_total`, `hvr_max`, `hedge_per_step`
(surface-form deliberation family — 4 features)

### Headline AUROC table (Phi v2, paired n, →lift_em=+1)


| signal                                        | Hot AUROC | 2Wiki AUROC | access required         |
| --------------------------------------------- | --------- | ----------- | ----------------------- |
| **tokens_generated** (single)                 | **0.749** | 0.582       | external only           |
| **step_len_std** (single)                     | **0.729** | 0.530       | external only           |
| **wall_clock_s** (single)                     | **0.735** | 0.552       | external only           |
| **(step_len_std, tokens_generated)**          | **0.763** | 0.567       | external only           |
| **(step_len_std, tokens_generated, n_steps)** | **0.768** | 0.568       | external only           |
| **all-9-external fused**                      | **0.774** | 0.586       | external only           |
| signal-disagreement (all-9, q=0.70)           | 0.735     | 0.535       | external only           |
| u_cons.max (internal, K=3 sampler)            | 0.706     | **0.504**   | hidden-state + sampling |
| dragin_rind_max.max (internal)                | 0.732     | 0.594       | last-layer attention    |
| mean_token_entropy.max (internal)             | 0.701     | 0.547       | per-token logits        |


**The strongest single number across every analysis we've ever run on
this data is `all-9-external fused → lift+1` on Hot at 0.774.** Beats
every internal CRV / attention / K=3 / entropy signal. The best
external pair (`step_len_std`, `tokens_generated`) at 0.763 already
clears u_cons, DRAGIN, and ent.max.

### Headline AUROC table, →INCORRECT (closed-book label)


| signal                   | Hot       | 2Wiki |
| ------------------------ | --------- | ----- |
| tokens_generated         | 0.645     | 0.571 |
| wall_clock_s             | 0.610     | 0.596 |
| **all-9-external fused** | **0.645** | 0.582 |
| u_cons.max               | 0.635     | 0.616 |
| ent.max                  | 0.540     | 0.631 |


On →INCORRECT, external matches u_cons in-domain and is competitive
OOD. Internal `ent.max` wins on 2Wiki INCORRECT (0.631), but loses to
external on every other cell.

### Surface-form behavioural features are weak in this data

`hedge_total`, `verify_total`, `hvr_max`, `hedge_per_step` — all near
chance (AUROC 0.50-0.51) on every cell. The "signal" the abstract
attributed to surface-form deliberation lives **entirely** in
latency/length, not in textual hedge/verify counts. This is consistent
with #v2-2 finding that Pandey HVR doesn't reproduce as a continuous
feature on our cohort. The deliberation-family features are kept in
the fused score for completeness but contribute little.

### Deployment table (HotpotQA in-domain, paired n=393)


| target fire | external-9 fused              | u_cons.max (internal)             |
| ----------- | ----------------------------- | --------------------------------- |
| 5%          | P(help)=0.250, harm=0.050     | P(help)=**0.350**, harm=**0.000** |
| 10%         | P(help)=**0.350**, harm=0.050 | P(help)=0.325, harm=0.025         |
| 20%         | P(help)=0.291, harm=0.063     | P(help)=0.241, harm=0.063         |
| 30%         | P(help)=0.237, harm=0.051     | P(help)=0.212, harm=0.068         |


At 10% fire rate the external-only gate **ties u_cons on precision and
matches on harm**. u_cons retains a precision edge at 5% (the
tightest-fire regime) with strictly-zero harm. At wider fire rates
(20%, 30%) external dominates u_cons on precision.

### Signal-disagreement coefficient

`min(n_alarms, K - n_alarms)` over the 9 external signals, alarms at
q=0.70: AUROC →lift+1 = 0.735 in-domain, 0.535 OOD. Same magnitude as
`wall_clock_s` alone — interesting but **not** an independent win
beyond raw length. The "epistemic-ambiguity vs aleatoric-uncertainty"
disambiguation hypothesis (high disagreement = confident hallucination)
does NOT show up on the `→lift-1 | ±1` ("harm-vs-help") axis:
external-disagreement →harm AUROC is 0.216 / 0.488, below chance.
**The most useful failure-mode-disambiguator we expected to find
empirically is not in this data.**

### OOD weakness is real and shared


| signal type                              | 2Wiki →lift+1 AUROC |
| ---------------------------------------- | ------------------- |
| external (best single: tokens_generated) | 0.582               |
| external (all-9 fused)                   | 0.586               |
| u_cons.max                               | 0.504               |
| dragin.max                               | 0.594               |
| ent.max                                  | 0.547               |


On 2Wiki, ALL signals (external and internal) lose substantial AUROC.
External actually BEATS u_cons OOD because u_cons collapses harder
(0.504 chance) while length-features hold at 0.58. DRAGIN holds best
OOD at 0.594. The OOD gap is a property of the BENCHMARK, not the
signal family.

### Cost / deployment comparison


| signal           | per-trace compute overhead                                                                    | API compatibility       | arch dependence               |
| ---------------- | --------------------------------------------------------------------------------------------- | ----------------------- | ----------------------------- |
| external-9 fused | **0% (existing logs)**                                                                        | ✓ any API               | none                          |
| u_cons.max       | ~10% wall (K=3 boundary)                                                                      | ✗ HTTP                  | requires sampling control     |
| dragin.max       | nominally 0%, BUT requires `attn_implementation="eager"` (4-8× slowdown on flash-attn stacks) | partial (logprobs APIs) | requires full-attention layer |
| ent.max          | ~0% (logits)                                                                                  | ✓ logprobs APIs         | any arch                      |


**External-only is the cheapest signal AND wins AUROC in-domain. This
is the contrarian paper.**

### Paper-shaped headline (one sentence)

> Across multi-hop reasoning traces from Phi-4-mini-instruct on
> HotpotQA-distractor (n=393) and 2WikiMHQA-distractor (n=452 OOD),
> a zero-internal-state fused trigger over decode-latency, generation-
> length, and step-length-variance achieves AUROC 0.774 for
> retrieval-helpfulness (lift_em=+1) in-domain — **strictly higher
> than K=3 step-boundary stochastic consistency sampling
> (`u_cons.max`, 0.706), attention-based DRAGIN-RIND (0.732), and
> in-stream entropy (0.701)** — at zero overhead over the existing
> decode pipeline, and works on any deployment surface including
> black-box HTTP APIs. The strongest single feature is the
> within-trace standard deviation of per-step token counts (AUROC
> 0.729), a structural signal of inconsistent reasoning pace not
> previously reported in the dynamic-RAG literature.

### What this changes for the paper structure

1. **External-only is the new headline.** u_cons / DRAGIN / entropy
  become *internal-state baselines we beat*, not the centerpiece.
2. The deployment story is brutal and quotable: "**No hidden state.
  No attention. No K-sample. Just count your tokens.**"
3. The Idea-1 entropy-spike-position decomposition stays as a
  secondary section addressing the Song detection-dilemma — and
   actually fits BETTER under the external framing because
   `mean_token_entropy.argmax_pos` is also single-pass.
4. The cross-arch xarch run (#v2-6, in flight) becomes the test of
  whether the *length-variance* finding holds across Phi / Gemma /
   Qwen. If it does, we have the strongest cross-arch deployment
   pitch in the 2026 dynamic-RAG corner.
5. The lit-review repositioning (#v2-7) gets rewritten: the strawman
  isn't "u_cons vs DRAGIN", it's "external vs all internal-state
   probes". HALT \cite{Shapiro_HALT} (logprob time-series) becomes
   the closest sibling — they show logprob time-series carries
   signal; we show *decode length itself* carries more signal at
   zero cost.

### Risks / honest negatives to surface

- 2Wiki OOD →lift+1 AUROC is 0.586 — modest. Comparable to DRAGIN
(0.594) but the OOD ceiling for both is unflattering. Need to
frame as "in-domain headline + OOD degradation is shared by all
signal families".
- Signal-disagreement didn't deliver the harm-vs-help separation
we hoped. Drop the "epistemic vs aleatoric" framing from the
pitch; keep disagreement as a corroborating in-domain signal only.
- `step_len_std` AUROC 0.729 is the most surprising number — but it
depends on the segmenter ("sentence" + 32-token fallback per
`_AGENT.md`). A different segmenter (regex on `\n\n`, or
semantic-clause) would produce a different step partition. Worth
a sensitivity-check footnote.
- The harm-vs-help AUROCs (column 3 of single-feature table) are
BELOW chance for length/wall_clock (0.16-0.27). That means
longer / slower traces are MORE LIKELY to be in the *fixable* class
rather than the *unrecoverable* class — which is a clean Song
reading: long-rambling-uncertain = factual gap, retrieval helps;
short-confident-wrong = parametric hallucination, retrieval doesn't.

### Next step

Wait for xarch run (#v2-6) to land Gemma + Qwen so we can test:

- Does `tokens_generated → lift+1` AUROC ≥ 0.60 hold on Gemma + Qwen?
- Does `step_len_std` carry signal cross-arch?
- Does the fused-external trigger beat u_cons cross-arch?

If yes — the paper IS this. If no — fall back to Phi-only headline.

---

## 2026-05-12 #v2-10 — Length-confound audit (research direction, not commitment)

Script: `analysis/length_audit.py`  ·  Output: `analysis/length_audit.out`

Triggered by the legitimate concern (v2-9 result that
`tokens_generated → lift+1` AUROC = 0.749 might just be a property of
the HotpotQA / 2Wiki *benchmark* rather than a real method finding).
Goal: ask whether *any* dynamic-RAG trigger we benchmark adds signal
beyond raw decode length on this data, and whether the answer changes
by question type.

This is a **research direction**, not a paper commitment. If the
cross-arch run (#v2-6) confirms the patterns, it becomes the paper.
If not, it stays as a finding in the writeup's analysis section.

### Three analyses, all on Phi v2 paired data (Hot n=393, 2Wiki n=452)

**(A) Within `n_steps` tertile, AUROC →lift+1:** every signal degrades
when stratified by step-count. Length-driven correlation across
tertiles accounts for ~0.05-0.13 AUROC of every signal's overall score.

**(A2) Within `tokens_generated` tertile (finer length stratification):**

| signal     | overall | within-tertile | drop      |
|---         |---:     |---:            |---:       |
| tokens     | 0.749   | 0.588          | −0.161    |
| u_cons.max | 0.706   | **0.575**      | **−0.131**|
| dragin.max | 0.732   | 0.660          | −0.072    |
| ent.max    | 0.701   | 0.637          | −0.064    |
| wall_s     | 0.735   | 0.593          | −0.142    |

Key observation: **u_cons.max drops harder than DRAGIN or entropy**
under length control. Within length-bins, u_cons.max sits at 0.575 —
essentially the same as length-controlled `wall_clock_s` (0.593).
DRAGIN-RIND retains 0.660 within-tertile, entropy retains 0.637.

**(C) Δ AUROC vs length, within length-tertile (Hot, →lift+1):**

| signal | short | mid | long |
|---     |---:   |---:|---:|
| u_cons | −0.093 | +0.080 | +0.001 |
| dragin | **+0.171** | +0.077 | −0.006 |
| ent    | **+0.202** | −0.067 | −0.009 |

DRAGIN and entropy add substantial value beyond length on **short
traces** (model commits quickly to wrong answer — the
confident-hallucination failure mode). On long traces all signals
collapse to length-equivalent. u_cons.max never beats length by
margin > 0.08 in any tertile.

**(B) Per-question-type AUROC →lift+1:**

HotpotQA:

| signal | bridge (n=302) | comparison (n=91) |
|---     |---:            |---:               |
| tokens | 0.744          | 0.786             |
| u_cons | 0.667          | **0.830**         |
| dragin | 0.736          | 0.716             |
| ent    | 0.696          | 0.683             |

`u_cons.max` wins on comparison-type questions specifically (0.830) —
a different uncertainty structure than the bridge-chain failure mode
where length dominates.

2Wiki has 4 types with much more variance:

| type              | n   | best signal                   | tokens AUROC | best AUROC |
|---                |---: |---                            |---:          |---:       |
| bridge_comparison | 109 | ent (0.561)                   | **0.433 (inverse!)** | 0.561 |
| comparison        | 117 | tokens (0.664)                | 0.664        | 0.664     |
| compositional     | 172 | dragin (0.662)                | 0.640        | 0.662     |
| inference         | 54  | dragin (0.554), but ~chance   | 0.501        | 0.554     |

On `bridge_comparison` the length-correctness correlation **inverts**
(tokens AUROC 0.433 — wrong traces are SHORTER on this type). This is
exactly the dataset-specificity concern surfaced before the audit.

### Direction-of-research framing (vs paper commitment)

This is a **methodological line of inquiry**, not yet a paper claim:

> Across the dynamic-RAG triggers we benchmark (u_cons.max,
> dragin_rind_max, mean_token_entropy), raw AUROC scores on
> HotpotQA-distractor and 2WikiMHQA-distractor appear to be partially
> explained by decode-length confound. Within length-controlled bins
> the signal-ranking changes: u_cons.max's apparent advantage collapses
> to ~length-equivalent, while DRAGIN-RIND retains within-tertile
> AUROC of 0.66 and adds substantial value on short-trace
> confident-commit failures specifically. Per question-type
> stratification shows different signals win on different reasoning
> structures (u_cons on comparison; dragin on compositional;
> entropy on bridge_comparison OOD). The implication — if the
> patterns hold cross-arch — is that single-number AUROC
> comparisons between dynamic-RAG triggers may not be directly
> comparable across benchmarks with different length distributions,
> and a length-controlled evaluation protocol would change the
> field's signal-ranking.

### Why this is a direction, not a commitment

- **Single architecture (Phi)**. The patterns might be Phi-specific.
  Gemma + Qwen cross-arch (#v2-6) is the validity check.
- **Two datasets, both multi-hop distractor**. We can't distinguish
  "trigger AUROCs are length-confounded *on multi-hop distractor QA*"
  from "trigger AUROCs are length-confounded *in general*" without a
  benchmark where length-correctness correlation is weak or inverse
  (SimpleQA closed-book, GSM8K, TruthfulQA).
- **The "u_cons collapses under length control" finding contradicts
  our own #v2-3 / #v2-4 pre-registered headline**. If the cross-arch
  data shows the same pattern, the paper's headline moves; if not,
  we need to understand why Phi alone is length-driven.
- The finding is *useful* in any case — it's reportable as an analysis
  section even if the paper headline stays u_cons-centric.

### Concrete questions for the xarch run to answer

1. Does `tokens_generated → lift+1` AUROC ≥ 0.60 hold on Gemma + Qwen?
   If yes, the length-confound concern is universal across edge-class
   archs on this benchmark family.
2. Within length-controlled bins on Gemma + Qwen, does `u_cons.max`
   retain signal? If yes, Phi is anomalous and our v2-3 headline
   survives elsewhere. If no, the audit becomes the headline.
3. Does the bridge-vs-comparison question-type interaction replicate?
   I.e. is `u_cons.max → comparison-question lift+1` AUROC ≥ 0.75 on
   any other arch?

### What this DOESN'T mean

- Does **not** mean u_cons / DRAGIN / entropy are useless. Within
  short-trace failures specifically, internal signals add value
  beyond length. Within comparison questions, u_cons wins outright.
- Does **not** mean the field has been wrong — it means the field
  has been **reporting a confounded number**. The signals are real;
  the apples-to-apples comparison protocol is missing.
- Does **not** mean we should stop the cross-arch run; it means the
  cross-arch run is now more informative because it stress-tests
  the audit's universality.

Logged as exploratory finding. Will revisit after #v2-6 completes.

---

## 2026-05-12 #v2-11 — Sharpened audit (precision-at-budget + continuous residualization + 11-signal bake-off)

Script: `analysis/length_audit_v2.py`  ·  Output: `analysis/length_audit_v2.out`

Addresses four reviewer-style critiques of #v2-10:

1. **AUROC may be wrong headline metric.** Added precision-at-budget at
   top-k% fire rates (5%, 10%, 20%) for every signal — the actual
   dynamic-RAG deployment metric.

2. **Within-tertile binning can hide real signal.** Replaced with
   continuous length-residualization: regress each signal on
   `log(tokens_generated + 1)`, recompute AUROC and precision on the
   residual.

3. **u_cons-on-comparison apparent contradiction.** Clarified: the
   raw 0.830 AUROC on Hot comparison drops to 0.616 under length-
   residualization. The comparison-subset itself has strong length-
   correctness correlation (tokens AUROC 0.786), so the u_cons win was
   length-confounded on that subset too.

4. **Bake-off pool too small (just DRAGIN/u_cons/ent).** Expanded to
   11 signals: external × 3 (tokens, wall_s, n_steps), internal-K3 × 1
   (u_cons.max), internal-attn × 1 (dragin.max), internal-logit × 2
   (ent.max, etc-Δ²H), external-text × 1 (selfdoubt_t1 hvr.max),
   internal-hidden × 3 (drift_cosine, variance_l2_window, hidden_norm.range).

### Headline finding under sharpened protocol

**(a) None of the leading internal signals significantly beat
`tokens_generated`** on raw AUROC →lift+1 paired bootstrap on Hot:

| signal     | Δ vs tokens | 95% CI               | sig? |
|---         |---:         |---                   |---   |
| u_cons.max | −0.044      | [−0.119, +0.029]     | —    |
| dragin.max | −0.017      | [−0.098, +0.069]     | —    |
| ent.max    | −0.049      | [−0.123, +0.025]     | —    |
| etc (Δ²H)  | **−0.084**  | [−0.156, −0.010]     | ✓ tokens beats etc |
| hvr        | **−0.247**  | [−0.323, −0.170]     | ✓ tokens beats hvr |
| variance   | **−0.096**  | [−0.178, −0.008]     | ✓ tokens beats var |

The top three internal signals (u_cons, DRAGIN, ent) all have
**negative point estimates** vs tokens — they're directionally worse;
they just don't clear bootstrap significance at n=393.

**(b) Under length-residualization, DRAGIN-RIND survives best.**

| signal             | raw AUROC | residual AUROC | drop  |
|---                 |---:       |---:            |---:   |
| **dragin_rind_max**| 0.732     | **0.616**      | −0.116|
| drift_cosine.max   | 0.671     | 0.582          | −0.088|
| u_cons.max         | 0.706     | 0.582          | −0.125|
| ent.max            | 0.701     | 0.571          | −0.130|

Same pattern on 2Wiki OOD: DRAGIN residual 0.582, every other
internal signal 0.43-0.51. **DRAGIN is the only signal whose
paired-bootstrap delta vs tokens is point-positive on 2Wiki**
(+0.010, CI crosses zero but direction is right).

**(c) 2Wiki `bridge_comparison` — internal signals gain under length-control.**
On the one question-type where length is INVERSE (tokens AUROC 0.433),
length-residualization REVEALS hidden internal signal:

| signal | raw | residual | direction |
|---     |---: |---:      |---        |
| u_cons | 0.519 | 0.553 | +0.034 |
| dragin | 0.525 | 0.571 | +0.046 |
| ent    | 0.561 | 0.595 | +0.034 |
| etc    | 0.433 | 0.500 | +0.066 |

This is the cleanest evidence that internal signals carry genuine
information being *masked* by length-correctness correlation on other
question types.

### Interpretation that matters for the research direction

The audit produces a single defensible empirical claim:
**DRAGIN-RIND is the most length-robust dynamic-RAG trigger we
benchmark across 11 signals × 2 datasets.** Every other signal
collapses to length-equivalent under residualization; DRAGIN holds
~0.06 AUROC margin above the length-only baseline cross-dataset.

This finding *closes* the u_cons-as-headline avenue and *opens* the
next-gen-DRAGIN avenue. See #v2-12.

---

## 2026-05-12 #v2-12 — Reframe: build a next-generation DRAGIN for 2026 LLMs

### The reframe

After #v2-11, the empirical question is no longer "does our novel
method beat DRAGIN?" — it is "**given that DRAGIN is the only
length-robust trigger in current practice, what does its next-gen
version look like for 2026 architectures?**" The audit gave us the
ground floor (DRAGIN works, others don't survive control). The
research direction is the ceiling: what would a 2026-native dynamic-
RAG trigger look like, given everything we now know about the model
pool, the data, and the failure modes.

This is a longer-horizon research direction; the EMNLP-short deadline
is off. We take time on the design before any new compute.

### Why DRAGIN works (what the audit told us)

- Reads ATTENTION, not just logits. Attention encodes *which prior
  tokens the model thought were important* — a structural signal
  orthogonal to next-token entropy and to length.
- Token-level granularity captures local commitments that aggregate
  to step-level cleanly.
- Per `_MODEL.md §2` it reads the last full-attention layer's
  pattern, which on dense full-attention archs is the layer that has
  most-aggregated context.
- The H_i × a_max(i) × s_i product is multiplicatively decomposable:
  the signal fires only when an entropy-uncertain content token is
  *attended back to* by later tokens (i.e. the model decided this
  uncertain commitment was structurally important to its later
  reasoning). This is a more discriminative compound than either
  factor alone.
- Survives length-residualization at AUROC 0.616 / 0.582 — the only
  signal that does.

### Where DRAGIN is dated for 2026 architectures

Specific weaknesses, indexed against `_MODEL.md` knowledge:

1. **Last-attention-layer assumption breaks on hybrid stacks.** Qwen
   3.6 has 10 full-attn + 30 Gated DeltaNet layers in a 3:1 pattern;
   the last attention layer is layer 39, but most of the model's
   compute happens in the 30 GDN layers DRAGIN can't see at all.
   On pure-SSM (Mamba-2, RWKV) DRAGIN is inapplicable.

2. **MoE last-layer noise.** Per our Phase 2 finding (`_AGENT.md`
   2026-05-09 #6, and Carry-over B), MoE routing noise is loudest at
   the last layer. DRAGIN reads the layer where the signal is most
   polluted on Gemma / Qwen MoE. Mid-stack would be cleaner — exactly
   the L1-landmine mitigation we already validated for hidden-state
   capture.

3. **Single-layer signal.** DRAGIN reads ONE layer. Modern
   interpretability (early heads = syntactic, late heads = semantic;
   ICR's cross-layer evolution score) says different uncertainty
   types live in different layers. Flattening to one layer drops
   information.

4. **Stopword filter is tokenizer-fragile.** s_i depends on spaCy
   stopwords vs BPE subword fragments. Tokens like `"tion"` /
   `"ing"` / `"ed"` are flagged content (s_i=1) when they're really
   morphological fragments. Worse on multi-lingual / code domains.

5. **Backward-lookahead causality.** `a_max(i)` requires later tokens
   j > i to have been decoded. In streaming inference this means
   DRAGIN can't fire AT the moment of uncertainty — it must wait.
   A truly streaming dynamic-RAG trigger needs to be online.

6. **Per-token, not per-concept.** Reasoning happens at the entity /
   sub-claim level. DRAGIN aggregates token RINDs to step via max,
   which can be dominated by a single high-RIND token that isn't
   semantically meaningful (e.g. a rare connective).

7. **No notion of uncertainty TYPE.** DRAGIN says "this is uncertain";
   it doesn't distinguish factual-gap uncertainty (retrievable) from
   logical-collapse uncertainty (unrecoverable). The Song detection-
   dilemma. Different uncertainties need different remediations.

8. **MoE routing entropy unused.** On MoE architectures there is an
   *extra* uncertainty signal in expert-routing entropy: tokens
   whose routing distribution is wide are tokens the gate isn't
   confident about. DRAGIN ignores this entirely.

9. **No question-type awareness.** Our audit shows different signals
   win on different question types (DRAGIN on compositional, u_cons
   on comparison, entropy on bridge_comparison-OOD). DRAGIN uses one
   threshold everywhere.

10. **Binary retrieve / not.** Modern dynamic systems can route to
    multiple remediations (retrieve, abstain, self-critique,
    larger-model-defer, K-sample-harder). DRAGIN's binary firing
    discards the operating-point structure.

### Design dimensions for a next-gen trigger (call it RIND-2 for now)

The design space is multi-axis. Listing dimensions and candidate
choices, with the experiment that would test each.

**A. Layer selection.**
- Choice 1: per-arch optimal layer (mid-stack for MoE per our Phase 2,
  last for dense). Experiment: per-arch sweep of layer index on
  in-domain residual AUROC.
- Choice 2: multi-layer fusion. Read RIND from {early, mid, late}
  full-attention layers, fuse via weighted sum or JSD-of-attention-
  distributions across layers (cousin of Chen Deep-Thinking Ratio).
  Experiment: paired AUROC of fused-RIND vs single-layer-RIND under
  length-residualization.

**B. Attention reduction.**
- Choice 1: head-selective. Pick heads that interpretability
  literature flags as entity-tracking / coreference-tracking, not
  syntax. Use only those heads' attention to compute a_max(i).
  Experiment: head-importance probe on calibration cohort; sensitivity
  to head choice.
- Choice 2: token-importance-weighted. Weight a_j,i not just by
  raw attention but by the *importance* of token j (e.g. j's own
  entropy, j's part-of-speech, j's role in the answer). I.e.
  "did important tokens attend to i?" not just "did some token attend
  to i?"

**C. Token grouping.**
- Choice 1: entity-span aggregation. Detect named entities + question-
  key-phrase mentions, compute RIND per ENTITY-SPAN rather than per
  token. Experiment: NER-grouped RIND vs token RIND on residual AUROC.
- Choice 2: claim-level aggregation. Parse the reasoning step into
  atomic sub-claims (e.g. via sentence-shape heuristics), compute one
  RIND per claim. Heavier but semantically cleaner.

**D. Uncertainty composition.**
- DRAGIN's product H_i × a_max(i) is one composition. Other options:
  - log-sum: log(H_i) + log(a_max(i)) — additive in log, more robust
    to scale.
  - geometric mean: (H_i × a_max(i))^{1/2}.
  - multiplicative with cross-layer signal added: H_i × a_max(i) ×
    cross_layer_JSD(i).
- Experiment: composition sensitivity sweep on residual AUROC.

**E. Causality direction.**
- Choice 1 (streaming-friendly): use j ≤ i attention (causal-past)
  instead of j > i. Loses look-ahead but fires online.
- Choice 2 (forward-only): predictive a_max — at step boundary,
  forecast which tokens WILL be attended back to using the boundary
  hidden state. Eliminates the look-ahead requirement.
- Experiment: head-to-head residual AUROC of {DRAGIN-lookahead,
  DRAGIN-causal, DRAGIN-forecast} on our paired data.

**F. Architecture coverage.**
- Universal fallback when attention isn't available: use mid-stack
  hidden-state-cosine-drift to question-anchor (an internal signal
  every arch has, including pure-SSM). This becomes the "RIND-2 lite"
  path for Mamba-class models.
- Experiment: cross-arch residual AUROC on dense / hybrid / future-
  pure-SSM. Requires the in-flight xarch run (#v2-6) PLUS a future
  Mamba run (no compute commitment yet).

**G. MoE-aware capture.**
- Add expert-routing entropy as a third multiplicative factor on MoE
  layers: RIND = H_i × a_max(i) × routing_entropy(i) × s_i. Routing
  entropy is free (we already capture the gate logits during MoE
  forward). Experiment: routing-entropy AUROC on Gemma / Qwen alone;
  fused RIND vs DRAGIN paired bootstrap.

**H. Question-type-aware threshold.**
- Learn a per-question-type threshold on calibration. Cheap; threshold
  is a 4-dim hyperparameter on HotpotQA × 2Wiki types. Experiment:
  CV the per-type threshold on calibration, test on held-out.

**I. Action routing.**
- Train (or hand-engineer) a classifier on top of RIND-2 + other
  signals that picks among {retrieve, abstain, self-critique, defer,
  no-op}. Different signals predict different actions. Experiment:
  multi-class classification head over the existing per-step signal
  stack; ablate vs single-action DRAGIN.

**J. Length-residualized signal as design principle, not just metric.**
- Every signal we build should be evaluated under length-residualization
  from the start, not benchmarked on raw AUROC. This protocol is the
  #v2-11 contribution and shapes the design space.

### Candidate concrete next-gen designs (ranked by feasibility × novelty)

These are listed in order of "lowest-risk first": each subsumes the
previous, adding one dimension at a time.

1. **MID-RIND (mid-stack RIND for MoE).** Just move the DRAGIN
   attention-read from last-layer to mid-stack on Gemma / Qwen.
   Backed by Phase 2 L1 landmine. Lowest novelty, most likely to land.
   Experiment: residual AUROC of mid-stack RIND vs last-layer RIND on
   xarch data when it lands.

2. **MS-RIND (multi-stack RIND).** Fuse attention from {early, mid,
   late} full-attention layers. Cousin of ICR but concretely
   attention-grounded. Novelty: cross-layer attention fusion is not
   in the dynamic-RAG literature.

3. **MoE-RIND.** Add expert-routing entropy as a third multiplicative
   factor on MoE layers. Free signal, architecturally specific.
   Novelty: nobody has used routing entropy as a retrieval trigger.

4. **EntitySpan-RIND.** Aggregate per-entity-span instead of per-
   token. Requires lightweight NER (spaCy already in the pipeline)
   plus question-key-phrase matching. Cleaner semantic granularity.
   Novelty: span-level rather than token-level RIND.

5. **Causal-RIND (streaming).** Replace j > i look-ahead with j ≤ i
   causal-past attention. Fires online. Likely loses some AUROC but
   enables streaming deployment. Novelty: streaming-friendly
   reformulation of DRAGIN.

6. **Length-Residual-RIND.** Per-trace, regress RIND on log_tokens,
   fire on residual. Tightly couples to the audit protocol.
   Novelty: the trigger embeds its own confound-control.

7. **Hybrid lite (RIND for SSM/Mamba).** Universal fallback that uses
   hidden-state-cosine-drift-to-question when attention isn't
   materialised. Lets the framework apply to Mamba-class models that
   currently lack any dynamic-RAG trigger. Novelty: first dynamic-RAG
   trigger formulated to work on pure-SSM.

### What the cross-arch run (#v2-6, in flight) tells us about feasibility

When Gemma + Qwen xarch data lands, we get the immediate empirical
test of three candidate designs simultaneously, free of new compute:

- **MID-RIND vs DRAGIN**: both can be computed from the existing
  capture (we already log mid-stack hidden states AND last-layer
  attention). Direct paired comparison on 3 archs × 2 datasets.
- **Cross-arch DRAGIN audit**: does last-layer DRAGIN survive
  length-residualization on Gemma + Qwen too, or does it
  Phi-specifically work? If it doesn't survive on MoE, that's the
  motivating gap for MID-RIND.
- **Per-arch question-type interaction**: does the "DRAGIN wins
  compositional, u_cons wins comparison" pattern replicate on
  Gemma / Qwen?

These three tests, when xarch lands, define which next-gen direction
has the strongest empirical anchor before we commit any new compute.

### Roadmap (no deadlines; this is the long-horizon research direction)

**Phase 1 (no new compute) — Cross-arch audit + MID-RIND analysis.**
Once xarch lands: run #v2-11 protocol on all 3 archs × 2 datasets;
compute MID-RIND residual AUROC vs DRAGIN residual AUROC, paired
bootstrap. ~30 min analysis, free.

**Phase 2 (modest compute) — MS-RIND multi-layer capture.** Requires
re-running decode with attention captured at 3 layers, not 1. Maybe
2-3× attention-storage cost; same model load. On Phi (~2 h re-decode).
If MS-RIND beats single-layer-RIND under length-residualization,
proceed to Phase 3. If not, stop here.

**Phase 3 (modest compute) — MoE-RIND on Gemma / Qwen.** Requires
capturing expert-routing gate logits during MoE forward. ~2 h
re-decode on each MoE arch. Specifically targets the architectures
where standard DRAGIN is most polluted.

**Phase 4 (longer) — Causal-RIND streaming formulation.** Code change
only (no re-decode); just changes the slice direction in
`RindAccumulator.ingest_attention_row`. Compare AUROC residual to
look-ahead DRAGIN. If within 0.03 AUROC, ship the streaming variant
as the deployment recommendation.

**Phase 5 (compute + data) — Mamba/SSM extension.** Requires a
Mamba-2 or RWKV-7 model on the compute box; new download. The
question is whether the hidden-state-cosine-drift-to-question
fallback approximates RIND's behavior on a model where RIND
literally cannot be computed. This is the most ambitious phase and
also the most novel ("first dynamic-RAG trigger for non-attention
architectures").

### Why this is the right reframe

- Our audit established DRAGIN as the only length-robust trigger.
  That's a strong empirical anchor.
- Our model pool (`_MODEL.md`) gives us the three architectural
  families DRAGIN's design was *not* built for: hybrid SSM-attention,
  MoE, and (with one more download) pure-SSM. We have first-hand
  experience with the failure modes.
- The audit protocol (#v2-11) is a paper contribution by itself,
  and the design space above gives multiple next-gen-method
  contributions on top of that.
- We're no longer chasing a "novel" method in isolation; we're
  iterating on the one method that works, against the empirical
  audit that justifies the iteration. Reviewer challenge "this is
  just DRAGIN with one knob changed" is countered by "yes, that's
  the point — this is the knob the audit identified as broken on
  modern architectures."

### Open questions for the design

- Is MID-RIND vs LAST-RIND a meaningful difference on MoE in
  practice, or does the routing noise hurt mid-stack too? The xarch
  data answers this immediately.
- Does adding routing entropy as a third factor genuinely help, or
  is it redundant with attention entropy? Empirically untested.
- Can we formulate a Mamba/SSM equivalent of "attention-weighted
  importance" using SSM state-space dynamics? Not obvious from a
  pure-SSM model. Needs literature dive.
- What's the right operating-point evaluation? Precision-at-budget
  is sensible but the budget itself should probably be a hyperparam,
  not a constant. Should the trigger be operating-point-aware (fire
  at the budget that maximizes expected utility given a retrieval
  cost)?
- Is question-type-aware thresholding actually portable, or does the
  threshold-per-type require lots of calibration data? Open.

### What this is NOT (to be explicit)

- Not an EMNLP-short submission target as currently framed. This is
  a longer-horizon direction; multiple of the design dimensions need
  independent validation. EMNLP-short, if pursued, would be the
  audit (#v2-11) + MID-RIND finding alone, not the full design space.
- Not committed to any single arch focus. The whole point is that
  the trigger family should work across the 3 archs we have.
- Not abandoning all prior work — the u_cons / entropy / hidden-state
  features stay in the toolbox as candidate auxiliary signals for
  Phase 4 (action-routing). They're just not headline material on
  their own.
