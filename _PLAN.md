# _PLAN.md

> Build the smallest working version first, then add one capability per step.
> Target: Linux + NVIDIA / CUDA. Mac/Metal deferred.
> Each phase ends with something we can **see** in the terminal.

## Thesis (one sentence)

In agentic LLM reasoning at long context, **memory pressure is being solved by architecture (linear attention, hybrid KV designs); reasoning-trajectory drift is not** — and the cheapest place to detect that drift before it produces a hallucination is in the model's **hidden states at logical step boundaries**, *not* in its output tokens, its `<think>` syntax, or its self-verification text. Latent-state gating is a **hardware-native** trigger that fires earlier than text-parsing alternatives, on every architectural class we test.

> *Reframe history (collapsed):*
>
> - **Pivot 1 (pre-Phase 1):** "edge / strict VRAM" → "KV-cache pressure under agentic long context". See `_AGENT.md` 2026-05-09 #3b.
> - **Pivot 2 (Phase 1 finding):** Phase 1 measured the headline KV curve as essentially flat on Qwen3.6-A3B (hybrid linear/full attention, ~80 MB total VRAM growth across 5 000 tokens, see `_findings.md` Phase 1 + `_AGENT.md` 2026-05-09 #4). KV-pressure-as-headline-cost no longer holds.
> - **Pivot 3 (current, post-Phase 1, "cognitive efficiency"):** From "memory is the cost" to "**memory is being solved; reasoning-trajectory drift is the open frontier**." This is the EMNLP-2026-era framing: contrast a Dense SLM, a Standard MoE, and a Hybrid-Linear MoE in the same edge-VRAM theater (6–20 GB), show that all three exhibit hidden-state drift before hallucination regardless of where their KV curves fall, and argue that a latent-state gate is hardware-native and architecture-agnostic. See `_AGENT.md` 2026-05-09 #5.

---

## Guiding rules

- One concept per phase. Split if it grows.
- Every phase ends with a visible artifact: console output, a file, or a number.
- Pinned versions, scripted installs, `sysinfo` on every run.
- Memory is *measured*, not feared. KV-cache pressure is an experimental variable, not an obstacle.
- Update `_AGENT.md` after every phase: what worked, what broke, what changed.

---

## Engine

Hugging Face `transformers` + `accelerate` + `bitsandbytes` (NF4 where it helps, bf16 native where it's faster — see `_AGENT.md` 2026-05-09 #3b/#4 for which-when). Single supported framework. Hidden states via `output_hidden_states=True` and forward hooks on a configurable layer subset. Generator pool (Phase 1.4 + downstream): three models bracketing the 2026 architectural design space — **Dense SLM** (Phi-4-Mini, ~6 GB bf16), **Standard MoE** (TBD, ~16 GB at NF4 — see Phase 1.4 model-availability checklist), **Hybrid-Linear MoE** (Qwen3.6-35B-A3B, NF4 hybrid for the edge-theater experiments / bf16 native for the on-A800 deep-dive runs). All three should fit in a 6–20 GB consumer-VRAM theater so the comparison reads as edge-realistic.

---

## Tentative landmines (unverified — re-evaluate as each phase touches them)

> Hypotheses to falsify, not adopted truths. Delete an entry once the phase that touches it shows it doesn't bite.

- **L1 — MoE routing variance ≠ uncertainty variance (Phase 2.3).** The router selects different experts per token, so naive mean-pool of last-layer hidden states absorbs "expert-switching noise" unrelated to reasoning uncertainty. Possible mitigations: (a) hook a *pre-routing* mid-stack layer; (b) track router logits / expert-distribution entropy alongside hidden states. Verify on day one with the Phase 0.5 smoke check.
- **L2 — `bitsandbytes` NF4 activation trap (Phase 2.1).** NF4 compresses weights; forward-pass activations are still bf16/fp16. If the forward hook keeps captured hidden states attached to the autograd graph and on-device, a 5k-token generation OOMs the A40. Hooks must `.detach().to("cpu", dtype=torch.float32)` immediately on capture, or stream to a ring buffer. Budget VRAM for capture explicitly in Phase 1.2.
- **L3 — Simulation strawman risk (Phase 3.2).** Heuristic GRIP simulation (entropy spike + discourse-marker prior) is not GRIP. Paper framing must be: "we beat *heuristic semantic routing* (ReaLM-Retrieve sim); GRIP-sim is an upper-bound proxy for *what if semantic detection were perfect*." Do not claim a win over GRIP itself.
- **L4 — Day-one MoE noise check (Phase 0.5 smoke).** Compute hidden-state L2-norm difference between a highly predictable continuation (`"The capital of France is "` → `Paris`) and an ambiguous prefix at the same position. If diff is dominated by router-switching noise on the MoE generator, L1 is real and Phase 2.3 needs a redesign before more code is written.
- **L5 — Phase 4.4 calibration leakage.** The Phase 2 finding that Qwen relies on drift while Phi/Gemma rely on variance means thresholds *must* be set per architecture (Phase 4.4 ablation d). If those thresholds are fit on the same traces used to report Phase 4.3's trajectory-drift F1 numbers, reviewers will (correctly) call data leakage. Mitigation: **the 200-question HotpotQA slice ships pre-partitioned 60/20/20 (calibration / validation / test) inside `data/hotpotqa_distractor_200.jsonl`** with `slice_sha256` pinned in the manifest. Calibration is the only split allowed to set thresholds; test is opened once at write-up. Same partition shared across all three architectural classes so cross-arch comparisons hold. Enforced in `scripts/04_dataset.py`; must be respected by every downstream Phase 4 script.

---

## Repo layout (target)

```
recollection.local/
├── _PLAN.md                # this file (the plan, forward-looking)
├── _AGENT.md               # experimental log (mechanics, backward-looking)
├── _draft.tex              # paper-prose corpus (superset for the 4-page final; sideways-looking)
├── _findings.md            # hand-curated paper-quality findings doc (updated only on user request)
├── README.md
├── environment.yml         # conda env: python, cmake, pinned pip deps
├── scripts/
│   ├── setup.sh            # linux+cuda only; conda + torch + env
│   ├── download_models.sh  # hf cli; revisions captured at first run; phase1.4 / phase1.4-full subcommands
│   ├── snapshot.sh         # package the repo (excl. models) into a single tar.gz at end of every phase
│   ├── sysinfo.py          # OS, RAM, GPU + VRAM, torch/transformers/bnb versions
│   ├── 00_smoke.py         # load + 32-token gen, hidden-state shape, L4 probe
│   ├── 01_kv_pressure.py   # 5k-token uninterrupted gen, VRAM checkpoints (Phase 1)
│   ├── plot_phase1_4.py    # bracketing plot across 3 architectures
│   ├── crv/                # CRV adapter package (hooks, boundaries, vectors, triggers)
│   ├── 02_crv_probe.py     # hidden-state hook + step boundary + CRV (Phase 2)
│   ├── plot_phase2.py      # per-arch drift / variance traces + signal summary
│   ├── 03_baselines.py     # ReaLM-Retrieve sim, GRIP sim, random  (Phase 3, NOT YET WRITTEN)
│   └── 04_showdown.py      # earliness + drift + cost across all gates (Phase 4, NOT YET WRITTEN)
├── models/                 # gitignored
├── data/                   # gitignored
├── runs/                   # JSONL + log + sysinfo + summary + CRV .npz; tracked in git EXCEPT *.npz
└── figs/                   # plots (gitignored, regenerable)
```

---

## Reproducibility contract

1. **Install path is scripted.** `scripts/setup.sh` confirms Linux+CUDA, builds the conda env, installs torch from the cu124 wheel index, and writes `runs/env_lockfile.txt` with `pip freeze` for exact replay. Non-CUDA hosts fail loudly with a clear message.
2. **Versions are floor-pinned.** `environment.yml` uses `>=` pins for the live stack. Exact resolved versions live in `runs/env_lockfile.txt` after a clean install; tighten there if drift starts hurting.
3. **Models are scripted with revisions captured.** `scripts/download_models.sh` writes resolved commit SHAs to `models/<name>.revision` on first download. After verifying a run is good, hardcode the SHA into the script's `REV_*` vars.
4. **Every run logs sysinfo.** `scripts/sysinfo.py` → OS, kernel, CPU, RAM, GPU + VRAM, backend, torch/transformers/bnb/accelerate versions, current process RSS. Phase scripts call it at startup.
5. **Every run logs memory.** Before model load, after model load, after first generation, at every 500-token checkpoint during long generations.

---

## Phase 0 — Environment + smoke

**Goal:** clean GPU box reaches a HF + bnb + torch-CUDA stack that loads the MoE generator and exposes hidden states.

> Hardware update (2026-05-09): Phase 0 closed on A40 46 GB with a `--no-quant` bf16 + CPU-offload fallback (NF4 path blocked by upstream bnb bugs on Qwen3.6-MoE; see `_AGENT.md` 2026-05-09 #2c). Re-baselined on **A800 PCIe 80 GB** (`_AGENT.md` 2026-05-09 #3b): Qwen3.6-35B-A3B loads fully on GPU at bf16 native (~~68 GB VRAM, 12.6 tok/s, ~11 GB headroom for KV / activations). NF4 hybrid path also works (~~65 GB, 8.9 tok/s) but is slower and noisier; bf16 native is canonical for Phase 1+.

- **0.1** `setup.sh` conda probe handles `/opt/miniforge3/bin/conda` (vast.ai-style boxes); env builds from `environment.yml`; torch installs from cu124 wheel index; lockfile written.
- **0.2** `sysinfo.py` reports CUDA visibility, device name, total/free VRAM, torch + transformers + bnb + accelerate versions.
- **0.3** `download_models.sh` pulls `microsoft/Phi-4-mini-instruct` by default and `Qwen/Qwen3.6-35B-A3B` (no `-Instruct` suffix repo exists; chat template bundled) with `all`. Revisions captured + pinned in script.
- **0.4** `00_smoke.py` loads + generates 32 tokens with `output_hidden_states=True`, prints tok/s, peak VRAM, hidden-state tensor shape. Phi-4-mini at 4-bit NF4 ✓ (replicates within noise on A800). **Qwen3.6-35B-A3B canonical path on A800: bf16 native fully on GPU (`--no-quant --gpu-budget 76GiB`), 12.6 tok/s @ ~68 GB VRAM**, no upstream-bug exposure. NF4 hybrid (Linears NF4 + grouped experts bf16) also loads cleanly at 80 GB but is 42 % slower and adds quantization noise on the L4 probe — kept as a fallback for smaller-VRAM cards.
- **0.5** **L4 landmine probe inside `00_smoke.py`** ran on both Phi (dense) and Qwen (MoE). On the MoE generator at bf16 native (A800): hidden L2 Δ ~0.5 % (0.563 / 111.4), entropy Δ 1.72 nats,  `Paris` p=0.554 — replicates the A40 bf16+CPU baseline almost exactly. **L1 confirmed across both boxes: MoE noise *does* degrade the L2 signal further than dense (Phi: ~1.1 %, Qwen: ~0.5 %).** Phase 2.3 must not start from L2-of-mean-pool.
- [info] **0.5b** Pre-quantize-on-disk scaffold (`tools/quantize_qwen.py`) preserved for future smaller-VRAM scenarios. Blocked by grouped MoE expert tensors stored as 3-D `nn.Parameter` inside `Qwen3_5MoeExperts`. Detailed in `_AGENT.md` (2026-05-09 #2b).
- [info] **0.5c** Upstream NF4 blockers (transformers#43032 streaming-load regression + bnb's missing grouped-MoE-experts pattern) are *still present* on the installed stack but no longer bite at 80 GB VRAM. Patch text + analysis preserved in `_AGENT.md` (2026-05-09 #2c) for revisit if Phase 4 ever wants to report numbers on consumer GPUs.

**Visible artifact:** A800 baseline + Qwen MoE bf16-native smoke + L4 probe results in `_AGENT.md` 2026-05-09 #3b. Logs in `runs/smoke_qwen_bf16_a800.log`.

---

## Phase 1 — KV-pressure baseline

**Goal:** quantify the cost the latent-boundary gate is supposed to save. Produces the paper's intro figure.

- **1.1** Generator: Qwen3.6-35B-A3B at bf16 native fully on cuda:0 (canonical path on A800 80 GB; see `_AGENT.md` 2026-05-09 #3b). Reflector deferred to Phase 2.5+ (not used by Phase 1 KV-pressure measurement). Chat template threaded via `tokenizer.apply_chat_template(..., add_generation_prompt=True)` inside `scripts/01_kv_pressure.py`.
- **1.2** `scripts/01_kv_pressure.py`: single 5,000-token uninterrupted greedy generation with manual loop + explicit `DynamicCache`. VRAM, peak VRAM, KV growth, rolling tok/s, instantaneous tok/s, prefill latency, decode-step1-vs-stepN latency, EOS-seen-at all logged at every 500-token checkpoint.
- **1.3** Three prompt classes (factual / multi-hop / ambiguous) × 5 000 tokens. Outputs: `runs/phase1_kv_pressure.jsonl` (30 checkpoint rows), `runs/phase1_kv_pressure_summary.jsonl` (3 summary rows), `figs/kv_pressure_baseline.png`. **Headline:** mean ~12.85 tok/s, **total KV growth 132–141 MB across 5 000 tokens**, decode-step1→stepN slowdown 0.96–0.98× (effectively flat). Detailed in `_AGENT.md` 2026-05-09 #4.
- [x] **1.4** **Three-way architectural-class baseline — DONE.** Ran `scripts/01_kv_pressure.py` on three generators bracketing the 2026 attention-design spectrum, sequentially on the A800 at bf16 native. Detailed results in `_AGENT.md` 2026-05-09 #6; bracketing plot in `figs/phase1_4_kv_growth.png`. Headline numbers (mean across 3 prompt classes, KV growth at step 5000): **Phi-4-mini 128 KB/tok / 643 MB / 47 tok/s; Gemma 4 26B A4B 221 KB/tok / 1 105 MB / 13 tok/s; Qwen3.6-35B-A3B 27 KB/tok / 135 MB / 13 tok/s.** Surprise: sliding-window attention bounds *compute* but *not memory* in HF's `DynamicCache` (no auto-eviction), making Gemma 4's KV slope steeper than Phi's. Bracketing argument now lives in **two dimensions** (compute vs memory pressure), not one.

  *Skipped (parked) — `google/gemma-4-31B-it`* (62.6 GB downloaded, same sliding-window+global attention pattern as 26B A4B). Same architectural class — would be a redundant fourth line on the bracketing plot. Reserved for **Phase 4.4 ablation (d): MoE-vs-dense within Gemma family**, all else held constant.

**Visible ✓ (Phase 1.3):** `figs/kv_pressure_baseline.png` + three numbers (peak VRAM 66 258 MB, tok/s @ 500 ≈ 12.7, tok/s @ 5000 ≈ 13.0). Qwen3.6 detail across 3 prompt classes.

**Visible ✓ (Phase 1.4):** `figs/phase1_4_kv_growth.png` (the headline bracketing plot — three distinct slopes), plus `figs/phase1_4_arch_comparison.png` (raw VRAM panels) and per-checkpoint detail in `runs/phase1_4_*.jsonl`. **Phase 1 closed.**

---

## Phase 2 — Latent-boundary engineering (the contribution)

**Goal:** detect "model is about to hallucinate" from hidden states **before** the next token is sampled — *and prove the signal exists even on architectures where the memory-pressure curve is flat.* Per the cognitive-efficiency pivot (`_AGENT.md` 2026-05-09 #5), Phase 2 is where the paper demonstrates that latent-state drift is the architecture-independent failure mode left over after sliding-window / linear-attention designs have "solved" the memory bottleneck.

- [x] **2.1** Hidden-state hook via HuggingFace `output_hidden_states=True` (deliberately not `register_forward_hook` — public API is uniform across our three architectural classes). `.detach().to("cpu", float32)` on capture (L2 mitigation). ≤3 % throughput overhead at steady state on $5\,000$-token traces; ~11–14 % overhead on 200-token smokes is warmup-dominated.
- [x] **2.1b** Engine + architecture-agnostic adapter (`scripts/crv/`): zero per-arch code. Verified end-to-end on Phi-4-mini (dense GQA), Gemma 4 26B A4B (sliding-window + sparse global), and Qwen3.6-35B-A3B ($3{:}1$ Gated DeltaNet + full).
- [x] **2.2** Step-boundary detector with three strategies (`sentence`, `think_block`, `fixed`); `sentence` swept end-to-end across $3 \times 3 \times 5\,000$ tokens × 2 layer modes. The `think_block` and `fixed` strategies are implemented but not yet swept across the matrix — see Future Work F3 in `_draft.tex`.
- [x] **2.3** CRV (Concept Representation Vector) computed per logical step via mean-pool of selected layers, with cosine drift vs previous CRV and rolling-window L2-norm variance. Both `last_layer` and `mid_stack` modes ran. **L1 fired hard at last-layer on Qwen3.6 (mean drift 0.479, near-saturated); mid-stack mitigation recovered the signal (mean drift 0.235).** Detailed in `_AGENT.md` 2026-05-09 #8 (last-layer) and #9 (mid-stack).
- [x] **2.4** Trigger rule (`CRVTrigger`) implemented with `or`/`and`/`drift`/`variance`/`none` modes. **Thresholds NOT calibrated** — defaults fire ~99 % on the MoE classes and are placeholder-only. Calibration is a Phase 4 task once labelled traces exist.
- [ ] **2.5** **Stop-gate sanity probe** — *deferred to Phase 4.1*. Cannot execute meaningfully without correct/hallucinated trace pairs. The labelled HotpotQA / MuSiQue split assembled in Phase 4.1 is the prerequisite. If at that point the latent-state signal does not separate on every architectural class, the paper has to retreat from "architecture-agnostic." For now, the per-step features and CRV vectors are saved to disk (`runs/phase2_crv_*_{last_layer,mid_stack}.{jsonl,npz}`) so the eventual sanity probe is offline and cheap.

**Visible (Phase 2 ✓):** `figs/phase2_drift_per_arch{,_midstack}.png`, `figs/phase2_variance_per_arch{,_midstack}.png`, `figs/phase2_signal_summary{,_midstack}.png` (six figures total, last-layer + mid-stack matched pairs). Per-step JSONL + CRV `.npz` files in `runs/phase2_crv_*`. Headline numbers in `_AGENT.md` #8/#9 and `_draft.tex` Tables 4 & 5.

---

## Phase 3 — SOTA baselines (the comparison set)

**Goal:** make the latent gate beatable in principle and beaten in practice.

- [x] **3.1** **ReaLM-Retrieve simulation** — `RealmRetrieveSimGate` in `scripts/crv/baseline_gates.py`. Trigger = (per-step RSU score > τ_rsu) OR (discourse-marker matched). RSU stand-in is **per-step mean next-token entropy of the generator** (no trained verifier available); divergence stamped on every row's `simulation_caveats`. Discourse pattern set is curated from ReaLM-Retrieve / GRIP qualitative examples (`let me verify`, `actually,`, `wait,`, `I need to check`, `not entirely sure`, ...).
- [x] **3.2** **GRIP simulation** — `GripSimGate`. Trigger = (boundary-token entropy z-score > τ_z AND discourse-marker matched) OR (boundary-token entropy z-score > τ_strong). Z-score is computed against a 20-step rolling window; stat update happens *after* scoring so a current spike isn't part of its own baseline. L3 simulation gap (no trained `<retrieve>`-token classifier head; entropy z-score substitutes for classifier logit) stamped per row.
- [x] **3.3** **Random-rate baseline** — `RandomGate` with deterministic seed (default `20260509`, logged). Fires at a rate **post-hoc rate-matched** to the average of the three real gates' per-step fire rates over the same trace. `evaluate()` records placeholder rows during the live decode, then `calibrate_rate(...)` + `finalize(...)` materializes Bernoulli decisions after the trace ends. Calibration recipe (mean of per-gate rates, not concatenated rate) stored in summary metadata.
- [x] **3.4** Unified trigger schema across all four gates: `{run_id, model, prompt_class, step_idx, token_idx, gate_name, fired, score, reason, simulation_caveats}` → `runs/phase3_triggers_<model>_<prompt>_<boundary>_<layermode>.jsonl`. Per-step input record (text + entropies + CRV features) saved alongside in `..._steps.jsonl` so Phase 4 can re-derive gate decisions at new thresholds without re-running the generator. Per-trace summary (fire rates, fire counts, gate metadata, seed) in `..._summary.json`.

**Driver:** `scripts/03_baselines.py` re-uses the Phase 2 manual decode loop with two additions on top: per-token next-token entropy capture (~1 ms/decode-step on the A800; effectively free) and per-step text decode. All four gates evaluate online on the same physical trace — no model swap noise (cf. "Things to watch" in `_AGENT.md`).

**Smoke ✓:** Phi-4-mini × factual × 200 tokens, last-layer CRV. 5 boundary steps committed, 20 trigger rows written, schema conformant, L3 caveats stamped on every realm/grip row, random gate calibrated to p=0.267 = mean(0.80, 0.00, 0.00). Throughput 40.5 tok/s — entropy capture overhead is in the noise.

**Visible:** four gates running on the same generation traces, per-trace JSONL + per-step JSONL + summary JSON. Full sweep (3 models × 3 prompt classes × 5000 tokens × 2 layer modes = 18 traces) is the next compute spend; deferred until user kicks it off.

---

## Phase 4 — Empirical showdown

**Goal:** prove the latent gate is **hardware-native** — it fires earlier than text-parsing alternatives (ReaLM-Retrieve, GRIP) because it observes the model's *internal* state at *layer-level granularity*, before a token is even sampled, on **every architecture class** in Phase 1.4. Or document where it doesn't.

> *Hardware-native earliness, defined:* a latent-state trigger fires inside the forward pass at the layer subset of interest, *before* the next token is decoded. A discourse-marker / verifier-score trigger has to wait for the token to be sampled, decoded, and possibly buffered into a phrase before it can match. Latent triggers are therefore "earlier" by at least one decode step — and on long `<think>` blocks, where text-parsing baselines wait for sentence-end inside the chain, can be earlier by tens to hundreds of tokens. This is the rhetorical move the paper is making with "hardware-native."

- **4.1** **First sweep ran 2026-05-09 (200 q × 3 arch × `mid_stack`); REJECTED for headline use** — protocol audit (entry `_AGENT.md` 2026-05-10 #11) showed Qwen ran in default-thinking mode (100 % hit-cap inside `<think>`), Gemma ran in default-no-thinking (the only "honest" trace), Phi ran greedy with no system role and degenerated into repetition loops (75 % hit-cap, EM | hit-cap = 4 %). The three architectures were measured under three different effective protocols. Re-run as 4.1b under Option A. The 4.1 JSONLs (`runs/phase4_qa_*_mid_stack.jsonl`) are preserved as a regression baseline only.
- **4.1b** **Re-run under Option A (unified protocol):** `enable_thinking=False` for Qwen (other two default to no-thinking under their own templates); explicit `{"role": "system"}` message for all three; greedy decoding everywhere with `repetition_penalty=1.05` for Phi only (degeneration guard, documented asymmetry); early-stop on `Answer: <text>\n` line in addition to EOS; `max_new_tokens=1024`. **Always** run `scripts/04b_protocol_smoke.py` first on a new box to verify each model's chat-template render and short-generation behavior matches its current model card (training cutoff < these 2026 releases — model cards are source of truth, smoke is the bridge to actual on-disk template SHAs). Outputs to `runs/phase4_qa_*_mid_stack_v2.jsonl`. Multi-hop dataset is the same ~200-question HotpotQA-distractor slice from 4.1, commit hash pinned at ingest. **The slice ships pre-partitioned 60/20/20 into calibration / validation / test, stratified on `(level × type)`, with the partition baked into the on-disk JSONL `split` column and pinned by `slice_sha256` in the manifest** (see `scripts/04_dataset.py`). The same partition is shared across all three architectural classes — per-architecture splits would invalidate any cross-architecture claim. Phase 4.4 threshold-leakage contract:

  - **calibration (120 q)** is the *only* split allowed to fit per-architecture thresholds (`τ_drift`, `τ_var`, GRIP `τ_z`, ReaLM `τ_rsu`, etc.). Phase 2.3 / 2.4 already noted Qwen relies on drift while Phi/Gemma rely on variance, so thresholds *must* be calibrated per architecture; this is exactly the place that risks leakage if not partitioned.
  - **validation (40 q)** for layer-mode (`last_layer` vs `mid_stack`) and boundary-strategy choices, plus any other model-selection knob.
  - **test (40 q)** opened **once** at write-up; never used to fit anything. Headline 4.2 / 4.3 / 4.5 numbers come from this split only.

  Run trace generation + four-gate evaluation on all three Phase 1.4 architectural classes, all three splits at once (one pass; the splits are then read offline by 4.2 / 4.3 / 4.4 / 4.5 with the appropriate-split filter).
- **4.2** **Earliness metric (hardware-native framing):** for each gate × each model, log the *step index relative to the eventual hallucinated token* of first fire on hallucinated traces. Two cuts: (a) per-architecture-class mean earliness, (b) per-gate distribution of earliness deltas vs the latent gate. The headline claim *was* **"latent gate fires earlier than text-parsing alternatives on every architecture class, including Qwen3.6 where the baseline is hybrid-linear and memory pressure is flat."** **First showdown (2026-05-10 #12) does NOT support this** at fire rate 0.30 (or 0.10 / 0.05). Drift-only latent: **wins on Gemma vs ReaLM/GRIP, loses to random by 1pp**; **loses to ReaLM on Qwen by 5pp**; inconclusive on Phi (n=11 eligible). The latent-OR ablation flips Qwen back to a latent win (+0.272 vs ReaLM +0.260). **Headline must retreat — see `_AGENT.md` 2026-05-10 #12 for the three framing options (narrow / honest negative / methodological retire-ReaLM-baseline). No paper-text past Phase 3 until this fork is resolved.**
- **4.3** **Trajectory-drift metric:** final-answer F1 / EM conditioned on (gate fired at step k vs k+1 vs not fired). Define drift as monotonic answer-quality decay with delayed firing. **First-pass result: signal is real on Qwen** (latest-firing quintile EM=1.0 vs earliest-firing EM=0.667; never-fired EM=0.8 — easy questions don't trip the gate). Weak / inconclusive on Phi/Gemma in this pass.
- **4.4** Ablations: (a) variance vs drift vs both; (b) layer subset (last only vs middle); (c) step-boundary definition (sentence vs fixed-stride); (d) **architectural class** (dense vs sliding-window MoE vs hybrid-linear MoE — does the optimal CRV layer / trigger threshold transfer across classes, or does each architecture need its own calibration?). **(a) + (d) ran together in #13 as the geometry-aware pivot:** the optimal latent mode is architecture-dependent and **does NOT transfer** across classes. Per-arch calibration-driven picks: **Phi → variance, Gemma → OR, Qwen → OR**. Under this geometry-aware policy (`--auto-latent-mode`), latent beats ReaLM/GRIP/random on **2 of 3 architectures** (Gemma narrowly, Qwen decisively). Phi remains an honest loss but no longer wrong-sign. The mid-stack CRV is universal; the heuristic on top of it is not — this becomes the paper's pivot. Detail: `_AGENT.md` #13.
- **4.5** Cost: per-gate × per-model total wall-clock and peak VRAM across the 200-question run. Pareto plot in `figs/efficiency_pareto.png`. **Cost framing:** report not just KV-memory savings (which on Qwen3.6 are minimal, by design) but also (i) **avoided corrective-retrieval round-trips** (each = embedding + index lookup + N regenerated tokens, where N is large for late-firing gates), (ii) **avoided wasted thinking** (tokens generated past the latent gate's fire point on hallucinated traces, weighted by per-token cost on each architecture), and (iii) the latency advantage from the layer-level fire point itself. **First-pass cost (validation, n=40):** Qwen EM=0.706 @ 145 tokens/q, Gemma EM=0.525 @ 199 tokens/q, Phi EM=0.182 @ 160 tokens/q. Qwen wins on both axes simultaneously (most accurate at lowest cost); separate Pareto win independent of the gate analysis.

**Visible (first pass ✓):** `runs/phase4_showdown_v2.csv`, `runs/phase4_showdown_v2_summary.json` (now carries `latent_mode_chosen` + `latent_mode_calib_scores` per arch from #13), `figs/phase4_earliness.png`, `figs/phase4_drift.png`, `figs/phase4_pareto.png`, `notebooks/phase4_showdown_walkthrough.ipynb`. **Post-#13 headline (geometry-aware, auto-latent-mode, fr=0.30, validation):** Phi (variance) beats ReaLM only; Gemma (OR) **beats ReaLM + GRIP + random**; Qwen (OR) **beats ReaLM + GRIP + random** decisively. Paper title pivots toward *"Geometry-Aware Latent Gating in 2026 Edge LLMs."* Bootstrap CIs + Phase 4.6 intervention pilot are the remaining gates before paper-text edits.

- **4.5b** **Sensitivity: Phi-only rerun with `min_steps=2`** — **DONE (2026-05-10).** n_eligible 11 → 18; latent gate Δfrac improves (−0.060 → +0.273) but **still loses to ReaLM (+0.519) and random (+0.430)**; mechanism evidence holds (`never_fired` quintile EM=0.875). Phi reclassified from INCONCLUSIVE to **HONEST LOSS**. Detail in `_AGENT.md` #12 §4.5b.
- **4.5c** **Sensitivity: Gemma system-role ablation** — **DONE (2026-05-10).** EM 0.500 → **0.570** when system text moves back into user message, recovering the 4.1 number almost exactly. **The 4.1b EM dip was protocol-induced, not noise.** Operational fix locked in: **`inline_system` becomes a per-model field in `MODEL_PROTOCOL` — `True` for Gemma, `False` for Phi and Qwen.** Showdown on the 4.5c traces: latent gate Δfrac improves to +0.060 (from +0.036 in 4.1b) but random also improves to +0.084, so latent still loses to random by 2.4pp on Gemma at fr=0.30. **Headline does not flip; mechanism evidence sharpens** — Gemma latent quintile EM gap widens to 50pp (q1=0.375 → q4=0.875). ReaLM Δfrac flipped from −0.130 to −0.057 across the protocol change, **strengthening Option C** (ReaLM-Retrieve simulator is protocol-fragile and should be retired/steelmanned). Detail in `_AGENT.md` #12 §4.5c.
- **4.6** **Intervention arm (NEW; gated on A/B/C resolution):** pause-and-correct on gate fire, measure EM lift over no-intervention baseline. If the latent gate produces an EM lift even when its earliness Δfrac ties random, *intervention utility* (not earliness) becomes the headline metric. This is the strongest reframing if Option B (honest negative on earliness) wins the writeup fork.

---

## Futures (deferred from current critical path)

> Logged here so thejudy don't get lost. Not blocking Phase 1–4. Status reflects **how strongly we want them to land**, not when.

- **F1 — Consumer-GPU validation arm (SHOULD happen).** Pull `unsloth/Qwen3-30B-A3B-bnb-4bit` (or whichever pre-baked NF4 snapshot has working grouped-MoE-experts). Run the Phase 1 KV-pressure curve and a Phase 4 showdown subset on it, *on the same A800 box* — peak VRAM during the run will land in the 18–22 GB range, demonstrating the same code + same gates fit a 24 GB consumer GPU. Turns the library-bug story (`_AGENT.md` 2026-05-09 #2c) from "we couldn't" into "we did, and yes upstream still has gaps." Adds 1 day, materially strengthens the paper's "works at the regime we claim" credibility.
- **F2 — `scripts/_compat.py` runtime monkey-patch (if time allows).** Package the `transformers#43032` 2-line patch (preserved in `_AGENT.md` 2026-05-09 #2c) as a runtime monkey-patch so it survives env recreation without a hand-edit of the installed `core_model_loading.py`. Defensive only; not needed by any current critical-path script (bf16 native on A800 doesn't exercise the broken path; F1's Unsloth snapshot ships its own NF4 layout). Useful insurance the moment we want NF4 of Qwen3.6 on a smaller card.
- **F3 — Option B re-run: all-thinking-ON arm (COULD happen).** Mirror Phase 4.1b but enable thinking on Qwen (default) and Gemma (`enable_thinking=True` + `<|think|>` in system prompt). Phi has no thinking mode and falls out of the comparison — either skip Phi for this arm (clean 2-way MoE comparison) or document the asymmetry. Requires `max_new_tokens≥2048` (Qwen's `<think>` blocks routinely run 1k+ tokens on multi-hop), so ~3-4× the Option A compute. **Tells us whether the latent-state gate's earliness/drift signal is invariant to thinking mode** — useful evidence either way. Defer to after the headline 4.1b numbers land. See `_AGENT.md` 2026-05-10 #11 for diagnosis.
- **F4 — Option C re-run: thinking-mode ablation (SHOULD happen if F3 lands).** Run *both* modes side-by-side as a Phase 4.4 ablation cell (e), 6 conditions instead of 3. Two cuts: (i) does the latent gate's earliness ranking vs ReaLM/GRIP hold in both modes? (ii) do per-architecture trigger thresholds calibrated in one mode transfer to the other (a deployability claim — operators don't always know which mode they're serving)? Cleanest "we considered the protocol confound and it's not driving the result" answer for reviewers. Compute: F3 + F1 cost combined; the calibration-leakage contract (L5) extends naturally because the splits are dataset-level, not protocol-level.
- **F5 — Writeup-framing fork resolution (MUST happen before any paper text past Phase 3).** First showdown produced a mixed result (`_AGENT.md` 2026-05-10 #12). Three options on the table: **A** narrow ("on Qwen, latent-OR beats ReaLM by 1pp on n=34"), **B** honest negative ("drift-only latent does not beat heuristic baselines on HotpotQA-distractor; we report and pivot to intervention"), **C** methodological ("ReaLM simulator fires at step 1 of nearly every trace under Option A — retire it as a baseline and re-run with a steelmanned alternative"). 4.5b + 4.5c sensitivities + 4.6 intervention pilot are the inputs to the decision; do not write past Phase 3 in `_draft.tex` until F5 is closed.

---

## Path to EMNLP short paper (post-#13/#14 assessment)

> Trifecta verdict (`_AGENT.md` #14): **we have a defensible short paper.** Title direction: *"Geometry-Aware Latent Gating in 2026 Edge LLMs: One Adapter, Per-Architecture Heuristic."*

### What's in the can (paper-ready, no further compute)

- Phase 1.4 architectural-bracketing plot (free §3 motivation figure).
- Phase 2 mid-stack CRV adapter (≤3 % overhead, 3 architectures).
- Phase 3 four-gate online evaluation harness (latent + ReaLM-sim + GRIP-sim + random).
- Phase 4.1b sweep under unified Option-A protocol with smoke-validator gate.
- Phase 4 #13 geometry-aware result: 2 of 3 architectures beat all baselines incl. random under calibration-driven per-arch mode selection. Mechanism evidence (quintile-EM gap) on all three.

### Critical-path tasks before submission (revised after #15 bootstrap)

1. ~~Bootstrap CIs on per-arch Δfrac.~~ **DONE (#15).** `scripts/04c_bootstrap.py`, `runs/phase4_bootstrap_v2.json`. **Headline numbers to cite verbatim:**
   - **Gemma (OR):** Δfrac = **+0.045 [+0.002, +0.089]**, n=40. **Significantly beats ReaLM (+0.175 ✓) and GRIP (+0.030 ✓)** in paired bootstrap at 95 %. Ties random.
   - **Qwen (OR):** Δfrac = **+0.272 [+0.122, +0.434]**, n=34. Significantly > 0; latent vs random margin **trends positive (P=0.93) but not 95 %-significant** on n=34.
   - **Phi (variance):** Δfrac = **+0.076 [−0.017, +0.167]**, n=11. Inconclusive; **significantly loses to GRIP** (margin CI [−0.252, −0.018]).
2. ~~Rebuild headline figure.~~ **DONE (#15).** `scripts/04d_headline_figure.py` → `figs/phase4_headline_geometry_aware.{png,pdf}` — single-panel arch×gate, calibration-chosen mode hatched, 95 % CI error bars.
3. **Rewrite `_draft.tex` §intro / §method / §results / §conclusion around the geometry-aware pivot.** ~½ day. **Headline framing (post-#15):** *"Geometry-aware latent gating significantly outperforms both text-parsing baselines (ReaLM, GRIP) on the standard MoE (Gemma) at 95 %; the same approach yields a large effect (+0.272) on the hybrid-linear MoE (Qwen) that trends toward beating random but is power-limited at n=34."* Phi sits in §limitations.
4. **Honest §limitations** with four bullets: Phi as honest negative (significantly loses to GRIP); Qwen latent vs random not 95 %-significant on n=34 (option to validate on test split, see below); auto-mode advantage degrades at fr<0.3; Gemma's effect depends on protocol (4.5c shows mode pick flips to variance under inline-system).
5. **(NEW) Test-split validation in §6.** L5 contract permits opening test once, post-decision. Rerun the geometry-aware analysis on test (n=40 per arch) as a single-shot replication check. If Qwen latent-vs-random clears 95 % on the *combined* valid+test (n=80), we can promote that to a primary claim; if not, paper still works on Gemma. Decision must be made *before* opening test — see GO/NO-GO below.

### Out of scope for the EMNLP short (move to future work / next paper)

- **Phase 4.6 intervention pilot** — different paper. Trying to fit it into 4 pages halves both contributions.
- **Steelmanned ReaLM baseline** (Option C) — defensible to leave ReaLM as-is with an honest framing line.
- **F1/F2** consumer-GPU validation, monkey-patch.
- **F3/F4** thinking-mode arm — strong as a follow-up, not headline.
- **Phase 2.5** stop-gate sanity probe — subsumed by #13 geometry-aware ablation.

### Revised Go / no-go gate (post-#15)

The original criterion ("Qwen vs random clears 95 %") **did not clear** (P(>0)=0.931). Revised:

- **GO** (recommended): adopt the softened criterion *"latent gate significantly beats both text-parsing baselines (ReaLM, GRIP) on at least one architecture at 95 %."* **Gemma clears this cleanly** (margins +0.175 and +0.030, both 95 %-significant). Headline becomes the Gemma result; Qwen is the secondary architecture with a "large effect, trends to beat random, power-limited at n=34" framing; Phi is the honest negative.
- **OPTIONAL "primary-Qwen" promotion**: open the test split (n=80 combined valid+test on Qwen) once, post-decision, as the §6 replication check. *Only do this if everything else is locked* — opening test for model selection violates L5.
- **NO-GO** (not recommended): pivot to honest-negative writeup (`_AGENT.md` #12 Option B). Defensible but sacrifices the Gemma significance result.
- **STALL** if the user wants 4.6 intervention in this submission (still adds ~2 weeks; converts short → long).

---

## Definition of done for the prototype

One command on CUDA (`scripts/setup.sh && scripts/download_models.sh all && python scripts/04_showdown.py --questions 200`) prints:

1. System + memory baseline.
2. Per-gate trigger trace and earliness.
3. Trajectory-drift table.
4. Pareto cost summary.

