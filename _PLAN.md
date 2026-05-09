# _PLAN.md

> Build the smallest working version first, then add one capability per step.
> Target: Linux + NVIDIA / CUDA. Mac/Metal deferred.
> Each phase ends with something we can **see** in the terminal.

## Thesis (one sentence)

On edge MoE, the cheapest place to detect that a reasoning model is about to hallucinate is in its **hidden states at logical step boundaries** — not in its output tokens, not in its `<think>` syntax, not in its self-verification text.

---

## Guiding rules

- One concept per phase. Split if it grows.
- Every phase ends with a visible artifact: console output, a file, or a number.
- Pinned versions, scripted installs, `sysinfo` on every run.
- Memory is *measured*, not feared. KV-cache pressure is an experimental variable, not an obstacle.
- Update `_AGENT.md` after every phase: what worked, what broke, what changed.

---

## Engine

Hugging Face `transformers` + `accelerate` + `bitsandbytes` 4-bit NF4 on CUDA. Single supported path. Hidden states via `output_hidden_states=True` and forward hooks on a configurable layer subset.

---

## Tentative landmines (unverified — re-evaluate as each phase touches them)

> Hypotheses to falsify, not adopted truths. Delete an entry once the phase that touches it shows it doesn't bite.

- **L1 — MoE routing variance ≠ uncertainty variance (Phase 2.3).** The router selects different experts per token, so naive mean-pool of last-layer hidden states absorbs "expert-switching noise" unrelated to reasoning uncertainty. Possible mitigations: (a) hook a *pre-routing* mid-stack layer; (b) track router logits / expert-distribution entropy alongside hidden states. Verify on day one with the Phase 0.5 smoke check.
- **L2 — `bitsandbytes` NF4 activation trap (Phase 2.1).** NF4 compresses weights; forward-pass activations are still bf16/fp16. If the forward hook keeps captured hidden states attached to the autograd graph and on-device, a 5k-token generation OOMs the A40. Hooks must `.detach().to("cpu", dtype=torch.float32)` immediately on capture, or stream to a ring buffer. Budget VRAM for capture explicitly in Phase 1.2.
- **L3 — Simulation strawman risk (Phase 3.2).** Heuristic GRIP simulation (entropy spike + discourse-marker prior) is not GRIP. Paper framing must be: "we beat *heuristic semantic routing* (ReaLM-Retrieve sim); GRIP-sim is an upper-bound proxy for *what if semantic detection were perfect*." Do not claim a win over GRIP itself.
- **L4 — Day-one MoE noise check (Phase 0.5 smoke).** Compute hidden-state L2-norm difference between a highly predictable continuation (`"The capital of France is "` → `Paris`) and an ambiguous prefix at the same position. If diff is dominated by router-switching noise on the MoE generator, L1 is real and Phase 2.3 needs a redesign before more code is written.

---

## Repo layout (target)

```
recollection.local/
├── _PLAN.md                # this file
├── _AGENT.md               # experimental log
├── README.md
├── environment.yml         # conda env: python, cmake, pinned pip deps
├── scripts/
│   ├── setup.sh            # linux+cuda only; conda + torch + env
│   ├── download_models.sh  # huggingface-cli; revisions captured at first run
│   ├── sysinfo.py          # OS, RAM, GPU + VRAM, torch/transformers/bnb versions
│   ├── 00_smoke.py         # load + 32-token gen, hidden-state shape, L4 probe
│   ├── 01_kv_pressure.py   # 5k-token uninterrupted gen, VRAM checkpoints
│   ├── crv/                # Concept Representation Vector implementations
│   ├── 02_crv_probe.py     # hidden-state hook + step boundary + CRV
│   ├── 03_baselines.py     # ReaLM-Retrieve sim, GRIP sim, random
│   └── 04_showdown.py      # earliness + drift + cost across all gates
├── models/                 # gitignored
├── data/                   # gitignored
├── runs/                   # JSONL logs per run (gitignored)
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

**Goal:** clean A40 box reaches a HF + bnb + torch-CUDA stack that loads the MoE generator at 4-bit and exposes hidden states.

- [ ] **0.1** `setup.sh` conda probe handles `/opt/miniforge3/bin/conda` (vast.ai-style boxes); env builds from `environment.yml`; torch installs from cu124 wheel index; lockfile written.
- [ ] **0.2** `sysinfo.py` reports CUDA visibility, device name, total/free VRAM, torch + transformers + bnb + accelerate versions.
- [ ] **0.3** `download_models.sh` pulls `microsoft/Phi-4-mini-instruct` by default (small smoke target) and `Qwen/Qwen3.6-35B-A3B-Instruct` with `all`. Revisions captured.
- [ ] **0.4** `00_smoke.py` loads at 4-bit NF4, generates 32 tokens with `output_hidden_states=True`, prints tok/s, peak VRAM, hidden-state tensor shape.
- [ ] **0.5** **L4 landmine probe inside `00_smoke.py`:** predictable vs ambiguous prefix → hidden-state L2 norm + next-token entropy. Predictable should be visibly *lower* hidden-state variance than ambiguous and *lower* entropy. If MoE routing noise dominates, L1 is real and Phase 2 design needs a rethink.

**Visible artifact:** A40 baseline appended to `_AGENT.md` (tok/s, peak VRAM, hidden-state shape, L4 numbers).

---

## Phase 1 — KV-pressure baseline

**Goal:** quantify the cost the latent-boundary gate is supposed to save. Produces the paper's intro figure.

- [ ] **1.1** Generator: Qwen3.6-35B-A3B at 4-bit. Reflector: Phi-4-mini.
- [ ] **1.2** `01_kv_pressure.py`: single 5,000-token uninterrupted generation. Log VRAM at each 500-token checkpoint, latency per token, total wall-clock.
- [ ] **1.3** Repeat 1.2 for three prompt classes (factual, multi-hop, deliberately ambiguous). `runs/phase1_kv_pressure.jsonl`. Plot VRAM-vs-tokens and tok/s-vs-tokens.

**Visible:** `figs/kv_pressure_baseline.png` + three numbers (peak VRAM, tok/s at 500, tok/s at 5000).

---

## Phase 2 — Latent-boundary engineering (the contribution)

**Goal:** detect "model is about to hallucinate" from hidden states **before** the next token is sampled.

- [ ] **2.1** Hidden-state hook: forward-hook the generator to capture configurable-layer-subset hidden states at every step. Throughput regression budget ≤5%. If we blow it, fall back to capturing every Nth step and document. Hooks use `.detach().to("cpu", float32)` on capture (L2 mitigation).
- [ ] **2.1b** Keep CRV / step-boundary / trigger code engine-agnostic: pass tensors through a small adapter so HF/bnb internals don't leak into trigger logic.
- [ ] **2.2** Step-boundary detector: define "logical step" operationally — sentence-end inside `<think>`, numbered list item, or fixed-stride window of N tokens. Sweep N. Tag every step with a boundary label.
- [ ] **2.3** **CRV (Concept Representation Vector):** per-step summary of hidden state. Start with mean-pool of last-layer hidden states across the step, optional projection. Track running variance and cosine drift between consecutive step CRVs. Variants in `scripts/crv/`. **If L1 fires from Phase 0.5, swap last-layer for a pre-routing mid-stack layer here.**
- [ ] **2.4** Trigger rule: fire when CRV variance OR drift crosses an empirical threshold. Calibrate offline on the Phase 4.1 dataset.
- [ ] **2.5** **Stop-gate sanity probe:** known-correct trace vs known-hallucinated trace from Phase 1 logs; confirm separation > 0 on at least one CRV variant. **If no variant separates, stop and rethink before Phase 3 — paper has no contribution.**

**Visible:** `runs/phase2_crv.jsonl` per-step `(token_idx, hidden_norm, variance, drift, fired)`; one figure overlaying variance traces on correct vs hallucinated runs.

---

## Phase 3 — SOTA baselines (the comparison set)

**Goal:** make the latent gate beatable in principle and beaten in practice.

- [ ] **3.1** **ReaLM-Retrieve simulation:** trigger on Reasoning Step Uncertainty Score (small verifier's per-step score) AND on discourse markers (`let me verify`, `actually`, `I need to check`). Thin wrapper around the same generator.
- [ ] **3.2** **GRIP simulation:** heuristic control-token classifier; fire retrieval when classifier predicts a `<retrieve>` token would appear next. Without trained weights, simulate via (next-token entropy spike + discourse-marker prior). Document the simulation gap explicitly (L3).
- [ ] **3.3** **Random-rate baseline:** fire at the same per-step rate as the average of the three trained gates. Log seed.
- [ ] **3.4** Unified trigger schema across all four gates: `{step_idx, token_idx, gate_name, fired, score, reason}` → `runs/phase3_triggers.jsonl`.

**Visible:** four gates running on the same generation traces, one CSV summary.

---

## Phase 4 — Empirical showdown

**Goal:** prove the latent gate fires earlier and prevents trajectory drift better than ReaLM-Retrieve / GRIP / random — or document where it doesn't.

- [ ] **4.1** Multi-hop dataset: ~200-question slice of HotpotQA-distractor or MuSiQue. Commit hash pinned at ingest.
- [ ] **4.2** **Earliness metric:** for each gate, log token index of first fire on hallucinated traces.
- [ ] **4.3** **Trajectory-drift metric:** final-answer F1 / EM conditioned on (gate fired at step k vs k+1 vs not fired). Define drift as monotonic answer-quality decay with delayed firing.
- [ ] **4.4** Ablations: (a) variance vs drift vs both; (b) layer subset (last only vs middle); (c) step-boundary definition (sentence vs fixed-stride).
- [ ] **4.5** Cost: per-gate total wall-clock and peak VRAM across the 200-question run. Pareto plot in `figs/efficiency_pareto.png`.

**Visible:** the paper's headline figure (earliness × drift × cost) and `runs/phase4_showdown.csv`.

---

## Definition of done for the prototype

One command on CUDA (`scripts/setup.sh && scripts/download_models.sh all && python scripts/04_showdown.py --questions 200`) prints:

1. System + memory baseline.
2. Per-gate trigger trace and earliness.
3. Trajectory-drift table.
4. Pareto cost summary.
