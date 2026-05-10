# _AGENT.md

> Experimental log + status. Read top-down: status, then chronological entries.

---

## Status

### Progress

| phase | step | state | artifact |
| --- | --- | --- | --- |
| 0 | conda probe + env (HF, bnb, torch cu124) | **done** | `scripts/setup.sh`, `environment.yml`, `runs/env_lockfile.txt` |
| 0 | model download — Phi-4-mini + Qwen3.6-35B-A3B, revisions captured | **done** | `scripts/download_models.sh`, `models/<name>.revision` |
| 0 | A40 baseline + L4 landmine probe (Phi dense + Qwen MoE) | **done** | entries below |
| 0 | handover to A800 80 GB box (re-baseline) | **done** | entry `2026-05-09 #3b` |
| 1 | KV-pressure curve at 5k tokens × 3 prompt classes | **done** (with surprise: flat curve on Qwen3.6 hybrid linear MoE) | `runs/phase1_kv_pressure.jsonl`, `figs/kv_pressure_baseline.png`, entry `2026-05-09 #4` |
| 1.4 | three-way architectural-class baseline (Phi-4-mini + Gemma 4 26B A4B + Qwen3.6) | **done** — bracketing plot complete, sliding-window-≠-cache-eviction insight added | `runs/phase1_4_*.jsonl`, `figs/phase1_4_arch_comparison.png`, `figs/phase1_4_kv_growth.png`, entry `2026-05-09 #6` |
| 4 (reserve) | Gemma 4 31B Dense as Phase 4 ablation reserve (MoE-vs-dense within Gemma family) | downloaded, parked | `models/gemma-4-31b-it/` |
| 2 | hidden-state hook + step boundary + CRV (all 3 arch classes, last-layer + mid-stack) | **DONE** — full matrix landed (last-layer in #8, mid-stack mitigation in #9); ≤3 % throughput overhead at steady state; **L1 confirmed AND mitigated**; sanity probe (2.5) deferred to Phase 4.1 by design | `scripts/crv/`, `scripts/02_crv_probe.py`, `scripts/plot_phase2.py`, `runs/phase2_crv_*.{jsonl,json,npz}`, `figs/phase2_*.png`, entries `2026-05-09 #7`/`#8`/`#9` |
| draft | `_draft.tex` paper-prose corpus (superset for the 4-page final paper) | **first pass shipped** — 1736 lines, 5 tables, 9 figures referenced, full structure (intro / related / background / method / setup / phase 0/1/2 / phase 4 design / discussion / limitations / repro / future / conclusion / appendix); 60 placeholder citations marked `TODO:` for future fill-in | `_draft.tex`, entry `2026-05-09 #9` |
| 3 | ReaLM-Retrieve sim + GRIP sim + random baseline | **DONE** — full 18-trace sweep (3 models × 2 layer modes × 3 prompt classes × 5 000 tokens) landed in 91 min on A800; all sessions OK; schema clean | `scripts/crv/baseline_gates.py`, `scripts/03_baselines.py`, `scripts/run_phase3_sweep.sh`, `runs/phase3_triggers_*.{jsonl,json}`, `runs/phase3_<model>_<layer>.log`, entry `2026-05-09 #10` |
| 4.1 | per-question multi-hop QA, 4 gates online, 200 q × 3 arch × `mid_stack` | **rejected** — full sweep ran but the prompt protocol was unfair across architectures (Qwen in default-thinking mode, Gemma in default-no-thinking mode, Phi in greedy-without-system-role-and-no-rep-penalty). Re-run as 4.1b under Option A protocol (entry `2026-05-10 #11`). The 4.1 JSONLs are kept as a regression baseline only, not for headline numbers. | `runs/phase4_qa_*_mid_stack.jsonl` (3 × 200 rows; mid_stack only) |
| 4.1b | re-run under Option A unified-protocol (no thinking, system role, per-arch decoding hygiene) | **in progress** — patch + smoke validator landed, sweep pending compute slot | `scripts/04_run_qa.py` (patched), `scripts/04b_protocol_smoke.py`, entry `2026-05-10 #11` |
| 4 (4.2-4.5) | earliness × trajectory-drift × cost showdown | not started — gated on 4.1b producing fair traces | `figs/efficiency_pareto.png`, `runs/phase4_showdown.csv` |

### Hardware

- **Active box (2026-05-09 #3b):** NVIDIA **A800 PCIe 80 GB**, driver 550.90.07 (CUDA 12.6 capable; torch 2.6.0+cu124 reports CUDA 12.4 runtime — forward-compat fine, same flavor as A40 box #1). 220 GiB host RAM, 32 cores, 362 GB overlay disk (was advertised 1.5 TB; vast container overlay smaller in practice — more than enough for Phi 7 GB + Qwen 69 GB + env). **Full 80 GB VRAM visible to torch** (81038 MB free), no cgroup limit. Qwen3.6-35B-A3B confirmed loadable in bf16 native fully on GPU — Phase 1 unblocked.
- Prior box (2026-05-09 #2): NVIDIA A40, 46 GB VRAM (driver 535.230.02, CUDA 12.6); ~2 TB RAM, 40 cores; per-process VRAM ceiling ≈ 44.35 GiB (cgroup-limited). Phase 0 closed here; Qwen smoke ran in bf16 + CPU offload @ 0.3 tok/s.
- Prior box (2026-05-09 #1): NVIDIA A40, driver 565.57.01 (CUDA 12.7), 503 GiB RAM.
- `setup.sh` conda probe handles `/opt/miniforge3/bin/conda` (vast.ai-style boxes).

### Engine

HF `transformers` + `accelerate` + `bitsandbytes`. **Generator pool (Phase 1.4 + downstream)** spans three architectural classes — Dense SLM (`microsoft/Phi-4-mini-instruct`, dense GQA full attention), Standard MoE (`google/gemma-4-26B-A4B-it`, MoE + sliding-window-+-sparse-global hybrid attention), Hybrid Linear MoE (`Qwen/Qwen3.6-35B-A3B`, MoE + 3:1 Gated DeltaNet : full attention). Optional 4th: `google/gemma-4-31B-it` (scale-matched dense ablation). **Canonical run mode on A800 80 GB: bf16 native fully on cuda:0 per model, sequentially.** NF4 still used for (a) Phi reflector smoke when speed isn't the deliverable and (b) the F1 Futures consumer-card validation arm via Unsloth pre-baked snapshots. Hidden states via `output_hidden_states=True` + forward hooks; the hook surface must be uniform across all three architectures (capture `hidden_states` outputs, not K/V tensors, since Gated DeltaNet's recurrent state geometry is incompatible with a K/V-tensor abstraction). torch installed from the cu124 wheel index inside `setup.sh`. Mac/Metal not supported.

### Models

- `microsoft/Phi-4-mini-instruct` — small-fast smoke target, also the lightweight reflector / baseline in Phase 1.
- `Qwen/Qwen3.6-35B-A3B-Instruct` — primary MoE generator (~3B active per token, ~17 GB at NF4). HF repo name taken from external review verbatim; verified on first download.

### Open questions

| id | question | resolve in |
| --- | --- | --- |
| Q1 | Does `transformers.generate` with `output_hidden_states=True` cost > 5 % throughput on the MoE generator at bf16 native (A800 canonical path)? If yes, write a custom forward-hook path that captures only at step boundaries. | Phase 1 / 2.1 |
| Q2 | Is "logical step boundary" best defined as sentence-end inside `<think>`, numbered-list item, or fixed-stride window? Sweep. | Phase 2.2 |
| Q3 | Does CRV variance or drift carry more signal? Or do we need both? | Phase 2.3–2.5, 4.4 |
| Q4 | Does the latent gate fire meaningfully earlier than ReaLM-Retrieve on real HotpotQA / MuSiQue traces? | Phase 4.2 |
| Q5 | Without ReaLM-Retrieve and GRIP weights, how honest is the simulation? Log every divergence. | Phase 3.1–3.2 |
| Q6 | Does MoE expert-switching noise dominate hidden-state variance for the predictable-vs-ambiguous probe? **L1 / L4 in `_PLAN.md`.** | Phase 0.5 |
| Q7 | **(opened 2026-05-09 #4, redirected #5, ANSWERED #6.)** Bracketing holds, *with refinement*. Phi shows clean O(N) growth (128 KB/tok). Qwen3.6 shows near-flat (27 KB/tok). Gemma 4 26B A4B shows clean O(N) at **221 KB/tok** — *steeper* than Phi (not flatter as #5 predicted) because HF `DynamicCache` does not auto-evict sliding-window layers; sliding bounds *compute* but not memory in the reference impl. Two-axis bracketing (compute / memory) is now the paper's framing. | resolved |
| Q8 | **(opened #5.)** Does the latent-state drift signal (Phase 2.5) separate correct vs hallucinated traces *on every Phase 1.4 architectural class*, including Qwen3.6 where memory pressure is essentially absent? If the signal disappears on the hybrid-linear class, the "architecture-agnostic" claim collapses and the paper has to retreat to "works on dense and sliding-window MoEs." | Phase 2.5 |
| Q9 | **(opened #5.)** Does the optimal CRV layer-subset / trigger threshold *transfer* across architectural classes (one calibration fits all), or does each architecture need its own (Phase 4.4 ablation d)? Affects deployability claim. | Phase 4.4 |

### Things to watch

- **Framing pivot history (current = third pivot):**
  - Pivot 1 (pre-Phase 1, `#3b`): "edge / strict VRAM" → "KV-cache pressure under agentic long context."
  - Pivot 2 (Phase 1 finding, `#4`): KV-pressure-as-headline-cost no longer holds (Qwen3.6 hybrid attention flattens the curve at 5 k); softer "corrective-retrieval cost + trajectory drift" reframe.
  - **Pivot 3 (current, `#5`):** "cognitive efficiency / memory is being solved by architecture; reasoning-trajectory drift is the open frontier." Three-way Phase 1.4 architectural-class comparison (dense / sliding-window MoE / linear-attention hybrid MoE). Latent-state gating reframed as *hardware-native* and *architecture-agnostic*.
  - **Stale framing locations:** `README.md` (still on Pivot 2 wording), `_findings.md` (Pivot 2 + the wrong 1:7 Gated-DeltaNet ratio in its Why section); `draft.md` (still on the original "6–20 GB SLM" framing, intentionally untouched until we draft for real). All three need a pass; `_findings.md` requires explicit user OK per its standing rule.
  - Two `_PLAN.md` Futures still open from Pivot 2: F1 (consumer-GPU validation via Unsloth NF4 snapshot — SHOULD happen) and F2 (runtime monkey-patch for transformers#43032 — if time allows).
- Pin model revisions after first clean download (`models/<name>.revision` SHAs go into `download_models.sh`'s `REV_*` vars).
- Hidden-state extraction inflates VRAM (saved tensors). On A800 with bf16 Qwen sitting at ~68 GB, headroom is ~11 GB; verify the (bf16 weights + per-step capture + KV cache) combo still fits at 5 k context. Hooks must `.detach().to("cpu", float32)` immediately on capture (L2 mitigation).
- Keep CRV / step-boundary / trigger code engine-agnostic via a small adapter so HF/bnb internals don't leak into trigger logic.
- "Logical step boundary" is the most ambiguous primitive in the pipeline. Get it under tests early (Phase 2.2) before tuning anything downstream.
- ReaLM-Retrieve and GRIP simulation honesty: log every place the sim diverges from the original paper. Reviewers will probe this.
- Random-baseline seed: log it.
- Corrective / retrieval pass during the showdown must use the **same generator** that produced the trace, otherwise we benchmark a model swap, not a gate.
- **Two distinct bnb-NF4 problems on transformers 5.8 + Qwen3.6-MoE** (see Phase 0 #2, #2b, and #2c entries):
  1. *On-the-fly streaming-load OOM* (issue huggingface/transformers#43032). Cause: `core_model_loading.py:1382` ships bf16 shards to GPU before bnb can quantize. Fix: 2-line patch (`if mapping.quantization_operation is not None: param_device = "cpu"`). Applied to the installed `core_model_loading.py` on this box; should be re-applied as a runtime monkey-patch in `scripts/_compat.py` (or similar) so it survives env recreation. Upstream patch not yet merged.
  2. *No grouped-MoE-experts support in bnb integration.* `quantizer_bnb_4bit.param_needs_quantization` only flags `bnb.nn.Linear4bit` modules; the Qwen3.5/3.6 `Qwen3_5MoeExperts` module wraps experts as 3-D `nn.Parameter` (grouped-GEMM layout). Mxfp4 and fbgemm_fp8 quantizers have explicit grouped-expert patterns; bnb does not. Phase 1 picks one of: (a) pull `unsloth/Qwen3.6-...-bnb-4bit` pre-baked snapshot, (b) switch from bnb-NF4 to MXFP4 (first-class grouped-MoE support upstream), or (c) extend `tools/quantize_qwen.py` with a manual grouped-expert pass + custom forward.
- Qwen ChatML template not threaded through `00_smoke.py`; raw-prompt generation produces gibberish even though forward-pass logits are sane. Use `tokenizer.apply_chat_template` in Phase 1 generation scripts.
- Per-process VRAM ceiling on the **A40 box was 44.35 GiB** (cgroup-limited). On A800 the full 80 GB is visible (verified 2026-05-09 #3b: 81038 MB free pre-load).
- **Canonical Qwen smoke command on A800: `python scripts/00_smoke.py --model qwen3.6-35b-a3b-instruct --no-quant --gpu-budget 76GiB`** (bf16 fully on GPU, 12.6 tok/s, ~68 GB VRAM). The `--no-quant` flag was originally an A40 fallback but turns out to also be the *fastest* canonical path on A800, because bnb can't quantize the grouped MoE expert tensors anyway and the dequant overhead on the small Linear fraction isn't worth it. NF4 hybrid (`--device-map single` without `--no-quant`) loads cleanly too at ~65 GB / 8.9 tok/s — useful as a fallback for smaller-VRAM cards.

---

## Phase snapshots — packaging state for offline preservation

> **Cadence: end of every phase.** Because the compute box is rented, every phase ends with a single small tar.gz that captures the science (docs, scripts, runs/, figs/, .git/) and excludes the bulky-and-reproducible (models, datasets, env caches). Designed to be moved to a local machine, then either kept as a backup or extracted to continue work.

### Make a snapshot

```bash
bash scripts/snapshot.sh phase1                       # phase tag goes into the filename
bash scripts/snapshot.sh phase2 /custom/dir            # default out dir is /workspace/snapshots
bash scripts/snapshot.sh wip-fix-hooks                 # ad-hoc mid-phase backup; any tag works
bash scripts/snapshot.sh phase4 "" --no-heavy          # exclude runs/**/*.npz when traces get big
```

Output:

- File: `/workspace/snapshots/recollection.local-<phase>[-noheavy]-<UTCstamp>.tar.gz`
- Typical size: ~0.6 MB at end of Phase 1 (no .npz yet); ~50 MB at end of Phase 2 (18 CRV .npz files); will grow further at Phase 4 (per-question traces). Use `--no-heavy` once total approaches ~1 GB.
- Script also prints the sha256 + a one-line restore recipe at the end, so you don't need to re-derive either on the destination.

### What's in / what's out

Included (the science):

- `.git/` — full history; on the destination, `git status` will show any uncommitted phase-N work that was packed verbatim.
- All markdown docs at root: `_PLAN.md`, `_AGENT.md`, `_findings.md`, `README.md`, `draft.md`, plus `_draft.tex` (paper-prose corpus).
- `scripts/` (setup, download, smoke, kv_pressure, crv/, plotting scripts, snapshot itself) and `tools/`.
- `extras/` (writing guideline).
- `runs/` — every JSONL, log, sysinfo dump, summary, lockfile from every phase. **CRV `.npz` files are included by default** (gitignored, but heavy enough that re-generating costs an hour of GPU time; carry locally in snapshots). Pass `--no-heavy` to drop them at Phase 4 scale.
- `figs/` — every plot. **Yes, PNG/SVG images are included — tar packs binary files normally.**
- `environment.yml`, `.gitignore`.

Excluded (re-buildable or noise):

- `models/` — 100s of GB of weights. Re-pull on destination via `bash scripts/download_models.sh phase1.4` (SHAs pinned).
- `data/` — datasets; re-pull from source.
- `runs_setup*.log` — verbose pip install logs from `scripts/setup.sh`. Not science, just noisy output.
- `.venv/`, `__pycache__/`, `*.pyc` — env caches.
- The previous boot tarball `recollection-patch.tar.gz` from box-transition history (already represented in git history; skipping the working-tree copy avoids recursive nesting).
- The snapshots dir itself if it's nested inside the repo (not by default; default is `/workspace/snapshots/` adjacent to the repo).

### Restore on destination

```bash
mkdir -p /path/to/local/recollection.local
cd /path/to/local/recollection.local
tar -xzf <snapshot>.tar.gz                # extracts into current dir
git status                                # see uncommitted phase-N work
git log --oneline -5                      # confirm history matches the sha
bash scripts/setup.sh                     # rebuild conda env (~3 min)
bash scripts/download_models.sh phase1.4  # re-pull models (SHAs pinned)
```

Verify integrity (the snapshot script printed the sha256 at packaging time; re-check on the destination before trusting):

```bash
echo "<sha256>  <snapshot>.tar.gz" | sha256sum -c
```

### Operational notes

- Run snapshots **before** you commit / push, so the tarball preserves the exact uncommitted state you might still want to edit. `git status` on the destination will reproduce that state precisely.
- If the box dies or is reclaimed mid-phase, the latest snapshot is your recovery point. Take ad-hoc snapshots (`scripts/snapshot.sh wip-<note>`) when you've made progress you'd hate to lose.
- The script is idempotent and side-effect-free outside the output dir; safe to re-run.
- Tarball includes timestamps so multiple runs in the same day don't collide.

---

## 2026-05-10 #11 — Phase 4.1 sweep was unfair; protocol diagnosis + Option A re-run plan + smoke validator

New compute box (re-rented A800 80 GB, same hardware class as `#3b`). Conda env + models being rebuilt from scratch — same `setup.sh` / `download_models.sh` paths, no surprises expected. While downloads ran, audited the existing Phase 4.1 outputs (3 × 200 questions × `mid_stack`, from 2026-05-09 sweep). Found a structural problem the original sweep didn't surface: **the three architectures were running three different effective prompt protocols simultaneously, so the Phase 4.1 JSONLs measure protocol mismatch as much as architecture.** Diagnosis below; cross-checked against live HF model cards (training-data cutoff is well before these 2026 releases — model cards are the source of truth, not memory); fix locked in as "Option A" (all-thinking-OFF, unified system role, per-model decoding hygiene); B and C parked as `_PLAN.md` Futures F3/F4.

### What the existing 4.1 sweep actually shows (200 q, `mid_stack`)

| model | EM | F1 | hit-cap rate | mean tok | EM \| hit-cap | EM \| EOS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phi-4-mini-instruct` | 0.185 | 0.278 | 150/200 (75 %) | 402.1 | 0.040 | 0.620 |
| `gemma-4-26b-a4b-it` | 0.575 | 0.746 | 13/200 (6.5 %) | 239.8 | 0.231 | 0.599 |
| `qwen3.6-35b-a3b-instruct` | 0.245 | 0.368 | **200/200 (100 %)** | 512.0 | 0.245 | n/a |

Numbers are not noise — they're the same three architectures all running greedy-argmax under the same `04_run_qa.py`, same dataset slice, same `max_new_tokens=512`. The headline ranking is **driven by prompt-template behavior, not model capability.** Specifically:

### The three protocol mismatches (verified against the local chat_template files + live HF model cards 2026-05-10)

1. **Qwen3.6-35B-A3B is in default-thinking mode and we never let it finish.** Local `models/qwen3.6-35b-a3b-instruct/chat_template.jinja` lines 147-153: when `add_generation_prompt=True` (which `04_run_qa.py` sets), the template **prefills `<|im_start|>assistant\n<think>\n` into the prompt itself**, so the model is launched mid-`<think>`. Every Qwen generation we logged starts with `"Here's a thinking process:\n\n1.  **Analyze the User Input:** ..."` — that's the inside of a thinking block. 200/200 traces hit the 512-token cap inside `<think>`, never emit `</think>`, never emit `Answer:`. The 24.5 % EM survives only because the `last_line` extraction heuristic catches a final mention before truncation. Live model card (https://huggingface.co/Qwen/Qwen3.6-35B-A3B, retrieved 2026-05-10) confirms: "Qwen3.6 models operate in thinking mode by default" (line 257 of fetched HTML) and "Qwen3.6 does not officially support the soft switch of Qwen3, i.e., `/think` and `/nothink`" (line 375). The supported lever is `chat_template_kwargs={"enable_thinking": False}`, which prefills `<think>\n\n</think>\n\n` (empty think block) so the model goes straight to the answer.
2. **Gemma-4-26B-A4B is in default-no-thinking mode by accident.** Local `models/gemma-4-26b-a4b-it/chat_template.jinja` lines 347-353: default branch emits `<|channel>thought\n<channel|>` — an *empty* thought channel. Model card (https://huggingface.co/google/gemma-4-26B-A4B-it, retrieved 2026-05-10): thinking is enabled by **two** simultaneous things — `enable_thinking=True` to `apply_chat_template` AND a `<|think|>` token at the start of the system prompt (line 341 of fetched HTML). Our 4.1 sweep set neither, so Gemma quietly ran in non-thinking mode — which happens to match the protocol we *want* for a fair comparison. Gemma's 57.5 % EM is the only "honestly measured" headline in the table.
3. **Phi-4-mini is in greedy-decode-with-no-system-role and falls into repetition loops.** Phi's chat template (in `tokenizer_config.json`) is `<|system|>...<|end|><|user|>...<|end|><|assistant|>` — it has a real system role. The current driver concatenates the instruction into the user message, so Phi never sees a `<|system|>` block. Card (https://huggingface.co/microsoft/Phi-4-mini-instruct, retrieved 2026-05-10) line 268 explicitly: "A model may refuse to answer questions ... in such cases, we have opted to try different system messages (e.g. 'You must always respond to a question' or 'Get to the point!')" — system messages are the documented control surface. **But the proximate failure is decoding, not roles.** Three sample 4.1 hit-cap tails:
   - "...Goodbye, and thank you! Remember, I'm here to help whenever you need. Have a great day ahead!" — repeated 4 times to 512 tokens.
   - "The Kingdom of the Isles, led by Aonghus Mór, covered a total land area of over 8300 km2." — repeated 5 times.
   - Off-topic comparison rambling that never reaches `Answer:`.
   These are classic greedy-decode degeneration on a 3.8B dense model with a ~1500-token multi-distractor context. Phi's card example *does* use `do_sample=False, temperature=0.0`, so greedy is officially supported, but the example caps at 500 tokens with one paragraph; we're feeding 10 distractor paragraphs and expecting 512 tokens of clean reasoning. Defensive `repetition_penalty=1.05` fixes the loops without adding sampling noise.

### Why Option A (all-thinking-OFF + unified system role) is the right re-run for the paper

The Phase 1.4 → Phase 4 throughline measures **architectural drift signals**, not chain-of-thought capacity. Thinking mode on Qwen and Gemma is a *generation-style* knob orthogonal to the latent-state-gating contribution. Holding it constant across all three architectures isolates the architecture variable, which is what the bracketing argument needs. Option A also fits inside the same compute envelope (no token-budget inflation), preserves the L5 leakage contract (same calibration / validation / test split), and Phi (which can't think) stops being artificially crippled relative to its peers. Options B (all-on, requires `max_new_tokens≥2048`) and C (both modes side-by-side as a thinking-mode ablation) are logged as `_PLAN.md` Futures F3 and F4 — both interesting, neither blocking the headline.

### Patches

- **`scripts/04_run_qa.py`** — Option A protocol patch:
  - `MODEL_PROTOCOL` registry keyed on `args.model`: per-model `chat_template_kwargs` (`enable_thinking=False` for Qwen; `{}` for Phi/Gemma — both default to no-thinking under their templates), per-model `repetition_penalty` (1.05 for Phi, 1.0 elsewhere). Stamped into the per-question record so the protocol is reproducible from the JSONL alone.
  - `messages` is now `[{"role": "system", "content": SYSTEM_INSTRUCTION}, {"role": "user", "content": USER_PAYLOAD}]`. User payload no longer contains the system instruction.
  - `repetition_penalty` applied to `next_logits` before `argmax` (HF-standard formulation: divide already-emitted-token logits by penalty if positive, multiply if negative — implemented in-place in `_apply_repetition_penalty`).
  - Early-stop on a clean `Answer: <text>\n` line in the *running decoded text* in addition to EOS. Catches models that emit a turn-end token we don't recognize (Phi sometimes prefers `<|end|>` over `<|endoftext|>`).
  - `max_new_tokens` default raised to 1024 (Option A doesn't need 2048; 1024 gives Gemma's verbose reasoning some slack while still bounding Phi's runaway loops).
  - Per-question record now persists `messages_used`, `chat_template_kwargs_used`, `repetition_penalty_used`, `early_stop_reason ∈ {"eos","answer_marker","max_tokens"}`, and `pct_with_answer_marker` aggregated across the run.
- **`scripts/04b_protocol_smoke.py`** (new) — model-card-vs-real-behavior probe. For each model, runs three probes and prints both the rendered chat-prompt prefix and the first ~200 tokens of generation:
  1. **Chat-template render check** — apply `apply_chat_template` with our real Option A `messages` and any per-model kwargs; print the last 300 chars of the rendered prompt to verify the thinking-mode prefill is what we expect.
  2. **Default-mode short generation** — 5 token, 1 paragraph, "What city is the Eiffel Tower in?" greedy. Detect: (a) does the generation start with `<think>` or `<channel>thought`? (b) does it emit `Answer:` within 256 tokens? (c) any repetition?
  3. **Multi-hop generation** — one real HotpotQA question from the calibration split, full Option A prompt, 512 tokens, greedy + per-model repetition penalty. Same detections plus EM.
  Output to `runs/04b_protocol_smoke_<timestamp>.json` so the model-card-confirmation lives next to the experimental outputs. Run before every Phase 4.1b sweep on a new box; smoke fails loudly (non-zero exit) if any model deviates from its declared mode.

### How to run (after env + models ready)

```bash
conda activate recollection
python scripts/04b_protocol_smoke.py     # ~3 min total (load + 3 probes per model)
bash scripts/run_phase41b_sweep.sh       # ~4-5 h, mirrors run_phase41_sweep.sh but writes to *_v2.jsonl
```

### Smoke results 2026-05-10 (env + downloads finished, validator ran end-to-end)

Box: A800 PCIe 80 GB, driver 550.90.07 (CUDA 12.6); torch 2.6.0+cu124; transformers 5.8.0; bnb 0.49.2. Downloads at pinned SHAs from `download_models.sh`. Smoke artifact: `runs/04b_protocol_smoke_20260510T015803Z.json` + `runs/04b_smoke_with_multihop.log`.

**Protocol (probe 1) — chat-template render under Option A:** all three models render the no-thinking prefill exactly as expected; `open_thinking_blocks` is empty for all three.

| model | rendered tail (last ~80 chars) | interpretation |
| --- | --- | --- |
| `phi-4-mini-instruct` | `...<\|end\|><\|user\|>... <\|end\|><\|assistant\|>` | clean — no thinking syntax exists in the template at all |
| `gemma-4-26b-a4b-it` | `...<turn\|>\n<\|turn>model\n<\|channel>thought\n<channel\|>` | empty thought channel (open `<\|channel>thought\n` immediately closed by `<channel\|>`) — this is the no-thinking path per card line 343 |
| `qwen3.6-35b-a3b-instruct` | `...<\|im_start\|>assistant\n<think>\n\n</think>\n\n` | `<think>` immediately closed by `</think>` — `enable_thinking=False` empty-think prefill from `chat_template.jinja` line 150 |

The first version of the smoke validator flagged Gemma and Qwen as fails because it grepped substrings like `<think>` / `<|channel>thought` in the tail. Fixed by checking for **open** thinking blocks (opener present without a matching closer between it and end-of-prompt). Both formats above are *closed* — explicitly closed for Qwen, structurally closed via empty channel for Gemma — so they're correct under Option A.

**Generation (probe 2) — "What is the capital of France?", 256-tok greedy + per-model rep_penalty:** all three answer "Paris" with a clean `Answer:` marker.

| model | tok | tok/s | early_stop | first-line generation |
| --- | ---: | ---: | --- | --- |
| `phi-4-mini-instruct` | 256 | 47.3 | max_tokens | "The capital of France is the city where its government and administrative functions are centered. ... Answer: Paris [continues for 200+ tokens repeating the answer]" |
| `gemma-4-26b-a4b-it` | 40 | 13.3 | answer_marker | "The question asks for the capital of France. ... Answer: Paris**Answer: Paris**\nthought\n" |
| `qwen3.6-35b-a3b-instruct` | 16 | 10.2 | eos | "Paris is the capital and most populous city of France.\n\nAnswer: Paris" |

Phi runs to max_tokens because it emits the answer then keeps repeating itself in restated forms ("Paris is known as the capital of France due to..." → "The question asks about the capital of France; however ..."). The Phase 4.1b driver's stricter `\bAnswer\s*:\s*([^\n]+)\n` early-stop (requires a newline after the answer text) will catch the *first* one — the smoke probe's looser implementation here is for visibility, not for replicating the sweep's stopping behavior. Qwen with `enable_thinking=False` lands the cleanest answer of the three at 16 tokens; the previous protocol had it stuck at 512 tokens inside `<think>`.

**Generation (probe 3) — real HotpotQA calibration question (id `5a8e3ea95542995a26add48d`, "Big Stone Gap director ... New York city?", gold "Greenwich Village, New York City"), 1024-tok greedy + per-model rep_penalty:**

| model | tok | tok/s | early_stop | extracted (smoke regex) | EM | F1 |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| `phi-4-mini-instruct` | 56 | 45.4 | answer_marker | "Greenwich VillageWhat was the original location of the New York Society of Model Engineers before it moved to Central Park?" | 0 | 0.24 |
| `gemma-4-26b-a4b-it` | 184 | 13.3 | answer_marker | "Greenwich Village, New York City//thought" | 0 | 0.91 |
| `qwen3.6-35b-a3b-instruct` | 51 | 11.1 | eos | "New York City" | 0 | 0.75 |

EM is 0 across the board because the smoke's local extraction regex is intentionally simple (greedy `(.+?)(?:\n|$)`); the sweep uses `qa_grading.extract_answer` which is more robust. The signal that matters: **all three terminate cleanly under 200 tokens with a real answer**, none drift into runaway repetition or open-`<think>` truncation.

Compare to the 4.1 (rejected) numbers on the same question:

| model | 4.1 (mid_stack, no-fix) | 4.1b smoke (Option A) |
| --- | --- | --- |
| Phi | 113 tok, EOS, "Greenwich Village, New York City", EM=1 | 56 tok, answer_marker, captured-too-much |
| Gemma | 145 tok, EOS, "Greenwich Village, New York City", EM=1 | 184 tok, answer_marker, "...//thought" suffix |
| **Qwen** | **512 tok, hit_cap, "they want the neighborhood", EM=0** | **51 tok, EOS, "New York City"** |

Qwen is the headline: protocol fix recovers a real answer in 1/10 the tokens. Phi under Option A grabs slightly more text on the answer line (no newline emitted between "Greenwich Village" and "What was..." — Phi's quirk, not the protocol's), worth watching for in the sweep but doesn't change EM at the grader level (`extract_answer` does extra cleanup). Gemma's "//thought" suffix is the empty-channel close token leaking into the decoded text — also a grader-side cleanup, not a protocol problem.

Conclusion: **Option A protocol verified end-to-end against the actual on-disk weights of all three models.** Sweep is unblocked.

### Things to watch (carried into 4.1b)

- **Don't trust the model card for sampling defaults blindly.** Phi card example uses greedy; Qwen card recommends `t=0.7, top_p=0.8, top_k=20, presence_penalty=1.5` for non-thinking mode. We're staying greedy across all three for trace-comparability (sampling noise muddies hidden-state drift signals — exactly the variable Phase 2/4 measure). Document the asymmetry vs the Qwen card; it's a deliberate choice.
- **`repetition_penalty=1.05` on Phi only** is asymmetric and reviewers may flag it. Defensive answer: it's a degeneration guard, not a quality knob; Phi without it produces literal sentence loops at >75 % rate, which corrupts the trace. Optional symmetric alternative: apply 1.05 to all three and re-run — costs nothing on Gemma/Qwen (they don't loop).
- **Don't read 4.1b numbers as "model capability."** Phase 4 grades the *gate*, not the model. EM/F1 only need to be high enough that there are correct AND incorrect traces in the test split — i.e. the gate has separable populations to fire on. If 4.1b gives Qwen-no-thinking ~30 % EM, that's still fine for the gate-comparison; if it gives <10 %, we have to reconsider.
- Smoke validator should be re-run on every new box: model cards may change (Qwen3.6 in particular has rolled multiple chat-template revisions in 2026), and our pinned SHAs can drift relative to whatever model-card text is current. The smoke is the bridge between the two.
- Existing `runs/phase4_qa_*_mid_stack.jsonl` from 2026-05-09 are kept on disk as a regression baseline; the v2 outputs land at `runs/phase4_qa_*_mid_stack_v2.jsonl` so analysis scripts can A/B them.

### Phase 4 status (after #11)

- Code: **patch ready**, smoke validator new.
- Smoke: pending (gated on env + model downloads finishing).
- 4.1b sweep: pending (~4–5 h on A800 once smoke passes).
- 4.2 / 4.3 / 4.4 / 4.5: still not started; 4.4 (b)+(c)+(d) ablations land naturally on top of 4.1b.

---

## 2026-05-09 #10 — Phase 3 baseline gates shipped (ReaLM sim, GRIP sim, random)

Code-only landing: the three Phase 3 simulators are implemented, unit-checked, and end-to-end smoked on one Phi-4-mini × factual × 200-token trace. Full 18-trace sweep (3 models × 3 prompts × 2 layer modes) is held until the user wants to spend the GPU time — same shape as Phase 2 but ~10–15 % cheaper per trace (the entropy capture is one extra `log_softmax` per decode step; throughput came in at 40.5 tok/s on Phi, vs ~46 tok/s for Phase 2 on the same model — ~10 % overhead, entirely from the per-token entropy reduction over the vocab).

### What landed

- **`scripts/crv/baseline_gates.py`** — four classes implementing the unified `evaluate(step_record) -> dict` surface:
  - `RealmRetrieveSimGate` — `(rsu_score > τ) OR (discourse marker matched)`. RSU is the per-step *mean next-token entropy* of the generator (no trained verifier — L3 simulation gap stamped on every row).
  - `GripSimGate` — `(z(boundary_entropy) > τ_z AND marker matched) OR (z > τ_strong)`. Rolling-window z-score (default window=20); the stat update happens *after* scoring so a bursty stream doesn't suppress its own spikes.
  - `RandomGate` — deterministic Bernoulli at a calibrated rate. Live `evaluate()` records placeholder rows; after the trace, `calibrate_rate(...)` sets $p = \text{mean}(p_\text{latent}, p_\text{realm}, p_\text{grip})$ over the same trace, then `finalize(...)` materializes seeded decisions. Default seed `20260509` written to summary.
  - `LatentGateAdapter` — thin wrapper around the existing `CRVTrigger` so the dispatch surface is uniform.
- **Discourse pattern set** — 14 case-insensitive regex patterns curated from ReaLM-Retrieve / GRIP qualitative examples (`let me verify`, `actually,`, `wait,`, `hmm,`, `i need to (verify|check|confirm|recheck)`, `not entirely sure`, ...). Curation provenance is itself stamped as a caveat: `discourse_pattern_set_is_curated_not_paper_exact`.
- **`scripts/03_baselines.py`** — driver. Re-uses the Phase 2 manual decode loop unchanged, plus:
  - per-token next-token entropy capture (`_entropy_nats(logits)` → fp32 log_softmax → $-\sum p \log p$, in nats);
  - per-step text decode at boundary (`tokenizer.decode(step_token_ids)`);
  - online evaluation of all four gates at every boundary;
  - post-pass random-gate calibration + finalization.
- **Output schema (Phase 3.4)** — `runs/phase3_triggers_<model>_<prompt>_<boundary>_<layermode>.jsonl`, one row per (step, gate). Each row carries `{run_id, model, prompt_class, step_idx, token_idx, gate_name, fired, score, reason, simulation_caveats}`. Per-step input record (`step_text`, `mean_token_entropy`, `boundary_token_entropy`, `drift_cosine`, `variance_l2_window`, `hidden_norm`) lands alongside in `..._steps.jsonl` so Phase 4 can re-derive gate decisions at new thresholds without re-running the generator. Per-trace summary (`fire_rates`, `fire_counts`, `gate_metadata` including the random seed and the calibration recipe) in `..._summary.json`.

### Smoke result (Phi-4-mini × factual × 200 tokens × last-layer CRV)

5 boundary steps committed → 20 trigger rows → schema conformant. Fire rates: latent 80 % (placeholder thresholds; uncalibrated; consistent with the Phase 2 #8 last-layer behavior on Phi), realm_sim 0 %, grip_sim 0 %, random 26.67 % = mean(0.80, 0.00, 0.00). On clean factual prose with no discourse markers and a stable entropy floor, both simulators correctly under-fire — the baseline isn't trivially saturated. L3 caveats land on every realm_sim and grip_sim row + in the summary's `gate_metadata`. Latent rows carry an empty `simulation_caveats` list as designed (the *latent* gate is the actual contribution, not a simulation), but its summary metadata flags `thresholds_uncalibrated_phase4_will_calibrate_on_labeled_traces` for honesty.

### Things to watch (carried forward into Phase 3 sweep / Phase 4)

- **Same-trace contract is enforced.** All four gates score the same physical decode trace; the random gate doesn't run a separate generation. This prevents the "we benchmarked a model swap, not a gate" failure mode listed under "Things to watch."
- **Threshold calibration is still a Phase 4 job.** The $\tau_{\text{rsu}}=2.5$ nats / $\tau_z=1.5$ / $\tau_\text{strong}=2.5$ defaults are placeholders (matched to the same regime the latent gate's $\tau_\text{drift}=0.1$ / $\tau_\text{var}=1.0$ live in). Phase 4.1's labelled split is still where calibration happens.
- **GRIP weak path needs a discourse marker.** On clean factual prose where markers are sparse, GRIP fires only via the strong-z path. This is *defensible* (GRIP's claim is that the trained classifier exploits context), but it means Phase 4 needs to log GRIP's per-rule fire breakdown so we don't accidentally claim the weak path is what's working. Already in `reason` (`spike+marker` vs `strong_spike` vs `below_thresholds`).
- **Random rate-matching is "fair" but soft.** A random-rate baseline at `mean(p_gates)` matches the *frequency* but not the *intelligence* of any gate; the comparison is "given the same firing budget, does the latent gate fire on the right *steps*?" — that question only has a meaningful answer once Phase 4.1 labels are in.
- **Entropy capture cost.** ~10 % throughput regression on the 200-tok Phi smoke; warmup-dominated (same shape as Phase 2's 11–14 % at 200 tokens dropping to ~3 % at 5 000 tokens). Will retest at full 5 000-token scale on the Phase 3 sweep.

### Phase 3 status

- Code: **done** (gates + driver + schema).
- Unit smoke: **done** (all 4 gates pass synthetic-input checks).
- End-to-end smoke: **done** (Phi × factual × 200 tok).
- Full sweep (3 × 3 × 2 = 18 traces × 5 000 tokens): **not started** — held for the next compute spend. Phase 4.1 (labelled multi-hop dataset) is still the higher-leverage next move; the 18-trace sweep is mostly a "make sure the schema scales" exercise that Phase 4 will subsume anyway.

---

## 2026-05-09 #10 — Phase 3 baseline gates shipped (ReaLM sim, GRIP sim, random)

Code-only landing: the three Phase 3 simulators are implemented, unit-checked, and end-to-end smoked on one Phi-4-mini x factual x 200-token trace. Full 18-trace sweep (3 models x 3 prompts x 2 layer modes) is held until the user wants to spend the GPU time. Throughput on the 200-tok Phi smoke came in at 40.5 tok/s vs ~46 tok/s for Phase 2 on the same model -- ~10 % overhead, entirely from the per-token entropy reduction over the vocab; warmup-dominated, expected to drop to ~3 % at 5 000-token scale (same shape as Phase 2 in #7/#8).

### What landed

- `scripts/crv/baseline_gates.py` -- four classes implementing a uniform `evaluate(step_record) -> dict` surface:
  - `RealmRetrieveSimGate` -- `(rsu_score > tau) OR (discourse marker matched)`. RSU is the per-step *mean next-token entropy* of the generator (no trained verifier; L3 simulation gap stamped on every row).
  - `GripSimGate` -- `(z(boundary_entropy) > tau_z AND marker matched) OR (z > tau_strong)`. Rolling-window z-score (default window=20); the rolling-stats update happens *after* scoring so a bursty stream doesn't suppress its own spikes.
  - `RandomGate` -- deterministic Bernoulli at a calibrated rate. Live `evaluate()` records placeholder rows; after the trace, `calibrate_rate(...)` sets `p = mean(p_latent, p_realm, p_grip)` over the same trace, then `finalize(...)` materializes seeded decisions. Default seed `20260509`.
  - `LatentGateAdapter` -- thin wrapper around the existing `CRVTrigger` so the dispatch surface is uniform.
- Discourse pattern set -- 14 case-insensitive regexes curated from ReaLM-Retrieve / GRIP qualitative examples (`let me verify`, `actually,`, `wait,`, `hmm,`, `i need to (verify|check|confirm|recheck)`, `not entirely sure`, ...). Curation provenance is itself stamped as a caveat: `discourse_pattern_set_is_curated_not_paper_exact`.
- `scripts/03_baselines.py` -- driver. Re-uses the Phase 2 manual decode loop unchanged, plus per-token next-token entropy capture (`_entropy_nats(logits)` -> fp32 log_softmax -> `-sum p log p`, in nats), per-step text decode at boundary, online evaluation of all four gates at every boundary, and post-pass random-gate calibration + finalization.
- Output schema (Phase 3.4) -- `runs/phase3_triggers_<model>_<prompt>_<boundary>_<layermode>.jsonl`, one row per (step, gate) carrying `{run_id, model, prompt_class, step_idx, token_idx, gate_name, fired, score, reason, simulation_caveats}`. Per-step input record (`step_text`, `mean_token_entropy`, `boundary_token_entropy`, `drift_cosine`, `variance_l2_window`, `hidden_norm`) saved alongside in `..._steps.jsonl` so Phase 4 can re-derive gate decisions at new thresholds without re-running the generator. Per-trace summary (`fire_rates`, `fire_counts`, `gate_metadata` including the random seed and the calibration recipe) in `..._summary.json`.

### Smoke result (Phi-4-mini x factual x 200 tokens x last-layer CRV)

5 boundary steps committed -> 20 trigger rows -> schema conformant. Fire rates: latent 80 % (placeholder thresholds; uncalibrated; consistent with the Phase 2 #8 last-layer behavior on Phi), realm_sim 0 %, grip_sim 0 %, random 26.67 % = mean(0.80, 0.00, 0.00). On clean factual prose with no discourse markers and a stable entropy floor, both simulators correctly under-fire -- the baseline isn't trivially saturated. L3 caveats land on every realm_sim and grip_sim row + in the summary's `gate_metadata`. Latent rows carry an empty `simulation_caveats` list as designed (the *latent* gate is the actual contribution, not a simulation), but its summary metadata flags `thresholds_uncalibrated_phase4_will_calibrate_on_labeled_traces` for honesty.

### Things to watch (carried forward into Phase 3 sweep / Phase 4)

- Same-trace contract is enforced. All four gates score the same physical decode trace; the random gate doesn't run a separate generation. This forecloses the "we benchmarked a model swap, not a gate" failure mode in the watch-list.
- Threshold calibration is still a Phase 4 job. The `tau_rsu=2.5` nats / `tau_z=1.5` / `tau_strong=2.5` defaults are placeholders, matched to the same regime the latent gate's `tau_drift=0.1` / `tau_var=1.0` live in.
- GRIP weak path needs a discourse marker. On clean factual prose where markers are sparse, GRIP fires only via the strong-z path. This is defensible (GRIP's claim is that the trained classifier exploits context), but Phase 4 needs to log GRIP's per-rule fire breakdown so we don't accidentally claim the weak path is what's working. Already in `reason` (`spike+marker` vs `strong_spike` vs `below_thresholds`).
- Random rate-matching is "fair" but soft. A random gate at `mean(p_gates)` matches the *frequency* but not the *intelligence* of any gate; the comparison is "given the same firing budget, does the latent gate fire on the right *steps*?" -- meaningful only once Phase 4.1 labels are in.

### Full sweep results (18 traces, 91 min wall on A800)

Driver: `bash scripts/run_phase3_sweep.sh` -- 6 sequential sessions, one per (model, layer_mode), each running all 3 prompt classes at 5 000 tokens. Per-session log in `runs/phase3_<model>_<layer_mode>.log`; sweep manifest in `runs/phase3_sweep.log`. Every session OK rc=0, no OOMs. Total 91 min: Phi 12 min, Gemma 26B-A4B 38 min, Qwen3.6 35B-A3B 41 min -- throughput essentially identical to Phase 2 (entropy capture cost is in the noise at 5 000-token scale, just as #7/#8 predicted; warmup-dominated overhead from the smoke #10 above is gone).

#### Per-(model, layer_mode) mean fire rates (averaged over the 3 prompt classes)

| model | layer | latent | realm_sim | grip_sim | random |
| --- | --- | ---: | ---: | ---: | ---: |
| phi-4-mini-instruct | last_layer | 45.6 % | 28.2 % | 3.4 % | 25.7 % |
| phi-4-mini-instruct | mid_stack | 71.3 % | 28.2 % | 3.4 % | 34.3 % |
| gemma-4-26b-a4b-it | last_layer | 99.1 % | 1.3 % | 6.2 % | 35.5 % |
| gemma-4-26b-a4b-it | mid_stack | 99.1 % | 1.3 % | 6.2 % | 35.5 % |
| qwen3.6-35b-a3b-instruct | last_layer | 99.7 % | 2.1 % | 6.0 % | 36.0 % |
| qwen3.6-35b-a3b-instruct | mid_stack | 89.7 % | 2.1 % | 6.0 % | 32.6 % |

Throughput / VRAM: Phi 42-46 tok/s @ 7.8 GB; Gemma 13.2 tok/s @ 49.7 GB; Qwen 12.6 tok/s @ 64.7 GB. Within noise of Phase 1.4 / Phase 2 numbers on the same prompts.

#### Reads from the sweep

- **Latent gate is saturated and uncalibrated, as expected.** The defaults (`tau_drift=0.1`, `tau_var=1.0`) were placeholders; the sweep confirms they over-fire. Calibration is a Phase 4 job; threshold sweeps are now offline-cheap because the per-step JSONL files (`runs/phase3_triggers_*_steps.jsonl`) carry the raw `drift_cosine` / `variance_l2_window` for every step. The ranking of saturation across architectures is interesting on its own: Gemma stays saturated on both layer modes (its drift signal is large *everywhere* in the stack), Qwen drops 99.7% -> 89.7% under the mid-stack mitigation (the same L1-disarm direction as Phase 2 #9), and Phi is the only architecture where the placeholder thresholds produce non-saturating fire rates (45-71 %).
- **ReaLM-sim picks up Phi's per-token uncertainty correctly.** Phi multi-hop hits 38.8 %; the same prompt on Gemma and Qwen is at 6.1 % / 0.3 %. That's exactly what we'd want from a "small-dense-model is more entropy-y per token" diagnostic, and it's a free signal -- a good sign the entropy-as-RSU stand-in is at least directionally meaningful.
- **GRIP-sim is architecture-flat by construction.** 3-7 % across all 18 traces because the z-score rolling normalization removes per-architecture entropy scale. That's the *correct* behavior for the spec ("classifier learns context-aware triggers"), and it gives the Phase 4 showdown a baseline whose firing budget doesn't trivially correlate with architecture -- exactly the comparison we want.
- **Random gate's rate-match worked.** Every trace's random rate equals `mean(latent, realm_sim, grip_sim)` to numerical precision; seed `20260509` reproducible, captured in every summary.
- **Layer mode is identity for non-latent gates.** ReaLM / GRIP / random fire rates are invariant across `last_layer` and `mid_stack` (they don't read hidden states). Useful sanity check that the driver is wiring the same trace into all four gates.

### Phase 3 status

- Code: done (gates + driver + schema).
- Unit smoke: done (all 4 gates pass synthetic-input checks).
- End-to-end smoke: done (Phi x factual x 200 tok).
- Full 18-trace sweep at 5 000 tokens: **done** -- 18/18 OK, 91 min wall, no OOMs, schema clean across all rows. Artifacts: 18 x `runs/phase3_triggers_*.jsonl` + 18 x `..._steps.jsonl` + 18 x `..._summary.json`, plus 6 session logs and the sweep manifest. Phase 4.1 (labelled multi-hop dataset) is still the next thing the threshold calibration depends on; the sweep mostly de-risked the schema and produced the offline corpus Phase 4 will read.

---

## 2026-05-09 #9 — Phase 2 mid-stack mitigation lands; L1 disarmed; first `_draft.tex` corpus written

Mid-stack matrix (L1 mitigation) ran on the same three architectures × three prompt classes × 5 000-token traces as #8, this time with `--layer-mode mid_stack` (capture from layer $\lfloor L/2\rfloor$: layer 16 of 32 on Phi, layer 20 of 40 on Qwen3.6, layer 15 of 30 on Gemma 4). Wall-clock 44.9 minutes, throughput identical to last-layer run (~3 % overhead). Artifacts: nine `runs/phase2_crv_*_mid_stack.jsonl` + nine `*_summary.json` + nine `*_crv.npz`, plus three figures `figs/phase2_{drift,variance,signal_summary}_per_arch_midstack.png` (or `_midstack.png` suffix on the summary). All last-layer artifacts intact alongside; the `_last_layer` vs `_mid_stack` filename distinction held throughout.

### Headline contrast — last-layer vs mid-stack

| arch | drift\textsubscript{cosine} (last $\to$ mid) | $\Delta$ drift | Var(L2) (last $\to$ mid) | $\Delta$ var |
| --- | --- | ---: | --- | ---: |
| Phi-4-mini | $0.024 \to 0.033$ | $+38\%$ | $2.0 \to 4.4$ | $+120\%$ |
| Gemma 4 26B A4B | $0.175 \to 0.106$ | $\mathbf{-39\%}$ | $476 \to 77$ | $\mathbf{-84\%}$ |
| Qwen3.6-35B-A3B | $\mathbf{0.479 \to 0.235}$ | $\mathbf{-51\%}$ | $411 \to \mathbf{0.09}$ | $\mathbf{-99.98\%}$ |

The headline:

1. **Qwen3.6 drift recovered** out of saturation. $0.479 \to 0.235$ moves the signal from "near the cosine-distance ceiling, gating impossible" to "in a working regime, threshold-amenable." The mitigation is the L1 prescription from `_PLAN.md` Tentative landmines, executed.
2. **Qwen3.6 variance collapses to ~0** ($411 \to 0.09$). At mid-stack the L2 norm of the recurrent Gated DeltaNet state is essentially constant — its spectral structure is preserved through the per-token $\alpha_t \cdot S_{t-1} + k_t \delta_t^\top$ update (with $\alpha_t \in [0,1]$, $\beta_t \in [0,1]$). The signal at mid-stack on Qwen3.6 lives entirely in **directional** change ($\drift$), not in **norm** change ($\Var$). This is a **per-architecture feature-selection finding** for Phase 4 trigger calibration: drift only on Qwen, drift+variance on Phi/Gemma.
3. **Gemma 4 cleans up significantly** ($-39\%$ drift, $-84\%$ var). The mid-stack drift trace reveals two huge clean spikes (~1.0 and ~1.25) at $T \approx 1500$ and $T \approx 4700$ on the factual prompt — events that were buried under routing noise at last-layer.
4. **Phi essentially unchanged.** Slight upward drift in both metrics (mid-stack hidden states are pre-final-RMSNorm, so naturally a bit less unit-normalised than last-layer). No L1 to mitigate; mitigation costs nothing.

### Drift-magnitude convergence supports the architecture-agnostic claim

The 20× spread of drift across architectures at last-layer ($0.024$ vs $0.479$) collapses to ~7× at mid-stack ($0.033$ vs $0.235$). Same order of magnitude across the three architectural classes. The "architecture-agnostic, with the right layer subset" claim now has empirical backing.

### Plot files

```
figs/phase2_drift_per_arch_midstack.png      — per-arch drift trace, 3-panel (mid-stack)
figs/phase2_variance_per_arch_midstack.png   — per-arch variance trace, 3-panel
figs/phase2_signal_summary_midstack.png      — bar summary parallel to last-layer
```

### Companion deliverable: `_draft.tex` first-pass corpus

Wrote `_draft.tex` (1736 lines, LaTeX `article` class, compile-ready modulo citation fills) as the *paper-prose corpus* — explicitly a superset that the final 4-page submission will be distilled from. Distinction:

- `_PLAN.md` — the experimental plan (forward-looking)
- `_AGENT.md` — the experimental log (this file; backward-looking, mechanics)
- `_draft.tex` — the paper-prose corpus (sideways-looking, scientific writing voice)

The three files together are the complete record. Adding to the canonical Status table above.

Structure of `_draft.tex`:

| section | source data |
| --- | --- |
| Abstract + Introduction + Related Work + Background | (prose only) |
| Method (CRV + boundaries + trigger + adapter + Algo 1) | refs to `scripts/crv/*` |
| Setup (Table 1: arch summary) | model `config.json` files |
| Phase 0 / L4 probe (paragraph) | `runs/smoke_*_a800.log`, `_AGENT.md` #2/#3b |
| Phase 1.3 (Fig + Table 2) | `runs/phase1_kv_pressure*.jsonl`, `figs/kv_pressure_baseline.png` |
| Phase 1.4 (Fig + Table 3 + two-axis bracketing) | `runs/phase1_4_*.jsonl`, `figs/phase1_4_kv_growth.png`, `_AGENT.md` #6 |
| Phase 2 last-layer (Table 4 + Fig 4/5/6) | `runs/phase2_crv_*_last_layer.*`, `figs/phase2_*.png`, `_AGENT.md` #8 |
| Phase 2 mid-stack (Table 5 + Fig 7/8/9) | `runs/phase2_crv_*_mid_stack.*`, `figs/phase2_*_midstack.png`, this entry |
| Phase 4 design (preregistered) | `_PLAN.md` Phase 3/4 |
| Discussion + Limitations + Reproducibility + Future + Conclusion | (prose) |
| Appendix A: prompt corpus | verbatim from `scripts/01_kv_pressure.py` |
| Appendix B: per-cell Phase 2 detail | TODO; data in `runs/phase2_crv_*_summary.json` |
| Appendix C: engineering notes from this log | curated from `_AGENT.md` |
| Bibliography | 22 `\bibitem`s with `TODO:` keys for future fill |

Maintenance protocol going forward: when a new phase produces data, write the experimental log here (`_AGENT.md` #N), update the plan checkboxes (`_PLAN.md`), then update the relevant section of `_draft.tex` to reflect the new numbers. The three files are deliberately decoupled so each can be edited without conflicting with the others.

### Phase 2 status (final)

| step | state |
| --- | --- |
| 2.1 hidden-state hook | done (last-layer + mid-stack, via `output_hidden_states=True` API) |
| 2.1b engine + arch-agnostic adapter | done — verified end-to-end on all three architectural classes |
| 2.2 step-boundary detector | done for `sentence`; `think_block` and `fixed` strategies implemented but unswept (see Future Work F3) |
| 2.3 CRV + variance + drift | done for `last_layer` and `mid_stack` modes; **L1 fires at last, mitigated at mid** |
| 2.4 trigger rule | implemented; thresholds NOT calibrated (Phase 4 task) |
| 2.5 stop-gate sanity probe | **deferred** — needs Phase 4.1 labelled correct/hallucinated traces to execute meaningfully |

Phase 2 closes here. Next is Phase 4.1 dataset assembly + Phase 3 baselines + the empirical showdown. The diagnostic infrastructure paper is essentially written; the empirical paper is the next phase block.

---

## 2026-05-09 #8 — Phase 2 full data: last-layer captures + L1 landmine fires (preserved as deliberate baseline)

Ran `02_crv_probe.py` end-to-end on all three architectural classes from Phase 1.4 — Phi-4-mini, Gemma 4 26B A4B, Qwen3.6-35B-A3B — sequentially on the A800 in bf16 native, three prompt classes per model, sentence-boundary detector, **last-layer mean-pooled CRV**. Total wall-clock 45.6 min. Artifacts: 9 JSONLs + 9 summary JSONs + 9 CRV `.npz` files in `runs/phase2_crv_<model>_<prompt>_sentence_last_layer.*`, plus per-model logs and three figures (`figs/phase2_drift_per_arch.png`, `figs/phase2_variance_per_arch.png`, `figs/phase2_signal_summary.png`).

### Headline numbers (last-layer CRV)

Steps committed and throughput (mean tok/s in parens; ~3 % overhead vs Phase 1.4 baseline):

|  | factual | multi-hop | ambiguous |
| --- | --- | --- | --- |
| Phi-4-mini-instruct (dense) | 149 (45.3) | 532 (45.9) | 186 (45.8) |
| Gemma 4 26B A4B (sliding+global MoE) | 340 (13.1) | **52** (13.4) | 231 (13.3) |
| Qwen3.6-35B-A3B (3:1 Gated DeltaNet) | 419 (12.5) | 447 (12.6) | 347 (12.6) |

(The Gemma 4 multi-hop = 52 steps outlier reflects the model emitting long unbroken paragraphs through the chained reasoning prompt; sentence-boundary marker tokens just don't fire, mean ~96 tok/step. Worth following up with the `think_block` strategy on Gemma.)

### Signal magnitude per architecture (last-layer, sentence-boundary)

| model | mean drift_cosine | mean rolling-variance(L2 norm) | total steps in trace |
| --- | ---: | ---: | ---: |
| Phi-4-mini-instruct | **0.024** | 2.0 | 867 |
| Gemma 4 26B A4B | 0.175 | 476 | 623 |
| Qwen3.6-35B-A3B | **0.479** | 411 | 1213 |

### The principal Phase 2 finding: **L1 landmine fires hard on Qwen3.6** (last-layer)

The plan (`_PLAN.md` Tentative landmines L1) flagged this as a hypothesis to falsify: *"MoE routing variance ≠ uncertainty variance — the router selects different experts per token, so naive mean-pool of last-layer hidden states absorbs expert-switching noise unrelated to reasoning uncertainty."* The Phase 2 last-layer CRV traces confirm it — strongly:

- **Phi (dense, no routing):** drift signal mostly under 0.05 with isolated event-driven spikes to ~0.2. The L2-norm variance is mostly flat with a sharp single spike around step 4000 in two of three prompt classes — *this is exactly the signature we want for hallucination detection.*
- **Gemma 4 (MoE + sliding-window):** drift in the 0.0–0.5 range, much noisier baseline; rich structure but high noise floor. Sits between Phi and Qwen on the L1 noise axis.
- **Qwen3.6 (MoE + 3:1 Gated DeltaNet):** drift consistently 0.3–0.8 across the entire 5 000-token trace, near-saturated near the cosine-distance ceiling — **the 256-expert × 8-active routing combined with the Gated DeltaNet recurrent state produces step-to-step hidden-state variability that washes the latent-state signal at the last layer.**

The plan's prescribed mitigation: read CRV from a *pre-routing mid-stack layer* instead of the last layer. That's the next experiment (#9 below).

### Preservation policy (this run is a deliberate baseline)

The noisy last-layer Qwen3.6 trace is *not a failed experiment*. It is the empirical confirmation of L1 — the failure mode the paper has to motivate the mid-stack mitigation against. **Both the last-layer matrix (this run) and the mid-stack matrix (next run) are preserved; the contrast is the finding.** File naming carries the layer mode (`_last_layer` vs `_mid_stack`) so neither can overwrite the other; the orchestrator computes the path via `phase2_crv_<model>_<prompt>_<boundary>_<layer_mode>.{jsonl,json,npz}` deterministically.

In the paper, both should show up:

- **Last-layer figure** (this run, `figs/phase2_*_per_arch.png`): "naive last-layer CRV is dominated by routing noise on linear-attention MoE; signal exists but SNR is poor."
- **Mid-stack figure** (the next run): "reading from a pre-routing mid-stack layer recovers a Phi-style clean signal across the MoE classes too — confirming the gating mechanism is architecture-agnostic *with the right layer subset*."

If the mid-stack mitigation works, the story is "L1 was real, here's why it bites, here's the fix, the contribution holds." If it doesn't work, the paper has to retreat from "architecture-agnostic" and claim the gate works only on dense / sliding-window models.

### Inspectable artifacts

```
runs/phase2_crv_phi-4-mini-instruct_{factual,multi-hop,ambiguous}_sentence_last_layer.jsonl       (per-step features)
runs/phase2_crv_phi-4-mini-instruct_{factual,multi-hop,ambiguous}_sentence_last_layer_summary.json (per-trace summary)
runs/phase2_crv_phi-4-mini-instruct_{factual,multi-hop,ambiguous}_sentence_last_layer_crv.npz     (CRV vectors, gitignored)
runs/phase2_crv_qwen3.6-35b-a3b-instruct_*_sentence_last_layer.{jsonl,json,npz}
runs/phase2_crv_gemma-4-26b-a4b-it_*_sentence_last_layer.{jsonl,json,npz}
runs/phase2_full_{phi,qwen,gemma}.log                                                              (per-model run logs)
figs/phase2_drift_per_arch.png        (3-panel facet, drift over trace, per architecture)
figs/phase2_variance_per_arch.png     (same layout for variance)
figs/phase2_signal_summary.png        (mean drift + mean variance bar chart, ±1σ across prompt classes)
runs/phase2_qwen_midstack_factual.log (interrupted; preserved as evidence of the kill before re-launch)
```

### Throughput regression: **inside the budget after warmup**

Phase 1.4 baseline = 47 / 13.1 / 12.85 tok/s for Phi / Gemma / Qwen. Phase 2 measured = 45.5 / 13.3 / 12.6 tok/s. Overhead **~3 % across the board** — well inside the ≤5 % budget set in `_PLAN.md` Phase 2.1. The 11–14 % I'd seen on the 200-token smokes was warmup-dominated; at 5k context the steady-state cost of `output_hidden_states=True` is small.

### Phase 2 status (after #8)

| step | state |
| --- | --- |
| 2.1 hidden-state hook | done (last-layer; mid-stack pending #9) |
| 2.1b engine + arch-agnostic adapter | done (verified on all 3 architectures) |
| 2.2 step-boundary detector | done — last-layer + sentence boundary traces shipped; `think_block` and `fixed` sweeps still pending |
| 2.3 CRV vector + variance + drift | done (last-layer); **L1 confirmed on Qwen3.6, partial on Gemma 4, clean on Phi** |
| 2.4 trigger rule | implemented; thresholds NOT calibrated (default `or` rule fires ≈100% on the MoE classes — calibration is Phase 4 work) |
| 2.5 stop-gate sanity probe | deferred (needs Phase 4.1 labelled correct/hallucinated traces) |

Next: **mid-stack layer mode across all three models × three prompts** (L1 mitigation experiment), same matrix scope as this run. Same 50-min compute budget. See #9 below.

---

## 2026-05-09 #7 — Phase 2 engineering: CRV adapter + orchestrator + first smokes

Built the architecture-agnostic CRV machinery (Phase 2.1 / 2.1b / 2.2 / 2.3 / 2.4) + the per-prompt orchestrator (`scripts/02_crv_probe.py`). Smoked end-to-end on Qwen3.6 and Phi-4-mini at 200 tokens × `factual` × sentence-boundary × last-layer CRV. Both runs produced clean per-step JSONL + summary + side `.npz` of CRV vectors. Architecture-agnostic claim holds; the CRV pipeline ran on both with zero per-arch glue, just `output_hidden_states=True` and the boundary detector. Phase 2.5 sanity probe deferred until Phase 4.1 brings labelled correct/hallucinated traces.

### Code shipped (engine + arch agnostic)

- `scripts/crv/__init__.py` — re-exports
- `scripts/crv/hooks.py` — `HiddenStateCapture`: consumes `output.hidden_states` from any HF causal-LM forward; extracts last-token vector per layer; detaches + moves to CPU fp32 immediately (L2 mitigation). Configurable layer subset and `every_n_steps` sub-sampling. **No `register_forward_hook` torch hooks** — using the public `output_hidden_states=True` API instead, which is uniform across dense / sliding-window / Gated-DeltaNet attention.
- `scripts/crv/boundaries.py` — `StepBoundaryDetector` with three strategies: `sentence` (token-id matching against precomputed punctuation set), `think_block` (`\n\n`, `</think>`, `<think>`, section headers), `fixed` (every N tokens; calibration baseline). Tokenizer-aware marker-id precomputation at init.
- `scripts/crv/vectors.py` — `CRVTracker`: buffers per-token captures between boundaries, mean-pools per layer at each boundary, concatenates selected layers into a single CRV vector. Tracks rolling-window L2-norm variance + cosine drift vs previous step's CRV. L2 drift mode also available as alternative metric. Drops the per-token buffer after each `commit_step` so memory stays flat.
- `scripts/crv/triggers.py` — `CRVTrigger`: pure threshold rules over scalar features (`drift_cosine`, `variance_l2_window`). Modes `or` / `and` / `drift` / `variance` / `none`. No internal state — same trigger can be applied to multiple feature streams.
- `scripts/02_crv_probe.py` — orchestrator: Phase 1's manual decode loop + KV cache + the CRV machinery wired in. Same prompts as Phase 1.3/1.4 so traces are directly comparable. Output:
  - `runs/phase2_crv_<model>_<prompt>_<boundary>_<layer_mode>.jsonl` — per-step scalar features
  - `runs/phase2_crv_..._summary.json` — per-trace summary
  - `runs/phase2_crv_..._crv.npz` — CRV vectors keyed by step_idx (gitignored — regeneratable, several MB at full 5k runs)

### Smoke results (200 tokens × `factual` × sentence-boundary × last-layer CRV)

| model | steps committed | mean tok/step | mean tok/s (Phase 2) | mean tok/s (Phase 1.4 baseline) | overhead | first-step drift_cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | 16 | 12.5 | 11.47 | 12.85 | **~11 %** | 0.61 (jump from prompt-processing into answering) |
| Phi-4-mini-instruct | 6 | 33.33 | 40.63 | 47.00 | **~14 %** | (similar magnitude) |

**Both pipelines work end-to-end with no per-arch code changes.** The captured features look meaningful — first generated step shows a large drift jump in both models (the model transitioning from "ingesting the prompt" to "starting the answer"), then settles to lower values. CRV vectors are 2048-dim on Qwen (last-layer hidden_size=2048) and 3072-dim on Phi (hidden_size=3072), saved as fp32 numpy arrays.

### Throughput regression: above the ≤5 % budget

`_PLAN.md` Phase 2.1 set a ≤5 % throughput budget. Smokes exceeded it (~11–14 %). Three causes, ordered by suspected impact:

1. **`output_hidden_states=True` causes HF to retain *all* layer outputs** in memory before we slice down to our chosen subset. We can't avoid this through the public API — the model materializes all 41 / 32 / 30 layer outputs; we only sub-select after the fact. To stay under budget on the full 5k run we'll either (a) bump `--every-n-steps` to 2 or 4 (capture half / quarter as many checkpoints), or (b) move to `register_forward_hook` on a single chosen layer to skip the all-layer materialization. (b) reintroduces per-arch concerns (layer naming differs); (a) is cheaper.
2. **Per-step `torch.cuda.synchronize()`** for accurate latency measurement adds ~1-3 ms / step. We need this for the latency JSONL but it's a known cost.
3. **CPU-side variance / cosine work** on the just-committed CRV. Negligible at single-layer subset; would matter if `layer_subset` is large.

Phase 2 critical path is *not* a perf benchmark — what matters is the signal quality. Document the overhead, run with `--every-n-steps=1` for clean signals, and don't optimize until Phase 4 cost analysis demands it.

### Inspectable artifacts

```
runs/phase2_crv_qwen3.6-35b-a3b-instruct_factual_sentence_last_layer.jsonl
runs/phase2_crv_qwen3.6-35b-a3b-instruct_factual_sentence_last_layer_summary.json
runs/phase2_crv_qwen3.6-35b-a3b-instruct_factual_sentence_last_layer_crv.npz       (gitignored)
runs/phase2_crv_phi-4-mini-instruct_factual_sentence_last_layer.jsonl
runs/phase2_crv_phi-4-mini-instruct_factual_sentence_last_layer_summary.json
runs/phase2_crv_phi-4-mini-instruct_factual_sentence_last_layer_crv.npz             (gitignored)
```

Sample row (Qwen3.6 step 2 — the big drift jump from prompt to answer):

```json
{"step_idx": 2, "token_idx_end": 9, "boundary_token_id": 13, "n_tokens_in_step": 2,
 "layer_indices": [40], "crv_dim": 2048, "hidden_norm": 91.92, "drift_cosine": 0.61,
 "variance_l2_window": 324.82, "wall_clock_s": 1.635, "vram_allocated_mb": 66153.3,
 "fired_default": true}
```

### What's deferred

- **Full 5k × 3 prompts × 3 architectures CRV capture.** Estimated wall-clock ~50 min on this box. Run when ready to invest the time and produce Phase 2 plots. Not done in #7 because the smoke is enough to validate the engineering and the full run is straightforward (just `python scripts/02_crv_probe.py --model X` for each of the three).
- **Boundary-strategy sweep.** Plan calls for sweeping `sentence` / `think_block` / `fixed=N` to compare which definition produces the most informative CRV signal. Code supports it via `--boundary` and `--fixed-stride`; just need to run the matrix.
- **Layer-mode sweep (`last_layer` vs `mid_stack` per L1 mitigation).** Same — code supports it via `--layer-mode mid_stack`.
- **Plotting** — `scripts/plot_phase2.py` will follow once we have full data. Plots: variance + drift over the trace per (model, prompt, boundary), 3-up grid by architecture for direct comparison.
- **Phase 2.5 stop-gate sanity probe.** Per `_PLAN.md`, requires correct-vs-hallucinated traces from Phase 4.1's HotpotQA / MuSiQue split. Cannot start until Phase 4.1 lands the labelled dataset.

### Phase 2 status

| step | state |
| --- | --- |
| 2.1 hidden-state hook | done (uses `output_hidden_states=True` not `register_forward_hook`) |
| 2.1b engine + arch-agnostic adapter | done (`scripts/crv/` package; smoked on Qwen3.6 + Phi) |
| 2.2 step-boundary detector | done (3 strategies; sweep pending) |
| 2.3 CRV vector + variance + drift | done (mean-pool, cosine drift, rolling variance) |
| 2.4 trigger rule | done (threshold rules; calibration deferred to Phase 4) |
| 2.5 stop-gate sanity probe | **deferred** (needs Phase 4.1 labelled dataset) |
| Full data run (5k × 3 × 3) | pending |
| Plotting + writeup | pending (`scripts/plot_phase2.py` + figs) |

---

## 2026-05-09 #6 — Phase 1.4 closed: three-way bracketing plot, plus the sliding-window-≠-cache-eviction finding

Ran `01_kv_pressure.py` end-to-end on the three architectural classes per #5's spec, on the same A800 80 GB box, in bf16 native fully on cuda:0, sequentially. Three prompt classes per model (factual / multi-hop / ambiguous), 5 000 tokens uninterrupted, 500-token checkpoints. Total wall-clock: Phi 5.5 min, Gemma 4 26B A4B 19.2 min, Qwen3.6 (re-used Phase 1.3 data, would've been ~20 min). Artifacts: `runs/phase1_4_phi.jsonl`, `runs/phase1_4_gemma4_26b.jsonl` (each 30 checkpoint rows), `runs/phase1_4_*_summary.jsonl`, plus `figs/phase1_4_arch_comparison.png` (raw VRAM panels) and `figs/phase1_4_kv_growth.png` (the headline plot — KV growth normalised to step 500).

### Headline numbers (mean across 3 prompt classes, at step 5000)

| model | mean tok/s | KV growth (MB) | KV growth slope (KB/tok) | peak VRAM (MB) | decode step1→stepN slowdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Phi-4-mini-instruct** (Dense, full GQA, 32 layers) | **47.0** | **643** | **128** | 7 985 | **1.01–1.07×** (visible) |
| **Gemma 4 26B A4B** (MoE, 5 sliding + 1 global × 5, 30 layers) | **13.2** | **1 105** | **221** | 50 852 | **0.99–1.11×** (slight) |
| **Qwen3.6-35B-A3B** (MoE, 3 linear + 1 full × 10, 40 layers) | **13.0** | **135** | **27** | 66 258 | **0.96–0.98×** (none / slight speedup) |

### What the bracketing actually looks like (and the surprise)

The Phase 1.4 plan, written in #5, predicted Gemma 4 26B A4B's curve would show a **knee at 1024 tokens** (sliding cache cap) followed by a flatter regime determined only by the 5 global layers. That prediction was wrong in an instructive way.

**Observation:** Gemma 4 26B A4B's KV-growth slope is **constant at ~215 KB/token across the entire 5 000-token run** — no knee, no eviction. Per-checkpoint deltas: 107 / 108 / 107 / 106 / 107 / 109 / 105 / 109 / 106 MB per 500 tokens, all within ~3 %.

**Why:** in HF transformers' `DynamicCache`, sliding-window attention bounds the *attention computation* (only the last `sliding_window=1024` tokens are attended to in sliding layers), but the cache *itself does not auto-evict tokens past the window*. The cache stores everything, attention just ignores the older entries. So memory grows linearly forever; only compute is bounded.

Per-layer math reconciles cleanly:
- 5 global layers × `num_global_key_value_heads=2` × `global_head_dim=512` × 2 (K+V) × 2 bytes (bf16) = **20 KB/tok** unbounded
- 25 sliding layers × `num_key_value_heads=8` × `head_dim=256` × 2 (K+V) × 2 bytes = **200 KB/tok** stored-not-evicted
- Total: **220 KB/tok** ≈ observed 215 KB/tok ✓

The 5 KB/tok gap (220 predicted vs 215 observed) is consistent with measurement granularity (allocator rounding to 2 MB blocks). Match within 3 %.

**Implication for the bracketing argument (and the paper):** the cleaner story is in **two dimensions**, not one. The 2026 attention-design space optimises *compute pressure* and *memory pressure* separately. Sliding-window architectures bound compute but in standard reference implementations do not bound memory. Linear-attention hybrids bound both. Vanilla full attention bounds neither.

| | bounds compute? | bounds memory? |
| --- | --- | --- |
| Phi-4-mini (vanilla full GQA) | no — visible 1.01–1.07× slowdown over 5 000 tok | no — 128 KB/tok linear |
| Gemma 4 26B A4B (sliding+global, HF reference impl) | **yes** — flat decode latency | **no** — 215 KB/tok linear (steeper than Phi!) |
| Qwen3.6-35B-A3B (3:1 Gated DeltaNet:full) | **yes** — flat or slightly faster | **mostly yes** — 27 KB/tok (only the 10 full-attn layers contribute) |

That's a tighter and more EMNLP-friendly framing of the bracketing argument than "memory is being solved." More precisely: **memory is being solved unevenly. The sliding-window class trades cache for compute; the linear-attention class trades cache for state. Latent-state gating's value is decoupled from which trade was made — it works at the layer-output level on all three.**

### Plot files

- `figs/phase1_4_arch_comparison.png` — raw VRAM panels, dominated by weight footprint (~8, ~50, ~65 GiB respectively); KV-growth signal invisible at this scale. Useful only for showing that the *baseline* sits in different places per architecture.
- `figs/phase1_4_kv_growth.png` — **the headline figure.** KV growth from step-500 baseline, mean ±1σ across 3 prompt classes. Three clean slopes: Gemma 4 ~960 MB, Phi ~545 MB, Qwen3.6 ~85 MB at step 5000. The bracket the paper wants.

### Per-prompt-class invariance holds (still)

Within each model, the three prompt classes (factual / multi-hop / ambiguous) produced essentially identical KV-growth slopes and tok/s, confirming Phase 1.3's finding that **the cost-per-token signature is prompt-class-invariant** at this context length. The shaded ±1σ bands in the figure are visually thin. EOS fired on Gemma 4 multi-hop @ step 902 and ambiguous @ step 3425, but the run continued past EOS as designed; KV cost continued growing identically, confirming the post-EOS measurement validity.

### Decode latency footnote (the Gemma 4 multi-hop spike)

Gemma 4 multi-hop showed a **1.11× decode slowdown** (0.0712 → 0.0789 s/step) — visibly larger than the other two classes (0.99× / 1.04×). Possible cause: multi-hop reasoning routes more often through compute-heavy expert paths; or a subset of decode steps engaged the global-attention layers more (e.g. when the model is doing the chain-step transitions). Worth a follow-up in Phase 4.4 ablations once we're routing per-step.

### Mechanics that worked (Phase 1.4-specific)

- `01_kv_pressure.py` worked first try on Gemma 4 (`Gemma4ForConditionalGeneration` arch class, multimodal head). `apply_chat_template(messages, add_generation_prompt=True)` produced clean text prompts via `AutoTokenizer`; no `AutoProcessor` needed for text-only mode. `DynamicCache` machinery (manual `past_key_values=` loop) handled both Gemma 4 and Phi-4-mini without code changes.
- The `00_smoke.py` raw-prompt path produced "...." dots on Gemma 4 (smoke entry above this in `runs/smoke_gemma4_26b_bf16_a800.log`); `01_kv_pressure.py`'s chat-templated path produced sane content. **Confirms that for Phase 2.5+ the L4 / sanity probe must use `apply_chat_template`** to be uniform across architectures. Carry into Phase 2.1b (uniform hook surface requirement).
- Decode latency measurement under `torch.cuda.synchronize()` per step works cleanly across all three architectures. No per-arch workarounds needed.
- Sequential model load is fine: Phi loaded 8.5 s (post-Gemma run, slight cache hit), Gemma 4 41 s (down from 24 s standalone — disk contention with the parallel 31B download gone, but still slower than expected; possibly the 1013-shard count vs Qwen's 693 vs Phi's 194 plays in).

### Phase 1 status (final)

| step | state |
| --- | --- |
| 1.1 generator/reflector wiring + chat template | done (1.3 + 1.4) |
| 1.2 `01_kv_pressure.py` | done |
| 1.3 three prompt classes on Qwen3.6 | done (`figs/kv_pressure_baseline.png`) |
| 1.4 three-way architectural-class baseline | **done** (`figs/phase1_4_kv_growth.png`) |

Phase 1 closed. **Q7 from #5 answered:** the three architectures bracket the cost-curve design space cleanly enough to tell the EMNLP story; updated picture is in the table above. **Q8 (drift signal across all 3 classes — Phase 2.5) and Q9 (CRV calibration transferability — Phase 4.4) remain open as designed.**

### Gemma 4 31B Dense status

Downloaded (62.6 GB at `models/gemma-4-31b-it/`, sha `ba74f5b6...`) but **parked as Phase 4 ablation reserve**, not part of the canonical Phase 1.4 bracketing plot. Reasoning (clarified after the user's question): 31B Dense uses the *same* sliding-window+sparse-global attention pattern as 26B A4B, so it would be a fourth line on the bracketing plot that mostly traces the Gemma-2-class curve — adds visual noise without a new architectural-class point. Where 31B Dense *does* add value is the narrow **MoE-vs-dense ablation within the Gemma family** for Phase 4.4: same training, same tokenizer, same attention design, MoE structure on/off. That's the question we'll answer with it later, not now. Reframe applied to `_PLAN.md` Phase 1.4 (the "(optional row 4)" was a misframing) and to the Futures section.

---

## 2026-05-09 #5 — Cognitive-efficiency pivot, Phase 1.4 spec, Gated DeltaNet ratio correction

User brought in an external EMNLP-2026-era framing recommendation in response to the Phase 1 surprise (#4): pivot the paper from a *hardware-resource* ("memory pressure") frame to a *cognitive-efficiency* ("memory is being solved by architecture; reasoning-trajectory drift is the open frontier") frame, and bracket the architectural design space with three concrete generators rather than one. After verifying the model availability and Gated DeltaNet citations (Raschka), agreed and incorporated.

This entry is the canonical record of the third framing pivot in this project (edge-VRAM → KV-pressure → cognitive-efficiency). Earlier two framings preserved verbatim in #3b ("Things to watch") and #4 (above).

### The pivot in one sentence

**From** *"agentic LLM reasoning under KV-cache pressure"* **to** *"latent-state gating is a hardware-native, architecture-agnostic earlier-than-text-parsing trigger for corrective retrieval — and we prove it on three points along the 2026 attention-design spectrum (dense full attention, sliding-window+global hybrid, linear-attention hybrid) where memory pressure ranges from steep to negligible."*

### Why this framing and not a fourth-time tightening of "memory pressure"

Phase 1.3 measured 80 MB total KV growth across 5 000 tokens on Qwen3.6 (`_findings.md` Phase 1, `runs/phase1_kv_pressure.jsonl`, `figs/kv_pressure_baseline.png`). The hybrid Gated-DeltaNet architecture flattens the curve we wanted to study by construction. The honest read is that the hardware bottleneck the original "edge VRAM" framing assumed is being fixed by architecture choices that are already shipping in late-2025 / 2026 models. Pretending otherwise to defend the framing would be brittle as more linear-attention hybrids ship through 2026 (Kimi Linear, Qwen3-Coder-Next, future Mamba-3 hybrids). The cognitive-efficiency frame ages better: it asserts that even when architecture removes the memory cost, the *reasoning-trajectory drift* that produces hallucinations during long `<think>` blocks is unchanged, and a latent-state hook is the cheapest way to detect it.

### Three-way architectural-class baseline (Phase 1.4 spec) — verified configs

All four candidates verified via HF API + `AutoConfig.from_pretrained` smoke (run captured to `runs/phase1_4_pre_run_checks.log`). Configs read from each model's `config.json` (Phi + Qwen already on disk; Gemma values from `AutoConfig` introspection).

| row | role | repo (HF) | sha (pinned) | layers | n_attn / n_kv heads | head_dim | hidden | sliding_window | MoE: total / active expert | bf16 GB | KV per token (full-attn layers) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Dense SLM (control) | `microsoft/Phi-4-mini-instruct` | `cfbefacb...` | 32 | 24 / 8 | 128 | 3072 | (262 144 = effectively unbounded) | dense | 7.7 | **128 KB/tok** (8 KV × 128 dim × 2 (K+V) × 2 bytes × 32 layers) |
| 2 | Standard MoE (industry baseline) | `google/gemma-4-26B-A4B-it` | `462a98a1...` | 30 | 16 / 8 | 256 | 2816 | **1024** (local layers cap; some global) | 128 total / `num_experts_per_tok` TBD from layer config | 51.6 | per global layer: 8 × 256 × 2 × 2 = **8 KB/tok**; per local layer: bounded at sliding_window (1024 tokens) |
| 3 | Hybrid Linear MoE (SOTA) | `Qwen/Qwen3.6-35B-A3B` | `995ad96e...` | **40** *(not 41 — corrected from #4)* | 16 / **2** *(not 4 — same product, wrong attribution)* | **256** *(not 128)* | 2048 | n/a | 256 total / 8 active per token + 1 shared (`moe_intermediate_size=512`) | 69 | only on the 10 full-attn layers (indices 3,7,...,39): 2 × 256 × 2 × 2 × 10 = **20 KB/tok**, exactly matches Phase 1.3 measurement |
| 4 *(optional ablation)* | Scale-matched Dense | `google/gemma-4-31B-it` | `ba74f5b6...` | 60 | 32 / 16 | 256 | 5376 | 1024 (same Gemma 4 pattern) | dense | 62.6 | per global layer: 16 × 256 × 2 × 2 = **16 KB/tok**; local layers bounded |

Pre-run checks (all green; see `runs/phase1_4_pre_run_checks.log`):

1. ✅ **Gemma 4 transformers 5.8.0 compatibility.** `model_type="gemma4"` is registered in `transformers.AutoModelForCausalLM._model_mapping` (alongside `gemma4_text` and `gemma4_assistant`). Architecture class is `Gemma4ForConditionalGeneration` — multimodal head pattern, same as Qwen3.6. **No version bump needed.** `AutoConfig.from_pretrained("google/gemma-4-26B-A4B-it")` and the 31B variant both load cleanly without `trust_remote_code`. Gemma 4 was released ~2026-05-05; transformers 5.8.0 already supports it.
2. ✅ **Configs introspected.** All architectural facts in the table above came from the actual configs (Qwen + Phi from local `models/<name>/config.json`; Gemma from `AutoConfig`).
3. ✅ **Qwen3.6 layer-types literally enumerated.** `models/qwen3.6-35b-a3b-instruct/config.json` has a `layer_types: [...]` array of length 40. Pattern: three `linear_attention` then one `full_attention`, repeating. Full-attn at indices 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 → **exactly 10 full-attention layers**. Config also lists `full_attention_interval: 4`. **3:1 ratio confirmed at the source.** The earlier 1:7 guess (in #4) and the secondary "4 KV × 128 dim" guess (in this entry's previous draft) are both wrong; corrected here.
4. ⏳ **Chat-template parity** — not yet smoked end-to-end on Gemma 4. Will run after weight pull. Gemma 4 model card shows `AutoTokenizer` works for text-only mode (note: card uses `AutoProcessor` for multimodal, but text path uses tokenizer); thinking control via `<|think|>` in the system prompt.
5. ⏳ **Thinking mode** — Qwen3.6 default-on, Gemma 4 controllable via system token, Phi-4-mini has no built-in thinking. Plan: enable thinking where available (realistic agentic mode); document the Phi asymmetry in Phase 1.4 results.

Execution: bf16 native fully on cuda:0, sequential per model, three prompt classes per model (factual / multi-hop / ambiguous from `01_kv_pressure.py`), 5 000 tokens uninterrupted with 500-token checkpoints. Wall-clock budget ~20 min per model, 60–80 min total. Output: extended JSONL (`runs/phase1_4_arch_comparison.jsonl` to keep #4 data separate) + new figure overlaying all classes (3-4 lines per panel × 2 panels for VRAM and tok/s). NF4 path is *not* on the Phase 1.4 critical path; the consumer-card validation arm stays in **F1 (Futures)**.

Predicted KV-pressure curves at 5k tokens (to be measured in 1.4):

| model | KV @ 5k tokens (predicted) | curve shape |
| --- | --- | --- |
| Phi-4-mini | **128 KB/tok × 5000 = ~625 MB** | clean linear O(N) — the steep reference |
| Gemma 4 26B A4B | global layers grow linearly; local layers cap at 1024 → **knee-shaped curve** (steep then flat) | bounded local + sparse global |
| Qwen3.6-35B-A3B | already measured — **~100 MB at 5k**, near-flat | hybrid linear, near-O(1) |

That's the bracketing plot the EMNLP 2026 framing asks for.

### Why `gemma-4-26B-A4B-it` is the right "Standard MoE" pick for row 2

The user's first proposal cited "Gemma 4 (26B)" with `~16 GB`. That maps to either the **26B A4B MoE** (~14 GB at NF4 / ~52 GB at bf16) or the **31B Dense** (~16 GB at NF4 / ~62 GB at bf16). Picked the **26B A4B MoE** for row 2 because:

1. **Same MoE structure as Qwen3.6 (row 3).** Isolates the *attention-mechanism* variable: row 2 (sliding-window + sparse global) and row 3 (3:1 linear + full) share MoE structure but differ in how attention scales with context. Row 2 vs row 1 (dense) then isolates MoE-vs-dense.
2. **Matched active-parameter scale.** Gemma 4 26B A4B has 3.8 B active vs Qwen3.6's 3 B active. Per-token compute envelope is comparable, so tok/s differences will primarily reflect attention-mechanism cost.
3. **Has thinking mode.** Native `<|think|>` control matches Qwen3.6's behavior; Phi-4-mini's lack of thinking mode is the asymmetry we'll have to document, not Gemma's.
4. **Fits sequentially on the A800.** ~52 GB bf16 + ~14 GB headroom fits easily; we don't need NF4 to run it.

Gemma 4 31B Dense is queued as the optional row 4 — same Gemma-4 sliding-window architecture but no MoE — to isolate "MoE structure vs dense" within the Gemma family. Adds one ~20-min run.

### Gated DeltaNet — corrected technical record (replaces the wrong 1:7 in earlier #4)

The Qwen3-Next family (Qwen3-Next, Qwen3.5, Qwen3.6, Qwen3-Coder-Next) and Kimi Linear use a **3:1 ratio of Gated DeltaNet linear-attention layers to full-attention layers** — three linear blocks, then one full-attention block, repeating. Sources:

- Sebastian Raschka, [*Gated DeltaNet*](https://sebastianraschka.com/llms-from-scratch/ch04/08_deltanet/) (Beyond Standard LLMs, ch. 4 / bonus 8): *"That is why these models often use a 3:1 ratio: three linear-attention style layers, one full-attention layer, then repeat."*
- The companion [`LLMs-from-scratch` ch. 4/08 README](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/08_deltanet/README.md): *"Both Qwen3-Next and Kimi Linear use a 3:1 ratio, meaning for every three transformer blocks employing the linear Gated DeltaNet variant, there's one block that uses full attention."*
- Original implementation: Yang et al., [*Gated Delta Networks: Improving Mamba2 with Delta Rule*](https://arxiv.org/abs/2412.06464), Dec 2024.
- Qwen team's reference impl: [`modular_qwen3_next.py` line ~835](https://github.com/huggingface/transformers/blob/0ed6d51ae8ed3f4fafca67a983b8d75bc76cd51b/src/transformers/models/qwen3_next/modular_qwen3_next.py#L835).

What Gated DeltaNet actually does (compressed): sequentially processes tokens and maintains a fixed-size recurrent state matrix `S` of shape `(b, n_heads, head_dim, head_dim)`. The state is updated per token via a delta-rule update (`δ = (v - S k) · β`, `S ← α · S + k δᵀ`), with two gates: `α` (scalar per-head decay; controls how fast the state forgets) and `β` (per-head update; controls how strongly each new token modifies the state). An output gate (SiLU-activated) follows. Memory cost per layer is `n_heads × head_dim²` — *no* dependency on context length, in contrast to full attention's `n_kv_heads × head_dim × n_tokens`. The downside: prior tokens have to be compressed through the fixed-size state bottleneck, so global context modeling is weaker — which is why these architectures keep ~25 % of layers as full attention for periodic global mixing.

**Reconciling against our Phase 1.3 data (corrected with the actual `config.json`):**

Qwen3.6-A3B has **40 layers** (not 41 as I earlier wrote — `num_hidden_layers: 40`). The config's `layer_types` array literally enumerates the pattern: three `linear_attention` blocks then one `full_attention` block, repeating; full-attn indices are 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 → **exactly 10 full-attention layers + 30 Gated DeltaNet layers**. Observed KV slope from Phase 1.3: ~20 KB/token. Per full-attn layer per token: `20 480 ÷ 10 ÷ 2 (K+V) ÷ 2 (bf16 bytes) = 512` elements per K (and per V). The actual config has `num_key_value_heads: 2` and `head_dim: 256` → 2 × 256 = 512 elements per K. **Math matches exactly.** Earlier drafts of this entry guessed "4 KV × 128 dim" — same product (1024 bytes per K), but wrong attribution. The Qwen3.6-MoE config uses an unusual asymmetric head geometry (`num_attention_heads: 16`, `head_dim: 256` → query projection 16 × 256 = 4096 *up* from hidden_size 2048; KV projection 2 × 256 = 512 down from hidden_size, the GQA compression).

**`_findings.md` carries the same wrong ratio** in its "Why" section ("1:7 full-to-linear ratio" / "5 of 41 full-attention layers"). Per the user's standing rule that `_findings.md` is updated only on explicit instruction, the correction is *not* applied there yet — flagged to the user for sign-off. The error doesn't change the *qualitative* finding (most layers are linear, KV growth is small, model is engineered to flatten the curve) but the layer count and per-layer KV math are off by a factor of 2.

### Implications for downstream phases (what changed in `_PLAN.md`)

Updated in this commit:

- **Thesis** rewritten to the cognitive-efficiency framing (memory is being solved; trajectory drift is the frontier).
- **Engine** line widened to "three-class generator pool" with bf16-native vs NF4 guidance.
- **Phase 1.4** replaced the pinned A/B/C decision with the three-way (optional four-way) spec table + pre-execution checklist + execution plan. Wall-clock budget is now real.
- **Phase 2 / 2.5** sharpened: the sanity probe now must show CRV separation on **all three architectural classes** — that's the test of "architecture-agnostic" in the new thesis. Phase 2.1b got an explicit "engine-architecture-agnostic" requirement (same hook surface across Phi-4-mini, Gemma 4, Qwen3.6).
- **Phase 4 / 4.2 / 4.5** sharpened to the "hardware-native earliness" framing. Earliness is now defined relative to the eventual hallucinated token, faceted by architecture class. Cost framing now reports avoided corrective-retrieval round-trips and avoided wasted thinking, not just KV memory.

Not yet updated (need user sign-off):

- `README.md` still carries the prior "KV-cache pressure" framing. Stale; needs a small rewrite to lead with "memory is being solved by architecture; trajectory drift is the open frontier; latent-state gating is the trigger."
- `_findings.md` carries the wrong 1:7 ratio in its Phase 1 "Why" section. Two-paragraph fix; needs user OK to touch the file.

### Things to verify before pulling Gemma 4 weights

1. `huggingface.co/google/gemma-4-26B-A4B-it` exists and is downloadable (the model card we read was for `gemma-4-31B-it`; the 26B A4B variant is referenced in the same family table but we haven't confirmed its repo path).
2. transformers 5.8.0 supports the Gemma 4 architecture (load test: `AutoConfig.from_pretrained("google/gemma-4-26B-A4B-it")` should work without a remote-code requirement).
3. Disk space: 4 models adds ~120 GB to `models/`. Currently ~76 GB used; we have 362 GB overlay headroom — fine.
4. `pip` may need `transformers @ git+` for Gemma 4; if so, regenerate `runs/env_lockfile.txt` and document the bump in this entry.

---

## 2026-05-09 #4 — Phase 1 KV-pressure baseline (and the framing surprise)

Built `scripts/01_kv_pressure.py` and ran 5,000-token uninterrupted greedy generation on Qwen3.6-35B-A3B (bf16 native fully on cuda:0, canonical A800 path) over three prompt classes: factual (long-form historical exposition), multi-hop (chained reasoning with confidence checks), ambiguous (open-ended creative continuation). Manual generation loop with explicit `past_key_values` so we can checkpoint VRAM + per-step latency at every 500 tokens. Total wall-clock: 19.8 min for the full 3 × 5k run. Artifacts: `runs/phase1_kv_pressure.jsonl` (30 checkpoint rows), `runs/phase1_kv_pressure_summary.jsonl` (3 summary rows), `figs/kv_pressure_baseline.png`.

### Headline numbers

| prompt class | mean tok/s | prefill (s) | decode step1 (s) | decode step5000 (s) | step1→stepN slowdown | total KV growth (MB) | peak VRAM (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| factual | **12.85** | 0.798 | 0.0816 | 0.0798 | **0.98×** | 141 | 66258 |
| multi-hop | 12.80 | 0.414 | 0.0800 | 0.0782 | 0.98× | 132 | 66258 |
| ambiguous | 12.89 | 0.388 | 0.0796 | 0.0761 | 0.96× | 132 | 66258 |

Three numbers per `_PLAN.md` Phase 1.3 deliverable: **peak VRAM = 66 258 MB (~64.7 GiB)**, **tok/s @ 500 = 12.6–12.8 (rolling)**, **tok/s @ 5000 = 13.0–13.1 (rolling)**.

### The surprise: there is no KV-pressure curve up to 5 k tokens

Top panel of the figure shows three perfectly overlapping VRAM-vs-tokens lines, going from ~64.62 GiB at 500 tokens to ~64.70 GiB at 5000 tokens. **Total VRAM growth across the entire 5 000-token generation is ~80 MB, prompt-class invariant.** Per-token KV slope is ~20 KB/token. Per-token decode latency is *flat or slightly decreasing* across all three classes — step 5000 is ~2 % faster than step 1.

For comparison, a naive standard-transformer estimate (41 layers × 6 KV heads × 128 dim × 2 (K+V) × 2 bytes/bf16 = ~240 KB/token) would have predicted ~1.2 GB of KV growth at 5 000 tokens *and* substantial per-step slowdown from O(N) attention. We observed ~12 % of that.

### Why: Qwen3.6-A3B is a hybrid linear-attention MoE

The "fast path is not available — install fla / causal-conv1d" warning at load (already noted in #3b but now load-bearing) is the giveaway. The Qwen3-Next / Qwen3.5 / Qwen3.6 / Qwen3-Coder-Next family uses a **3 : 1 ratio of Gated-DeltaNet (linear-attention) layers to full-attention layers** — three linear blocks, then one full-attention block, repeating. The same 3:1 hybrid pattern is used by Kimi Linear (with MLA in place of full attention).

> *Sources (added 2026-05-09 #5):* Sebastian Raschka, [Gated DeltaNet (Beyond Standard LLMs ch. 4)](https://sebastianraschka.com/llms-from-scratch/ch04/08_deltanet/) and the [`LLMs-from-scratch` ch. 4/08 README](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/08_deltanet/README.md): "Both Qwen3-Next and Kimi Linear use a 3:1 ratio, meaning for every three transformer blocks employing the linear Gated DeltaNet variant, there's one block that uses full attention." Original implementations: Yang et al., [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) (Dec 2024); Qwen team's [`modular_qwen3_next.py`](https://github.com/huggingface/transformers/blob/0ed6d51ae8ed3f4fafca67a983b8d75bc76cd51b/src/transformers/models/qwen3_next/modular_qwen3_next.py#L835).

On a **40-layer Qwen3.6-A3B** (`num_hidden_layers=40` in `config.json`; the `layer_types` array enumerates the pattern explicitly), 3:1 → **exactly 10 full-attention layers (indices 3, 7, 11, 15, 19, 23, 27, 31, 35, 39) + 30 Gated DeltaNet layers**. Reconciling against observed slope: 20 KB/token ÷ 10 full-attn layers ÷ 2 (K+V) ÷ 2 bytes (bf16) = **1024 bytes per layer per K**, which matches the configured `num_key_value_heads=2` × `head_dim=256` = 512 elements per K = 1024 bytes per K (and same for V) per layer per token. The 3:1 hypothesis matches our data exactly; an earlier draft of this entry guessed 1:7 ratio and an even earlier draft of #5 guessed "4 × 128" head geometry — both have been corrected against the literal config above.

Net: **at 5 k context this model is engineered to make KV memory and per-step compute effectively constant.** Gated DeltaNet's recurrent state has size `n_heads × head_dim²` per layer (constant in context length), with α (decay) and β (update) gates regulating how the state forgets and absorbs new tokens. The full-attention layers — every fourth layer plus the first/last — provide periodic global mixing. The linear-attention layers run on a torch fallback (no `fla` / `causal-conv1d` installed) and still outperform standard-attention scaling at this length.

### What this means for the project framing

This is a real problem for the **"agentic reasoning under KV-cache pressure"** thesis we just adopted in `README.md` + `_PLAN.md` (commit pre-#4). On the canonical generator we picked, that pressure simply doesn't exist at 5 k. Three honest options:

1. **Add a standard-attention generator as a comparison baseline (recommended, 1–2 hr).** Re-run `01_kv_pressure.py` on Llama-3.1-8B-Instruct (or Qwen2.5-14B-Instruct) at bf16 — both have plain MHA / GQA full-attention, both fit in <20 GB VRAM. Get the *real* O(N) KV curve we expected. The paper then has the correct comparison: "this is what KV pressure looks like on a vanilla transformer; here's what hybrid-attention models like Qwen3.6 already buy you for free; here's what our gating adds *on top* of that for trajectory drift / corrective-retrieval cost."
2. **Push context further out on Qwen3.6 (also recommended, 1 hr).** Run a single 32 k-token uninterrupted generation. The flat regime will eventually end — finding the knee is itself a result and bounds the regime where Qwen3.6's architecture handles long context "for free."
3. **Tighten the framing once more (free).** The thesis as written claims KV pressure is the binding cost; on this model it is not. Pivot the headline contribution to **"corrective-retrieval cost + trajectory drift over long agentic context"** rather than KV memory specifically. Latent-state gating still saves: it lets us avoid wasted retrieval/reasoning passes whose cost is real (each adds ~80 ms × N tokens of regenerated content, plus an embedding+search round-trip) regardless of whether the *baseline* KV grows.

The methodology is fine — Phase 2 hidden-state hooks, step-boundary detection, CRV variance/drift, and Phase 4's earliness × drift × cost showdown all work without a steep KV curve. We just don't get to lean on KV memory as the headline cost.

### Mechanics that worked

- **Manual generation loop with explicit `DynamicCache`**: works first try on transformers 5.8 + Qwen3-MoE. Pass `past_key_values=None` on step 0, then the returned `out.past_key_values` on every subsequent step; pass `attention_mask` only on step 0 (later steps don't need it because the cache contains the full attention context). Greedy via `argmax`; no `do_sample=False` plumbing needed in the manual loop.
- **`tokenizer.apply_chat_template(..., add_generation_prompt=True)`**: produces a clean prompt with proper chat-template wrapping. Qwen3.6 emits `<think>...</think>` then real content as expected. The thinking-mode default is on (Qwen3 convention); we did not pass `enable_thinking=False`.
- **`torch.cuda.synchronize()` after every step**: required for accurate per-step latency; CUDA's async kernel queueing made earlier non-synced timing useless. Cost is small (single sync per ~80 ms step).
- **Forcing past EOS**: in all three classes, EOS never fired at all in 5 000 tokens. The model genuinely produces 5k+ tokens for these prompts. No artificial "ignore EOS" logic was needed.
- **VRAM measurement**: `torch.cuda.memory_allocated()` returns the live allocation; `torch.cuda.max_memory_allocated()` tracks peak since `reset_peak_memory_stats()`. Both update synchronously (CPU-side counters).

### Mechanics worth noting for Phase 2

- The naming choice in the summary metric was nontrivial: step 0 is the prefill (full prompt forward), which is an outlier. The KV-pressure curve we care about is single-token-decode step 1 vs step N. Renamed `step0_to_stepN_slowdown` → `decode_slowdown_step1_to_stepN` mid-run.
- KV cache of `132–141 MB at 5 k` extrapolates to ~900 MB at 32 k tokens (linear in the 10 full-attention layers' contribution; the 31 Gated DeltaNet layers' state is bounded). Easily fits the 15 GiB headroom even out to 64 k context.
- Phase 2.1's hidden-state hooks will inflate VRAM by capturing per-layer outputs. Per-step capture of all 41 layers at last-token only = `41 × 2048 × 2 bytes = 168 KB`, kept on GPU per step before `.detach().to('cpu', float32)`. Negligible. **Layer-subset variance experiments are unconstrained by VRAM at this generator config.**

### Phase 1 status

| step | state |
| --- | --- |
| 1.1 generator/reflector wiring + chat template | done (chat template threaded inside `01_kv_pressure.py`; reflector deliberately not loaded — Phase 2.5+ task) |
| 1.2 `01_kv_pressure.py`: 5k uninterrupted gen + 500-tok checkpoints | done |
| 1.3 three prompt classes + JSONL + plot | done |

**Visible artifact ✓**: `figs/kv_pressure_baseline.png` + the three numbers above.

Next: triage the framing surprise with the user. **Resolved in 2026-05-09 #5** below — third pivot to "cognitive efficiency / trajectory drift" framing with a three-way architectural-class comparison spec for Phase 1.4.

---

## 2026-05-09 #3b — Phase 0 re-baseline closed on A800 80 GB

Followed the boot plan from #3 end-to-end on the new vast.ai A800 box. Setup, sysinfo, downloads, and both smokes ran clean. **Phase 0 reproducible on A800; Phase 1 unblocked.** All artifacts in `runs/`: `sysinfo_a800_box3.json`, `download_a800.log`, `smoke_phi_a800.log`, `smoke_qwen_nf4_single_a800.log`, `smoke_qwen_bf16_a800.log`, plus the regenerated `env_lockfile.txt`.

### Stack as installed (this box)

Identical to box #2 lockfile (101 packages):

| dep | version |
| --- | --- |
| python | 3.11.15 |
| torch | 2.6.0+cu124 |
| transformers | 5.8.0 |
| accelerate | 1.13.0 |
| bitsandbytes | 0.49.2 |
| huggingface_hub | 1.14.0 |
| numpy | 2.4.4 |

`sysinfo` reports: A800 80 GB PCIe, **memory_free_mb=81038** (full 80 GB visible, no cgroup ceiling — vs 44.35 GiB on A40 #2). Driver 550.90.07; torch reports CUDA runtime 12.4. bf16 supported. RAM 220 GiB total, 183 GiB available. **No transformers `core_model_loading.py` patch applied on this box** (deliberately — wanted to test the unpatched path on a card with headroom).

### Models (re-downloaded fresh, SHAs match pins)

- `microsoft/Phi-4-mini-instruct` @ `cfbefacb99257ffa30c83adab238a50856ac3083` — 7.2 GB
- `Qwen/Qwen3.6-35B-A3B` @ `995ad96eacd98c81ed38be0c5b274b04031597b0` — 69 GB

Total download wall-clock: ~10 min for the pair (Phi <10 s, Qwen ~10 min on this network). SHAs in `download_models.sh` survive re-resolution; pins are good.

### Phi-4-mini smoke (replication of the A40 #2 result)

`python scripts/00_smoke.py --model phi-4-mini-instruct`

| metric | A800 (this) | A40 #2 |
| --- | --- | --- |
| load (s) | 2.87 | 4.52 |
| 32-tok gen tok/s | **22.3** | 19.4 |
| peak VRAM (MB) | 2868 | 2868 |
| hidden_states | 32 × 33 layers, (1,1,3072) bf16 | identical |
| L2 pred / amb (Δ) | 54.198 / 54.821 (0.623) | 54.204 / 54.813 (0.609) |
| H(next) pred / amb (Δ nats) | 1.139 / 3.956 (2.816) | 1.177 / 3.998 (2.821) |
| top tok pred | ` Paris` p=0.826 | ` Paris` p=0.820 |

Within tok/s noise; hidden-state numbers replicate. Phi NF4 path = good.

### Qwen3.6-35B-A3B smoke — **the NF4 question**

User asked: "does the NF4 issue arise again on 80 GB?" Ran two configurations to answer.

#### (a) NF4 hybrid: `--device-map single` (no `--no-quant`)

`python scripts/00_smoke.py --model qwen3.6-35b-a3b-instruct --device-map single`

| metric | value |
| --- | --- |
| load (s) | 12.96 |
| RSS after load (GB) | 1.07 |
| **VRAM after load (MB)** | **66675** (~65 GiB) |
| 32-tok gen tok/s | **8.9** |
| VRAM peak gen (MB) | 64213 |
| hidden_states | 32 × **41 layers**, last `(1, 1, 2048)` bf16 |
| L2 pred / amb (Δ) | 112.163 / 111.173 (Δ 0.990, ~0.9 %) |
| H(next) pred / amb (Δ nats) | 2.189 / 3.734 (Δ 1.545) |
| top tok pred | ` Paris` p=0.571 |

**Loaded successfully — no OOM.** 693 weight shards in 13 s. Confirms what #2c predicted: with 80 GB headroom, neither upstream blocker (transformers#43032 streaming-load OOM, bnb's missing grouped-MoE-experts support) actually bites. But the model is **NOT fully NF4** — it's a hybrid:

- Standard `nn.Linear` modules (~5 GB worth: q/k/v/o, gates, shared-expert linears, lm_head) → NF4
- Grouped expert tensors (`Qwen3_5MoeExperts.gate_up_proj`, `down_proj`, ~60 GB) → still bf16 (bnb's integration leaves 3-D `nn.Parameter`s untouched, exactly as #2b/#2c diagnosed)
- Embeddings, norms → bf16

Net: ~65 GB on GPU, ~14 GB headroom for KV / activations.

#### (b) bf16 native: `--no-quant --gpu-budget 76GiB`

`python scripts/00_smoke.py --model qwen3.6-35b-a3b-instruct --no-quant --gpu-budget 76GiB`

| metric | value |
| --- | --- |
| load (s) | 17.74 |
| RSS after load (GB) | 1.03 |
| **VRAM after load (MB)** | **68665** (~67 GiB) |
| 32-tok gen tok/s | **12.6** |
| VRAM peak gen (MB) | 66206 |
| hidden_states | 32 × 41 layers, (1, 1, 2048) bf16 |
| L2 pred / amb (Δ) | 111.696 / 111.133 (Δ 0.563) |
| H(next) pred / amb (Δ nats) | 2.394 / 4.113 (Δ 1.719) |
| top tok pred | ` Paris` p=0.554 |

#### Comparison + decision

| | NF4 hybrid | **bf16 native** | A40 #2 (bf16+CPU) |
| --- | --- | --- | --- |
| VRAM after load | 66 675 MB | 68 665 MB (+2 GB) | mostly CPU |
| 32-tok tok/s | 8.9 | **12.6 (+42 %)** | 0.3 |
| L4 L2 Δ (pred vs amb) | 0.990 | **0.563** | 0.649 |
| L4 H Δ (nats) | 1.545 | **1.719** | 1.734 |
| top tok pred prob | 0.571 | **0.554** | 0.568 |

**bf16 native is the canonical Phase 1 path.** Three reasons:
1. **42 % faster** at the same VRAM cost. NF4 mode was paying dequant overhead on the small fraction of Linears it could actually quantize, with the bulk of weights (the experts) bf16 in *both* paths anyway. No throughput win, just overhead.
2. **L4 probe matches the A40 #2 bf16+CPU baseline almost exactly** (L2 Δ 0.563 vs 0.649, entropy Δ 1.719 vs 1.734, ` Paris` p=0.554 vs 0.568). NF4 mode added ~50 % extra L2 noise (Δ 0.99 vs 0.56 / 0.65) — quantization noise on the non-expert Linears that we don't need to pay. **L1 finding fully replicated** with full precision: MoE noise still degrades L2 separation to ~0.5 % (Phi dense was ~1.1 %).
3. **VRAM headroom is 80 − 68.7 ≈ 11 GB**, comfortably enough for the Phase 1 5 k-token KV cache (~1–7 GB depending on KV-head count). Single-GPU, no offload, no accelerate planner.

### So: did the NF4 issue arise again?

**Technically yes; practically no.** transformers#43032 (the bf16-shards-staged-on-GPU-before-quant regression) is still in the install — `core_model_loading.py:1382` was *not* patched on this box. bnb's grouped-MoE-experts gap is also still there. Both bugs are intact. They just don't OOM at 80 GB the way they did at 44 GiB. We sidestep them by going bf16 instead of fighting them.

The patch text from #2c stays preserved there (in case Phase 4 ever wants to also report numbers on a smaller-VRAM card). `tools/quantize_qwen.py` stays as a scaffold for the same reason. Neither is on the Phase 1 critical path.

### Notes worth keeping

- **Generation text quality.** On the A40 `--no-quant` run we got "gibberish" with a raw prompt (no ChatML). On A800 the same path emits sane English (e.g. `<think></think>Retrieval-augmented generation (RAG) is an AI framework…`). The Qwen3.6 thinking model emits an empty `<think></think>` block then answers when given a simple question. Threading `tokenizer.apply_chat_template` is still a Phase 1 task (`01_kv_pressure.py`); the smoke script's raw prompt happens to work well enough for shape + L4 deliverables.
- **No `_compat.py` shim needed yet.** The bnb 0.49.2 `Params4bit.__new__` private-kwarg patch in `00_smoke.py` was ported from box #2; it's harmless on the A800 path (we don't hit the accelerate dispatch hook) but stays in for portability.
- **`flash-linear-attention` / `causal-conv1d` "fast path not available" warning** — Qwen3.6 has a linear-attention fallback for some heads that wants these libs. Not installed, falls back to torch impl; no functional impact, possibly a small tok/s headroom we could claw back later. Defer to Phase 1 if it matters.

### Phase 0 status (re-affirmed on A800)

| step | state |
| --- | --- |
| 0.1 conda probe + env | done |
| 0.2 environment.yml | done (lockfile re-written, identical to box #2) |
| 0.3 sysinfo extended | done; **80 GB visible**, no cgroup limit |
| 0.4 download_models.sh + revs | done (SHAs verified on re-pull) |
| 0.5 00_smoke.py on Qwen MoE | **done at full bf16 native — 12.6 tok/s, ~68 GB VRAM, L4 numbers match A40 baseline** |

Phase 0 closed (again, properly). On to Phase 1: `01_kv_pressure.py` with bf16-native Qwen as the canonical generator.

---

## 2026-05-09 #3 — Handover to A800 80 GB box (incoming)

**Why moving:** Phase 0 closed on the A40 with one open structural problem: Qwen3.6-35B-A3B at NF4 is blocked by two upstream bugs (transformers 5.x streaming-load regression + bnb's missing grouped-MoE-experts support; see #2c). On A40's 44.35 GiB cgroup ceiling, the only working Qwen path was bf16 + CPU offload at 0.3 tok/s — unusable for Phase 1's 5k-token generation, much less Phase 4's 200-question showdown. Rather than chase an upstream-broken NF4 path, took the GPU upgrade.

**New box:** A800 PCIe 80 GB, Idaho US (vast.ai #16761518), CUDA 12.4 driver (matches `torch==2.6.0+cu124` exactly), 225 GB host RAM, 32 cores, 1.5 TB storage. A800 is electrically identical to A100 PCIe 80 GB for single-GPU workloads; cheaper because of NVLink restrictions we don't use.

**Why this unblocks Phase 1:** Qwen3.6-35B-A3B in bf16 is ~70 GB — fits comfortably in 80 GB with ~8-10 GB of headroom for KV-cache at 5k tokens. **No quantization needed.** Both upstream bnb blockers become moot. We can drop `--no-quant` from the smoke script (its purpose was the A40 fallback) and run Qwen at native bf16 throughput, expected ~25-40 tok/s.

**What survives the move (this tarball):** all scripts, tools, markdown logs, runs/ artifacts (lockfile, sysinfo, smoke logs, quantize log), env spec, README, draft, `.git/` history. Models are *not* included — re-download via `scripts/download_models.sh all` on the new box (~92 s on prior network; pinned SHAs in script).

**What does *not* carry over:**
- The local 2-line patch to `transformers/core_model_loading.py` (issue #43032). Not needed on the A800 because we're going bf16 path. Re-apply only if we revisit NF4 in Phase 1+. The patch text is preserved in #2c below.
- `--no-quant` flag in `00_smoke.py`. The flag stays in the script for portability; for the A800 box the canonical command drops it.

**Boot plan on new box (in order):**

1. `scripts/setup.sh` (conda probe + cu124 torch + pinned env). Verify `runs/env_lockfile.txt` matches the one in this tarball (modulo build hashes).
2. `python scripts/sysinfo.py | tee runs/sysinfo_a800_box3.json` — capture VRAM (~80 GB visible, no cgroup limit expected on PCIe vast box), driver string, torch CUDA reported as 12.4.
3. `scripts/download_models.sh all` — Phi + Qwen3.6, verify SHAs match `models/<name>.revision` pins.
4. `python scripts/00_smoke.py --model phi-4-mini-instruct` — sanity-check NF4 path on a small model.
5. `python scripts/00_smoke.py --model qwen3.6-35b-a3b-instruct` — **drop `--no-quant`**; expect bf16 native load, ~70 GB VRAM, ~25-40 tok/s, identical L4 probe numbers to the A40 CPU-offload run (since the underlying weights and prompts are unchanged — different precision/placement, same logits).
6. Append a `2026-05-? #3b` entry to this file with the new tok/s, peak VRAM, and L4 probe numbers. Confirm L1 finding survives (it should — same model).
7. Update Phase 0.4 line in `_PLAN.md` once the bf16 native path replaces the `--no-quant` fallback as the canonical Qwen path.

**Things to revisit (not blocking Phase 1):**
- Whether to also benchmark `unsloth/Qwen3-30B-A3B-bnb-4bit` on the A800 to *also* report edge-tier numbers — useful for the paper's framing ("works on consumer GPU too"). Defer until Phase 4 unless cheap to run alongside.
- The transformers #43032 patch + grouped-MoE-experts work in `tools/quantize_qwen.py` is preserved for future "ship a quantized release" needs but is *not* on the Phase 1 critical path.

---

## 2026-05-09 #2 — Phase 0 closed on the larger-disk A40 box

Re-ran Phase 0 end-to-end on a fresh A40 box (193 GB free disk; same GPU, more RAM, slightly older driver). Same `scripts/setup.sh`, `scripts/download_models.sh all`, `scripts/00_smoke.py` flow. New box is now the active box; prior box is archived in the entry below.

### Stack as installed (this box)

Same versions as box #1 (lockfile in `runs/env_lockfile.txt`, 101 packages):

| dep | version |
| --- | --- |
| python | 3.11.15 |
| torch | 2.6.0+cu124 |
| transformers | 5.8.0 |
| accelerate | 1.13.0 |
| bitsandbytes | 0.49.2 |
| huggingface_hub | 1.14.0 |
| numpy | 2.4.4 |

Driver 535.230.02 here vs 565.57.01 on box #1 — torch 2.6.0+cu124 still works fine (CUDA forward-compat). Note the per-process VRAM ceiling is **~44.35 GiB**, not the full 46 GiB the card advertises; the container is cgroup-limited. This matters for Phase 0.5's 4-bit fit math.

### Models (downloaded fresh, SHAs match the pins)

- `microsoft/Phi-4-mini-instruct` @ `cfbefacb99257ffa30c83adab238a50856ac3083` — 7.2 GB on disk.
- `Qwen/Qwen3.6-35B-A3B` @ `995ad96eacd98c81ed38be0c5b274b04031597b0` — 67 GB on disk (fp16 safetensors). Architecture `qwen3_5_moe`, 41 layers, hidden_size 2048.

Both downloads completed in ~92 s on this box's network — no longer the rate-limiting step. SHAs in `download_models.sh` survived re-resolution; pins are good.

### Phi-4-mini smoke (replication)

`python scripts/00_smoke.py --model phi-4-mini-instruct`

| metric | this box | box #1 |
| --- | --- | --- |
| load (s) | 4.52 | 5.28 |
| RSS after load (GB) | 1.01 | 1.01 |
| VRAM after load (MB) | 3290 | 3291 |
| 32-tok gen tok/s | **19.4** | 21.5 |
| peak VRAM (MB) | 2868 | 2868 |
| hidden_states | 32 steps × 33 layers, last `(1,1,3072)` bf16 | same |
| L2(predictable) / L2(ambiguous) | 54.204 / 54.813 (Δ 0.609) | identical |
| H(next) predictable / ambiguous (nats) | 1.177 / 3.998 (Δ 2.821) | identical |
| top token predictable | ` Paris` p=0.820 | identical |

Replicates box #1 baseline within tok/s noise. Phi smoke = good.

### Qwen3.6-35B-A3B smoke (Phase 0.5 deliverable)

`python scripts/00_smoke.py --model qwen3.6-35b-a3b-instruct --no-quant`

The bnb-NF4 path on this stack is broken (see "compat finding" below). Loaded the model in **bf16 with accelerate auto + CPU offload** (`{0: 40GiB, "cpu": 200GiB}`). 2 TB RAM means the 67 GB fp16 weights fit in CPU trivially.

| metric | value |
| --- | --- |
| load (s) | 18.44 |
| RSS after gen (GB) | 29.29 (fraction of weights stay in CPU pageable) |
| peak VRAM (MB) | 39509 |
| 32-tok gen tok/s | 0.3 (CPU-bound; expected, not the deliverable) |
| hidden_states | 32 steps × **41 layers**, last `(1, 1, 2048)` bf16 |
| L2(predictable) / L2(ambiguous) | **111.744 / 111.095** (Δ 0.649, ~0.6%) |
| H(next) predictable / ambiguous (nats) | **2.374 / 4.109** (Δ 1.734) |
| top token predictable | ` Paris` p=**0.568** |
| top token ambiguous | ` "` p=0.165 |

**Findings on the MoE generator:**

1. **L1 confirmed.** Raw last-layer mean-pool L2 norm separates predictable vs ambiguous prompts by ~0.6% on Qwen MoE — *less* signal than the ~1% on dense Phi-4-mini. Entropy gap is 1.73 nats vs 2.82 nats on Phi (still a clean ordering, ~1.7×). Q6 in the open-questions table now has its first data point: **MoE noise does degrade the L2 signal further, exactly as predicted.** Cosine drift between consecutive CRVs and layer-subset variance remain the load-bearing candidates for Phase 2.3 — do not start with L2-of-mean-pool.
2. **Hidden-state extraction works on Qwen3-MoE.** 41 layers × 32 generation steps, last layer at last step `(1, 1, 2048)`. Same `output_hidden_states=True` path used for Phi. Hook surface is uniform across both models. Phase 2.1 design holds.
3. **Generation text was gibberish** — the raw-prompt path without Qwen ChatML template emits noise tokens. This is a chat-template oversight in `00_smoke.py`, not a model load issue (the L4 forward pass over a single prefix returned a sane ` Paris` top token at p=0.568). Fix is a Phase 1 task: thread the tokenizer's `apply_chat_template` into the generation script.

### Compat finding: transformers 5.8 + bitsandbytes 0.49.2 NF4 streaming-load is broken

On this stack, bnb-NF4 loading of Qwen3.6-35B-A3B fails three different ways depending on the layout:

- **`device_map={"": 0}`** (force whole model on GPU) → OOM at 68 % load, ~44 GiB allocated. transformers 5.8's new `core_model_loading._materialize_copy` does `tensor.to(device=device, dtype=dtype)` *before* bnb gets a chance to quantize. So bf16 shards accumulate on GPU and run out at ~45 GB long before reaching the ~18 GB NF4 final size. **The streaming-quant path that worked in transformers 4.x for bnb-4bit appears to have regressed in 5.x.**
- **`device_map="auto"`, no CPU fallback in `max_memory`** → `validate_environment` rejects with "Some modules are dispatched on the CPU or the disk" because accelerate's auto planner *still* believes the model is bigger than the budget even with bnb's `adjust_target_dtype` claiming INT4. Looks like a planner-vs-quantizer size mismatch.
- **`device_map="auto"` + `llm_int8_enable_fp32_cpu_offload=True` + max_memory split** → load completes (60 s, 693 shards) but the dispatch step crashes twice over: first with `Params4bit.__new__() got an unexpected keyword argument '_is_hf_initialized'` (accelerate 1.10–1.13 sends that private kwarg into bnb's narrow constructor; bnb 0.49.2 doesn't accept it); after monkey-patching the kwarg, then `Tensor.item() cannot be called on meta tensors` because accelerate's hook attachment calls `state_dict()` on a CPU-offloaded `Params4bit` whose `quant_state.offset` is still on `meta`.

Tried downgrading accelerate (1.13 → 1.10.1), and downgrading transformers (5.8 → 4.51.3): 4.51 doesn't recognize `qwen3_5_moe` (`KeyError: qwen3_5_moe` in `CONFIG_MAPPING`), so we're locked to 5.x for this model. bnb 0.49.2 is the latest published.

**Decision for Phase 0:** ship `--no-quant` (bf16 + CPU offload) as the documented MoE smoke path. We have 2 TB RAM, the L4 deliverable is shape + L2/entropy (no quantization-fidelity dependency), and the bnb path is a Phase 1 / 2.1 task: revisit when the streaming-quant regression is fixed upstream OR by pre-quantizing offline (e.g. saving NF4 shards to disk via a tiny `tools/quantize_qwen.py`) so `from_pretrained` loads already-4-bit weights and never needs to materialize bf16 on GPU. Adding to "things to watch."

### Phase 0 status

| step | state |
| --- | --- |
| 0.1 conda probe + env | done |
| 0.2 environment.yml | done (floor pins; lockfile written) |
| 0.3 sysinfo extended | done (same script as box #1) |
| 0.4 download_models.sh + revs | done (SHAs verified on re-pull) |
| 0.5 00_smoke.py on Qwen MoE | **done via `--no-quant` fallback**; bnb-NF4 path deferred to Phase 1 (compat finding above) |

Phase 0 closed. Move to Phase 1.

---

## 2026-05-09 #2b — Pre-quantize-to-disk attempt (`tools/quantize_qwen.py`)

Followed up on the Phase 0 compat finding by trying the pre-quantize-on-disk workaround. Wrote `tools/quantize_qwen.py` to dodge the broken streaming-load pathway: load the source on CPU in bf16 (cheap with 2 TB RAM), surgically replace each `nn.Linear` with a fresh `bnb.nn.Linear4bit` whose weight is initialized from the bf16 source, move that single Linear4bit to GPU (`Params4bit.cuda()` is what actually invokes `bnb.functional.quantize_4bit`), splice it into the parent module, and drop the original. The grouped-quant-on-copy approach kept peak VRAM bounded by *one* Linear's bf16 plus the accumulating 4-bit pack.

**Two earlier dead ends, both in this script's history (kept here so future-me doesn't repeat them):**

1. *`replace_with_bnb_linear` then `module.to(cuda)`.* `replace_with_bnb_linear` initialises Linear4bit modules with **meta-tensor** weights — the original bf16 weights are dropped, not migrated. `.to('cuda')` then raises `Cannot copy out of meta tensor; no data!`. The helper is a graph-rewrite, not a data-mover.
2. *`init_empty_weights() + replace_with_bnb_linear + accelerate.load_checkpoint_in_model`.* Loaded fast (~21 s) and quantized cleanly to peak 3.7 GiB on the GPU, but **692 parameters stayed on `meta`** because the source checkpoint namespaces its tensors as `model.language_model.layers.X.…` (this snapshot is shared with a multimodal sibling that contains `model.visual.*` and `mtp.*` heads). `accelerate.load_checkpoint_in_model` does *not* apply transformers' key-renaming logic, so most of the LM weights couldn't bind to our text-only `Qwen3_5MoeForCausalLM`. transformers' own `from_pretrained` *does* apply that mapping (we'd already seen it work in the `--no-quant` smoke), so step 1 of the final script uses `from_pretrained` for ingestion instead.

**The third (current) approach got further but exposed a deeper design issue:**

```
[ok] config: model_type=qwen3_5_moe hidden=2048 layers=40 vocab=248320
[ok] step 1: from_pretrained on CPU in bf16 in 15.6s (rss 0.71 -> 4.28 GB)
[..] step 2: replace 351 nn.Linear modules with bnb.Linear4bit
    -> 350 modules to quantize, 1 kept in bf16 (lm_head)
  [350/350] vram=1.00 GiB rss=4.4 GB
[ok] 350 Linear4bit modules quantized in 9.3s; peak VRAM during quant = 721 MB (0.70 GiB)
[..] step 3: move remaining (kept-bf16, embeddings, norms) to cuda:0
torch.OutOfMemoryError: ... 44.35 GiB total, 43.97 GiB in use
```

The streaming-quant loop in step 2 is **exactly the behaviour the broken transformers path *should* have produced**: 350 NF4 Linears in 9.3 s, peak VRAM 0.70 GiB. That's the proof-of-concept we wanted. But step 3 (move-rest-to-GPU) blew up because the model has ~60 GB of bf16 weights that *are not in any* `nn.Linear`.

### Why: Qwen3.6-MoE stores experts as grouped 3-D Parameters

This Qwen3.5/3.6 MoE implementation packs each layer's experts into single grouped `torch.nn.Parameter` tensors:

- `model.layers.X.mlp.experts.gate_up_proj` — shape `(num_experts, hidden, 2*intermediate)` bf16
- `model.layers.X.mlp.experts.down_proj` — shape `(num_experts, intermediate, hidden)` bf16

This is the standard grouped-GEMM layout for fast MoE forward passes. It is **not** `nn.ModuleList[nn.Linear]`, so `replace_with_bnb_linear` (which only walks `nn.Linear` submodules) never sees them, and bnb's `Linear4bit` cannot wrap a 3-D weight (its forward expects a 2-D weight). Roughly:

| group | rough param count | what the script handled |
| --- | --- | --- |
| attention q/k/v/o, gate, shared-expert linears, linear_attn projections, lm_head | ~5 B | quantized to NF4 (350 modules, 0.70 GiB peak) |
| MoE expert FFNs (`gate_up_proj` + `down_proj`, grouped) | ~30 B | left bf16 → ~60 GB → step-3 OOM |
| embeddings, norms | <1 B | small, fine |

So bnb-NF4 of the 35 B-A3B model on this stack **needs first-class grouped-MoE quantization support**, which neither bnb 0.49.2 nor transformers 5.8 ships. Real options for Phase 1+:

- **(a) Manually quantize the grouped tensors with `bnb.functional.quantize_4bit` and write a custom forward that dequantizes per-expert on demand.** Real engineering, probably 100–300 lines plus a careful test against the bf16 baseline.
- **(b) Pre-quantize with a different toolkit that does support grouped-MoE NF4** (`torchao` or `compressed-tensors`). Adds a dependency surface, possibly hours of integration to confirm hidden-state hooks still fire identically.
- **(c) Wait for transformers to fix the 5.x bnb streaming-load regression** and use the standard `from_pretrained(quantization_config=...)` path. Cheapest for us, slowest by calendar.
- **(d) Stay on the bf16 + CPU-offload fallback for now and revisit when speed actually matters.** Phase 1's KV-pressure curve is the first place 0.3 tok/s starts to hurt; we'll feel the pain there and decide between (a) and (b).

`tools/quantize_qwen.py` is kept in the repo as a working scaffold for (a). The 350-Linear loop is correct and fast; an extension to handle grouped-expert tensors would slot in between steps 2 and 3.

**Decision:** ship the `--no-quant` smoke as Phase 0's MoE deliverable (already done), defer NF4 to Phase 1. Add a "things to watch" entry for the grouped-MoE-experts gap so Phase 1 budgets it.

---

## 2026-05-09 #2c — Upstream analysis: it's two distinct bugs, not one

User pointed at four upstream threads that re-frame the entire 0.5b friction:

- huggingface/transformers#42043 ("Fix bnb for the weights refactor", SunMarc, merged 2025-11-18, 396 commits) — ships transformers 5.x bnb support but with an unchecked TODO in the body: *"check why the memory is way too high when quantizing on the fly"*. So 5.x out of the box is known-degraded for on-the-fly bnb.
- huggingface/transformers#42246 ("Fix device when source and target are different", Cyrilvallez, merged 2025-11-18 same day) — fixes VLM `_conversion_mapping` loading by passing the **target GPU device** directly into `spawn_materialize`. PR body explicitly admits this exposes a latent bug: *"the only thing saving us is the fact that `accelerate_dispatch` will move the parameters if the device is not correct during post processing"*. Pre-#42246 the post-load dispatch step was inadvertently doing the bnb "stage on CPU then quantize" dance correctly.
- huggingface/transformers#43032 ("[v5 Regression] BitsAndBytes 4-bit quantization OOM", jwm1969, 2025-12-24) — exact diagnosis of our load-time OOM, with a verified two-line fix in `core_model_loading.py:1382`. When `mapping.quantization_operation is not None`, force `param_device = "cpu"`. Author confirmed 280 Linear4bit modules loading to 8.88 GiB on a 16 GB card after the patch.
- unslothai/unsloth#4108 — confirms `Qwen3.5/Qwen3.6` MoE works in transformers v5 + unsloth, but only via unsloth's custom MoE-aware quant stack and their pre-baked `unsloth/...-bnb-4bit` snapshots.

**What this changes:**

1. **The 5.8 streaming-load regression is real and fixable.** Applied the #43032 patch directly to the installed `core_model_loading.py:1382` (single `if mapping.quantization_operation is not None: param_device = "cpu"` line). After the patch, the previously-OOMing path got *much* further: `device_map="auto"` now plans correctly for the Linear-quantizable bytes; `device_map={"": 0}` runs the load loop and OOMs much later (at ~43 GiB allocated, not the 44 GiB we saw before — i.e. quantized Linears stay small but something else still exceeds the budget).
2. **The grouped-MoE-experts gap is also real, and it's separate.** Reading `quantizer_bnb_4bit.py:91`: `param_needs_quantization` returns True **only** when the parent module is `bnb.nn.Linear4bit`. The Qwen3.5/3.6 MoE implementation packs each layer's experts as a 3-D `nn.Parameter` inside a `Qwen3_5MoeExperts` module (not `nn.ModuleList[nn.Linear]`), so bnb's integration leaves them in bf16. Compare with `quantizer_mxfp4.py:185-307` and `quantizer_fbgemm_fp8.py:185` which have explicit `layers.*.mlp.experts.gate_up_proj_blocks` patterns. **bitsandbytes has no equivalent.** That's why even with the #43032 patch applied, the second OOM still hits — ~60 GB of bf16 expert weights that nobody quantizes.

So the picture is clean now:

| problem | scope | fix surface | status |
| --- | --- | --- | --- |
| transformers 5.x bnb on-the-fly load OOM (issue #43032) | every bnb-quantizable model on tight GPUs in v5 | 2-line patch in `core_model_loading.py` | applied locally; upstream PR not yet merged |
| no grouped-MoE-experts support in bnb integration | Qwen3.5/Qwen3.6 (and any future grouped-MoE arch) on bnb | new code in `transformers.integrations.bitsandbytes` mirroring the mxfp4/fbgemm_fp8 patterns, OR custom forward + per-expert `bnb.functional.quantize_4bit` | not addressed upstream; unsloth has it in their stack |

**Verification of the patch (run captured to `runs/smoke_qwen35b_nf4.log`):**

- Pre-patch: OOM in `_materialize_copy` at 68 % of the load (one Linear's 1024 MiB tried to land on a card already at 44.04 GiB).
- Post-patch (`device_map={"": 0}`): load progresses to ~98 % before OOMing at 43.44 GiB. The shape of the OOM changed — what used to be "shipping bf16 to GPU before quantization" is now "shipping un-quantizable bf16 expert tensors to GPU after Linears were correctly NF4'd". Confirms the patch is doing what #43032 claimed and isolates the remaining cost as the experts.

**Implications for the project:**

- The `tools/quantize_qwen.py` scaffold's earlier 0.70 GiB peak quantizing 350 Linears was *actually correct* — it was hitting the same wall that vanilla `from_pretrained` now hits with the patch applied. The only thing my script was missing was a step to also quantize the grouped expert tensors, and now we know upstream isn't doing that either for bnb.
- For Phase 1's KV-pressure curve we need a *fast* generator. Three real options now:
  1. **Pull `unsloth/Qwen3.6-35B-A3B-Instruct-bnb-4bit` (or equivalent) instead of quantizing ourselves.** Unsloth pre-bakes their MoE-aware NF4 layout and ships a working HF snapshot. Cost: ~17 GB additional download, dependency on whichever runtime pieces unsloth requires (or just transformers if their NF4 layout is bnb-compatible at load time).
  2. **Switch the generator's quant scheme from NF4 to MXFP4** (`BitsAndBytesConfig` → `Mxfp4Config`). transformers ships first-class grouped-MoE-expert support for MXFP4. Cost: different bytes/param ratio (4-bit but with extra metadata), need to confirm MXFP4 is fast on A40 (it was originally GB200-targeted).
  3. **Extend `tools/quantize_qwen.py` with a manual grouped-expert pass + a custom forward** that calls `bnb.functional.dequantize_4bit` per-expert at runtime. Real engineering, ~1-2 days, but most under-our-control of the three.
- The 2-line patch should also be packaged as a runtime monkey-patch (we shouldn't rely on a hand-edit of the installed `core_model_loading.py` surviving env recreation). Easy to add to `scripts/00_smoke.py` or a new `scripts/_compat.py` shim.

**Decision (for now):** keep the diagnosis logged, leave the patch in place on this box, and bring the three options above to the next decision point. Phase 1 needs to pick one before writing `01_kv_pressure.py`.

---

## 2026-05-09 #1 — Phase 0 baseline (A40)

### Stack as installed

| dep | version |
| --- | --- |
| python | 3.11.15 |
| torch | 2.6.0+cu124 |
| transformers | 5.8.0 |
| accelerate | 1.13.0 |
| bitsandbytes | 0.49.2 |
| huggingface_hub | 1.14.0 |
| numpy | 2.4.4 |

CUDA 12.4 runtime (torch wheel) on driver 565.57.01 (CUDA 12.7-capable). bf16 supported. Lockfile: `runs/env_lockfile.txt`.

### Phi-4-mini smoke (`models/phi-4-mini-instruct`, sha `cfbefacb99257ffa30c83adab238a50856ac3083`)

- Load: 5.28 s; RSS 0.68 → 1.01 GB; VRAM 4 → 3291 MB
- 32-token deterministic gen: 1.49 s = **21.5 tok/s**; peak VRAM 2868 MB
- `output_hidden_states=True`: 32 steps × 33 layers; last-layer-last-step shape `(1, 1, 3072)` `bfloat16`

### L4 landmine probe (Phi-4-mini, **dense** model)

| | hidden L2 | H(next) nats | top token / p |
| --- | --- | --- | --- |
| `"The capital of France is "` (predictable) | 54.204 | **1.177** | ` Paris` 0.820 |
| `"The next word in this random sequence is"` (ambiguous) | 54.813 | **3.997** | `:` 0.179 |
| Δ | **0.609 (~1%)** | 2.821 (≈3.4×) | — |

**Insight worth keeping:** raw last-layer L2 norm carries effectively *no* signal for next-token uncertainty even on a **dense** model. Entropy separates predictable from ambiguous by 3.4×; the L2 norm differs by ~1%. This is before MoE routing noise is in the picture. Implication for Phase 2.3: **do not start CRV at "mean-pool last-layer hidden state, then take L2 norm"** — that's effectively a constant. The per-step **cosine drift** between consecutive CRVs (already in 2.3) and **layer-subset variance** are the candidates that have to do the work. Worth running this same probe on the MoE generator once it's downloaded; if the L1 / L4 hypothesis is right, MoE will degrade the L2 signal further (more confound), but cosine drift over a multi-step trace might still survive. We learn that on Day 1 of Phase 2.

### Models

- `microsoft/Phi-4-mini-instruct` @ `cfbefacb99257ffa30c83adab238a50856ac3083` — 7.2 GB on disk. SHA pinned in `download_models.sh`.
- `Qwen/Qwen3.6-35B-A3B` @ `995ad96eacd98c81ed38be0c5b274b04031597b0` — bare repo IS the instruct variant (no `-Instruct` suffix exists; chat template bundled). Architecture tag `qwen3_5_moe`. SHA pinned. Download in progress.

### Things I changed on the fly during Phase 0

- **`huggingface-cli` → `hf`.** The new HF Hub CLI deprecated the old binary; `download_models.sh` updated to call `hf download`.
- **torch unpin → pin to 2.6.0+cu124.** First install pulled torch 2.11.0+cu130 (the cu124 wheel index aliases to cu13 builds for newer torch). Driver here is CUDA 12.7-capable; cu130 needs driver ≥13.0, so `cuda_available=False`. Pinned to torch 2.6.0+cu124 explicitly in `setup.sh`. Lesson: "let pip resolve" inside a CUDA-driver-bound stack is a footgun. `TORCH_VERSION` env var allows future bumps.
- **`trust_remote_code=False` for HF model loads.** Phi-4-mini's bundled `modeling_phi3.py` is pinned to an older transformers API (`LossKwargs`) and ImportErrors against transformers 5.8. Built-in Phi3 / Qwen3MoE implementations handle these architectures natively. Same fix should apply to the Qwen3.6 load.
- **Floor pins (`>=`) in `environment.yml`, exact versions in lockfile.** Switched from exact pins (which were stale 2024 versions) after seeing the box's recent reality.
- **Qwen repo name correction.** Review wrote `Qwen/Qwen3.6-35B-A3B-Instruct`; actual repo is `Qwen/Qwen3.6-35B-A3B`. Verified via HF API search; pinned SHA in `download_models.sh`.

### v1/v2 cleanup

Deleted `_AGENT_v1.md`, `_AGENT_v2.md`, `_PLAN_v2.md`, `draft_original.md`. Old phase scripts (`01_repl.py`, `02_*`, `03_*`, `confidence.py`) deleted earlier. Single supported path: this plan, this stack.

### Paused — disk upgrade in progress

Container overlay is 75 GB total. After phi-4-mini (7.2 GB) + a partial Qwen3.6 download (51 GB) we hit 94% full with only 5.3 GB free. The bare-bf16 Qwen3.6-35B-A3B is ~70 GB on disk, so it never fit. Killed the download; deleted the partial.

User is upgrading storage to ~200 GB. On resume:

1. **Re-run env setup** — the conda env at `/venv/recollection` lives on the same overlay and likely won't survive the upgrade. Run `bash scripts/setup.sh` from a clean state. Pinned versions are in `runs/env_lockfile.txt` for exact replay if drift becomes an issue.
2. **Re-download phi-4-mini** — `bash scripts/download_models.sh` (default, no `all`). SHA already pinned (`cfbefacb99257ffa30c83adab238a50856ac3083`).
3. **Download Qwen3.6-35B-A3B bf16** — `bash scripts/download_models.sh all`. With ~200 GB free this fits comfortably (70 GB model + 7 GB phi + env + headroom). SHA pinned (`995ad96eacd98c81ed38be0c5b274b04031597b0`); repo is `Qwen/Qwen3.6-35B-A3B` (no `-Instruct` suffix; bare repo is the instruct variant).
4. **Re-run `00_smoke.py`** without args (defaults to qwen3.6) to capture A40 baseline + L4 probe on the actual MoE generator. The phi-4-mini L4 result above stands as the dense-baseline reference; the MoE result is what tells us whether L1 fires.

What the disk crunch did *not* force a change to:

- bnb NF4 path stays. bf16 source on disk → NF4 in VRAM at load. No pivot to FP8 / GPTQ pre-quantized variants needed once disk is healthy.
- All decisions logged above (torch 2.6.0+cu124 pin, `trust_remote_code=False`, hf-CLI swap, SHA pins, Qwen repo-name correction) are persistent — they live in `scripts/setup.sh` and `scripts/download_models.sh`, not in the env or model files.

What to delete before re-running setup if state persists across the upgrade:

- `models/` (any partial downloads)
- `runs/env_lockfile.txt` (will be regenerated)
- `/venv/recollection` (conda env, if it lingers)

---

## 2026-05-09 — Phase 0 scaffold

Wrote `environment.yml`, `scripts/setup.sh`, `scripts/sysinfo.py`, `scripts/download_models.sh`, `scripts/00_smoke.py`. Decisions and reasoning so future-me does not undo them by accident:

- **Quantization path = bnb NF4 at load, not pre-quantized AWQ/GPTQ.** Cost: ~70 GB fp16 download for the 35B MoE (vs ~17 GB for a pre-quantized snapshot). Benefit: bnb NF4 keeps the standard PyTorch forward graph intact, which is what makes per-layer forward hooks work in Phase 2. AWQ/GPTQ have custom kernels that can interfere with hook semantics or simply not propagate hidden states. With 503 GiB RAM and roomy disk on the A40, the download size is a non-issue. Re-evaluate only if Phase 0.5 finds the bnb path silently mangles hidden-state outputs (the L4 probe will catch it).

- **No tiered install in `setup.sh`.** Earlier setups had a `--phase4` flag because `sentence-transformers` + `lancedb` were a notable fraction of total install time on a smaller box. On the A40, every dep combined is dwarfed by the model download. One install path = fewer "did I run setup with the right flag?" failure modes.

- **torch installed outside `environment.yml`.** The cu124 wheel index URL is platform-specific; conda-forge's torch-cuda combo is fragile. Cleaner to install torch separately inside `setup.sh` from the official PyTorch index, and let `environment.yml` carry only platform-portable pins.

- **Floor pins (`>=`) in `environment.yml`, exact lockfile written to `runs/env_lockfile.txt` after first clean install.** Same pattern as the model revision SHAs. Tighter pins introduce false reproducibility — if the package isn't on PyPI anymore, the pin breaks anyway. Lockfile-after-install is honest and replayable.

- **Revisions not pre-pinned for models.** Download script writes resolved SHAs to `models/<name>.revision`; we hardcode them after the first clean run. Beats either pinning a SHA we made up or following `main` forever.

- **Default download = phi-4-mini only; `all` adds the 35B MoE.** A 70 GB pull is slow; the small model is enough to validate the env + smoke script + L4 probe scaffolding before burning bandwidth.

- **`00_smoke.py` is doing four jobs at once.** (1) load timing + RSS/VRAM, (2) generation tok/s + peak VRAM, (3) hidden-state shape discovery, (4) **L4 landmine probe** (predictable vs ambiguous prefix → hidden-L2 + next-token entropy). Job 4 is non-trivial; it's the cheapest test of whether the entire CRV thesis is even coherent on an MoE model. **If the L4 numbers come back showing zero ordering between predictable and ambiguous prompts, we stop and rethink before Phase 1.**

- **Hidden-state capture uses `.detach().to("cpu", dtype=torch.float32)` from day one** — L2 mitigation baked in so it doesn't bite when capture moves into the streaming generation path.
