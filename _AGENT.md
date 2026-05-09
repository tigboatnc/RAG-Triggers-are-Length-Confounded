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
| 0 | handover to A800 80 GB box (re-baseline pending) | **in flight** | entry `2026-05-09 #3` |
| 1 | KV-pressure curve at 5k tokens × 3 prompt classes | not started | `runs/phase1_kv_pressure.jsonl`, `figs/kv_pressure_baseline.png` |
| 2 | hidden-state hook + step boundary + CRV + sanity probe | not started | `scripts/crv/`, `runs/phase2_crv.jsonl` |
| 3 | ReaLM-Retrieve sim + GRIP sim + random baseline | not started | `runs/phase3_triggers.jsonl` |
| 4 | earliness × trajectory-drift × cost showdown | not started | `figs/efficiency_pareto.png`, `runs/phase4_showdown.csv` |

### Hardware

- **Next box (handover pending, 2026-05-09 #3):** NVIDIA **A800 PCIe 80 GB**, Idaho US (vast.ai #16761518). 1.52 TB/s HBM bw, **CUDA 12.4 driver (exact match for `torch==2.6.0+cu124`, zero forward-compat risk)**, 225 GB host RAM, 32 cores, 1.5 TB storage, 6-month max duration. A800 is the China-export A100 variant — same compute, same VRAM, same per-card bandwidth; only NVLink is reduced (irrelevant for single-GPU). 80 GB VRAM means **Qwen3.6-35B-A3B fits in bf16 natively** — no NF4 needed for Phase 1, sidesteps both upstream bnb blockers (#43032 streaming-load and grouped-MoE-experts gap).
- Prior active box (2026-05-09 #2): NVIDIA A40, 46 GB VRAM (driver 535.230.02, CUDA 12.6); ~2 TB RAM, 40 cores; per-process VRAM ceiling ≈ 44.35 GiB (cgroup-limited). Phase 0 closed here; Qwen smoke ran in bf16 + CPU offload @ 0.3 tok/s.
- Prior box (2026-05-09 #1): NVIDIA A40, driver 565.57.01 (CUDA 12.7), 503 GiB RAM.
- `setup.sh` conda probe handles `/opt/miniforge3/bin/conda` (vast.ai-style boxes).

### Engine

HF `transformers` + `accelerate` + `bitsandbytes` 4-bit NF4. Hidden states via `output_hidden_states=True` + forward hooks. torch installed from the cu124 wheel index inside `setup.sh`. Mac/Metal not supported.

### Models

- `microsoft/Phi-4-mini-instruct` — small-fast smoke target, also the lightweight reflector / baseline in Phase 1.
- `Qwen/Qwen3.6-35B-A3B-Instruct` — primary MoE generator (~3B active per token, ~17 GB at NF4). HF repo name taken from external review verbatim; verified on first download.

### Open questions

| id | question | resolve in |
| --- | --- | --- |
| Q1 | Does `transformers.generate` with `output_hidden_states=True` cost > 5% throughput on the MoE generator at 4-bit? If yes, write a custom forward-hook path that captures only at step boundaries. | Phase 0.4 / 2.1 |
| Q2 | Is "logical step boundary" best defined as sentence-end inside `<think>`, numbered-list item, or fixed-stride window? Sweep. | Phase 2.2 |
| Q3 | Does CRV variance or drift carry more signal? Or do we need both? | Phase 2.3–2.5, 4.4 |
| Q4 | Does the latent gate fire meaningfully earlier than ReaLM-Retrieve on real HotpotQA / MuSiQue traces? | Phase 4.2 |
| Q5 | Without ReaLM-Retrieve and GRIP weights, how honest is the simulation? Log every divergence. | Phase 3.1–3.2 |
| Q6 | Does MoE expert-switching noise dominate hidden-state variance for the predictable-vs-ambiguous probe? **L1 / L4 in `_PLAN.md`.** | Phase 0.5 |

### Things to watch

- Pin model revisions after first clean download (`models/<name>.revision` SHAs go into `download_models.sh`'s `REV_*` vars).
- Hidden-state extraction inflates VRAM (saved tensors). Verify the (4-bit weights + per-step capture) combo still fits 46 GB at 5k context.
- Keep CRV / step-boundary / trigger code engine-agnostic via a small adapter so HF/bnb internals don't leak into trigger logic.
- "Logical step boundary" is the most ambiguous primitive in the pipeline. Get it under tests early (Phase 2.2) before tuning anything downstream.
- ReaLM-Retrieve and GRIP simulation honesty: log every place the sim diverges from the original paper. Reviewers will probe this.
- Random-baseline seed: log it.
- Corrective / retrieval pass during the showdown must use the **same generator** that produced the trace, otherwise we benchmark a model swap, not a gate.
- **Two distinct bnb-NF4 problems on transformers 5.8 + Qwen3.6-MoE** (see Phase 0 #2, #2b, and #2c entries):
  1. *On-the-fly streaming-load OOM* (issue huggingface/transformers#43032). Cause: `core_model_loading.py:1382` ships bf16 shards to GPU before bnb can quantize. Fix: 2-line patch (`if mapping.quantization_operation is not None: param_device = "cpu"`). Applied to the installed `core_model_loading.py` on this box; should be re-applied as a runtime monkey-patch in `scripts/_compat.py` (or similar) so it survives env recreation. Upstream patch not yet merged.
  2. *No grouped-MoE-experts support in bnb integration.* `quantizer_bnb_4bit.param_needs_quantization` only flags `bnb.nn.Linear4bit` modules; the Qwen3.5/3.6 `Qwen3_5MoeExperts` module wraps experts as 3-D `nn.Parameter` (grouped-GEMM layout). Mxfp4 and fbgemm_fp8 quantizers have explicit grouped-expert patterns; bnb does not. Phase 1 picks one of: (a) pull `unsloth/Qwen3.6-...-bnb-4bit` pre-baked snapshot, (b) switch from bnb-NF4 to MXFP4 (first-class grouped-MoE support upstream), or (c) extend `tools/quantize_qwen.py` with a manual grouped-expert pass + custom forward.
- Qwen ChatML template not threaded through `00_smoke.py`; raw-prompt generation produces gibberish even though forward-pass logits are sane. Use `tokenizer.apply_chat_template` in Phase 1 generation scripts.
- Per-process VRAM ceiling on the **A40 box was 44.35 GiB** (cgroup-limited). On the incoming A800 80 GB PCIe box the expected ceiling is the full 80 GB; verify with `sysinfo.py` on first boot before re-doing VRAM math.
- On the A800: drop `--no-quant` from the canonical Qwen smoke command. The flag remains in the script for portability but is no longer the default path.

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
