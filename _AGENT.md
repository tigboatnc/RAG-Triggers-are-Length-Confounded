# _AGENT.md

> Experimental log + status. Read top-down: status, then chronological entries.

---

## Status

### Progress

| phase | step | state | artifact |
| --- | --- | --- | --- |
| 0 | conda probe + env (HF, bnb, torch cu124) | **in progress** | `scripts/setup.sh`, `environment.yml`, `runs/env_lockfile.txt` |
| 0 | model download — Phi-4-mini + Qwen3.6-35B-A3B, revisions captured | not started | `scripts/download_models.sh`, `models/<name>.revision` |
| 0 | A40 baseline + L4 landmine probe | not started | new entry below + `_AGENT.md` baseline section |
| 1 | KV-pressure curve at 5k tokens × 3 prompt classes | not started | `runs/phase1_kv_pressure.jsonl`, `figs/kv_pressure_baseline.png` |
| 2 | hidden-state hook + step boundary + CRV + sanity probe | not started | `scripts/crv/`, `runs/phase2_crv.jsonl` |
| 3 | ReaLM-Retrieve sim + GRIP sim + random baseline | not started | `runs/phase3_triggers.jsonl` |
| 4 | earliness × trajectory-drift × cost showdown | not started | `figs/efficiency_pareto.png`, `runs/phase4_showdown.csv` |

### Hardware

- NVIDIA A40, 46 GB VRAM (CUDA 12.7, driver 565.57.01); 503 GiB RAM, 40 cores; Ubuntu 24.04 in container `b62fd22e2b17`.
- Conda at `/opt/miniforge3/bin/conda` (vast.ai-style, not on PATH); `setup.sh` probe handles it.

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

---

## 2026-05-09 — Phase 0 baseline (A40)

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
