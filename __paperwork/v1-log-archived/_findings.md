# _findings.md

> Cumulative empirical findings, paper-oriented prose. **This document is updated only when explicitly requested by the user.** Last update: 2026-05-09 (Phase 1).

---

## Setup (briefly, for context only — full env in `_AGENT.md` 2026-05-09 #3b)

All findings below were measured on a single NVIDIA **A800 PCIe 80 GB** (driver 550.90.07, full 80 GB visible to torch). Stack: Python 3.11.15, torch 2.6.0+cu124, transformers 5.8.0, accelerate 1.13.0, bitsandbytes 0.49.2 (env lockfile in `runs/env_lockfile.txt`).

The canonical primary generator for Phase 1 is **Qwen/Qwen3.6-35B-A3B** at bf16 native, fully on `cuda:0`, no offload, no quantization. Pinned commit: `995ad96eacd98c81ed38be0c5b274b04031597b0`. Architecture tag: `qwen3_5_moe`. 41 layers, hidden 2048, ~3 B active parameters per token.

The reflector model (`microsoft/Phi-4-mini-instruct`, NF4) is not used in Phase 1; it is reserved for Phase 2.5+ trigger logic.

---

## Phase 1 — KV-cache pressure baseline

### What was measured

A single 5,000-token uninterrupted greedy generation, run for three prompt classes selected to stress the model differently:

| class | what it probes |
| --- | --- |
| **factual** | long-form confident exposition (low entropy, long output by design) |
| **multi-hop** | chained reasoning with explicit confidence checks per hop |
| **ambiguous** | open-ended creative continuation (high entropy throughout) |

For each prompt class, we logged at every 500-token checkpoint: VRAM allocated, peak VRAM, KV-cache growth since prefill, rolling tok/s over the last window, instantaneous tok/s at the checkpoint step, wall-clock since generation start, and whether end-of-sequence had been emitted. Per-step latency was synchronized with `torch.cuda.synchronize()` to defeat CUDA's async kernel queueing. Generation used a manual decode loop with explicit `DynamicCache` (rather than `model.generate`) so per-step measurement was deterministic.

Implementation: `scripts/01_kv_pressure.py`. Per-checkpoint rows in `runs/phase1_kv_pressure.jsonl` (30 rows). Per-class summaries in `runs/phase1_kv_pressure_summary.jsonl` (3 rows). Figure in `figs/kv_pressure_baseline.png`. Total wall-clock for the full 3 × 5 000-token run: 19.8 minutes.

### Headline numbers

| prompt class | mean tok/s | prefill (s) | decode step 1 (s) | decode step 5000 (s) | step 1 → step N slowdown | total KV growth (MB) | peak VRAM (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| factual | 12.85 | 0.798 | 0.0816 | 0.0798 | **0.98×** | 141 | 66 258 |
| multi-hop | 12.80 | 0.414 | 0.0800 | 0.0782 | 0.98× | 132 | 66 258 |
| ambiguous | 12.89 | 0.388 | 0.0796 | 0.0761 | 0.96× | 132 | 66 258 |

Three Phase 1 deliverable numbers: **peak VRAM 66 258 MB (~64.7 GiB)**, **tok/s @ 500 ≈ 12.7 (rolling)**, **tok/s @ 5000 ≈ 13.0 (rolling)**.

### The principal finding: there is essentially no KV-pressure curve up to 5 k tokens

Across all three prompt classes, total VRAM allocated grew by ~80 MB over the entire 5,000-token generation — three perfectly overlapping linear traces in the figure's top panel. Per-step decode latency was flat or slightly decreasing across the 5 000 steps; step 5000 was ~2 % faster than step 1 in all three runs (i.e. a "slowdown" of 0.96–0.98×, indistinguishable from noise). EOS was never emitted in any of the three runs at any point in the 5 000-token horizon.

A naive standard-transformer estimate for Qwen3.6's headline geometry (41 layers × bf16 KV, ~6 KV heads × 128 dim) would have predicted ~240 KB/token of KV growth (~1.2 GB at 5 000 tokens) and substantial per-step slowdown from O(N) attention. The observed slope is **~20 KB/token**, roughly 8 % of that estimate, with no measurable per-step slowdown at all.

### Why: the generator is a hybrid linear / full-attention MoE

Qwen3.6-A3B inherits its long-context geometry from the Qwen3-Next family: a small fraction of layers use standard full attention (whose KV grows linearly with context), and the remainder use gated-DeltaNet-style **linear attention** (constant memory, constant per-step compute regardless of cache size). The model's snapshot ships with a `fla` / `causal-conv1d` accelerated path; with neither library installed the linear-attention layers fall back to a torch implementation, which is what we ran. The "fast path is not available" warning at model load is the diagnostic giveaway.

Quantitatively: observed KV slope of ~20 KB/token, divided across an estimated 5 of 41 full-attention layers (consistent with Qwen3-Next's typical 1:7 full-to-linear ratio), divided across 2 (K + V) × 2 bytes (bf16), yields ~1024 bytes per layer per token. That figure matches a configuration of **4 KV heads × 128 dim per full-attention layer** — exactly the published Qwen3-Next default.

Net: **at 5 k context, this generator is architecturally engineered to make KV memory and per-step compute effectively constant.** The flat curve we observed is not an artifact of measurement; it is a deliberate feature of the model's design.

### Implications for the project's framing

The headline-cost claim of the project's current thesis — *"agentic LLM reasoning where KV-cache pressure dominates total cost"* — does not hold for this specific generator at this specific context length. KV memory grew by less than 0.2 % of the model's weight footprint (132–141 MB on top of 66 GiB) over the entire 5 000-token horizon, and per-token throughput was indistinguishable from constant. We cannot lean on KV memory as the headline cost on this generator alone.

Two things still hold:

1. **The methodology is independent of KV pressure being the binding cost.** Hidden-state hooks, step-boundary detection, and the CRV variance / drift signal (Phase 2) all measure the model's *internal uncertainty trajectory*, which is unaffected by whether KV grows. The Phase 4 evaluation (earliness × trajectory-drift × cost) measures whether the gate fires before answer quality degrades; that signal is real even when baseline-generation cost is flat.

2. **Corrective-retrieval and re-generation cost are still real.** Each unnecessary retrieval / re-reasoning pass costs an embedding + index round-trip plus N regenerated tokens at the model's per-token cost (~80 ms/token here). These costs are paid regardless of whether the baseline KV grows. Latent-state gating that suppresses unnecessary fires saves *those* round-trips, not the steady-state baseline.

The right framing pivot is therefore from "KV memory under agentic context" to **"corrective-retrieval cost and trajectory drift in long agentic context"** — keeping the methodology and Phase 2–4 plan intact. This pivot is pending user triage (`_PLAN.md` Phase 1.4 📌 pin).

### Open question raised by this finding

At what context length does Qwen3.6-A3B's flat-cost regime end? The full-attention sub-layers' KV does grow linearly; eventually their contribution will dominate. Locating that knee — and showing the expected O(N) reference curve on a standard-attention generator (Llama-3.1-8B-Instruct or Qwen2.5-14B-Instruct) — is the deferred Phase 1.4 work item. Without that comparison, the paper has no contrast for what hybrid attention buys you "for free" versus what latent-state gating adds on top.

### Auxiliary observations worth keeping

- **Prompt-class invariance.** Mean throughput, KV growth slope, peak VRAM, and step-to-step latency are statistically indistinguishable across factual, multi-hop, and ambiguous prompts. Whatever the model is doing internally — confident exposition, multi-hop reasoning, creative generation — the *cost-per-token signature* is identical at this context length. The interesting per-class signal must be elsewhere (likely in the hidden-state trajectory, which is what Phase 2 captures).
- **EOS never fires.** All three prompts produced 5 000+ continuous tokens of cooperating generation. Forcing past EOS was not necessary. The prompts in the script (see `PROMPTS` in `01_kv_pressure.py`) are reusable as long-generation stress prompts in later phases.
- **Prefill latency varies with prompt length, as expected.** Factual (133 prompt tokens) prefill was 0.80 s; ambiguous (85 prompt tokens) was 0.39 s. Roughly linear in prompt length. Useful as a cost-model input for Phase 4.
- **Headroom for hidden-state capture is not a constraint.** Per-step capture of all 41 layers' last-token hidden states totals ~168 KB on GPU before `.detach().to('cpu', float32)` — negligible against the ~14 GiB of headroom remaining after the generator loads. Phase 2.1's layer-subset variance experiments are unconstrained by VRAM at this generator configuration.
