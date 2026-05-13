# _MODEL.md

> Internal model documentation. Triangulated from three sources at every claim:
> **(card)** = the live HuggingFace model card retrieved 2026-05-10;
> **(config)** = the on-disk `models/<name>/config.json` of the pinned snapshot;
> **(obs)** = our own measurements from Phase 0/1/2/3/4 runs and the
> `04b_protocol_smoke.py` validator. When two sources disagree, both are
> reported and the disagreement is flagged inline. Knowledge cutoff for the
> assistant building this file is well before the 2026 releases below — *do
> not* trust unverified memory; **treat the model cards as primary, validate
> against on-disk config and our own outputs.**
>
> Read alongside [`_PLAN.md`](_PLAN.md) (forward-looking design) and
> [`_AGENT.md`](_AGENT.md) (chronological experiment log). This file is the
> static "what these models are" reference; `_AGENT.md` is the dynamic "what
> we measured today."

## Why this file exists

The Phase 1.4 generator pool is three 2026-released models whose chat
templates, default decoding regimes, and attention designs differ in ways
that are easy to mis-remember. The Phase 4.1 (rejected) sweep documented in
[`_AGENT.md` 2026-05-10 #11](_AGENT.md) was a direct casualty of that
mis-remembering: Qwen ran in default-thinking mode and hit the token cap
on every question, Gemma silently ran in non-thinking mode (the only
"honest" trace), and Phi ran greedy without a system role and degenerated
into repetition loops. The Option A re-run protocol is locked in, but the
broader lesson is: **whenever the assistant touches one of these three
models, read this file first.** New box, new env, new sweep — re-run
`scripts/04b_protocol_smoke.py` and compare against the canonical numbers
in §6 below before trusting any new run.

## 1. The generator pool

| local name | HF repo | role in Phase 1.4 bracket | total params | active params | architecture | weights on disk (bf16) | pinned SHA |
| --- | --- | --- | ---: | ---: | --- | ---: | --- |
| `phi-4-mini-instruct` | `microsoft/Phi-4-mini-instruct` | **Dense SLM** | 3.8 B | 3.8 B | dense decoder, GQA, full attention | ~7.2 GB | `cfbefacb99257ffa30c83adab238a50856ac3083` |
| `gemma-4-26b-a4b-it` | `google/gemma-4-26B-A4B-it` | **Standard MoE** | 25.2 B | 3.8 B | MoE + 5:1 sliding-window-then-global hybrid | ~49 GB | `462a98a12e28e2cbcfccaf78fe41e3e50235e6ae` |
| `qwen3.6-35b-a3b-instruct` | `Qwen/Qwen3.6-35B-A3B` | **Hybrid Linear MoE** | 35 B | ~3 B | MoE + 3:1 Gated DeltaNet : full attention | ~67 GB | `995ad96eacd98c81ed38be0c5b274b04031597b0` |
| `qwen2.5-7b-instruct` | `Qwen/Qwen2.5-7B-Instruct` | **Dense (RQ1 reference, 2025)** | 7.6 B | 7.6 B | dense decoder, GQA 28Q/4KV, full attention all 28 layers | ~14.2 GB | `a09a35458c702b33eeacc393d103063234e8bc28` |

**Qwen3.6 status (RQ1):** dropped from the active model pool —
`enable_thinking=False` collapses to direct-answer mode on the multi-hop
QA tasks RQ1 evaluates (n_steps=[2,1,1,1,1] on smoke; tokens 10-32);
`enable_thinking=True` requires non-greedy sampling per card, breaking
length-residualization comparability with Phi/Gemma. Documented
limitation; see `_rq1-agent.md` Experiment P0 Qwen failure section.

**Qwen2.5-7B-Instruct (RQ1, added 2026-05-13):** dense decoder added as
the canonical 2025 RAG-paper backbone (used by SR-RAG Wu 2025; DyPRAG
Tan 2025 family). Replaces Qwen3.6 in the active pool. Plays cleanly
with Option-A greedy + no thinking-mode kwargs. Apache-2.0 license.
Fits comfortably on 24 GB GPUs (14.5 GB allocated under `eager`-attn
loader + KV cache).

(Optional 4th — `google/gemma-4-31b-it`, 30.7 B dense, parked under
`models/gemma-4-31b-it/` for the Phase 4.4 ablation (d) "MoE-vs-dense
within Gemma family." Not loaded for any current critical path.)

The three together bracket the 2026 attention-design spectrum at
**matched active-parameter scale** (~3-4 B active per token), so per-token
compute is comparable and tok/s differences in our runs primarily reflect
the attention mechanism rather than model size. Memory footprint differs
~10× across the three, by design.

## 2. Architecture details (verified config-vs-card)

### Phi-4-mini-instruct

| field | (card) | (config) | (obs) |
| --- | --- | --- | --- |
| total parameters | 3.8 B | implied | n/a |
| `num_hidden_layers` | — | **32** | mid_stack capture = layer 16 |
| `hidden_size` | — | **3072** | hidden-state tensor `[B, T, 3072]` confirmed in Phase 2 |
| `num_attention_heads` | — | **24** | — |
| `num_key_value_heads` | — | 8 (GQA 3:1) | — |
| `vocab_size` | 200 064 (incl. padding) | **200 064** | — |
| `max_position_embeddings` | 128 K | **131 072** | matches |
| `sliding_window` | not advertised | **262 144** (effectively no sliding — larger than max_position) | full attention everywhere |
| training stack | synthetic + filtered web; SFT + DPO | n/a | — |
| created | 2025-02 (HF) | n/a | — |
| `model_type` | — | `phi3` (re-uses Phi-3 architecture class in transformers) | — |

Notes:

- **Card's "5.1B / 8B / 30.7B" Gemma table includes embeddings**; Phi card
  doesn't break that out. The 3.8 B Phi figure is post-embedding "active"
  per the card text.
- Phi-4-mini is a Phi-3-class architecture (still loaded by the `Phi3`
  module under transformers 5.8). Don't confuse with Phi-4 14B (different
  arch class).
- `sliding_window=262144` is set above `max_position_embeddings=131072`,
  which means it never engages — every layer is full causal attention.

### Gemma-4-26B-A4B-it

| field | (card) | (config) | (obs) |
| --- | --- | --- | --- |
| total parameters | 25.2 B | implied | n/a |
| active parameters | **3.8 B** (per token, after MoE routing) | implied | matches Phi for compute-bracket fairness |
| `num_hidden_layers` | — | **30** | mid_stack capture = layer 15 |
| `hidden_size` | — | **2816** | confirmed |
| `num_attention_heads` | — | **16** | — |
| `num_key_value_heads` | — | 8 (GQA 2:1) | — |
| `vocab_size` | — | **262 144** | — |
| `max_position_embeddings` | 256 K | **262 144** | matches |
| `num_experts` | 128 total + 1 shared | **128** | shared expert handled inside MoE module, not in config field |
| `top_k_experts` (active) | **8 routed** | **8** | matches |
| `moe_intermediate_size` | — | **704** | per-expert FFN width |
| sliding window | 1024 tok | **1024** | matches; small global cache, large sliding cache |
| layer pattern | hybrid sliding+global, "final layer always global" | **5:1** = 25 sliding + 5 full (config `layer_types`); final layer is `full_attention` | matches card statement |
| modalities | text + image (vision encoder ~550 M) | text path = `gemma4_text` sub-config, vision path = `gemma4_vision` sub-config | we use text path only via `AutoModelForCausalLM` |
| RoPE pattern | full layers use Proportional RoPE (p-RoPE) | `rope_parameters.full_attention.rope_type = "proportional"`, sliding layers use default | matches |
| `model_type` | — | `gemma4` | requires transformers ≥ 5.8 |

Notes:

- The card says 25.2 B total, 3.8 B active, and lists 8 active / 128 total
  + 1 shared expert. Config doesn't have a separate field for the shared
  expert; it's wired in the MoE block directly.
- **Sliding window = 1024 tokens** is much smaller than Qwen's no-sliding
  setup. Phase 1.4 found that this *bounds compute per layer* but **not
  KV memory**, because HF's `DynamicCache` doesn't auto-evict entries
  outside the sliding window. See `_AGENT.md` 2026-05-09 #6 for the full
  finding.
- Vision encoder is loaded by default but unused by our text-only
  `AutoModelForCausalLM` path. VRAM cost includes it.

### Qwen3.6-35B-A3B (instruct — chat template bundled, no `-Instruct` repo)

| field | (card) | (config) | (obs) |
| --- | --- | --- | --- |
| total parameters | 35 B (from name; "A3B" = ~3 B active) | implied | matches |
| active parameters | ~3 B per token | implied | matches Phi/Gemma for compute-bracket fairness |
| `num_hidden_layers` | — | **40** | mid_stack capture = layer 20 |
| `hidden_size` | — | **2048** | smaller than Phi/Gemma despite more total params (MoE economy) |
| Gated DeltaNet heads | "32 for V, 16 for QK" | `linear_num_value_heads=32, linear_num_key_heads=16, linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4` | matches card text exactly |
| Gated Attention heads | "16 for Q and 2 for KV", `head_dim=256`, RoPE dim 64 | `num_attention_heads=16, num_key_value_heads=2`; `partial_rotary_factor=0.25` × hidden ⇒ rotary dim 64 | matches |
| `vocab_size` | — | **248 320** | — |
| `max_position_embeddings` | 256 K (262 144) | **262 144** | matches; card recommends keeping ≥128K context to "preserve thinking capabilities" |
| `num_experts` | 256 | **256** | — |
| `num_experts_per_tok` (active) | 8 routed + 1 shared | **8** | shared expert wired in MoE block |
| `moe_intermediate_size` | 512 | **512** | smaller per-expert FFN than Gemma |
| layer pattern | hybrid Gated DeltaNet + Gated Attention | `layer_types`: 30 `linear_attention` + 10 `full_attention` interleaved **3:1** (3 linear, 1 full, ×10) | matches card; canonical "3:1" framing in `_PLAN.md` |
| MTP head | yes (Multi-Token Prediction) | `mtp_num_hidden_layers=1`, `mtp_use_dedicated_embeddings=false` | not used by our greedy decode loop |
| reasoning parser | `qwen3` (per card vLLM/SGLang flag) | n/a | we don't run a separate reasoning parser; we tokenize the prefilled `<think>` directly |
| `model_type` | — | `qwen3_5_moe` (ARCH still under "3.5" name in transformers; Qwen3.6 is a checkpoint update on the 3.5 architecture) | requires `transformers ≥ 5.8` |

Notes:

- Architecturally Qwen3.6 = Qwen3.5 with re-trained weights and a tweaked
  chat template (notably: Qwen3.5 supported `/think` and `/nothink` soft
  switches, **Qwen3.6 dropped them** — see card §"Instruct (Non-Thinking)
  Mode", line 375 of fetched HTML). Use `chat_template_kwargs={"enable_thinking": False}`.
- "Number of Activated Experts: 8 Routed + 1 Shared" (card line 40 of HF
  fetch) means each token always sees the 1 shared expert plus 8
  routed-Top-K experts → 9 expert MLPs per token per MoE layer.
- The on-disk `mamba_ssm_dtype=float32` field implies the Gated DeltaNet
  recurrent state is kept in fp32 even when surrounding weights are bf16
  — relevant for Phase 2's `output_hidden_states` capture, since the
  hidden-state output passes through the recurrent block.
- Unlike Gemma 4, Qwen3.6 is **text-only**, no vision encoder.

### Qwen2.5-7B-Instruct (RQ1 reference, 2025; added 2026-05-13)

| spec | card claim | on-disk config / measured | match? |
|---|---|---|---|
| `architectures` | `Qwen2ForCausalLM` (card §"Architecture") | `Qwen2ForCausalLM` | matches |
| `model_type` | qwen2 | `qwen2` | matches; supported by transformers ≥4.37 |
| total params | 7.61 B (card "7.61B") | **7,615,616,512** | matches |
| non-embedding params | 6.53 B (card) | (not separately measured) | derivable |
| n layers | 28 (card) | `num_hidden_layers=28` | matches |
| hidden size | — | `hidden_size=3584` | — |
| GQA | "28 for Q and 4 for KV" (card) | `num_attention_heads=28`, `num_key_value_heads=4` | matches; **7×-compressed KV** vs Q |
| intermediate size | — | `intermediate_size=18944` | — |
| layer pattern | dense (card §"Architecture") | `is_hybrid=False`; all 28 layers `full_attention` | matches; **no hybrid stack** |
| `vocab_size` | — | `152064` (model) / `151643` (tokenizer specials) | — |
| `max_position_embeddings` | "Full 131,072 tokens" (card §"Introduction") | `32768` natively; YaRN config in card extends to 128K | **on-disk default = 32K** unless rope_scaling YaRN is enabled |
| `rope_scaling` | YaRN optional (card §"Processing Long Texts") | `{"rope_theta": 1e6, "rope_type": "default"}` | default mode |
| `tie_word_embeddings` | — | `False` | output head separate from embed |
| dtype | bf16 native | `torch.bfloat16` | matches |

- DRAGIN attention layer (last full-attn) = **layer 27** (tuple-index
  27, model-layer 27) — every layer is full attention so the index is
  just `n_layers - 1`.
- Mid-stack capture layer for v2 pipeline = **layer 14** (n_layers // 2).
- No vision encoder, no thinking-mode mechanism, no MoE routing.
- License: apache-2.0 (free).

## 3. Chat templates and the prompting protocol (Option A)

Each model's chat template lives at one of two paths:

- `models/<name>/tokenizer_config.json` field `chat_template` (Phi-4-mini),
  **or**
- `models/<name>/chat_template.jinja` (Gemma 4, Qwen 3.6).

Treat the on-disk file as ground truth — it's what `apply_chat_template`
actually evaluates. Cards quote excerpts but can lag.

### Phi-4-mini

```text
{% for message in messages %}
  {% if message['role'] == 'system' and 'tools' in message and message['tools'] is not none %}
    {{ '<|' + message['role'] + '|>' + message['content'] + '<|tool|>' + message['tools'] + '<|/tool|>' + '<|end|>' }}
  {% else %}
    {{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}
  {% endif %}
{% endfor %}
{% if add_generation_prompt %}{{ '<|assistant|>' }}{% else %}{{ eos_token }}{% endif %}
```

- Has a real `system` role (`<|system|>...<|end|>`). Card §"Chat Format"
  example confirms: `messages = [{"role":"system",...}, {"role":"user",...}]`.
- No thinking-mode mechanism in the template at all. Phi-4-mini does not
  support `<think>` or any equivalent.
- `eos_token = <|endoftext|>` (id varies by tokenizer). The model
  sometimes prefers to emit `<|end|>` to signal turn-end instead — our
  `04_run_qa.py` early-stop on `Answer:\s*[^\n]+\n` covers this case.

### Gemma 4 26B A4B

Long template; the Option A-relevant tail:

```jinja
{%- if add_generation_prompt -%}
    {%- if ns.prev_message_type != 'tool_response' and ns.prev_message_type != 'tool_call' -%}
        {{- '<|turn>model\n' -}}
        {%- if not enable_thinking | default(false) -%}
            {{- '<|channel>thought\n<channel|>' -}}
        {%- endif -%}
    {%- endif -%}
{%- endif -%}
```

- Default (no `enable_thinking=True`) prefills **`<|channel>thought\n<channel|>`** — an *empty* thought channel that signals "no thinking; go straight to the answer." Card §2 line 343: "if thinking is disabled, the model will still generate the tags but with an empty thought block."
- Thinking is enabled by **two** simultaneous knobs: `enable_thinking=True`
  to `apply_chat_template` AND a `<|think|>` token at the start of the
  system prompt (card §2 line 341). We do neither.
- Real `system` role supported (card §2 line 339: "Compared to Gemma 3, the
  models use standard `system`, `assistant`, and `user` roles").
- `<turn|>` and `<|turn>` are paired delimiter tokens (note the order;
  it's the close-then-open glyph layout, not a typo).

### Qwen2.5-7B-Instruct (RQ1, added 2026-05-13)

Standard ChatML, no thinking-mode mechanism. The chat template's
generation-prompt tail is just `<|im_start|>assistant\n` — no
`<think>` tag prefill. Empty `chat_template_kwargs={}` works.

Smoke (`runs/04c_new_model_smoke_qwen2.5-7b-instruct_20260513T142310Z.json`):
- Rendered prompt tail (300 chars): standard ChatML, ends at
  `<|im_start|>assistant\n` with no thinking-block prefill.
- `open_thinking_blocks = []` (none detected).
- 256-tok greedy probe: 16 tokens, **eos** stop, "Reasoning: ...
  Answer: Paris" — perfect Option-A format compliance.
- 3-question HotpotQA-pilot smoke: all 3 reached eos cleanly,
  emitted `Step 1: ... Answer: ...` chain-of-thought, throughput
  ~41 tok/s on real-question prompts.

`system` role supported; standard ChatML message format. No
`enable_thinking`/`preserve_thinking` kwargs apply.

### Qwen3.6-35B-A3B

Tail (chat_template.jinja lines 147-153):

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}
```

- **Default = thinking ON.** Template prefills `<think>\n` and the model
  is launched mid-thought. It will emit reasoning, eventually `</think>`,
  then the actual answer. Card §"Quickstart" line 257: "Qwen3.6 models
  operate in thinking mode by default."
- **`enable_thinking=False`** prefills `<think>\n\n</think>\n\n` — an
  empty closed think block, so the model goes straight to the answer.
  Card §"Instruct Mode" line 411: pass `chat_template_kwargs={"enable_thinking": False}`.
- The Qwen 3 soft switches `/think` and `/nothink` **do not work in 3.6**
  (card line 375). Don't try.
- `system` role supported (template lines 62-65). System message must be
  first in the messages list.
- `preserve_thinking=True` is a separate kwarg (card §"Preserve Thinking"
  line 422) that retains historical `<think>` blocks across multi-turn
  conversations. Not used by our single-turn QA runs.

### Option A protocol (Phase 4.1b, locked in 2026-05-10)

| model | `messages[0]` role | `chat_template_kwargs` | `repetition_penalty` | rationale |
| --- | --- | --- | ---: | --- |
| `phi-4-mini-instruct` | `system` | `{}` | **1.05** | Defensive degeneration guard; greedy alone produces sentence loops on 1500-tok contexts (`_AGENT.md` 2026-05-10 #11). |
| `gemma-4-26b-a4b-it` | `system` | `{}` | 1.0 | Default = no thinking. |
| `qwen3.6-35b-a3b-instruct` | `system` | `{"enable_thinking": False}` | 1.0 | (Excluded from RQ1; under-generation under this kwarg.) |
| `qwen2.5-7b-instruct` | `system` | `{}` | 1.0 | RQ1 reference backbone. No thinking-mode mechanism; smoke 04c confirmed Option-A behaviour. |

Common: `add_generation_prompt=True`; greedy decode (`argmax` of next
logits with optional rep penalty); early-stop on `\bAnswer\s*:\s*[^\n]+\n`
in addition to EOS; `max_new_tokens=1024` (Option A bound).

The same `MODEL_PROTOCOL` dict is duplicated at the top of
`scripts/04_run_qa.py` and `scripts/04b_protocol_smoke.py` for clarity;
**keep them in lockstep.** Smoke verifies on-disk weights match the
declared protocol; sweep wrapper aborts if smoke fails.

## 4. Sampling defaults — what the cards recommend vs what we use

The cards diverge on sampling. We override all three to **greedy** for
trace-comparability (sampling noise muddies hidden-state drift signals,
which is exactly what Phase 2/4 measure). Reviewers may push back; the
defensible answer is "we trade per-model peak quality for cross-arch
trace stability; gate metrics, not generation quality, are what we
report."

| model | (card) recommended | (obs) we run |
| --- | --- | --- |
| Phi-4-mini | `do_sample=False, temperature=0.0` (card §"Code section" example) | greedy + `repetition_penalty=1.05` |
| Gemma 4 | `temperature=1.0, top_p=0.95` (card §1 line 333) | greedy |
| Qwen 3.6 (thinking) | `t=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0` (card line 251) | greedy (mode forced off) |
| Qwen 3.6 (instruct/no-think) | `t=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0` (card line 253) | greedy |
| Qwen 2.5 7B Instruct | `t=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05` (card §"Quickstart" — Qwen2.5 generation_config.json on disk: `temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05`) | greedy + `repetition_penalty=1.0` (matches Phi/Gemma/Qwen3.6 contract; no observed degenerate loops on smoke) |

If a separate sampling-comparison run is ever requested, mirror the card
recommendations exactly and stamp the divergence in the per-question
record.

## 5. EOS / stop tokens

| model | `eos_token` | special turn-end | observed escape valves |
| --- | --- | --- | --- |
| Phi-4-mini | `<|endoftext|>` | `<|end|>` (per chat template) | sometimes emits `<|end|>` instead of EOS; smoke caught Phi running to `max_tokens` after a clean `Answer: Paris` (just kept restating itself); answer-line early-stop catches this. |
| Gemma 4 | `<eos>` (template-defined) | `<turn|>` | clean EOS in our smoke (40 tok, EOS @ answer line). |
| Qwen 3.6 | `<|im_end|>` (ChatML) | — | clean EOS in our smoke (16 tok, EOS @ answer line). |
| Qwen 2.5 7B | `<|im_end|>` id=151645 (ChatML) | pad=`<|endoftext|>` id=151643 | clean EOS in 04c smoke (16 tok, EOS @ answer line). 3-question HotpotQA-pilot smoke: all 3 reached EOS cleanly. |

The Phase 4 driver tracks both `eos_seen_at` (token id == eos_id) and
`answer_marker_seen_at` (regex-matched `Answer:` line). Both get logged.

## 6. Canonical numbers — what we have measured ourselves

These are the numbers to compare against when validating a new compute
box, a new env, or a re-run. All on **A800 PCIe 80 GB**, bf16 native,
fully on `cuda:0`, `transformers 5.8.0`, `torch 2.6.0+cu124`, `bnb 0.49.2`.

### Load + smoke (Phase 0 / `04b_protocol_smoke`)

| model | weights load | RSS post-load | VRAM post-load | first-token gen | tok/s steady (32-tok smoke) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phi-4-mini | ~4-9 s | ~1.0 GB | **~8.1 GB** | ~50 ms | **~47 tok/s** |
| Gemma 4 26B A4B | ~15-20 s | — | **~50 GB** | — | **~13 tok/s** |
| Qwen 3.6 35B A3B | ~35-40 s | — | **~67-68 GB** (bf16 native) | — | **~10-12 tok/s** |
| Qwen 2.5 7B Instruct | **~3.4 s** | ~1.0 GB | **~14.5 GB** (allocated) / ~16.6 GB (peak with cache) | ~50 ms | **~21 tok/s** (256-tok smoke; ~41 tok/s on real-question prompts per 04c) |

Qwen also has a fully-quantized NF4 hybrid path (~65 GB / ~8.9 tok/s) as a
fallback for smaller VRAM cards; bf16 native is the canonical A800 path
because bnb cannot quantize the grouped `Qwen3_5MoeExperts` tensor and
the dequant overhead on the small Linear fraction isn't worth it. See
`_AGENT.md` 2026-05-09 #2c for the upstream bug history.

Qwen 2.5 7B Instruct fits comfortably on a 24 GB GPU (RTX 3090) under
the eager-attn loader with K=5 sampler in scope.

### Phase 1.4 KV-pressure (5 000-token uninterrupted generation, mean across 3 prompt classes)

| model | mean tok/s | KV growth at step 5000 (MB) | KV growth slope (KB/tok) | peak VRAM (MB) | step1→stepN slowdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phi-4-mini | **47** | **643** | **128** | ~8 700 | ~1.0× |
| Gemma 4 26B A4B | **13** | **1 105** | **221** | ~50 800 | ~1.0× |
| Qwen 3.6 35B A3B | **13** | **135** | **27** | **66 258** | ~0.97× |

The two-axis bracket (KV growth vs total compute) is the Phase 1
headline. Note Gemma 4's KV growth is **steeper than Phi's** despite
sliding-window attention — the HF `DynamicCache` doesn't auto-evict
entries outside the sliding window, so sliding bounds compute but not
memory. This is a real and reportable finding (`_AGENT.md` 2026-05-09 #6).

### Phase 2 hidden-state hook overhead

| model | tok/s with `output_hidden_states=True` | overhead vs Phase 1.4 |
| --- | ---: | ---: |
| Phi-4-mini | 45.5 | ~3 % |
| Gemma 4 26B A4B | 13.3 | ~−2 % (within noise) |
| Qwen 3.6 35B A3B | 12.6 | ~2 % |

Well inside the ≤5 % budget set in `_PLAN.md` Phase 2.1. The 11–14 %
overhead on 200-tok smokes is warmup-dominated and goes away at 5k scale.

### Phase 4.1b protocol smoke (2026-05-10, single calibration question)

Question: id `5a8e3ea95542995a26add48d`, "The director of the romantic
comedy 'Big Stone Gap' is based in what New York city?", gold "Greenwich
Village, New York City".

| model | tok generated | early_stop | model output (extracted) |
| --- | ---: | --- | --- |
| Phi-4-mini | 56 | answer_marker | `Greenwich Village` (then runs into the next question line — answer-line regex captured both) |
| Gemma 4 26B A4B | 184 | answer_marker | `Greenwich Village, New York City//thought` (extra `thought` artifact from empty-channel close) |
| Qwen 3.6 35B A3B | **51** | **eos** | `New York City` |

Compare to **Phase 4.1 (rejected)** on the same question:

| model | 4.1 (mid_stack) | 4.1b (Option A smoke) | delta |
| --- | --- | --- | --- |
| Phi | EOS @ 113, "Greenwich Village, New York City", EM=1 | answer_marker @ 56, captured-too-much | model side issue (smoke regex over-captured), driver should be fine |
| Gemma | EOS @ 145, "Greenwich Village, New York City", EM=1 | answer_marker @ 184, "...//thought" suffix | grader-side cleanup needed |
| Qwen | hit_cap @ 512, "they want the neighborhood", EM=0 | **EOS @ 51, "New York City"** | **protocol fix recovers a real answer in 1/10 the tokens** |

## 7. Known model quirks (observed, not from the cards)

- **Phi runaway repetition without rep_penalty.** On 1500-token
  multi-distractor contexts under greedy decode, Phi-4-mini reliably
  enters sentence-level repetition loops ("Goodbye, and thank you!"
  ×4, "The Kingdom of the Isles ... covered a total land area of over
  8300 km2." ×5). 75% (150/200) hit-cap rate in the Phase 4.1 sweep.
  `repetition_penalty=1.05` makes this go away in our smoke; full sweep
  validation pending.
- **Phi answer-line newline elision.** Phi sometimes emits
  `Answer: Greenwich VillageWhat was the original location...` with no
  newline between answer and next sentence. The strict `\n`-terminated
  `Answer:` regex captures everything to the *next* newline, which
  may be 100+ chars later. Affects the smoke probe; the actual driver
  uses `qa_grading.extract_answer` which has additional cleanup.
- **Gemma "thought" suffix leak.** Decoded text occasionally ends with
  literal `thought\n` after the answer, because the empty-channel close
  glyph round-trips through the tokenizer as the word "thought". Visible
  in the Phase 4.1b smoke; cleaned up in `qa_grading.extract_answer`.
- **Qwen "thinking process" preamble.** Even with `enable_thinking=False`
  prefilled (empty closed `<think>`), Qwen sometimes opens with
  `"Here's a thinking process:\n\n1. **Analyze..."` as plain prose —
  i.e. it's *imitating* its training-data thinking-block style in the
  body of the answer. The smoke shows this is rare under Option A
  (1500-tok contexts; 51-tok response on the multi-hop question). Heavy
  in the Phase 4.1 (rejected) sweep where thinking was actually on.
- **Gemma EOS during long generations.** During the Phase 1.4 5k-token
  uninterrupted run, Gemma 4 emitted EOS at step 902 (multi-hop class)
  and step 3425 (ambiguous). The driver continues past EOS by design
  for KV-pressure measurement — KV cost continued growing identically
  past the EOS, validating the post-EOS measurement protocol.

## 8. Library / runtime constraints

- **`transformers ≥ 5.8.0`** required for both `gemma4` and `qwen3_5_moe`
  arch classes. Verified working: 5.8.0.
- **`bitsandbytes ≥ 0.49`** import-clean on the env, but **does not
  support grouped-MoE-experts quantization** (`Qwen3_5MoeExperts` is a
  3-D `nn.Parameter`, not a `Linear4bit`). Two failure modes documented
  in `_AGENT.md` 2026-05-09 #2c. Workarounds:
  1. Pull `unsloth/Qwen3.6-...-bnb-4bit` pre-baked snapshot (Futures F1).
  2. Switch to MXFP4 (first-class grouped-MoE support upstream).
  3. Manual grouped-expert pass via `tools/quantize_qwen.py` (parked).
- **`torch 2.6.0+cu124`** — pinned in `setup.sh` because pip's "latest"
  torch (2.11+cu130) wants driver ≥13.0 and the A800 box ships driver
  12.6. Setup script handles the auto-downgrade.
- **`flash-linear-attention` + `causal-conv1d`** — Qwen 3.6's Gated
  DeltaNet falls back to a torch implementation when these aren't
  installed. Our smoke saw the warning "[transformers] The fast path is
  not available because one of the required library is not installed."
  The torch fallback works correctly but is slower; check Phase 1.4
  numbers if installing them changes the bracket comparison.

## 9. Validation checklist for a new compute box

Run in this order:

1. `bash scripts/setup.sh` (env builds; torch downgrade to 2.6+cu124 if
   needed; lockfile written to `runs/env_lockfile.txt`).
2. `bash scripts/download_models.sh phase1.4` (~125 GB, all three pinned
   SHAs).
3. `python scripts/sysinfo.py` — confirms CUDA visible, A800 detected,
   matches the §6 hardware row.
4. `python scripts/04_dataset.py` — rebuilds `data/hotpotqa_distractor_200.jsonl`
   and verifies `slice_sha256` matches the manifest.
5. **`python scripts/04b_protocol_smoke.py`** — must report
   `[ok] all 3 models pass Option A protocol smoke`. Compare its
   `runs/04b_protocol_smoke_<ts>.json` against the §6 "Phase 4.1b
   protocol smoke" canonical numbers. **Sweep is blocked until this
   passes** (the wrapper script enforces it).
6. `bash scripts/run_phase41b_sweep.sh` — Phase 4.1b headline run
   (~1.5-2.5 h on A800).

## 10. Where to extend this file

- A new model joins the pool → add a row to §1, a section to §2-3-7, a
  protocol entry to §3 (Option A), a sampling row to §4, an EOS row to
  §5, and update the smoke validator's `PROTOCOL` dict + the driver's
  `MODEL_PROTOCOL`.
- A new architecture-relevant finding (e.g. a new attention quirk that
  affects Phase 2 hidden-state geometry) → add to §7, cross-reference
  the relevant `_AGENT.md` entry.
- A new sampling regime is adopted → update §4 and add a row to the
  per-question record schema.
- The HF model card changes materially (e.g. Qwen pushes a chat-template
  revision) → re-fetch, diff against the on-disk template, re-run the
  smoke, update §3, and stamp the date in `_AGENT.md`. Do **not** silently
  bump pinned SHAs; that breaks the reproducibility contract.
