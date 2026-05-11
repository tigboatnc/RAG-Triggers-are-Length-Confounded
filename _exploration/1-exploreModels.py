"""1-exploreModels.py — Model Exploration Primer (Learning Path Step 1)

Loads each of the three project models one at a time and runs a structured
battery of probes designed to teach you *what these models are* from the
inside out.  All results are written to a single JSON log file that the
companion notebook (1-exploreModels.ipynb) reads to build a visual primer.

Probes (in order, per model):
  P1  Identity        — "Who are you?" (see what the model says about itself)
  P2  Chat template   — render the same messages list, show the raw string
  P3  Architecture    — layer count, hidden dim, vocab, attention config,
                        MoE config if present, parameter census
  P4  Hidden-state    — forward a short prompt, capture all layer hidden
                        states, report shapes, norms, dtype
  P5  Uncertainty     — predictable vs ambiguous prefix (L2 norm + entropy)
  P6  Reasoning       — a short multi-hop question (no paragraphs) to see
                        how each model *sounds* when it reasons

Usage (on compute cluster):
    python _exploration/1-exploreModels.py                     # all 3
    python _exploration/1-exploreModels.py --model phi         # just phi
    python _exploration/1-exploreModels.py --model gemma       # just gemma
    python _exploration/1-exploreModels.py --model qwen        # just qwen

Output:
    _exploration/1-exploreModels_log.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Model registry — mirrors scripts/download_models.sh local names
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "phi": {
        "local_dir": "models/phi-4-mini-instruct",
        "hf_repo": "microsoft/Phi-4-mini-instruct",
        "family": "Phi-4 (Microsoft)",
        "arch_class": "Dense Decoder (GQA)",
        "lineage": (
            "Phi-1 (1.3B, 2023) → Phi-2 (2.7B) → Phi-3/3.5-mini (3.8B) → "
            "Phi-4-mini (3.8B, Feb 2025).  Same Phi-3 architecture class in "
            "transformers (model_type='phi3'), but with 200K vocabulary, GQA, "
            "and shared input/output embeddings added in Phi-4-mini."
        ),
        "comparable_models": [
            "Llama-3.2-3B-Instruct (Meta, dense 3B)",
            "Gemma-2-2B-it (Google, dense 2.6B)",
            "StableLM-2-1.6B (Stability AI)",
        ],
        "chat_template_kwargs": {},
        "repetition_penalty": 1.05,
        "what_makes_it_special": (
            "Dense — every parameter fires on every token.  Smallest model in "
            "the pool, trained heavily on synthetic 'textbook-like' data for "
            "reasoning.  No thinking mode, no MoE.  The simplest architecture "
            "in our comparison: a standard Transformer decoder with Grouped "
            "Query Attention (GQA).  Think of it as the 'control' model."
        ),
    },
    "gemma": {
        "local_dir": "models/gemma-4-26b-a4b-it",
        "hf_repo": "google/gemma-4-26B-A4B-it",
        "family": "Gemma 4 (Google DeepMind)",
        "arch_class": "Sparse MoE + Sliding-Window/Global Hybrid Attention",
        "lineage": (
            "Gemma-1 (2B/7B, Feb 2024) → Gemma-2 (2B/9B/27B, Jun 2024) → "
            "Gemma-3 (1B–27B, Mar 2025) → Gemma-4 (E2B/E4B/26B-A4B/31B, "
            "May 2025).  The 26B-A4B is the first MoE Gemma: 25.2B total, "
            "only 3.8B active per token via top-8-of-128 expert routing."
        ),
        "comparable_models": [
            "Mixtral-8x7B (Mistral, classic MoE, 46.7B total / ~13B active)",
            "DeepSeek-V2-Lite (16B-A2.4B, MoE + MLA attention)",
            "Qwen2-57B-A14B (Alibaba, MoE)",
        ],
        "chat_template_kwargs": {},
        "repetition_penalty": 1.0,
        "what_makes_it_special": (
            "MoE with a hybrid attention pattern: 25 sliding-window layers "
            "(window=1024 tokens) + 5 full-attention layers in a 5:1 ratio, "
            "final layer always global.  Also multimodal (vision encoder "
            "bundled but unused in our text-only path).  The sliding window "
            "bounds compute per layer but — surprise — does NOT bound KV "
            "memory in HuggingFace's DynamicCache implementation."
        ),
    },
    "qwen": {
        "local_dir": "models/qwen3.6-35b-a3b-instruct",
        "hf_repo": "Qwen/Qwen3.6-35B-A3B",
        "family": "Qwen 3.6 (Alibaba / Qwen Team)",
        "arch_class": "Hybrid Linear MoE (Gated DeltaNet + Full Attention)",
        "lineage": (
            "Qwen-1 (7B/14B/72B, Sep 2023) → Qwen-1.5 → Qwen-2 (0.5–72B, "
            "Jun 2024) → Qwen-2.5 → Qwen-3 (0.6–235B, Apr 2025) → "
            "Qwen-3.5 (Feb 2025, same arch) → Qwen-3.6 (May 2025, same "
            "arch, retrained weights + updated chat template).  Architecture "
            "tag in transformers is still 'qwen3_5_moe'."
        ),
        "comparable_models": [
            "Jamba-1.5 (AI21, Mamba-Attention hybrid, 52B-A12B)",
            "RWKV-6 (RWKV Foundation, pure linear-attention / RNN)",
            "Mamba-2 (Tri Dao, state-space model)",
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "repetition_penalty": 1.0,
        "what_makes_it_special": (
            "The most exotic architecture: 3:1 Gated DeltaNet (linear "
            "attention, O(1) per token) interleaved with full quadratic "
            "attention, plus 256-expert MoE (8 routed + 1 shared = 9 active "
            "per token).  35B total parameters but only ~3B active.  Thinking "
            "mode on by default — must pass enable_thinking=False to get "
            "direct answers.  Grouped expert tensors are 3-D nn.Parameters, "
            "NOT nn.Linear — this breaks standard quantization."
        ),
    },
}

# Probes
IDENTITY_PROMPT = "Who are you? Answer in exactly one sentence."
PREDICTABLE_PROMPT = "The capital of France is"
AMBIGUOUS_PROMPT = "The next word in this random sequence is"
REASONING_PROMPT = (
    "Albert Einstein was born in Ulm.  Ulm is a city in which country?  "
    "That country's current capital is which city?  Answer step by step, "
    "then give a final answer in the form 'Answer: <city>'."
)
TEMPLATE_TEST_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
]


def _safe_json(obj: Any) -> Any:
    """Make an object JSON-serializable (handles tensors, numpy, etc)."""
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(x) for x in obj]
    return str(obj)


def _entropy_nats(logits) -> float:
    import torch
    with torch.inference_mode():
        x = logits.detach().to(torch.float32).reshape(-1)
        log_p = torch.log_softmax(x, dim=-1)
        p = log_p.exp()
        h = -(p * log_p).sum().item()
    return float(h)


def _count_parameters(model) -> dict[str, int]:
    """Count total, trainable, and estimate bf16 size."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "approx_bf16_bytes": total * 2,
        "approx_bf16_gb": round(total * 2 / 1e9, 2),
    }


def _inspect_architecture(model, config) -> dict[str, Any]:
    """Pull architecture details from config + model structure."""
    info: dict[str, Any] = {}
    info["model_type"] = getattr(config, "model_type", "unknown")
    info["num_hidden_layers"] = getattr(config, "num_hidden_layers", None)
    info["hidden_size"] = getattr(config, "hidden_size", None)
    info["num_attention_heads"] = getattr(config, "num_attention_heads", None)
    info["num_key_value_heads"] = getattr(config, "num_key_value_heads", None)
    info["vocab_size"] = getattr(config, "vocab_size", None)
    info["max_position_embeddings"] = getattr(config, "max_position_embeddings", None)

    # MoE fields
    info["num_experts"] = getattr(config, "num_experts", None) or getattr(config, "num_local_experts", None)
    info["num_experts_per_tok"] = getattr(config, "num_experts_per_tok", None) or getattr(config, "top_k_experts", None)
    info["moe_intermediate_size"] = getattr(config, "moe_intermediate_size", None)

    # Sliding window
    info["sliding_window"] = getattr(config, "sliding_window", None)

    # Layer types (Gemma/Qwen hybrid patterns)
    layer_types = getattr(config, "layer_types", None)
    if layer_types:
        from collections import Counter
        counts = Counter(layer_types)
        info["layer_type_counts"] = dict(counts)
        info["layer_type_pattern"] = _describe_pattern(layer_types)

    # Qwen-specific linear attention fields
    for field in ["linear_num_key_heads", "linear_num_value_heads",
                  "linear_key_head_dim", "linear_value_head_dim"]:
        val = getattr(config, field, None)
        if val is not None:
            info[field] = val

    # Text sub-config (Gemma 4 wraps text config)
    if hasattr(config, "text_config"):
        tc = config.text_config
        info["_note"] = "text_config found (multimodal wrapper); values above from text_config"
        info["num_hidden_layers"] = getattr(tc, "num_hidden_layers", info["num_hidden_layers"])
        info["hidden_size"] = getattr(tc, "hidden_size", info["hidden_size"])
        info["num_attention_heads"] = getattr(tc, "num_attention_heads", info["num_attention_heads"])
        info["num_key_value_heads"] = getattr(tc, "num_key_value_heads", info["num_key_value_heads"])
        info["vocab_size"] = getattr(tc, "vocab_size", info["vocab_size"])
        info["max_position_embeddings"] = getattr(tc, "max_position_embeddings", info["max_position_embeddings"])
        info["num_experts"] = getattr(tc, "num_experts", info["num_experts"]) or getattr(tc, "num_local_experts", info["num_experts"])
        info["num_experts_per_tok"] = getattr(tc, "num_experts_per_tok", info["num_experts_per_tok"]) or getattr(tc, "top_k_experts", info["num_experts_per_tok"])
        info["moe_intermediate_size"] = getattr(tc, "moe_intermediate_size", info["moe_intermediate_size"])
        info["sliding_window"] = getattr(tc, "sliding_window", info["sliding_window"])
        lt = getattr(tc, "layer_types", None)
        if lt:
            from collections import Counter
            info["layer_type_counts"] = dict(Counter(lt))
            info["layer_type_pattern"] = _describe_pattern(lt)

    # Top-level module names (first 2 levels)
    top_modules = []
    for name, _ in model.named_children():
        top_modules.append(name)
    info["top_level_modules"] = top_modules

    return {k: v for k, v in info.items() if v is not None}


def _describe_pattern(layer_types: list[str]) -> str:
    """Describe the repeating pattern in layer_types."""
    n = len(layer_types)
    for period in range(1, n + 1):
        if n % period != 0:
            continue
        pattern = layer_types[:period]
        if all(layer_types[i] == pattern[i % period] for i in range(n)):
            reps = n // period
            short = " → ".join(pattern)
            return f"{reps} × ({short})"
    return " → ".join(layer_types)


def load_model(model_dir: str):
    """Load model + tokenizer in bf16 native on cuda:0."""
    import torch

    # bnb compat shim (from 00_smoke.py)
    try:
        import bitsandbytes.nn as _bnb_nn
        if not getattr(_bnb_nn.Params4bit, "_recollect_kwargs_patched", False):
            _orig_new = _bnb_nn.Params4bit.__new__
            _allowed = {
                "data", "requires_grad", "quant_state", "blocksize",
                "compress_statistics", "quant_type", "quant_storage", "module",
                "bnb_quantized",
            }
            def _patched_new(cls, data=None, requires_grad=False, **kwargs):
                kwargs = {k: v for k, v in kwargs.items() if k in _allowed}
                return _orig_new(cls, data=data, requires_grad=requires_grad, **kwargs)
            _bnb_nn.Params4bit.__new__ = _patched_new
            _bnb_nn.Params4bit._recollect_kwargs_patched = True
    except ImportError:
        pass

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  [..] loading tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    print(f"  [..] loading model (bf16, device_map={{'': 0}})")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model.eval()
    load_time = time.perf_counter() - t0

    vram_mb = int(torch.cuda.max_memory_allocated() / 1024**2)
    print(f"  [ok] loaded in {load_time:.1f}s, VRAM ~{vram_mb} MB")
    return tokenizer, model, load_time, vram_mb


def generate_text(model, tokenizer, prompt_or_messages, max_new_tokens=128,
                  chat_template_kwargs=None, repetition_penalty=1.0):
    """Generate text from either a raw string or a messages list."""
    import torch

    if isinstance(prompt_or_messages, list):
        try:
            text = tokenizer.apply_chat_template(
                prompt_or_messages, tokenize=False,
                add_generation_prompt=True,
                **(chat_template_kwargs or {}),
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                prompt_or_messages, tokenize=False,
                add_generation_prompt=True,
            )
    else:
        text = prompt_or_messages

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[-1]

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    if repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty

    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    gen_time = time.perf_counter() - t0
    new_tokens = out.shape[-1] - prompt_len
    tok_per_s = new_tokens / gen_time if gen_time > 0 else 0

    response = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
    return {
        "response": response.strip(),
        "prompt_tokens": prompt_len,
        "new_tokens": int(new_tokens),
        "gen_time_s": round(gen_time, 3),
        "tok_per_s": round(tok_per_s, 1),
    }


def probe_hidden_states(model, tokenizer, prompt: str) -> dict[str, Any]:
    """Single forward pass with output_hidden_states=True, report geometry."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model(**inputs, output_hidden_states=True, use_cache=False)

    hs = out.hidden_states  # tuple of (n_layers+1) tensors
    n_layers = len(hs)
    shapes = [tuple(h.shape) for h in hs]

    # Norms at last token position for each layer
    norms = []
    for h in hs:
        vec = h[:, -1, :].detach().to("cpu", dtype=torch.float32)
        norms.append(round(float(vec.norm().item()), 4))

    return {
        "prompt": prompt,
        "prompt_tokens": int(inputs.input_ids.shape[-1]),
        "num_hidden_layers_returned": n_layers,
        "first_layer_shape": shapes[0],
        "last_layer_shape": shapes[-1],
        "hidden_dim": shapes[-1][-1],
        "dtype": str(hs[-1].dtype),
        "l2_norms_per_layer": norms,
        "l2_norm_first": norms[0],
        "l2_norm_mid": norms[n_layers // 2],
        "l2_norm_last": norms[-1],
    }


def probe_uncertainty(model, tokenizer) -> dict[str, Any]:
    """Predictable vs ambiguous prefix: L2 norm + entropy separation."""
    import torch

    results = {}
    for label, prompt in [("predictable", PREDICTABLE_PROMPT),
                          ("ambiguous", AMBIGUOUS_PROMPT)]:
        ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            fwd = model(**ids, output_hidden_states=True, use_cache=False)

        last_hs = fwd.hidden_states[-1][:, -1, :].detach().to("cpu", dtype=torch.float32)
        logits = fwd.logits[:, -1, :].detach().to("cpu", dtype=torch.float32)
        probs = logits.softmax(dim=-1)
        entropy = _entropy_nats(logits)
        top_p, top_id = probs.max(dim=-1)
        top_token = tokenizer.decode([int(top_id.item())])
        l2 = float(last_hs.norm().item())

        results[label] = {
            "prompt": prompt,
            "hidden_l2": round(l2, 4),
            "entropy_nats": round(entropy, 4),
            "top_token": top_token.strip(),
            "top_prob": round(float(top_p.item()), 4),
        }

    l2_delta = abs(results["predictable"]["hidden_l2"] - results["ambiguous"]["hidden_l2"])
    h_delta = results["ambiguous"]["entropy_nats"] - results["predictable"]["entropy_nats"]
    results["separation"] = {
        "l2_abs_delta": round(l2_delta, 4),
        "l2_pct_delta": round(100 * l2_delta / max(results["predictable"]["hidden_l2"], 1e-9), 2),
        "entropy_delta_nats": round(h_delta, 4),
        "entropy_ratio": round(
            results["ambiguous"]["entropy_nats"] / max(results["predictable"]["entropy_nats"], 1e-9), 2
        ),
    }
    return results


def probe_chat_template(tokenizer, chat_template_kwargs: dict) -> dict[str, Any]:
    """Show the raw rendered chat template for the test messages."""
    try:
        rendered = tokenizer.apply_chat_template(
            TEMPLATE_TEST_MESSAGES, tokenize=False,
            add_generation_prompt=True,
            **(chat_template_kwargs or {}),
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            TEMPLATE_TEST_MESSAGES, tokenize=False,
            add_generation_prompt=True,
        )
        chat_template_kwargs = {"_note": "kwargs not supported by this tokenizer"}

    # Tokenize to see special tokens
    token_ids = tokenizer.encode(rendered)

    return {
        "messages_input": TEMPLATE_TEST_MESSAGES,
        "chat_template_kwargs": chat_template_kwargs,
        "rendered_text": rendered,
        "rendered_token_count": len(token_ids),
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "special_tokens_count": len(tokenizer.all_special_tokens),
        "vocab_size": tokenizer.vocab_size,
    }


def run_probes_for_model(key: str, spec: dict) -> dict[str, Any]:
    """Run all probes for one model.  Returns the full result dict."""
    import torch

    print(f"\n{'='*70}")
    print(f"  MODEL: {spec['family']}  ({key})")
    print(f"  Repo:  {spec['hf_repo']}")
    print(f"  Arch:  {spec['arch_class']}")
    print(f"{'='*70}")

    model_dir = spec["local_dir"]
    if not Path(model_dir).exists():
        print(f"  [!!] {model_dir} not found — skipping.")
        return {"error": f"model dir {model_dir} not found"}

    result: dict[str, Any] = {
        "key": key,
        "hf_repo": spec["hf_repo"],
        "family": spec["family"],
        "arch_class": spec["arch_class"],
        "lineage": spec["lineage"],
        "comparable_models": spec["comparable_models"],
        "what_makes_it_special": spec["what_makes_it_special"],
    }

    tokenizer, model, load_time, vram_mb = load_model(model_dir)
    result["load_time_s"] = round(load_time, 2)
    result["vram_after_load_mb"] = vram_mb

    # P1: Identity
    print("\n  [P1] Identity probe...")
    msgs = [{"role": "user", "content": IDENTITY_PROMPT}]
    result["p1_identity"] = generate_text(
        model, tokenizer, msgs, max_new_tokens=80,
        chat_template_kwargs=spec["chat_template_kwargs"],
        repetition_penalty=spec["repetition_penalty"],
    )
    print(f"       → {result['p1_identity']['response'][:120]}...")

    # P2: Chat template
    print("  [P2] Chat template inspection...")
    result["p2_chat_template"] = probe_chat_template(
        tokenizer, spec["chat_template_kwargs"]
    )

    # P3: Architecture
    print("  [P3] Architecture introspection...")
    result["p3_architecture"] = _inspect_architecture(model, model.config)
    result["p3_architecture"]["param_census"] = _count_parameters(model)

    # P4: Hidden-state geometry
    print("  [P4] Hidden-state geometry probe...")
    result["p4_hidden_states"] = probe_hidden_states(
        model, tokenizer, "The quick brown fox jumps over the lazy dog"
    )

    # P5: Uncertainty separation
    print("  [P5] Uncertainty separation probe...")
    result["p5_uncertainty"] = probe_uncertainty(model, tokenizer)
    sep = result["p5_uncertainty"]["separation"]
    print(f"       L2 Δ = {sep['l2_pct_delta']}%  |  H Δ = {sep['entropy_delta_nats']} nats")

    # P6: Reasoning
    print("  [P6] Reasoning probe...")
    msgs = [{"role": "user", "content": REASONING_PROMPT}]
    result["p6_reasoning"] = generate_text(
        model, tokenizer, msgs, max_new_tokens=256,
        chat_template_kwargs=spec["chat_template_kwargs"],
        repetition_penalty=spec["repetition_penalty"],
    )
    print(f"       → {result['p6_reasoning']['response'][:120]}...")

    # Cleanup
    del model
    del tokenizer
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"  [ok] {key} probes complete, GPU memory released.\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="Model Exploration Primer")
    parser.add_argument(
        "--model", default="all",
        choices=["all", "phi", "gemma", "qwen"],
        help="Which model(s) to probe (default: all three sequentially).",
    )
    parser.add_argument(
        "--out", default="_exploration/1-exploreModels_log.json",
        help="Output JSON log path.",
    )
    args = parser.parse_args()

    if args.model == "all":
        keys = ["phi", "gemma", "qwen"]
    else:
        keys = [args.model]

    print(f"Model Exploration Primer — probing {len(keys)} model(s)")
    print(f"Output → {args.out}\n")

    # Load existing results if any (allows incremental runs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except Exception:
            existing = {}

    results = existing.get("models", {})
    for key in keys:
        spec = MODEL_REGISTRY[key]
        result = run_probes_for_model(key, spec)
        results[key] = _safe_json(result)

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "script": "1-exploreModels.py",
        "models": results,
    }

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n{'='*70}")
    print(f"  All probes complete.  Results → {out_path}")
    print(f"  Next: open 1-exploreModels.ipynb to build the visual primer.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
