"""Phase 0.5 smoke test (v3).

Loads Qwen3.6-35B-A3B-Instruct in 4-bit NF4 via bitsandbytes, generates 32
tokens with output_hidden_states=True, and reports:

  - load wall-clock + RSS + VRAM after load
  - tokens/sec for the generation
  - last-layer hidden-state tensor shape (so we know what to hook in Phase 2)
  - peak VRAM during generation
  - L4 landmine check: L2 norm of the last-layer hidden state at the position
    immediately following a *predictable* prefix vs an *ambiguous* prefix.
    If MoE expert-switching noise dominates real "prediction certainty"
    variance, the two will be indistinguishable -> L1 in _PLAN.md is real
    and Phase 2.3 needs a redesign before any more code is written.

Usage:
    conda run -n recollection --no-capture-output python scripts/00_smoke.py
    # or with the smaller model for a faster smoke:
    python scripts/00_smoke.py --model phi-4-mini-instruct
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sysinfo import _process_rss_gb, report, vram_used_mb  # type: ignore[import-not-found]

DEFAULT_MODEL_DIR = "qwen3.6-35b-a3b-instruct"
MAX_TOKENS = 32

# L4 landmine probes: same length, same syntactic position, very different
# next-token predictability. Ambiguous prompt picked so no single token wins.
PREDICTABLE_PROMPT = "The capital of France is"
AMBIGUOUS_PROMPT = "The next word in this random sequence is"


def _format_mb(mb: int | None) -> str:
    return "n/a" if mb is None else f"{mb} MB"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_DIR,
        help="Subdir under models/ (default: qwen3.6-35b-a3b-instruct)",
    )
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument(
        "--prompt",
        default="In one sentence, what is retrieval-augmented generation?",
    )
    args = parser.parse_args()

    model_dir = Path("models") / args.model
    if not model_dir.exists():
        print(f"[!!] missing {model_dir}. Run: bash scripts/download_models.sh all")
        sys.exit(1)

    info = report()
    rss_before_import = _process_rss_gb() or 0.0
    vram_before = vram_used_mb()
    print(f"\n[..] importing torch + transformers + bitsandbytes ...")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rss_after_import = _process_rss_gb() or 0.0
    print(f"[ok] import: rss {rss_before_import:.3f} -> {rss_after_import:.3f} GB")

    if not torch.cuda.is_available():
        print("[!!] CUDA not available; v3 requires a CUDA GPU.")
        sys.exit(1)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"[..] loading {model_dir} (4-bit NF4, bf16 compute)")
    t0 = time.perf_counter()
    # trust_remote_code=False on purpose: bundled modeling_phi3.py in the HF
    # snapshot is pinned to an older transformers API (e.g. LossKwargs) and
    # ImportErrors against current transformers. The built-in Phi3 / Qwen
    # implementations in transformers 5.x handle these architectures natively.
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    load_secs = time.perf_counter() - t0
    rss_after_load = _process_rss_gb() or 0.0
    vram_after_load = vram_used_mb()
    print(
        f"[ok] loaded in {load_secs:.2f}s | "
        f"rss {rss_after_import:.3f} -> {rss_after_load:.3f} GB | "
        f"VRAM {_format_mb(vram_before)} -> {_format_mb(vram_after_load)}"
    )

    # --- Generation ---------------------------------------------------------
    print(f"\n[..] prompt: {args.prompt!r}")
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
    gen_secs = time.perf_counter() - t0
    n_new = out.sequences.shape[-1] - inputs.input_ids.shape[-1]
    text = tokenizer.decode(out.sequences[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    peak_vram_mb = int(torch.cuda.max_memory_allocated() / 1024**2)
    rss_after_gen = _process_rss_gb() or 0.0
    vram_after_gen = vram_used_mb()

    # out.hidden_states is a tuple of length n_new; each entry is a tuple of
    # n_layers tensors of shape (batch, seq_len_at_step, hidden_dim).
    hs_steps = len(out.hidden_states)
    hs_layers = len(out.hidden_states[0]) if hs_steps else 0
    last_layer_last_step = (
        out.hidden_states[-1][-1] if hs_steps and hs_layers else None
    )
    hs_shape = tuple(last_layer_last_step.shape) if last_layer_last_step is not None else None
    hs_dtype = (
        str(last_layer_last_step.dtype) if last_layer_last_step is not None else None
    )

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(text)
    print("=" * 60)
    print(
        f"[ok] {n_new} tokens in {gen_secs:.2f}s = {n_new / gen_secs:.1f} tok/s | "
        f"rss {rss_after_load:.3f} -> {rss_after_gen:.3f} GB | "
        f"VRAM peak={peak_vram_mb} MB (live={_format_mb(vram_after_gen)})"
    )
    print(
        f"[ok] hidden_states: {hs_steps} steps x {hs_layers} layers; "
        f"last-layer-last-step shape={hs_shape} dtype={hs_dtype}"
    )

    # --- L4 landmine probe --------------------------------------------------
    # On a single forward pass over each prompt, take the last-layer hidden
    # state at the FINAL token of the prefix. Compute its L2 norm and the
    # entropy of the next-token distribution. The predictable prompt should
    # show a sharper distribution; if the hidden-state norms are
    # indistinguishable AND the entropy gap is small relative to the norm
    # diff, MoE routing noise is the dominant signal and L1 is real.
    print("\n[..] L4 landmine probe (predictable vs ambiguous prefix)")
    probe = {}
    for label, prompt in (("predictable", PREDICTABLE_PROMPT), ("ambiguous", AMBIGUOUS_PROMPT)):
        ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            fwd = model(**ids, output_hidden_states=True, use_cache=False)
        # Detach + cpu + fp32 immediately (Landmine L2 fix in _PLAN.md).
        last_hs = fwd.hidden_states[-1][:, -1, :].detach().to("cpu", dtype=torch.float32)
        logits = fwd.logits[:, -1, :].detach().to("cpu", dtype=torch.float32)
        probs = logits.softmax(dim=-1)
        entropy = float(-(probs * (probs.clamp_min(1e-12)).log()).sum().item())
        top_p, top_id = probs.max(dim=-1)
        top_token = tokenizer.decode([int(top_id.item())])
        l2 = float(last_hs.norm().item())
        probe[label] = {
            "prompt": prompt,
            "hidden_l2": round(l2, 4),
            "next_entropy_nats": round(entropy, 4),
            "top_token": top_token,
            "top_prob": round(float(top_p.item()), 4),
        }
        print(
            f"  [{label:>11}] hidden_L2={l2:.3f}  H(next)={entropy:.3f} nats  "
            f"top='{top_token}' p={float(top_p.item()):.3f}"
        )
    l2_diff = abs(probe["predictable"]["hidden_l2"] - probe["ambiguous"]["hidden_l2"])
    h_diff = probe["ambiguous"]["next_entropy_nats"] - probe["predictable"]["next_entropy_nats"]
    print(
        f"  [diff] hidden_L2 |delta|={l2_diff:.3f}  "
        f"entropy delta (amb-pred)={h_diff:.3f} nats"
    )
    if h_diff <= 0:
        print(
            "  [!!] entropy ordering inverted (ambiguous <= predictable). Either the "
            "ambiguous prompt isn't actually ambiguous to the model, or something is "
            "very wrong upstream. Investigate before trusting the L2 numbers."
        )
    print(
        "  Read: hidden-state L2 alone tells us little; the *ratio* of "
        "(L2 variance across prompts) to (entropy variance across prompts) is the "
        "L4 signal. Track it across more prompt pairs in Phase 2.5 before "
        "committing to mean-pool CRV."
    )

    print(
        f"\n[summary] device={info['gpu']['details']} "
        f"final_rss={rss_after_gen:.3f} GB peak_vram={peak_vram_mb} MB "
        f"hs_shape={hs_shape}"
    )


if __name__ == "__main__":
    main()
