"""Phase 1: KV-cache pressure baseline (v3).

Measures how KV-cache memory, peak VRAM, and per-token latency grow over a
5,000-token uninterrupted greedy generation with the canonical Phase 1
generator (Qwen3.6-35B-A3B, bf16 native fully on cuda:0). Runs over three
prompt classes and emits:

  runs/phase1_kv_pressure.jsonl   one row per checkpoint (default every 500 tok)
  figs/kv_pressure_baseline.png   two panels: VRAM-vs-tokens, tok/s-vs-tokens

This is Phase 1.2 + 1.3 from _PLAN.md. Generator path matches the canonical
A800 80 GB regime documented in _AGENT.md 2026-05-09 #3b: bf16 native, fully on
cuda:0, no offload, no NF4. The reflector (Phi-4-mini) is NOT loaded here —
Phase 1 is generator-only KV-pressure measurement; the reflector lands in
Phase 2.5 / 3.x trigger logic.

Why a manual generation loop instead of `model.generate(...)`:
  - We need per-step latency at every token, plus VRAM at every checkpoint.
  - `generate` does this poorly: even with `output_scores=True` the timing is
    aggregated, and getting at `past_key_values` mid-stream is awkward.
  - The manual loop is ~30 lines and gives us deterministic checkpoint
    semantics. Matches the planned Phase 2.1 hook-shape so we won't have to
    re-architect for hidden-state capture.

Why we force generation past EOS:
  - Phase 1 measures the *cost* curve, not answer quality. KV-cache grows the
    same whether the token is semantic or post-EOS noise. We log when EOS
    would have fired so the cost curve can be split into "real generation"
    vs "synthetic stress" if needed downstream.

Usage:
  python scripts/01_kv_pressure.py
  python scripts/01_kv_pressure.py --prompt-class factual --max-new-tokens 1000
  python scripts/01_kv_pressure.py --no-plot   # data only, plot later
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sysinfo import _process_rss_gb, vram_used_mb  # type: ignore[import-not-found]

DEFAULT_MODEL_DIR = "qwen3.6-35b-a3b-instruct"

# Three prompt classes from _PLAN.md Phase 1.3. Each is engineered to keep
# Qwen3.6 cooperating for thousands of tokens (thinking model, so simple
# questions terminate fast — these probe the 5k regime intentionally).
PROMPTS: dict[str, str] = {
    "factual": (
        "Explain in detail the major events of the French Revolution from "
        "1789 to 1799. Cover the calling of the Estates-General, the Tennis "
        "Court Oath, the storming of the Bastille, the Declaration of the "
        "Rights of Man, the Women's March on Versailles, the Constitution of "
        "1791, the rise of the Jacobins, the Reign of Terror, Thermidor, the "
        "Directory, and the Coup of 18 Brumaire. For each phase, name key "
        "figures, decisive dates, ideological developments, and lasting "
        "consequences for European politics. Be exhaustive."
    ),
    "multi-hop": (
        "Reason through the following multi-hop chain step by step, showing "
        "your work at each hop and stating your confidence at each step:\n"
        "(a) Identify a scientist who won the Nobel Prize in Physics twice.\n"
        "(b) Name the years of both prizes and the specific contributions "
        "cited each time.\n"
        "(c) Identify their primary research affiliation at the time of the "
        "second prize, and the country.\n"
        "(d) Identify one student or close collaborator of theirs who later "
        "won a Nobel Prize.\n"
        "(e) Identify the institution that student was at when they won, and "
        "the year of their prize.\n"
        "(f) Reflect on which step you were least confident about and why."
    ),
    "ambiguous": (
        "Continue the following open-ended story for as long as you can, "
        "weaving in unexpected plot twists, at least three unrelated "
        "metaphors per paragraph, shifts in narrative voice, and at least "
        "one digression into invented historical detail per page. Do not "
        "rush to a conclusion.\n\n"
        "'The forest was silent that evening, until a figure stepped through "
        "the line of cedars and—'"
    ),
}


def _format_mb(mb: int | None) -> str:
    return "n/a" if mb is None else f"{mb} MB"


def load_generator(model_dir: Path):
    """Load Qwen3.6 in bf16 native, fully on cuda:0. Canonical Phase 1 path."""
    import torch

    # Compat shim: see scripts/00_smoke.py for context. Harmless on the bf16
    # path (we don't hit Params4bit here) but keeps the env consistent.
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

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[..] loading {model_dir} (bf16 native, device_map={{'': 0}})")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model.eval()
    load_secs = time.perf_counter() - t0
    vram_after = vram_used_mb()
    rss_after = _process_rss_gb() or 0.0
    print(
        f"[ok] loaded in {load_secs:.2f}s | "
        f"rss {rss_after:.3f} GB | "
        f"VRAM {_format_mb(vram_after)}"
    )
    return tokenizer, model


def run_one_prompt(
    label: str,
    prompt: str,
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
    checkpoint_every: int,
) -> tuple[list[dict], dict]:
    """Generate `max_new_tokens` greedy tokens with KV cache; checkpoint every N.

    Returns (checkpoint_rows, summary_dict). The summary captures the prompt
    metadata + the EOS-would-have-fired-at marker for downstream filtering.
    """
    import torch

    messages = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(chat_text, return_tensors="pt").to(model.device)
    prompt_tokens = int(encoded.input_ids.shape[-1])

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    rows: list[dict] = []
    cache = None
    input_ids = encoded.input_ids
    attention_mask = encoded.attention_mask
    eos_id = tokenizer.eos_token_id
    eos_seen_at: int | None = None
    step_latencies: list[float] = []

    print(
        f"\n[{label}] prompt_tokens={prompt_tokens} max_new={max_new_tokens} "
        f"chk_every={checkpoint_every}"
    )

    # Prefill (step 0): full prompt in. Subsequent steps: only the new token.
    t_start = time.perf_counter()
    vram_before = torch.cuda.memory_allocated() / (1024**2)

    for step in range(max_new_tokens):
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask if step == 0 else None,
                past_key_values=cache,
                use_cache=True,
            )
        next_logits = out.logits[:, -1, :]
        next_token = next_logits.argmax(dim=-1, keepdim=True)
        cache = out.past_key_values
        input_ids = next_token
        attention_mask = None
        torch.cuda.synchronize()
        step_latency = time.perf_counter() - t0
        step_latencies.append(step_latency)

        tok_id = int(next_token.item())
        if eos_id is not None and tok_id == eos_id and eos_seen_at is None:
            eos_seen_at = step + 1

        if (step + 1) % checkpoint_every == 0 or (step + 1) == max_new_tokens:
            wall_clock = time.perf_counter() - t_start
            tokens_so_far = step + 1
            window = step_latencies[-checkpoint_every:]
            rolling_tok_s = len(window) / sum(window) if window else 0.0
            instant_tok_s = 1.0 / step_latency if step_latency > 0 else 0.0
            vram_alloc_mb = torch.cuda.memory_allocated() / (1024**2)
            vram_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
            kv_growth_mb = vram_alloc_mb - vram_before
            row = {
                "prompt_class": label,
                "prompt_tokens": prompt_tokens,
                "step_idx": tokens_so_far,
                "tokens_generated": tokens_so_far,
                "wall_clock_s": round(wall_clock, 3),
                "rolling_tok_per_s": round(rolling_tok_s, 3),
                "instantaneous_tok_per_s": round(instant_tok_s, 3),
                "vram_allocated_mb": round(vram_alloc_mb, 1),
                "vram_peak_mb": round(vram_peak_mb, 1),
                "kv_growth_mb_since_step0": round(kv_growth_mb, 1),
                "eos_seen_at": eos_seen_at,
            }
            rows.append(row)
            print(
                f"  [{label:>9}] step={tokens_so_far:>4}  "
                f"wall={wall_clock:6.1f}s  "
                f"rolling={rolling_tok_s:5.2f} tok/s  "
                f"instant={instant_tok_s:5.2f} tok/s  "
                f"vram={vram_alloc_mb:7.0f}MB  "
                f"peak={vram_peak_mb:7.0f}MB  "
                f"kv_growth={kv_growth_mb:5.0f}MB  "
                f"eos@{eos_seen_at}"
            )

    # step 0 is the prefill (full prompt forward + first decoded token); it is
    # an outlier in latency because of prompt length + model warmup. The KV-
    # pressure curve we actually care about is "did single-token decode get
    # slower as KV grew?" → compare step 1 vs step N, both single-token decodes.
    prefill_latency = step_latencies[0]
    decode_latencies = step_latencies[1:] if len(step_latencies) > 1 else [prefill_latency]
    decode_first = decode_latencies[0]
    decode_last = decode_latencies[-1]
    decode_slowdown = decode_last / decode_first if decode_first > 0 else 1.0
    summary = {
        "prompt_class": label,
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "total_wall_clock_s": round(time.perf_counter() - t_start, 3),
        "mean_tok_per_s": round(len(step_latencies) / sum(step_latencies), 3),
        "prefill_latency_s": round(prefill_latency, 4),
        "decode_step1_latency_s": round(decode_first, 4),
        "decode_stepN_latency_s": round(decode_last, 4),
        "decode_slowdown_step1_to_stepN": round(decode_slowdown, 3),
        "eos_seen_at": eos_seen_at,
        "vram_at_step0_mb": round(vram_before, 1),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
    }
    print(
        f"[{label}] done. mean={summary['mean_tok_per_s']} tok/s | "
        f"prefill={summary['prefill_latency_s']}s | "
        f"decode step1->stepN: {summary['decode_step1_latency_s']}s -> "
        f"{summary['decode_stepN_latency_s']}s ({summary['decode_slowdown_step1_to_stepN']:.2f}x) | "
        f"peak_vram={summary['peak_vram_mb']} MB | eos@{eos_seen_at}"
    )
    return rows, summary


def plot_kv_pressure(rows: list[dict], fig_path: Path) -> None:
    """Two-panel figure: VRAM-vs-tokens (top) and tok/s-vs-tokens (bottom)."""
    import matplotlib.pyplot as plt

    by_class: dict[str, list[dict]] = {}
    for r in rows:
        by_class.setdefault(r["prompt_class"], []).append(r)

    fig, (ax_v, ax_t) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    colors = {"factual": "#1f77b4", "multi-hop": "#d62728", "ambiguous": "#2ca02c"}

    for label, points in by_class.items():
        xs = [p["tokens_generated"] for p in points]
        vrams = [p["vram_allocated_mb"] / 1024 for p in points]
        tps = [p["rolling_tok_per_s"] for p in points]
        c = colors.get(label, "#888888")
        ax_v.plot(xs, vrams, marker="o", label=label, color=c)
        ax_t.plot(xs, tps, marker="o", label=label, color=c)

    ax_v.set_ylabel("VRAM allocated (GiB)")
    ax_v.set_title("KV-cache pressure baseline (Qwen3.6-35B-A3B, bf16 native, A800 80 GB)")
    ax_v.grid(alpha=0.3)
    ax_v.legend(loc="lower right")

    ax_t.set_ylabel("Rolling tok/s (over checkpoint window)")
    ax_t.set_xlabel("Tokens generated")
    ax_t.grid(alpha=0.3)
    ax_t.legend(loc="upper right")

    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=140)
    print(f"[ok] plot written to {fig_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_DIR,
        help="Subdir under models/ (default: qwen3.6-35b-a3b-instruct)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=5000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument(
        "--prompt-class",
        default="all",
        choices=["all", *PROMPTS.keys()],
    )
    parser.add_argument(
        "--output",
        default="runs/phase1_kv_pressure.jsonl",
        help="JSONL output (one row per checkpoint).",
    )
    parser.add_argument(
        "--summary-output",
        default="runs/phase1_kv_pressure_summary.jsonl",
        help="JSONL output (one row per prompt class).",
    )
    parser.add_argument("--fig-path", default="figs/kv_pressure_baseline.png")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    model_dir = Path("models") / args.model
    if not model_dir.exists():
        print(f"[!!] missing {model_dir}. Run: bash scripts/download_models.sh all")
        sys.exit(1)

    tokenizer, model = load_generator(model_dir)

    selected = list(PROMPTS.keys()) if args.prompt_class == "all" else [args.prompt_class]
    all_rows: list[dict] = []
    summaries: list[dict] = []
    for label in selected:
        rows, summary = run_one_prompt(
            label,
            PROMPTS[label],
            tokenizer,
            model,
            args.max_new_tokens,
            args.checkpoint_every,
        )
        all_rows.extend(rows)
        summaries.append(summary)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")
    print(f"[ok] {len(all_rows)} checkpoint rows -> {output_path}")

    summary_path = Path(args.summary_output)
    with summary_path.open("w") as f:
        for s in summaries:
            f.write(json.dumps(s) + "\n")
    print(f"[ok] {len(summaries)} summary rows -> {summary_path}")

    print("\n=== headline numbers ===")
    for s in summaries:
        print(
            f"  {s['prompt_class']:>9}: mean={s['mean_tok_per_s']:5.2f} tok/s  "
            f"prefill={s['prefill_latency_s']:.4f}s  "
            f"decode_step1={s['decode_step1_latency_s']:.4f}s  "
            f"decode_stepN={s['decode_stepN_latency_s']:.4f}s  "
            f"slowdown={s['decode_slowdown_step1_to_stepN']:.2f}x  "
            f"peak_vram={s['peak_vram_mb']:.0f} MB  "
            f"eos@{s['eos_seen_at']}"
        )

    if not args.no_plot:
        plot_kv_pressure(all_rows, Path(args.fig_path))


if __name__ == "__main__":
    main()
