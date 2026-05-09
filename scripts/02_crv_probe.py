"""Phase 2: latent-state hooks + step boundaries + CRV (the core contribution).

For each (model, prompt class) pair, generate up to `max_new_tokens` greedy
tokens with the manual decode loop from Phase 1, but with two additions:

  1. Pass `output_hidden_states=True` to every forward — `HiddenStateCapture`
     reads `out.hidden_states` and records the last-token hidden state per
     layer to CPU fp32.
  2. After each generated token, ask `StepBoundaryDetector` whether this
     token completes a logical step. If yes, `CRVTracker.commit_step(...)`
     mean-pools the per-token states accumulated since the previous boundary,
     emits a CRV scalar-feature row (norm + drift + variance), and stores
     the full CRV vector to a side .npz file.

Outputs per (model, prompt_class, boundary_strategy):

  runs/phase2_crv_<model>_<prompt>_<strategy>.jsonl   one row per logical step
  runs/phase2_crv_<model>_<prompt>_<strategy>_crv.npz CRV vectors keyed by step_idx
  runs/phase2_crv_<model>_<prompt>_<strategy>_summary.json  per-trace summary

The .npz files are heavy (a few MB each); they're listed in .gitignore but
remain on the box. The .jsonl + summary.json files travel in phase snapshots.

Usage:
  python scripts/02_crv_probe.py                                   # all 3 prompts on Qwen3.6, sentence boundary
  python scripts/02_crv_probe.py --model phi-4-mini-instruct
  python scripts/02_crv_probe.py --boundary fixed --fixed-stride 32
  python scripts/02_crv_probe.py --max-new-tokens 200 --prompt-class factual  # smoke test
  python scripts/02_crv_probe.py --layer-mode mid_stack            # L1 mitigation (pre-routing)
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
from crv import CRVTracker, CRVTrigger, HiddenStateCapture, StepBoundaryDetector  # type: ignore[import-not-found]

DEFAULT_MODEL_DIR = "qwen3.6-35b-a3b-instruct"

# Same prompts as Phase 1.3/1.4 so the CRV traces are comparable to the KV-pressure data.
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
    """Load a generator in bf16 native, fully on cuda:0. Same path as Phase 1.4."""
    import torch

    # Compat shim — see scripts/00_smoke.py and scripts/01_kv_pressure.py for context.
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
    print(
        f"[ok] loaded in {load_secs:.2f}s | "
        f"rss {_process_rss_gb() or 0:.3f} GB | "
        f"VRAM {_format_mb(vram_used_mb())}"
    )
    return tokenizer, model


def run_one_prompt(
    label: str,
    prompt: str,
    tokenizer: Any,
    model: Any,
    args: argparse.Namespace,
) -> tuple[list[dict], dict, dict]:
    """Drive the manual decode loop with hooks + CRV tracker.

    Returns (per_step_features, summary, crv_vectors_payload).
    """
    import torch

    messages = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(chat_text, return_tensors="pt").to(model.device)
    prompt_tokens = int(encoded.input_ids.shape[-1])

    # Resolve layer subset.
    n_layers = getattr(model.config, "num_hidden_layers", None)
    if n_layers is None and hasattr(model.config, "text_config"):
        n_layers = model.config.text_config.num_hidden_layers
    if n_layers is None:
        raise RuntimeError("cannot read num_hidden_layers from model.config")
    # `out.hidden_states` is a tuple of length n_layers + 1 (embedding + per layer).
    last_layer_idx = n_layers  # index in the hidden_states tuple
    mid_layer_idx = n_layers // 2

    if args.layer_subset is not None:
        capture_subset = list(args.layer_subset)
    elif args.layer_mode == "last_layer":
        capture_subset = [last_layer_idx]
    elif args.layer_mode == "mid_stack":
        capture_subset = [mid_layer_idx]
    elif args.layer_mode == "last_and_mid":
        capture_subset = [mid_layer_idx, last_layer_idx]
    else:
        capture_subset = None  # capture all; CRVTracker will resolve

    capture = HiddenStateCapture(
        layer_subset=capture_subset,
        every_n_steps=args.every_n_steps,
    )
    boundaries = StepBoundaryDetector(
        strategy=args.boundary,
        n_tokens=args.fixed_stride,
        tokenizer=tokenizer,
    )
    tracker = CRVTracker(
        layer_subset=capture_subset,
        layer_mode=args.layer_mode if args.layer_mode in {"last_layer", "mid_stack"} else "last_layer",
        window_size=args.window_size,
        drift_metric="cosine",
    )
    trigger = CRVTrigger(
        threshold_drift=args.threshold_drift,
        threshold_variance=args.threshold_variance,
        mode="or",
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    features: list[dict] = []
    cache = None
    input_ids = encoded.input_ids
    attention_mask = encoded.attention_mask
    eos_id = tokenizer.eos_token_id
    eos_seen_at: int | None = None
    step_latencies: list[float] = []

    print(
        f"\n[{label}] prompt_tokens={prompt_tokens} "
        f"max_new={args.max_new_tokens} "
        f"boundary={args.boundary} (stride={args.fixed_stride}) "
        f"layer_mode={args.layer_mode}  capture_subset={capture_subset}"
    )

    t_start = time.perf_counter()
    vram_before = torch.cuda.memory_allocated() / (1024**2)

    for step in range(args.max_new_tokens):
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask if step == 0 else None,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,  # ← Phase 2.1
            )
        next_logits = out.logits[:, -1, :]
        next_token = next_logits.argmax(dim=-1, keepdim=True)
        cache = out.past_key_values
        token_id = int(next_token.item())

        captured = capture.record(step, out)
        tracker.add_token(captured)

        # Boundary-driven: commit_step when the just-generated token closes a logical step.
        if boundaries.is_boundary(token_id):
            feature = tracker.commit_step(
                step_idx=boundaries._step_count,
                token_idx=step + 1,
                boundary_token_id=token_id,
            )
            if feature is not None:
                wall_clock = time.perf_counter() - t_start
                feature["wall_clock_s"] = round(wall_clock, 3)
                feature["vram_allocated_mb"] = round(
                    torch.cuda.memory_allocated() / (1024**2), 1
                )
                feature["fired_default"] = trigger.fired(feature)
                features.append(feature)

        if eos_id is not None and token_id == eos_id and eos_seen_at is None:
            eos_seen_at = step + 1

        input_ids = next_token
        attention_mask = None
        torch.cuda.synchronize()
        step_latencies.append(time.perf_counter() - t0)

        if args.progress_every and (step + 1) % args.progress_every == 0:
            wall = time.perf_counter() - t_start
            steps_committed = boundaries._step_count
            print(
                f"  [{label:>9}] tok={step+1:>5} steps={steps_committed:>4} "
                f"wall={wall:6.1f}s  "
                f"vram={torch.cuda.memory_allocated()/(1024**2):7.0f}MB  "
                f"eos@{eos_seen_at}"
            )

    # Flush a final partial step (any tokens since last boundary).
    final_feature = tracker.commit_step(
        step_idx=boundaries._step_count + 1,
        token_idx=args.max_new_tokens,
        boundary_token_id=None,
    )
    if final_feature is not None:
        final_feature["wall_clock_s"] = round(time.perf_counter() - t_start, 3)
        final_feature["vram_allocated_mb"] = round(
            torch.cuda.memory_allocated() / (1024**2), 1
        )
        final_feature["fired_default"] = trigger.fired(final_feature)
        features.append(final_feature)

    summary = {
        "model_name": args.model,
        "prompt_class": label,
        "boundary_strategy": args.boundary,
        "fixed_stride": args.fixed_stride if args.boundary == "fixed" else None,
        "layer_mode": args.layer_mode,
        "capture_subset": capture_subset,
        "n_layers_total": n_layers,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "tokens_generated": len(step_latencies),
        "logical_steps_committed": len(features),
        "tokens_per_step_mean": (
            round(len(step_latencies) / max(1, len(features)), 2)
        ),
        "total_wall_clock_s": round(time.perf_counter() - t_start, 3),
        "mean_tok_per_s": round(len(step_latencies) / sum(step_latencies), 3),
        "vram_at_step0_mb": round(vram_before, 1),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
        "eos_seen_at": eos_seen_at,
        "n_fired_default": sum(1 for f in features if f.get("fired_default")),
        "capture_stats": capture.stats,
        "boundary_stats": boundaries.stats,
        "trigger_thresholds": {
            "drift": trigger.threshold_drift,
            "variance": trigger.threshold_variance,
            "mode": trigger.mode,
        },
    }
    print(
        f"[{label}] done. {summary['logical_steps_committed']} steps from "
        f"{summary['tokens_generated']} tokens "
        f"(mean {summary['tokens_per_step_mean']} tok/step) | "
        f"mean={summary['mean_tok_per_s']} tok/s | "
        f"fired_default={summary['n_fired_default']}/{summary['logical_steps_committed']} | "
        f"peak_vram={summary['peak_vram_mb']} MB"
    )

    crvs, step_indices = tracker.stacked_crvs()
    crv_payload = {
        "step_idx": step_indices,
        "crv": crvs.cpu().numpy() if crvs.numel() > 0 else None,
        "capture_subset": capture_subset,
        "n_layers_total": n_layers,
    }
    return features, summary, crv_payload


def write_outputs(
    args: argparse.Namespace,
    label: str,
    features: list[dict],
    summary: dict,
    crv_payload: dict,
) -> None:
    import numpy as np

    runs_dir = Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    base = f"phase2_crv_{args.model}_{label}_{args.boundary}_{args.layer_mode}"
    jsonl_path = runs_dir / f"{base}.jsonl"
    summary_path = runs_dir / f"{base}_summary.json"
    npz_path = runs_dir / f"{base}_crv.npz"

    with jsonl_path.open("w") as f:
        for row in features:
            f.write(json.dumps(row) + "\n")

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    if crv_payload["crv"] is not None:
        np.savez(
            npz_path,
            step_idx=np.array(crv_payload["step_idx"], dtype=np.int32),
            crv=crv_payload["crv"].astype(np.float32),
            capture_subset=np.array(crv_payload["capture_subset"] or [], dtype=np.int32),
        )
        crv_size_mb = npz_path.stat().st_size / (1024**2)
    else:
        crv_size_mb = 0.0

    print(
        f"  wrote: {jsonl_path}  ({len(features)} rows)\n"
        f"         {summary_path}\n"
        f"         {npz_path}  ({crv_size_mb:.2f} MB)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_DIR,
        help="Subdir under models/ (default: qwen3.6-35b-a3b-instruct)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=5000)
    parser.add_argument(
        "--prompt-class",
        default="all",
        choices=["all", *PROMPTS.keys()],
    )
    parser.add_argument(
        "--boundary",
        default="sentence",
        choices=["sentence", "think_block", "fixed"],
    )
    parser.add_argument(
        "--fixed-stride",
        type=int,
        default=32,
        help="Token stride for boundary='fixed'.",
    )
    parser.add_argument(
        "--layer-mode",
        default="last_layer",
        choices=["last_layer", "mid_stack", "last_and_mid"],
        help=(
            "Which layer(s) to use for the CRV. 'mid_stack' is the L1 "
            "mitigation candidate (pre-routing-noise reference)."
        ),
    )
    parser.add_argument(
        "--layer-subset",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Explicit layer indices into out.hidden_states (length n_layers+1). "
            "Overrides --layer-mode if given."
        ),
    )
    parser.add_argument(
        "--every-n-steps",
        type=int,
        default=1,
        help="Sub-sample capture stride. 1 = capture every decode step.",
    )
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--threshold-drift", type=float, default=0.1)
    parser.add_argument("--threshold-variance", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    model_dir = Path("models") / args.model
    if not model_dir.exists():
        print(f"[!!] missing {model_dir}. Run: bash scripts/download_models.sh phase1.4")
        sys.exit(1)

    print(f"[..] importing torch + transformers + bitsandbytes ...")
    import torch  # noqa: F401  (just primes the env)

    tokenizer, model = load_generator(model_dir)

    selected = list(PROMPTS.keys()) if args.prompt_class == "all" else [args.prompt_class]

    headline = []
    for label in selected:
        features, summary, crv_payload = run_one_prompt(
            label, PROMPTS[label], tokenizer, model, args
        )
        write_outputs(args, label, features, summary, crv_payload)
        headline.append(summary)

    print("\n=== headline (Phase 2 per-trace summary) ===")
    for s in headline:
        print(
            f"  {s['prompt_class']:>9}: "
            f"steps={s['logical_steps_committed']:>4} "
            f"(~{s['tokens_per_step_mean']} tok/step) "
            f"mean={s['mean_tok_per_s']:5.2f} tok/s "
            f"fired={s['n_fired_default']:>3} "
            f"peak_vram={s['peak_vram_mb']:.0f} MB "
            f"eos@{s['eos_seen_at']}"
        )


if __name__ == "__main__":
    main()
