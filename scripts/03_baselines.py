"""Phase 3: SOTA baselines (the comparison set).

Drives the same manual decode loop as ``02_crv_probe.py`` but additionally:

  - records the **decoded text** of each logical step (since the previous
    boundary), so discourse-marker detection has something to scan;
  - records the **per-token next-token entropy** (mean over tokens *inside*
    the step), the ReaLM-Retrieve RSU stand-in;
  - records the **boundary-token next-token entropy**, the GRIP classifier
    stand-in;
  - evaluates **all four gates** at every boundary on the same physical
    trace (no model swap; the random gate is post-hoc rate-matched after
    the trace finishes);
  - emits one row per (step, gate) into ``runs/phase3_triggers_<...>.jsonl``
    matching the unified Phase 3.4 schema.

The driver intentionally re-runs the trace rather than reading Phase 2
JSONLs because Phase 2's saved features don't include text or entropy.
Throughput cost is small: one extra ``log_softmax`` + sum over the vocab
per decode step (~1 ms on the A800 for these models).

Usage:
  python scripts/03_baselines.py                                     # all 3 prompts on Qwen, mid-stack, sentence boundaries
  python scripts/03_baselines.py --model phi-4-mini-instruct
  python scripts/03_baselines.py --max-new-tokens 200 --prompt-class factual   # smoke
  python scripts/03_baselines.py --layer-mode last_layer                       # comparison run
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
from crv import (  # type: ignore[import-not-found]
    CRVTracker,
    CRVTrigger,
    GripSimGate,
    HiddenStateCapture,
    LatentGateAdapter,
    RandomGate,
    RealmRetrieveSimGate,
    StepBoundaryDetector,
)

DEFAULT_MODEL_DIR = "qwen3.6-35b-a3b-instruct"

# Same prompts as Phase 1.3 / 1.4 / 2 so the four-gate comparison sits on
# top of the same generation traces as the latent-only Phase 2 result.
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
    """Load a generator in bf16 native, fully on cuda:0. Same path as Phase 2."""
    import torch

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


def _entropy_nats(logits) -> float:
    """H = -sum p * log p, computed in bf16-safe fp32 from logits.

    ``logits`` is shape (1, vocab). We log_softmax in fp32 to avoid bf16
    underflow in the tail. Returns a Python float in nats (natural log).
    """
    import torch

    with torch.inference_mode():
        x = logits.detach().to(torch.float32).reshape(-1)
        log_p = torch.log_softmax(x, dim=-1)
        p = log_p.exp()
        h = -(p * log_p).sum().item()
    return float(h)


def run_one_prompt(
    label: str,
    prompt: str,
    tokenizer: Any,
    model: Any,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], dict]:
    """Drive the manual decode loop with hooks + CRV tracker + 3 sim gates.

    Returns
    -------
    trigger_rows
        Long-format list, one row per (step, gate). Final shape after
        random-gate finalization. Direct write target for
        ``runs/phase3_triggers_<...>.jsonl``.
    step_records
        One row per logical step with its raw inputs (text, entropies,
        CRV features). Useful for debugging and for Phase 4 to re-derive
        gate decisions with different thresholds without re-running the
        generator. Written to ``..._steps.jsonl``.
    summary
        Per-trace summary, including per-gate fire rates and gate metadata.
    """
    import torch

    messages = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(chat_text, return_tensors="pt").to(model.device)
    prompt_tokens = int(encoded.input_ids.shape[-1])

    n_layers = getattr(model.config, "num_hidden_layers", None)
    if n_layers is None and hasattr(model.config, "text_config"):
        n_layers = model.config.text_config.num_hidden_layers
    if n_layers is None:
        raise RuntimeError("cannot read num_hidden_layers from model.config")
    last_layer_idx = n_layers
    mid_layer_idx = n_layers // 2

    if args.layer_mode == "last_layer":
        capture_subset: list[int] = [last_layer_idx]
    elif args.layer_mode == "mid_stack":
        capture_subset = [mid_layer_idx]
    else:
        raise ValueError(
            f"unsupported layer_mode for Phase 3: {args.layer_mode!r}; "
            "use last_layer or mid_stack (matched to Phase 2 sweep)"
        )

    capture = HiddenStateCapture(
        layer_subset=capture_subset, every_n_steps=1
    )
    boundaries = StepBoundaryDetector(
        strategy=args.boundary,
        n_tokens=args.fixed_stride,
        tokenizer=tokenizer,
    )
    tracker = CRVTracker(
        layer_subset=capture_subset,
        layer_mode=args.layer_mode,
        window_size=args.window_size,
        drift_metric="cosine",
    )
    latent_trigger = CRVTrigger(
        threshold_drift=args.threshold_drift,
        threshold_variance=args.threshold_variance,
        mode="or",
    )

    latent_gate = LatentGateAdapter(latent_trigger)
    realm_gate = RealmRetrieveSimGate(rsu_threshold_nats=args.realm_rsu_threshold)
    grip_gate = GripSimGate(
        z_threshold=args.grip_z_threshold,
        strong_z_threshold=args.grip_strong_z_threshold,
        window_size=args.grip_window,
    )
    random_gate = RandomGate(seed=args.random_seed)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cache = None
    input_ids = encoded.input_ids
    attention_mask = encoded.attention_mask
    eos_id = tokenizer.eos_token_id
    eos_seen_at: int | None = None
    step_latencies: list[float] = []

    # Per-step accumulators (text + entropy across the tokens *between* boundaries).
    step_token_ids: list[int] = []
    step_token_entropies: list[float] = []

    step_records: list[dict] = []
    trigger_rows: list[dict] = []

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
                output_hidden_states=True,
            )
        next_logits = out.logits[:, -1, :]

        # Phase 3 addition: per-token entropy. Cheap (~1ms) and the GRIP /
        # ReaLM stand-ins both consume this signal.
        token_entropy = _entropy_nats(next_logits)

        next_token = next_logits.argmax(dim=-1, keepdim=True)
        cache = out.past_key_values
        token_id = int(next_token.item())

        captured = capture.record(step, out)
        tracker.add_token(captured)

        step_token_ids.append(token_id)
        step_token_entropies.append(token_entropy)

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

                step_text = tokenizer.decode(
                    step_token_ids, skip_special_tokens=False
                )
                mean_entropy = (
                    sum(step_token_entropies) / len(step_token_entropies)
                    if step_token_entropies else 0.0
                )
                boundary_entropy = step_token_entropies[-1] if step_token_entropies else 0.0

                step_record = {
                    "step_idx": int(feature["step_idx"]),
                    "token_idx_end": int(feature["token_idx_end"]),
                    "n_tokens_in_step": len(step_token_ids),
                    "boundary_token_id": int(token_id),
                    "step_text": step_text,
                    "mean_token_entropy": round(mean_entropy, 6),
                    "boundary_token_entropy": round(boundary_entropy, 6),
                    # Pass the CRV scalar features through, so step JSONL is
                    # self-contained for offline re-evaluation.
                    "drift_cosine": float(feature.get("drift_cosine", 0.0)),
                    "variance_l2_window": float(feature.get("variance_l2_window", 0.0)),
                    "hidden_norm": float(feature.get("hidden_norm", 0.0)),
                    "wall_clock_s": feature["wall_clock_s"],
                    "vram_allocated_mb": feature["vram_allocated_mb"],
                }
                step_records.append(step_record)

                # Online evaluation of the three "real" gates. The random
                # gate gets called too — it just returns a placeholder
                # row, real decisions land in finalize() after calibration.
                for gate in (latent_gate, realm_gate, grip_gate, random_gate):
                    g = gate.evaluate(step_record)
                    g.update({
                        "step_idx": step_record["step_idx"],
                        "token_idx": step_record["token_idx_end"],
                    })
                    trigger_rows.append(g)

                step_token_ids = []
                step_token_entropies = []

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

    # Post-pass: rate-match the random gate to the average of the three
    # real gates' fire rates over the same trace, then finalize().
    by_gate: dict[str, list[bool]] = {"latent": [], "realm_sim": [], "grip_sim": []}
    for row in trigger_rows:
        if row["gate_name"] in by_gate:
            by_gate[row["gate_name"]].append(bool(row["fired"]))

    rate = random_gate.calibrate_rate([by_gate["latent"], by_gate["realm_sim"], by_gate["grip_sim"]])
    finalized_random = random_gate.finalize(step_records)

    # Replace the placeholder random rows with the calibrated decisions.
    trigger_rows = [r for r in trigger_rows if r["gate_name"] != "random"]
    for r in finalized_random:
        # Match the schema: gate row has {step_idx, token_idx, gate_name,
        # fired, score, reason, simulation_caveats}.
        trigger_rows.append(r)

    # Sort for tidy on-disk output: by (step_idx, gate_name).
    gate_order = {"latent": 0, "realm_sim": 1, "grip_sim": 2, "random": 3}
    trigger_rows.sort(key=lambda r: (r["step_idx"], gate_order.get(r["gate_name"], 99)))

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
        "logical_steps_committed": len(step_records),
        "tokens_per_step_mean": round(
            len(step_latencies) / max(1, len(step_records)), 2
        ),
        "total_wall_clock_s": round(time.perf_counter() - t_start, 3),
        "mean_tok_per_s": round(len(step_latencies) / sum(step_latencies), 3),
        "vram_at_step0_mb": round(vram_before, 1),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
        "eos_seen_at": eos_seen_at,
        "fire_rates": {
            "latent": (sum(by_gate["latent"]) / len(by_gate["latent"])) if by_gate["latent"] else 0.0,
            "realm_sim": (sum(by_gate["realm_sim"]) / len(by_gate["realm_sim"])) if by_gate["realm_sim"] else 0.0,
            "grip_sim": (sum(by_gate["grip_sim"]) / len(by_gate["grip_sim"])) if by_gate["grip_sim"] else 0.0,
            "random": rate,
        },
        "fire_counts": {
            "latent": sum(by_gate["latent"]),
            "realm_sim": sum(by_gate["realm_sim"]),
            "grip_sim": sum(by_gate["grip_sim"]),
            "random": sum(1 for r in finalized_random if r["fired"]),
        },
        "gate_metadata": {
            "latent": latent_gate.metadata(),
            "realm_sim": realm_gate.metadata(),
            "grip_sim": grip_gate.metadata(),
            "random": random_gate.metadata(),
        },
        "boundary_stats": boundaries.stats,
        "capture_stats": capture.stats,
    }

    print(
        f"[{label}] done. {summary['logical_steps_committed']} steps from "
        f"{summary['tokens_generated']} tokens "
        f"(mean {summary['tokens_per_step_mean']} tok/step) | "
        f"mean={summary['mean_tok_per_s']} tok/s | "
        f"fire latent={summary['fire_counts']['latent']} "
        f"realm={summary['fire_counts']['realm_sim']} "
        f"grip={summary['fire_counts']['grip_sim']} "
        f"random={summary['fire_counts']['random']} (p={rate:.3f}) | "
        f"peak_vram={summary['peak_vram_mb']} MB"
    )

    return trigger_rows, step_records, summary


def write_outputs(
    args: argparse.Namespace,
    label: str,
    trigger_rows: list[dict],
    step_records: list[dict],
    summary: dict,
) -> None:
    runs_dir = Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    base = f"phase3_triggers_{args.model}_{label}_{args.boundary}_{args.layer_mode}"
    triggers_path = runs_dir / f"{base}.jsonl"
    steps_path = runs_dir / f"{base}_steps.jsonl"
    summary_path = runs_dir / f"{base}_summary.json"

    run_id = f"{args.model}|{label}|{args.boundary}|{args.layer_mode}"
    with triggers_path.open("w") as f:
        for row in trigger_rows:
            row_out = {"run_id": run_id, "model": args.model, "prompt_class": label, **row}
            f.write(json.dumps(row_out) + "\n")
    with steps_path.open("w") as f:
        for row in step_records:
            f.write(json.dumps(row) + "\n")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"  wrote: {triggers_path}  ({len(trigger_rows)} rows)\n"
        f"         {steps_path}     ({len(step_records)} rows)\n"
        f"         {summary_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=5000)
    parser.add_argument(
        "--prompt-class", default="all", choices=["all", *PROMPTS.keys()]
    )
    parser.add_argument(
        "--boundary", default="sentence",
        choices=["sentence", "think_block", "fixed"],
    )
    parser.add_argument("--fixed-stride", type=int, default=32)
    parser.add_argument(
        "--layer-mode", default="mid_stack",
        choices=["last_layer", "mid_stack"],
        help="mid_stack matches Phase 2's L1-mitigation canonical run.",
    )
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--threshold-drift", type=float, default=0.1)
    parser.add_argument("--threshold-variance", type=float, default=1.0)
    parser.add_argument("--realm-rsu-threshold", type=float, default=2.5)
    parser.add_argument("--grip-z-threshold", type=float, default=1.5)
    parser.add_argument("--grip-strong-z-threshold", type=float, default=2.5)
    parser.add_argument("--grip-window", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260509)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    model_dir = Path("models") / args.model
    if not model_dir.exists():
        print(f"[!!] missing {model_dir}. Run: bash scripts/download_models.sh phase1.4")
        sys.exit(1)

    print("[..] importing torch + transformers + bitsandbytes ...")
    import torch  # noqa: F401  (primes the env)

    tokenizer, model = load_generator(model_dir)

    selected = list(PROMPTS.keys()) if args.prompt_class == "all" else [args.prompt_class]

    headline: list[dict] = []
    for label in selected:
        trigger_rows, step_records, summary = run_one_prompt(
            label, PROMPTS[label], tokenizer, model, args
        )
        write_outputs(args, label, trigger_rows, step_records, summary)
        headline.append(summary)

    print("\n=== headline (Phase 3 per-trace summary) ===")
    for s in headline:
        fr = s["fire_rates"]
        print(
            f"  {s['prompt_class']:>9}: "
            f"steps={s['logical_steps_committed']:>4}  "
            f"latent={fr['latent']:.2%} "
            f"realm={fr['realm_sim']:.2%} "
            f"grip={fr['grip_sim']:.2%} "
            f"random={fr['random']:.2%} | "
            f"mean={s['mean_tok_per_s']:5.2f} tok/s "
            f"peak_vram={s['peak_vram_mb']:.0f} MB"
        )


if __name__ == "__main__":
    main()
