"""Canonical decode helpers — chat-template loading, model-loading, EM/F1
grading — shared by the SOTA-capture script `04f_run_qa_sota.py`.

For the paper's headline pool we never invoke this script's `main()`
directly; it is kept because `04f_run_qa_sota.py` imports it for the
canonical loaders and the per-model chat-template / repetition-penalty
protocol map (MODEL_PROTOCOL below).
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
from qa_grading import exact_match, extract_answer, token_f1  # type: ignore[import-not-found]

DEFAULT_DATASET_PATH = "data/hotpotqa_distractor_200.jsonl"
DEFAULT_MANIFEST_PATH = "data/hotpotqa_distractor_200.manifest.json"


SYSTEM_INSTRUCTION = (
    "You are a careful reasoner. Read the provided paragraphs, think step "
    "by step, and answer the question concisely. End your response with a "
    "single line of the form 'Answer: <your answer>' where <your answer> "
    "is as short as possible (typically 1-5 words)."
)


# "Option A" per-model protocol — chat-template flags + repetition-penalty
# guards for the two headline-pool architectures.  Both default to
# no-thinking-mode (Qwen 2.5 has no thinking-mode mechanism in its chat
# template, just standard ChatML; Gemma 4 26B A4B emits an empty thought
# channel by default and we keep enable_thinking unset).
MODEL_PROTOCOL: dict[str, dict[str, Any]] = {
    "qwen2.5-7b-instruct": {
        "chat_template_kwargs": {},
        "repetition_penalty": 1.0,
    },
    "gemma-4-26b-a4b-it": {
        "chat_template_kwargs": {},
        "repetition_penalty": 1.0,
    },
}

import re as _re  # noqa: E402

_ANSWER_MARKER_RE = _re.compile(r"\bAnswer\s*:\s*([^\n]+)\n", _re.IGNORECASE)


def _format_paragraphs(paragraphs: list[dict[str, Any]]) -> str:
    """Render the 10 paragraphs into a model-readable block.

    Order is preserved from the dataset (no shuffling -- a different
    question entirely if we did). Title is shown so the model can learn to
    reference paragraphs by name when reasoning.
    """
    parts = []
    for p in paragraphs:
        body = p["joined_text"].strip()
        parts.append(f"[Paragraph {p['paragraph_idx']}: {p['title']}]\n{body}")
    return "\n\n".join(parts)


def _build_user_payload(question: str, paragraphs: list[dict[str, Any]], inline_system: bool = False) -> str:
    """Option A: user payload no longer contains the system instruction.
    `inline_system=True` is the 4.5c ablation: stuff SYSTEM_INSTRUCTION into the
    user message instead of using a system role, to test whether the EM dip on
    Gemma between 4.1 (system-in-user, accidental) and 4.1b (system role) is
    protocol-induced."""
    head = f"{SYSTEM_INSTRUCTION}\n\n" if inline_system else ""
    return (
        f"{head}Paragraphs:\n\n{_format_paragraphs(paragraphs)}\n\n"
        f"Question: {question}\n\n"
        f"Reason step by step, then end with 'Answer: ...'."
    )


def _build_messages(question: str, paragraphs: list[dict[str, Any]], inline_system: bool = False) -> list[dict]:
    user = {"role": "user", "content": _build_user_payload(question, paragraphs, inline_system=inline_system)}
    if inline_system:
        return [user]
    return [{"role": "system", "content": SYSTEM_INSTRUCTION}, user]


def _apply_repetition_penalty(logits, generated_ids: list[int], penalty: float):
    """HF-standard CTRL repetition penalty applied in-place to a single-step row.

    No-op when penalty == 1.0 or no tokens have been generated yet.
    """
    if penalty == 1.0 or not generated_ids:
        return logits
    import torch
    ids = torch.tensor(list(set(generated_ids)), device=logits.device, dtype=torch.long)
    selected = logits[0, ids]
    selected = torch.where(selected > 0, selected / penalty, selected * penalty)
    logits[0, ids] = selected
    return logits


def _format_mb(mb: int | None) -> str:
    return "n/a" if mb is None else f"{mb} MB"


def load_generator(model_dir: Path, attn_implementation: str | None = None):
    """Load a generator in bf16 native, fully on cuda:0 (canonical Phase 1.4 path).

    ``attn_implementation`` defaults to None (HF picks: sdpa/flash on GPU).
    Pass ``"eager"`` from the SOTA driver (04f) when ``output_attentions=True``
    is needed downstream — sdpa silently returns ``out.attentions = (None, ...)``.
    """
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
    _from_pretrained_kwargs: dict = dict(
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    if attn_implementation is not None:
        _from_pretrained_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        **_from_pretrained_kwargs,
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
    """H = -sum p * log p, computed in fp32 from bf16 logits, in nats."""
    import torch

    with torch.inference_mode():
        x = logits.detach().to(torch.float32).reshape(-1)
        log_p = torch.log_softmax(x, dim=-1)
        p = log_p.exp()
        h = -(p * log_p).sum().item()
    return float(h)


def _resolve_layer_subset(model, layer_mode: str) -> list[int]:
    n_layers = getattr(model.config, "num_hidden_layers", None)
    if n_layers is None and hasattr(model.config, "text_config"):
        n_layers = model.config.text_config.num_hidden_layers
    if n_layers is None:
        raise RuntimeError("cannot read num_hidden_layers from model.config")
    last_layer_idx = n_layers
    mid_layer_idx = n_layers // 2
    if layer_mode == "last_layer":
        return [last_layer_idx]
    if layer_mode == "mid_stack":
        return [mid_layer_idx]
    raise ValueError(f"unsupported layer_mode: {layer_mode!r}")


def run_one_question(
    record: dict[str, Any],
    tokenizer: Any,
    model: Any,
    capture_subset: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Generate one trace + grade it. Returns the per-question record to JSONL."""
    import torch

    proto = MODEL_PROTOCOL.get(args.model, {"chat_template_kwargs": {}, "repetition_penalty": 1.0})
    chat_template_kwargs = proto["chat_template_kwargs"]
    repetition_penalty = float(proto["repetition_penalty"])

    messages = _build_messages(
        record["question"], record["paragraphs"], inline_system=getattr(args, "inline_system", False)
    )
    try:
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
        )
    except TypeError:
        # Tokenizer's chat template does not accept the kwarg(s); fall back to
        # default render. Logged in the per-question record so the trace is
        # honest about whether enable_thinking actually took effect.
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        chat_template_kwargs = {"_unsupported": dict(chat_template_kwargs)}
    encoded = tokenizer(chat_text, return_tensors="pt").to(model.device)
    prompt_tokens = int(encoded.input_ids.shape[-1])

    capture = HiddenStateCapture(layer_subset=capture_subset, every_n_steps=1)
    boundaries = StepBoundaryDetector(
        strategy="sentence", n_tokens=32, tokenizer=tokenizer
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

    cache = None
    input_ids = encoded.input_ids
    attention_mask = encoded.attention_mask
    eos_id = tokenizer.eos_token_id
    eos_seen_at: int | None = None

    generated_token_ids: list[int] = []
    step_token_ids: list[int] = []
    step_token_entropies: list[float] = []

    step_records: list[dict] = []
    trigger_rows: list[dict] = []

    early_stop_reason = "max_tokens"
    answer_marker_seen_at: int | None = None

    t_start = time.perf_counter()

    for step in range(args.max_new_tokens):
        with torch.inference_mode():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask if step == 0 else None,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
            )
        next_logits = out.logits[:, -1, :].clone()  # clone so we can mutate for rep_penalty
        token_entropy = _entropy_nats(next_logits)

        next_logits = _apply_repetition_penalty(next_logits, generated_token_ids, repetition_penalty)
        next_token = next_logits.argmax(dim=-1, keepdim=True)
        cache = out.past_key_values
        token_id = int(next_token.item())

        captured = capture.record(step, out)
        tracker.add_token(captured)

        generated_token_ids.append(token_id)
        step_token_ids.append(token_id)
        step_token_entropies.append(token_entropy)

        if eos_id is not None and token_id == eos_id:
            eos_seen_at = step + 1
            early_stop_reason = "eos"

        # Cheap periodic answer-marker check on the running decoded text.
        # Catches models that emit a turn-end token we don't recognize as EOS.
        # Decode is O(step) per call so only run every 16 steps -- amortized
        # cost is negligible vs the forward pass.
        if (
            answer_marker_seen_at is None
            and (step + 1) % 16 == 0
            and len(generated_token_ids) >= 16
        ):
            tail_text = tokenizer.decode(generated_token_ids[-256:], skip_special_tokens=True)
            m = _ANSWER_MARKER_RE.search(tail_text)
            if m:
                answer_marker_seen_at = step + 1
                if early_stop_reason == "max_tokens":
                    early_stop_reason = "answer_marker"

        if boundaries.is_boundary(token_id):
            feature = tracker.commit_step(
                step_idx=boundaries._step_count,
                token_idx=step + 1,
                boundary_token_id=token_id,
            )
            if feature is not None:
                step_text = tokenizer.decode(step_token_ids, skip_special_tokens=False)
                mean_entropy = (
                    sum(step_token_entropies) / len(step_token_entropies)
                    if step_token_entropies else 0.0
                )
                boundary_entropy = (
                    step_token_entropies[-1] if step_token_entropies else 0.0
                )
                step_record = {
                    "step_idx": int(feature["step_idx"]),
                    "token_idx_end": int(feature["token_idx_end"]),
                    "n_tokens_in_step": len(step_token_ids),
                    "boundary_token_id": int(token_id),
                    "step_text": step_text,
                    "mean_token_entropy": round(mean_entropy, 6),
                    "boundary_token_entropy": round(boundary_entropy, 6),
                    "drift_cosine": float(feature.get("drift_cosine", 0.0)),
                    "variance_l2_window": float(feature.get("variance_l2_window", 0.0)),
                    "hidden_norm": float(feature.get("hidden_norm", 0.0)),
                }
                step_records.append(step_record)

                for gate in (latent_gate, realm_gate, grip_gate, random_gate):
                    g = gate.evaluate(step_record)
                    g.update({
                        "step_idx": step_record["step_idx"],
                        "token_idx": step_record["token_idx_end"],
                    })
                    trigger_rows.append(g)

                step_token_ids = []
                step_token_entropies = []

        input_ids = next_token
        attention_mask = None
        torch.cuda.synchronize()

        if eos_seen_at is not None:
            break  # Stop on EOS so the trace doesn't run past the answer.
        if answer_marker_seen_at is not None:
            break  # Option A: also stop on a clean Answer:<text>\n line.

    wall = time.perf_counter() - t_start
    tokens_generated = len(generated_token_ids)

    by_gate: dict[str, list[bool]] = {"latent": [], "realm_sim": [], "grip_sim": []}
    for row in trigger_rows:
        if row["gate_name"] in by_gate:
            by_gate[row["gate_name"]].append(bool(row["fired"]))
    rate = random_gate.calibrate_rate(
        [by_gate["latent"], by_gate["realm_sim"], by_gate["grip_sim"]]
    )
    finalized_random = random_gate.finalize(step_records)
    trigger_rows = [r for r in trigger_rows if r["gate_name"] != "random"]
    trigger_rows.extend(finalized_random)
    gate_order = {"latent": 0, "realm_sim": 1, "grip_sim": 2, "random": 3}
    trigger_rows.sort(key=lambda r: (r["step_idx"], gate_order.get(r["gate_name"], 99)))

    fire_counts = {
        "latent": sum(by_gate["latent"]),
        "realm_sim": sum(by_gate["realm_sim"]),
        "grip_sim": sum(by_gate["grip_sim"]),
        "random": sum(1 for r in finalized_random if r["fired"]),
    }
    fire_rates = {
        k: (v / max(1, len(step_records))) for k, v in fire_counts.items()
    }
    fire_rates["random"] = rate  # use the calibration rate verbatim

    full_generation = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    extracted, extract_strategy = extract_answer(full_generation)
    em = exact_match(extracted, record["answer"])
    f1 = token_f1(extracted, record["answer"])

    return {
        "id": record["id"],
        "split": record.get("split", "unknown"),
        "level": record.get("level", ""),
        "type": record.get("type", ""),
        "question": record["question"],
        "gold_answer": record["answer"],
        "model": args.model,
        "layer_mode": args.layer_mode,
        "prompt_tokens": prompt_tokens,
        "tokens_generated": tokens_generated,
        "wall_clock_s": round(wall, 3),
        "tok_per_s": round(tokens_generated / wall, 3) if wall > 0 else 0.0,
        "eos_seen_at": eos_seen_at,
        "answer_marker_seen_at": answer_marker_seen_at,
        "early_stop_reason": early_stop_reason,
        "chat_template_kwargs_used": chat_template_kwargs,
        "repetition_penalty_used": repetition_penalty,
        "inline_system": bool(getattr(args, "inline_system", False)),
        "n_steps": len(step_records),
        "model_answer_extracted": extracted,
        "model_answer_extract_strategy": extract_strategy,
        "model_answer_full_generation_tail": full_generation[-400:],
        "em": em,
        "f1": round(f1, 4),
        "label_correct_em": bool(em),
        "label_correct_f1_50": bool(f1 >= 0.5),
        "fire_counts": fire_counts,
        "fire_rates": {k: round(v, 4) for k, v in fire_rates.items()},
        "steps": step_records,
        "gate_decisions": trigger_rows,
        "trigger_thresholds": {
            "drift": args.threshold_drift,
            "variance": args.threshold_variance,
            "realm_rsu": args.realm_rsu_threshold,
            "grip_z": args.grip_z_threshold,
            "grip_strong_z": args.grip_strong_z_threshold,
            "random_seed": args.random_seed,
            "_note": "PLACEHOLDER thresholds; Phase 4.4 will refit on calibration split only.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--split", default="all",
        choices=["all", "calibration", "validation", "test"],
        help="Filter to one split (default: all 200 questions).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap question count for smoke runs.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=1024,
        help="Default raised from 512 to 1024 in Phase 4.1b (Option A re-run).",
    )
    parser.add_argument(
        "--layer-mode", default="mid_stack",
        choices=["last_layer", "mid_stack"],
    )
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--threshold-drift", type=float, default=0.1)
    parser.add_argument("--threshold-variance", type=float, default=1.0)
    parser.add_argument("--realm-rsu-threshold", type=float, default=2.5)
    parser.add_argument("--grip-z-threshold", type=float, default=1.5)
    parser.add_argument("--grip-strong-z-threshold", type=float, default=2.5)
    parser.add_argument("--grip-window", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260509)
    parser.add_argument(
        "--out-suffix", default="",
        help="Append to output filename (e.g. '_smoke') for multiple parallel runs.",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--inline-system", action="store_true",
        help="4.5c ablation: stuff SYSTEM_INSTRUCTION into the user message instead "
             "of using a system role. Replicates the 4.1 (rejected) Gemma protocol "
             "but with everything else from 4.1b held constant.",
    )
    args = parser.parse_args()

    model_dir = Path("models") / args.model
    if not model_dir.exists():
        print(f"[!!] missing {model_dir}.")
        sys.exit(1)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[!!] missing dataset {dataset_path}. Run scripts/04_dataset.py first.")
        sys.exit(1)

    questions: list[dict[str, Any]] = []
    with dataset_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if args.split != "all" and row.get("split") != args.split:
                continue
            questions.append(row)
    if args.limit:
        questions = questions[: args.limit]

    print(
        f"[..] dataset {dataset_path} -> {len(questions)} questions "
        f"(split filter: {args.split}{'; limit ' + str(args.limit) if args.limit else ''})"
    )

    print(f"[..] importing torch + transformers + bitsandbytes ...")
    import torch  # noqa: F401  (primes the env)

    tokenizer, model = load_generator(model_dir)
    capture_subset = _resolve_layer_subset(model, args.layer_mode)
    print(f"[..] layer_mode={args.layer_mode}  capture_subset={capture_subset}")

    runs_dir = Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = runs_dir / f"phase4_qa_{args.model}_{args.layer_mode}{args.out_suffix}.jsonl"
    print(f"[..] writing to {out_path}")

    sum_em = 0.0
    sum_f1 = 0.0
    sum_tokens = 0
    sum_wall = 0.0
    extract_strategy_counts: dict[str, int] = {}
    sweep_t0 = time.perf_counter()

    with out_path.open("w") as f:
        for q_idx, record in enumerate(questions):
            try:
                row = run_one_question(record, tokenizer, model, capture_subset, args)
            except Exception as e:
                print(f"  [q{q_idx}] {record['id']} FAILED: {type(e).__name__}: {e}")
                row = {
                    "id": record["id"],
                    "split": record.get("split"),
                    "model": args.model,
                    "layer_mode": args.layer_mode,
                    "error": f"{type(e).__name__}: {e}",
                }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

            if "error" not in row:
                sum_em += row["em"]
                sum_f1 += row["f1"]
                sum_tokens += row["tokens_generated"]
                sum_wall += row["wall_clock_s"]
                strat = row["model_answer_extract_strategy"]
                extract_strategy_counts[strat] = extract_strategy_counts.get(strat, 0) + 1

            if (q_idx + 1) % args.progress_every == 0 or q_idx == len(questions) - 1:
                elapsed = time.perf_counter() - sweep_t0
                done = q_idx + 1
                avg_per_q = elapsed / done
                eta = avg_per_q * (len(questions) - done)
                print(
                    f"  [{done:>3}/{len(questions)}] {record['id']:24} "
                    f"split={row.get('split','?'):12} "
                    f"em={row.get('em','-')} f1={row.get('f1','-')} "
                    f"tok={row.get('tokens_generated','-'):>4} "
                    f"steps={row.get('n_steps','-'):>3} "
                    f"strategy={row.get('model_answer_extract_strategy','-'):14} "
                    f"| avg/q={avg_per_q:5.1f}s  eta={eta/60:5.1f}m"
                )

    n_done = sum(1 for _ in [None] * len(questions))  # placeholder; overwrite below
    # Actually count successful rows by re-reading (avoids tracking a parallel count).
    n_ok = 0
    with out_path.open() as f:
        for line in f:
            row = json.loads(line)
            if "error" not in row:
                n_ok += 1
    n_err = len(questions) - n_ok

    if n_ok > 0:
        print(
            f"\n=== headline (Phase 4.1 -- {args.model} / {args.layer_mode}) ===\n"
            f"  n_ok           = {n_ok}  (n_err = {n_err})\n"
            f"  EM             = {sum_em/n_ok:.4f}  ({int(sum_em)}/{n_ok})\n"
            f"  token-F1       = {sum_f1/n_ok:.4f}\n"
            f"  mean tok/q     = {sum_tokens/n_ok:.1f}\n"
            f"  mean wall/q    = {sum_wall/n_ok:.2f} s\n"
            f"  extract strat. = {extract_strategy_counts}\n"
            f"  out            = {out_path}"
        )
    else:
        print(f"[!!] no questions completed successfully ({n_err} errors)")


if __name__ == "__main__":
    main()
