"""Phase 4.8/4.9/4.10 driver: per-question multi-hop QA + true-SOTA captures.

Drop-in replacement for ``scripts/04_run_qa.py`` that *additionally* captures
the auxiliary signals the three true-SOTA baseline gates need:

  - DRAGIN ``RIND = H_i * a_max(i) * s_i`` over content tokens
    (requires ``output_attentions=True`` on the forward pass).
  - ReaLM-Retrieve RSUS = α·U_verb + β·U_ent + γ·U_cons, with
    α=0.40, β=0.35, γ=0.25, τ=0.65 (Guo et al. Footnote 1).
  - SELFDOUBT-Full: Tier-1 HVR fused with U_verb by per-model z-scores.

Reuses ``04_run_qa`` for chat-template / model-loading / grading / dataset
plumbing (imported via ``importlib.import_module("04_run_qa")``) and only
diverges in the decode loop, where the SOTA captures are interleaved.

Output JSONL schema is a superset of ``runs/phase4_qa_<model>_<lm>.jsonl``:
each step record gains the fields documented in
``crv/baseline_gates.py`` ("Step-record contract for the SOTA gates").

Compute envelope (per `_AGENT.md` 2026-05-11 #17):

  - Verbalized-confidence probe:    ~12 extra forward steps per step boundary.
  - Consistency sampling (K=3, 40t): ~120 extra forward steps per step boundary.
  - Entity entropy:                 CPU-only spaCy + BM25; negligible.
  - DRAGIN attention capture:       no extra compute (just ``output_attentions``).

Estimated wall-clock at ~8 steps/question, 200 questions, 3 archs on A800:
~15-18 hours total (matches Guo et al.'s reported overhead).

Usage (compute box):
  python scripts/04f_run_qa_sota.py --model qwen3.6-35b-a3b-instruct --limit 5  # smoke
  python scripts/04f_run_qa_sota.py --model phi-4-mini-instruct                  # full
  python scripts/04f_run_qa_sota.py --model gemma-4-26b-a4b-it
  python scripts/04f_run_qa_sota.py --model qwen3.6-35b-a3b-instruct --no-cons   # skip U_cons

TODO(compute-box):
  - Verify ``output_attentions=True`` returns expected shape on each arch
    (Qwen 3.6 hybrid stack may need ``attn_implementation="eager"`` because
    flash-attention does not materialise attentions). See smoke test 04h.
  - On Qwen 3.6, the layer index of the last *full-attention* block must
    be passed via ``--dragin-attn-layer-idx``. The full-attention indices
    are read from ``config.layer_types``; the smoke test prints them.
  - Per-step wall-clock budget: the consistency sampler dominates. If
    runtime overruns, drop K from 3 to 2 (``--cons-k 2``) — Guo et al.
    Footnote 2 documents that this trades 0.4% AUROC for ~33% speed-up.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Reuse the canonical loaders / helpers from 04_run_qa.
qa = import_module("04_run_qa")  # type: ignore

from crv import (  # type: ignore[import-not-found]
    CRVTracker,
    CRVTrigger,
    DraginGate,
    EtcGate,
    HiddenStateCapture,
    LatentGateAdapter,
    RandomGate,
    RealmRsusGate,
    SelfDoubtFullGate,
    SelfDoubtTier1Gate,
    StepBoundaryDetector,
    hedge_verify_counts,
)
from crv.sota_capture import (  # type: ignore[import-not-found]
    ConsistencySampler,
    EntityEntropyScorer,
    RindAccumulator,
    VerbalizedConfidenceProbe,
    compute_token_entropy_nats,
)
from qa_grading import exact_match, extract_answer, token_f1  # type: ignore[import-not-found]


def _resolve_dragin_attn_layer(model: Any, override: int | None) -> int:
    """Pick which entry of ``out.attentions`` DRAGIN reads.

    Returns an index into the **attentions tuple**, not necessarily the
    model-layer index. They diverge on Qwen 3.6 and other hybrid stacks:

    * Dense / sliding-window stacks (Phi-4-mini, Gemma 4): every layer
      contributes an entry to ``out.attentions``; tuple length equals
      ``num_hidden_layers``. The last full-attention layer's model-index
      equals its tuple-index. Picking ``num_full - 1`` and ``n_layers - 1``
      give the same answer.
    * Hybrid (Qwen 3.6, 40 layers, 3:1 GDN : full pattern): transformers
      5.8 returns a **compact** tuple of length 10 containing only the
      full-attention layers' attentions — the GDN layers do NOT produce
      None entries, they are dropped. The model-layer index 39 of the
      rightmost full-attn layer is therefore wrong; the correct tuple
      index is 9. We compute it as ``count(full_attention layers in
      layer_types) - 1``.

    For backwards compatibility with calling code that wants the
    model-layer index (e.g. for logging), use ``override`` to pin it
    explicitly. The smoke test #18 / production sweep #19+ both read
    the tuple index returned here.
    """
    if override is not None:
        return int(override)
    cfg = model.config
    layer_types = getattr(cfg, "layer_types", None)
    n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg.text_config, "num_hidden_layers", None)
    if layer_types:
        full_positions = [i for i, t in enumerate(layer_types) if "full" in str(t).lower()]
        if not full_positions:
            return int(n_layers) - 1
        if len(full_positions) < len(layer_types):
            # Hybrid stack: attentions tuple is compact (full-attn-only).
            return len(full_positions) - 1
        # All-full stack: tuple-index == model-index.
        return full_positions[-1]
    return int(n_layers) - 1


def run_one_question_sota(
    record: dict[str, Any],
    tokenizer: Any,
    model: Any,
    capture_subset: list[int],
    probes: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    proto = qa.MODEL_PROTOCOL.get(args.model, {"chat_template_kwargs": {}, "repetition_penalty": 1.0})
    chat_template_kwargs = proto["chat_template_kwargs"]
    repetition_penalty = float(proto["repetition_penalty"])

    messages = qa._build_messages(
        record["question"], record["paragraphs"], inline_system=getattr(args, "inline_system", False)
    )
    try:
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
        )
    except TypeError:
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        chat_template_kwargs = {"_unsupported": dict(chat_template_kwargs)}
    encoded = tokenizer(chat_text, return_tensors="pt").to(model.device)
    prompt_tokens = int(encoded.input_ids.shape[-1])

    # Existing capture / boundary / latent infra (unchanged from 04_run_qa).
    capture = HiddenStateCapture(layer_subset=capture_subset, every_n_steps=1)
    boundaries = StepBoundaryDetector(strategy="sentence", n_tokens=32, tokenizer=tokenizer)
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
    etc_gate = EtcGate(d2h_threshold=args.etc_threshold)
    selfdoubt_t1 = SelfDoubtTier1Gate(hvr_threshold=args.selfdoubt_t1_threshold)
    dragin_gate = DraginGate(rind_threshold=args.dragin_threshold)
    realm_rsus = RealmRsusGate(rsus_threshold=args.rsus_threshold)
    # SelfDoubt-Full needs calibration moments; we run with no-moments here so
    # every per-step decision falls back to Tier-1 HVR. Moment fitting happens
    # offline in 04g_sota_showdown.py (calibration split).
    selfdoubt_full = SelfDoubtFullGate(z_threshold=args.selfdoubt_full_z)
    random_gate = RandomGate(seed=args.random_seed)

    # New: per-question RIND accumulator and per-question entity scorer.
    rind_acc = RindAccumulator(prompt_len=prompt_tokens)
    ent_scorer = (
        probes["entity_scorer_factory"](record["paragraphs"])
        if probes.get("entity_scorer_factory") and not args.no_ent
        else None
    )

    dragin_layer_idx = probes["dragin_layer_idx"]

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
                output_attentions=True,
            )
        next_logits = out.logits[:, -1, :].clone()
        token_entropy = compute_token_entropy_nats(next_logits)

        next_logits = qa._apply_repetition_penalty(next_logits, generated_token_ids, repetition_penalty)
        next_token = next_logits.argmax(dim=-1, keepdim=True)
        cache = out.past_key_values
        token_id = int(next_token.item())

        # Decode just this token's surface text for the stopword filter.
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)

        captured = capture.record(step, out)
        tracker.add_token(captured)

        # DRAGIN: observe the newly generated token, then ingest the new
        # attention row that just arrived in out.attentions[dragin_layer_idx].
        rind_acc.observe_new_token(token_text=token_text, entropy=token_entropy)
        # out.attentions may be None on archs/configs that don't support
        # attentions (e.g. flash-attention-only); skip gracefully.
        attns = getattr(out, "attentions", None)
        if attns is not None and dragin_layer_idx < len(attns):
            attn_layer = attns[dragin_layer_idx]
            if attn_layer is not None:
                rind_acc.ingest_attention_row(attn_layer)

        generated_token_ids.append(token_id)
        step_token_ids.append(token_id)
        step_token_entropies.append(token_entropy)

        if eos_id is not None and token_id == eos_id:
            eos_seen_at = step + 1
            early_stop_reason = "eos"

        if (
            answer_marker_seen_at is None
            and (step + 1) % 16 == 0
            and len(generated_token_ids) >= 16
        ):
            tail_text = tokenizer.decode(generated_token_ids[-256:], skip_special_tokens=True)
            m = qa._ANSWER_MARKER_RE.search(tail_text)
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
                boundary_entropy = step_token_entropies[-1] if step_token_entropies else 0.0

                # SOTA per-step features ----------------------------------
                rind = rind_acc.commit_step()
                rind_max = float(rind["rind_max"]) if rind else 0.0

                h_count, v_count = hedge_verify_counts(step_text)
                hvr_val = h_count / (v_count + 1)

                # ReaLM-RSUS captures (each can be disabled via flags).
                u_verb: float | None = None
                u_verb_text = ""
                if not args.no_verb and probes.get("verbal_probe") is not None:
                    try:
                        u_verb = probes["verbal_probe"].probe(cache, token_id)
                    except Exception as exc:  # pragma: no cover
                        u_verb = None
                        u_verb_text = f"probe_error:{type(exc).__name__}"

                u_ent: float | None = None
                ent_meta: dict[str, Any] = {}
                if ent_scorer is not None:
                    e = ent_scorer.score(step_text)
                    u_ent = float(e["u_ent"])
                    ent_meta = {"n_entities": e["n_entities"], "u_ent_H_nats": e["mean_entropy_nats"]}

                u_cons: float | None = None
                cons_meta: dict[str, Any] = {}
                if not args.no_cons and probes.get("consistency_sampler") is not None:
                    try:
                        c = probes["consistency_sampler"].score(cache, token_id)
                        u_cons = float(c["u_cons"])
                        cons_meta = {
                            "mean_pairwise_cos": c["mean_pairwise_cosine"],
                            "n_samples": c["n_samples"],
                        }
                    except Exception as exc:  # pragma: no cover
                        u_cons = None
                        cons_meta = {"sampler_error": type(exc).__name__}

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
                    # ---- SOTA-baseline features ----
                    "dragin_rind_max": rind_max,
                    "dragin_n_content_tokens": int(rind["n_content_tokens"]) if rind else 0,
                    "hvr": float(hvr_val),
                    "hedge_count": int(h_count),
                    "verify_count": int(v_count),
                    "u_verb": u_verb,
                    "u_verb_note": u_verb_text,
                    "u_ent": u_ent,
                    "u_ent_meta": ent_meta,
                    "u_cons": u_cons,
                    "u_cons_meta": cons_meta,
                }
                step_records.append(step_record)

                for gate in (
                    latent_gate, etc_gate, selfdoubt_t1, dragin_gate,
                    realm_rsus, selfdoubt_full, random_gate,
                ):
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
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if eos_seen_at is not None:
            break
        if answer_marker_seen_at is not None:
            break

    wall = time.perf_counter() - t_start
    tokens_generated = len(generated_token_ids)

    # Calibrate the random gate post-hoc on the active per-step decisions
    # of the *new* active pool {latent, etc, selfdoubt_t1, dragin, realm_rsus,
    # selfdoubt_full}, matching the convention in 04_run_qa.
    by_gate: dict[str, list[bool]] = {
        "latent": [], "etc": [], "selfdoubt_t1": [], "dragin": [],
        "realm_rsus": [], "selfdoubt_full": [],
    }
    for row in trigger_rows:
        if row["gate_name"] in by_gate:
            by_gate[row["gate_name"]].append(bool(row["fired"]))
    rate = random_gate.calibrate_rate(list(by_gate.values()))
    finalized_random = random_gate.finalize(step_records)
    trigger_rows = [r for r in trigger_rows if r["gate_name"] != "random"]
    trigger_rows.extend(finalized_random)

    fire_counts = {g: sum(by_gate[g]) for g in by_gate}
    fire_counts["random"] = sum(1 for r in finalized_random if r["fired"])
    fire_rates = {k: (v / max(1, len(step_records))) for k, v in fire_counts.items()}
    fire_rates["random"] = rate

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
        "sota_config": {
            "dragin_attn_layer_idx": dragin_layer_idx,
            "rsus_weights": realm_rsus.weights,
            "rsus_threshold": realm_rsus.rsus_threshold,
            "u_verb_enabled": not args.no_verb,
            "u_ent_enabled": not args.no_ent,
            "u_cons_enabled": not args.no_cons,
            "_note": "PLACEHOLDER thresholds for selfdoubt_full/dragin; 04g calibrates on calibration split only.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-dir", default=None,
                        help="Override the implicit models/<model> path. "
                             "Useful when running from a v2 RUN_DIR with non-default layout.")
    parser.add_argument("--output", default=None,
                        help="Override the implicit runs/phase4_qa_<model>_<layer_mode><out_suffix>.jsonl path. "
                             "Required for v2 multi-phase runs to avoid clobbering across phases.")
    parser.add_argument("--dataset", default=qa.DEFAULT_DATASET_PATH)
    parser.add_argument("--manifest", default=qa.DEFAULT_MANIFEST_PATH)
    parser.add_argument("--split", default="all", choices=["all", "calibration", "validation", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--layer-mode", default="mid_stack", choices=["last_layer", "mid_stack"])
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--threshold-drift", type=float, default=0.1)
    parser.add_argument("--threshold-variance", type=float, default=1.0)
    parser.add_argument("--etc-threshold", type=float, default=0.5)
    parser.add_argument("--selfdoubt-t1-threshold", type=float, default=1.0)
    parser.add_argument("--selfdoubt-full-z", type=float, default=1.0)
    parser.add_argument("--dragin-threshold", type=float, default=1.0)
    parser.add_argument("--dragin-attn-layer-idx", type=int, default=None,
                        help="Override the layer index whose attentions DRAGIN reads. "
                             "Default = auto-detect last full-attention layer.")
    parser.add_argument("--rsus-threshold", type=float, default=0.65)
    parser.add_argument("--random-seed", type=int, default=20260509)
    parser.add_argument("--cons-k", type=int, default=3)
    parser.add_argument("--cons-max-new", type=int, default=40)
    parser.add_argument("--no-verb", action="store_true",
                        help="Skip the verbalized-confidence KV-fork probe.")
    parser.add_argument("--no-ent", action="store_true",
                        help="Skip the spaCy + BM25 entity-entropy scorer.")
    parser.add_argument("--no-cons", action="store_true",
                        help="Skip the K-sample consistency continuations.")
    parser.add_argument("--out-suffix", default="_sota")
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--inline-system", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else Path("models") / args.model
    if not model_dir.exists():
        print(f"[!!] missing {model_dir}.")
        sys.exit(1)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[!!] missing dataset {dataset_path}.")
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

    import torch  # noqa: F401

    tokenizer, model = qa.load_generator(model_dir, attn_implementation="eager")
    capture_subset = qa._resolve_layer_subset(model, args.layer_mode)
    dragin_layer_idx = _resolve_dragin_attn_layer(model, args.dragin_attn_layer_idx)
    print(f"[..] layer_mode={args.layer_mode}  capture_subset={capture_subset}")
    print(f"[..] dragin_attn_layer_idx={dragin_layer_idx} (override={args.dragin_attn_layer_idx})")
    if hasattr(model.config, "layer_types"):
        print(f"[..] layer_types[:8]={list(model.config.layer_types)[:8]} "
              f"... layer_types[-4:]={list(model.config.layer_types)[-4:]}")

    # Build the per-question probes once; entity scorer is rebuilt per question
    # because its BM25 index is over that question's 10 distractor paragraphs.
    probes: dict[str, Any] = {"dragin_layer_idx": dragin_layer_idx}
    if not args.no_verb:
        probes["verbal_probe"] = VerbalizedConfidenceProbe(model, tokenizer, max_probe_tokens=12)
        print(f"[..] verbalized confidence probe ready (max_probe_tokens=12)")
    if not args.no_cons:
        probes["consistency_sampler"] = ConsistencySampler(
            model, tokenizer,
            n_samples=args.cons_k,
            max_new_tokens=args.cons_max_new,
            temperature=0.6, top_p=0.9,
            seed=args.random_seed,
        )
        print(f"[..] consistency sampler ready (K={args.cons_k}, T=0.6, top_p=0.9)")
    if not args.no_ent:
        probes["entity_scorer_factory"] = lambda paragraphs: EntityEntropyScorer(paragraphs)
        # Warm spaCy + sentence-transformers caches early so any missing
        # dependency surfaces before the first per-question run.
        try:
            _ = EntityEntropyScorer([{"joined_text": "warmup"}])
            print(f"[..] entity entropy scorer ready (spaCy + rank_bm25)")
        except Exception as exc:
            print(f"[!!] entity scorer warmup failed ({exc}); disabling --no-ent")
            probes.pop("entity_scorer_factory")
            args.no_ent = True

    runs_dir = Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = runs_dir / f"phase4_qa_{args.model}_{args.layer_mode}{args.out_suffix}.jsonl"
    print(f"[..] writing {out_path}")

    n_total = len(questions)
    n_em_correct = 0
    t_total = time.perf_counter()
    with out_path.open("w") as f:
        for i, q in enumerate(questions):
            row = run_one_question_sota(q, tokenizer, model, capture_subset, probes, args)
            n_em_correct += int(row["em"])
            f.write(json.dumps(row) + "\n")
            f.flush()
            if (i + 1) % max(1, args.progress_every) == 0 or (i + 1) == n_total:
                elapsed = time.perf_counter() - t_total
                print(
                    f"[..] {i + 1:>3}/{n_total}  em={n_em_correct}/{i + 1}  "
                    f"steps={row.get('n_steps','-'):>3}  "
                    f"tok={row['tokens_generated']:>4}  "
                    f"wall={row['wall_clock_s']:.1f}s  "
                    f"elapsed={elapsed/60:.1f}min"
                )

    print(f"[ok] wrote {n_total} rows to {out_path}")
    print(f"[ok] em rate: {n_em_correct}/{n_total} = {n_em_correct/n_total:.3f}")


if __name__ == "__main__":
    main()
