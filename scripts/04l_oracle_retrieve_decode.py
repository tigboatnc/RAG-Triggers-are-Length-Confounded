"""Phase 4 / #23 — Oracle-retrieve re-decode for downstream-QA utility evaluation.

Pre-registered design (commit this BEFORE running on the compute box):

  Goal:
    For every question in the HotpotQA-distractor 200 slice, produce a
    second model answer obtained by re-prompting the model with ONLY the
    supporting-fact paragraphs (distractors removed). This is the "oracle
    retrieval" condition. Identical decoding protocol to #20 / Option A
    (greedy, repetition_penalty per MODEL_PROTOCOL, early-stop on EOS or
    `Answer:` marker, `enable_thinking=False` for Qwen).

  Why this is the right experiment:
    Δfrac is a calibration proxy. The downstream-utility question is:
    "does triggering retrieval where the gate fires actually flip
    incorrect answers to correct?". HotpotQA-distractor lets us simulate
    this without a real retriever — the gold supporting paragraphs are
    already in the prompt; the oracle-retrieve condition simply hides
    the distractors. The closed-book EM/F1 (already on disk from #20) +
    the oracle-retrieve EM/F1 (this script's output) lets the offline
    scorer (04m) compute per-policy {EM, F1} for any gate.

  Outputs (one per JSONL line, mirroring 04_run_qa's schema for the
  fields we keep):
    id, split, model, oracle_retrieve=True,
    n_supporting_paragraphs, n_distractors_removed,
    model_answer_extracted, em, f1,
    tokens_generated, early_stop_reason,
    chat_template_kwargs_used.

  Notes on enable_thinking=False (Qwen):
    Empirically validated on HotpotQA in #20: 200 traces, all produced
    non-trivial reasoning chains (median n_steps ~4-5). The 2Wiki
    under-generation seen in the smoke for #21 was domain-specific.
    This script logs `tokens_generated` per trace so any collapse vs
    the closed-book run is detectable in the scorer.

  Usage (compute box, gzn env):

    python scripts/04l_oracle_retrieve_decode.py \\
        --model qwen3.6-35b-a3b-instruct \\
        --dataset data/hotpotqa_distractor_200.jsonl \\
        --out runs/phase4_qa_qwen3.6-35b-a3b-instruct_oracle_retrieve.jsonl

  Expected wall clock: ~30-60 min for 200 questions on a single A100.
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

# Reuse the canonical decode helpers from 04_run_qa so the protocol is
# byte-for-byte identical to #20 (modulo paragraph filtering).
rq = import_module("04_run_qa")  # type: ignore


def _filter_to_supporting(paragraphs: list[dict[str, Any]], supporting_facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep paragraphs whose title appears in supporting_facts. Preserve order."""
    sup_titles = {sf["title"] for sf in supporting_facts}
    kept = [p for p in paragraphs if p["title"] in sup_titles]
    return kept


def run_one_question(record: dict[str, Any], tokenizer: Any, model: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Lightweight decode loop — no CRV / gate / hidden-state capture.

    Mirrors 04_run_qa.run_one_question's decode path exactly (same chat
    template, same repetition penalty, same answer-marker early stop,
    same EOS stop, same max_new_tokens), but drops everything we don't
    need for plain answer extraction.
    """
    import torch

    proto = rq.MODEL_PROTOCOL.get(args.model, {"chat_template_kwargs": {}, "repetition_penalty": 1.0})
    chat_template_kwargs = proto["chat_template_kwargs"]
    repetition_penalty = float(proto["repetition_penalty"])

    paragraphs_full = record["paragraphs"]
    paragraphs_sup = _filter_to_supporting(paragraphs_full, record["supporting_facts"])
    n_sup = len(paragraphs_sup)
    n_removed = len(paragraphs_full) - n_sup

    if n_sup == 0:
        # Defensive — should not happen on HotpotQA but log and skip cleanly.
        return {
            "id": record["id"], "split": record.get("split", "unknown"),
            "oracle_retrieve": True, "error": "no_supporting_paragraphs",
            "n_supporting_paragraphs": 0,
            "n_distractors_removed": n_removed,
        }

    messages = rq._build_messages(record["question"], paragraphs_sup, inline_system=False)
    try:
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs
        )
    except TypeError:
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        chat_template_kwargs = {"_unsupported": dict(chat_template_kwargs)}
    encoded = tokenizer(chat_text, return_tensors="pt").to(model.device)
    prompt_tokens = int(encoded.input_ids.shape[-1])

    cache = None
    input_ids = encoded.input_ids
    attention_mask = encoded.attention_mask
    eos_id = tokenizer.eos_token_id

    generated_token_ids: list[int] = []
    early_stop_reason = "max_tokens"
    answer_marker_seen_at: int | None = None
    eos_seen_at: int | None = None

    t_start = time.perf_counter()
    for step in range(args.max_new_tokens):
        with torch.inference_mode():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask if step == 0 else None,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=False,
            )
        next_logits = out.logits[:, -1, :].clone()
        next_logits = rq._apply_repetition_penalty(next_logits, generated_token_ids, repetition_penalty)
        next_token = next_logits.argmax(dim=-1, keepdim=True)
        cache = out.past_key_values
        token_id = int(next_token.item())
        generated_token_ids.append(token_id)

        if eos_id is not None and token_id == eos_id:
            eos_seen_at = step + 1
            early_stop_reason = "eos"

        if (
            answer_marker_seen_at is None
            and (step + 1) % 16 == 0
            and len(generated_token_ids) >= 16
        ):
            tail_text = tokenizer.decode(generated_token_ids[-256:], skip_special_tokens=True)
            m = rq._ANSWER_MARKER_RE.search(tail_text)
            if m:
                answer_marker_seen_at = step + 1
                if early_stop_reason == "max_tokens":
                    early_stop_reason = "answer_marker"

        input_ids = next_token
        attention_mask = None
        torch.cuda.synchronize()

        if eos_seen_at is not None:
            break
        if answer_marker_seen_at is not None:
            break

    wall = time.perf_counter() - t_start
    tokens_generated = len(generated_token_ids)
    full_generation = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    extracted, extract_strategy = rq.extract_answer(full_generation)
    em = rq.exact_match(extracted, record["answer"])
    f1 = rq.token_f1(extracted, record["answer"])

    return {
        "id": record["id"],
        "split": record.get("split", "unknown"),
        "level": record.get("level", ""),
        "type": record.get("type", ""),
        "question": record["question"],
        "gold_answer": record["answer"],
        "model": args.model,
        "oracle_retrieve": True,
        "n_supporting_paragraphs": n_sup,
        "n_distractors_removed": n_removed,
        "prompt_tokens": prompt_tokens,
        "tokens_generated": tokens_generated,
        "wall_clock_s": round(wall, 3),
        "tok_per_s": round(tokens_generated / wall, 3) if wall > 0 else 0.0,
        "eos_seen_at": eos_seen_at,
        "answer_marker_seen_at": answer_marker_seen_at,
        "early_stop_reason": early_stop_reason,
        "chat_template_kwargs_used": chat_template_kwargs,
        "repetition_penalty_used": repetition_penalty,
        "model_answer_extracted": extracted,
        "model_answer_extract_strategy": extract_strategy,
        "model_answer_full_generation_tail": full_generation[-400:],
        "em": em,
        "f1": f1,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model-dir", default=None,
                   help="Path to the local HF model dir. Defaults to "
                        "`models/<model>` (the project convention used in "
                        "04_run_qa.py); pass explicitly if different.")
    p.add_argument("--dataset", default="data/hotpotqa_distractor_200.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--limit", type=int, default=None, help="dev knob; first N rows")
    p.add_argument("--attn-impl", default=None,
                   help="HF attn_implementation; default = HF auto (sdpa).")
    args = p.parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else Path("models") / args.model
    if not model_dir.exists():
        print(f"[err] model dir not found: {model_dir} — pass --model-dir explicitly.")
        return 1
    tokenizer, model = rq.load_generator(model_dir, attn_implementation=args.attn_impl)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_done = 0
    n_em = 0
    f1_sum = 0.0

    with Path(args.dataset).open() as f, out_path.open("w") as fout:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if args.limit is not None and n_done >= args.limit:
                break
            rec = json.loads(line)
            try:
                row = run_one_question(rec, tokenizer, model, args)
            except Exception as e:
                row = {
                    "id": rec.get("id"), "split": rec.get("split", "unknown"),
                    "oracle_retrieve": True, "error": f"{type(e).__name__}: {e}",
                }
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            n_done += 1
            if "em" in row:
                n_em += int(row["em"])
                f1_sum += float(row["f1"])
                print(f"[{n_done:>3}] id={rec['id']}  em={row['em']}  f1={row['f1']:.3f}  "
                      f"tok={row.get('tokens_generated')}  reason={row.get('early_stop_reason')}  "
                      f"n_sup={row.get('n_supporting_paragraphs')}/{row.get('n_supporting_paragraphs',0)+row.get('n_distractors_removed',0)}")
            else:
                print(f"[{n_done:>3}] id={rec['id']}  ERROR: {row.get('error')}")

    print()
    print(f"[ok] wrote {out_path}")
    print(f"     n={n_done}  EM={n_em}/{n_done} ({n_em/max(1,n_done)*100:.1f}%)  "
          f"meanF1={f1_sum/max(1,n_done):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
