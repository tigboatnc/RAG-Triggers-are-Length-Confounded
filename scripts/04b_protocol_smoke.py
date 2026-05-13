"""Phase 4.1b protocol smoke — verify each generator behaves under Option A.

Why this exists: model cards for the three Phase 1.4 generators were published
in 2026, after this assistant's training cutoff. We cannot trust memorised
defaults; we read the live model card, derive the Option A protocol, and then
*verify on the actual on-disk weights* that the chat-template render and
short-generation behaviour match expectations. Run on every new compute box
before the Phase 4.1b sweep.

Three probes per model:

1. Chat-template render check — apply the Option A `messages` array, print
   the last 300 chars of the rendered prompt. Verify thinking-mode prefill
   tokens (or their absence) match the per-model contract.

2. Default-mode short generation — 1-paragraph factual question, 256 tokens
   greedy. Detect: (a) thinking-mode opener `<think>` / `<channel>thought`,
   (b) emission of `Answer:` marker, (c) trivial repetition.

3. Multi-hop generation — one real HotpotQA calibration question, full
   Option A prompt, 1024 tokens, greedy + per-model `repetition_penalty`.
   Detect the same plus EM / F1 against gold.

Output: human-readable stdout + machine-readable
`runs/04b_protocol_smoke_<timestamp>.json`. Exits non-zero if any model
deviates from its declared mode (thinking opener detected when expected off,
or no `Answer:` marker on the multi-hop probe).

Usage:
  python scripts/04b_protocol_smoke.py
  python scripts/04b_protocol_smoke.py --models phi-4-mini-instruct
  python scripts/04b_protocol_smoke.py --skip-multihop   # for quick template-only check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


# Per-model Option A protocol. Keep this in lockstep with `MODEL_PROTOCOL`
# in `04_run_qa.py` — the smoke is meaningless if the two diverge.
PROTOCOL = {
    "phi-4-mini-instruct": {
        "chat_template_kwargs": {},
        "repetition_penalty": 1.05,
        "thinking_expected": False,
        "card_url": "https://huggingface.co/microsoft/Phi-4-mini-instruct",
    },
    "gemma-4-26b-a4b-it": {
        "chat_template_kwargs": {},
        "repetition_penalty": 1.0,
        "thinking_expected": False,
        "card_url": "https://huggingface.co/google/gemma-4-26B-A4B-it",
    },
    "qwen3.6-35b-a3b-instruct": {
        "chat_template_kwargs": {"enable_thinking": False},
        "repetition_penalty": 1.0,
        "thinking_expected": False,
        "card_url": "https://huggingface.co/Qwen/Qwen3.6-35B-A3B",
    },
    "qwen2.5-7b-instruct": {
        # No thinking mode in Qwen 2.5; standard ChatML; apache-2.0.
        # Added 2026-05-13 as RQ1 reference backbone (used by SR-RAG,
        # DyPRAG family). Smoke 04c_new_model_smoke confirmed clean
        # Option-A behaviour.
        "chat_template_kwargs": {},
        "repetition_penalty": 1.0,
        "thinking_expected": False,
        "card_url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
    },
}


SYSTEM_INSTRUCTION = (
    "You are a careful reasoner. Read the provided paragraphs, think step "
    "by step, and answer the question concisely. End your response with a "
    "single line of the form 'Answer: <your answer>' where <your answer> "
    "is as short as possible (typically 1-5 words)."
)

PROBE1_USER = (
    "Question: What is the capital of France?\n\n"
    "Reason briefly, then end with 'Answer: ...'."
)

# Tokens that, if generated as the FIRST few tokens, indicate thinking-mode is on.
# We check the decoded text, not token ids, so it's tokenizer-agnostic.
THINKING_OPENERS = [
    "<think>",
    "<|think|>",
    "<|channel>thought",
    "Here's a thinking process",
    "Here is a thinking process",
    "Let me think",
]

ANSWER_MARKER_RE = re.compile(r"\bAnswer\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)


def _load_one(model_dir: Path):
    """Bare-bones loader. Mirrors `load_generator` in 04_run_qa.py."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  [..] loading {model_dir} (bf16, device_map={{'': 0}})")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model.eval()
    dt = time.perf_counter() - t0
    print(f"  [ok] loaded in {dt:.1f}s")
    return tok, model


def _apply_repetition_penalty(logits, generated_ids, penalty: float):
    """HF-standard CTRL repetition penalty applied to a single-step logits row."""
    import torch

    if penalty == 1.0 or not generated_ids:
        return logits
    ids = torch.tensor(list(set(generated_ids)), device=logits.device, dtype=torch.long)
    selected = logits[0, ids]
    selected = torch.where(selected > 0, selected / penalty, selected * penalty)
    logits[0, ids] = selected
    return logits


def _greedy_decode(
    model,
    tok,
    prompt_text: str,
    max_new_tokens: int,
    repetition_penalty: float = 1.0,
    stop_on_answer_line: bool = True,
):
    """Single-trace greedy decode with optional repetition penalty + answer-line stop."""
    import torch

    enc = tok(prompt_text, return_tensors="pt").to(model.device)
    input_ids = enc.input_ids
    attention_mask = enc.attention_mask
    cache = None
    eos_id = tok.eos_token_id
    generated: list[int] = []
    early_stop = "max_tokens"

    decoded_so_far = ""
    answer_marker_seen = False

    t0 = time.perf_counter()
    for step in range(max_new_tokens):
        with torch.inference_mode():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask if step == 0 else None,
                past_key_values=cache,
                use_cache=True,
            )
        logits = out.logits[:, -1, :].clone()
        cache = out.past_key_values
        logits = _apply_repetition_penalty(logits, generated, repetition_penalty)
        nxt = logits.argmax(dim=-1, keepdim=True)
        tid = int(nxt.item())
        generated.append(tid)

        if eos_id is not None and tid == eos_id:
            early_stop = "eos"
            break

        if stop_on_answer_line and step % 8 == 7:
            decoded_so_far = tok.decode(generated, skip_special_tokens=True)
            m = ANSWER_MARKER_RE.search(decoded_so_far)
            if m and "\n" in decoded_so_far[m.start():]:
                early_stop = "answer_marker"
                answer_marker_seen = True
                break

        input_ids = nxt
        attention_mask = None

    wall = time.perf_counter() - t0
    text = tok.decode(generated, skip_special_tokens=True)
    if not answer_marker_seen and ANSWER_MARKER_RE.search(text):
        answer_marker_seen = True
    return {
        "tokens_generated": len(generated),
        "wall_s": round(wall, 2),
        "tok_per_s": round(len(generated) / wall, 2) if wall > 0 else 0.0,
        "early_stop": early_stop,
        "text": text,
        "answer_marker_seen": answer_marker_seen,
    }


def _detect_thinking(text: str, lookback: int = 200) -> str | None:
    head = text[:lookback]
    for opener in THINKING_OPENERS:
        if opener in head:
            return opener
    return None


def _detect_repetition(text: str, n: int = 30, copies: int = 3) -> str | None:
    """Naive: any 30-char substring appearing >= 3 times back-to-back."""
    if len(text) < n * copies:
        return None
    for i in range(len(text) - n * copies):
        snippet = text[i : i + n]
        if snippet * copies in text:
            return snippet
    return None


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _em_f1(pred: str, gold: str) -> tuple[int, float]:
    np_, ng = _normalize(pred), _normalize(gold)
    em = int(np_ == ng)
    pred_toks, gold_toks = np_.split(), ng.split()
    if not pred_toks or not gold_toks:
        return em, 0.0
    common = set(pred_toks) & set(gold_toks)
    if not common:
        return em, 0.0
    p = sum(min(pred_toks.count(t), gold_toks.count(t)) for t in common) / len(pred_toks)
    r = sum(min(pred_toks.count(t), gold_toks.count(t)) for t in common) / len(gold_toks)
    return em, (2 * p * r) / (p + r)


def _format_paragraphs(paragraphs: list[dict[str, Any]]) -> str:
    parts = []
    for p in paragraphs:
        body = p["joined_text"].strip()
        parts.append(f"[Paragraph {p['paragraph_idx']}: {p['title']}]\n{body}")
    return "\n\n".join(parts)


def _option_a_messages(user_payload: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_payload},
    ]


def probe_one_model(model_name: str, multihop_record: dict | None) -> dict:
    proto = PROTOCOL[model_name]
    model_dir = Path("models") / model_name
    if not model_dir.exists():
        return {
            "model": model_name,
            "error": f"model dir missing: {model_dir}",
        }

    print(f"\n{'='*72}\n[probe] {model_name} (card: {proto['card_url']})\n{'='*72}")
    tok, model = _load_one(model_dir)
    result: dict[str, Any] = {
        "model": model_name,
        "protocol": {
            "chat_template_kwargs": proto["chat_template_kwargs"],
            "repetition_penalty": proto["repetition_penalty"],
            "thinking_expected": proto["thinking_expected"],
            "card_url": proto["card_url"],
        },
        "fail_reasons": [],
    }

    print("\n--- probe 1: chat-template render ---")
    try:
        rendered = tok.apply_chat_template(
            _option_a_messages(PROBE1_USER),
            tokenize=False,
            add_generation_prompt=True,
            **proto["chat_template_kwargs"],
        )
    except TypeError as e:
        result["fail_reasons"].append(f"chat_template_kwargs not accepted: {e}")
        rendered = tok.apply_chat_template(
            _option_a_messages(PROBE1_USER), tokenize=False, add_generation_prompt=True
        )
    tail = rendered[-300:]

    # Distinguish OPEN vs CLOSED thinking-block prefills.
    #   - Qwen `enable_thinking=False`: prefills `<think>\n\n</think>\n\n` (closed).
    #   - Qwen default: prefills `<think>\n` only (OPEN -> model continues inside).
    #   - Gemma no-thinking: prefills `<|channel>thought\n<channel|>` (closed).
    #   - Gemma `enable_thinking=True`: prefills `<|think|>` in system + open thought channel.
    # We grep the last 200 chars for each opener and check whether its matching
    # closer appears AFTER it. Open => bad under Option A; closed => good.
    last200 = rendered[-200:]
    open_blocks: list[str] = []

    def _check(opener: str, closer: str):
        idx = last200.rfind(opener)
        if idx < 0:
            return
        if closer in last200[idx + len(opener):]:
            return  # closed before prompt ends -> empty/closed thinking block
        open_blocks.append(f"{opener!r} (no matching {closer!r} after it)")

    _check("<think>", "</think>")
    _check("<|think|>", "<|/think|>")
    _check("<|channel>thought", "<channel|>")

    has_open_thinking = bool(open_blocks)
    print(f"  rendered_len={len(rendered)} chars; tail (last 300):")
    print(f"  {tail!r}")
    print(f"  open_thinking_blocks={open_blocks}  expected_off={not proto['thinking_expected']}")
    result["probe1_chat_template"] = {
        "rendered_len": len(rendered),
        "rendered_tail_300": tail,
        "open_thinking_blocks": open_blocks,
    }
    if has_open_thinking and not proto["thinking_expected"]:
        result["fail_reasons"].append(
            f"open thinking-mode prefill in rendered prompt under Option A: {open_blocks}"
        )

    print("\n--- probe 2: short factual generation (256 tok greedy) ---")
    g = _greedy_decode(model, tok, rendered, max_new_tokens=256,
                       repetition_penalty=proto["repetition_penalty"])
    opener = _detect_thinking(g["text"])
    rep = _detect_repetition(g["text"])
    paris_in = "Paris" in g["text"]
    print(f"  tok={g['tokens_generated']}  early_stop={g['early_stop']}  tok/s={g['tok_per_s']}")
    print(f"  thinking_opener_first200chars={opener!r}")
    print(f"  repetition_substring={rep!r}")
    print(f"  contains 'Paris'={paris_in}  answer_marker={g['answer_marker_seen']}")
    print(f"  generation (first 500 chars): {g['text'][:500]!r}")
    result["probe2_short_factual"] = {
        **g,
        "thinking_opener_seen": opener,
        "repetition_substring": rep,
        "contains_paris": paris_in,
    }
    if opener and not proto["thinking_expected"]:
        result["fail_reasons"].append(
            f"unexpected thinking opener in short generation: {opener!r}"
        )
    if rep:
        result["fail_reasons"].append(f"repetition detected: {rep!r}")
    if not paris_in:
        result["fail_reasons"].append("'Paris' not found in capital-of-France answer")
    if not g["answer_marker_seen"] and g["early_stop"] != "eos":
        result["fail_reasons"].append("no Answer: marker emitted on short factual probe")

    if multihop_record is not None:
        print("\n--- probe 3: real HotpotQA multi-hop (1024 tok) ---")
        user_payload = (
            f"Paragraphs:\n\n{_format_paragraphs(multihop_record['paragraphs'])}\n\n"
            f"Question: {multihop_record['question']}\n\n"
            f"Reason step by step, then end with 'Answer: ...'."
        )
        rendered_mh = tok.apply_chat_template(
            _option_a_messages(user_payload),
            tokenize=False,
            add_generation_prompt=True,
            **proto["chat_template_kwargs"],
        )
        print(f"  prompt tokens (decoded len): {len(rendered_mh)} chars; question: {multihop_record['question'][:80]!r}")
        g3 = _greedy_decode(model, tok, rendered_mh, max_new_tokens=1024,
                            repetition_penalty=proto["repetition_penalty"])
        m = ANSWER_MARKER_RE.search(g3["text"])
        pred = m.group(1).strip() if m else g3["text"].splitlines()[-1].strip()[:80]
        em, f1 = _em_f1(pred, multihop_record["answer"])
        opener3 = _detect_thinking(g3["text"])
        rep3 = _detect_repetition(g3["text"])
        print(f"  tok={g3['tokens_generated']}  early_stop={g3['early_stop']}  tok/s={g3['tok_per_s']}")
        print(f"  gold={multihop_record['answer']!r}  pred={pred!r}  EM={em}  F1={f1:.3f}")
        print(f"  thinking_opener={opener3!r}  repetition={rep3!r}")
        result["probe3_multihop"] = {
            "question": multihop_record["question"],
            "gold": multihop_record["answer"],
            "pred": pred,
            "em": em,
            "f1": round(f1, 4),
            **g3,
            "thinking_opener_seen": opener3,
            "repetition_substring": rep3,
        }
        if opener3 and not proto["thinking_expected"]:
            result["fail_reasons"].append(
                f"unexpected thinking opener in multi-hop: {opener3!r}"
            )
        if rep3:
            result["fail_reasons"].append(f"multi-hop repetition: {rep3!r}")
        if not g3["answer_marker_seen"]:
            result["fail_reasons"].append("multi-hop probe did not emit Answer: marker")

    import torch
    del model
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(PROTOCOL.keys()))
    parser.add_argument("--skip-multihop", action="store_true")
    parser.add_argument("--dataset", default="data/hotpotqa_distractor_200.jsonl")
    parser.add_argument("--out-dir", default="runs")
    args = parser.parse_args()

    multihop_record = None
    if not args.skip_multihop:
        ds = Path(args.dataset)
        if ds.exists():
            with ds.open() as f:
                for line in f:
                    row = json.loads(line)
                    if row.get("split") == "calibration":
                        multihop_record = row
                        break
            if multihop_record is None:
                print(f"[!!] no calibration-split row in {ds}; skipping multi-hop probe")
        else:
            print(f"[!!] dataset {ds} missing; skipping multi-hop probe (rerun scripts/04_dataset.py)")

    results = []
    for m in args.models:
        if m not in PROTOCOL:
            print(f"[!!] unknown model {m}; allowed: {list(PROTOCOL.keys())}")
            continue
        results.append(probe_one_model(m, multihop_record))

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.out_dir) / f"04b_protocol_smoke_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": ts,
        "option": "A",
        "protocol_used": PROTOCOL,
        "results": results,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*72}\n[summary] wrote {out_path}\n{'='*72}")
    n_fail = 0
    for r in results:
        fr = r.get("fail_reasons", [])
        status = "OK " if not fr else "FAIL"
        if fr:
            n_fail += 1
        print(f"  [{status}] {r['model']:32}  fail_reasons={fr}")
    if n_fail:
        print(f"\n[!!] {n_fail}/{len(results)} models FAILED protocol smoke. Do not run 4.1b sweep until fixed.")
        return 2
    print(f"\n[ok] all {len(results)} models pass Option A protocol smoke. Safe to run 4.1b sweep.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
