"""Phase 4.1c — generic new-model protocol-smoke + _MODEL.md fact-collection.

Companion to ``scripts/04b_protocol_smoke.py``. Where 04b validates the
three pinned Phi/Gemma/Qwen3.6 archs in lockstep with the v2 PROTOCOL
dict, this script probes ANY new model (one at a time) and dumps a
JSON readout with every field needed to update ``_MODEL.md`` §1-§6.

Use case (RQ1 model-pool expansion 2026-05-13):
    python scripts/04c_new_model_smoke.py \
        --model qwen2.5-7b-instruct \
        --model-dir models/qwen2.5-7b-instruct \
        --thinking-mode off
    python scripts/04c_new_model_smoke.py \
        --model llama-2-7b-chat-hf \
        --model-dir models/llama-2-7b-chat-hf \
        --thinking-mode na

Output (one file per invocation):
    runs/04c_new_model_smoke_<model>_<UTCts>.json

Which contains:
    - §1 row data: HF repo (auto-derived from config), model_type, layer
      count, total params, hidden size, weights GB on disk, pinned SHA
      from models/<name>.revision if present.
    - §2 architecture: layer_types pattern (hybrid?), GQA Q/KV heads,
      max_position_embeddings, vocab_size, RoPE config, EOS / BOS
      special tokens. Anything we can read off the loaded model.config.
    - §3 chat template: tail of rendered Option-A messages, with and
      without each thinking-mode kwarg if applicable. Detection of OPEN
      thinking blocks (the v1 #20 trap).
    - §6 canonical numbers: load wall, VRAM allocated/reserved, decode
      tok/s on a 256-token greedy run with the requested rep penalty,
      EOS / answer-marker behaviour.
    - Behavioural sanity: a 5-question HotpotQA-pilot smoke if
      data/hotpotqa_distractor_25_pilot.jsonl is present (else skip).

The JSON is designed to be pasted into a chat with the agent verbatim;
the agent then writes the matching _MODEL.md sections without needing
to load the model itself.

Hard-fails are *only* for unrecoverable load errors. Behavioural
oddities (open thinking block, no Answer: marker, etc.) are reported
but not exit-failed — this is a fact-collection probe, not a gate.
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


SYSTEM_INSTRUCTION = (
    "You are a careful reasoner. Read the provided paragraphs, think step "
    "by step, and answer the question concisely. End your response with a "
    "single line of the form 'Answer: <your answer>' where <your answer> "
    "is as short as possible (typically 1-5 words)."
)

PROBE_USER = (
    "Question: What is the capital of France?\n\n"
    "Reason briefly, then end with 'Answer: ...'."
)

THINKING_OPENERS = [
    "<think>", "<|think|>", "<|channel>thought",
    "Here's a thinking process", "Here is a thinking process", "Let me think",
]

ANSWER_MARKER_RE = re.compile(r"\bAnswer\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _dir_size_gb(p: Path) -> float:
    total = 0
    for f in p.rglob("*"):
        if f.is_file() and f.suffix in (".safetensors", ".bin", ".pt", ".pth"):
            total += f.stat().st_size
    return round(total / (1024**3), 2)


def _read_revision(model_dir: Path) -> str | None:
    rev_file = model_dir.parent / f"{model_dir.name}.revision"
    if rev_file.exists():
        return rev_file.read_text().strip().split("\n")[0]
    cache = model_dir / ".cache"
    if cache.exists():
        for snap in (cache / "huggingface" / "download").rglob("*.metadata"):
            try:
                meta = json.loads(snap.read_text())
                if "etag" in meta:
                    return meta["etag"]
            except Exception:
                pass
    return None


def _load_one(model_dir: Path, dtype_name: str = "bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  [..] loading {model_dir} (dtype={dtype_name}, device_map={{'':0}})", flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    dtype = getattr(torch, dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        device_map={"": 0},
        dtype=dtype,
        attn_implementation="eager",  # match v2 capture pipeline (DRAGIN needs attentions)
    )
    model.eval()
    load_wall = round(time.perf_counter() - t0, 2)
    print(f"  [ok] loaded in {load_wall}s", flush=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return tok, model, load_wall


def _vram_mb() -> dict[str, int]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        return {
            "available": True,
            "allocated_mb": int(torch.cuda.memory_allocated() / (1024**2)),
            "reserved_mb": int(torch.cuda.memory_reserved() / (1024**2)),
            "max_allocated_mb": int(torch.cuda.max_memory_allocated() / (1024**2)),
            "device_name": torch.cuda.get_device_name(0),
            "device_total_mb": int(torch.cuda.get_device_properties(0).total_memory / (1024**2)),
        }
    except Exception as e:
        return {"available": False, "err": str(e)}


def _arch_summary(model) -> dict[str, Any]:
    cfg = model.config
    out: dict[str, Any] = {
        "model_type": getattr(cfg, "model_type", None),
        "architectures": list(getattr(cfg, "architectures", []) or []),
        "n_params_total": int(sum(p.numel() for p in model.parameters())),
        "n_params_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "hidden_size": getattr(cfg, "hidden_size", None) or getattr(getattr(cfg, "text_config", None), "hidden_size", None),
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None) or getattr(getattr(cfg, "text_config", None), "num_hidden_layers", None),
        "num_attention_heads": getattr(cfg, "num_attention_heads", None),
        "num_key_value_heads": getattr(cfg, "num_key_value_heads", None),
        "intermediate_size": getattr(cfg, "intermediate_size", None),
        "vocab_size": getattr(cfg, "vocab_size", None),
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
        "rope_theta": getattr(cfg, "rope_theta", None),
        "rope_scaling": getattr(cfg, "rope_scaling", None),
        "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", None),
        "torch_dtype": str(getattr(cfg, "torch_dtype", None)),
    }
    # Hybrid stack detection
    layer_types = getattr(cfg, "layer_types", None)
    if layer_types:
        from collections import Counter
        out["layer_types_summary"] = dict(Counter(layer_types))
        out["layer_types_first8"] = list(layer_types[:8])
        out["layer_types_last4"] = list(layer_types[-4:])
        out["is_hybrid"] = len(set(layer_types)) > 1
    else:
        out["layer_types_summary"] = None
        out["is_hybrid"] = False
    # MoE detection
    for moe_attr in ("num_experts", "num_local_experts", "num_routed_experts"):
        v = getattr(cfg, moe_attr, None)
        if v is not None:
            out[f"moe_{moe_attr}"] = int(v)
    return out


def _eos_specials(tok) -> dict[str, Any]:
    return {
        "eos_token": getattr(tok, "eos_token", None),
        "eos_token_id": int(tok.eos_token_id) if tok.eos_token_id is not None else None,
        "bos_token": getattr(tok, "bos_token", None),
        "bos_token_id": int(tok.bos_token_id) if tok.bos_token_id is not None else None,
        "pad_token": getattr(tok, "pad_token", None),
        "pad_token_id": int(tok.pad_token_id) if tok.pad_token_id is not None else None,
        "added_special_tokens": list(getattr(tok, "additional_special_tokens", []) or []),
        "vocab_size": tok.vocab_size,
    }


def _render_chat(tok, kwargs: dict[str, Any]) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": PROBE_USER},
    ]
    try:
        rendered = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
        used_kwargs = kwargs
        accepted = True
    except TypeError:
        rendered = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        used_kwargs = {}
        accepted = False

    last200 = rendered[-200:]
    open_blocks: list[str] = []
    def chk(opener, closer):
        idx = last200.rfind(opener)
        if idx < 0: return
        if closer in last200[idx + len(opener):]: return
        open_blocks.append(f"{opener!r} (no matching {closer!r} after)")
    chk("<think>", "</think>")
    chk("<|think|>", "<|/think|>")
    chk("<|channel>thought", "<channel|>")

    return {
        "kwargs_passed": kwargs,
        "kwargs_accepted": accepted,
        "kwargs_used_at_render_time": used_kwargs,
        "rendered_len_chars": len(rendered),
        "rendered_tail_300": rendered[-300:],
        "open_thinking_blocks": open_blocks,
        "has_open_thinking_block": bool(open_blocks),
    }


def _apply_repetition_penalty(logits, generated_ids, penalty: float):
    import torch
    if penalty == 1.0 or not generated_ids:
        return logits
    ids = torch.tensor(list(set(generated_ids)), device=logits.device, dtype=torch.long)
    selected = logits[0, ids]
    selected = torch.where(selected > 0, selected / penalty, selected * penalty)
    logits[0, ids] = selected
    return logits


def _greedy_decode(model, tok, prompt_text: str, max_new_tokens: int = 256, repetition_penalty: float = 1.0):
    import torch
    enc = tok(prompt_text, return_tensors="pt").to(model.device)
    input_ids = enc.input_ids
    attn_mask = enc.attention_mask
    cache = None
    eos_id = tok.eos_token_id
    generated: list[int] = []
    early = "max_tokens"
    answer_marker_seen = False

    t0 = time.perf_counter()
    for step in range(max_new_tokens):
        with torch.inference_mode():
            out = model(
                input_ids=input_ids,
                attention_mask=attn_mask if step == 0 else None,
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
            early = "eos"; break
        if step % 8 == 7:
            txt = tok.decode(generated, skip_special_tokens=True)
            m = ANSWER_MARKER_RE.search(txt)
            if m and "\n" in txt[m.start():]:
                early = "answer_marker"
                answer_marker_seen = True
                break
        input_ids = nxt
        attn_mask = None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    text = tok.decode(generated, skip_special_tokens=True)
    if not answer_marker_seen and ANSWER_MARKER_RE.search(text):
        answer_marker_seen = True

    head = text[:200]
    thinking_opener = next((o for o in THINKING_OPENERS if o in head), None)

    return {
        "tokens_generated": len(generated),
        "wall_s": round(wall, 2),
        "tok_per_s": round(len(generated) / wall, 2) if wall > 0 else 0.0,
        "early_stop": early,
        "answer_marker_seen": answer_marker_seen,
        "thinking_opener_in_head_200": thinking_opener,
        "text_first_400": text[:400],
        "text_last_200": text[-200:],
    }


def _hotpot_smoke(model, tok, repetition_penalty: float, max_new_tokens: int = 512) -> dict[str, Any] | None:
    """Optional small HotpotQA-distractor smoke if the pilot slice is on disk."""
    slice_path = Path("data/hotpotqa_distractor_25_pilot.jsonl")
    if not slice_path.exists():
        return None
    rows = []
    with slice_path.open() as f:
        for i, line in enumerate(f):
            if i >= 3: break
            rows.append(json.loads(line))
    out = []
    for r in rows:
        para_block = "\n\n".join(
            f"[Paragraph {p['paragraph_idx']}: {p['title']}]\n{p['joined_text']}"
            for p in r["paragraphs"][:5]
        )
        user = f"Paragraphs:\n{para_block}\n\nQuestion: {r['question']}\n\nReason step by step, then end with 'Answer: ...'."
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user},
        ]
        try:
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        d = _greedy_decode(model, tok, prompt, max_new_tokens=max_new_tokens,
                           repetition_penalty=repetition_penalty)
        d["q_id"] = r["id"]
        d["gold_answer"] = r["answer"]
        out.append(d)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="logical name (e.g. qwen2.5-7b-instruct)")
    p.add_argument("--model-dir", required=True, help="local path with weights")
    p.add_argument("--thinking-mode", choices=["off", "on", "na"], default="na",
                   help="off => pass enable_thinking=False; on => pass True; na => no kwargs")
    p.add_argument("--repetition-penalty", type=float, default=1.0,
                   help="proposed Option-A rep penalty for this model")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--skip-hotpot", action="store_true")
    p.add_argument("--out-dir", default="runs")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists() or not any(model_dir.glob("*.safetensors")):
        print(f"[!!] missing weights at {model_dir}/*.safetensors", file=sys.stderr)
        return 2

    tok, model, load_wall = _load_one(model_dir, dtype_name=args.dtype)
    vram = _vram_mb()
    arch = _arch_summary(model)
    eos = _eos_specials(tok)

    print(f"\n--- §3 chat template render (kwargs={args.thinking_mode}) ---")
    if args.thinking_mode == "off":
        chat = _render_chat(tok, {"enable_thinking": False})
    elif args.thinking_mode == "on":
        chat = _render_chat(tok, {"enable_thinking": True})
    else:
        chat = _render_chat(tok, {})
    print(f"  rendered_tail_300 = {chat['rendered_tail_300']!r}")
    print(f"  open_thinking_blocks = {chat['open_thinking_blocks']}")

    print(f"\n--- §6 canonical decode ({args.max_new_tokens} tok greedy, rep={args.repetition_penalty}) ---")
    rendered_for_probe = chat["rendered_tail_300"]  # not the prompt; rebuild full prompt:
    full_prompt = tok.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": PROBE_USER},
        ],
        tokenize=False, add_generation_prompt=True,
        **(chat["kwargs_used_at_render_time"]),
    )
    decode = _greedy_decode(model, tok, full_prompt,
                            max_new_tokens=args.max_new_tokens,
                            repetition_penalty=args.repetition_penalty)
    print(f"  tokens_generated={decode['tokens_generated']} early_stop={decode['early_stop']} "
          f"tok/s={decode['tok_per_s']} answer_marker_seen={decode['answer_marker_seen']}")

    hotpot = None
    if not args.skip_hotpot:
        print(f"\n--- behavioural smoke: 3 questions from data/hotpotqa_distractor_25_pilot.jsonl ---")
        hotpot = _hotpot_smoke(model, tok, args.repetition_penalty)
        if hotpot is None:
            print("  (slice not on disk; skipped)")
        else:
            for h in hotpot:
                print(f"  q={h['q_id'][:12]} gold='{h['gold_answer']}' "
                      f"tok={h['tokens_generated']} stop={h['early_stop']} "
                      f"answer_marker={h['answer_marker_seen']}")

    out = {
        "ts_utc": _ts(),
        "model_name": args.model,
        "model_dir": str(model_dir),
        "weights_size_gb": _dir_size_gb(model_dir),
        "pinned_revision": _read_revision(model_dir),
        "load_wall_s": load_wall,
        "vram": vram,
        "section_1_summary": {
            "model_type": arch["model_type"],
            "n_params_total_M": round(arch["n_params_total"] / 1e6, 1),
            "weights_gb_on_disk": _dir_size_gb(model_dir),
        },
        "section_2_architecture": arch,
        "section_3_chat_template": chat,
        "section_5_eos_specials": eos,
        "section_6_decode_probe": {
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty,
            "result": decode,
        },
        "behavioural_hotpot_smoke": hotpot,
        "_args": vars(args),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"04c_new_model_smoke_{args.model}_{_ts()}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[ok] wrote {out_path}")
    print(f"[ok] paste this file's contents into the agent chat to get _MODEL.md updates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
