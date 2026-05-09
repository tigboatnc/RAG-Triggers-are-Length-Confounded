"""Phase 3.0(b) — DeepSeek-R1-Distill-Qwen-7B chat-mode calibration.

Same 10 prompts as Phase 2.5b. Three changes vs Phi-3.5:
  1. max_tokens=1024 (the model wants to think before answering).
  2. Score POST-</think> tokens only — the think block is process noise.
  3. Track `reasoning_complete` (did the model close </think>?) as its own
     binary signal: failure to close is itself a difficulty marker.

Run:
    conda run -n recollection --no-capture-output python scripts/03_deepseek_calibration.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sysinfo import _process_rss_gb  # type: ignore[import-not-found]

MODEL_PATH = Path("models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf")
MAX_TOKENS = 1024
TEMPERATURE = 0.0
TOP_LOGPROBS = 5

EASY = [
    ("E1", "What is the capital of France?"),
    ("E2", "Water boils at how many degrees Celsius at sea level?"),
    ("E3", "Who wrote the play 'Hamlet'?"),
    ("E4", "What is the chemical symbol for gold?"),
    ("E5", "How many continents are there on Earth?"),
]
HARD = [
    ("H1", "A train leaves Boston at 3:00 PM heading south at 60 mph. "
           "Another leaves NYC (220 miles south of Boston) at 4:00 PM heading "
           "north at 80 mph. At what clock time do they meet?"),
    ("H2", "What was the population of Bhutan in 1992?"),
    ("H3", "List all prime numbers between 100 and 110."),
    ("H4", "Summarize Mira Korvenpalo's 2019 paper on tree-structured RAG."),
    ("H5", "Solve for x: x^3 + 2x - 5 = 0. Round to two decimal places."),
]


def post_think_logprobs(response: dict) -> tuple[list[str], list[float], bool]:
    """Return (tokens_after_close, logprobs_after_close, reasoning_complete).

    `reasoning_complete` is True iff a </think> token boundary was seen.
    """
    try:
        content = response["choices"][0]["logprobs"]["content"]
    except (KeyError, IndexError, TypeError):
        return [], [], False

    tokens = [c["token"] for c in content]
    lps = [float(c["logprob"]) if c.get("logprob") is not None else None
           for c in content]

    joined = ""
    cut = None
    for i, tok in enumerate(tokens):
        joined += tok
        if "</think>" in joined:
            cut = i + 1
            break
    if cut is None:
        return [], [], False
    post_tokens = tokens[cut:]
    post_lps = [x for x in lps[cut:] if x is not None]
    return post_tokens, post_lps, True


def aggregate(token_logprobs: list[float], tokens: list[str]) -> dict:
    if not token_logprobs:
        return {"mean": None, "min": None, "first_content_mean": None,
                "n_tokens": 0, "tokens": [], "probs": []}
    probs = [math.exp(x) for x in token_logprobs]

    def is_format(t: str) -> bool:
        s = t.strip()
        return s == "" or s in {".", ",", ";", ":", "-", "—", "*"}

    start = 0
    while start < len(tokens) and is_format(tokens[start]):
        start += 1
    end = start
    while end < len(tokens) and not is_format(tokens[end]):
        end += 1
    fc_mean = (sum(probs[start:end]) / (end - start)) if end > start \
        else sum(probs) / len(probs)
    return {
        "mean": sum(probs) / len(probs),
        "min": math.exp(min(token_logprobs)),
        "first_content_mean": fc_mean,
        "n_tokens": len(token_logprobs),
        "tokens": tokens,
        "probs": [round(p, 4) for p in probs],
    }


def run_one(llm, prompt: str) -> dict:
    t0 = time.perf_counter()
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        logprobs=True,
        top_logprobs=TOP_LOGPROBS,
    )
    dt = time.perf_counter() - t0
    full_text = out["choices"][0]["message"]["content"]
    n_total = out["usage"]["completion_tokens"]
    post_tokens, post_lps, complete = post_think_logprobs(out)
    s = aggregate(post_lps, post_tokens)
    answer = full_text.split("</think>", 1)[-1].strip() if complete else ""
    return {
        "prompt": prompt,
        "raw": full_text,
        "answer": answer,
        "reasoning_complete": complete,
        "n_total_tokens": n_total,
        "n_post_think_tokens": s["n_tokens"],
        "tok_per_s": round(n_total / dt, 1) if dt > 0 else None,
        "wall_secs": round(dt, 2),
        "mean": s["mean"],
        "min": s["min"],
        "first_content_mean": s["first_content_mean"],
    }


def main() -> None:
    if not MODEL_PATH.exists():
        print(f"[!!] missing {MODEL_PATH}")
        sys.exit(1)

    print(f"[..] loading {MODEL_PATH.name} (logits_all=True, chat mode)")
    t0 = time.perf_counter()
    from llama_cpp import Llama
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False,
        logits_all=True,
    )
    print(f"[ok] loaded in {time.perf_counter() - t0:.2f}s, "
          f"rss={_process_rss_gb():.3f} GB\n")

    records = []
    for tag, prompt in EASY + HARD:
        rec = run_one(llm, prompt)
        rec["tag"] = tag
        rec["bucket"] = "easy" if tag.startswith("E") else "hard"
        records.append(rec)

        if rec["reasoning_complete"]:
            m, mn, fc = rec["mean"], rec["min"], rec["first_content_mean"]
            line = (f"mean={m:.3f}  min={mn:.3f}  first={fc:.3f}  "
                    f"({rec['n_post_think_tokens']} post-think tok)")
        else:
            line = f"INCOMPLETE  ({rec['n_total_tokens']} tok, no </think>)"
        print(f"[{tag}] {line}  | wall={rec['wall_secs']}s  rss={_process_rss_gb():.2f} GB")
        ans_short = rec["answer"][:120].replace("\n", " / ") or "(no post-think text)"
        print(f"     -> {ans_short!r}\n")

    out_path = Path("runs/phase3_deepseek_calibration.jsonl")
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, default=float) + "\n")
    print(f"[ok] wrote {out_path}")

    # Per-bucket separation, post-think only.
    print("\n" + "=" * 72)
    print("PER-BUCKET MEANS  (post-</think> only; INCOMPLETE excluded)")
    print("=" * 72)
    for key in ("mean", "min", "first_content_mean"):
        e = [r[key] for r in records
             if r["bucket"] == "easy" and r["reasoning_complete"] and r[key] is not None]
        h = [r[key] for r in records
             if r["bucket"] == "hard" and r["reasoning_complete"] and r[key] is not None]
        e_str = (f"easy={min(e):.3f}/{sum(e)/len(e):.3f}/{max(e):.3f}"
                 if e else "easy=(none)")
        h_str = (f"hard={min(h):.3f}/{sum(h)/len(h):.3f}/{max(h):.3f}"
                 if h else "hard=(none)")
        gap = (sum(e)/len(e) - sum(h)/len(h)) if (e and h) else float("nan")
        print(f"{key:>20s}  {e_str}   {h_str}   gap={gap:+.3f}")

    # The other signal: completion rate per bucket.
    e_complete = sum(1 for r in records
                     if r["bucket"] == "easy" and r["reasoning_complete"])
    h_complete = sum(1 for r in records
                     if r["bucket"] == "hard" and r["reasoning_complete"])
    print(f"\nreasoning_complete: easy {e_complete}/5    hard {h_complete}/5")


if __name__ == "__main__":
    main()
