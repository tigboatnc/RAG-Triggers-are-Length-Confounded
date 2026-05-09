"""Phase 2.5b — same 10 prompts, but via chat-completion (proper template).

Hypothesis: Phi-3.5's "Question:/Answer:" drift comes from us using base
text completion. `create_chat_completion` applies the model's chat template
(<|user|>...<|end|><|assistant|>...), which should make the model respond
instead of autocomplete.

If true:
  - format-drift tokens vanish from the output
  - `mean` and `min` should stop being inverted
  - the easy/hard signal should appear in *all three* aggregators

Run:
    conda run -n recollection --no-capture-output python scripts/02b_chat_calibration.py
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

MODEL_PATH = Path("models/Phi-3.5-mini-instruct-Q4_K_M.gguf")
MAX_TOKENS = 40
TEMPERATURE = 0.0
TOP_LOGPROBS = 5

# Same prompts as 02_calibration.py (kept terse; chat mode adds the system
# scaffolding for us, so no need for "Answer:" suffixes).
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


def aggregate(token_logprobs: list[float], tokens: list[str]) -> dict:
    """Mean / min / first_content_mean over chosen-token logprobs."""
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


def extract_logprobs(response: dict) -> tuple[list[float], list[str]]:
    """Chat-completion logprobs schema (OpenAI-style):
        choices[0].logprobs.content = [
            {token: str, logprob: float, top_logprobs: [...]},
            ...
        ]
    """
    try:
        content = response["choices"][0]["logprobs"]["content"]
    except (KeyError, IndexError, TypeError):
        return [], []
    tokens = [c["token"] for c in content]
    lps = [float(c["logprob"]) for c in content if c.get("logprob") is not None]
    return lps, tokens


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
    lps, tokens = extract_logprobs(out)
    s = aggregate(lps, tokens)
    answer = out["choices"][0]["message"]["content"].strip()
    return {
        "prompt": prompt,
        "answer": answer,
        "n_tokens": s["n_tokens"],
        "tok_per_s": round(s["n_tokens"] / dt, 1) if dt > 0 else None,
        "mean": s["mean"],
        "min": s["min"],
        "first_content_mean": s["first_content_mean"],
        "tokens": s["tokens"],
        "probs": s["probs"],
    }


def main() -> None:
    if not MODEL_PATH.exists():
        print(f"[!!] missing {MODEL_PATH}")
        sys.exit(1)

    print(f"[..] loading {MODEL_PATH.name} (logits_all=True, chat mode)")
    from llama_cpp import Llama
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False,
        logits_all=True,
    )
    print(f"[ok] loaded, rss={_process_rss_gb():.3f} GB\n")

    records = []
    for tag, prompt in EASY + HARD:
        rec = run_one(llm, prompt)
        rec["tag"] = tag
        rec["bucket"] = "easy" if tag.startswith("E") else "hard"
        records.append(rec)
        m = rec["mean"]; mn = rec["min"]; fc = rec["first_content_mean"]
        print(f"[{tag}] mean={m if m is None else f'{m:.3f}'}  "
              f"min={mn if mn is None else f'{mn:.3f}'}  "
              f"first={fc if fc is None else f'{fc:.3f}'}  | "
              f"{rec['n_tokens']} tok @ {rec['tok_per_s']} tok/s")
        print(f"     -> {rec['answer'][:120].replace(chr(10), ' / ')!r}\n")

    out_path = Path("runs/phase2_chat_calibration.jsonl")
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, default=float) + "\n")
    print(f"[ok] wrote {out_path}")

    print("\n" + "=" * 72)
    print(f"PER-BUCKET MEANS  (lower-mid-upper)   [chat mode]")
    print("=" * 72)
    for key in ("mean", "min", "first_content_mean"):
        e = [r[key] for r in records if r["bucket"] == "easy" and r[key] is not None]
        h = [r[key] for r in records if r["bucket"] == "hard" and r[key] is not None]
        if not e or not h:
            print(f"{key:>20s}  (n/a)")
            continue
        e_mid = sum(e) / len(e)
        h_mid = sum(h) / len(h)
        print(f"{key:>20s}  easy={min(e):.3f}/{e_mid:.3f}/{max(e):.3f}   "
              f"hard={min(h):.3f}/{h_mid:.3f}/{max(h):.3f}   "
              f"easy-hard gap={e_mid - h_mid:+.3f}")


if __name__ == "__main__":
    main()
