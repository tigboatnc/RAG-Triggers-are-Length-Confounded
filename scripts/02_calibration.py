"""Phase 2.4 — confidence calibration on hand-picked prompts.

Five "easy" prompts the model should answer confidently and correctly.
Five "hard" prompts (multi-hop, obscure-fact, arithmetic, list-recall)
where we expect either uncertainty or hallucination.

We score each prompt with all three aggregators from confidence.py and
print a markdown table. We also write runs/phase2_calibration.jsonl with
the full per-prompt records for later inspection.

Run:
    conda run -n recollection --no-capture-output python scripts/02_calibration.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from confidence import summarize  # type: ignore[import-not-found]
from sysinfo import _process_rss_gb  # type: ignore[import-not-found]

MODEL_PATH = Path("models/Phi-3.5-mini-instruct-Q4_K_M.gguf")
MAX_TOKENS = 40
TEMPERATURE = 0.0  # deterministic so reruns are comparable
TOP_LOGPROBS = 5

EASY = [
    ("E1", "The capital of France is"),
    ("E2", "Water boils at how many degrees Celsius at sea level? Answer:"),
    ("E3", "Who wrote the play 'Hamlet'? Answer:"),
    ("E4", "The chemical symbol for gold is"),
    ("E5", "How many continents are there on Earth? Answer:"),
]

HARD = [
    # multi-hop arithmetic
    ("H1", "A train leaves Boston at 3:00 PM heading south at 60 mph. "
           "Another leaves NYC (220 miles south of Boston) at 4:00 PM heading "
           "north at 80 mph. At what clock time do they meet? Answer:"),
    # obscure / niche fact, likely hallucinated
    ("H2", "What was the population of Bhutan in 1992? Answer:"),
    # short combinatorics requiring exact recall
    ("H3", "List all prime numbers between 100 and 110. Answer:"),
    # OOD synthesis (we deliberately invent a name)
    ("H4", "Summarize Mira Korvenpalo's 2019 paper on tree-structured RAG. Answer:"),
    # numerical reasoning
    ("H5", "Solve for x: x^3 + 2x - 5 = 0. Round to two decimal places. Answer:"),
]


def run_one(llm, prompt: str) -> dict:
    t0 = time.perf_counter()
    out = llm(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        logprobs=TOP_LOGPROBS,
        echo=False,
    )
    dt = time.perf_counter() - t0
    s = summarize(out)
    return {
        "prompt": prompt,
        "answer": out["choices"][0]["text"].strip(),
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

    print(f"[..] loading {MODEL_PATH.name} (logits_all=True)")
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
        print(f"[{tag}] mean={rec['mean']:.3f}  min={rec['min']:.3f}  "
              f"first={rec['first_content_mean']:.3f}  | "
              f"{rec['n_tokens']} tok @ {rec['tok_per_s']} tok/s")
        print(f"     -> {rec['answer'][:120].replace(chr(10), ' / ')!r}\n")

    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    out_path = runs_dir / "phase2_calibration.jsonl"
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, default=float) + "\n")
    print(f"[ok] wrote {out_path}")

    # Markdown summary table.
    print("\n" + "=" * 72)
    print("CALIBRATION TABLE")
    print("=" * 72)
    print(f"{'tag':<4}  {'bucket':<5}  {'mean':>6}  {'min':>6}  "
          f"{'first':>6}  prompt")
    for rec in records:
        prompt_short = rec["prompt"][:50].replace("\n", " ")
        print(f"{rec['tag']:<4}  {rec['bucket']:<5}  "
              f"{rec['mean']:>6.3f}  {rec['min']:>6.3f}  "
              f"{rec['first_content_mean']:>6.3f}  {prompt_short!r}")

    # Aggregate per bucket so we can see separation.
    def bucket_stats(records, bucket, key):
        vals = [r[key] for r in records if r["bucket"] == bucket]
        return min(vals), sum(vals) / len(vals), max(vals)

    print("\n" + "=" * 72)
    print(f"PER-BUCKET MEANS  (lower-mid-upper)")
    print("=" * 72)
    for key in ("mean", "min", "first_content_mean"):
        e_lo, e_mid, e_hi = bucket_stats(records, "easy", key)
        h_lo, h_mid, h_hi = bucket_stats(records, "hard", key)
        sep = e_mid - h_mid
        print(f"{key:>20s}  easy={e_lo:.3f}/{e_mid:.3f}/{e_hi:.3f}   "
              f"hard={h_lo:.3f}/{h_mid:.3f}/{h_hi:.3f}   "
              f"easy-hard gap={sep:+.3f}")


if __name__ == "__main__":
    main()
