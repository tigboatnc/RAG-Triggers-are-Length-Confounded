"""Phase 3.0 step 1 — inspect DeepSeek-R1-Distill-Qwen-7B's chat output.

Goals (no calibration yet):
  1. Confirm the model loads on Metal and answers something.
  2. See whether/how it emits <think>...</think> reasoning blocks.
  3. Decide how confidence should be scored: whole output, post-</think> only,
     think-block separately?

Run:
    conda run -n recollection --no-capture-output python scripts/03_deepseek_inspect.py
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from confidence import summarize  # type: ignore[import-not-found]
from sysinfo import _process_rss_gb  # type: ignore[import-not-found]

MODEL_PATH = Path("models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf")

PROMPTS = [
    ("E1_easy", "What is the capital of France?"),
    ("H5_hard_math", "Solve for x: x^3 + 2x - 5 = 0. Round to two decimal places."),
]


def split_think(text: str) -> tuple[str, str]:
    """Split text into (think_block, post_think_answer)."""
    m = re.search(r"<think>(.*?)</think>(.*)$", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    if "<think>" in text and "</think>" not in text:
        return text.split("<think>", 1)[1].strip(), ""
    return "", text.strip()


def main() -> None:
    if not MODEL_PATH.exists():
        print(f"[!!] missing {MODEL_PATH}")
        sys.exit(1)

    print(f"[..] loading {MODEL_PATH.name}  (logits_all=True)")
    t0 = time.perf_counter()
    from llama_cpp import Llama
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False,
        logits_all=True,
    )
    load_secs = time.perf_counter() - t0
    rss = _process_rss_gb()
    print(f"[ok] loaded in {load_secs:.2f}s, rss={rss:.3f} GB\n")

    for tag, prompt in PROMPTS:
        print("=" * 72)
        print(f"{tag}: {prompt}")
        print("=" * 72)
        t0 = time.perf_counter()
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,           # plenty of room for a think block
            temperature=0.0,
            logprobs=True,
            top_logprobs=5,
        )
        dt = time.perf_counter() - t0
        text = out["choices"][0]["message"]["content"]
        n_tokens = out["usage"]["completion_tokens"]
        print(f"[gen] {n_tokens} tok in {dt:.2f}s = {n_tokens / dt:.1f} tok/s\n")

        think, answer = split_think(text)
        print(f"--- raw output ({len(text)} chars) ---")
        print(text)
        print()
        print(f"--- think block ({len(think)} chars) ---")
        print(think[:600] + ("..." if len(think) > 600 else ""))
        print()
        print(f"--- post-think answer ({len(answer)} chars) ---")
        print(answer)
        print()

        # Confidence over the whole generation:
        whole = summarize(out)
        print(f"[whole] mean={whole['mean']:.3f}  min={whole['min']:.3f}  "
              f"first_content_mean={whole['first_content_mean']:.3f}  "
              f"({whole['n_tokens']} tok)")

        # Confidence restricted to tokens AFTER the </think> tag, if we can
        # find it. We do this by walking tokens until we see "</think>".
        tokens = whole["tokens"]
        probs = whole["probs"]
        cut = None
        joined = ""
        for i, tok in enumerate(tokens):
            joined += tok
            if "</think>" in joined:
                cut = i + 1
                break
        if cut is not None and cut < len(tokens):
            post_probs = probs[cut:]
            if post_probs:
                post_mean = sum(post_probs) / len(post_probs)
                post_min = min(post_probs)
                print(f"[post]  mean={post_mean:.3f}  min={post_min:.3f}  "
                      f"({len(post_probs)} tok after </think>)")
        else:
            print("[post]  (no </think> found — model didn't close the block)")
        print()

    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    out_path = runs_dir / "phase3_deepseek_inspect.json"
    out_path.write_text(json.dumps({"note": "see stdout"}, indent=2))
    print(f"[ok] inspection done. Final RSS: {_process_rss_gb():.3f} GB")


if __name__ == "__main__":
    main()
