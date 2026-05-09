"""Phase 2.1 — inspect the logprobs schema returned by llama-cpp-python.

The original draft assumed `response['logits']` exists; we don't trust that.
This script runs one short prompt with `logprobs=5` and prints the full
top-level structure plus a single-token slice, so we can write the gate
against the *actual* schema.

Run:
    conda run -n recollection --no-capture-output python scripts/02_inspect_logprobs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sysinfo import _process_rss_gb  # type: ignore[import-not-found]

MODEL_PATH = Path("models/Phi-3.5-mini-instruct-Q4_K_M.gguf")
PROMPT = "The capital of France is"
MAX_TOKENS = 6
TOP_LOGPROBS = 5


def shape(obj, depth: int = 0, max_depth: int = 4) -> str:
    """Compact recursive type/shape description."""
    pad = "  " * depth
    if depth > max_depth:
        return f"{pad}..."
    if isinstance(obj, dict):
        lines = [f"{pad}dict (keys: {list(obj.keys())})"]
        for k, v in obj.items():
            lines.append(f"{pad}  .{k} ->")
            lines.append(shape(v, depth + 2, max_depth))
        return "\n".join(lines)
    if isinstance(obj, list):
        n = len(obj)
        if n == 0:
            return f"{pad}list (empty)"
        lines = [f"{pad}list (len={n}, first item:)"]
        lines.append(shape(obj[0], depth + 1, max_depth))
        return "\n".join(lines)
    if isinstance(obj, str):
        s = obj if len(obj) < 60 else obj[:57] + "..."
        return f"{pad}str  {s!r}"
    return f"{pad}{type(obj).__name__}  {obj!r}"


def main() -> None:
    if not MODEL_PATH.exists():
        print(f"[!!] missing {MODEL_PATH}")
        sys.exit(1)

    print(f"[..] loading {MODEL_PATH.name}")
    from llama_cpp import Llama
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False,
        logits_all=True,  # required by llama-cpp-python for logprobs=N at generation
    )
    print(f"[ok] loaded, rss={_process_rss_gb():.3f} GB")

    print(f"[..] prompt: {PROMPT!r}, max_tokens={MAX_TOKENS}, "
          f"logprobs={TOP_LOGPROBS}")
    out = llm(
        PROMPT,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        logprobs=TOP_LOGPROBS,
        echo=False,
    )

    print("\n" + "=" * 60)
    print("TOP-LEVEL SHAPE")
    print("=" * 60)
    print(shape(out))

    choice = out["choices"][0]
    lp = choice.get("logprobs")
    print("\n" + "=" * 60)
    print("choices[0]['logprobs'] keys: " + str(list(lp.keys()) if lp else None))
    print("=" * 60)
    if lp:
        tokens = lp.get("tokens", [])
        token_logprobs = lp.get("token_logprobs", [])
        top_logprobs = lp.get("top_logprobs", [])
        print(f"tokens                  ({len(tokens)})  : {tokens}")
        print(f"token_logprobs          ({len(token_logprobs)})  : {token_logprobs}")
        print(f"top_logprobs[0]              : "
              f"{json.dumps(top_logprobs[0] if top_logprobs else None, indent=2, default=float)}")

        # The number we actually want for the confidence gate:
        import math
        chosen_probs = [math.exp(x) if x is not None else None
                        for x in token_logprobs]
        valid = [p for p in chosen_probs if p is not None]
        avg = sum(valid) / len(valid) if valid else float("nan")
        print(f"\nexp(token_logprobs)     : {[round(p, 4) if p else None for p in chosen_probs]}")
        print(f"avg confidence          : {avg:.4f}")

    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    snapshot_path = runs_dir / "phase2_logprobs_schema.json"
    snapshot_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[ok] wrote full response to {snapshot_path}")


if __name__ == "__main__":
    main()
