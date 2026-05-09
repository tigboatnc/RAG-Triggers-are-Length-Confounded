"""Phase 1 REPL (chat mode).

Tiny interactive loop on top of a single GGUF using `create_chat_completion`
so the model's chat template is applied (Phase 2.5 lesson). Every turn prints:
  - the assistant answer
  - tokens generated, wall time, tokens/sec
  - process RSS delta (so we can see the KV cache grow)
  - mean / min / first_content_mean confidence

Multi-turn history is kept; type ':reset' to clear it. ':sys' re-prints sysinfo.

Run:
    conda run -n recollection --no-capture-output \
        python scripts/01_repl.py --model models/Phi-3.5-mini-instruct-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sysinfo import _process_rss_gb, report  # type: ignore[import-not-found]
from confidence import summarize  # type: ignore[import-not-found]

DEFAULT_MODEL = "models/Phi-3.5-mini-instruct-Q4_K_M.gguf"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 REPL on a GGUF model.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Path to GGUF file.")
    p.add_argument("--n_ctx", type=int, default=4096, help="Context window in tokens.")
    p.add_argument("--n_gpu_layers", type=int, default=-1,
                   help="-1 = offload everything to Metal/CUDA.")
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--no-sysinfo", action="store_true",
                   help="Skip the sysinfo dump at startup.")
    p.add_argument("--top_logprobs", type=int, default=5,
                   help="Top-N logprobs per token (cost: full-vocab logits).")
    p.add_argument("--show-tokens", action="store_true",
                   help="Print per-token (text, prob) list each turn.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[!!] missing {model_path}. Run: bash scripts/download_models.sh")
        sys.exit(1)

    if not args.no_sysinfo:
        report()

    print(f"\n[..] importing llama_cpp ...")
    from llama_cpp import Llama
    rss_after_import = _process_rss_gb() or 0.0

    print(f"[..] loading {model_path.name} (n_ctx={args.n_ctx}, "
          f"n_gpu_layers={args.n_gpu_layers})")
    t0 = time.perf_counter()
    llm = Llama(
        model_path=str(model_path),
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        verbose=False,
        logits_all=True,  # required by llama-cpp-python to expose token_logprobs at call time
    )
    load_secs = time.perf_counter() - t0
    rss_after_load = _process_rss_gb() or 0.0
    print(
        f"[ok] loaded in {load_secs:.2f}s | "
        f"rss {rss_after_import:.3f} -> {rss_after_load:.3f} GB "
        f"(+{rss_after_load - rss_after_import:.3f} GB)"
    )
    print("\nType your prompt. Commands: 'exit' / ':sys' / ':reset' / Ctrl-D.\n")

    turn = 0
    rss_prev = rss_after_load
    messages: list[dict[str, str]] = []
    while True:
        try:
            prompt = input(f"[{turn}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit", ":q"}:
            break
        if prompt == ":sys":
            report()
            continue
        if prompt == ":reset":
            messages = []
            print("[ok] history cleared\n")
            continue

        messages.append({"role": "user", "content": prompt})
        t0 = time.perf_counter()
        out = llm.create_chat_completion(
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            logprobs=True,
            top_logprobs=args.top_logprobs,
        )
        gen_secs = time.perf_counter() - t0
        text = out["choices"][0]["message"]["content"].strip()
        messages.append({"role": "assistant", "content": text})
        n_tokens = out["usage"]["completion_tokens"]
        rss_now = _process_rss_gb() or 0.0
        delta = rss_now - rss_prev
        conf = summarize(out)

        print("\n" + text + "\n")
        print(
            f"[turn {turn}] {n_tokens} tok / {gen_secs:.2f}s "
            f"= {n_tokens / gen_secs:.1f} tok/s | "
            f"rss {rss_prev:.3f} -> {rss_now:.3f} GB "
            f"({'+' if delta >= 0 else ''}{delta:.3f})"
        )
        print(
            f"[conf]   mean={conf['mean']:.3f}  "
            f"min={conf['min']:.3f}  "
            f"first_content_mean={conf['first_content_mean']:.3f}"
        )
        if args.show_tokens:
            pairs = [f"{t!r}={p}" for t, p in zip(conf["tokens"], conf["probs"])]
            print("[tokens] " + "  ".join(pairs))
        rss_prev = rss_now
        turn += 1

    print(f"\n[ok] {turn} turn(s). Final RSS: {rss_prev:.3f} GB")


if __name__ == "__main__":
    main()
