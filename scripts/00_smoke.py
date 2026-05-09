"""Phase 0.6 smoke test.

Loads the Phase-0 GGUF, runs one prompt, prints output + tokens/sec + RSS deltas.
This is the first 'visible' artifact: model speaks back, with memory accounting.

Run:
    conda run -n recollection --no-capture-output python scripts/00_smoke.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sysinfo import _process_rss_gb, report  # type: ignore[import-not-found]

MODEL_PATH = Path("models/Phi-3.5-mini-instruct-Q4_K_M.gguf")
PROMPT = "In one sentence, what is retrieval-augmented generation?"
N_CTX = 4096
N_GPU_LAYERS = -1  # offload everything to Metal
MAX_TOKENS = 96


def main() -> None:
    if not MODEL_PATH.exists():
        print(f"[!!] missing {MODEL_PATH}. Run: bash scripts/download_models.sh")
        sys.exit(1)

    info = report()
    rss_before_import = _process_rss_gb() or 0.0

    print(f"\n[..] importing llama_cpp ...")
    from llama_cpp import Llama
    rss_after_import = _process_rss_gb() or 0.0
    print(f"[ok] import: rss {rss_before_import:.3f} -> {rss_after_import:.3f} GB")

    print(f"[..] loading {MODEL_PATH.name} (n_ctx={N_CTX}, n_gpu_layers={N_GPU_LAYERS})")
    t0 = time.perf_counter()
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )
    load_secs = time.perf_counter() - t0
    rss_after_load = _process_rss_gb() or 0.0
    print(
        f"[ok] loaded in {load_secs:.2f}s | "
        f"rss {rss_after_import:.3f} -> {rss_after_load:.3f} GB "
        f"(+{rss_after_load - rss_after_import:.3f} GB)"
    )

    print(f"\n[..] prompt: {PROMPT!r}")
    t0 = time.perf_counter()
    out = llm(
        PROMPT,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        echo=False,
    )
    gen_secs = time.perf_counter() - t0
    text = out["choices"][0]["text"].strip()
    n_tokens = out["usage"]["completion_tokens"]
    rss_after_gen = _process_rss_gb() or 0.0

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(text)
    print("=" * 60)
    print(
        f"[ok] {n_tokens} tokens in {gen_secs:.2f}s "
        f"= {n_tokens / gen_secs:.1f} tok/s | "
        f"rss {rss_after_load:.3f} -> {rss_after_gen:.3f} GB"
    )
    print(f"[summary] backend={info['gpu']['backend']} "
          f"chip={info['gpu'].get('details')} "
          f"final_rss={rss_after_gen:.3f} GB / total={info['ram']['total_gb']} GB")


if __name__ == "__main__":
    main()
