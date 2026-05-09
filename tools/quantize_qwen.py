"""Pre-quantize a HF model snapshot to NF4 and save to disk.

Workaround for the transformers 5.8 + bitsandbytes 0.49.2 NF4 streaming-load
regression diagnosed in `_AGENT.md` (2026-05-09 #2). When `from_pretrained` is
asked to load a 35B-class model with `quantization_config=BitsAndBytesConfig(...)`
on a single tight-VRAM GPU (≤44 GiB), transformers 5.x's new
`core_model_loading._materialize_copy` ships bf16 shards to the GPU *before*
bnb gets a chance to quantize, OOMing well before reaching the ~18 GB NF4
final size.

This script sidesteps the load-time race entirely:

  1. Load the source snapshot on CPU in bf16 via `AutoModelForCausalLM.from_pretrained`.
     Transformers' from_pretrained handles the checkpoint's key remapping
     (e.g. `model.language_model.layers.X` -> `model.layers.X` for VL-derived
     snapshots that share an LM with the conditional-generation sibling). With
     2 TB RAM, holding 67 GB of bf16 weights is cheap.
  2. Walk every `nn.Linear` (skipping a configurable set such as `lm_head`).
     For each one, construct a fresh `bnb.nn.Linear4bit`, attach the source
     bf16 weight as a `bnb.nn.Params4bit`, move that single module to GPU
     (Params4bit's `.cuda()` is where bnb actually runs `quantize_4bit` and
     stores the 4-bit packed bytes), then splice it back into the parent module.
     The original bf16 Linear is dropped, freeing its CPU storage immediately.
     Peak VRAM during the loop ≈ (largest single Linear's bf16 + accumulating
     4-bit). For a 35B MoE model this stays under ~3 GiB.
  3. Move the remaining non-quantizable modules (embeddings / norms / lm_head)
     to GPU in their original dtype.
  4. `save_pretrained` writes the 4-bit-packed shards plus a
     `quantization_config` entry into `config.json`. Subsequent
     `from_pretrained(..., quantization_config=...)` materializes only ~18 GB
     total (per-shard much smaller), which fits the 44 GiB GPU budget and
     skips the broken streaming-quant path entirely.

We avoid `accelerate.dispatch_model` and `load_checkpoint_in_model` because
both re-trigger the `Params4bit.__new__` kwarg incompatibility and the
`Tensor.item() on meta` crashes documented in `_AGENT.md`. Doing the swap
manually keeps us inside well-tested module-replacement territory.

Usage:
    python tools/quantize_qwen.py \
        --src models/qwen3.6-35b-a3b-instruct \
        --dst models/qwen3.6-35b-a3b-instruct-nf4
"""

from __future__ import annotations

import argparse
import gc
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))

from sysinfo import _process_rss_gb, vram_used_mb  # type: ignore[import-not-found]


def _format_gb(mb: int | None) -> str:
    return "n/a" if mb is None else f"{mb / 1024:.2f} GiB"


def _set_submodule(root, dotted_name: str, new_module) -> None:
    """torch.nn doesn't expose `set_submodule` until 2.4; do it by hand."""
    parent_path, _, attr = dotted_name.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, attr, new_module)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--src",
        default="models/qwen3.6-35b-a3b-instruct",
        help="Source model directory (bf16 safetensors snapshot).",
    )
    parser.add_argument(
        "--dst",
        default="models/qwen3.6-35b-a3b-instruct-nf4",
        help="Destination directory for the NF4 snapshot.",
    )
    parser.add_argument(
        "--keep-bf16",
        nargs="*",
        default=["lm_head"],
        help="Module name substrings to leave in bf16 (default: lm_head).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Target device for quantization (must be CUDA for bnb).",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=100,
        help="Print VRAM/RSS after every N quantized Linear4bit modules.",
    )
    args = parser.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not src.exists():
        sys.exit(f"[!!] missing source: {src}")
    if dst.exists() and any(dst.iterdir()):
        sys.exit(f"[!!] destination not empty: {dst} (delete or pick a different --dst)")
    dst.mkdir(parents=True, exist_ok=True)

    print("[..] importing torch + transformers + bitsandbytes")
    import torch
    from torch import nn
    import bitsandbytes as bnb
    import bitsandbytes.nn as _bnb_nn

    # Compat shim: see _AGENT.md "compat finding". We don't go through
    # accelerate's dispatch_model here, but transformers' save_pretrained may
    # still construct Params4bit indirectly. Keep the kwarg-dropping shim on.
    if not getattr(_bnb_nn.Params4bit, "_recollect_kwargs_patched", False):
        _orig_new = _bnb_nn.Params4bit.__new__
        _allowed = {
            "data", "requires_grad", "quant_state", "blocksize",
            "compress_statistics", "quant_type", "quant_storage", "module",
            "bnb_quantized",
        }
        def _patched_new(cls, data=None, requires_grad=False, **kwargs):
            kwargs = {k: v for k, v in kwargs.items() if k in _allowed}
            return _orig_new(cls, data=data, requires_grad=requires_grad, **kwargs)
        _bnb_nn.Params4bit.__new__ = _patched_new
        _bnb_nn.Params4bit._recollect_kwargs_patched = True

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        sys.exit("[!!] CUDA not available; bnb 4-bit quantization requires a GPU.")
    target = torch.device(args.device)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    cfg = AutoConfig.from_pretrained(str(src))
    text_cfg = getattr(cfg, "text_config", cfg)
    print(
        f"[ok] config: model_type={cfg.model_type} "
        f"hidden={getattr(text_cfg, 'hidden_size', '?')} "
        f"layers={getattr(text_cfg, 'num_hidden_layers', '?')} "
        f"vocab={getattr(text_cfg, 'vocab_size', '?')}"
    )

    rss0 = _process_rss_gb() or 0.0
    print(f"\n[..] step 1: from_pretrained on CPU in bf16 (rss baseline {rss0:.2f} GB)")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(src),
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_secs = time.perf_counter() - t0
    rss_after = _process_rss_gb() or 0.0
    print(f"[ok] loaded on CPU in {load_secs:.1f}s | rss {rss0:.2f} -> {rss_after:.2f} GB")

    n_total_lin = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    skip_set = list(args.keep_bf16)
    print(
        f"\n[..] step 2: replace {n_total_lin} nn.Linear modules with bnb.Linear4bit "
        f"(skip name-substrings: {skip_set})"
    )
    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(target)
    targets: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(s in name for s in skip_set):
                continue
            targets.append((name, module))
    print(f"    -> {len(targets)} modules to quantize, {n_total_lin - len(targets)} kept in bf16")

    n_done = 0
    for name, old in targets:
        # Build the empty Linear4bit shell first (cpu meta-ish; nothing
        # quantized yet because no Params4bit data is attached).
        new = _bnb_nn.Linear4bit(
            old.in_features,
            old.out_features,
            bias=old.bias is not None,
            compute_dtype=torch.bfloat16,
            quant_type="nf4",
            compress_statistics=True,
        )
        # Build a Params4bit holding the source bf16 weight. Constructing on
        # CPU (data still on cpu) is cheap; the .to(target) below is what
        # actually invokes bnb's quantize_4bit on the GPU.
        p = _bnb_nn.Params4bit(
            old.weight.data,
            requires_grad=False,
            quant_type="nf4",
            compress_statistics=True,
        )
        new.weight = p
        if old.bias is not None:
            new.bias = nn.Parameter(old.bias.data.to(dtype=torch.bfloat16), requires_grad=False)
        # Move + quantize. Params4bit overrides `.to(cuda)` to run quantize_4bit;
        # after this call new.weight holds the 4-bit packed bytes plus quant_state.
        new = new.to(target)
        # Splice into parent. Drop the old Linear so its bf16 storage is GC'd.
        _set_submodule(model, name, new)
        del old, p
        n_done += 1
        if n_done % args.report_every == 0 or n_done == len(targets):
            gc.collect()
            torch.cuda.empty_cache()
            vram = vram_used_mb()
            rss = _process_rss_gb() or 0.0
            print(
                f"  [{n_done:4d}/{len(targets)}] "
                f"vram={_format_gb(vram)} rss={rss:.1f} GB  ({name})"
            )
    quant_secs = time.perf_counter() - t0
    peak_mb = int(torch.cuda.max_memory_allocated(target) / 1024**2)
    print(
        f"[ok] {n_done} Linear4bit modules quantized in {quant_secs:.1f}s; "
        f"peak VRAM during quant = {peak_mb} MB ({_format_gb(peak_mb)})"
    )

    print(
        f"\n[..] step 3: move remaining (kept-bf16, embeddings, norms) to {target}"
    )
    model.to(target)
    gc.collect()
    torch.cuda.empty_cache()
    vram_final = vram_used_mb()
    rss_final = _process_rss_gb() or 0.0
    print(
        f"[ok] final on-GPU footprint: vram={_format_gb(vram_final)} rss={rss_final:.1f} GB"
    )

    # Tell save_pretrained how to round-trip this model.
    model.config.quantization_config = bnb_config.to_dict()
    model.is_quantized = True
    model.is_loaded_in_4bit = True

    print(f"\n[..] step 4: save_pretrained -> {dst}")
    t0 = time.perf_counter()
    model.save_pretrained(str(dst), safe_serialization=True)
    save_secs = time.perf_counter() - t0
    print(f"[ok] model written in {save_secs:.1f}s")

    print(f"[..] saving tokenizer + chat template")
    tok = AutoTokenizer.from_pretrained(str(src))
    tok.save_pretrained(str(dst))
    chat_jinja = src / "chat_template.jinja"
    if chat_jinja.exists():
        shutil.copy2(chat_jinja, dst / "chat_template.jinja")
        print(f"[ok] copied chat_template.jinja")

    on_disk_bytes = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
    print(f"\n[ok] done. {dst.name} on disk = {on_disk_bytes / 1024**3:.2f} GiB")
    print(
        f"    Next: `python scripts/00_smoke.py --model {dst.name}` "
        "(no --no-quant flag needed)."
    )


if __name__ == "__main__":
    main()
