"""Phase 4.8/4.9/4.10 smoke test: validate every SOTA-capture piece in isolation.

Run this on the compute box BEFORE kicking off the full ``04f_run_qa_sota.py``
sweep. It exercises one model end-to-end on a single 200-question dataset
row and verifies each capture component independently:

  T1. spaCy + rank_bm25 + sentence-transformers imports cleanly and the
      ``EntityEntropyScorer`` returns a finite u_ent in [0, 1] on a
      short fixture string.
  T2. The model loads with ``output_attentions=True`` and the forward
      pass produces a non-None ``out.attentions`` tuple of the right
      shape per layer.
  T3. ``RindAccumulator`` ingests one attention row without raising and
      returns a finite ``rind_max >= 0``.
  T4. ``VerbalizedConfidenceProbe.probe()`` returns a number in [0, 1]
      OR ``None`` (parsing failure is acceptable; we just verify no
      crash) AND the KV cache is restored bit-exactly to its pre-fork
      sequence length.
  T5. ``ConsistencySampler.score()`` returns a finite u_cons in [0, 1]
      AND the KV cache is restored. (Skipped if --skip-cons; the sampler
      is the most expensive piece.)
  T6. End-to-end: run 1 question through ``run_one_question_sota`` with
      ``--max-new-tokens 64`` and verify the output JSON has every new
      step-record key.

Usage (on compute box):
  python scripts/04h_sota_smoke.py --model qwen3.6-35b-a3b-instruct
  python scripts/04h_sota_smoke.py --model phi-4-mini-instruct --skip-cons
  python scripts/04h_sota_smoke.py --model gemma-4-26b-a4b-it

Exit code 0 on all-pass, non-zero on first failure. Writes a brief
JSONL report at ``runs/phase4_sota_smoke_<model>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from importlib import import_module
qa = import_module("04_run_qa")  # type: ignore
sota = import_module("04f_run_qa_sota")  # type: ignore

from crv.sota_capture import (  # type: ignore[import-not-found]
    ConsistencySampler,
    EntityEntropyScorer,
    RindAccumulator,
    VerbalizedConfidenceProbe,
    compute_token_entropy_nats,
    parse_verbal_confidence,
    stopword_filter,
)


def _is_finite(x: float | None) -> bool:
    return x is not None and isinstance(x, float) and x == x and x not in (float("inf"), float("-inf"))


def t1_cpu_only(report: dict) -> bool:
    print("[T1] CPU-only deps: spaCy + rank_bm25 + sentence-transformers")
    try:
        paragraphs = [
            {"joined_text": "Cleopatra ruled Egypt from 51 BC to 30 BC."},
            {"joined_text": "Mark Antony married Cleopatra after Julius Caesar."},
            {"joined_text": "The Battle of Actium was in 31 BC."},
        ]
        scorer = EntityEntropyScorer(paragraphs)
        e = scorer.score("Cleopatra and Mark Antony fought at Actium.")
        ok = (
            _is_finite(float(e["u_ent"]))
            and 0.0 <= float(e["u_ent"]) <= 1.0
            and int(e["n_entities"]) >= 1
        )
        report["T1"] = {"u_ent": e["u_ent"], "n_entities": e["n_entities"], "ok": bool(ok)}
        print(f"     -> u_ent={e['u_ent']:.3f}  n_entities={e['n_entities']}  {'OK' if ok else 'FAIL'}")
        return bool(ok)
    except Exception as exc:
        report["T1"] = {"error": str(exc), "trace": traceback.format_exc(), "ok": False}
        print(f"     -> FAIL: {exc}")
        return False


def t2_attention_capture(model, tokenizer, dragin_layer_idx: int, report: dict) -> bool:
    print("[T2] forward pass with output_attentions=True")
    import torch
    try:
        prompt = "The capital of France is"
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                use_cache=True,
                output_hidden_states=False,
                output_attentions=True,
            )
        attns = getattr(out, "attentions", None)
        if attns is None:
            report["T2"] = {"error": "out.attentions is None", "ok": False}
            print("     -> FAIL: out.attentions is None. "
                  "Probably flash-attn-only; set attn_implementation='eager'.")
            return False
        a = attns[dragin_layer_idx]
        if a is None:
            report["T2"] = {
                "n_layers": len(attns),
                "layer_idx": dragin_layer_idx,
                "error": "this layer's attentions are None (likely linear/GDN layer)",
                "ok": False,
            }
            print(f"     -> FAIL: attentions at layer {dragin_layer_idx} are None. "
                  f"On hybrid archs, override --dragin-attn-layer-idx to a full-attention layer.")
            return False
        shape = list(a.shape)
        report["T2"] = {
            "n_layers": len(attns),
            "layer_idx": dragin_layer_idx,
            "shape": shape,
            "ok": True,
        }
        print(f"     -> attentions[{dragin_layer_idx}].shape={shape}  OK")
        return True
    except Exception as exc:
        report["T2"] = {"error": str(exc), "trace": traceback.format_exc(), "ok": False}
        print(f"     -> FAIL: {exc}")
        return False


def t3_rind_accumulator(model, tokenizer, dragin_layer_idx: int, report: dict) -> bool:
    print("[T3] RindAccumulator end-to-end on a 4-token forward")
    import torch
    try:
        prompt = "Q: 2 + 2 ="
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = int(enc.input_ids.shape[-1])
        acc = RindAccumulator(prompt_len=prompt_len)
        cache = None
        input_ids = enc.input_ids
        attention_mask = enc.attention_mask
        for step in range(4):
            with torch.inference_mode():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask if step == 0 else None,
                    past_key_values=cache,
                    use_cache=True,
                    output_attentions=True,
                )
            cache = out.past_key_values
            next_logits = out.logits[:, -1, :]
            tok = int(next_logits.argmax(dim=-1).item())
            txt = tokenizer.decode([tok], skip_special_tokens=False)
            h = compute_token_entropy_nats(next_logits)
            acc.observe_new_token(txt, h)
            attns = out.attentions
            if attns is not None and attns[dragin_layer_idx] is not None:
                acc.ingest_attention_row(attns[dragin_layer_idx])
            input_ids = torch.tensor([[tok]], device=model.device)
        rind = acc.commit_step()
        ok = rind is not None and _is_finite(rind["rind_max"]) and rind["rind_max"] >= 0.0
        report["T3"] = {"rind": rind, "ok": bool(ok)}
        print(f"     -> {rind}  {'OK' if ok else 'FAIL'}")
        return bool(ok)
    except Exception as exc:
        report["T3"] = {"error": str(exc), "trace": traceback.format_exc(), "ok": False}
        print(f"     -> FAIL: {exc}")
        return False


def t4_verbal_probe(model, tokenizer, report: dict) -> bool:
    print("[T4] VerbalizedConfidenceProbe: fork + restore + parse")
    import torch
    try:
        prompt = "Q: 2 + 2 = 4. Step 1: I checked the arithmetic."
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                use_cache=True,
            )
        cache = out.past_key_values
        last_tok = int(enc.input_ids[0, -1].item())
        # Snapshot the cache shape *before* the probe.
        probe = VerbalizedConfidenceProbe(model, tokenizer, max_probe_tokens=12)
        len_before = probe._cache_seq_len(cache)
        u_verb = probe.probe(cache, last_tok)
        len_after = probe._cache_seq_len(cache)
        cache_ok = len_after == len_before
        u_ok = u_verb is None or (0.0 <= u_verb <= 1.0)
        ok = bool(cache_ok and u_ok)
        report["T4"] = {
            "u_verb": u_verb,
            "len_before": len_before,
            "len_after": len_after,
            "cache_restored": cache_ok,
            "ok": ok,
        }
        print(f"     -> u_verb={u_verb}  len_before={len_before}  len_after={len_after}  "
              f"{'OK' if ok else 'FAIL'}")
        return ok
    except Exception as exc:
        report["T4"] = {"error": str(exc), "trace": traceback.format_exc(), "ok": False}
        print(f"     -> FAIL: {exc}")
        return False


def t5_consistency_sampler(model, tokenizer, report: dict, skip: bool) -> bool:
    if skip:
        print("[T5] ConsistencySampler: SKIPPED (--skip-cons)")
        report["T5"] = {"skipped": True, "ok": True}
        return True
    print("[T5] ConsistencySampler: 3 samples, T=0.6, top_p=0.9, then cache restore")
    import torch
    try:
        prompt = "Q: 2 + 2 = 4. Step 1: I checked the arithmetic. Step 2:"
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                use_cache=True,
            )
        cache = out.past_key_values
        last_tok = int(enc.input_ids[0, -1].item())
        sampler = ConsistencySampler(model, tokenizer, n_samples=3, max_new_tokens=20)
        len_before = sampler._cache_seq_len(cache)
        c = sampler.score(cache, last_tok)
        len_after = sampler._cache_seq_len(cache)
        cache_ok = len_after == len_before
        u_ok = _is_finite(c["u_cons"]) and 0.0 <= c["u_cons"] <= 1.0
        ok = bool(cache_ok and u_ok)
        report["T5"] = {
            "u_cons": c["u_cons"],
            "mean_pairwise_cos": c["mean_pairwise_cosine"],
            "len_before": len_before,
            "len_after": len_after,
            "cache_restored": cache_ok,
            "ok": ok,
        }
        print(f"     -> u_cons={c['u_cons']:.3f}  mean_cos={c['mean_pairwise_cosine']:.3f}  "
              f"len_before={len_before}  len_after={len_after}  {'OK' if ok else 'FAIL'}")
        return ok
    except Exception as exc:
        report["T5"] = {"error": str(exc), "trace": traceback.format_exc(), "ok": False}
        print(f"     -> FAIL: {exc}")
        return False


def t6_end_to_end(model, tokenizer, dragin_layer_idx: int, dataset_path: Path, model_name: str,
                  report: dict, skip_cons: bool) -> bool:
    print("[T6] one full question through run_one_question_sota (max_new_tokens=64)")
    try:
        # Pull the first row.
        row = None
        with dataset_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                break
        if row is None:
            report["T6"] = {"error": "dataset empty", "ok": False}
            return False

        layer_subset = qa._resolve_layer_subset(model, "mid_stack")

        ns = argparse.Namespace(
            model=model_name,
            layer_mode="mid_stack",
            max_new_tokens=64,
            window_size=10,
            threshold_drift=0.1,
            threshold_variance=1.0,
            etc_threshold=0.5,
            selfdoubt_t1_threshold=1.0,
            selfdoubt_full_z=1.0,
            dragin_threshold=1.0,
            dragin_attn_layer_idx=dragin_layer_idx,
            rsus_threshold=0.65,
            random_seed=20260509,
            cons_k=3,
            cons_max_new=20,
            no_verb=False,
            no_ent=False,
            no_cons=bool(skip_cons),
            inline_system=False,
        )

        probes = {"dragin_layer_idx": dragin_layer_idx}
        probes["verbal_probe"] = VerbalizedConfidenceProbe(model, tokenizer, max_probe_tokens=12)
        if not skip_cons:
            probes["consistency_sampler"] = ConsistencySampler(
                model, tokenizer, n_samples=3, max_new_tokens=20, temperature=0.6, top_p=0.9, seed=20260509,
            )
        probes["entity_scorer_factory"] = lambda paragraphs: EntityEntropyScorer(paragraphs)

        t0 = time.perf_counter()
        result = sota.run_one_question_sota(row, tokenizer, model, layer_subset, probes, ns)
        wall = time.perf_counter() - t0

        # Check that every new step-record key is present on at least one step.
        required_keys = {
            "dragin_rind_max", "hvr", "u_verb", "u_ent", "u_cons",
            "hedge_count", "verify_count",
        }
        steps = result.get("steps") or []
        if not steps:
            report["T6"] = {"error": "no steps emitted; max_new_tokens too low or step detector silent",
                            "wall_s": wall, "ok": False}
            print(f"     -> FAIL: no step records emitted ({wall:.1f}s)")
            return False
        first = steps[0]
        missing = [k for k in required_keys if k not in first]
        ok = not missing
        report["T6"] = {
            "n_steps": len(steps),
            "wall_s": round(wall, 3),
            "tokens_generated": result["tokens_generated"],
            "missing_step_keys": missing,
            "first_step_excerpt": {k: first.get(k) for k in sorted(required_keys)},
            "fire_counts": result["fire_counts"],
            "ok": bool(ok),
        }
        print(f"     -> {len(steps)} steps in {wall:.1f}s; missing keys: {missing or 'none'}  "
              f"{'OK' if ok else 'FAIL'}")
        if ok:
            print(f"     fire_counts: {result['fire_counts']}")
        return ok
    except Exception as exc:
        report["T6"] = {"error": str(exc), "trace": traceback.format_exc(), "ok": False}
        print(f"     -> FAIL: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default=qa.DEFAULT_DATASET_PATH)
    parser.add_argument("--dragin-attn-layer-idx", type=int, default=None)
    parser.add_argument("--skip-cons", action="store_true",
                        help="Skip T5 + the consistency leg of T6 (use to triage faster).")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_path = Path(args.out or f"runs/phase4_sota_smoke_{args.model}.json")
    report: dict = {"model": args.model, "args": vars(args), "results": {}}

    print(f"\n=== Phase 4 SOTA smoke test on {args.model} ===\n")

    # T1 needs no model.
    t1_ok = t1_cpu_only(report["results"])
    if not t1_ok:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print("\n[!!] T1 failed before model load. Fix CPU deps first.")
        return 1

    model_dir = Path("models") / args.model
    if not model_dir.exists():
        report["model_dir_missing"] = str(model_dir)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\n[!!] missing {model_dir}; aborting before GPU tests.")
        return 1

    tokenizer, model = qa.load_generator(model_dir)
    dragin_layer_idx = sota._resolve_dragin_attn_layer(model, args.dragin_attn_layer_idx)
    print(f"\n[..] dragin_attn_layer_idx={dragin_layer_idx}")
    if hasattr(model.config, "layer_types"):
        print(f"[..] layer_types: {list(model.config.layer_types)}")

    all_ok = True
    all_ok &= t2_attention_capture(model, tokenizer, dragin_layer_idx, report["results"])
    all_ok &= t3_rind_accumulator(model, tokenizer, dragin_layer_idx, report["results"])
    all_ok &= t4_verbal_probe(model, tokenizer, report["results"])
    all_ok &= t5_consistency_sampler(model, tokenizer, report["results"], skip=args.skip_cons)
    all_ok &= t6_end_to_end(
        model, tokenizer, dragin_layer_idx, Path(args.dataset), args.model,
        report["results"], skip_cons=args.skip_cons,
    )

    report["overall_ok"] = bool(all_ok)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[ok] wrote {out_path}")
    print(f"[{'OK' if all_ok else 'FAIL'}] overall smoke: {all_ok}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
