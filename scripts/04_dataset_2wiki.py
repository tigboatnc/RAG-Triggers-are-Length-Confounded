"""Phase 4 / #21 OOD slice assembly: 200-question 2WikiMultiHopQA-distractor
slice for the zero-shot transfer test (Qwen3.6 only, frozen thresholds from
HotpotQA #20 calibration).

Mirrors ``scripts/04_dataset.py`` (HotpotQA) but for 2WikiMultiHopQA. All 200
questions are tagged ``split = "ood_test"`` on disk: there is no calibration
or validation partition because *no thresholds and no model-selection knobs
are fit on this dataset* — this is a frozen-threshold OOD transfer test, not
an in-distribution evaluation. See ``_AGENT.md`` #21 for the full
pre-registration.

Schema is otherwise byte-compatible with the HotpotQA slice so the existing
sweep driver (``scripts/04f_run_qa_sota.py``) reads it without modification:

  - split       (str): "ood_test" for every row
  - split_idx   (int): row index in the source split
  - id          (str): 2Wiki ``_id``
  - question    (str)
  - answer      (str)
  - type        (str): "compositional" | "inference" | "comparison" | "bridge_comparison"
  - level       (str): "" (2Wiki has no difficulty annotation; kept for schema parity)
  - paragraphs  (list): [{paragraph_idx, title, sentences, joined_text}, ...]
  - supporting_facts (list): [{title, sent_id}, ...]

2WikiMultiHopQA source: the original release (Ho et al. 2020) stores
``context`` as ``list[ [title, list[str_sentences]] ]`` (10 entries: ~2
supporting + ~8 distractors). HF mirrors generally preserve this shape but
field names vary. We try the most common mirror first
(``xanhho/2WikiMultihopQA``), fall back to alternatives, and stamp the
chosen mirror in the manifest so the audit trail is unambiguous. If none
load cleanly the user is told which one to try with --hf-name.

Usage:
    python scripts/04_dataset_2wiki.py                  # default: 200 q, seed 20260511
    python scripts/04_dataset_2wiki.py --n 5            # smoke slice
    python scripts/04_dataset_2wiki.py --hf-name voidful/2WikiMultihopQA  # override mirror
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

# Candidate HF mirrors, in priority order. The first one that loads
# successfully wins; the chosen mirror is recorded in the manifest.
DEFAULT_MIRRORS = [
    "xanhho/2WikiMultihopQA",
    "voidful/2WikiMultihopQA",
    "iohadrubin/2wikimultihop",
]
DATASET_SPLIT_CANDIDATES = ["validation", "dev", "test"]

DEFAULT_N = 200
DEFAULT_SEED = 20260511
SPLIT_TAG = "ood_test"

# 2Wiki's four reasoning types per Ho et al. 2020 §3.
KNOWN_TYPES = {"compositional", "inference", "comparison", "bridge_comparison"}


def _stratified_indices(items: list[dict[str, Any]], seed: int) -> list[int]:
    """Round-robin draw from per-type buckets so the 200-q slice mirrors the
    source distribution over the four reasoning types as closely as possible.

    Returns the indices into ``items`` to include in the slice. Ordering of
    the returned list is then sorted ascending so the on-disk JSONL is
    reproducible from (source-split-fingerprint, seed) alone.
    """
    by_type: dict[str, list[int]] = {}
    for i, item in enumerate(items):
        t = str(item.get("type", "unknown"))
        by_type.setdefault(t, []).append(i)

    cell_seed_base = seed
    for t in list(by_type.keys()):
        cell_seed = int.from_bytes(
            hashlib.sha256(f"{cell_seed_base}|{t}".encode("utf-8")).digest()[:8],
            "big",
        )
        rng = random.Random(cell_seed)
        rng.shuffle(by_type[t])

    return by_type


def _normalize_paragraphs(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize a 2Wiki ``context`` entry to the HotpotQA-compatible shape
    used by the rest of the pipeline.

    2Wiki canonical shape: ``context = [[title, [sent1, sent2, ...]], ...]``.
    Some HF mirrors flatten this into parallel arrays (``{title: [...],
    sentences: [[...]]}``) to match HotpotQA. We handle both.
    """
    ctx = item.get("context")
    if ctx is None:
        raise ValueError("missing 'context' field; cannot normalize paragraphs")

    paragraphs: list[dict[str, Any]] = []

    # Shape A: HotpotQA-style parallel arrays.
    if isinstance(ctx, dict) and "title" in ctx and "sentences" in ctx:
        titles = ctx["title"]
        sentences_lists = ctx["sentences"]
        for i, (title, sents) in enumerate(zip(titles, sentences_lists)):
            paragraphs.append({
                "paragraph_idx": i,
                "title": str(title),
                "sentences": [str(s) for s in sents],
                "joined_text": "".join(str(s) for s in sents),
            })
        return paragraphs

    # Shape B: canonical 2Wiki list-of-tuples.
    if isinstance(ctx, list):
        for i, entry in enumerate(ctx):
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                title, sents = entry[0], entry[1]
                paragraphs.append({
                    "paragraph_idx": i,
                    "title": str(title),
                    "sentences": [str(s) for s in sents],
                    "joined_text": "".join(str(s) for s in sents),
                })
            elif isinstance(entry, dict) and "title" in entry:
                # Shape C: some mirrors wrap each paragraph as a dict.
                sents = entry.get("sentences") or entry.get("text") or []
                if isinstance(sents, str):
                    sents = [sents]
                paragraphs.append({
                    "paragraph_idx": i,
                    "title": str(entry["title"]),
                    "sentences": [str(s) for s in sents],
                    "joined_text": "".join(str(s) for s in sents),
                })
            else:
                raise ValueError(
                    f"unrecognized paragraph entry shape at index {i}: {type(entry).__name__}; "
                    f"expected [title, [sentences]] tuple or {{title, sentences}} dict"
                )
        return paragraphs

    raise ValueError(f"unrecognized 'context' shape: {type(ctx).__name__}")


def _normalize_supporting_facts(item: dict[str, Any]) -> list[dict[str, Any]]:
    """2Wiki canonical shape: ``supporting_facts = [[title, sent_id], ...]``.
    HotpotQA-style parallel-array form is also handled."""
    sf = item.get("supporting_facts")
    if sf is None:
        return []

    if isinstance(sf, dict) and "title" in sf and "sent_id" in sf:
        titles = sf["title"]
        sent_ids = sf["sent_id"]
        return [
            {"title": str(t), "sent_id": int(s)}
            for t, s in zip(titles, sent_ids)
        ]

    if isinstance(sf, list):
        out: list[dict[str, Any]] = []
        for entry in sf:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                out.append({"title": str(entry[0]), "sent_id": int(entry[1])})
            elif isinstance(entry, dict):
                out.append({
                    "title": str(entry.get("title", "")),
                    "sent_id": int(entry.get("sent_id", 0)),
                })
        return out

    return []


def _load_first_available(
    mirrors: list[str],
    split_candidates: list[str],
) -> tuple[Any, str, str]:
    """Try mirrors in order; for each, try the named splits. Return
    (dataset, mirror_name, split_name) for the first that loads. Raises
    RuntimeError with a per-mirror diagnostic if all fail."""
    from datasets import load_dataset  # type: ignore[import-not-found]

    errors: list[str] = []
    for mirror in mirrors:
        for split_name in split_candidates:
            try:
                print(f"[..] trying {mirror} / split={split_name}")
                ds = load_dataset(mirror, split=split_name, trust_remote_code=False)
                print(f"[ok] loaded {mirror} / split={split_name} ({len(ds)} rows)")
                return ds, mirror, split_name
            except Exception as exc:  # pragma: no cover (network-bound)
                errors.append(f"{mirror}/{split_name}: {type(exc).__name__}: {exc}")
                continue
    msg = "\n  ".join(errors)
    raise RuntimeError(
        f"could not load any 2WikiMultiHopQA mirror. Tried:\n  {msg}\n"
        f"Pass --hf-name <user>/<repo> to override the mirror list."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument(
        "--name", default=None,
        help="Override slice name. Default: 2wikimultihop_distractor_{n}",
    )
    parser.add_argument(
        "--hf-name", default=None,
        help="Override the HF dataset mirror (e.g. 'xanhho/2WikiMultihopQA'). "
             "If unset, tries a list of common mirrors in priority order.",
    )
    parser.add_argument(
        "--hf-split", default=None,
        help="Override the HF source split name. Default tries validation/dev/test.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slice_name = args.name or f"2wikimultihop_distractor_{args.n}"

    print("[..] importing datasets ...")

    mirrors = [args.hf_name] if args.hf_name else list(DEFAULT_MIRRORS)
    split_candidates = [args.hf_split] if args.hf_split else list(DATASET_SPLIT_CANDIDATES)

    ds, chosen_mirror, chosen_split = _load_first_available(mirrors, split_candidates)

    info = ds.info
    dataset_revision = getattr(info, "version", None)
    fingerprint = getattr(ds, "_fingerprint", None)
    n_total = len(ds)
    print(f"[ok] {chosen_mirror}/{chosen_split}: {n_total} rows, fingerprint={fingerprint}")

    if args.n > n_total:
        print(f"[!!] requested n={args.n} > {n_total}; clamping")
        args.n = n_total

    # Materialize lightweight records first so we can stratify on `type`
    # before sampling. Memory is fine -- 12K rows of metadata is < 50 MB.
    raw_records: list[dict[str, Any]] = []
    type_counts: dict[str, int] = {}
    schema_problems: list[str] = []
    for i in range(n_total):
        item = ds[i]
        t = str(item.get("type", "unknown"))
        if t not in KNOWN_TYPES:
            schema_problems.append(f"row {i}: unexpected type='{t}'")
        type_counts[t] = type_counts.get(t, 0) + 1
        raw_records.append({"_source_idx": i, "type": t, "_item": item})

    if schema_problems:
        print(f"[!!] {len(schema_problems)} rows have unexpected `type` values; "
              f"first 5: {schema_problems[:5]}")
        print(f"     observed type distribution: {type_counts}")
    else:
        print(f"[ok] all rows match the 4 known 2Wiki types: {type_counts}")

    by_type = _stratified_indices(raw_records, args.seed)

    # Round-robin draw from each type bucket until we have args.n picks.
    target_n = args.n
    target_per_type = {t: max(1, round(target_n * (len(ids) / n_total))) for t, ids in by_type.items()}
    overshoot = sum(target_per_type.values()) - target_n
    # If rounding overshot, trim from the largest bucket(s); if undershot, add.
    if overshoot > 0:
        for t in sorted(target_per_type, key=lambda k: -target_per_type[k]):
            if overshoot == 0:
                break
            if target_per_type[t] > 1:
                target_per_type[t] -= 1
                overshoot -= 1
    elif overshoot < 0:
        for t in sorted(target_per_type, key=lambda k: -target_per_type[k]):
            if overshoot == 0:
                break
            target_per_type[t] += 1
            overshoot += 1

    print(f"[..] per-type target draw: {target_per_type} (sum={sum(target_per_type.values())})")

    selected_source_indices: list[int] = []
    for t, target in target_per_type.items():
        avail = by_type.get(t, [])
        take = min(target, len(avail))
        selected_source_indices.extend(avail[:take])
    selected_source_indices.sort()

    # Build normalized records.
    records: list[dict[str, Any]] = []
    norm_errors: list[str] = []
    for src_i in selected_source_indices:
        item = raw_records[src_i]["_item"]
        try:
            paragraphs = _normalize_paragraphs(item)
            supporting_facts = _normalize_supporting_facts(item)
        except Exception as exc:
            norm_errors.append(f"row source_idx={src_i}: {type(exc).__name__}: {exc}")
            continue
        records.append({
            "split_idx": int(src_i),
            "id": str(item.get("_id") or item.get("id") or f"row_{src_i}"),
            "question": str(item.get("question", "")),
            "answer": str(item.get("answer", "")),
            "type": str(item.get("type", "")),
            "level": "",  # 2Wiki has no level field; kept empty for schema parity
            "paragraphs": paragraphs,
            "supporting_facts": supporting_facts,
        })

    if norm_errors:
        print(f"[!!] {len(norm_errors)} normalization errors; first 3:")
        for e in norm_errors[:3]:
            print(f"     {e}")
        if len(records) < target_n * 0.9:
            print(f"[FATAL] only {len(records)}/{target_n} rows normalized cleanly. "
                  f"Check the mirror schema with --hf-name <alternative>.")
            sys.exit(2)

    # Write JSONL with hash + manifest.
    out_path = out_dir / f"{slice_name}.jsonl"
    manifest_path = out_dir / f"{slice_name}.manifest.json"

    hasher = hashlib.sha256()
    id_to_split: dict[str, str] = {}
    per_type_in_slice: dict[str, int] = {}
    with out_path.open("w") as f:
        for record in records:
            record_with_split = {"split": SPLIT_TAG, **record}
            line = json.dumps(record_with_split, ensure_ascii=False)
            hasher.update(line.encode("utf-8"))
            hasher.update(b"\n")
            f.write(line + "\n")
            id_to_split[record["id"]] = SPLIT_TAG
            per_type_in_slice[record["type"]] = per_type_in_slice.get(record["type"], 0) + 1

    slice_sha256 = hasher.hexdigest()

    manifest = {
        "dataset_name": chosen_mirror,
        "dataset_config": "default",
        "dataset_split": chosen_split,
        "dataset_version": str(dataset_revision) if dataset_revision is not None else None,
        "dataset_fingerprint": fingerprint,
        "n_total_in_split": n_total,
        "sample_size": len(records),
        "sample_seed": args.seed,
        "sample_strategy": (
            "stratified on `type` (the 4 reasoning types); per-type counts "
            "proportional to source distribution; round-robin draw from "
            "per-type buckets shuffled with seed."
        ),
        "slice_sha256": slice_sha256,
        "slice_path": str(out_path),
        "split": {
            "policy": (
                "single ood_test split (no calibration, no validation). This "
                "slice is evaluated under HotpotQA-frozen thresholds per "
                "_AGENT.md #21 pre-registration; no knobs are tuned on it."
            ),
            "tag": SPLIT_TAG,
            "counts": {SPLIT_TAG: len(records)},
            "per_type_counts_in_slice": per_type_in_slice,
            "per_type_counts_in_source": type_counts,
            "contract": [
                "ood_test: opened under frozen Qwen thresholds + mode from HotpotQA-distractor calibration (#20).",
                "No threshold fitting on this slice. No model selection on this slice. See _AGENT.md #21.",
            ],
            "id_to_split": id_to_split,
        },
        "schema_per_row": [
            "split (str): always 'ood_test'",
            "split_idx (int): row index in the source split",
            "id (str): 2Wiki _id",
            "question (str)",
            "answer (str)",
            "type (str): 'compositional' | 'inference' | 'comparison' | 'bridge_comparison'",
            "level (str): always '' (2Wiki has no difficulty annotation)",
            "paragraphs: [{paragraph_idx, title, sentences, joined_text}, ...]",
            "supporting_facts: [{title, sent_id}, ...]",
        ],
        "preregistration_entry": "_AGENT.md #21 (2026-05-11)",
        "frozen_thresholds_source": "runs/phase4_bootstrap_dragin.json (entry #20)",
    }
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(
        f"[ok] wrote {len(records)} questions to {out_path}\n"
        f"     slice_sha256       = {slice_sha256}\n"
        f"     mirror             = {chosen_mirror}/{chosen_split}\n"
        f"     per-type in slice  = {per_type_in_slice}\n"
        f"     manifest           = {manifest_path}"
    )


if __name__ == "__main__":
    main()
