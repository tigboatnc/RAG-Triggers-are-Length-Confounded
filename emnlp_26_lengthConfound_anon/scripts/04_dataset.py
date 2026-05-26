"""Phase 4.1 dataset assembly: 200-question HotpotQA-distractor slice
with a strict, stratified calibration / validation / test partition baked
into the on-disk file.

We pull the HuggingFace mirror of HotpotQA (config: distractor, validation
split). The distractor split gives, for each question:

  - the question (English, multi-hop, requires combining 2 paragraphs)
  - a list of 10 paragraphs (2 supporting + 8 distractors)
  - the gold short-form answer
  - the supporting-fact (paragraph-title, sentence-idx) annotations

We deterministically sample N questions with a fixed seed, then split
them 60 / 20 / 20 into calibration / validation / test, **stratified on
(level x type)** so each split sees the same difficulty mix.

The split is pinned in two places:

  1. Each row in the JSONL carries a ``split`` field
     (``"calibration" | "validation" | "test"``).
  2. The manifest enumerates the (id -> split) map and records the
     ``slice_sha256`` over the JSONL bytes; downstream scripts can verify
     the file hasn't been mutated.

The split contract (Phase 4.4 leakage prevention):

  - Per-architecture latent thresholds (tau_drift, tau_var), GRIP / ReaLM
    thresholds, and any other knob that gets *fit* may only consult the
    ``calibration`` rows. The same split (not per-arch) applies across all
    three architectural classes so cross-architecture claims stay valid.
  - Layer-mode and boundary-strategy choices, plus other "search-space"
    knobs, may consult ``validation``.
  - ``test`` is opened once at write-up. Anything that touches it before
    the headline numbers are frozen is treated as leakage.

Every downstream Phase 4 script reads from these two files only -- no
live HF calls during generation.

Usage:
    python scripts/04_dataset.py                      # default: 200 q, seed 20260509, 60/20/20
    python scripts/04_dataset.py --n 50               # smoke slice (still 60/20/20 stratified)
    python scripts/04_dataset.py --seed 1337
    python scripts/04_dataset.py --calibration-frac 0.5 --validation-frac 0.25  # override 60/20/20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

DATASET_NAME = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"
DATASET_SPLIT = "validation"
DEFAULT_N = 200
DEFAULT_SEED = 20260509

# 60 / 20 / 20 by default. Calibration is the largest because it's the
# only split allowed to fit thresholds per architecture (Phase 4.4
# threshold-leakage guard); val/test are equal so we don't accidentally
# bias either toward an easier difficulty mix.
DEFAULT_CALIB_FRAC = 0.60
DEFAULT_VAL_FRAC = 0.20
# test_frac = 1 - calib_frac - val_frac (computed)


def _stratified_split(
    items: list[dict[str, Any]],
    calib_frac: float,
    val_frac: float,
    seed: int,
) -> dict[int, str]:
    """Assign each item to one of {calibration, validation, test}, stratified
    on (level, type) so each split sees the same difficulty mix.

    Returns a dict keyed by item index (in the input list order) -> split label.

    Stratification works per (level, type) cell: shuffle items in the cell
    deterministically by ``seed``, then take the first ``round(calib_frac * n)``
    as calibration, the next ``round(val_frac * n)`` as validation, and
    the rest as test. Because we round, the resulting per-cell sizes can
    differ by 1 from the global fractions; the global totals are reported
    in the manifest so any drift is visible.
    """
    test_frac = 1.0 - calib_frac - val_frac
    if test_frac < 0:
        raise ValueError(
            f"calib_frac + val_frac must be <= 1 (got {calib_frac} + {val_frac})"
        )

    by_cell: dict[tuple[str, str], list[int]] = {}
    for i, item in enumerate(items):
        key = (item.get("level", "unknown"), item.get("type", "unknown"))
        by_cell.setdefault(key, []).append(i)

    assignment: dict[int, str] = {}
    for key, ids in by_cell.items():
        # random.Random doesn't accept tuples; derive a per-cell int seed
        # from a sha256 of the cell key + global seed so each cell gets a
        # different shuffle but the whole partition is reproducible from
        # `seed` alone.
        cell_seed = int.from_bytes(
            hashlib.sha256(f"{seed}|{key[0]}|{key[1]}".encode("utf-8")).digest()[:8],
            "big",
        )
        rng = random.Random(cell_seed)
        shuffled = list(ids)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_calib = round(calib_frac * n)
        n_val = round(val_frac * n)
        # Guarantee the per-cell sizes don't exceed n.
        n_val = min(n_val, n - n_calib)
        for j, idx in enumerate(shuffled):
            if j < n_calib:
                assignment[idx] = "calibration"
            elif j < n_calib + n_val:
                assignment[idx] = "validation"
            else:
                assignment[idx] = "test"
    return assignment


def _normalize_paragraphs(item: dict[str, Any]) -> list[dict[str, Any]]:
    """HotpotQA paragraphs are stored as parallel arrays under `context`:

        context.title:     list[str]   (length 10)
        context.sentences: list[list[str]]   (length 10, each inner list = sentences)

    We normalize to a list of {title, sentences, paragraph_idx} dicts.
    """
    titles = item["context"]["title"]
    sentences_lists = item["context"]["sentences"]
    paragraphs: list[dict[str, Any]] = []
    for i, (title, sents) in enumerate(zip(titles, sentences_lists)):
        paragraphs.append({
            "paragraph_idx": i,
            "title": title,
            "sentences": list(sents),
            "joined_text": "".join(sents),
        })
    return paragraphs


def _normalize_supporting_facts(item: dict[str, Any]) -> list[dict[str, Any]]:
    titles = item["supporting_facts"]["title"]
    sent_ids = item["supporting_facts"]["sent_id"]
    return [
        {"title": t, "sent_id": int(s)}
        for t, s in zip(titles, sent_ids)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument(
        "--name",
        default=None,
        help="Override slice name. Default: hotpotqa_distractor_{n}",
    )
    parser.add_argument(
        "--calibration-frac", type=float, default=DEFAULT_CALIB_FRAC,
        help="Fraction of the slice used for threshold calibration (default 0.60).",
    )
    parser.add_argument(
        "--validation-frac", type=float, default=DEFAULT_VAL_FRAC,
        help="Fraction used for layer-mode / boundary-strategy choices (default 0.20). "
             "Test fraction = 1 - calibration - validation.",
    )
    parser.add_argument(
        "--split-seed", type=int, default=None,
        help="Separate seed for the train/val/test partition. Defaults to --seed.",
    )
    parser.add_argument(
        "--exclude-ids-from", action="append", default=[],
        help="Path to another slice JSONL whose `id`s should be excluded "
             "from this sample. May be passed multiple times. Used to build "
             "disjoint slices for v2 (e.g. exclude the v1 200-q calibration set).",
    )
    parser.add_argument(
        "--all-test", action="store_true",
        help="Override the 60/20/20 split: tag every row split='test'. Use when "
             "this slice is purely a test set and calibration came from a prior slice.",
    )
    args = parser.parse_args()
    if args.split_seed is None:
        args.split_seed = args.seed

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slice_name = args.name or f"hotpotqa_distractor_{args.n}"

    print(f"[..] importing datasets ...")
    from datasets import load_dataset

    print(f"[..] loading {DATASET_NAME} (config={DATASET_CONFIG}, split={DATASET_SPLIT})")
    ds = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        trust_remote_code=False,
    )

    # Capture provenance: dataset rev hash from the cached info object.
    info = ds.info
    dataset_revision = getattr(info, "version", None)
    download_checksums = getattr(info, "download_checksums", None) or {}
    fingerprint = ds._fingerprint  # huggingface-internal but stable

    n_total = len(ds)
    print(f"[ok] loaded {n_total} questions, fingerprint={fingerprint}")
    if args.n > n_total:
        print(f"[!!] requested n={args.n} > {n_total}; clamping")
        args.n = n_total

    excluded_ids: set[str] = set()
    for excl_path in args.exclude_ids_from:
        with open(excl_path) as ef:
            for line in ef:
                line = line.strip()
                if not line:
                    continue
                excluded_ids.add(str(json.loads(line)["id"]))
    if excluded_ids:
        print(f"[..] excluding {len(excluded_ids)} ids from {args.exclude_ids_from}")

    rng = random.Random(args.seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    if excluded_ids:
        indices = [i for i in indices if str(ds[i]["id"]) not in excluded_ids]
    selected = sorted(indices[: args.n])

    # Materialize the records first so we can stratify on (level, type)
    # before writing the file. Memory is fine -- 200 questions x ~10 small
    # paragraphs each is < 5 MB.
    records: list[dict[str, Any]] = []
    for idx in selected:
        item = ds[idx]
        records.append({
            "split_idx": int(idx),
            "id": str(item["id"]),
            "question": str(item["question"]),
            "answer": str(item["answer"]),
            "type": str(item.get("type", "")),
            "level": str(item.get("level", "")),
            "paragraphs": _normalize_paragraphs(item),
            "supporting_facts": _normalize_supporting_facts(item),
        })

    if args.all_test:
        split_assignment = ["test"] * len(records)
    else:
        split_assignment = _stratified_split(
            records,
            calib_frac=args.calibration_frac,
            val_frac=args.validation_frac,
            seed=args.split_seed,
        )

    split_counts: dict[str, int] = {"calibration": 0, "validation": 0, "test": 0}
    cell_split_counts: dict[str, dict[str, int]] = {}
    out_path = out_dir / f"{slice_name}.jsonl"
    manifest_path = out_dir / f"{slice_name}.manifest.json"

    hasher = hashlib.sha256()
    id_to_split: dict[str, str] = {}
    with out_path.open("w") as f:
        for i, record in enumerate(records):
            split = split_assignment[i]
            record_with_split = {"split": split, **record}
            line = json.dumps(record_with_split, ensure_ascii=False)
            hasher.update(line.encode("utf-8"))
            hasher.update(b"\n")
            f.write(line + "\n")
            split_counts[split] += 1
            cell_key = f"{record['level']}|{record['type']}"
            cell_split_counts.setdefault(cell_key, {"calibration": 0, "validation": 0, "test": 0})
            cell_split_counts[cell_key][split] += 1
            id_to_split[record["id"]] = split

    slice_sha256 = hasher.hexdigest()
    test_frac = 1.0 - args.calibration_frac - args.validation_frac

    manifest = {
        "dataset_name": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "dataset_split": DATASET_SPLIT,
        "dataset_version": str(dataset_revision) if dataset_revision is not None else None,
        "dataset_fingerprint": fingerprint,
        "download_checksums_keys": list(download_checksums.keys()),
        "n_total_in_split": n_total,
        "sample_size": len(records),
        "sample_seed": args.seed,
        "sample_strategy": "Random.shuffle then take first --n; indices sorted ascending in output",
        "slice_sha256": slice_sha256,
        "slice_path": str(out_path),
        "split": {
            "policy": "stratified on (level, type) per Phase 4.4 leakage guard",
            "fractions": {
                "calibration": args.calibration_frac,
                "validation": args.validation_frac,
                "test": round(test_frac, 6),
            },
            "seed": args.split_seed,
            "counts": split_counts,
            "per_cell_counts": cell_split_counts,
            "contract": [
                "calibration: only split allowed to FIT per-architecture thresholds (tau_drift, tau_var, GRIP / ReaLM thresholds, etc.).",
                "validation: layer-mode and boundary-strategy choices, plus other model-selection knobs.",
                "test: opened ONCE at write-up; never used to fit anything; same partition shared across all architectural classes.",
            ],
            "id_to_split": id_to_split,
        },
        "schema_per_row": [
            "split (str): 'calibration' | 'validation' | 'test'",
            "split_idx (int): row index in the original validation split",
            "id (str): HotpotQA question id",
            "question (str)",
            "answer (str): gold short answer",
            "type (str): 'comparison' | 'bridge' | ...",
            "level (str): 'easy' | 'medium' | 'hard'",
            "paragraphs: [{paragraph_idx, title, sentences, joined_text}, ...]",
            "supporting_facts: [{title, sent_id}, ...]",
        ],
    }
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(
        f"[ok] wrote {len(records)} questions to {out_path}\n"
        f"     split counts  = {split_counts} (frac calib={args.calibration_frac}, val={args.validation_frac}, test={round(test_frac,4)})\n"
        f"     slice_sha256  = {slice_sha256}\n"
        f"     manifest      = {manifest_path}"
    )
    print("     per-(level, type) cell counts:")
    for cell, counts in sorted(cell_split_counts.items()):
        print(f"       {cell:20} -> {counts}")


if __name__ == "__main__":
    main()
