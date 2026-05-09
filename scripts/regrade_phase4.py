"""Offline re-grade Phase 4.1 JSONL outputs with the current extractor.

Reads the existing per-question full-generation tail (and the extracted
answer + strategy), re-runs ``extract_answer`` on a longer-tail-friendly
buffer, recomputes EM and token-F1 vs the gold, and writes the updated
file in place plus a side-by-side diff summary.

Used after the placeholder-echo fix in ``qa_grading.py`` to recover Qwen
labels without re-running the 2 h 20 min generation.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from qa_grading import exact_match, extract_answer, token_f1  # type: ignore[import-not-found]


def regrade(path: Path) -> None:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    n_changed = 0
    em_before = 0
    em_after = 0
    f1_before_sum = 0.0
    f1_after_sum = 0.0
    strat_changes: Counter[tuple[str, str]] = Counter()

    for r in rows:
        if "error" in r:
            continue
        prev_pred = r.get("model_answer_extracted", "")
        prev_strat = r.get("model_answer_extract_strategy", "")
        prev_em = int(r.get("em", 0))
        prev_f1 = float(r.get("f1", 0.0))

        # Re-run extraction on the saved tail (the field we have on disk).
        # Note: we only kept the last 400 chars of the generation, so the
        # re-grade can fail to find an "Answer:" that fell off the start of
        # the tail. That's still strictly better than the old extractor on
        # the same data, since the placeholder-echo filter doesn't depend
        # on context length.
        tail = r.get("model_answer_full_generation_tail", "")
        new_pred, new_strat = extract_answer(tail)
        new_em = exact_match(new_pred, r["gold_answer"])
        new_f1 = token_f1(new_pred, r["gold_answer"])

        em_before += prev_em
        em_after += new_em
        f1_before_sum += prev_f1
        f1_after_sum += new_f1

        if new_pred != prev_pred or new_strat != prev_strat:
            n_changed += 1
            strat_changes[(prev_strat, new_strat)] += 1

        r["model_answer_extracted"] = new_pred
        r["model_answer_extract_strategy"] = new_strat
        r["em"] = new_em
        r["f1"] = round(new_f1, 4)
        r["label_correct_em"] = bool(new_em)
        r["label_correct_f1_50"] = bool(new_f1 >= 0.5)

        # Also recompute fire_rates so the random-rate reflects the
        # (unchanged) latent/realm/grip rates with the same calibration
        # recipe -- but those rates don't change since the underlying
        # gate decisions in `gate_decisions` are unchanged. Skip.

    n_ok = sum(1 for r in rows if "error" not in r)
    print(f"\n{path.name}: {n_ok} rows, {n_changed} changed by re-grade")
    print(f"  EM:  {em_before/n_ok:.4f}  ->  {em_after/n_ok:.4f}   (delta {em_after-em_before:+d}/{n_ok})")
    print(f"  F1:  {f1_before_sum/n_ok:.4f}  ->  {f1_after_sum/n_ok:.4f}")
    if strat_changes:
        print(f"  strategy transitions:")
        for (a, b), c in strat_changes.most_common():
            print(f"    {a:18} -> {b:18}  ({c})")

    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {path}")


def main() -> None:
    runs_dir = Path("runs")
    paths = sorted(runs_dir.glob("phase4_qa_*_mid_stack.jsonl"))
    paths = [p for p in paths if not p.name.endswith("_smoke.jsonl")]
    print(f"[..] re-grading {len(paths)} files")
    for p in paths:
        regrade(p)


if __name__ == "__main__":
    main()
