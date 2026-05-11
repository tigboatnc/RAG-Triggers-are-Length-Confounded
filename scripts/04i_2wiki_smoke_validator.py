"""Phase 4 / #21 §smoke — pre-sweep validator for the 2Wiki OOD transfer.

Reads a 5-question smoke JSONL produced by ``scripts/04f_run_qa_sota.py``
with ``--out-suffix _2wiki_smoke`` and checks the four criteria pinned in
``_AGENT.md`` #21 §smoke:

  1. The 2Wiki paragraphs rendered into Qwen's chat template without
     truncation or schema mismatch (proxied by: every smoke row has a
     non-empty ``question`` and a non-trivial ``prompt_tokens`` count,
     and ``n_steps >= 1`` for at least 4 of 5 rows).
  2. The ``Answer:`` early-stop regex fires on at least 3 of 5 questions
     (``early_stop_reason == "answer_marker"`` OR ``answer_marker_seen_at``
     is not None).
  3. The DRAGIN attention layer auto-detect picked the correct tuple
     index (should be **9** for Qwen3.6 hybrid stack per #18 §T2).
  4. Every per-step record carries non-None values for the five scalars
     the OOD analyzer reads: ``drift_cosine``, ``variance_l2_window``,
     ``mean_token_entropy``, ``hvr``, ``dragin_rind_max``.

Exits non-zero if any check fails. The compute-box wrapper
(``scripts/run_2wiki_ood.sh``) refuses to start the 200-q sweep if the
exit code is non-zero.

Usage:
    python scripts/04i_2wiki_smoke_validator.py
    python scripts/04i_2wiki_smoke_validator.py --traces runs/<custom>.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_TRACES = "runs/phase4_qa_qwen3.6-35b-a3b-instruct_mid_stack_2wiki_smoke.jsonl"
EXPECTED_DRAGIN_TUPLE_IDX_QWEN36 = 9  # see _AGENT.md #18 §T2 / #19 §setup

REQUIRED_STEP_FIELDS = [
    "drift_cosine",
    "variance_l2_window",
    "mean_token_entropy",
    "hvr",
    "dragin_rind_max",
]


def _load(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing trace file {path}")
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def check_render(rows: list[dict]) -> tuple[bool, str]:
    """Criterion 1: chat-template render + boundary detection produces a
    non-trivial trace on at least 4 of 5 rows."""
    n_total = len(rows)
    if n_total == 0:
        return False, "no rows in trace file"
    n_with_steps = sum(1 for r in rows if int(r.get("n_steps", 0)) >= 1)
    n_with_question = sum(1 for r in rows if str(r.get("question", "")).strip())
    n_with_prompt = sum(1 for r in rows if int(r.get("prompt_tokens", 0)) > 0)
    threshold = max(1, int(0.8 * n_total))
    ok = n_with_steps >= threshold and n_with_question == n_total and n_with_prompt == n_total
    detail = (
        f"{n_with_steps}/{n_total} rows have n_steps >= 1; "
        f"{n_with_question}/{n_total} have non-empty question; "
        f"{n_with_prompt}/{n_total} have prompt_tokens > 0"
    )
    return ok, detail


def check_answer_marker(rows: list[dict]) -> tuple[bool, str]:
    """Criterion 2: at least 3 of 5 rows hit the Answer: early-stop regex."""
    n_total = len(rows)
    if n_total == 0:
        return False, "no rows"
    n_hit = sum(
        1 for r in rows
        if r.get("early_stop_reason") == "answer_marker"
        or r.get("answer_marker_seen_at") is not None
    )
    threshold = max(1, min(3, int(0.6 * n_total)))
    return n_hit >= threshold, f"{n_hit}/{n_total} rows hit Answer: regex (need >= {threshold})"


def check_dragin_layer(rows: list[dict]) -> tuple[bool, str]:
    """Criterion 3: the auto-detected DRAGIN tuple index is 9 (Qwen3.6 hybrid).

    The driver records this either at the top level (per-question repetition
    of a sweep-level config) or under a config sub-dict; we look in both
    places to be format-agnostic.
    """
    candidates: list[int] = []
    for r in rows:
        for key in ("dragin_attn_layer_idx", "dragin_layer_idx"):
            v = r.get(key)
            if v is None:
                cfg = r.get("config") or {}
                v = cfg.get(key)
            if isinstance(v, int):
                candidates.append(v)
                break
    if not candidates:
        return False, "no row recorded a DRAGIN attention-layer index"
    unique = set(candidates)
    if unique != {EXPECTED_DRAGIN_TUPLE_IDX_QWEN36}:
        return False, (
            f"expected tuple index {EXPECTED_DRAGIN_TUPLE_IDX_QWEN36}, "
            f"got {sorted(unique)} across {len(candidates)} rows"
        )
    return True, f"tuple index {EXPECTED_DRAGIN_TUPLE_IDX_QWEN36} on {len(candidates)} rows"


def check_step_fields(rows: list[dict]) -> tuple[bool, str]:
    """Criterion 4: every per-step record has non-None values for the five
    scalars the OOD analyzer relies on."""
    n_total_steps = 0
    n_missing: dict[str, int] = {f: 0 for f in REQUIRED_STEP_FIELDS}
    for r in rows:
        steps = r.get("steps") or []
        for s in steps:
            n_total_steps += 1
            for f in REQUIRED_STEP_FIELDS:
                if s.get(f) is None:
                    n_missing[f] += 1
    if n_total_steps == 0:
        return False, "zero per-step records across all rows"
    any_missing = any(v > 0 for v in n_missing.values())
    detail = (
        f"{n_total_steps} step records; missing-counts = "
        + ", ".join(f"{k}={v}" for k, v in n_missing.items())
    )
    return (not any_missing), detail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", default=DEFAULT_TRACES)
    parser.add_argument("--strict", action="store_true",
                        help="Treat any [warn] as [FAIL].")
    args = parser.parse_args()

    path = Path(args.traces)
    print(f"[..] 2Wiki OOD smoke validator")
    print(f"     traces = {path}")
    print()

    try:
        rows = _load(path)
    except Exception as exc:
        print(f"[FAIL] could not load traces: {exc}")
        return 2

    checks = [
        ("1. chat-template render + boundary detection", check_render(rows)),
        ("2. Answer: early-stop regex fires",            check_answer_marker(rows)),
        ("3. DRAGIN attention tuple index",              check_dragin_layer(rows)),
        ("4. per-step features populated",               check_step_fields(rows)),
    ]

    any_fail = False
    for label, (ok, detail) in checks:
        tag = "[ok]  " if ok else "[FAIL]"
        if not ok:
            any_fail = True
        print(f"  {tag} {label}")
        print(f"         {detail}")
    print()

    if any_fail:
        print("[FAIL] one or more smoke criteria failed. Do NOT launch the 200-q sweep.")
        print("       Copy this output into a new _AGENT.md entry under #21 §smoke-fail.")
        return 1
    print(f"[ok] all 4 smoke criteria passed on {len(rows)} rows. Proceed to the 200-q sweep.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
