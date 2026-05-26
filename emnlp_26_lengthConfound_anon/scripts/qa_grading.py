"""HotpotQA-standard answer normalization + EM + token-F1, plus an
answer-extraction heuristic for free-form model output.

The normalize_answer / token-F1 implementation is a transcription of the
official ``hotpot_evaluate_v1.py`` recipe (which is itself a SQuAD-1.1
descendant): lowercase, strip articles, collapse whitespace, drop
punctuation, then F1 over whitespace tokens. This stays inline rather
than depending on the ``evaluate`` package so the pipeline runs offline
and the grading is byte-stable across runs.

Answer extraction tries, in order:

  1. The text after the last "Answer:" / "Final answer:" / "Final
     Answer:" / "## Answer" marker, up to the next blank line / end of
     output / closing markdown fence.
  2. The text after a "</think>" close tag, if present and not empty.
  3. The last non-empty line.
  4. The whole generation.

The strategy that fired is reported alongside the extracted string so we
can audit what fraction of questions extracted via fallback. (Phase 4.3
relies on this label; if fallback rate is high we may want to tighten
the prompt.)
"""

from __future__ import annotations

import re
import string
from collections import Counter


# ---- HotpotQA-standard normalization & metrics -------------------------------

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})
_WS_RE = re.compile(r"\s+")


def normalize_answer(s: str) -> str:
    """HotpotQA-standard answer normalization."""
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def exact_match(prediction: str, gold: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    """Whitespace-token F1 over normalized answers (HotpotQA-standard).

    Returns 0 for either side empty unless both are empty (then 1.0 by
    convention, matching the official script's edge case)."""
    p = normalize_answer(prediction).split()
    g = normalize_answer(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p)
    recall = num_same / len(g)
    return 2 * precision * recall / (precision + recall)


# ---- Answer extraction --------------------------------------------------------

_ANSWER_MARKERS = (
    "final answer:",
    "final_answer:",
    "answer:",
    "## answer",
    "**answer**",
    "**final answer**",
)


_LOWER_UPPER_SEAM_RE = re.compile(r"([a-z\)\],])([A-Z])")
_MAX_ANSWER_CHARS = 100  # HotpotQA gold answers are 1-5 words; 100 chars is generous


def _strip_trailing_garbage(s: str) -> str:
    """Trim closing markdown fences / common end-of-turn tokens / trailing
    whitespace from an extracted answer span, then aggressively truncate
    where the model glued its answer onto continuation text without a
    separator (the "NewportYour response..." pattern observed on small
    open-source models that loop past their answer)."""
    s = s.strip()
    for tail in ("```", "</s>", "<|endoftext|>", "<|im_end|>", "<|end|>"):
        if s.endswith(tail):
            s = s[: -len(tail)].strip()
    # First non-empty line only.
    for line in s.splitlines():
        if line.strip():
            s = line.strip()
            break
    # Truncate at first sentence-end punctuation followed by space/end.
    sent_end = re.search(r"[.!?](\s|$)", s)
    if sent_end:
        s = s[: sent_end.start() + 1]
    # Truncate at first lowercase->uppercase seam (concatenation artefact).
    # Skip the first char so a sentence that legitimately starts with a
    # capital ("Greenwich Village, New York City") isn't split. Split is
    # only applied if the seam is past char 1, since proper text uses a
    # space between capitalized words.
    seam = _LOWER_UPPER_SEAM_RE.search(s, 1)
    if seam is not None:
        s = s[: seam.start() + 1]
    # Hard cap.
    if len(s) > _MAX_ANSWER_CHARS:
        s = s[:_MAX_ANSWER_CHARS]
    return s.strip().rstrip(".").strip()


def extract_answer(generation: str) -> tuple[str, str]:
    """Try to find the model's final answer inside ``generation``.

    Returns ``(answer_string, strategy)`` where ``strategy`` is one of:
    "answer_marker", "think_close", "last_line", "fallback_full".
    """
    if not generation:
        return "", "fallback_full"

    text = generation
    text_lower = text.lower()

    # Find every "Answer:" / "Final answer:" / etc. marker, then pick the
    # last one whose payload doesn't immediately start with a template
    # placeholder character ("<your answer>", "<X>", "[answer]"). Some
    # reasoning models (notably Qwen3.6) self-quote the prompt's placeholder
    # text as part of meta-reflection ("- Check constraints: 'Answer: <your
    # answer>' ..."); those occurrences should be skipped.
    candidates: list[tuple[int, int]] = []
    for marker in _ANSWER_MARKERS:
        start = 0
        while True:
            pos = text_lower.find(marker, start)
            if pos == -1:
                break
            candidates.append((pos, len(marker)))
            start = pos + 1
    candidates.sort()  # ascending by position

    def _is_placeholder_echo(span: str) -> bool:
        s = span.lstrip()
        return s.startswith("<") or s.startswith("[")

    real_pos = -1
    real_len = 0
    for pos, mlen in candidates:
        span = text[pos + mlen:]
        if _is_placeholder_echo(span):
            continue
        cleaned = _strip_trailing_garbage(span)
        if cleaned:
            real_pos = pos
            real_len = mlen  # don't break; we want the last "real" marker
    if real_pos != -1:
        span = text[real_pos + real_len:]
        cleaned = _strip_trailing_garbage(span)
        if cleaned:
            return cleaned, "answer_marker"

    think_pos = text_lower.rfind("</think>")
    if think_pos != -1:
        tail = text[think_pos + len("</think>") :]
        cleaned = _strip_trailing_garbage(tail)
        if cleaned:
            return cleaned, "think_close"

    for line in reversed(text.splitlines()):
        if line.strip():
            return _strip_trailing_garbage(line), "last_line"

    return _strip_trailing_garbage(text), "fallback_full"


__all__ = [
    "normalize_answer",
    "exact_match",
    "token_f1",
    "extract_answer",
]
