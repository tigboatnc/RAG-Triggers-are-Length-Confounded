"""Confidence aggregators over per-token logprobs.

Schema (from Phase 2.1):
    response['choices'][0]['logprobs']['token_logprobs']  # list[float]

We expose a few aggregations because Phase 2.1 already showed that a naive
mean is dragged down by trailing format tokens (newlines, BPE spillovers).
Phase 2.4 will pick which one(s) to keep.
"""

from __future__ import annotations

import math
from typing import Any


def _logprobs_block(response: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return response["choices"][0]["logprobs"]
    except (KeyError, IndexError, TypeError):
        return None


def _chosen_logprobs(response: dict[str, Any]) -> list[float]:
    """Works for both raw-completion ('token_logprobs') and chat-completion
    ('content': [{token, logprob, ...}, ...]) schemas."""
    block = _logprobs_block(response)
    if not block:
        return []
    if "token_logprobs" in block:  # raw completion
        return [float(x) for x in block["token_logprobs"] if x is not None]
    if "content" in block:  # chat completion
        return [float(c["logprob"]) for c in block["content"]
                if c.get("logprob") is not None]
    return []


def _chosen_tokens(response: dict[str, Any]) -> list[str]:
    block = _logprobs_block(response)
    if not block:
        return []
    if "tokens" in block:
        return list(block["tokens"])
    if "content" in block:
        return [c["token"] for c in block["content"]]
    return []


def mean_prob(response: dict[str, Any]) -> float | None:
    """Average probability across all generated tokens."""
    lps = _chosen_logprobs(response)
    if not lps:
        return None
    probs = [math.exp(x) for x in lps]
    return sum(probs) / len(probs)


def min_prob(response: dict[str, Any]) -> float | None:
    """Lowest per-token probability. Penalizes any single shaky step."""
    lps = _chosen_logprobs(response)
    if not lps:
        return None
    return math.exp(min(lps))


def first_content_mean(response: dict[str, Any]) -> float | None:
    """Mean probability over the first contiguous run of *content* tokens.

    Strips leading and trailing whitespace-only / format tokens. Captures the
    intuition that the gate cares about the answer, not the formatting.
    """
    tokens = _chosen_tokens(response)
    lps = _chosen_logprobs(response)
    if not tokens or not lps or len(tokens) != len(lps):
        return None

    def is_format(tok: str) -> bool:
        s = tok.strip()
        return s == "" or s in {".", ",", ";", ":", "-", "—", "*"}

    start = 0
    while start < len(tokens) and is_format(tokens[start]):
        start += 1
    end = start
    while end < len(tokens) and not is_format(tokens[end]):
        end += 1
    if end <= start:
        return mean_prob(response)
    window = lps[start:end]
    probs = [math.exp(x) for x in window]
    return sum(probs) / len(probs)


def summarize(response: dict[str, Any]) -> dict[str, float | None]:
    """One-shot: every aggregator + the per-token list, for logging."""
    tokens = _chosen_tokens(response)
    lps = _chosen_logprobs(response)
    probs = [round(math.exp(x), 4) for x in lps]
    return {
        "mean": mean_prob(response),
        "min": min_prob(response),
        "first_content_mean": first_content_mean(response),
        "n_tokens": len(lps),
        "tokens": tokens,
        "probs": probs,
    }
