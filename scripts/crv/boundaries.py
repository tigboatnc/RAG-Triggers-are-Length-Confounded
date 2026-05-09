"""Step-boundary detector (Phase 2.2).

A "logical step" in the paper is the unit at which the model should be
checked for drift / hallucination. Three operational definitions to sweep:

  - sentence    — boundary fires on tokens that decode to (or end with)
                  one of: '.', '!', '?', '\\n', or the various unicode
                  variants that appear in chat / reasoning models.
  - think_block — boundary fires on tokens decoded to '\\n\\n' OR on
                  the special `<think>` / `</think>` markers when the
                  model uses them. Approximates "end of a reasoning step".
  - fixed       — boundary every `n_tokens` regardless of content. Useful
                  as a calibration baseline (no semantic structure).

Per `_PLAN.md` Phase 2.2 + Phase 4.4 ablation (c), this is meant to be
swept; pick a strategy at construction time and run multiple variants in
parallel for comparison.

Implementation note: detection is per-token (called after each generated
token) so it's O(1) per token and adds zero throughput overhead. The
sentence-end token-id set is precomputed once at init from the tokenizer.
"""

from __future__ import annotations

from typing import Any


class StepBoundaryDetector:
    """Decide whether the just-generated token completes a logical step.

    Parameters
    ----------
    strategy
        One of "sentence", "think_block", "fixed". See module docstring.
    n_tokens
        Stride for "fixed" strategy. Default 64. Ignored for other strategies.
    tokenizer
        HF tokenizer instance. Required for "sentence" and "think_block"
        to precompute the marker-token id set.
    extra_marker_strings
        Optional iterable of additional sentinel strings whose token-ids
        should be treated as boundaries. Useful e.g. to add `</think>`.
    """

    SENTENCE_MARKERS: tuple[str, ...] = (
        ".", "!", "?", "\n", "\n\n",
        "。", "？", "！",  # chinese / japanese sentence-end punctuation
        ".\n", "!\n", "?\n",
    )
    THINK_BLOCK_MARKERS: tuple[str, ...] = (
        "\n\n", "</think>", "<think>",
        # Common reasoning-model section delimiters:
        "\n###", "\n##", "\n**",
    )

    def __init__(
        self,
        strategy: str = "sentence",
        n_tokens: int = 64,
        tokenizer: Any | None = None,
        extra_marker_strings: tuple[str, ...] | None = None,
    ) -> None:
        if strategy not in {"sentence", "think_block", "fixed"}:
            raise ValueError(f"unknown strategy: {strategy!r}")
        self.strategy = strategy
        self.n_tokens = max(1, int(n_tokens))
        self.tokenizer = tokenizer
        self._token_count = 0
        self._step_count = 0
        self._marker_ids: set[int] = set()
        if strategy in {"sentence", "think_block"}:
            if tokenizer is None:
                raise ValueError(
                    f"strategy={strategy!r} requires a tokenizer for marker-id precomputation"
                )
            markers = (
                self.SENTENCE_MARKERS if strategy == "sentence" else self.THINK_BLOCK_MARKERS
            )
            if extra_marker_strings:
                markers = tuple(markers) + tuple(extra_marker_strings)
            self._marker_ids = self._compute_marker_ids(tokenizer, markers)

    @staticmethod
    def _compute_marker_ids(tokenizer: Any, markers: tuple[str, ...]) -> set[int]:
        """Encode each marker string a few ways (raw, with leading space, etc.)
        and return the set of single-token ids that decode to a sentence
        boundary. Multi-token encodings are skipped — we only care about the
        moment a single decoded token matches a boundary."""
        ids: set[int] = set()
        for s in markers:
            for encode_form in (s, " " + s, "\n" + s):
                try:
                    encoded = tokenizer.encode(encode_form, add_special_tokens=False)
                except Exception:
                    continue
                # Single-token-id markers fire cleanly per-step.
                if len(encoded) == 1:
                    ids.add(int(encoded[0]))
                # Some tokenizers may produce a leading "space" token + the punct
                # as second token; capture that second id too.
                elif len(encoded) == 2:
                    ids.add(int(encoded[1]))
        return ids

    def is_boundary(self, token_id: int, generated_text_so_far: str | None = None) -> bool:
        """Called after each token is generated. Returns True if this token
        completes a logical step. Side-effect-free — caller is responsible
        for advancing CRVTracker etc."""
        self._token_count += 1
        fired: bool

        if self.strategy == "fixed":
            fired = (self._token_count % self.n_tokens) == 0
        elif self.strategy in {"sentence", "think_block"}:
            fired = int(token_id) in self._marker_ids
        else:
            raise RuntimeError(f"unreachable: strategy={self.strategy!r}")

        if fired:
            self._step_count += 1
        return fired

    def reset(self) -> None:
        self._token_count = 0
        self._step_count = 0

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "n_tokens": self.n_tokens,
            "tokens_seen": self._token_count,
            "steps_committed": self._step_count,
            "marker_id_count": len(self._marker_ids),
        }
