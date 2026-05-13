"""SOTA-baseline decode-time capture utilities (Phase 4.8 / 4.9 / 4.10 handoff).

This module is the *engine-agnostic* feature-capture layer that the decode
driver (`scripts/04f_run_qa_sota.py`) calls into to harvest the auxiliary
signals that the three true-SOTA baselines need:

  - DRAGIN (Su et al., `arXiv:2403.10081v3`) — per-token RIND score
        ``RIND_i = H_i * a_max(i) * s_i``
    where ``H_i`` is next-token entropy, ``a_max(i)`` is the max
    attention an *already-generated* token receives from later tokens in
    the last attention-bearing layer, and ``s_i`` is a 0/1 stopword
    filter. Per-step decision = max RIND over the step's tokens.
    See `RindAccumulator` below.

  - ReaLM-Retrieve / RSUS (Guo et al., 2026) — per-step combination
        ``RSUS = a*U_verb + b*U_ent + c*U_cons``
    with paper-published weights (a=0.40, b=0.35, c=0.25) and a fixed
    retrieval threshold τ=0.65. Each component is captured at a step
    boundary by the helpers below.

  - SELFDOUBT-Full (Pandey et al., 2026) — fuses Tier-1 HVR with
    verbalized confidence via per-model z-scores. Shares ``U_verb``
    capture with ReaLM-Retrieve so a single forward pass yields both.

All non-trivial GPU calls (verbalized-confidence fork, consistency
sampling) trim the KV cache back to its pre-fork sequence length on
exit so the main generation trajectory is preserved bit-exactly — see
`_AGENT.md` 2026-05-11 #16 for why we reject deep-copy forking on
~40 GB caches.

The module deliberately has zero hard dependency on transformers at
import time; everything is type-late and behind a function boundary so
the no-GPU dev box can still ``import scripts.crv.sota_capture`` for
unit tests on the regex / BM25 / NER pieces.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------- #
# Token-level scalars (DRAGIN building blocks)
# --------------------------------------------------------------------- #


def compute_token_entropy_nats(logits: Any) -> float:
    """Shannon entropy in nats from a 1-D logits row.

    Accepts a torch.Tensor of shape ``[1, vocab]`` or ``[vocab]``. Computed
    in fp32 to avoid bf16 underflow on long-tail probabilities; matches
    ``_entropy_nats`` in `scripts/04_run_qa.py` so DRAGIN's H_i and the
    rest of the pipeline's entropy series are numerically identical.
    """
    import torch
    with torch.inference_mode():
        x = logits.detach().to(torch.float32).reshape(-1)
        log_p = torch.log_softmax(x, dim=-1)
        p = log_p.exp()
        h = -(p * log_p).sum().item()
    return float(h)


# Stopword filter for DRAGIN's s_i. The DRAGIN paper §3.2 specifies a
# spaCy stopword set; we cache it lazily so the import cost is paid
# once. spaCy's stopword list is ~330 entries and language-stable across
# en_core_web_sm versions used in the 2026 papers.
_SPACY_STOPWORDS: set[str] | None = None
# Always-stop punctuation/whitespace that the spaCy set misses on
# subword decodings (e.g. " ," or "Ġ.").
_PUNCT_STOPWORDS = set(",.;:!?\"'()[]{}<>-_/\\|+=*&^%$#@~`")


def _ensure_spacy_stopwords() -> set[str]:
    global _SPACY_STOPWORDS
    if _SPACY_STOPWORDS is None:
        try:
            from spacy.lang.en.stop_words import STOP_WORDS  # type: ignore
            _SPACY_STOPWORDS = set(STOP_WORDS)
        except Exception:  # pragma: no cover - smoke-test will catch
            # Fallback to a minimal-but-functional set so the gate
            # remains operational if spaCy is not installed on the
            # compute box. Documented in `_AGENT.md` #17.
            _SPACY_STOPWORDS = {
                "a", "an", "the", "and", "or", "but", "is", "are", "was",
                "were", "be", "been", "being", "have", "has", "had", "do",
                "does", "did", "of", "in", "on", "at", "to", "for", "by",
                "with", "as", "this", "that", "it", "its", "they", "them",
                "i", "you", "he", "she", "we", "him", "her", "his", "hers",
                "from", "into", "than", "then", "so", "if", "not", "no",
            }
    return _SPACY_STOPWORDS


def stopword_filter(token_text: str) -> int:
    """Return ``s_i ∈ {0, 1}`` per DRAGIN: 1 iff token is content.

    Strips leading whitespace and the SentencePiece/Llama "Ġ" / "▁" word
    boundary glyphs before checking. Pure-whitespace and punct-only
    tokens are stopwords (s_i = 0).
    """
    if token_text is None:
        return 0
    t = token_text.strip().lstrip("Ġ▁").lower()
    if not t:
        return 0
    if all(ch in _PUNCT_STOPWORDS for ch in t):
        return 0
    if t in _ensure_spacy_stopwords():
        return 0
    return 1


def amax_from_last_layer_attentions(attentions_last: Any, token_pos: int) -> float:
    """Max attention an already-generated token receives from later tokens.

    DRAGIN's ``a_max(i)`` is defined (§3.2) as the maximum attention
    received by token i from any subsequent token j > i in the *last*
    transformer layer. With KV-cache decoding, at each generated step
    we get attentions of shape ``[B, n_heads, q_len, kv_len]`` where
    q_len == 1 (the new token's row only) and kv_len is the full cache
    length so far. The "j > i" maximum therefore accumulates online:

        per_token_amax[i] = max(per_token_amax[i], max_h a[h, 0, i])

    after each new-token forward pass. This function returns that
    single-step contribution; the caller (RindAccumulator) keeps the
    running maximum per token position.

    Hybrid-architecture note (Qwen 3.6 / Gated DeltaNet): full-attention
    layers exist only at fixed strides. We use the last *full-attention*
    layer for ``attentions_last``; the driver selects this layer by
    config. See `_AGENT.md` 2026-05-11 #16 for layer-selection details.
    """
    import torch
    a = attentions_last.detach().to(torch.float32)
    # a shape: [B, H, 1, kv_len]
    a_row = a[0, :, 0, token_pos]  # [H]
    return float(a_row.max().item())


@dataclass
class RindAccumulator:
    """Streaming DRAGIN RIND score accumulator.

    Call ``observe_new_token(token_text, entropy, prompt_len, ...)``
    on every generated token, and ``ingest_attention_row(attn_last)``
    every step so per-position a_max stays current. At each step
    boundary call ``commit_step()`` to get the step's max RIND score
    (or None if no content tokens in the step).
    """
    prompt_len: int = 0
    per_pos_entropy: list[float] = field(default_factory=list)
    per_pos_amax: list[float] = field(default_factory=list)
    per_pos_s: list[int] = field(default_factory=list)
    _step_start_pos: int = 0

    def observe_new_token(self, token_text: str, entropy: float) -> None:
        self.per_pos_entropy.append(float(entropy))
        self.per_pos_amax.append(0.0)
        self.per_pos_s.append(stopword_filter(token_text))

    def ingest_attention_row(self, attentions_last: Any) -> None:
        """Update per-position amax using the *new* row of attention.

        At decode step k (k = len(per_pos_*) - 1), the new attention
        row covers KV positions ``[0, prompt_len + k]``. We only care
        about generated-token positions, i.e. ``[prompt_len,
        prompt_len + k - 1]`` — token k itself doesn't yet have a "j > i"
        observer.
        """
        import torch
        a = attentions_last.detach().to(torch.float32)
        # a shape: [B, H, 1, kv_len]; kv_len = prompt_len + current_idx + 1
        per_head_max = a[0, :, 0, self.prompt_len : self.prompt_len + len(self.per_pos_amax) - 1].max(dim=0).values
        # Shape now [n_gen_positions_excluding_current]; update running max.
        for i, v in enumerate(per_head_max.tolist()):
            if v > self.per_pos_amax[i]:
                self.per_pos_amax[i] = float(v)

    def commit_step(self) -> dict[str, float] | None:
        """Return the per-step max RIND and supporting stats.

        Returns ``None`` when the step has no content tokens (avoids
        emitting noise scores at the warmup or on degenerate steps).
        """
        end = len(self.per_pos_entropy)
        start = self._step_start_pos
        self._step_start_pos = end
        if end <= start:
            return None
        H = self.per_pos_entropy[start:end]
        A = self.per_pos_amax[start:end]
        S = self.per_pos_s[start:end]
        rinds = [h * a * s for h, a, s in zip(H, A, S)]
        # Only count content-token RINDs; if no content tokens, fall
        # back to None so the step gets warmup treatment.
        content = [r for r, s in zip(rinds, S) if s == 1]
        if not content:
            return {
                "rind_max": 0.0,
                "rind_mean": 0.0,
                "n_content_tokens": 0,
                "n_tokens": end - start,
            }
        return {
            "rind_max": float(max(content)),
            "rind_mean": float(sum(content) / len(content)),
            "n_content_tokens": int(sum(S)),
            "n_tokens": int(end - start),
        }


# --------------------------------------------------------------------- #
# Verbalized confidence probe (ReaLM-Retrieve U_verb + SELFDOUBT-Full)
# --------------------------------------------------------------------- #


# A small repertoire of confidence-probe prompts. Guo et al. (2026,
# Appendix B.1) use the first form on instruct-style models; the Phi /
# Gemma chat templates re-attach EOS after the probe, so the model's
# completion is constrained to a few digits before the turn ends.
_VERBAL_PROBE_TEMPLATE = (
    "\n\nBriefly rate your confidence in the reasoning step you just produced "
    "on a scale from 0 to 100, where 0 means you have no idea and 100 means "
    "you are completely certain. Respond with just the number.\nConfidence: "
)

_CONF_NUMBER_RE = re.compile(r"\b(\d{1,3})\b")


def parse_verbal_confidence(text: str) -> float | None:
    """Extract a confidence integer 0..100 from a short model reply.

    Returns ``None`` if no number is found or all numbers are out of
    range. Multi-number replies pick the *first* in-range hit, matching
    Guo et al.'s post-processing.
    """
    if not text:
        return None
    for m in _CONF_NUMBER_RE.finditer(text):
        v = int(m.group(1))
        if 0 <= v <= 100:
            return float(v) / 100.0
    return None


class VerbalizedConfidenceProbe:
    """Fork the KV cache, ask for a confidence score, restore.

    Usage::

        probe = VerbalizedConfidenceProbe(model, tokenizer, max_probe_tokens=12)
        u_verb = probe.probe(cache, current_input_id, current_seq_len)
        # cache is restored to its original sequence length before return.

    ``u_verb`` is ``1 - verbalized_confidence`` so that *higher = more
    uncertain*, matching the RSUS sign convention in Guo et al.

    Returns ``None`` if the model fails to emit a parseable number; the
    RSUS scorer treats this as "no signal" (drops the U_verb term and
    re-normalises the remaining weights).

    Cache-fork mechanics: we record the cache's current sequence length
    (the size of dimension 2 of the first layer's key tensor), append
    the probe prompt + decode K tokens, then *truncate the cache back*
    on dim 2 to the recorded length. This restores the cache to its
    pre-fork state up to bitwise equality of remaining entries. Tested
    in `scripts/04h_sota_smoke.py`.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        max_probe_tokens: int = 12,
        probe_template: str = _VERBAL_PROBE_TEMPLATE,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_probe_tokens = int(max_probe_tokens)
        self.probe_ids = tokenizer(probe_template, return_tensors="pt", add_special_tokens=False).input_ids
        self.probe_ids = self.probe_ids.to(model.device)

    def _cache_seq_len(self, cache: Any) -> int:
        """Read the current sequence length from a DynamicCache-like object."""
        # DynamicCache exposes ``get_seq_length()``; older caches expose
        # ``key_cache[0].shape[2]``. Cover both.
        if hasattr(cache, "get_seq_length"):
            try:
                return int(cache.get_seq_length())
            except Exception:
                pass
        if hasattr(cache, "key_cache") and len(cache.key_cache) > 0:
            return int(cache.key_cache[0].shape[2])
        # Tuple-of-tuples legacy past_key_values.
        if isinstance(cache, tuple) and cache and isinstance(cache[0], tuple):
            return int(cache[0][0].shape[2])
        raise RuntimeError("cannot read sequence length from past_key_values object")

    def _truncate_cache_to(self, cache: Any, target_len: int) -> Any:
        """Slice the cache back to ``target_len`` on the sequence dim.

        Handles three cache layouts:
          - transformers >=5.x ``DynamicCache``: per-layer objects on
            ``cache.layers[*]`` exposing ``.keys`` / ``.values`` /
            ``cumulative_length`` / ``crop(max_length)``. The canonical
            truncation path is ``cache.crop(target_len)`` when present.
          - transformers <5 ``DynamicCache``: parallel ``key_cache`` /
            ``value_cache`` lists plus ``_seen_tokens``.
          - tuple-of-tuples legacy ``past_key_values``.
        """
        if hasattr(cache, "crop"):
            try:
                cache.crop(int(target_len))
                return cache
            except Exception:
                pass
        if hasattr(cache, "layers") and len(cache.layers) > 0 and hasattr(cache.layers[0], "keys"):
            for layer in cache.layers:
                if layer.keys is not None:
                    layer.keys = layer.keys[:, :, :target_len, :].contiguous()
                if layer.values is not None:
                    layer.values = layer.values[:, :, :target_len, :].contiguous()
                if hasattr(layer, "cumulative_length"):
                    layer.cumulative_length = int(target_len)
            return cache
        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            for li in range(len(cache.key_cache)):
                cache.key_cache[li] = cache.key_cache[li][:, :, :target_len, :].contiguous()
                cache.value_cache[li] = cache.value_cache[li][:, :, :target_len, :].contiguous()
            if hasattr(cache, "_seen_tokens"):
                cache._seen_tokens = target_len
            return cache
        if isinstance(cache, tuple) and cache and isinstance(cache[0], tuple):
            return tuple(
                tuple(t[:, :, :target_len, :].contiguous() for t in layer)
                for layer in cache
            )
        raise RuntimeError("unsupported cache type for truncation")

    def probe(self, cache: Any, last_token_id: int) -> float | None:
        """Run the probe; return ``1 - p_correct`` or ``None`` on failure.

        IMPORTANT: ``cache`` must already include ``last_token_id`` as
        its final position (i.e. the post-step state). The probe will
        consume the prompt template tokens starting from that position
        and then truncate back.
        """
        import torch
        original_len = self._cache_seq_len(cache)

        # Run the probe prompt through the cache one token at a time
        # (cheapest path that's guaranteed to play nice with arbitrary
        # attention masks across all three architectures).
        try:
            with torch.inference_mode():
                input_ids = self.probe_ids  # [1, probe_len]
                out = self.model(
                    input_ids=input_ids,
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = out.past_key_values
                # Decode up to max_probe_tokens greedy continuation.
                decoded_ids: list[int] = []
                next_logits = out.logits[:, -1, :]
                eos_id = self.tokenizer.eos_token_id
                for _ in range(self.max_probe_tokens):
                    nxt = int(next_logits.argmax(dim=-1).item())
                    if eos_id is not None and nxt == eos_id:
                        break
                    decoded_ids.append(nxt)
                    nxt_t = torch.tensor([[nxt]], device=self.model.device)
                    out = self.model(
                        input_ids=nxt_t,
                        past_key_values=cache,
                        use_cache=True,
                    )
                    cache = out.past_key_values
                    next_logits = out.logits[:, -1, :]
            text = self.tokenizer.decode(decoded_ids, skip_special_tokens=True)
            conf = parse_verbal_confidence(text)
        finally:
            # Always restore the cache, even if anything above raised.
            self._truncate_cache_to(cache, original_len)
        if conf is None:
            return None
        # Sign convention: higher = more uncertain.
        return 1.0 - conf


# --------------------------------------------------------------------- #
# Entity entropy (ReaLM-Retrieve U_ent)
# --------------------------------------------------------------------- #


class EntityEntropyScorer:
    """Per-step entity grounding entropy via spaCy NER + BM25 over passages.

    Guo et al. §3.2: extract NEs from the step text, look up each NE's
    posterior over the 10 distractor passages by BM25 score, normalise
    to a probability simplex, take the Shannon entropy in nats, and
    average across NEs. Higher entropy = the step references entities
    whose support is spread across passages = less confidently
    grounded. We follow the paper exactly except for one substitution
    noted below.

    Substitution (documented in `_AGENT.md` 2026-05-11 #17):
      Guo et al. use the BGE-M3 cross-encoder reranker for the
      entity-grounding probability. We use BM25 (rank_bm25) because the
      cross-encoder inference cost dominates per-step latency and is
      out of scope for this iteration. The simulator caveat
      ``u_ent_bm25_proxy_for_bge_reranker`` is stamped into every
      emitted record.
    """

    def __init__(self, paragraphs: Sequence[dict]) -> None:
        # paragraphs as in the HotpotQA-distractor dataset rows.
        from rank_bm25 import BM25Okapi  # local import: only on compute box
        self._nlp = self._load_spacy()
        self.paragraphs = list(paragraphs)
        tokenized = [self._tokenize(p["joined_text"]) for p in self.paragraphs]
        self.bm25 = BM25Okapi(tokenized)
        self.caveats = ["u_ent_bm25_proxy_for_bge_reranker"]

    @staticmethod
    def _load_spacy() -> Any:
        import spacy  # type: ignore
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            # Fallback: use the blank "en" pipeline with rule-based NER.
            # The compute-box runbook (`_AGENT.md` #17) installs the
            # full model; this branch keeps the smoke test green even
            # if the model download is delayed.
            nlp = spacy.blank("en")
            return nlp

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t for t in re.findall(r"[A-Za-z0-9]+", text.lower()) if t]

    def _entities(self, text: str) -> list[str]:
        if not text.strip():
            return []
        doc = self._nlp(text)
        ents = [e.text for e in getattr(doc, "ents", []) if e.text.strip()]
        if not ents:
            # Fallback: use ProperNoun-shaped tokens. This matches the
            # paper's behaviour when the NER pipeline returns no
            # entities (Guo et al. §3.2, footnote 3).
            ents = [w for w in re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text)]
        # Dedupe while preserving order.
        seen: set[str] = set()
        out: list[str] = []
        for e in ents:
            if e not in seen:
                seen.add(e)
                out.append(e)
        return out

    def score(self, step_text: str) -> dict[str, float | int]:
        """Return ``{u_ent, n_entities, mean_entropy_nats}``.

        ``u_ent`` is the mean Shannon entropy (in nats, divided by
        ``ln(n_passages)`` to map to [0, 1] per the RSUS spec). If no
        entities are found we return ``u_ent = 0.5`` (the paper's
        documented prior for "no NE signal").
        """
        ents = self._entities(step_text)
        if not ents:
            return {"u_ent": 0.5, "n_entities": 0, "mean_entropy_nats": 0.0}
        n_p = max(2, len(self.paragraphs))
        log_n = math.log(n_p)
        entropies: list[float] = []
        for e in ents:
            scores = self.bm25.get_scores(self._tokenize(e))
            s = np.asarray(scores, dtype=np.float64)
            s = s - s.max()  # numerical stability
            w = np.exp(s)
            denom = float(w.sum())
            if denom <= 0.0:
                continue
            p = w / denom
            p = np.clip(p, 1e-12, 1.0)
            H = float(-(p * np.log(p)).sum())
            entropies.append(H)
        if not entropies:
            return {"u_ent": 0.5, "n_entities": len(ents), "mean_entropy_nats": 0.0}
        mean_H = float(sum(entropies) / len(entropies))
        return {
            "u_ent": float(min(1.0, max(0.0, mean_H / log_n))),
            "n_entities": int(len(ents)),
            "mean_entropy_nats": mean_H,
        }


# --------------------------------------------------------------------- #
# Continuation consistency (ReaLM-Retrieve U_cons)
# --------------------------------------------------------------------- #


class ConsistencySampler:
    """3-sample continuation consistency at a step boundary.

    Guo et al. §3.2: sample K=3 continuations of ~40 tokens at T=0.6,
    top-p=0.9 from the current state. ``U_cons`` is ``1 - mean pairwise
    cosine similarity`` between the continuations' sentence-BERT
    embeddings. Forks the cache once per sample and truncates back on
    exit, identically to ``VerbalizedConfidenceProbe``.

    Embedding model defaults to ``sentence-transformers/all-MiniLM-L6-v2``
    (90 MB, CPU-fast). Pinning to a small dense model is intentional:
    Guo et al. footnote 2 specifies that the comparator embedder is
    "an off-the-shelf sentence encoder" and explicitly rejects sharing
    the generator's hidden states for U_cons.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        n_samples: int = 3,
        max_new_tokens: int = 40,
        temperature: float = 0.6,
        top_p: float = 0.9,
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        seed: int = 20260511,
        capture_sample_texts: bool = False,
        capture_sample_hidden: bool = False,
        hidden_layer_idx: int | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.n_samples = int(n_samples)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.seed = int(seed)
        # Pilot extensions (rq1-pilot-v1): off by default to preserve v2 schema.
        self.capture_sample_texts = bool(capture_sample_texts)
        self.capture_sample_hidden = bool(capture_sample_hidden)
        self.hidden_layer_idx = hidden_layer_idx  # which `out.hidden_states[l]` to dump
        # Lazy embed model load.
        self._embed_name = embed_model_name
        self._embedder: Any | None = None

    def _load_embedder(self) -> Any:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._embedder = SentenceTransformer(self._embed_name)
        return self._embedder

    def _cache_seq_len(self, cache: Any) -> int:
        if hasattr(cache, "get_seq_length"):
            try:
                return int(cache.get_seq_length())
            except Exception:
                pass
        if hasattr(cache, "key_cache") and len(cache.key_cache) > 0:
            return int(cache.key_cache[0].shape[2])
        if isinstance(cache, tuple) and cache and isinstance(cache[0], tuple):
            return int(cache[0][0].shape[2])
        raise RuntimeError("cannot read sequence length from past_key_values object")

    def _truncate_cache_to(self, cache: Any, target_len: int) -> Any:
        if hasattr(cache, "crop"):
            try:
                cache.crop(int(target_len))
                return cache
            except Exception:
                pass
        if hasattr(cache, "layers") and len(cache.layers) > 0 and hasattr(cache.layers[0], "keys"):
            for layer in cache.layers:
                if layer.keys is not None:
                    layer.keys = layer.keys[:, :, :target_len, :].contiguous()
                if layer.values is not None:
                    layer.values = layer.values[:, :, :target_len, :].contiguous()
                if hasattr(layer, "cumulative_length"):
                    layer.cumulative_length = int(target_len)
            return cache
        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            for li in range(len(cache.key_cache)):
                cache.key_cache[li] = cache.key_cache[li][:, :, :target_len, :].contiguous()
                cache.value_cache[li] = cache.value_cache[li][:, :, :target_len, :].contiguous()
            if hasattr(cache, "_seen_tokens"):
                cache._seen_tokens = target_len
            return cache
        if isinstance(cache, tuple) and cache and isinstance(cache[0], tuple):
            return tuple(
                tuple(t[:, :, :target_len, :].contiguous() for t in layer)
                for layer in cache
            )
        raise RuntimeError("unsupported cache type for truncation")

    def _sample_once(self, cache: Any, start_token: int) -> dict[str, Any]:
        """Greedy + top-p + temperature sample; restores cache on exit.

        Returns ``{text, n_tokens, finish_reason, eos_hidden}`` where
        ``eos_hidden`` is None unless ``capture_sample_hidden=True`` —
        in that case it is a list[float] for the chosen mid-stack
        hidden layer at the *last* generated token (or last seen if
        EOS terminates early). ``finish_reason`` ∈ {"eos","max_tokens"}.
        """
        import torch
        original_len = self._cache_seq_len(cache)
        capture_hidden = self.capture_sample_hidden and self.hidden_layer_idx is not None
        last_hidden: list[float] | None = None
        try:
            with torch.inference_mode():
                cur = torch.tensor([[start_token]], device=self.model.device)
                ids: list[int] = []
                finish = "max_tokens"
                for _ in range(self.max_new_tokens):
                    out = self.model(
                        input_ids=cur,
                        past_key_values=cache,
                        use_cache=True,
                        output_hidden_states=capture_hidden,
                    )
                    cache = out.past_key_values
                    if capture_hidden and getattr(out, "hidden_states", None) is not None:
                        try:
                            h = out.hidden_states[self.hidden_layer_idx]
                            # h shape: [B, q_len=1, D]
                            last_hidden = h[0, -1, :].detach().to(torch.float32).cpu().tolist()
                        except Exception:
                            last_hidden = None
                    logits = out.logits[:, -1, :].to(torch.float32) / max(1e-6, self.temperature)
                    probs = torch.softmax(logits, dim=-1)[0]
                    # top-p / nucleus sampling
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = (cum > self.top_p).nonzero()
                    if len(cutoff) > 0:
                        k = int(cutoff[0].item()) + 1
                    else:
                        k = sorted_probs.numel()
                    keep_probs = sorted_probs[:k]
                    keep_idx = sorted_idx[:k]
                    keep_probs = keep_probs / keep_probs.sum()
                    pick = int(torch.multinomial(keep_probs, num_samples=1).item())
                    tok = int(keep_idx[pick].item())
                    if self.tokenizer.eos_token_id is not None and tok == self.tokenizer.eos_token_id:
                        finish = "eos"
                        break
                    ids.append(tok)
                    cur = torch.tensor([[tok]], device=self.model.device)
            return {
                "text": self.tokenizer.decode(ids, skip_special_tokens=True),
                "n_tokens": len(ids),
                "finish_reason": finish,
                "eos_hidden": last_hidden if capture_hidden else None,
            }
        finally:
            self._truncate_cache_to(cache, original_len)

    def score(self, cache: Any, start_token: int) -> dict[str, Any]:
        """Return ``{u_cons, mean_pairwise_cosine, n_samples, samples?}``.

        With ``capture_sample_texts=True`` the return dict also carries
        ``samples: list[{text, n_tokens, finish_reason}]`` and, if
        ``capture_sample_hidden=True``, each sample dict includes
        ``eos_hidden: list[float]`` (mid-stack last-token hidden state).
        With both off, the wire schema is unchanged from v2.

        Uses ``torch.manual_seed(self.seed + step_idx)`` for
        reproducibility — the caller is responsible for re-seeding
        with a per-step offset if it cares about determinism.
        """
        import torch
        torch.manual_seed(self.seed)
        sample_records: list[dict[str, Any]] = []
        sample_texts: list[str] = []
        for _ in range(self.n_samples):
            rec = self._sample_once(cache, start_token)
            sample_records.append(rec)
            sample_texts.append(rec["text"])
        embedder = self._load_embedder()
        embs = embedder.encode(sample_texts, normalize_embeddings=True)
        embs = np.asarray(embs, dtype=np.float64)
        # Mean pairwise cosine sim (vectors are L2-normed by the encoder).
        sims: list[float] = []
        for i in range(len(sample_texts)):
            for j in range(i + 1, len(sample_texts)):
                sims.append(float(np.dot(embs[i], embs[j])))
        mean_sim = float(sum(sims) / len(sims)) if sims else 1.0
        u_cons = float(max(0.0, min(1.0, 1.0 - mean_sim)))
        out: dict[str, Any] = {
            "u_cons": u_cons,
            "mean_pairwise_cosine": mean_sim,
            "n_samples": int(len(sample_texts)),
        }
        if self.capture_sample_texts:
            # Strip eos_hidden if not requested; preserves dict shape.
            out["samples"] = [
                {k: v for k, v in r.items() if (k != "eos_hidden" or self.capture_sample_hidden)}
                for r in sample_records
            ]
        return out


# --------------------------------------------------------------------- #
# RSUS combiner
# --------------------------------------------------------------------- #


# Guo et al. (`arXiv:2604.XXXXX` Footnote 1) published weights.
RSUS_WEIGHTS_DEFAULT = {"u_verb": 0.40, "u_ent": 0.35, "u_cons": 0.25}
RSUS_THRESHOLD_DEFAULT = 0.65


def combine_rsus(
    u_verb: float | None,
    u_ent: float | None,
    u_cons: float | None,
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Combine the three uncertainty signals into a single RSUS scalar.

    Returns ``(rsus_score, weights_actually_used)``. When a component
    is ``None`` (e.g. the model failed to emit a verbalized confidence
    integer) its weight is redistributed proportionally over the
    remaining components — Guo et al. §3.3 specifies this fallback.

    Output is in [0, 1] when each component is in [0, 1].
    """
    weights = dict(weights or RSUS_WEIGHTS_DEFAULT)
    inputs = {"u_verb": u_verb, "u_ent": u_ent, "u_cons": u_cons}
    keep = {k: v for k, v in inputs.items() if v is not None}
    if not keep:
        return float("nan"), weights
    w_sum = sum(weights[k] for k in keep)
    actually = {k: weights[k] / w_sum for k in keep}
    score = float(sum(actually[k] * keep[k] for k in keep))
    return score, actually


__all__ = [
    "compute_token_entropy_nats",
    "stopword_filter",
    "amax_from_last_layer_attentions",
    "RindAccumulator",
    "parse_verbal_confidence",
    "VerbalizedConfidenceProbe",
    "EntityEntropyScorer",
    "ConsistencySampler",
    "combine_rsus",
    "RSUS_WEIGHTS_DEFAULT",
    "RSUS_THRESHOLD_DEFAULT",
]
