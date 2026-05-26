"""Per-step hidden-state capture (Phase 2.1).

We deliberately do NOT use `model.register_forward_hook(...)` here. The HF
public API already exposes per-layer hidden states via `output.hidden_states`
when `output_hidden_states=True` is passed to `model.forward(...)`. That's
the same surface across:

  - dense full-attention models (Phi-4-mini)
  - sliding-window + sparse-global hybrids (Gemma 4 family)
  - Gated-DeltaNet : full-attention hybrids (Qwen3-Next / Qwen3.6 family)

So this module just consumes that API. No per-arch glue, no
register-hook ordering nightmares, no surgery on the model graph. The
"hook" in `_PLAN.md` Phase 2.1 is really an interception point on the
forward output, not a torch hook handle.

Memory note: `output_hidden_states=True` is a small overhead during decode
(seq_len = 1) — ~`n_layers × hidden_dim × 2 bytes` per step on GPU before
we move it to CPU fp32. For Qwen3.6 that's ~336 KB / step, which fits the
≤5 % throughput-regression budget comfortably.
"""

from __future__ import annotations

from typing import Iterable

import torch


class HiddenStateCapture:
    """Extract last-token hidden states from a forward output, per layer.

    Parameters
    ----------
    layer_subset
        Iterable of layer indices to keep (after the embedding layer; i.e.
        index 0 == embedding output, index 1 == output of layer 1, ...,
        index N == output of last layer for an N-layer model). If None,
        capture all layers including the embedding entry. The model's
        `out.hidden_states` tuple has length `n_layers + 1`.
    every_n_steps
        Sub-sample the capture stream — record only steps where
        `step_idx % every_n_steps == 0`. Default 1 (capture every step).
        Use values >1 if you blow the throughput budget on long traces.
    device
        Where to store captured tensors. "cpu" by default, per the L2
        mitigation noted in `_PLAN.md` (tensors detached + moved to CPU
        fp32 immediately so they don't pin GPU memory or autograd graph).
    dtype
        Cast captured tensors to this dtype on capture. fp32 by default
        for downstream numerical stability of variance/cosine.
    """

    def __init__(
        self,
        layer_subset: Iterable[int] | None = None,
        every_n_steps: int = 1,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.layer_subset = list(layer_subset) if layer_subset is not None else None
        self.every_n_steps = max(1, int(every_n_steps))
        self.device = device
        self.dtype = dtype
        self._capture_count = 0
        self._skip_count = 0

    def record(self, step_idx: int, output) -> dict[int, torch.Tensor] | None:
        """Capture last-token hidden states from this forward output.

        Returns a dict {layer_index: 1-D tensor of shape (hidden_dim,)} or
        None if this step was skipped (sub-sampling) or the model wasn't
        configured for hidden-state output.
        """
        if step_idx % self.every_n_steps != 0:
            self._skip_count += 1
            return None

        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError(
                "HiddenStateCapture: model forward must be invoked with "
                "output_hidden_states=True. Got an output with no .hidden_states."
            )

        # `hidden_states` is a tuple of length n_layers + 1.
        # Each entry has shape (batch, seq_len, hidden_dim).
        # On decode (seq_len = 1) we want the only token; on prefill (seq_len = N)
        # we want the LAST token (the one whose logits became the next decode).
        indices = self.layer_subset if self.layer_subset is not None else range(len(hidden_states))
        captured: dict[int, torch.Tensor] = {}
        for layer_idx in indices:
            tensor = hidden_states[layer_idx]
            last_tok_vec = tensor[:, -1, :].squeeze(0)  # (hidden_dim,)
            captured[int(layer_idx)] = last_tok_vec.detach().to(
                device=self.device, dtype=self.dtype
            )

        self._capture_count += 1
        return captured

    @property
    def stats(self) -> dict[str, int]:
        return {
            "captures": self._capture_count,
            "skipped": self._skip_count,
            "every_n_steps": self.every_n_steps,
        }
