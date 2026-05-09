"""Concept Representation Vector tracker (Phase 2.3).

Per logical step, accumulate the per-token hidden states (one per
model.forward call within that step), mean-pool to produce a single
CRV vector, and compute the drift-and-variance signals that downstream
gating uses.

Two layer-subset modes (per `_PLAN.md` 2.3 + the L1 mitigation):

  - "last_layer"   — single layer, the final hidden state. Cheapest.
                     Probably noisy on MoE because of router switching.
  - "mid_stack"    — single layer near the middle of the stack
                     (`n_layers // 2`). Pre-routing-noise candidate.
  - "concat"       — concatenate selected layers' vectors per token,
                     then mean-pool. Higher fidelity, larger CRV.
  - explicit list  — caller-specified layer indices.

CRV is stored on CPU fp32 throughout. The buffer of *per-token* hidden
states for the current step is dropped after each `commit_step` call so
the in-memory footprint never grows unboundedly.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

import torch


class CRVTracker:
    """Accumulate hidden states between boundaries; emit a per-step CRV.

    Parameters
    ----------
    layer_subset
        Iterable of layer indices to use for CRV. The CRV is the
        concatenation of mean-pooled per-layer vectors (in the order
        provided). If None, falls back to a single-layer subset selected
        by `layer_mode`.
    layer_mode
        Used only when `layer_subset` is None. One of:
        - "last_layer" — use the last available layer in `HiddenStateCapture`'s
                         `captured` dict (highest-index key).
        - "mid_stack"  — use the layer closest to half the highest-index key.
    window_size
        Rolling window length (in steps) for variance computation.
        Smaller window = more reactive; larger = smoother. Default 10 steps.
    drift_metric
        One of "cosine" (drift = 1 - cos_sim, range [0, 2]) or
        "l2" (drift = ||CRV_t - CRV_{t-1}||_2). Cosine is bounded and
        scale-invariant — preferred default.
    """

    def __init__(
        self,
        layer_subset: Iterable[int] | None = None,
        layer_mode: str = "last_layer",
        window_size: int = 10,
        drift_metric: str = "cosine",
    ) -> None:
        self.layer_subset = list(layer_subset) if layer_subset is not None else None
        if layer_mode not in {"last_layer", "mid_stack"}:
            raise ValueError(f"unknown layer_mode: {layer_mode!r}")
        self.layer_mode = layer_mode
        if drift_metric not in {"cosine", "l2"}:
            raise ValueError(f"unknown drift_metric: {drift_metric!r}")
        self.drift_metric = drift_metric
        self.window_size = max(2, int(window_size))

        # Per-step state we keep across commit_step calls
        self._per_step_buffer: list[dict[int, torch.Tensor]] = []
        self._previous_crv: torch.Tensor | None = None
        self._norm_history: deque[float] = deque(maxlen=self.window_size)
        self._step_features: list[dict] = []
        self._committed_crvs: list[torch.Tensor] = []
        self._committed_step_indices: list[int] = []

    def add_token(self, hidden_states: dict[int, torch.Tensor] | None) -> None:
        """Accept a per-token capture (output of HiddenStateCapture.record).

        Skips silently if `hidden_states` is None (sub-sampled step).
        """
        if hidden_states is None:
            return
        self._per_step_buffer.append(hidden_states)

    def _resolve_layer_subset(self, available_keys: Iterable[int]) -> list[int]:
        keys = sorted(available_keys)
        if self.layer_subset is not None:
            missing = [i for i in self.layer_subset if i not in keys]
            if missing:
                raise KeyError(
                    f"CRVTracker: layer_subset {self.layer_subset} requested but "
                    f"layers {missing} were not captured. Available: {keys}"
                )
            return list(self.layer_subset)
        if self.layer_mode == "last_layer":
            return [keys[-1]]
        if self.layer_mode == "mid_stack":
            return [keys[len(keys) // 2]]
        raise RuntimeError("unreachable")

    def commit_step(
        self,
        step_idx: int,
        token_idx: int,
        boundary_token_id: int | None = None,
    ) -> dict | None:
        """Finalize the current logical step. Mean-pools the per-token
        hidden states, computes CRV + scalar features, resets the
        per-token buffer.

        Returns the feature dict (also retained internally), or None if
        no tokens were captured for this step (which shouldn't normally
        happen unless every step in this window was sub-sampled).
        """
        if not self._per_step_buffer:
            return None

        layer_indices = self._resolve_layer_subset(self._per_step_buffer[0].keys())

        # Mean-pool per layer across the tokens in this step.
        per_layer_means: list[torch.Tensor] = []
        for layer_idx in layer_indices:
            stacked = torch.stack(
                [hs[layer_idx] for hs in self._per_step_buffer]
            )  # (n_tokens_in_step, hidden_dim)
            per_layer_means.append(stacked.mean(dim=0))

        crv = torch.cat(per_layer_means, dim=-1)  # (sum(hidden_dims),)
        norm = float(crv.norm().item())

        # Drift vs previous CRV.
        if self._previous_crv is None:
            drift = 0.0
        else:
            if self.drift_metric == "cosine":
                cos = torch.nn.functional.cosine_similarity(
                    crv.unsqueeze(0), self._previous_crv.unsqueeze(0)
                ).item()
                drift = float(1.0 - cos)
            else:
                drift = float(torch.linalg.vector_norm(crv - self._previous_crv).item())

        # Rolling variance over the L2-norm history (windowed).
        self._norm_history.append(norm)
        variance = (
            float(torch.tensor(list(self._norm_history)).var(unbiased=True).item())
            if len(self._norm_history) > 1
            else 0.0
        )

        feature = {
            "step_idx": step_idx,
            "token_idx_end": token_idx,
            "boundary_token_id": int(boundary_token_id) if boundary_token_id is not None else None,
            "n_tokens_in_step": len(self._per_step_buffer),
            "layer_indices": layer_indices,
            "crv_dim": int(crv.shape[-1]),
            "hidden_norm": round(norm, 6),
            f"drift_{self.drift_metric}": round(drift, 6),
            "variance_l2_window": round(variance, 6),
        }

        self._step_features.append(feature)
        self._committed_crvs.append(crv)
        self._committed_step_indices.append(step_idx)
        self._previous_crv = crv
        self._per_step_buffer = []  # drop per-token buffer

        return feature

    @property
    def features(self) -> list[dict]:
        return list(self._step_features)

    def stacked_crvs(self) -> tuple[torch.Tensor, list[int]]:
        """Return (n_steps, crv_dim) tensor of all committed CRVs and the
        list of step_idx values they correspond to."""
        if not self._committed_crvs:
            return torch.empty(0), []
        return torch.stack(self._committed_crvs), list(self._committed_step_indices)
