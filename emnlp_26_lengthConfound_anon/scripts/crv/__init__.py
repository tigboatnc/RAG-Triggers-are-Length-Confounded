"""Concept Representation Vector (CRV) machinery for Phase 2.

The package is engine-agnostic and architecture-agnostic by design (Phase
2.1b in `_PLAN.md`): the public adapter consumes only HF
`CausalLMOutputWithPast.hidden_states` tuples and token-id streams, so it
works uniformly on Phi-4-mini (dense), Gemma 4 sliding-window MoE, and
Qwen3.6 hybrid Gated DeltaNet MoE without any per-architecture wiring.

Submodules:
  - `hooks`       — `HiddenStateCapture` extracts per-step hidden states
                    from a model output, detaches, moves to CPU fp32.
  - `boundaries`  — `StepBoundaryDetector` decides when a "logical step"
                    ends (sentence-end, fixed-stride, think-block).
  - `vectors`     — `CRVTracker` accumulates per-token hidden states between
                    boundaries, mean-pools to a per-step CRV, computes L2
                    norm + cosine drift vs previous step + rolling variance.
  - `triggers`    — `CRVTrigger` applies threshold rules to per-step
                    features and decides whether to fire (i.e. trigger
                    corrective retrieval / re-think).
"""

from .hooks import HiddenStateCapture
from .boundaries import StepBoundaryDetector
from .vectors import CRVTracker
from .triggers import CRVTrigger
from .baseline_gates import (
    DISCOURSE_PATTERNS,
    DraginGate,
    EtcGate,
    GripSimGate,
    HEDGE_PATTERNS,
    LatentGateAdapter,
    RandomGate,
    RealmRetrieveSimGate,
    RealmRsusGate,
    RSUS_THRESHOLD_DEFAULT,
    RSUS_WEIGHTS_DEFAULT,
    SelfDoubtFullGate,
    SelfDoubtTier1Gate,
    VERIFY_PATTERNS,
    discourse_marker_match,
    hedge_verify_counts,
)

__all__ = [
    "HiddenStateCapture",
    "StepBoundaryDetector",
    "CRVTracker",
    "CRVTrigger",
    "DISCOURSE_PATTERNS",
    "DraginGate",
    "EtcGate",
    "GripSimGate",
    "HEDGE_PATTERNS",
    "LatentGateAdapter",
    "RandomGate",
    "RealmRetrieveSimGate",
    "RealmRsusGate",
    "RSUS_THRESHOLD_DEFAULT",
    "RSUS_WEIGHTS_DEFAULT",
    "SelfDoubtFullGate",
    "SelfDoubtTier1Gate",
    "VERIFY_PATTERNS",
    "discourse_marker_match",
    "hedge_verify_counts",
]
