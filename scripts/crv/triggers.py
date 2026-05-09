"""Trigger rules over per-step CRV features (Phase 2.4).

Phase 2 ships a deliberately *simple* trigger family — threshold rules
over the scalar features produced by `CRVTracker`. Calibration of the
thresholds is deferred to Phase 4 (calibrate offline on Phase 4.1's
HotpotQA / MuSiQue split). For now we provide:

  - "drift"      — fire when cosine drift exceeds τ_drift
  - "variance"   — fire when rolling-variance-of-L2-norm exceeds τ_var
  - "or"         — fire when *either* exceeds threshold (default)
  - "and"        — fire when *both* exceed threshold (conservative)
  - "none"       — never fire (control / random-rate baseline scaffold)

The rules are pure functions of `feature` dicts produced by
`CRVTracker.commit_step`. No hidden state inside the trigger itself, so
multiple triggers can be applied to the same feature stream cheaply.
"""

from __future__ import annotations

from typing import Mapping


_DEFAULT_DRIFT_KEY = "drift_cosine"
_VARIANCE_KEY = "variance_l2_window"


class CRVTrigger:
    """Threshold-rule trigger over CRV scalar features.

    Parameters
    ----------
    threshold_drift
        Fire if drift exceeds this. Drift is read from the feature dict
        at `drift_key` (default `drift_cosine`). Cosine drift values are
        in [0, 2]; in practice "above 0.1" already means "the model's
        last-step latent state moved noticeably."
    threshold_variance
        Fire if rolling variance of L2 norm exceeds this. Scale is
        model-dependent (raw L2 norms vary by hidden_dim and architecture)
        so calibrate per-architecture.
    mode
        Combination rule: "or" / "and" / "drift" / "variance" / "none".
    drift_key
        Key in the feature dict to read the drift value from. Defaults to
        `drift_cosine`. If you constructed CRVTracker with `drift_metric="l2"`
        pass `drift_key="drift_l2"`.
    """

    VALID_MODES = ("or", "and", "drift", "variance", "none")

    def __init__(
        self,
        threshold_drift: float = 0.1,
        threshold_variance: float = 1.0,
        mode: str = "or",
        drift_key: str = _DEFAULT_DRIFT_KEY,
        variance_key: str = _VARIANCE_KEY,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")
        self.threshold_drift = float(threshold_drift)
        self.threshold_variance = float(threshold_variance)
        self.mode = mode
        self.drift_key = drift_key
        self.variance_key = variance_key

    def fired(self, feature: Mapping[str, float]) -> bool:
        if self.mode == "none":
            return False
        d = float(feature.get(self.drift_key, 0.0)) > self.threshold_drift
        v = float(feature.get(self.variance_key, 0.0)) > self.threshold_variance
        if self.mode == "or":
            return d or v
        if self.mode == "and":
            return d and v
        if self.mode == "drift":
            return d
        if self.mode == "variance":
            return v
        raise RuntimeError(f"unreachable mode: {self.mode}")

    def explain(self, feature: Mapping[str, float]) -> dict:
        d_val = float(feature.get(self.drift_key, 0.0))
        v_val = float(feature.get(self.variance_key, 0.0))
        return {
            "mode": self.mode,
            "drift_value": d_val,
            "drift_threshold": self.threshold_drift,
            "drift_exceeded": d_val > self.threshold_drift,
            "variance_value": v_val,
            "variance_threshold": self.threshold_variance,
            "variance_exceeded": v_val > self.threshold_variance,
            "fired": self.fired(feature),
        }
