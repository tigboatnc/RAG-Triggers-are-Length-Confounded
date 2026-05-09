"""Phase 3 baseline gates — simulated comparators for the latent CRV trigger.

Three gates live here:

  - ``RealmRetrieveSimGate``   — simulator for *ReaLM-Retrieve*. The original
    fires retrieval when a small trained verifier scores the current
    reasoning step as uncertain ("Reasoning Step Uncertainty Score", RSU)
    AND/OR when discourse markers like "let me verify" / "actually" /
    "I need to check" appear. We do not have access to ReaLM's trained
    verifier, so the RSU score is replaced by a *stand-in* — the per-step
    mean next-token entropy of the generator itself. This is documented
    inline as a Phase 3 caveat (L3 in ``_PLAN.md``).

  - ``GripSimGate``            — simulator for *GRIP* control-token gating.
    The original learns a classifier head that emits a `<retrieve>` /
    `<rethink>` control token at boundaries; without trained weights we
    approximate "the classifier would have fired" as
    (next-token entropy spike at the boundary, z-scored against a rolling
    window) AND (discourse-marker prior). Same L3 caveat applies.

  - ``RandomGate``              — Bernoulli baseline. The plan specifies firing
    at the *same per-step rate as the average of the trained gates* on the
    same trace, so this gate calibrates its rate **post-hoc** in
    ``calibrate_rate(...)`` once the trace's other gate decisions are known.
    The PRNG seed is captured in the gate's metadata.

All four gates (these three + the latent ``CRVTrigger``) emit rows that
match the unified Phase 3.4 schema::

    {
      "run_id":      str,           # provenance for joining across files
      "model":       str,
      "prompt_class": str,
      "layer_mode":  str | None,    # None for non-latent gates (informational)
      "step_idx":    int,           # logical step index at this boundary
      "token_idx":   int,           # absolute decode-step index of boundary
      "gate_name":   str,           # "latent" | "realm_sim" | "grip_sim" | "random"
      "fired":       bool,
      "score":       float,         # gate-internal score (higher = more "fire-y")
      "reason":      str,           # short human explanation of which sub-rule fired
      "simulation_caveats": list[str],  # L3 honesty: empty for latent + random; non-empty for the sims
    }

Discourse-marker patterns are precompiled regex (case-insensitive) so the
text scan is O(L) per step. The pattern set is defensible-and-public:
direct quotes from the ReaLM-Retrieve and GRIP papers' qualitative
examples plus a small handful of common reasoning-model fillers
("hmm,", "wait,", "on second thought").

Public API: each gate has ``evaluate(step_record)`` returning a dict
matching the schema above (minus ``run_id``/``model``/``prompt_class``,
which are stamped in by the driver). ``RandomGate`` additionally has
``calibrate_rate(other_gate_decisions)`` and ``finalize(step_records)``.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


DISCOURSE_PATTERNS: tuple[str, ...] = (
    # ReaLM-Retrieve qualitative examples (verify / re-check family).
    r"\blet me verify\b",
    r"\blet me double[- ]?check\b",
    r"\bi need to (verify|check|confirm|recheck)\b",
    r"\bi should (verify|check|confirm|recheck)\b",
    r"\bactually,",
    r"\bwait,",
    r"\bhmm,",
    r"\bon second thought\b",
    # GRIP-style "I'm uncertain about" cues.
    r"\bi'?m not (sure|certain)\b",
    r"\bi am not (sure|certain)\b",
    r"\bi don'?t (remember|recall|know)\b",
    r"\bnot entirely sure\b",
    # Self-correction starters.
    r"\bcorrection:",
    r"\bscratch that\b",
)

_DISCOURSE_RE = re.compile("|".join(DISCOURSE_PATTERNS), re.IGNORECASE)


def discourse_marker_match(text: str) -> str | None:
    """Return the first matching discourse phrase in ``text``, or None.

    Returns the literal matched substring (lowercased) so callers can log
    it as part of the gate's ``reason`` field for auditability.
    """
    if not text:
        return None
    m = _DISCOURSE_RE.search(text)
    return m.group(0).lower() if m else None


@dataclass
class _RollingStats:
    """Online mean / variance over a fixed window. Used by GRIP entropy z-score."""

    window_size: int = 20
    _values: list[float] = field(default_factory=list)

    def push(self, x: float) -> None:
        self._values.append(float(x))
        if len(self._values) > self.window_size:
            self._values.pop(0)

    @property
    def n(self) -> int:
        return len(self._values)

    @property
    def mean(self) -> float:
        return sum(self._values) / self.n if self._values else 0.0

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        m = self.mean
        var = sum((v - m) ** 2 for v in self._values) / (self.n - 1)
        return var ** 0.5

    def zscore(self, x: float) -> float:
        s = self.std
        return (float(x) - self.mean) / s if s > 0 else 0.0


# ---- ReaLM-Retrieve simulation ------------------------------------------------

class RealmRetrieveSimGate:
    """Simulator for ReaLM-Retrieve's RSU+discourse trigger.

    The decision is::

        fired = (rsu_score > rsu_threshold) OR (discourse marker matched)

    where ``rsu_score`` is the **per-step mean next-token entropy** of the
    generator (not the original paper's trained verifier). That substitution
    is the Phase 3 simulation gap; it's emitted in every row's
    ``simulation_caveats`` field.

    Step record contract (the ``step_record`` argument to ``evaluate``):

        - ``step_text``                 str   — text decoded since previous boundary
        - ``mean_token_entropy``        float — mean H(next_token | prefix) over
                                                the tokens *inside* this step,
                                                in nats
    """

    name = "realm_sim"

    def __init__(self, rsu_threshold_nats: float = 2.5) -> None:
        self.rsu_threshold = float(rsu_threshold_nats)

    def evaluate(self, step_record: Mapping[str, object]) -> dict:
        rsu = float(step_record.get("mean_token_entropy", 0.0))  # nats
        text = str(step_record.get("step_text", ""))
        marker = discourse_marker_match(text)

        rsu_fired = rsu > self.rsu_threshold
        marker_fired = marker is not None
        fired = rsu_fired or marker_fired

        reason_parts: list[str] = []
        if rsu_fired:
            reason_parts.append(f"rsu>{self.rsu_threshold:.2f}({rsu:.3f})")
        if marker_fired:
            reason_parts.append(f"marker:{marker!r}")
        if not reason_parts:
            reason_parts.append("below_thresholds")

        return {
            "gate_name": self.name,
            "fired": fired,
            "score": rsu,  # primary score is the entropy-as-RSU stand-in
            "reason": ";".join(reason_parts),
            "simulation_caveats": [
                "rsu_score_is_generator_entropy_not_trained_verifier",
                "discourse_pattern_set_is_curated_not_paper_exact",
            ],
        }

    def metadata(self) -> dict:
        return {
            "gate_name": self.name,
            "rsu_threshold_nats": self.rsu_threshold,
            "discourse_patterns": list(DISCOURSE_PATTERNS),
            "caveats": [
                "rsu_score_is_generator_entropy_not_trained_verifier",
                "discourse_pattern_set_is_curated_not_paper_exact",
            ],
        }


# ---- GRIP simulation ----------------------------------------------------------

class GripSimGate:
    """Simulator for GRIP control-token classifier.

    Without the trained classifier head, the firing rule is::

        fired = (boundary_entropy_zscore > z_threshold) AND (discourse_marker matched)
              OR (boundary_entropy_zscore > strong_z_threshold)

    The conjunction matches GRIP's "the classifier learns context-aware
    triggers"; we let a strong entropy spike alone fire the gate too,
    because the conjunction-only version under-fires badly when discourse
    markers are sparse (and the original GRIP would in that case have
    learned a different trigger pattern).

    Step record contract:

        - ``boundary_token_entropy``  float — H(next_token) at the boundary token
        - ``step_text``               str
    """

    name = "grip_sim"

    def __init__(
        self,
        z_threshold: float = 1.5,
        strong_z_threshold: float = 2.5,
        window_size: int = 20,
    ) -> None:
        self.z_threshold = float(z_threshold)
        self.strong_z_threshold = float(strong_z_threshold)
        self._stats = _RollingStats(window_size=window_size)

    def evaluate(self, step_record: Mapping[str, object]) -> dict:
        h = float(step_record.get("boundary_token_entropy", 0.0))
        z = self._stats.zscore(h)
        # Update rolling window AFTER scoring, so the current value isn't
        # part of its own baseline (avoids self-suppression of obvious
        # spikes when the spike-stream is bursty).
        self._stats.push(h)

        text = str(step_record.get("step_text", ""))
        marker = discourse_marker_match(text)

        weak_fired = (z > self.z_threshold) and (marker is not None)
        strong_fired = z > self.strong_z_threshold
        fired = weak_fired or strong_fired

        reason_parts: list[str] = [f"z={z:.2f}"]
        if marker:
            reason_parts.append(f"marker:{marker!r}")
        if strong_fired:
            reason_parts.append("strong_spike")
        elif weak_fired:
            reason_parts.append("spike+marker")
        else:
            reason_parts.append("below_thresholds")

        return {
            "gate_name": self.name,
            "fired": fired,
            "score": z,
            "reason": ";".join(reason_parts),
            "simulation_caveats": [
                "no_trained_control_token_classifier",
                "entropy_zscore_substitutes_for_classifier_logit",
                "discourse_pattern_set_is_curated_not_paper_exact",
            ],
        }

    def metadata(self) -> dict:
        return {
            "gate_name": self.name,
            "z_threshold": self.z_threshold,
            "strong_z_threshold": self.strong_z_threshold,
            "rolling_window_size": self._stats.window_size,
            "discourse_patterns": list(DISCOURSE_PATTERNS),
            "caveats": [
                "no_trained_control_token_classifier",
                "entropy_zscore_substitutes_for_classifier_logit",
                "discourse_pattern_set_is_curated_not_paper_exact",
            ],
        }


# ---- Random baseline ----------------------------------------------------------

class RandomGate:
    """Bernoulli baseline at a calibrated rate.

    The plan (Phase 3.3) specifies the random gate must fire at the same
    per-step rate as the *average* of the three trained gates. We can't
    know that rate online, so the workflow is:

      1. During the live run, ``evaluate(step_record)`` records *no decision
         yet* — it just remembers the step_idx so we know which steps
         existed.
      2. After the run finishes, the driver calls
         ``calibrate_rate(other_gate_decisions)`` with the per-step
         decisions of the (latent + realm_sim + grip_sim) gates.
      3. The driver then calls ``finalize(step_records)`` which returns one
         row per step with a deterministic seeded Bernoulli draw at the
         calibrated rate.

    The seed is captured in metadata for replay.
    """

    name = "random"

    def __init__(self, seed: int = 20260509) -> None:
        self.seed = int(seed)
        self._rate: float | None = None
        self._n_other_gates: int | None = None
        self._step_indices: list[int] = []

    def evaluate(self, step_record: Mapping[str, object]) -> dict:
        idx = int(step_record["step_idx"])
        self._step_indices.append(idx)
        # Online row: no decision yet. Driver writes finalized rows after
        # calibration. We still emit a placeholder so the downstream
        # accounting can detect "calibration not run yet" if it ever
        # encounters one of these.
        return {
            "gate_name": self.name,
            "fired": False,
            "score": float("nan"),
            "reason": "pending_calibration",
            "simulation_caveats": ["random_gate_pending_post_hoc_calibration"],
        }

    def calibrate_rate(self, other_gate_decisions: Sequence[Sequence[bool]]) -> float:
        """Set the random-gate fire rate to the mean of the per-gate fire rates.

        ``other_gate_decisions`` is a list of per-gate decision sequences
        (one bool per step). Per the plan: "fire at the same per-step
        rate as the *average* of the three trained gates." We compute
        each gate's fire rate independently then average those rates,
        which is more stable than concatenating-then-averaging when one
        gate fires far more than another.
        """
        if not other_gate_decisions:
            self._rate = 0.0
            self._n_other_gates = 0
            return 0.0
        rates = [
            (sum(1 for d in seq if d) / len(seq)) if seq else 0.0
            for seq in other_gate_decisions
        ]
        self._rate = sum(rates) / len(rates)
        self._n_other_gates = len(rates)
        return self._rate

    def finalize(self, step_records: Iterable[Mapping[str, object]]) -> list[dict]:
        """Materialize one Bernoulli row per step at the calibrated rate."""
        if self._rate is None:
            raise RuntimeError(
                "RandomGate.finalize called before calibrate_rate; "
                "the driver must compute the average gate rate first."
            )
        rng = random.Random(self.seed)
        out: list[dict] = []
        for record in step_records:
            fired = rng.random() < self._rate
            out.append({
                "gate_name": self.name,
                "fired": fired,
                "score": self._rate,
                "reason": f"bernoulli(p={self._rate:.4f})",
                "simulation_caveats": [],
                "step_idx": int(record["step_idx"]),
                "token_idx": int(record.get("token_idx_end", record.get("token_idx", -1))),
            })
        return out

    def metadata(self) -> dict:
        return {
            "gate_name": self.name,
            "seed": self.seed,
            "calibrated_rate": self._rate,
            "n_other_gates_used_for_calibration": self._n_other_gates,
            "calibration_recipe": "mean of per-gate fire rates over the same trace",
        }


# ---- Latent gate adapter (so all four share one dispatch surface) ------------

class LatentGateAdapter:
    """Thin wrapper around ``CRVTrigger`` matching the Phase 3 evaluate() shape.

    Reads ``drift_cosine`` / ``variance_l2_window`` directly from the CRV
    feature dict produced by ``CRVTracker.commit_step`` — same fields the
    Phase 2 ``02_crv_probe.py`` script already writes. No behavior change
    versus Phase 2; this adapter exists purely so the driver can iterate
    over a heterogeneous list of gates with one ``evaluate()`` call.
    """

    name = "latent"

    def __init__(self, trigger) -> None:
        self.trigger = trigger

    def evaluate(self, step_record: Mapping[str, object]) -> dict:
        explanation = self.trigger.explain(step_record)
        fired = bool(explanation["fired"])
        drift_v = explanation["drift_value"]
        var_v = explanation["variance_value"]
        score = max(
            drift_v / max(self.trigger.threshold_drift, 1e-9),
            var_v / max(self.trigger.threshold_variance, 1e-9),
        )
        reasons: list[str] = []
        if explanation["drift_exceeded"]:
            reasons.append(f"drift>{self.trigger.threshold_drift:.3f}({drift_v:.3f})")
        if explanation["variance_exceeded"]:
            reasons.append(f"var>{self.trigger.threshold_variance:.3f}({var_v:.3f})")
        if not reasons:
            reasons.append("below_thresholds")
        return {
            "gate_name": self.name,
            "fired": fired,
            "score": float(score),
            "reason": ";".join(reasons),
            "simulation_caveats": [],
        }

    def metadata(self) -> dict:
        return {
            "gate_name": self.name,
            "threshold_drift": self.trigger.threshold_drift,
            "threshold_variance": self.trigger.threshold_variance,
            "mode": self.trigger.mode,
            "caveats": [
                "thresholds_uncalibrated_phase4_will_calibrate_on_labeled_traces",
            ],
        }


__all__ = [
    "DISCOURSE_PATTERNS",
    "discourse_marker_match",
    "RealmRetrieveSimGate",
    "GripSimGate",
    "RandomGate",
    "LatentGateAdapter",
]
