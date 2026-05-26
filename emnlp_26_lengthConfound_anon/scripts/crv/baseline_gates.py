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


# ---- SELFDOUBT-Tier1: true HVR gate ------------------------------------------

# Seed marker dictionaries drawn from Pandey et al. (`arXiv:2604.06389v1`)
# page 1 and §3.1 seed examples. The paper specifies a per-model
# unsupervised discovery pipeline (90 unlabeled traces, multi-LLM
# consensus, BGE-M3 coherence filter); these seeds are a no-compute
# approximation. `_PLAN.md` Phase 4.10 covers the full discovery pipeline
# when compute is available. Patterns are case-insensitive whole-word
# matches; the comma-anchored fillers ("hmm,", "wait,") preserve their
# discourse-marker semantics without false-firing inside ordinary prose.
HEDGE_PATTERNS: tuple[str, ...] = (
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\bpossibly\b",
    r"\bprobably\b",
    r"\bseems\b",
    r"\bappears\b",
    r"\bnot (sure|certain|entirely sure)\b",
    r"\bi'?m not (sure|certain)\b",
    r"\bi am not (sure|certain)\b",
    r"\bi don'?t (remember|recall|know)\b",
    r"\bcould be wrong\b",
    r"\bmight be (wrong|incorrect)\b",
    r"\bmay be (wrong|incorrect)\b",
    r"\bhmm,",
    r"\bwait,",
    r"\bwell,",
)

VERIFY_PATTERNS: tuple[str, ...] = (
    r"\blet me (verify|check|confirm|re-?read|re-?check|see)\b",
    r"\bi (need to|should|will|'ll) (verify|check|confirm|recheck|reconsider|review|re-?read)\b",
    r"\bdouble[- ]?check\b",
    r"\bsubstitute back\b",
    r"\bplug (it )?in\b",
    r"\bcorrection:",
    r"\bactually,",
    r"\bon second thought\b",
    r"\bscratch that\b",
)

_HEDGE_RE = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)
_VERIFY_RE = re.compile("|".join(VERIFY_PATTERNS), re.IGNORECASE)


def hedge_verify_counts(text: str) -> tuple[int, int]:
    """Return ``(hedge_count, verify_count)`` for SELFDOUBT-Tier1 HVR."""
    if not text:
        return 0, 0
    return (len(_HEDGE_RE.findall(text)), len(_VERIFY_RE.findall(text)))


class SelfDoubtTier1Gate:
    """True Hedge-to-Verify Ratio gate per Pandey et al. (2026), §3.

    Per-step decision::

        HVR_t = h_t / (v_t + 1)
        fired = HVR_t > hvr_threshold

    where ``h_t`` and ``v_t`` are counts of hedge / verify regex matches in
    the step text. ``+1`` in the denominator is exactly the Laplace-smoothed
    form used in the SELFDOUBT paper (page 1: "hedge markers divided by the
    number of verify markers plus one").

    Adaptation note: the paper computes HVR over a *complete* reasoning
    trace and reports trace-level AUROC. We compute HVR on each step's
    text incrementally and fire when the per-step HVR crosses
    ``hvr_threshold``. This is our adaptation; documented in the paper
    §5. The Tier-1 ``HVR == 0 ⇒ predict correct`` gate from the SELFDOUBT
    paper is recoverable by setting ``hvr_threshold = 0.0`` and reading
    the ``fired`` field with inverted polarity.

    Step record contract:

        - ``step_text``  str  — text decoded since previous boundary
    """

    name = "selfdoubt_t1"

    def __init__(self, hvr_threshold: float = 1.0) -> None:
        self.hvr_threshold = float(hvr_threshold)

    def evaluate(self, step_record: Mapping[str, object]) -> dict:
        text = str(step_record.get("step_text", ""))
        h, v = hedge_verify_counts(text)
        hvr = h / (v + 1)
        fired = hvr > self.hvr_threshold
        reason = f"hvr={hvr:.3f}(h={h},v={v},thr={self.hvr_threshold:.2f})"
        return {
            "gate_name": self.name,
            "fired": fired,
            "score": float(hvr),
            "reason": reason,
            "simulation_caveats": [
                "marker_dictionary_is_seed_set_not_paper_discovery_pipeline",
                "hvr_computed_per_step_not_per_trace_as_in_paper",
            ],
        }

    def metadata(self) -> dict:
        return {
            "gate_name": self.name,
            "hvr_threshold": self.hvr_threshold,
            "hedge_patterns": list(HEDGE_PATTERNS),
            "verify_patterns": list(VERIFY_PATTERNS),
            "caveats": [
                "marker_dictionary_is_seed_set_not_paper_discovery_pipeline",
                "hvr_computed_per_step_not_per_trace_as_in_paper",
            ],
        }


# ---- ETC: Entropy-Trend Constraint (Li et al.) -------------------------------

class EtcGate:
    """Entropy-Trend Constraint gate per Li et al. (Modeling Uncertainty Trends).

    Per-step decision::

        D2H_t = H_{t+1} - 2 H_t + H_{t-1}        (second-order difference)
        fired = |D2H_t| > d2h_threshold

    where ``H_t`` is the per-step mean next-token entropy. The second
    difference captures "uncertainty turning points" — where an entropy
    trend changes curvature — which the ETC paper argues is a stronger
    timing signal than absolute entropy.

    Adaptation note: the paper formulates ETC on a per-*token* entropy
    series. Our step records expose per-*step* mean entropy only (the
    capture cost of per-token entropy was outside the Phase 2 budget).
    We therefore evaluate ETC on the step-mean series, which has
    coarser resolution than the original. Documented in the paper §5.

    At ``step_idx <= 1`` the second difference is undefined and we emit
    ``fired = False`` with ``score = nan``. Calibration ignores these
    rows so the threshold reflects only well-defined boundary points.

    Step record contract:

        - ``step_idx``           int    — 1-indexed logical step
        - ``mean_token_entropy`` float  — H_t in nats
    """

    name = "etc"

    def __init__(self, d2h_threshold: float = 0.5) -> None:
        self.d2h_threshold = float(d2h_threshold)
        self._history: list[float] = []

    def reset(self) -> None:
        """Call between traces so the per-trace entropy buffer is cleared."""
        self._history = []

    def evaluate(self, step_record: Mapping[str, object]) -> dict:
        h = float(step_record.get("mean_token_entropy", 0.0))
        self._history.append(h)

        # We need three consecutive entropies to compute D2H_{t-1}.
        # Fire at the middle point of the most recent triple; the
        # earliest two steps in any trace are necessarily not-fired.
        if len(self._history) < 3:
            return {
                "gate_name": self.name,
                "fired": False,
                "score": float("nan"),
                "reason": "warmup_<3_steps",
                "simulation_caveats": [
                    "d2h_on_step_mean_entropy_not_per_token_as_in_paper",
                ],
            }

        h_prev, h_mid, h_curr = self._history[-3], self._history[-2], self._history[-1]
        d2h = h_curr - 2.0 * h_mid + h_prev
        abs_d2h = abs(d2h)
        fired = abs_d2h > self.d2h_threshold
        return {
            "gate_name": self.name,
            "fired": fired,
            "score": float(abs_d2h),
            "reason": f"|D2H|={abs_d2h:.3f}(thr={self.d2h_threshold:.2f})",
            "simulation_caveats": [
                "d2h_on_step_mean_entropy_not_per_token_as_in_paper",
            ],
        }

    def metadata(self) -> dict:
        return {
            "gate_name": self.name,
            "d2h_threshold": self.d2h_threshold,
            "caveats": [
                "d2h_on_step_mean_entropy_not_per_token_as_in_paper",
            ],
        }



class DraginGate:
    """True DRAGIN RIND gate per Su et al. (`arXiv:2403.10081v3`), §3.2.

    Per-step decision::

        RIND_max(step) = max over content tokens i in step of H_i * a_max(i) * s_i
        fired = RIND_max(step) > rind_threshold

    where the per-token RIND triple is accumulated online during decoding
    by ``crv.sota_capture.RindAccumulator`` — see that module for the
    architecture-specific attention-layer selection (last full-attention
    layer on Qwen 3.6's hybrid stack; last layer on dense Phi / Gemma).

    Step record contract:

        - ``dragin_rind_max``  float — produced by RindAccumulator.commit_step()
    """

    name = "dragin"

    def __init__(self, rind_threshold: float = 1.0) -> None:
        self.rind_threshold = float(rind_threshold)

    def evaluate(self, step_record: Mapping[str, object]) -> dict:
        v = step_record.get("dragin_rind_max")
        if v is None:
            return {
                "gate_name": self.name,
                "fired": False,
                "score": float("nan"),
                "reason": "no_rind_max_captured",
                "simulation_caveats": [
                    "rind_unavailable_amax_capture_missing_for_this_step",
                ],
            }
        rind = float(v)
        fired = rind > self.rind_threshold
        return {
            "gate_name": self.name,
            "fired": fired,
            "score": rind,
            "reason": f"rind_max={rind:.3f}(thr={self.rind_threshold:.2f})",
            "simulation_caveats": [],
        }

    def metadata(self) -> dict:
        return {
            "gate_name": self.name,
            "rind_threshold": self.rind_threshold,
            "caveats": [],
        }


# Default RSUS weights and threshold from Guo et al. (2026) Footnote 1.
RSUS_WEIGHTS_DEFAULT: dict[str, float] = {"u_verb": 0.40, "u_ent": 0.35, "u_cons": 0.25}
RSUS_THRESHOLD_DEFAULT: float = 0.65


class RealmRsusGate:

    name = "realm_rsus"

    def __init__(
        self,
        rsus_threshold: float = RSUS_THRESHOLD_DEFAULT,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        self.rsus_threshold = float(rsus_threshold)
        self.weights = dict(weights or RSUS_WEIGHTS_DEFAULT)

    def _combine(self, u_verb, u_ent, u_cons) -> tuple[float, dict[str, float]]:
        components = {"u_verb": u_verb, "u_ent": u_ent, "u_cons": u_cons}
        keep = {k: float(v) for k, v in components.items() if v is not None and v == v}
        if not keep:
            return float("nan"), {}
        w_sum = sum(self.weights[k] for k in keep)
        actually = {k: self.weights[k] / w_sum for k in keep}
        score = float(sum(actually[k] * keep[k] for k in keep))
        return score, actually

    def evaluate(self, step_record: Mapping[str, object]) -> dict:
        u_verb = step_record.get("u_verb")
        u_ent = step_record.get("u_ent")
        u_cons = step_record.get("u_cons")
        score, actually = self._combine(u_verb, u_ent, u_cons)
        if score != score:  # NaN
            return {
                "gate_name": self.name,
                "fired": False,
                "score": float("nan"),
                "reason": "all_components_missing",
                "simulation_caveats": [
                    "no_uverb_uent_or_ucons_captured",
                ],
            }
        fired = score > self.rsus_threshold
        parts: list[str] = []
        parts.append(f"u_v={'NA' if u_verb is None else f'{float(u_verb):.3f}'}")
        parts.append(f"u_e={'NA' if u_ent is None else f'{float(u_ent):.3f}'}")
        parts.append(f"u_c={'NA' if u_cons is None else f'{float(u_cons):.3f}'}")
        reason = f"rsus={score:.3f}({', '.join(parts)})"
        caveats = ["u_ent_bm25_proxy_for_bge_reranker", "thresholded_rsus_no_rl_policy"]
        if u_verb is None:
            caveats.append("uverb_missing_weight_redistributed")
        return {
            "gate_name": self.name,
            "fired": fired,
            "score": score,
            "reason": reason,
            "simulation_caveats": caveats,
            "weights_used": actually,
        }

    def metadata(self) -> dict:
        return {
            "gate_name": self.name,
            "rsus_threshold": self.rsus_threshold,
            "weights": dict(self.weights),
            "caveats": [
                "thresholded_rsus_no_rl_policy",
                "u_ent_bm25_proxy_for_bge_reranker",
            ],
        }


class SelfDoubtFullGate:
    """Full SELFDOUBT gate per Pandey et al. (2026), §3.

    Combines the Tier-1 HVR signal with verbalized confidence by per-
    model z-score fusion::

        z_hvr  = (HVR_t  - mu_hvr)  / sigma_hvr
        z_verb = (U_verb - mu_verb) / sigma_verb
        SD_score = (z_hvr + z_verb) / 2
        fired = SD_score > z_threshold

    Distribution moments ``(mu_hvr, sigma_hvr, mu_verb, sigma_verb)`` are
    fitted on the calibration split at threshold-fitting time and frozen
    before validation evaluation (Phase 4.10 in `_PLAN.md`).

    Step record contract:
        - ``hvr``     float — Hedge-to-Verify Ratio
        - ``u_verb``  float | None — verbalized uncertainty (1 - conf)

    Per the paper, missing U_verb falls back to the Tier-1 HVR-only
    decision (``fired = HVR_t > hvr_only_threshold``); the simulator
    caveat is stamped accordingly.
    """

    name = "selfdoubt_full"

    def __init__(
        self,
        z_threshold: float = 1.0,
        hvr_only_threshold: float = 1.0,
        moments: Mapping[str, float] | None = None,
    ) -> None:
        self.z_threshold = float(z_threshold)
        self.hvr_only_threshold = float(hvr_only_threshold)
        # moments dict: {"mu_hvr", "sigma_hvr", "mu_verb", "sigma_verb"}
        self.moments: dict[str, float] = dict(moments or {})

    def fit_moments(
        self,
        calib_step_records: Iterable[Mapping[str, object]],
    ) -> None:
        """Fit (mu, sigma) for HVR and U_verb on the calibration split."""
        hvr_vals: list[float] = []
        verb_vals: list[float] = []
        for s in calib_step_records:
            h = s.get("hvr")
            if h is not None:
                hvr_vals.append(float(h))
            uv = s.get("u_verb")
            if uv is not None:
                verb_vals.append(float(uv))
        import statistics
        if len(hvr_vals) >= 2:
            self.moments["mu_hvr"] = statistics.mean(hvr_vals)
            self.moments["sigma_hvr"] = max(1e-6, statistics.pstdev(hvr_vals))
        if len(verb_vals) >= 2:
            self.moments["mu_verb"] = statistics.mean(verb_vals)
            self.moments["sigma_verb"] = max(1e-6, statistics.pstdev(verb_vals))

    def evaluate(self, step_record: Mapping[str, object]) -> dict:
        h = step_record.get("hvr")
        uv = step_record.get("u_verb")
        # Fall back to Tier-1 if either moment is missing or U_verb absent.
        if (
            uv is None
            or "mu_verb" not in self.moments
            or "mu_hvr" not in self.moments
            or h is None
        ):
            hvr_val = float(h) if h is not None else 0.0
            fired = hvr_val > self.hvr_only_threshold
            return {
                "gate_name": self.name,
                "fired": fired,
                "score": hvr_val,
                "reason": f"fallback_hvr_only(hvr={hvr_val:.3f},thr={self.hvr_only_threshold:.2f})",
                "simulation_caveats": [
                    "uverb_or_moments_missing_fell_back_to_tier1_hvr",
                ],
            }
        z_hvr = (float(h) - self.moments["mu_hvr"]) / self.moments["sigma_hvr"]
        z_verb = (float(uv) - self.moments["mu_verb"]) / self.moments["sigma_verb"]
        sd = (z_hvr + z_verb) / 2.0
        fired = sd > self.z_threshold
        return {
            "gate_name": self.name,
            "fired": fired,
            "score": float(sd),
            "reason": f"sd={sd:.3f}(z_hvr={z_hvr:.2f},z_verb={z_verb:.2f},thr={self.z_threshold:.2f})",
            "simulation_caveats": [
                "marker_dictionary_is_seed_set_not_paper_discovery_pipeline",
            ],
        }

    def metadata(self) -> dict:
        return {
            "gate_name": self.name,
            "z_threshold": self.z_threshold,
            "hvr_only_threshold": self.hvr_only_threshold,
            "moments": dict(self.moments),
            "caveats": [
                "marker_dictionary_is_seed_set_not_paper_discovery_pipeline",
            ],
        }


__all__ = [
    "DISCOURSE_PATTERNS",
    "discourse_marker_match",
    "HEDGE_PATTERNS",
    "VERIFY_PATTERNS",
    "hedge_verify_counts",
    "RealmRetrieveSimGate",
    "GripSimGate",
    "SelfDoubtTier1Gate",
    "EtcGate",
    "DraginGate",
    "RealmRsusGate",
    "SelfDoubtFullGate",
    "RSUS_WEIGHTS_DEFAULT",
    "RSUS_THRESHOLD_DEFAULT",
    "RandomGate",
    "LatentGateAdapter",
]
