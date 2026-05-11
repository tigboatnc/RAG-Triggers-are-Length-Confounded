"""Analysis helpers shared across try1..try5.

No new compute; pure post-hoc on existing trace JSONLs.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))
import importlib
sd = importlib.import_module("04g_sota_showdown")

V1_TRACES = {
    "phi":   ROOT / "runs/phase4_qa_phi-4-mini-instruct_mid_stack_dragin.jsonl",
    "gemma": ROOT / "runs/phase4_qa_gemma-4-26b-a4b-it_mid_stack_dragin.jsonl",
    "qwen":  ROOT / "runs/phase4_qa_qwen3.6-35b-a3b-instruct_mid_stack_dragin.jsonl",
}
V2_TRACES = {
    "phi_hotpot_v2": ROOT / "runs/v2_exp1_phi-4-mini-instruct/D1_closed_hotpot.jsonl",
    "phi_2wiki_v2":  ROOT / "runs/v2_exp1_phi-4-mini-instruct/D2_closed_wiki.jsonl",
}
ALL_TRACES = {**V1_TRACES, **V2_TRACES}

GATES = ("drift", "variance", "or", "etc", "hvr", "dragin")


def load_traces(name: str, min_steps: int = 3) -> list[dict]:
    rows = [json.loads(l) for l in ALL_TRACES[name].open() if l.strip()]
    return [r for r in rows if int(r.get("n_steps", 0)) >= min_steps]


def label(r: dict) -> bool:
    return bool(r.get("label_correct_em", r.get("em", False)))


def pool_step_scores(traces: list[dict], key: str) -> list[float]:
    out = []
    for r in traces:
        sc = sd._trace_scores(r)
        out.extend(float(v) for v in sc[key] if v is not None)
    return out


def trace_max_score(t: dict, key: str) -> float | None:
    sc = sd._trace_scores(t)
    vs = [float(v) for v in sc[key] if v is not None]
    return max(vs) if vs else None


def auroc(scores_pos: list[float], scores_neg: list[float]) -> float:
    """AUROC via Mann-Whitney U / (n_pos * n_neg). scores_pos = scores for the
    'positive' class (we use class=INCORRECT as positive throughout)."""
    if not scores_pos or not scores_neg:
        return float("nan")
    combined = [(s, 1) for s in scores_pos] + [(s, 0) for s in scores_neg]
    combined.sort(key=lambda x: x[0])
    n = len(combined)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    R_pos = sum(r for r, (_, lbl) in zip(ranks, combined) if lbl == 1)
    n_p = len(scores_pos); n_n = len(scores_neg)
    U = R_pos - n_p * (n_p + 1) / 2
    return U / (n_p * n_n)


def quantile(xs: list[float], p: float) -> float:
    if not xs: return float("nan")
    s = sorted(xs); i = int(p * (len(s) - 1))
    return s[i]


def fit_threshold(traces: list[dict], key: str, target: float,
                  granularity: str = "trace",
                  candidates: tuple = (.50,.60,.70,.75,.80,.85,.90,.93,.95,.97,.99)) -> float:
    """Pick τ so that at the chosen granularity the fire rate is closest to target.
    Returns 0.0 if no scores."""
    if granularity == "step":
        xs = pool_step_scores(traces, key)
        if not xs: return 0.0
        # In step-iso mode, simply pick the (1-target)-quantile of step scores.
        return quantile(xs, 1.0 - target)
    elif granularity == "trace":
        xs = pool_step_scores(traces, key)
        if not xs: return 0.0
        per_trace = [[float(v) for v in sd._trace_scores(t).get(key, []) if v is not None] for t in traces]
        cands = sorted({quantile(xs, q) for q in candidates})
        best, best_diff = cands[-1], float("inf")
        for thr in cands:
            fired = sum(1 for vs in per_trace if any(v > thr for v in vs))
            rate = fired / len(traces)
            d = abs(rate - target)
            if d < best_diff:
                best_diff, best = d, thr
        return float(best)
    else:
        raise ValueError(granularity)


def gate_fire_per_trace(traces: list[dict], gate: str, thresholds: dict,
                        rng: random.Random | None = None, random_p: float = 0.30,
                        latent_mode: str = "or") -> list[bool]:
    """Returns list of bool per trace: did the gate fire (at least one step)?"""
    out = []
    for t in traces:
        sc = sd._trace_scores(t)
        if gate == "drift":
            fired = any((v is not None and v > thresholds["drift"]) for v in sc["drift"])
        elif gate == "variance":
            fired = any((v is not None and v > thresholds["variance"]) for v in sc["variance"])
        elif gate == "or":
            fired = any((sc["drift"][i] > thresholds["drift"]) or (sc["variance"][i] > thresholds["variance"])
                        for i in range(len(sc["drift"])))
        elif gate == "etc":
            fired = any((v is not None and v > thresholds["etc"]) for v in sc["etc"])
        elif gate == "hvr":
            fired = any((v is not None and v > 0.0) for v in sc["hvr"])  # native Pandey
        elif gate == "dragin":
            fired = any((v is not None and v > thresholds["dragin"]) for v in sc["dragin"])
        elif gate == "random":
            if rng is None: rng = random.Random(0)
            n = int(t.get("n_steps", len(sc["drift"])))
            fired = False
            for _ in range(n):
                if rng.random() < random_p:
                    fired = True; break
        else:
            raise ValueError(gate)
        out.append(fired)
    return out


def delta_frac(fires: list[bool], labels: list[bool]) -> tuple[float, dict]:
    nF = nNF = nCF = nCNF = 0
    for f, c in zip(fires, labels):
        if f:
            nF += 1; nCF += int(c)
        else:
            nNF += 1; nCNF += int(c)
    if nF == 0 or nNF == 0:
        return float("nan"), {"n_fired": nF, "n_not_fired": nNF, "fire_rate": nF/len(fires) if fires else 0,
                              "p_c_f": float("nan"), "p_c_nf": float("nan")}
    pCF = nCF / nF; pCNF = nCNF / nNF
    return pCNF - pCF, {"n_fired": nF, "n_not_fired": nNF,
                        "fire_rate": nF / len(fires),
                        "p_c_f": pCF, "p_c_nf": pCNF}


def split_traces(traces: list[dict]) -> dict[str, list[dict]]:
    """v1 traces have split ∈ {calibration, validation, test}. Return dict."""
    out = {"calibration": [], "validation": [], "test": []}
    for r in traces:
        s = r.get("split", "calibration")
        if s not in out: out[s] = []
        out[s].append(r)
    return out
