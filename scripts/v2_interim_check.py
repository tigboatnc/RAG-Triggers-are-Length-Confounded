"""Interim F-style analyzer for v2 Experiment 1.

Reads C2_calibration.json + D1/D2 closed-book partial JSONLs and reports per-gate
fire rate, Δfrac proxy (P(correct|not fired) − P(correct|fired)), and a paired
bootstrap margin of latent − baseline on Δfrac. Does NOT need E1/E2; the
lift-conditional headline replaces this proxy with P(oracle helped | fired)
later, but the gate-ranking structure on Δfrac is usually directionally
predictive.

Usage (analysis box, no GPU, gzn env):

    python scripts/v2_interim_check.py \
        --run-dir runs/v2_exp1_phi-4-mini-instruct
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sd = __import__("04g_sota_showdown")

GATES = ("latent_drift", "latent_variance", "latent_or",
         "etc", "selfdoubt_t1", "dragin", "random")
BASELINES = ("etc", "selfdoubt_t1", "dragin", "random")
LATENT_MODES = ("drift", "variance", "or")


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def _firsts_multi(traces: list[dict], thresholds: dict, random_p: float,
                  seed: int) -> list[dict]:
    """Like sd.firsts_per_trace but emits first-fire indices for ALL three
    latent modes (drift / variance / or) plus etc / selfdoubt_t1 / dragin /
    random under the same frozen thresholds. No re-selection — same τ_drift
    and τ_variance applied with a different aggregation rule.
    """
    rng = random.Random(seed)
    out: list[dict] = []
    for t in traces:
        sc = sd._trace_scores(t)
        n_steps = int(t.get("n_steps", len(sc["drift"])))
        ff: dict[str, int | None] = {}
        for mode in LATENT_MODES:
            ff[f"latent_{mode}"] = sd._latent_first_fire(
                sc, mode, thresholds["drift"], thresholds["variance"])
        ff["etc"] = sd._first_fire_index(sc["etc"], thresholds["etc"])
        ff["selfdoubt_t1"] = sd._first_fire_index(sc["hvr"], thresholds["selfdoubt_t1"])
        ff["dragin"] = sd._first_fire_index(sc["dragin"], thresholds["dragin"])
        rfire = None
        for i in range(n_steps):
            if rng.random() < random_p:
                rfire = i + 1; break
        ff["random"] = rfire
        out.append({"id": t.get("id"), "n_steps": n_steps,
                    "label_correct": bool(t.get("label_correct_em", t.get("em", False))),
                    "first_fire": ff})
    return out


def _delta_frac(firsts: list[dict], gate: str) -> tuple[float, int, int, int, int]:
    """Return (Δfrac, n_fired, n_correct_fired, n_not_fired, n_correct_not_fired)."""
    nF = nCF = nNF = nCNF = 0
    for f in firsts:
        c = bool(f["label_correct"])
        fired = f["first_fire"].get(gate) is not None
        if fired:
            nF += 1; nCF += int(c)
        else:
            nNF += 1; nCNF += int(c)
    pCF = nCF / nF if nF else float("nan")
    pCNF = nCNF / nNF if nNF else float("nan")
    if math.isnan(pCF) or math.isnan(pCNF):
        df = float("nan")
    else:
        df = pCNF - pCF  # high = gate fires on incorrect (good)
    return df, nF, nCF, nNF, nCNF


def _paired_bootstrap_delta(firsts: list[dict], gA: str, gB: str,
                            B: int = 10_000, seed: int = 20260511) -> dict:
    """Resample traces with replacement; report Δfrac(gA) − Δfrac(gB) CIs + p."""
    rng = random.Random(seed)
    n = len(firsts)
    idxs = list(range(n))
    diffs = []
    for _ in range(B):
        sample = [firsts[rng.choice(idxs)] for _ in range(n)]
        dA, _, _, _, _ = _delta_frac(sample, gA)
        dB, _, _, _, _ = _delta_frac(sample, gB)
        if not (math.isnan(dA) or math.isnan(dB)):
            diffs.append(dA - dB)
    if not diffs:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "p_two_sided": float("nan"), "n_resamples_kept": 0}
    diffs.sort()
    m = sum(diffs) / len(diffs)
    lo = diffs[int(0.025 * (len(diffs) - 1))]
    hi = diffs[int(0.975 * (len(diffs) - 1))]
    pos = sum(1 for d in diffs if d > 0)
    p_one = min(pos, len(diffs) - pos) / len(diffs)
    p_two = 2 * p_one
    return {"mean": m, "ci_lo": lo, "ci_hi": hi, "p_two_sided": p_two,
            "n_resamples_kept": len(diffs)}


def _mcnemar(firsts: list[dict], gA: str, gB: str) -> dict:
    """McNemar on the binary `gate_fired ↔ label_incorrect` contingency.
    Reports discordant pairs (b, c) and continuity-corrected χ².
    """
    b = c = 0  # b: A fires & incorrect, B not; c: B fires & incorrect, A not
    for f in firsts:
        if f["label_correct"]:
            continue
        a = f["first_fire"].get(gA) is not None
        d = f["first_fire"].get(gB) is not None
        if a and not d: b += 1
        if d and not a: c += 1
    if b + c == 0:
        return {"b": b, "c": c, "chi2_cc": float("nan"), "p_approx": float("nan")}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = math.erfc(math.sqrt(chi2) / math.sqrt(2))
    return {"b": b, "c": c, "chi2_cc": chi2, "p_approx": p}


def analyze(label: str, slice_path: Path, cal: dict) -> dict:
    raw = _load(slice_path)
    n_raw = len(raw)
    rows = [r for r in raw if int(r.get("n_steps", 0)) >= 3]
    n_used = len(rows)
    em = sum(1 for r in rows if r.get("label_correct_em") or r.get("em"))
    em_rate = em / n_used if n_used else float("nan")

    firsts = _firsts_multi(rows, cal["thresholds"],
                           cal.get("random_p", 0.30), seed=20260511)

    per_gate = {}
    for g in GATES:
        df, nF, nCF, nNF, nCNF = _delta_frac(firsts, g)
        per_gate[g] = {
            "fire_rate": nF / n_used if n_used else float("nan"),
            "delta_frac": df,
            "p_correct_given_fired": (nCF / nF) if nF else float("nan"),
            "p_correct_given_not_fired": (nCNF / nNF) if nNF else float("nan"),
            "n_fired": nF, "n_not_fired": nNF,
        }

    chosen = f"latent_{cal['latent_mode_chosen']}"
    paired = {}
    for b in BASELINES:
        paired[b] = {
            "bootstrap": _paired_bootstrap_delta(firsts, chosen, b, B=2000),
            "mcnemar":   _mcnemar(firsts, chosen, b),
        }
    paired_mode_vs_mode = {}
    for a in ("latent_drift", "latent_variance", "latent_or"):
        if a == chosen: continue
        paired_mode_vs_mode[a] = {
            "bootstrap": _paired_bootstrap_delta(firsts, chosen, a, B=2000),
        }

    return {
        "label": label,
        "path": str(slice_path),
        "n_raw": n_raw, "n_used": n_used,
        "closed_book_em_rate": em_rate,
        "latent_mode_chosen": cal["latent_mode_chosen"],
        "per_gate": per_gate,
        "paired_latent_minus_baseline": paired,
        "paired_chosen_minus_other_mode": paired_mode_vs_mode,
    }


def _fmt(x, w=7, p=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return f"{'n/a':>{w}}"
    return f"{x:>{w}.{p}f}"


def render(report: dict) -> str:
    lines = []
    lines.append(f"=== {report['label']}  (n_used={report['n_used']}/{report['n_raw']}, "
                 f"closed-book EM={report['closed_book_em_rate']*100:.1f}%, "
                 f"chosen={report['latent_mode_chosen']}) ===")
    lines.append(f"{'gate':14}  {'fire%':>6}  {'Δfrac':>8}  {'P(C|F)':>8}  {'P(C|¬F)':>8}  {'nF':>4}/{'nNF':>4}")
    for g in GATES:
        d = report["per_gate"][g]
        lines.append(f"{g:14}  {d['fire_rate']*100:>5.1f}%  {_fmt(d['delta_frac'])}  "
                     f"{_fmt(d['p_correct_given_fired'])}  {_fmt(d['p_correct_given_not_fired'])}  "
                     f"{d['n_fired']:>4}/{d['n_not_fired']:>4}")
    lines.append("")
    lines.append("paired Δfrac (chosen latent mode vs other modes — diagnostic, no re-selection):")
    lines.append(f"{'other_mode':16}  {'mean':>8}  {'95% CI':>20}  {'p_boot':>8}")
    for a, d in report["paired_chosen_minus_other_mode"].items():
        bs = d["bootstrap"]
        ci = f"[{bs['ci_lo']:+.3f},{bs['ci_hi']:+.3f}]" if not math.isnan(bs['ci_lo']) else "n/a"
        lines.append(f"{a:16}  {_fmt(bs['mean'], 8)}  {ci:>20}  {_fmt(bs['p_two_sided'], 8)}")
    lines.append("")
    lines.append("paired Δfrac (chosen latent − baseline), B=2000 bootstrap + McNemar on incorrect-only discordants:")
    lines.append(f"{'baseline':14}  {'mean':>8}  {'95% CI':>20}  {'p_boot':>8}  {'p_Bonf':>8}  "
                 f"{'McN b/c':>10}  {'p_McN':>8}")
    k = len(BASELINES)
    alpha = 0.05 / k
    for b in BASELINES:
        bs = report["paired_latent_minus_baseline"][b]["bootstrap"]
        mc = report["paired_latent_minus_baseline"][b]["mcnemar"]
        ci = f"[{bs['ci_lo']:+.3f},{bs['ci_hi']:+.3f}]" if not math.isnan(bs['ci_lo']) else "n/a"
        p = bs["p_two_sided"]
        pflag = "*" if (not math.isnan(p) and p < alpha) else " "
        lines.append(f"{b:14}  {_fmt(bs['mean'], 8)}  {ci:>20}  "
                     f"{_fmt(p, 8)}{pflag}  α={alpha:.4f}  "
                     f"{mc['b']:>4}/{mc['c']:<4}  {_fmt(mc['p_approx'], 8)}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--save-json", default=None,
                   help="Optional path to dump structured report")
    args = p.parse_args()
    run = Path(args.run_dir)
    cal = json.loads((run / "C2_calibration.json").read_text())

    reports = []
    for label, name in (("HotpotQA (in-domain)", "D1_closed_hotpot.jsonl"),
                        ("2WikiMHQA (OOD)",       "D2_closed_wiki.jsonl")):
        path = run / name
        if not path.exists():
            print(f"[skip] {name} missing"); continue
        rep = analyze(label, path, cal)
        reports.append(rep)
        print(render(rep)); print()

    if args.save_json and reports:
        out = {"calibration": cal, "reports": reports}
        Path(args.save_json).write_text(json.dumps(out, indent=2))
        print(f"[ok] wrote {args.save_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
