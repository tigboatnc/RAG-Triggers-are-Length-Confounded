"""Phase 4 / #23 follow-up — lift-conditional gate diagnostics.

Reuses the data produced by 04l + #20 closed-book traces. Rather than
asking "does gate-triggered retrieval beat never-retrieve?" (which is
hopeless when the model already solves the closed-book task), asks:

  "Among traces where the gate fires, is the per-trace EM/F1 lift from
   oracle-retrieve higher than among traces where the gate does not fire?"

This is the lift-Δfrac analogue of the original Δfrac analysis — it
asks whether the gate selectively fires on the traces where retrieval
would actually help, controlling for "model already knew the answer".

Per-gate metrics reported (per split, full + n_steps>=3 sensitivity):

  lift_label per trace:
    +1  oracle correct, closed-book incorrect  (wrong → right)
     0  same EM either way                      (no change)
    -1  closed-book correct, oracle incorrect  (right → wrong)

  recall_em     = P(fires | lift_em = +1)        — caught helpful retrievals
  fpr_em        = P(fires | lift_em <= 0)        — fired when not helpful
  precision_em  = P(lift_em = +1 | fires)        — of fired traces, helpful frac
  harm_rate     = P(lift_em = -1 | fires)        — of fired traces, harmful frac

  mean_lift_em(fired) − mean_lift_em(not_fired)  paired bootstrap CI
  mean_lift_f1(fired) − mean_lift_f1(not_fired)  paired bootstrap CI

If gate_latent has paired lift margin ✱>0 (its fired-cohort has higher
mean lift than its not-fired cohort) AND gate_random does not, we have a
viable rescue narrative on the existing data with no new compute.

Usage:
    python scripts/04n_lift_conditional.py --model qwen3.6-35b-a3b-instruct
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from importlib import import_module
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sd = import_module("04g_sota_showdown")  # type: ignore

GATES = ["latent", "etc", "selfdoubt_t1", "dragin", "random"]


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _bootstrap_lift_margin(per_trace: list[dict], gate: str, metric: str,
                           B: int, rng: random.Random) -> dict:
    """Paired bootstrap on the trace index. Each resample computes
    mean_lift_fired − mean_lift_not_fired for the named gate / metric.
    Reports point + 95% CI + P(>0).
    """
    n = len(per_trace)
    if n == 0:
        return {"point": None, "ci_lo": None, "ci_hi": None, "p_gt_zero": None,
                "n_fired": 0, "n_not_fired": 0}

    def _margin(idx: list[int]) -> float | None:
        fired_vals, not_vals = [], []
        for i in idx:
            t = per_trace[i]
            v = t["lift"][metric]
            if t["fires"][gate]:
                fired_vals.append(v)
            else:
                not_vals.append(v)
        if not fired_vals or not not_vals:
            return None
        return (sum(fired_vals)/len(fired_vals)) - (sum(not_vals)/len(not_vals))

    point = _margin(list(range(n)))
    n_fired = sum(1 for t in per_trace if t["fires"][gate])
    pts: list[float] = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        m = _margin(idx)
        if m is not None:
            pts.append(m)
    if not pts:
        return {"point": point, "ci_lo": None, "ci_hi": None, "p_gt_zero": None,
                "n_fired": n_fired, "n_not_fired": n - n_fired}
    return {
        "point": point,
        "ci_lo": sd._pct(pts, 0.025), "ci_hi": sd._pct(pts, 0.975),
        "p_gt_zero": sum(1 for x in pts if x > 0) / len(pts),
        "n_fired": n_fired, "n_not_fired": n - n_fired,
    }


def _confusion(per_trace: list[dict], gate: str) -> dict:
    helpful = sum(1 for t in per_trace if t["lift"]["em"] == 1)
    harmful = sum(1 for t in per_trace if t["lift"]["em"] == -1)
    neutral = sum(1 for t in per_trace if t["lift"]["em"] == 0)
    fired = [t for t in per_trace if t["fires"][gate]]
    not_fired = [t for t in per_trace if not t["fires"][gate]]
    fired_helpful = sum(1 for t in fired if t["lift"]["em"] == 1)
    fired_harmful = sum(1 for t in fired if t["lift"]["em"] == -1)
    fired_neutral = sum(1 for t in fired if t["lift"]["em"] == 0)
    n = len(per_trace)
    return {
        "n": n,
        "lift_distribution": {"helpful_+1": helpful, "neutral_0": neutral, "harmful_-1": harmful},
        "n_fired": len(fired), "n_not_fired": len(not_fired),
        "recall_em":    (fired_helpful / helpful) if helpful else None,
        "fpr_em":       (fired_harmful + fired_neutral) / (harmful + neutral) if (harmful + neutral) else None,
        "precision_em": (fired_helpful / len(fired)) if fired else None,
        "harm_rate":    (fired_harmful / len(fired)) if fired else None,
        "base_rate_helpful": helpful / n if n else None,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3.6-35b-a3b-instruct")
    p.add_argument("--closed-book", default="runs/phase4_qa_{model}_mid_stack_dragin.jsonl")
    p.add_argument("--oracle",      default="runs/phase4_qa_{model}_oracle_retrieve.jsonl")
    p.add_argument("--bootstrap-json", default="runs/phase4_bootstrap_dragin.json")
    p.add_argument("--out", default="runs/phase4_lift_conditional.json")
    p.add_argument("-B", "--bootstrap", type=int, default=10000)
    p.add_argument("--min-steps", type=int, default=0,
                   help="0 = full cohort (primary), 3 = match Δfrac (sensitivity).")
    p.add_argument("--seed", type=int, default=20260511)
    args = p.parse_args()

    closed_rows = _load_jsonl(Path(args.closed_book.format(model=args.model)))
    oracle_rows = _load_jsonl(Path(args.oracle.format(model=args.model)))
    oracle_by_id = {r["id"]: r for r in oracle_rows if "em" in r}

    with Path(args.bootstrap_json).open() as f:
        boot20 = json.load(f)
    arch = boot20["per_arch"][args.model]
    thresholds, latent_mode = arch["thresholds"], arch["latent_mode_chosen"]
    random_p = float(arch.get("random_p", 0.30))

    rng = random.Random(args.seed)

    by_split: dict[str, list[dict]] = {"calibration": [], "validation": [], "test": []}
    for r in closed_rows:
        split = r.get("split")
        if split not in by_split:
            continue
        if int(r.get("n_steps", 0)) < args.min_steps:
            continue
        orow = oracle_by_id.get(r["id"])
        if not orow or "em" not in orow:
            continue
        pt = sd.firsts_per_trace([r], thresholds, latent_mode, random_p, rng)[0]
        fires = {g: (pt["first_fire"].get(g) is not None) for g in GATES}
        c_em, c_f1 = int(r["em"]), float(r["f1"])
        o_em, o_f1 = int(orow["em"]), float(orow["f1"])
        by_split[split].append({
            "id": r["id"],
            "closed": {"em": c_em, "f1": c_f1},
            "oracle": {"em": o_em, "f1": o_f1},
            "lift":   {"em": o_em - c_em, "f1": o_f1 - c_f1},
            "fires":  fires,
        })

    print(f"[..] model={args.model}  B={args.bootstrap}  min_steps={args.min_steps}")
    for split in ("calibration", "validation", "test"):
        print(f"     n_{split}={len(by_split[split])}")
    print()

    result: dict = {"args": vars(args), "per_split": {}}
    for split in ("calibration", "validation", "test"):
        rows = by_split[split]
        if not rows:
            continue
        result["per_split"][split] = {"n": len(rows), "per_gate": {}}
        conf_any = _confusion(rows, "latent")
        helpful = conf_any["lift_distribution"]["helpful_+1"]
        neutral = conf_any["lift_distribution"]["neutral_0"]
        harmful = conf_any["lift_distribution"]["harmful_-1"]

        print(f"--- {split.upper()} (n={len(rows)}) ---")
        print(f"  lift_em distribution:  helpful(+1)={helpful}  neutral(0)={neutral}  harmful(-1)={harmful}")
        print(f"  base rate of helpful retrieval: {helpful/len(rows)*100:.1f}%")
        print()
        print(f"  {'gate':14}  fires%  rec_em  prec_em  fpr_em  harm   "
              f"ΔlEM(fired−nf)    95% CI                 P>0   "
              f"ΔlF1(fired−nf)    95% CI                 P>0")
        for g in GATES:
            conf = _confusion(rows, g)
            m_em = _bootstrap_lift_margin(rows, g, "em", args.bootstrap, rng)
            m_f1 = _bootstrap_lift_margin(rows, g, "f1", args.bootstrap, rng)
            fr = conf["n_fired"] / len(rows) * 100
            def _fmtpct(x): return "n/a" if x is None else f"{x*100:5.1f}"
            def _fmtm(x):   return "n/a" if x is None else f"{x*100:+6.2f}"
            def _fmtp(x):   return "n/a" if x is None else f"{x:.3f}"
            print(f"  {g:14}  {fr:5.1f}  {_fmtpct(conf['recall_em'])}  "
                  f"{_fmtpct(conf['precision_em'])}   {_fmtpct(conf['fpr_em'])}  "
                  f"{_fmtpct(conf['harm_rate'])}   "
                  f"{_fmtm(m_em['point'])}   "
                  f"[{_fmtm(m_em['ci_lo'])}, {_fmtm(m_em['ci_hi'])}]   "
                  f"{_fmtp(m_em['p_gt_zero'])}   "
                  f"{_fmtm(m_f1['point'])}   "
                  f"[{_fmtm(m_f1['ci_lo'])}, {_fmtm(m_f1['ci_hi'])}]   "
                  f"{_fmtp(m_f1['p_gt_zero'])}")
            result["per_split"][split]["per_gate"][g] = {
                "fire_rate": fr/100, "confusion": conf,
                "margin_em": m_em, "margin_f1": m_f1,
            }
        print()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[ok] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
