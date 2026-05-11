"""Phase 4 / #23 — Downstream-QA utility scoring (offline).

Given:
  - closed-book traces from #20 (per-question EM/F1 + per-gate fire info)
  - oracle-retrieve answers from 04l (per-question EM/F1 under retrieval)

Compute per-policy {EM, F1} aggregates, where a policy decides for each
trace whether to use the closed-book or oracle-retrieve answer:

  never_retrieve      : closed-book always.
  always_retrieve     : oracle-retrieve always (ceiling, fires 100%).
  gate_<name>         : oracle-retrieve iff the gate fires on the trace,
                        closed-book otherwise.

Gates considered (with thresholds frozen from
`runs/phase4_bootstrap_dragin.json` per #20):
  - latent           (drift OR variance — Qwen mode)
  - etc
  - selfdoubt_t1
  - dragin
  - random           (matched fire rate, fixed seed)

Bootstrap recipe matches #20 / #22b:
  - 10,000-resample paired percentile.
  - Eligibility filter: `n_steps >= 3` for the policy comparison so the
    cohort matches the Δfrac analyses (otherwise gates can never fire on
    short traces and the policy collapses to never_retrieve).
  - Reported per split (calibration / validation / test).

Outputs:
  - `runs/phase4_utility.json`: full structured results.
  - stdout: pretty table per split.

Usage:
    /Users/sanegi/miniconda3/envs/gzn/bin/python scripts/04m_utility_scoring.py \\
        --model qwen3.6-35b-a3b-instruct
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

DEFAULT_BOOTSTRAP = 10000
DEFAULT_MIN_STEPS = 3
DEFAULT_SEED = 20260511

GATES = ["latent", "etc", "selfdoubt_t1", "dragin", "random"]
POLICIES = ["never", "always"] + [f"gate_{g}" for g in GATES]


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _gate_fires(trace: dict, thresholds: dict, latent_mode: str,
                rng_random_p: float, rng: random.Random) -> dict[str, bool]:
    """Did each gate fire ANYWHERE in the trace? Returns one bool per gate."""
    pt = sd.firsts_per_trace([trace], thresholds, latent_mode, rng_random_p, rng)[0]
    return {g: (pt["first_fire"].get(g) is not None) for g in GATES}


def _policy_outcome(closed_em: int, closed_f1: float,
                    oracle_em: int, oracle_f1: float,
                    fires: dict[str, bool]) -> dict[str, tuple[int, float]]:
    out: dict[str, tuple[int, float]] = {
        "never":  (closed_em, closed_f1),
        "always": (oracle_em, oracle_f1),
    }
    for g in GATES:
        if fires.get(g, False):
            out[f"gate_{g}"] = (oracle_em, oracle_f1)
        else:
            out[f"gate_{g}"] = (closed_em, closed_f1)
    return out


def _bootstrap_paired(per_trace_em: dict[str, list[int]],
                      per_trace_f1: dict[str, list[float]],
                      B: int, rng: random.Random) -> dict[str, dict]:
    """Paired percentile bootstrap on the trace index, for every policy.

    Reports point estimate + 95% CI for EM and F1 of each policy, plus
    paired margins (policy − never_retrieve, policy − always_retrieve).
    """
    n = len(next(iter(per_trace_em.values())))
    out: dict[str, dict] = {p: {} for p in POLICIES}
    out["__margins__"] = {}

    em_point = {p: (sum(per_trace_em[p]) / n if n else 0.0) for p in POLICIES}
    f1_point = {p: (sum(per_trace_f1[p]) / n if n else 0.0) for p in POLICIES}

    em_boot: dict[str, list[float]] = {p: [] for p in POLICIES}
    f1_boot: dict[str, list[float]] = {p: [] for p in POLICIES}
    margins_keys = [
        (p, "never")  for p in POLICIES if p != "never"
    ] + [
        (p, "always") for p in POLICIES if p != "always"
    ]
    margin_boot: dict[tuple[str, str], list[float]] = {k: [] for k in margins_keys}

    idx_range = list(range(n))
    for _ in range(B):
        idx = [rng.randrange(n) for _ in idx_range]
        em_b = {p: sum(per_trace_em[p][i] for i in idx) / n for p in POLICIES}
        f1_b = {p: sum(per_trace_f1[p][i] for i in idx) / n for p in POLICIES}
        for p in POLICIES:
            em_boot[p].append(em_b[p])
            f1_boot[p].append(f1_b[p])
        for (a, b) in margins_keys:
            margin_boot[(a, b)].append(f1_b[a] - f1_b[b])

    for p in POLICIES:
        em_lo, em_hi = sd._pct(em_boot[p], 0.025), sd._pct(em_boot[p], 0.975)
        f1_lo, f1_hi = sd._pct(f1_boot[p], 0.025), sd._pct(f1_boot[p], 0.975)
        out[p] = {
            "em_point": em_point[p], "em_ci": [em_lo, em_hi],
            "f1_point": f1_point[p], "f1_ci": [f1_lo, f1_hi],
        }
    for (a, b) in margins_keys:
        pts = margin_boot[(a, b)]
        lo, hi = sd._pct(pts, 0.025), sd._pct(pts, 0.975)
        point = (f1_point[a] - f1_point[b])
        p_gt_zero = sum(1 for x in pts if x > 0) / len(pts)
        out["__margins__"][f"{a}_minus_{b}_f1"] = {
            "point": point, "ci": [lo, hi], "p_gt_zero": p_gt_zero,
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3.6-35b-a3b-instruct")
    p.add_argument("--closed-book",
                   default="runs/phase4_qa_{model}_mid_stack_dragin.jsonl")
    p.add_argument("--oracle",
                   default="runs/phase4_qa_{model}_oracle_retrieve.jsonl")
    p.add_argument("--bootstrap-json", default="runs/phase4_bootstrap_dragin.json")
    p.add_argument("--out", default="runs/phase4_utility.json")
    p.add_argument("-B", "--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    p.add_argument("--min-steps", type=int, default=0,
                   help="0 = full 200 (primary); 3 = match Δfrac analyses (sensitivity).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args()

    closed_path = Path(args.closed_book.format(model=args.model))
    oracle_path = Path(args.oracle.format(model=args.model))
    if not closed_path.exists():
        print(f"[err] missing closed-book traces: {closed_path}")
        return 1
    if not oracle_path.exists():
        print(f"[err] missing oracle-retrieve answers: {oracle_path}")
        print(f"       run scripts/04l_oracle_retrieve_decode.py on the compute box first.")
        return 1

    closed_rows = _load_jsonl(closed_path)
    oracle_rows = _load_jsonl(oracle_path)
    oracle_by_id = {r["id"]: r for r in oracle_rows if "em" in r}

    with Path(args.bootstrap_json).open() as f:
        boot20 = json.load(f)
    arch = boot20["per_arch"].get(args.model)
    if arch is None:
        print(f"[err] no thresholds for {args.model} in {args.bootstrap_json}")
        return 1
    thresholds = arch["thresholds"]
    latent_mode = arch["latent_mode_chosen"]
    random_p = float(arch.get("random_p", 0.30))

    rng = random.Random(args.seed)

    rows_by_split: dict[str, list[dict]] = {"calibration": [], "validation": [], "test": []}
    n_no_oracle = 0
    n_too_short = 0

    for r in closed_rows:
        split = r.get("split")
        if split not in rows_by_split:
            continue
        if int(r.get("n_steps", 0)) < args.min_steps:
            n_too_short += 1
            continue
        orow = oracle_by_id.get(r["id"])
        if orow is None or "em" not in orow:
            n_no_oracle += 1
            continue
        fires = _gate_fires(r, thresholds, latent_mode, random_p, rng)
        per_pol = _policy_outcome(int(r["em"]), float(r["f1"]),
                                  int(orow["em"]), float(orow["f1"]),
                                  fires)
        rows_by_split[split].append({
            "id": r["id"],
            "n_steps": int(r.get("n_steps", 0)),
            "closed_em": int(r["em"]), "closed_f1": float(r["f1"]),
            "oracle_em": int(orow["em"]), "oracle_f1": float(orow["f1"]),
            "fires": fires, "per_policy": per_pol,
        })

    print(f"[..] model={args.model}  B={args.bootstrap}  min_steps={args.min_steps}")
    print(f"     n_calib={len(rows_by_split['calibration'])}  "
          f"n_val={len(rows_by_split['validation'])}  "
          f"n_test={len(rows_by_split['test'])}  "
          f"dropped_short={n_too_short}  missing_oracle={n_no_oracle}")
    print()

    result: dict = {
        "args": vars(args),
        "thresholds_from_20": thresholds,
        "latent_mode": latent_mode,
        "random_p": random_p,
        "per_split": {},
    }

    for split in ("calibration", "validation", "test"):
        rows = rows_by_split[split]
        if not rows:
            continue
        per_pol_em: dict[str, list[int]] = {p: [] for p in POLICIES}
        per_pol_f1: dict[str, list[float]] = {p: [] for p in POLICIES}
        fire_counts = {g: 0 for g in GATES}
        for r in rows:
            for p in POLICIES:
                em, f1 = r["per_policy"][p]
                per_pol_em[p].append(int(em))
                per_pol_f1[p].append(float(f1))
            for g in GATES:
                fire_counts[g] += int(r["fires"][g])

        ci = _bootstrap_paired(per_pol_em, per_pol_f1, args.bootstrap, rng)
        result["per_split"][split] = {
            "n": len(rows),
            "trace_fire_rates": {g: fire_counts[g] / len(rows) for g in GATES},
            "per_policy": {p: ci[p] for p in POLICIES},
            "margins": ci["__margins__"],
        }

        print(f"--- {split.upper()} (n={len(rows)}) ---")
        print(f"  trace fire rates: " + "  ".join(f"{g}={fire_counts[g]/len(rows)*100:.0f}%" for g in GATES))
        print()
        print(f"  {'policy':24}  fired%  {'EM':>6}  EM 95% CI            {'F1':>6}  F1 95% CI")
        for p in POLICIES:
            fr = ""
            if p.startswith("gate_"):
                g = p[len("gate_"):]
                fr = f"{fire_counts[g]/len(rows)*100:5.1f}"
            elif p == "always":
                fr = "100.0"
            elif p == "never":
                fr = "  0.0"
            d = ci[p]
            print(f"  {p:24}  {fr:>6}  {d['em_point']*100:5.1f}  "
                  f"[{d['em_ci'][0]*100:5.1f}, {d['em_ci'][1]*100:5.1f}]   "
                  f"{d['f1_point']*100:5.1f}  [{d['f1_ci'][0]*100:5.1f}, {d['f1_ci'][1]*100:5.1f}]")

        print()
        print(f"  --- Paired F1 margins (policy − reference, paired across traces) ---")
        for (a, b) in [(f"gate_{g}", "never")  for g in GATES] + \
                      [(f"gate_{g}", "always") for g in GATES] + \
                      [("always",  "never")]:
            key = f"{a}_minus_{b}_f1"
            m = ci["__margins__"].get(key)
            if m is None: continue
            lo, hi = m["ci"]
            sig = sd._sig(lo, hi)
            print(f"  {a:14} − {b:8}  ΔF1={m['point']*100:+6.2f}  "
                  f"[{lo*100:+6.2f}, {hi*100:+6.2f}]   P(>0)={m['p_gt_zero']:.3f}   {sig}")
        print()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[ok] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
