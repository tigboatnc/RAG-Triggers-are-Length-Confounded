"""Phase 4 / #21 — OOD transfer test on 2WikiMultiHopQA-distractor.

Applies Qwen3.6-35B-A3B thresholds **frozen on HotpotQA-distractor
calibration (#20)** zero-shot to the 2Wiki-distractor traces. No
threshold fitting, no mode selection. The latent aggregation mode (OR),
the four scalar thresholds, and the random-gate firing rate are read
from ``runs/phase4_bootstrap_dragin.json`` and applied to the new
trace file unchanged.

This script is L5-compliant by construction: every parameter that
could otherwise be fit on the OOD slice is loaded from the frozen
HotpotQA-calibration record, and the OOD slice has no
calibration/validation rows on disk (all 200 questions are tagged
``split = "ood_test"``).

Pre-registered in ``_AGENT.md`` #21 before any 2Wiki traces were
generated. The success / partial / null criteria in that entry are the
template the writeup should follow when reporting these numbers.

Reuses the analysis primitives from ``04g_sota_showdown.py`` (the #20
re-bootstrap) so the bootstrap recipe is identical: same paired
percentile bootstrap, same Δfrac sign convention, same eligibility
filter, same random seed.

Usage:
    python scripts/04i_ood_transfer.py
    python scripts/04i_ood_transfer.py --bootstrap 20000
    python scripts/04i_ood_transfer.py --traces runs/phase4_qa_qwen3.6-35b-a3b-instruct_mid_stack_2wiki_smoke.jsonl
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

# Reuse the canonical analysis primitives from #20.
sd = import_module("04g_sota_showdown")  # type: ignore

POOL = sd.POOL  # ["latent", "etc", "selfdoubt_t1", "dragin", "random"]
QWEN_KEY = "qwen3.6-35b-a3b-instruct"

DEFAULT_BOOTSTRAP_JSON = "runs/phase4_bootstrap_dragin.json"
DEFAULT_TRACES = "runs/phase4_qa_qwen3.6-35b-a3b-instruct_mid_stack_2wiki_ood.jsonl"
DEFAULT_OUT = "runs/phase4_ood_transfer_2wiki.json"


def _load_frozen_config(bootstrap_path: Path, arch_key: str) -> dict:
    """Read frozen Qwen thresholds + latent mode + random_p from the #20
    bootstrap output. Raises with a clear message if the file or arch entry
    is missing."""
    if not bootstrap_path.exists():
        raise FileNotFoundError(
            f"missing {bootstrap_path}. Run scripts/04g_sota_showdown.py first "
            f"to produce the frozen calibration record."
        )
    with bootstrap_path.open() as f:
        boot = json.load(f)
    per_arch = boot.get("per_arch") or {}
    if arch_key not in per_arch:
        raise KeyError(
            f"{bootstrap_path} has no entry for arch '{arch_key}'. "
            f"Available: {sorted(per_arch.keys())}"
        )
    rec = per_arch[arch_key]
    return {
        "thresholds": rec["thresholds"],
        "latent_mode": rec["latent_mode_chosen"],
        "random_p": float(rec.get("random_p", 0.30)),
        "calib_n_eligible": rec.get("n_calibration_eligible"),
        "calib_mode_scores": rec.get("latent_mode_calib_scores"),
    }


def _load_traces(traces_path: Path) -> list[dict]:
    if not traces_path.exists():
        raise FileNotFoundError(
            f"missing {traces_path}. Run scripts/04f_run_qa_sota.py on "
            f"data/2wikimultihop_distractor_200.jsonl first."
        )
    rows: list[dict] = []
    with traces_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bootstrap-json", default=DEFAULT_BOOTSTRAP_JSON,
                   help="Frozen HotpotQA-calibration record (entry #20).")
    p.add_argument("--traces", default=DEFAULT_TRACES,
                   help="2Wiki OOD trace JSONL produced by 04f_run_qa_sota.py.")
    p.add_argument("--arch", default=QWEN_KEY,
                   help="Arch key whose frozen config to apply (default: Qwen3.6).")
    p.add_argument("--bootstrap", "-B", type=int, default=10000)
    p.add_argument("--min-steps", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260511)
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()

    bootstrap_path = Path(args.bootstrap_json)
    traces_path = Path(args.traces)
    out_path = Path(args.out)

    frozen = _load_frozen_config(bootstrap_path, args.arch)
    rows = _load_traces(traces_path)
    eligible = [r for r in rows if int(r.get("n_steps", 0)) >= args.min_steps]

    rng = random.Random(args.seed)

    print(f"[..] OOD transfer test on 2WikiMultiHopQA-distractor")
    print(f"     arch                   = {args.arch}")
    print(f"     frozen thresholds from = {bootstrap_path}")
    print(f"     traces                 = {traces_path}  ({len(rows)} rows, "
          f"{len(eligible)} eligible at min_steps={args.min_steps})")
    print(f"     latent_mode (frozen)   = {frozen['latent_mode']}")
    print(f"     calib n_eligible       = {frozen['calib_n_eligible']}")
    print(f"     thresholds (frozen):")
    for k in ("drift", "variance", "etc", "selfdoubt_t1", "dragin"):
        print(f"       {k:13} > {frozen['thresholds'][k]:.4f}")
    print(f"     random_p (frozen)      = {frozen['random_p']}")
    print(f"     bootstrap B            = {args.bootstrap}")
    print(f"     seed                   = {args.seed}")
    print()

    if not eligible:
        print("[!!] no eligible traces; aborting.")
        return 2

    # Per-trace first-fire under frozen config.
    pt = sd.firsts_per_trace(
        eligible,
        thresholds=frozen["thresholds"],
        latent_mode=frozen["latent_mode"],
        random_p=frozen["random_p"],
        rng=rng,
    )

    # Per-gate Δfrac + bootstrap CI.
    print(f"  --- 2WIKI OOD (n={len(pt)}, frozen HotpotQA-cal config) ---")
    print(f"    {'gate':14}  fire%   point      95% CI                 sig")
    per_gate: dict[str, dict] = {}
    for g in POOL:
        fired = sum(1 for t in pt if t["first_fire"].get(g) is not None)
        fire_rate_trace = fired / len(pt) if pt else 0.0
        ci = sd.boot_ci(pt, g, args.bootstrap, rng)
        ci["trace_fire_rate"] = fire_rate_trace
        per_gate[g] = ci
        print(f"    {g:14}  {fire_rate_trace*100:5.1f}  {sd._fmt(ci['point'])}   "
              f"[{sd._fmt(ci['ci_lo'])}, {sd._fmt(ci['ci_hi'])}]   "
              f"{sd._sig(ci['ci_lo'], ci['ci_hi'])}")

    print(f"    --- paired margins (latent − baseline) ---")
    margins: dict[str, dict] = {}
    for opp in ("etc", "selfdoubt_t1", "dragin", "random"):
        mci = sd.boot_margin_ci(pt, "latent", opp, args.bootstrap, rng)
        margins[f"latent_minus_{opp}"] = mci
        p_str = "n/a" if mci.get("p_gt_zero") is None else f"P(>0)={mci['p_gt_zero']:.3f}"
        print(f"    latent-{opp:<13}  {sd._fmt(mci['point'])}   "
              f"[{sd._fmt(mci['ci_lo'])}, {sd._fmt(mci['ci_hi'])}]   "
              f"{p_str}   {sd._sig(mci.get('ci_lo'), mci.get('ci_hi'))}")
    print()

    # Verdict against the #21 pre-registered criteria.
    latent = per_gate.get("latent", {})
    point = latent.get("point")
    ci_lo = latent.get("ci_lo")
    ci_hi = latent.get("ci_hi")
    dragin_margin = margins.get("latent_minus_dragin", {})

    verdict = "null"
    if point is not None and ci_lo is not None and ci_hi is not None:
        latent_sig = ci_lo > 0
        dragin_sig = (
            dragin_margin.get("ci_lo") is not None
            and dragin_margin.get("ci_lo", -1) > 0
        )
        if latent_sig and dragin_sig:
            verdict = "strong_success"
        elif latent_sig:
            verdict = "success"
        elif point > 0 and dragin_margin.get("p_gt_zero") is not None:
            # Use latent Δfrac P(>0) via the bootstrap distribution width; approximate
            # by counting the proportion of the symmetric CI above zero.
            # boot_ci doesn't expose the raw draws, so fall back on point + ci_lo signs.
            verdict = "partial" if point > 0 else "null"

    print(f"[verdict] {verdict}  (per _AGENT.md #21 pre-registered criteria)")

    result = {
        "args": vars(args),
        "frozen_config": frozen,
        "traces_path": str(traces_path),
        "n_total": len(rows),
        "n_eligible": len(eligible),
        "min_steps": args.min_steps,
        "per_gate": per_gate,
        "margins": margins,
        "verdict": verdict,
        "preregistration": "_AGENT.md #21",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[ok] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
