"""v2 experiment analyzer — heavy logging, no pre-registration ceremony.

Reads closed-book traces + oracle-retrieve answers + frozen gate thresholds.
Computes everything we might possibly want for post-hoc analysis:

  - per-gate trace-level fire rate
  - lift-conditional confusion (recall, precision, FPR, harm rate)
  - paired bootstrap margin CIs (latent vs each baseline)
  - Bonferroni-corrected AND uncorrected p-values
  - McNemar test on binary lift_em
  - stratifications: by question type, by closed-book F1 band, by n_steps band
  - raw bootstrap arrays dumped to disk (so any post-hoc re-statistic is free)

Usage:

    python scripts/04o_exp_analyzer.py \
        --closed-book runs/v2_phi_hotpot_500.jsonl \
        --oracle      runs/v2_phi_hotpot_500_oracle.jsonl \
        --thresholds  runs/v2_phi_calibration.json \
        --out         runs/v2_phi_hotpot_500_analysis.json \
        --label       phi_hotpot500
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from importlib import import_module
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sd = import_module("04g_sota_showdown")  # type: ignore

GATES = ["latent", "etc", "selfdoubt_t1", "dragin", "random"]
BASELINES = ["etc", "selfdoubt_t1", "dragin", "random"]  # gates compared *against* latent


# -----------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# -----------------------------------------------------------------------
# Confusion matrix on lift labels
# -----------------------------------------------------------------------

def _confusion(rows: list[dict], gate: str) -> dict:
    n = len(rows)
    helpful = [r for r in rows if r["lift"]["em"] == 1]
    harmful = [r for r in rows if r["lift"]["em"] == -1]
    neutral = [r for r in rows if r["lift"]["em"] == 0]
    fired = [r for r in rows if r["fires"][gate]]
    not_fired = [r for r in rows if not r["fires"][gate]]
    fired_h = sum(1 for r in fired if r["lift"]["em"] == 1)
    fired_n = sum(1 for r in fired if r["lift"]["em"] == 0)
    fired_x = sum(1 for r in fired if r["lift"]["em"] == -1)
    return {
        "n": n,
        "n_helpful": len(helpful), "n_neutral": len(neutral), "n_harmful": len(harmful),
        "base_rate_helpful": len(helpful) / n if n else None,
        "n_fired": len(fired), "n_not_fired": len(not_fired),
        "fire_rate": len(fired) / n if n else 0.0,
        "recall_em":    fired_h / len(helpful) if helpful else None,
        "precision_em": fired_h / len(fired) if fired else None,
        "fpr_em":       (fired_x + fired_n) / (len(harmful) + len(neutral)) if (harmful or neutral) else None,
        "harm_rate":    fired_x / len(fired) if fired else None,
        # also F1-continuous versions (mean lift conditional)
        "mean_lift_em_fired":     (sum(r["lift"]["em"] for r in fired) / len(fired)) if fired else None,
        "mean_lift_em_not_fired": (sum(r["lift"]["em"] for r in not_fired) / len(not_fired)) if not_fired else None,
        "mean_lift_f1_fired":     (sum(r["lift"]["f1"] for r in fired) / len(fired)) if fired else None,
        "mean_lift_f1_not_fired": (sum(r["lift"]["f1"] for r in not_fired) / len(not_fired)) if not_fired else None,
    }


# -----------------------------------------------------------------------
# Paired bootstrap (margin: latent_X − baseline_X on the SAME resample)
# -----------------------------------------------------------------------

def _bootstrap_margins(rows: list[dict], B: int, rng: random.Random) -> dict:
    """For each (gate, baseline) pair, paired-bootstrap the margin in:
        - mean_lift_em(fired) − mean_lift_em(not_fired)   [self-discrimination]
        - mean_lift_f1 analogue
        - precision_em (P(lift=+1 | fires))               [against gate's own population]

    Also reports raw bootstrap arrays so anything post-hoc is free.
    """
    n = len(rows)

    def _margin(idx_list: list[int], gate: str, metric: str) -> float | None:
        fired_vals, not_vals = [], []
        for i in idx_list:
            t = rows[i]
            v = t["lift"][metric]
            if t["fires"][gate]:
                fired_vals.append(v)
            else:
                not_vals.append(v)
        if not fired_vals or not not_vals:
            return None
        return (sum(fired_vals)/len(fired_vals)) - (sum(not_vals)/len(not_vals))

    def _precision(idx_list: list[int], gate: str) -> float | None:
        fired = [rows[i] for i in idx_list if rows[i]["fires"][gate]]
        if not fired:
            return None
        helpful = sum(1 for r in fired if r["lift"]["em"] == 1)
        return helpful / len(fired)

    # Bootstrap arrays per gate per metric
    arr_lift_em:   dict[str, list[float | None]] = {g: [] for g in GATES}
    arr_lift_f1:   dict[str, list[float | None]] = {g: [] for g in GATES}
    arr_precision: dict[str, list[float | None]] = {g: [] for g in GATES}
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        for g in GATES:
            arr_lift_em[g].append(_margin(idx, g, "em"))
            arr_lift_f1[g].append(_margin(idx, g, "f1"))
            arr_precision[g].append(_precision(idx, g))

    def _summarize(xs_or_none: list[float | None]) -> dict:
        xs = [x for x in xs_or_none if x is not None]
        if not xs:
            return {"point": None, "ci_lo": None, "ci_hi": None,
                    "p_gt_zero": None, "n_valid": 0}
        return {
            "point":     None,  # filled in by caller using full-cohort
            "ci_lo":     sd._pct(xs, 0.025),
            "ci_hi":     sd._pct(xs, 0.975),
            "p_gt_zero": sum(1 for x in xs if x > 0) / len(xs),
            "n_valid":   len(xs),
        }

    per_gate: dict[str, dict] = {}
    for g in GATES:
        full_idx = list(range(n))
        per_gate[g] = {
            "lift_em":    {**_summarize(arr_lift_em[g]),
                           "point": _margin(full_idx, g, "em")},
            "lift_f1":    {**_summarize(arr_lift_f1[g]),
                           "point": _margin(full_idx, g, "f1")},
            "precision":  {**_summarize(arr_precision[g]),
                           "point": _precision(full_idx, g)},
        }

    # Paired margins: latent vs each baseline, paired on the same resample
    paired: dict[str, dict] = {}
    for g_a, g_b in [("latent", b) for b in BASELINES]:
        for metric in ("lift_em", "lift_f1", "precision"):
            arrs_a = {"lift_em": arr_lift_em, "lift_f1": arr_lift_f1, "precision": arr_precision}[metric][g_a]
            arrs_b = {"lift_em": arr_lift_em, "lift_f1": arr_lift_f1, "precision": arr_precision}[metric][g_b]
            diffs = [a - b for a, b in zip(arrs_a, arrs_b) if (a is not None and b is not None)]
            if not diffs:
                paired[f"{g_a}-{g_b}_{metric}"] = {"point": None, "ci_lo": None, "ci_hi": None,
                                                   "p_gt_zero": None, "n_valid": 0,
                                                   "bonferroni_sig_at_0.0125": None,
                                                   "uncorrected_sig_at_0.05": None}
                continue
            # Full-cohort point estimate
            full_idx = list(range(n))
            if metric in ("lift_em", "lift_f1"):
                metric_short = "em" if metric == "lift_em" else "f1"
                fa = _margin(full_idx, g_a, metric_short)
                fb = _margin(full_idx, g_b, metric_short)
            else:
                fa = _precision(full_idx, g_a)
                fb = _precision(full_idx, g_b)
            point = (fa - fb) if (fa is not None and fb is not None) else None
            p_gt = sum(1 for x in diffs if x > 0) / len(diffs)
            p_lt = sum(1 for x in diffs if x < 0) / len(diffs)
            two_sided_p = 2 * min(p_gt, p_lt)
            paired[f"{g_a}-{g_b}_{metric}"] = {
                "point":      point,
                "ci_lo":      sd._pct(diffs, 0.025),
                "ci_hi":      sd._pct(diffs, 0.975),
                "p_gt_zero":  p_gt,
                "two_sided_p": two_sided_p,
                "uncorrected_sig_at_0.05":   bool(two_sided_p < 0.05),
                "bonferroni_sig_at_0.0125":  bool(two_sided_p < 0.05 / len(BASELINES)),
                "n_valid":    len(diffs),
            }

    return {
        "per_gate":       per_gate,
        "paired_margins": paired,
        "raw":            {  # for post-hoc re-statistics
            "lift_em":   arr_lift_em,
            "lift_f1":   arr_lift_f1,
            "precision": arr_precision,
        },
    }


# -----------------------------------------------------------------------
# McNemar test on binary lift_em=+1
# -----------------------------------------------------------------------

def _mcnemar(rows: list[dict], gate_a: str, gate_b: str) -> dict:
    """McNemar on the 2x2 table of fired-helpful labels:
        b = a-fires-AND-helpful but b-doesn't,
        c = b-fires-AND-helpful but a-doesn't.
    Continuity-corrected chi^2 = (|b-c| - 1)^2 / (b+c).
    """
    a_only = b_only = both = neither = 0
    for r in rows:
        if r["lift"]["em"] != 1:
            continue  # only score on helpful cases
        af = r["fires"][gate_a]
        bf = r["fires"][gate_b]
        if af and not bf:   a_only += 1
        elif bf and not af: b_only += 1
        elif af and bf:     both += 1
        else:               neither += 1
    discordant = a_only + b_only
    if discordant == 0:
        return {"a_only": a_only, "b_only": b_only, "both": both, "neither": neither,
                "chi2": None, "p_two_sided": None}
    chi2 = (abs(a_only - b_only) - 1) ** 2 / discordant
    # p-value from chi^2 with 1 dof; use survival function approximation
    p = math.exp(-chi2 / 2.0)  # crude upper bound for df=1; exact would use erfc
    return {"a_only": a_only, "b_only": b_only, "both": both, "neither": neither,
            "chi2": chi2, "p_two_sided_approx": p}


# -----------------------------------------------------------------------
# Stratifications
# -----------------------------------------------------------------------

def _stratified_confusions(rows: list[dict], by: str, bins: list[tuple[str, callable]]) -> dict:
    out: dict[str, dict] = {}
    for bin_name, predicate in bins:
        subset = [r for r in rows if predicate(r)]
        out[bin_name] = {"n": len(subset), "per_gate": {g: _confusion(subset, g) for g in GATES}}
    return out


def _build_rows(closed: list[dict], oracle_by_id: dict[str, dict],
                thresholds: dict, latent_mode: str, random_p: float,
                rng: random.Random, splits_kept: set[str],
                min_steps: int) -> list[dict]:
    rows: list[dict] = []
    for r in closed:
        s = r.get("split")
        if splits_kept and s not in splits_kept:
            continue
        if int(r.get("n_steps", 0)) < min_steps:
            continue
        orow = oracle_by_id.get(r["id"])
        if not orow or "em" not in orow:
            continue
        pt = sd.firsts_per_trace([r], thresholds, latent_mode, random_p, rng)[0]
        fires = {g: (pt["first_fire"].get(g) is not None) for g in GATES}
        c_em, c_f1 = int(r["em"]), float(r["f1"])
        o_em, o_f1 = int(orow["em"]), float(orow["f1"])
        rows.append({
            "id":         r["id"],
            "split":      s,
            "type":       r.get("type", "unknown"),
            "level":      r.get("level", "unknown"),
            "n_steps":    int(r.get("n_steps", 0)),
            "closed":     {"em": c_em, "f1": c_f1},
            "oracle":     {"em": o_em, "f1": o_f1},
            "lift":       {"em": o_em - c_em, "f1": o_f1 - c_f1},
            "fires":      fires,
            "first_fire": {g: pt["first_fire"].get(g) for g in GATES},
        })
    return rows


# -----------------------------------------------------------------------
# Pretty-print
# -----------------------------------------------------------------------

def _pp(rows: list[dict], boot: dict, label: str) -> None:
    print()
    print("=" * 100)
    print(f"== {label}  (n={len(rows)})")
    print("=" * 100)

    helpful = sum(1 for r in rows if r["lift"]["em"] == 1)
    harmful = sum(1 for r in rows if r["lift"]["em"] == -1)
    neutral = sum(1 for r in rows if r["lift"]["em"] == 0)
    closed_em  = sum(r["closed"]["em"] for r in rows) / len(rows) if rows else 0
    oracle_em  = sum(r["oracle"]["em"] for r in rows) / len(rows) if rows else 0
    closed_f1  = sum(r["closed"]["f1"] for r in rows) / len(rows) if rows else 0
    oracle_f1  = sum(r["oracle"]["f1"] for r in rows) / len(rows) if rows else 0
    print(f"  lift_em:  helpful(+1)={helpful}  neutral(0)={neutral}  harmful(-1)={harmful}  "
          f"base_rate_helpful={helpful/len(rows)*100:.1f}%" if rows else "")
    print(f"  closed-book EM = {closed_em*100:.1f}%  F1 = {closed_f1*100:.1f}%")
    print(f"  oracle      EM = {oracle_em*100:.1f}%  F1 = {oracle_f1*100:.1f}%  "
          f"(Δ={(oracle_em-closed_em)*100:+.1f} EM, {(oracle_f1-closed_f1)*100:+.1f} F1)")
    print()
    print(f"  {'gate':14}  fire%  rec_em  prec_em  fpr_em  harm   Δlift_em(point  [95% CI])         "
          f"Δlift_f1(point  [95% CI])         prec(point [95% CI])")
    for g in GATES:
        b = boot["per_gate"][g]
        c = _confusion(rows, g)
        def _f(x, w=6): return "n/a".rjust(w) if x is None else f"{x*100:+.2f}".rjust(w)
        def _pct(x):    return "n/a" if x is None else f"{x*100:5.1f}"
        print(f"  {g:14}  {c['fire_rate']*100:4.1f}  {_pct(c['recall_em'])}  "
              f"{_pct(c['precision_em'])}   {_pct(c['fpr_em'])}  "
              f"{_pct(c['harm_rate'])}   "
              f"{_f(b['lift_em']['point'])} [{_f(b['lift_em']['ci_lo'])},{_f(b['lift_em']['ci_hi'])}]    "
              f"{_f(b['lift_f1']['point'])} [{_f(b['lift_f1']['ci_lo'])},{_f(b['lift_f1']['ci_hi'])}]    "
              f"{_f(b['precision']['point'])} [{_f(b['precision']['ci_lo'])},{_f(b['precision']['ci_hi'])}]")
    print()
    print(f"  Paired margins (latent − baseline). 2-sided p, Bonferroni @ α=0.05/{len(BASELINES)}={0.05/len(BASELINES):.4f}.")
    for metric in ("lift_em", "lift_f1", "precision"):
        print(f"    -- {metric} --")
        for baseline in BASELINES:
            m = boot["paired_margins"].get(f"latent-{baseline}_{metric}", {})
            if m.get("point") is None:
                continue
            sig = ""
            if m.get("bonferroni_sig_at_0.0125"):
                sig = "***bonf"
            elif m.get("uncorrected_sig_at_0.05"):
                sig = "*uncorr"
            print(f"      latent − {baseline:14}  Δ={m['point']*100:+6.2f}  "
                  f"[{m['ci_lo']*100:+6.2f}, {m['ci_hi']*100:+6.2f}]  "
                  f"p2-sided={m.get('two_sided_p', float('nan')):.4f}  "
                  f"P(>0)={m.get('p_gt_zero', float('nan')):.3f}  {sig}")
    print()
    # McNemar
    print("  McNemar on lift_em=+1 (helpful cohort only). Continuity-corrected χ², approx p.")
    for baseline in BASELINES:
        m = _mcnemar(rows, "latent", baseline)
        if m.get("chi2") is None:
            print(f"    latent vs {baseline:14}  zero discordant pairs (a_only={m['a_only']}, b_only={m['b_only']})")
        else:
            print(f"    latent vs {baseline:14}  a_only={m['a_only']:3d}  b_only={m['b_only']:3d}  "
                  f"both={m['both']:3d}  neither={m['neither']:3d}  "
                  f"χ²={m['chi2']:.3f}  p_approx={m['p_two_sided_approx']:.4f}")
    print()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--closed-book", required=True)
    p.add_argument("--oracle", required=True)
    p.add_argument("--thresholds", required=True,
                   help="JSON with arch-specific {thresholds, latent_mode_chosen, random_p}; "
                        "format compatible with phase4_bootstrap_dragin.json or v2 calibration JSON.")
    p.add_argument("--arch", default=None,
                   help="If thresholds JSON has a `per_arch` dict, pick this key; "
                        "otherwise the top-level keys are used as-is.")
    p.add_argument("--out", required=True)
    p.add_argument("--label", default="experiment")
    p.add_argument("-B", "--bootstrap", type=int, default=10000)
    p.add_argument("--min-steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260511)
    p.add_argument("--splits", nargs="+", default=None,
                   help="Restrict to a subset of split tags. Default: any split.")
    p.add_argument("--dump-raw", action="store_true",
                   help="Include the full per-resample bootstrap arrays in the JSON (large file).")
    args = p.parse_args()

    closed_rows = _load_jsonl(Path(args.closed_book))
    oracle_rows = _load_jsonl(Path(args.oracle))
    oracle_by_id = {r["id"]: r for r in oracle_rows if "em" in r}

    thr_doc = json.loads(Path(args.thresholds).read_text())
    if args.arch:
        if "per_arch" in thr_doc and args.arch in thr_doc["per_arch"]:
            arch_rec = thr_doc["per_arch"][args.arch]
        elif args.arch in thr_doc:
            arch_rec = thr_doc[args.arch]
        else:
            print(f"[err] arch {args.arch!r} not in thresholds doc"); return 1
    else:
        arch_rec = thr_doc
    thresholds   = arch_rec.get("thresholds") or arch_rec["thresholds_chosen"]
    latent_mode  = arch_rec.get("latent_mode_chosen") or arch_rec.get("latent_mode")
    random_p     = float(arch_rec.get("random_p", 0.30))

    splits_kept = set(args.splits) if args.splits else set()
    rng = random.Random(args.seed)
    rows = _build_rows(closed_rows, oracle_by_id, thresholds, latent_mode, random_p,
                       rng, splits_kept, args.min_steps)

    print(f"[..] {args.label}: built {len(rows)} rows  "
          f"(closed={len(closed_rows)}, oracle={len(oracle_by_id)}, "
          f"thresholds_mode={latent_mode}, B={args.bootstrap})")

    # Primary table (whole cohort)
    rng2 = random.Random(args.seed + 1)
    primary_boot = _bootstrap_margins(rows, args.bootstrap, rng2)
    _pp(rows, primary_boot, f"{args.label} :: all (n={len(rows)})")

    # Stratifications
    strata: dict = {}

    by_split = sorted({r["split"] for r in rows})
    for split in by_split:
        sub = [r for r in rows if r["split"] == split]
        if not sub:
            continue
        rng3 = random.Random(args.seed + hash(split) % 10000)
        sub_boot = _bootstrap_margins(sub, args.bootstrap, rng3)
        _pp(sub, sub_boot, f"{args.label} :: split={split} (n={len(sub)})")
        strata[f"split={split}"] = {"n": len(sub), "boot": sub_boot,
                                    "confusion": {g: _confusion(sub, g) for g in GATES}}

    by_type = sorted({r["type"] for r in rows})
    for t in by_type:
        sub = [r for r in rows if r["type"] == t]
        if len(sub) < 30:
            continue
        strata[f"type={t}"] = {"n": len(sub),
                               "confusion": {g: _confusion(sub, g) for g in GATES}}

    # Closed-book F1 band stratification
    bands = [
        ("cb_f1_0.00-0.25", lambda r: r["closed"]["f1"] < 0.25),
        ("cb_f1_0.25-0.50", lambda r: 0.25 <= r["closed"]["f1"] < 0.50),
        ("cb_f1_0.50-0.75", lambda r: 0.50 <= r["closed"]["f1"] < 0.75),
        ("cb_f1_0.75-1.00", lambda r: r["closed"]["f1"] >= 0.75),
    ]
    for name, pred in bands:
        sub = [r for r in rows if pred(r)]
        if len(sub) < 20:
            continue
        strata[name] = {"n": len(sub),
                        "confusion": {g: _confusion(sub, g) for g in GATES}}

    # n_steps band stratification
    step_bands = [
        ("nsteps_0-2",  lambda r: r["n_steps"] <= 2),
        ("nsteps_3-5",  lambda r: 3 <= r["n_steps"] <= 5),
        ("nsteps_6-9",  lambda r: 6 <= r["n_steps"] <= 9),
        ("nsteps_10+",  lambda r: r["n_steps"] >= 10),
    ]
    for name, pred in step_bands:
        sub = [r for r in rows if pred(r)]
        if len(sub) < 20:
            continue
        strata[name] = {"n": len(sub),
                        "confusion": {g: _confusion(sub, g) for g in GATES}}

    result: dict = {
        "args": vars(args),
        "label": args.label,
        "n_total": len(rows),
        "thresholds": thresholds,
        "latent_mode": latent_mode,
        "random_p": random_p,
        "per_trace": rows,           # full per-trace dump for post-hoc analysis
        "primary": primary_boot if args.dump_raw else _drop_raw(primary_boot),
        "strata": strata,
        "mcnemar": {f"latent_vs_{b}": _mcnemar(rows, "latent", b) for b in BASELINES},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[ok] wrote {out_path}")
    if args.dump_raw:
        print(f"     (raw bootstrap arrays included; expect large file)")
    return 0


def _drop_raw(boot: dict) -> dict:
    return {k: v for k, v in boot.items() if k != "raw"}


if __name__ == "__main__":
    sys.exit(main())
