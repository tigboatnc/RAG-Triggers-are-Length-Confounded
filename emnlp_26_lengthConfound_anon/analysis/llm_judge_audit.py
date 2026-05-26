"""LLM-as-Judge robustness check for the headline survivor matrix.

Replaces the exact-match (EM) correctness function with an LLM-issued
semantic-equivalence judgement on both arms of the paired-oracle target.
This is the Janiak 2025-style robustness check that the length-confound
literature demands (Janiak, Santilli, Ielanskyi all explicitly recommend
LLM-as-Judge over lexical-overlap correctness).

Recipe (matches Janiak EMNLP 2025): a small-tier OpenAI model issues a
binary YES/NO call on whether the predicted answer is semantically
equivalent to the gold answer. Both closed-book and oracle decodes are
judged with the same prompt; we then define

    lift_em_judge(q) = 1[ judge(a_o)=YES  AND  judge(a_c)=NO ]

and re-evaluate Δ = AUROC(s_resid) − AUROC(tokens) on the headline pool
under this new target. Cohen's κ between lift_em and lift_em_judge is
reported as the agreement metric (Janiak Cohen's κ = 0.799 for
GPT-4o-Mini vs human; we report κ vs EM as the within-paper anchor).

API setup
---------
The OpenAI Python SDK auto-reads OPENAI_API_KEY from the environment. Per
the official quickstart (https://developers.openai.com/api/docs/quickstart):

    export OPENAI_API_KEY="sk-..."

Two convenience fallbacks (in priority order):
  1. $OPENAI_API_KEY environment variable (standard)
  2. secrets/openai_api_key.txt (single-line file, gitignored)

Model choice
------------
Default model is gpt-5-mini (the 2026 successor to gpt-4o-mini, which is
what Janiak EMNLP 2025 used). Override with --model. For literal Janiak
replication pass --model gpt-4o-mini. For paranoid cost control pass
--model gpt-5-nano.

Cost @ gpt-5-mini, n=200 × 2 arms × 2 datasets × 2 archs = 1,600 calls of
~80 input + 5 output tokens ≈ $0.05 total. (gpt-5-nano: ~$0.01.
gpt-4o-mini: ~$0.10.)

Caching
-------
Every judgement is cached per (arch, dataset, qid, arm) under
analysis/llm_judge_cache/. Re-runs skip cached entries; the script is
resumable. Delete the cache directory to force a clean re-run.

Usage
-----
    python analysis/llm_judge_audit.py --arch qwen2.5-7b-instruct
    python analysis/llm_judge_audit.py --arch gemma-4-26b-a4b-it
    python analysis/llm_judge_audit.py --arch both                 # pooled
    python analysis/llm_judge_audit.py --arch both --model gpt-5-nano

Outputs
-------
    analysis/llm_judge_audit_<arch>.out    (per-arch + pooled summary)
    analysis/llm_judge_audit_<arch>.json   (machine-readable)
    analysis/llm_judge_cache/<arch>/<dataset>/<qid>_<arm>.json   (cache)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from signal_lib import (
    auroc_pos, bootstrap_auroc_diff, ols_residual,
)
from r1_paired_audit import load_paired as load_paired_base
from moonshot_explore import cabv_score, fct_score, laas_score, normalize_final

# Signal pool matches Table 1 / Appendix A.
SIGNALS = [
    ("dragin_max",       "DRAGIN-RIND.max"),
    ("ent_max",          "ent.max"),
    ("etc",              "etc (Δ²H max)"),
    ("flare_max",        "flare.max (1-min p)"),
    ("u_cons_max",       "u_cons.max"),
    ("seakr_k5_logdet",  "seakr_k5_proxy.logdet"),
    ("drift_max",        "drift_cosine.max"),
    ("var_max",          "var_l2_window.max"),
    ("hidden_range",     "hidden_norm.range"),
    ("hvr_max",          "hvr.max"),
    ("flip_k5",          "flip_rate K=5"),
    ("flip_k1",          "flip_rate K=1"),
    ("late_flips",       "late_flips"),
    ("min_majority",     "min_majority"),
    ("M1_cabv",          "CABV"),
    ("M8_fct",           "FCT"),
    ("M9_cabv_x_laas",   "CABV×LAAS"),
]

JUDGE_SYSTEM_PROMPT = """You are a strict semantic-equivalence judge for multi-hop QA.
Your task is to decide whether a model-predicted answer is semantically equivalent to a gold answer.
Reply with exactly one word: YES or NO. No explanation."""

JUDGE_USER_TEMPLATE = """Question: {question}
Gold answer: {gold}
Predicted answer: {pred}

Is the predicted answer semantically equivalent to the gold answer? Reply YES or NO."""


# ============================================================================ #
# API key resolution + client construction
# ============================================================================ #

def resolve_api_key() -> str:
    """Resolve the OpenAI API key from env var or secrets file.

    The secrets file is parsed leniently: comment lines (starting with '#')
    and blank lines are skipped; the first remaining non-blank line is
    treated as the key.  This means you can paste your key into the
    example file as-is, or into the canonical file without surrounding
    comments — both work.
    """
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    # Try both names so the user doesn't have to rename the example file.
    for fname in ("openai_api_key.txt", "openai_api_key.txt.example"):
        secrets_path = ROOT / "secrets" / fname
        if not secrets_path.exists():
            continue
        for raw_line in secrets_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("sk-REPLACE"):
                # template still has the placeholder — skip
                continue
            return line
    raise RuntimeError(
        "OpenAI API key not found.  Either:\n"
        "  (a) export OPENAI_API_KEY=\"sk-...\" in your shell, OR\n"
        f"  (b) put your key in {ROOT / 'secrets/openai_api_key.txt'} "
        "(one line, gitignored).\n"
        "See https://developers.openai.com/api/docs/quickstart for the "
        "standard pattern."
    )


def build_client():
    """Lazy-import openai so importing this module doesn't require the SDK."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai package not installed.  Add to environment.yml or:\n"
            "  pip install openai>=1.50\n"
        ) from exc
    api_key = resolve_api_key()
    return OpenAI(api_key=api_key)


# ============================================================================ #
# Judgement
# ============================================================================ #

_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
_NO_RE = re.compile(r"\bno\b", re.IGNORECASE)


def parse_judgement(text: str) -> int | None:
    """Parse YES/NO from the model output.  Returns 1, 0, or None."""
    if not text:
        return None
    text = text.strip()
    has_yes = bool(_YES_RE.search(text))
    has_no = bool(_NO_RE.search(text))
    if has_yes and not has_no:
        return 1
    if has_no and not has_yes:
        return 0
    # Ambiguous — fail-soft to None
    return None


def judge_one(client, model: str, question: str, gold: str, pred: str,
              max_retries: int = 6) -> tuple[int | None, str]:
    """One judgement call.  Returns (binary_label, raw_text)."""
    if not pred or pred.lower().strip() in ("", "none", "n/a"):
        return 0, "EMPTY_PRED"
    user_msg = JUDGE_USER_TEMPLATE.format(
        question=question.strip(),
        gold=gold.strip(),
        pred=pred.strip(),
    )
    # Responses API per https://developers.openai.com/api/docs/quickstart
    # Falls back to chat.completions for older model strings.
    last_err = None
    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=model,
                input=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                # GPT-5-class reasoning models consume max_output_tokens for
                # both internal reasoning AND the visible output.  With <64
                # the model can spend the whole budget on reasoning and emit
                # 0 visible tokens.  256 is comfortable headroom for a YES/NO
                # judge and still costs <$0.0001 per call.
                max_output_tokens=256,
            )
            # For GPT-5-class reasoning models, force minimal reasoning effort
            # so the model spends most of its budget on the visible answer
            # rather than on internal CoT.  gpt-4o-* family ignores this kwarg.
            if "gpt-5" in model.lower():
                kwargs["reasoning"] = {"effort": "minimal"}
            response = client.responses.create(**kwargs)
            raw_text = response.output_text
            return parse_judgement(raw_text), raw_text
        except Exception as exc:
            last_err = exc
            # Rate limit: long backoff (exponential, up to ~5 min).  Other
            # errors: short backoff.  GPT-5-mini TPM caps are tight on new
            # accounts; the long backoff buys time for the quota to refill.
            is_rate = "RateLimit" in type(exc).__name__ or "429" in str(exc)
            sleep_s = (8 * (2 ** attempt)) if is_rate else (2 ** attempt)
            sleep_s = min(sleep_s, 300)
            print(f"  [retry {attempt+1}/{max_retries} after {sleep_s}s] "
                  f"{type(exc).__name__}: {str(exc)[:100]}",
                  file=sys.stderr)
            time.sleep(sleep_s)
    return None, f"ERROR: {type(last_err).__name__}: {last_err}"


def log_line(log_path: Path, msg: str) -> None:
    """Append an ISO-timestamped line to the run log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with log_path.open("a") as f:
        f.write(f"[{ts}] {msg}\n")


def judge_with_cache(client, model: str, arch: str, dataset: str,
                     qid: str, arm: str, question: str, gold: str,
                     pred: str, cache_dir: Path,
                     log_path: Path | None = None) -> int | None:
    """Cache-aware judgement.  Returns the binary label (1/0/None)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{qid}_{arm}.json"
    if cache_path.exists():
        rec = json.loads(cache_path.read_text())
        # Only honor the cache when the model matches AND we got a real
        # label.  Failed calls (label is None) are retried on the next run.
        if rec.get("model") == model and rec.get("label") is not None:
            if log_path is not None:
                log_line(log_path,
                         f"CACHE-HIT  {arch}/{dataset}/{qid}/{arm}  label={rec['label']}")
            return rec.get("label")
    t0 = time.time()
    label, raw = judge_one(client, model, question, gold, pred)
    elapsed_ms = int((time.time() - t0) * 1000)
    cache_path.write_text(json.dumps({
        "qid": qid, "arm": arm, "arch": arch, "dataset": dataset,
        "model": model, "question": question, "gold": gold, "pred": pred,
        "label": label, "raw": raw, "ts": time.time(),
    }, indent=2))
    if log_path is not None:
        raw_preview = (raw or "").replace("\n", " ")[:80]
        log_line(log_path,
                 f"CALL       {arch}/{dataset}/{qid}/{arm:6s}  "
                 f"model={model}  label={label}  elapsed={elapsed_ms}ms  raw={raw_preview!r}")
    return label


# ============================================================================ #
# Trace loading (with moonshot signals)
# ============================================================================ #

def load_paired_full(cp: Path, op: Path) -> list[dict]:
    """Wrap load_paired_base + add the moonshot signals AND keep the raw
    question / gold / pred fields needed for judgement."""
    base_rows = load_paired_base(cp, op)
    if not base_rows:
        return []
    closed = {r["id"]: r for r in (json.loads(l) for l in cp.open() if l.strip())}
    oracle = {r["id"]: r for r in (json.loads(l) for l in op.open() if l.strip())}
    for row in base_rows:
        c = closed.get(row["id"])
        o = oracle.get(row["id"])
        if not c or not o:
            continue
        steps = c.get("steps") or []
        final_ans = normalize_final(c.get("model_answer_extracted") or "")
        cabv = cabv_score(steps)
        laas = laas_score(steps)
        fct = fct_score(steps, final_ans)
        row["M1_cabv"] = cabv
        row["M8_fct"] = fct
        row["M9_cabv_x_laas"] = (cabv * laas) if (cabv is not None and laas is not None) else None
        # Raw text needed for judgement
        row["question"] = c.get("question") or ""
        row["gold"] = c.get("gold_answer") or c.get("answer") or ""
        row["pred_closed"] = c.get("model_answer_extracted") or c.get("model_answer") or ""
        row["pred_oracle"] = o.get("model_answer_extracted") or o.get("model_answer") or ""
    return base_rows


def load_arch(arch: str) -> list[tuple[str, list[dict]]]:
    """Return [(dataset_label, rows), ...] for one arch."""
    base = ROOT / "runs" / f"v2_headline_{arch}"
    out = []
    for ds_label, df, of in [
        ("hotpot", "D1_closed_hotpot.jsonl", "E1_oracle_hotpot.jsonl"),
        ("wiki",   "D2_closed_wiki.jsonl",   "E2_oracle_wiki.jsonl"),
    ]:
        cp = base / df
        op = base / of
        if not cp.exists() or not op.exists():
            print(f"  [skip] {arch}/{ds_label}: trace files missing at {base}", file=sys.stderr)
            continue
        rows = load_paired_full(cp, op)
        if rows:
            out.append((ds_label, rows))
    return out


# ============================================================================ #
# Judgement pass over all traces (with cache)
# ============================================================================ #

def judge_arch(client, model: str, arch: str, cache_root: Path,
               log_path: Path | None = None,
               max_queries: int | None = None,
               sleep_s: float = 0.0) -> list[dict]:
    """Judge both arms of every paired trace for one arch.  Returns the
    list of rows with `closed_em_judge`, `oracle_em_judge`, `lift_em_judge`
    fields populated."""
    cells = load_arch(arch)
    if not cells:
        return []

    all_rows = []
    n_calls = 0
    for ds_label, rows in cells:
        cache_dir = cache_root / arch / ds_label
        print(f"  judging {arch} × {ds_label}: {len(rows)} pairs"
              f" ({2*len(rows)} calls; cached entries skipped)")
        if log_path is not None:
            log_line(log_path,
                     f"BEGIN-CELL {arch}/{ds_label}  n_pairs={len(rows)}")
        for row in rows:
            qid = str(row["id"])
            question = row.get("question", "")
            gold = row.get("gold", "")
            pred_closed = row.get("pred_closed", "")
            pred_oracle = row.get("pred_oracle", "")

            row["closed_em_judge"] = judge_with_cache(
                client, model, arch, ds_label, qid, "closed",
                question, gold, pred_closed, cache_dir, log_path=log_path
            )
            n_calls += 1
            if sleep_s > 0:
                time.sleep(sleep_s)
            row["oracle_em_judge"] = judge_with_cache(
                client, model, arch, ds_label, qid, "oracle",
                question, gold, pred_oracle, cache_dir, log_path=log_path
            )
            n_calls += 1
            if sleep_s > 0:
                time.sleep(sleep_s)

            # Build lift_em_judge — symmetric to lift_em definition
            cj = row["closed_em_judge"]
            oj = row["oracle_em_judge"]
            if cj is None or oj is None:
                row["lift_em_judge"] = None
            else:
                row["lift_em_judge"] = +1 if (oj == 1 and cj == 0) else (
                    -1 if (oj == 0 and cj == 1) else 0
                )

            if max_queries is not None and n_calls >= max_queries:
                print(f"  [budget] hit --max-queries={max_queries}, stopping early")
                all_rows.extend(rows)
                return all_rows
        all_rows.extend(rows)
    return all_rows


# ============================================================================ #
# Agreement: Cohen's kappa between lift_em and lift_em_judge
# ============================================================================ #

def cohens_kappa(a: list[int], b: list[int]) -> float:
    """Cohen's κ for two label vectors over the same items."""
    assert len(a) == len(b) and len(a) > 0
    n = len(a)
    cats = sorted(set(a) | set(b))
    obs = sum(1 for x, y in zip(a, b) if x == y) / n
    # Expected agreement under independence
    from collections import Counter
    pa = Counter(a); pb = Counter(b)
    exp = sum((pa[c] / n) * (pb[c] / n) for c in cats)
    if exp == 1.0:
        return float("nan")
    return (obs - exp) / (1.0 - exp)


# ============================================================================ #
# Re-evaluate the survivor matrix under lift_em_judge
# ============================================================================ #

def report_under_judge(rows: list[dict], B: int, label_fn,
                       target_name: str) -> str:
    """Re-run the headline survivor matrix under the given label function
    (lift_em_judge for the judge-based target).  Returns a formatted string."""
    valid = [r for r in rows if label_fn(r) is not None and r.get("tokens") is not None]
    n = len(valid)
    pop = sum(1 for r in valid if label_fn(r) == +1)
    base = pop / n if n else float("nan")

    lines = []
    lines.append("")
    lines.append("=" * 100)
    lines.append(f"  Headline survivor matrix under target = {target_name}")
    lines.append(f"  n={n}  pop={pop}  base={base:.3f}  B={B}")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"  {'signal':<30}  {'resAUC':>7}  {'Δ':>7}  {'95% CI':>22}  {'sig?':>5}")
    lines.append("  " + "-" * 80)

    # Tokens baseline (raw + residualized)
    valid_tok = [r for r in valid if r.get("tokens") is not None]
    is_pos = lambda r: label_fn(r) == +1
    tok_au = auroc_pos(valid_tok, "tokens", is_pos)
    res_tok_map, _ = ols_residual(valid_tok, "tokens", x_key="log_tokens")
    for r in valid_tok:
        r["__res_tokens__"] = res_tok_map[id(r)]
    res_tok_au = auroc_pos(valid_tok, "__res_tokens__", is_pos)
    lines.append(f"  {'tokens_generated':<30}  {res_tok_au:>7.3f}  "
                 f"{'—':>7}  (baseline, raw {tok_au:.3f})        —")

    for key, name in SIGNALS:
        sub = [r for r in valid if r.get(key) is not None]
        if len(sub) < 20:
            continue
        # Residualize
        resid_map, _ = ols_residual(sub, key, x_key="log_tokens")
        for r in sub:
            r[f"__res_{key}__"] = resid_map[id(r)]
        res_au = auroc_pos(sub, f"__res_{key}__", is_pos)
        # Paired bootstrap of (res_signal − tokens_raw) — same formula as Table 1
        m, lo, hi, _ = bootstrap_auroc_diff(
            sub, f"__res_{key}__", "tokens", is_pos, B=B, seed=11
        )
        sig = "✓" if (lo == lo and (lo > 0 or hi < 0)) else "—"
        lines.append(f"  {name:<30}  {res_au:>7.3f}  "
                     f"{m:>+7.3f}  [{lo:>+6.3f},{hi:>+6.3f}]   {sig:>5}")

    return "\n".join(lines)


# ============================================================================ #
# Main
# ============================================================================ #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="both",
                    choices=["qwen2.5-7b-instruct", "gemma-4-26b-a4b-it", "both"],
                    help="Which architecture(s) to judge")
    ap.add_argument("--model", default="gpt-5-mini",
                    help="OpenAI model id (default: gpt-5-mini). "
                    "Use 'gpt-4o-mini' for literal Janiak 2025 replication, "
                    "or 'gpt-5-nano' for cheapest.")
    ap.add_argument("--bootstrap-b", type=int, default=10_000)
    ap.add_argument("--max-queries", type=int, default=None,
                    help="Cap on total API calls (for dry-run / budget)")
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="Seconds to sleep between calls (to stay under TPM "
                    "limits on free / tier-1 accounts).  Default 1.0s.")
    args = ap.parse_args()

    client = build_client()
    cache_root = ROOT / "analysis" / "llm_judge_cache"
    log_path = ROOT / "analysis" / "llm_judge_audit.log"
    print(f"LLM-as-Judge robustness audit")
    print(f"  model:        {args.model}")
    print(f"  arch:         {args.arch}")
    print(f"  cache root:   {cache_root.relative_to(ROOT)}")
    print(f"  log file:     {log_path.relative_to(ROOT)}")
    print(f"  bootstrap-b:  {args.bootstrap_b}")
    if args.max_queries:
        print(f"  max-queries:  {args.max_queries}")
    print()

    log_line(log_path,
             f"RUN-START   model={args.model}  arch={args.arch}  "
             f"max_queries={args.max_queries}  B={args.bootstrap_b}")

    archs = ["qwen2.5-7b-instruct", "gemma-4-26b-a4b-it"] if args.arch == "both" else [args.arch]

    pool: list[dict] = []
    for arch in archs:
        rows = judge_arch(client, args.model, arch, cache_root,
                          log_path=log_path,
                          max_queries=args.max_queries,
                          sleep_s=args.sleep)
        if rows:
            pool.extend(rows)
            print(f"  [done] {arch}: {len(rows)} pairs judged")
        else:
            print(f"  [warn] {arch}: no rows loaded")

    if not pool:
        print("\n[abort] no traces loaded; nothing to report")
        return

    # Cohen's κ: lift_em (EM) vs lift_em_judge ------------------------------ #
    paired = [r for r in pool if r.get("lift_em") is not None
              and r.get("lift_em_judge") is not None]
    em_labels = [r["lift_em"] for r in paired]
    jd_labels = [r["lift_em_judge"] for r in paired]
    kappa = cohens_kappa(em_labels, jd_labels) if paired else float("nan")
    n_helped_em = sum(1 for r in paired if r["lift_em"] == +1)
    n_helped_judge = sum(1 for r in paired if r["lift_em_judge"] == +1)
    n_agree_helped = sum(1 for r in paired
                         if r["lift_em"] == +1 and r["lift_em_judge"] == +1)

    print()
    print("=" * 100)
    print("  Agreement: lift_em (EM) vs lift_em_judge (LLM-as-Judge)")
    print("=" * 100)
    print(f"  n_paired:                       {len(paired)}")
    print(f"  helped under EM:                {n_helped_em}")
    print(f"  helped under judge:             {n_helped_judge}")
    print(f"  helped under BOTH (intersect):  {n_agree_helped}")
    print(f"  Cohen's κ (3-class ±1/0):       {kappa:.3f}")
    if kappa < 0.4:
        print("  → κ<0.4: substantial disagreement between EM and judge "
              "(consistent with Ielanskyi 2025 critique)")
    elif kappa < 0.6:
        print("  → κ in [0.4, 0.6]: moderate agreement; judge identifies a "
              "somewhat different helped subset")
    else:
        print("  → κ≥0.6: substantial agreement; survivor matrix should "
              "transfer cleanly to judge label")

    # Headline survivor matrix under lift_em_judge -------------------------- #
    report = report_under_judge(
        pool, args.bootstrap_b,
        label_fn=lambda r: r.get("lift_em_judge"),
        target_name="lift_em_judge=+1 (LLM-as-Judge)",
    )
    print(report)

    # Also report the original EM-based matrix for direct side-by-side ------ #
    report_em = report_under_judge(
        pool, args.bootstrap_b,
        label_fn=lambda r: r.get("lift_em"),
        target_name="lift_em=+1 (EM, original)",
    )
    print(report_em)

    # Write outputs ---------------------------------------------------------- #
    arch_tag = args.arch.replace("/", "_")
    out_txt = ROOT / "analysis" / f"llm_judge_audit_{arch_tag}.out"
    out_json = ROOT / "analysis" / f"llm_judge_audit_{arch_tag}.json"

    full = []
    full.append(f"LLM-as-Judge robustness audit")
    full.append(f"  model:       {args.model}")
    full.append(f"  arch:        {args.arch}")
    full.append(f"  bootstrap-b: {args.bootstrap_b}")
    full.append(f"  n_pool:      {len(pool)}")
    full.append(f"  n_paired:    {len(paired)}")
    full.append(f"  Cohen κ:     {kappa:.3f}")
    full.append(report)
    full.append(report_em)
    out_txt.write_text("\n".join(full) + "\n")

    summary = {
        "model": args.model,
        "arch": args.arch,
        "bootstrap_b": args.bootstrap_b,
        "n_pool": len(pool),
        "n_paired": len(paired),
        "cohens_kappa_em_vs_judge": kappa,
        "n_helped_em": n_helped_em,
        "n_helped_judge": n_helped_judge,
        "n_helped_both": n_agree_helped,
    }
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n[ok] wrote {out_txt.relative_to(ROOT)}")
    print(f"[ok] wrote {out_json.relative_to(ROOT)}")
    log_line(log_path,
             f"RUN-END     n_pool={len(pool)}  n_paired={len(paired)}  "
             f"kappa={kappa:.3f}  out={out_txt.name}")


if __name__ == "__main__":
    main()
