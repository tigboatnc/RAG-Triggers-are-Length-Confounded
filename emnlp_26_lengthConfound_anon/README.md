# Length-Controlled Evaluation of Training-Free Dynamic-RAG Triggers

Code and results for the EMNLP 2026 short paper. The paper itself is the
documentation; this README only covers what's where, how to run things, and
how to map the artifacts back to the tables/figures in the paper.

## What's in here

```
emnlp_26_lengthConfound/
├── environment.yml                       — conda env spec
├── scripts/                              — decode pipeline (GPU)
│   ├── 04f_run_qa_sota.py                — closed-book decode + rq1-pilot-v1 schema + K=5 sampler
│   ├── 04l_oracle_retrieve_decode.py     — oracle re-decode (paired)
│   ├── 04_dataset.py                     — HotpotQA-distractor slicer
│   ├── 04_dataset_2wiki.py               — 2WikiMultiHopQA-distractor slicer
│   ├── qa_grading.py                     — EM / F1 graders
│   ├── sysinfo.py                        — env / GPU probing
│   ├── run_headline_n200.sh              — orchestrator (Qwen + Gemma × {HotpotQA, 2Wiki}, n=200)
│   ├── download_models.sh                — HF weight downloader with pinned revisions
│   ├── setup.sh                          — conda env bootstrap
│   └── crv/                              — per-step capture + signal extractors
│       ├── __init__.py
│       ├── baseline_gates.py             — DRAGIN, FLARE, ETC, SELFDOUBT-T1, etc.
│       ├── boundaries.py                 — spaCy sentence segmentation + 32-tok fallback
│       ├── hooks.py                      — forward-hook hidden-state capture
│       ├── sota_capture.py               — rq1-pilot-v1 schema capture class (K=5 sampler,
│       │                                  RIND accumulator, verbalised-conf probe, …)
│       ├── triggers.py                   — trigger-firing logic
│       └── vectors.py                    — CRV / drift / variance helpers
├── analysis/                             — post-hoc analysis (CPU only) + result files
│   ├── _common.py                        — AUROC primitive shared across analysis scripts
│   ├── signal_lib.py                     — signal extractors + bootstrap helpers (library)
│   ├── r1_paired_audit.py                — per-arch paired audit driver (takes --arch)
│   ├── moonshot_explore.py               — CABV, FCT (first-commitment-time) + headline
│   │                                       Table 1 / forest plot inputs
│   ├── a1_k1_vs_k5_ablation.py           — §4.5 K-ablation (K=5 vs K=1 flip_rate)
│   ├── uncertainty_line_audit.py         — App. C UNCERTAINTY-LINE head-to-head replication
│   ├── e7_bonferroni_and_spearman.py     — App. A (Bonferroni) + App. E (Spearman ρ)
│   ├── llm_judge_audit.py                — App. B LLM-as-Judge correctness-function
│   │                                       robustness (Janiak 2025 / Ielanskyi 2025 style)
│   ├── plot_survivor_forest.py           — Figure 1
│   └── *.out / *.json                    — result files (see "Results → paper artifact map")
├── externalCodeAudits/uncertainty-line/  — Vashurin 2025 reference implementation (App. C)
├── figs/
│   └── survivor_forest.{pdf,png}         — Figure 1 of the paper
├── runs/
│   └── README.txt                        — trace files (jsonl) live here after decoding;
│                                            not shipped (size); regenerate with run_headline_n200.sh
└── secrets/
    └── openai_api_key.txt.example        — fallback location for the OpenAI API key
                                            (copy to openai_api_key.txt; gitignored)
```

## Quick start (reproducing the numbers from already-decoded traces)

Assumes you have the trace JSONLs at `runs/v2_headline_<arch>/{D1,D2,E1,E2}_*.jsonl`
for `<arch> ∈ {qwen2.5-7b-instruct, gemma-4-26b-a4b-it}`. See "Decoding the
traces" below if you need to regenerate them.

```bash
# 1. Environment
conda env create -f environment.yml
conda activate recollection

# 2. Per-arch paired audit (raw + length-residualised AUROC, Δ vs tokens, B=10k)
python analysis/r1_paired_audit.py --arch qwen2.5-7b-instruct --bootstrap-b 10000
python analysis/r1_paired_audit.py --arch gemma-4-26b-a4b-it  --bootstrap-b 10000

# 3. Headline survivor matrix (Table 1) — CABV, FCT, etc.
python analysis/moonshot_explore.py --bootstrap-b 10000

# 4. §4.5 K=5 vs K=1 ablation (the null-result table)
python analysis/a1_k1_vs_k5_ablation.py --arch qwen2.5-7b-instruct --pilot-root v2_headline --bootstrap-b 10000
python analysis/a1_k1_vs_k5_ablation.py --arch gemma-4-26b-a4b-it  --pilot-root v2_headline --bootstrap-b 10000

# 5. Appendix C: UNCERTAINTY-LINE head-to-head replication
python analysis/uncertainty_line_audit.py --arch qwen2.5-7b-instruct
python analysis/uncertainty_line_audit.py --arch gemma-4-26b-a4b-it

# 6. Appendix A (Bonferroni) + Appendix E (Spearman ρ per cell)
python analysis/e7_bonferroni_and_spearman.py --bootstrap-b 10000

# 7. Figure 1 (forest plot — re-renders from analysis/e7_bonferroni.out)
python analysis/plot_survivor_forest.py

# 8. Appendix B: LLM-as-Judge correctness-function robustness (needs OpenAI API key — see below)
python analysis/llm_judge_audit.py --arch both --model gpt-5-mini --bootstrap-b 10000
```

All analysis scripts write to `analysis/*.out` and `analysis/*.json`, so the
pre-computed result files in this folder are overwritten on each run. The
shipped `.out` / `.json` files are the canonical numbers used in the paper.

## OpenAI API key (only needed for `llm_judge_audit.py`)

1. **`OPENAI_API_KEY` environment variable** (recommended):
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```
   Put this line in your `~/.zshrc` / `~/.bashrc` so it's persistent.
2. **`secrets/openai_api_key.txt`** (single-line fallback, gitignored):
   ```bash
   cp secrets/openai_api_key.txt.example secrets/openai_api_key.txt
   # edit secrets/openai_api_key.txt and replace `sk-REPLACE-ME` with your key
   ```

### Model choice

| `--model` | Use when | Approx cost for the full appendix ($n{=}800 \times 2$ arms) |
|---|---|---|
| `gpt-5-mini` (default) | balanced 2026-era judge; matches Janiak's GPT-4o-Mini-tier choice modernised | ~$0.05 |
| `gpt-4o-mini` | literal Janiak 2025 replication | ~$0.10 |
| `gpt-5-nano` | paranoid cost control | ~$0.01 |
| `gpt-5` | overkill but bulletproof | ~$1-2 |

`gpt-5-mini` is the sweet spot. Override with `--model gpt-4o-mini` only if
you specifically want to mirror Janiak's choice byte-for-byte.

### Caching

Every judgement is cached at `analysis/llm_judge_cache/<arch>/<dataset>/<qid>_<arm>.json`.
Re-running the script skips cached entries — the script is resumable. Delete
the cache directory to force a clean re-run.

## Decoding the traces (GPU pipeline, ~15 h wall on A800 80 GB)

```bash
# 1. Download model weights (requires HF_TOKEN env var for gated repos)
bash scripts/download_models.sh   # Qwen2.5-7B + Gemma-4-26B-A4B-it at pinned SHAs

# 2. Build dataset slices (~30 s each, HF-cached)
python scripts/04_dataset.py       --n 500 --seed 20260601 --name hotpotqa_distractor_500_v2test
python scripts/04_dataset_2wiki.py --n 500 --seed 20260601 --name 2wikimultihop_distractor_500_v2test

# 3. Headline decode (Qwen + Gemma, n=200/cell × 4 cells × {closed, oracle})
bash scripts/run_headline_n200.sh
```

Outputs land at `runs/v2_headline_<arch>/{B_smoke,D1_closed_hotpot,D2_closed_wiki,E1_oracle_hotpot,E2_oracle_wiki}.jsonl`.
Slice manifests with SHA-256 checksums at `runs/v2_headline_<arch>/checkpoints/slice_manifest.json`.

To run a single architecture, set `ARCHS=<arch>`. For a smaller cohort, set `N=100`.

## Models and data

Not shipped (size, license):

- **Models**: `models/<arch_name>/` is the expected location. `scripts/download_models.sh`
  pulls Qwen2.5-7B-Instruct and Gemma-4-26B-A4B-it at the HF revision SHAs
  hard-coded in that script. Set `HF_TOKEN` before running.
- **Datasets**: HotpotQA and 2WikiMultiHopQA are pulled by the dataset slicers
  from HF; the seeded slices are 500 questions each (first 200 used per cell).

## Results → paper artifact map

| Paper element | Source file |
|---|---|
| Abstract / §1 headline numbers (9 nominal, 4 Bonferroni) | `analysis/e7_bonferroni.out` |
| **Figure 1** (forest plot, survivor matrix) | `figs/survivor_forest.{pdf,png}` (from `analysis/e7_bonferroni.out` via `plot_survivor_forest.py`) |
| §4.1 / **Table 1** survivor matrix (4 Bonferroni-robust + 9 nominal-95% + below-line) | `analysis/moonshot_explore.out` (POOLED HEADLINE block) + `analysis/e7_bonferroni.out` |
| §4.3 / Table 2 per-cell FCT (Qwen × Hot 0.677, Qwen × Wiki 0.604, Gemma × Hot 0.642, Gemma × Wiki 0.784) | `analysis/moonshot_explore.out` (PER-CELL breakdown section) |
| §4.5 / Table 3 K=5 vs K=1 (Qwen −0.002, Gemma +0.033) | `analysis/a1_k1_vs_k5_r1_{qwen,gemma}.{out,json}` |
| Appendix A / Table 4 Bonferroni-corrected survivor set | `analysis/e7_bonferroni.out` |
| Appendix B / Table 5 LLM-as-Judge robustness (Cohen's κ=0.707 + survivor matrix under judge) | `analysis/llm_judge_audit_<arch>.{out,json}` |
| Appendix C / Table 6 ε_protocol Qwen 0.029, Gemma 0.027 (UNCERTAINTY-LINE replication) | `analysis/uncertainty_line_audit_{qwen2.5-7b-instruct,gemma-4-26b-a4b-it}.{out,json}` |
| Appendix D / Table 7 FCT vs Datta `t*` operational diff | (documentation only — no analyser) |
| Appendix E / Table 8 length-vs-signal Spearman ρ per cell | `analysis/e7_spearman.out` |

## Bootstrap formula

All paired-bootstrap CIs in the paper use

```
Δ = AUROC(s_resid) − AUROC(tokens_generated)
```

where `s_resid` is the OLS residual of the signal on `log(tokens + 1)` on the
same cohort. `B = 10,000` resamples. This matches the §3.2 paper formula and
the Vashurin et al. (2025) UNCERTAINTY-LINE protocol applied to the
dynamic-RAG trigger pool. The bootstrap is implemented in
`analysis/signal_lib.py::bootstrap_auroc_diff`; the residualisation in
`ols_residual` in the same module.

## Two-arch scope

Trace files for Qwen2.5-7B-Instruct and Gemma-4-26B-A4B-it are the headline
pool (n = 800 paired). The `analysis/` scripts run on either arch via the
`--arch` flag (`r1_paired_audit.py`, `uncertainty_line_audit.py`,
`a1_k1_vs_k5_ablation.py`). The `moonshot_explore.py` and `e7_*` scripts
load both archs and produce the pooled headline numbers.

## Hardware

Decoding: single NVIDIA A800 80 GB (or A100 80 GB / H100), driver 550+,
CUDA 12.4+. Total wall ~15 h for the two-arch headline decode.
Analysis: pure CPU + numpy, ~5 min total across all scripts at `B = 10,000`.