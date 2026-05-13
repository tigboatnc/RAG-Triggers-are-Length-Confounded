# _rq1-plan.md

> Forward-looking design + rationale + understanding for **RQ1**.
> Self-sufficient. Companion log: `_rq1-agent.md` (pure experiment ledger).
> Lit anchor: `litReview/-rq1-litReview.md` + `litReview/-rq1-litReview-sources.csv`.
> Per-model protocol: `_MODEL.md`.
>
> No claim derives statistical evidence from v1 cohorts or from the
> `u_cons` family (unpublished, exploratory throughout RQ1).

---

## Three goals of this work

Every experiment in `_rq1-agent.md` must serve at least one of:

1. **TEST the working hypothesis.** Of published parameter-free,
   training-free, decode-time dynamic-RAG triggers, which (if any)
   survive a length-controlled evaluation protocol on multi-hop
   closed-book QA paired with oracle retrieval? Falsify or confirm
   *DRAGIN-RIND is the sole length-robust survivor*.
2. **PREPARE FOR OTHER HYPOTHESES.** Heavy logging at every layer
   (per-trace, per-step, per-resample, per-sample), maximum-value
   capture per compute run, schema-superset trace dump. Post-hoc
   re-analysis must be possible without re-decoding. The cheapest
   re-decode is the one we never have to repeat.
3. **LEARN THE DOMAIN.** Each external method we audit teaches us
   something about the design space (what worked, why; what the
   field's failure modes are). The paper's "discussion / future work"
   section is built from this learning, not from speculation.

These are equally weighted. An experiment that only does (1) is
under-instrumented; an experiment that only does (3) is unfalsifiable.

---

## RQ1 statement

> Of published parameter-free, training-free, decode-time dynamic-RAG
> triggers, which (if any) survive a length-controlled evaluation
> protocol on multi-hop closed-book QA paired with oracle retrieval?

Operationalized: for each candidate trigger signal, fit
`signal ~ log(tokens_generated + 1)` by OLS on the cohort, recompute
AUROC and precision-at-budget on the residual, and run a paired
bootstrap (B=2000 default, B=10 000 for headline) for
`signal − tokens_generated` on raw AUROC. A "survivor" is a signal
whose **residualized AUROC and paired-margin point estimate are both
positive on at least one of the two evaluation datasets** (in-domain
weighted as primary).

Label: `lift_em ∈ {+1, 0, −1}` from a paired closed-book vs
oracle-retrieval re-decode of the same trace. Primary target
`lift_em = +1` ("retrieval would have helped"). Secondary target
`closed_em = 0` (descriptive only).

Honest contribution scope: **trigger-evaluation methodology + a
re-ranking of the 2024–2026 trigger pool under the new protocol**.
Positive headline if RQ1 lands as currently sketched: *"DRAGIN-RIND is
the only published dynamic-RAG trigger that survives length-controlled
evaluation; we propose the residualization protocol and show how the
field's signal-ranking changes under it."*

---

## Lit anchor (from `litReview/-rq1-litReview.md`)

The candidate trigger pool — methods named as primary dynamic-RAG
triggers in §3.1 / §4.2 / §4.4 of the lit review:

| trigger | citation | family |
|---|---|---|
| **DRAGIN-RIND** | Su et al. 2024 (2403.10081) | per-token attention × entropy × content-filter |
| **FLARE** | Jiang 2023 | token-probability threshold |
| **DynamicRAG** | Sun 2025 | RL-trained reranker (orthogonal to triggering, excluded) |
| **Adaptive-RAG** | Jeong 2024 | classifier-based question-level router (orthogonal, excluded) |
| **Layered Query Retrieval** | Huang 2024 | semantic-rule complexity router (orthogonal, excluded) |
| **Self-RAG** | Asai 2024 | trained reflection tokens (out of family — supervised, used as reference model only) |
| **SeaKR** | Su et al. 2025 | K=20 hidden-state EigenScore + perplexity + ln-entropy + energy |
| **DioR** | Guo et al. 2025 | trained MLP/RNN classifiers on IG features (out of family — supervised, excluded) |
| **MBA-RAG** | Tang 2024 | RL bandit on per-question features (trained per-arch) |
| **SelfCheckGPT** | Manakul 2023 | K-sample text-consistency (NLI / Ngram / BERTScore variants) |
| **DH-RAG, DoTA-RAG, OmniSearch, …** | various 2025 | full-system / multimodal / agentic (orthogonal) |

The lit review's own §4 evidence table flags two open problems with
direct bearing on RQ1:

> *"Evaluation remains overlap-heavy; cost/safety often neglected"*  (6/10 evidence; Brown SLR; Yang CRAG).
> *"Generalization under domain shift is still brittle"*  (4/10 evidence; Yang; Chen RGB).

RQ1 is positioned as a contribution to those two open problems,
restricted to the **parameter-free decode-time trigger subfamily** (with
trained-classifier methods either dropped or included as reference
rows with explicit caveat).

---

## Data inventory (what is on disk; the floor for further work)

### Models — final RQ1 pool (post-pilot revision, 2026-05-13 evening)

| model | generation | architecture | role | status |
|---|---|---|---|---|
| **Qwen2.5-7B-Instruct** | 2025 | dense, 28 layers, GQA 28Q/4KV | 2025 reassessment-cohort backbone (used by SR-RAG, DyPRAG family) | new — not yet downloaded |
| **Phi-4-mini-instruct** | 2026 | dense edge, 32 layers; mid-stack capture layer 16; DRAGIN tuple-index 31 | frontier dense edge | pilot complete |
| **Gemma-4-26B-A4B-it** | 2026 | MoE-sliding 5:1, 30 layers; mid-stack layer 15; DRAGIN tuple-index 29 | frontier MoE | pilot complete |
| ~~Llama-2-7B-Chat~~ | 2023 | dense | canonical baseline-paper backbone | **dropped** — slow gated download on compute box; transitive comparability via SR-RAG/DyPRAG literature using Qwen2.5; left as optional future addition (instructions retained in handover §) |
| ~~Qwen3.6-35B-A3B-Instruct~~ | 2026 | MoE-hybrid GDN 3:1 | excluded | **dropped** — Qwen3.6 limitation note below |

Final pool: **3 backbones** across 2 generations (2025 + 2026) and
2 architecture families (dense + MoE-sliding). Tighter table reads
better for a 4-page short paper than a 4-arch grid.

Excludes Qwen3.6 by principled limitation (chain-of-thought collapses
under matched-protocol greedy + `enable_thinking=False`); details in
`_rq1-agent.md` Experiment P0 Qwen failure section. Weights for Qwen3.6
can be deleted from compute box to free disk.

Llama-2-7B-Chat dropped as a current-campaign model only; the Meta
gated repo took too long to download on the compute box too. The
relevant 2024 trigger papers (DRAGIN, SeaKR, Self-RAG) used Llama-2-7B
as backbone, but Qwen2.5-7B has subsequent literature (SR-RAG 2025;
DyPRAG 2025 family) that benchmarks DRAGIN/FLARE on it — transitive
comparability holds. If you ever want to add Llama-2-7B-Chat back,
the download instructions remain in the handover section under
"Optional future addition: Llama-2-7B-Chat."

Decoding: Option-A protocol per `_MODEL.md §3` — greedy, per-arch
repetition penalty (Phi 1.05; others 1.0 unless smoke says otherwise),
early-stop on `Answer:` regex or EOS, `max_new_tokens` capped.

Qwen2.5-7B protocol is TBD pending the new-model smoke
(`scripts/04c_new_model_smoke.py`); proposed defaults (rep=1.0, no
thinking-mode kwargs — Qwen 2.5 has no thinking mode) are validated
by the smoke before any sweep.

### Datasets

| dataset | role | slice file | n |
|---|---|---|---|
| HotpotQA-distractor | in-domain | `data/hotpotqa_distractor_500_v2test.jsonl` | 500 |
| 2WikiMultiHopQA-distractor | zero-shot OOD | `data/2wikimultihop_distractor_500_v2test.jsonl` | 500 |

Both slices: seed `20260601`, provably disjoint from v1 200-question
calibration. 2WikiMHQA is treated as zero-shot OOD: thresholds and OLS
coefficients fit on in-domain are applied unchanged to OOD.

### Trace files (paired closed-book × oracle re-decode, Phi only)

| file | rows | source script |
|---|---:|---|
| `runs/v2_exp1_phi-4-mini-instruct/D1_closed_hotpot.jsonl` | 500 | `scripts/04f_run_qa_sota.py` |
| `runs/v2_exp1_phi-4-mini-instruct/D2_closed_wiki.jsonl` | 500 | same |
| `runs/v2_exp1_phi-4-mini-instruct/E1_oracle_hotpot.jsonl` | 500 | `scripts/04l_oracle_retrieve_decode.py` |
| `runs/v2_exp1_phi-4-mini-instruct/E2_oracle_wiki.jsonl` | 500 | same |

Paired-joined with all-signals-populated filter:

| dataset | n_paired | closed_incorrect | lift+1 | lift−1 | base helpful |
|---|---:|---:|---:|---:|---:|
| HotpotQA in-domain | **393** | 245 | 42 | ~37 | 0.107 |
| 2WikiMHQA OOD | **452** | 262 | 75 | ~54 | 0.166 |

### Per-step features captured (current schema)

Per reasoning step (boundary detector: spaCy sentence with 32-token fallback):

- `mean_token_entropy`, `boundary_token_entropy` — greedy entropies.
- `dragin_rind_max`, `dragin_n_content_tokens` — DRAGIN-RIND online.
- `drift_cosine`, `variance_l2_window`, `hidden_norm` — mid-stack CRV
  trajectory features (exploratory).
- `hvr`, `hedge_count`, `verify_count` — SELFDOUBT-T1 (Pandey HVR).
- `u_verb`, `u_ent`, `u_cons` — verbalized / entity / K=3-consistency
  probes (`u_cons` exploratory; `u_ent` null due to missing
  `rank_bm25` at decode time).

Per-trace summary: `tokens_generated`, `wall_clock_s`, `n_steps`,
`em`, `f1`, `label_correct_em`, full per-baseline gate-decision audit
trail.

**Schema gaps that motivate next-experiment captures (Goal 2):**
- per-token `max_logprob` arrays — needed for FLARE.
- K=20 sample hidden states per step — needed for SeaKR eigen_score.
- K=3 sample completion **texts** (we keep only the aggregate `u_cons`) — needed for SelfCheckGPT.

---

## Fresh compute-box bootstrap

Whole stack rebuilds from scratch on a new GPU box in well under an
hour. Long-form authoritative procedure: `_MODEL.md §9` (validation
checklist for a new compute box). Concrete command sequence below.

### Phi-only (single-arch experiments — E1, E2, parts of E7/E8/E9/E10)

```bash
bash scripts/setup.sh                                           # conda env, lockfile, ~3-4 min
bash scripts/download_models.sh                                 # Phi only, ~10 s warm
python scripts/sysinfo.py                                       # confirm CUDA + VRAM + versions
python scripts/04_dataset.py                                    # v1 200-q HotpotQA calibration cohort
python scripts/04_dataset.py \
    --n 500 --seed 20260601 --split-seed 20260601 \
    --exclude-ids-from data/hotpotqa_distractor_200.jsonl \
    --all-test --name hotpotqa_distractor_500_v2test            # in-domain test slice
python scripts/04_dataset_2wiki.py \
    --n 500 --seed 20260601 \
    --name 2wikimultihop_distractor_500_v2test                  # OOD test slice
```

### Multi-arch (E3 / E4 / E5 / E6)

Swap `download_models.sh` → `download_models.sh phase1.4` (pulls
Phi + Gemma-4-26B-A4B-it + Qwen3.6-35B-A3B at pinned SHAs in
`REV_*`, ~125 GB total) and run the cross-arch smoke before any
sweep:

```bash
bash scripts/download_models.sh phase1.4
python scripts/04b_protocol_smoke.py                            # must report `[ok] all 3 models pass Option A protocol smoke`
```

The wrapper scripts hard-fail on smoke failure. Sweep is blocked
until smoke passes.

### Artifacts: what rebuilds vs what carries over the wire

| artifact | source | size | wall clock |
|---|---|---|---|
| conda env | `environment.yml` + `setup.sh` | — | ~3–4 min |
| `runs/env_lockfile.txt` | `pip freeze` after setup | — | ~5 s |
| Phi-4-mini-instruct weights | HF pinned SHA in `download_models.sh` | ~7 GB | ~10 s warm |
| Gemma-4-26B-A4B-it weights | HF pinned SHA | ~47 + 1.6 GB | ~5–15 min |
| Qwen3.6-35B-A3B weights | HF pinned SHA | ~69 GB | ~5–15 min |
| `data/hotpotqa_distractor_200.jsonl` | `04_dataset.py` | ~5 MB | ~30 s (HF download dominates) |
| `data/hotpotqa_distractor_500_v2test.jsonl` | `04_dataset.py --n 500 --all-test --exclude-ids-from ...` | ~12 MB | ~5 s (HF cache warm) |
| `data/2wikimultihop_distractor_500_v2test.jsonl` | `04_dataset_2wiki.py --n 500` | ~9 MB | ~60–120 s (first-time mirror probe) |

Time budget end-to-end on a recent reference run (vast.ai A800
80 GB): ~9 min wall (setup 3:35, downloads 2:38, extras 0:35,
dataset rebuild 0:08, sysinfo 0:21). Plus the test-slice builds
(~30–60 s each on a warm HF cache).

All other state (model weights, scripts, env) is reproducible from
the repo. The `models/<name>.revision` files (created on first
download) are git-ignored but worth carrying in snapshots — they
fingerprint the exact HF commit. Same for `runs/sysinfo_*.json`
and `runs/env_lockfile.txt`.

### Hardware floor

A800 80 GB or A100 80 GB or H100; driver 550+; CUDA 12.4+ runtime;
≥200 GB host RAM; ≥150 GB free disk for the three-model pool. Phi
alone fits comfortably on a 40 GB card. **If the box does not match
the `_MODEL.md §6` hardware row — different driver, different CUDA,
different VRAM ceiling — stop and audit before continuing.** Do not
silently bump SHAs or the pinned torch / transformers / bnb versions
(it breaks the reproducibility contract).

### Submodule clone (RQ1-specific)

After the env is up, also init the trigger-pool baseline submodules
pinned in `.gitmodules`:

```bash
git submodule update --init --recursive externalCodeAudits/
```

Pulls 6 repos (DRAGIN, FLARE, MBA, seakr, self-rag, selfcheckgpt) at
the SHAs the STEP-2 audit was performed against. Read
`externalCodeAudits/README.md` for what each method is and why it is
in the pool.

---

## NEW: handover for fresh Cursor chat on compute box (2026-05-13 +)

This section is for the **next compute-box session in a new Cursor chat**.
The scaffolding (env, Phi/Gemma/Qwen3.6 weights, pilot data, scripts)
is already on the box from the prior session. The new session needs:

1. **Drop Qwen3.6 weights** to free ~67 GB disk:
   ```bash
   rm -rf models/qwen3.6-35b-a3b-instruct
   ```
   Documented limitation; pilot data + smoke output kept under
   `runs/v2_pilot_qwen3.6-35b-a3b-instruct/` (small).

2. **Download Qwen2.5-7B-Instruct** under HF token (`HF_TOKEN`
   already configured on the box; apache-2.0, ungated, ~14 GB):
   ```bash
   conda run -n recollection python -c "
   from huggingface_hub import snapshot_download
   snapshot_download(
       repo_id='Qwen/Qwen2.5-7B-Instruct',
       revision='a09a35458c702b33eeacc393d103063234e8bc28',
       local_dir='models/qwen2.5-7b-instruct',
       allow_patterns=['*.safetensors','*.json','*.txt','tokenizer*','LICENSE'],
       ignore_patterns=['*.bin','*.pth'],
       max_workers=4,
   )"
   echo "a09a35458c702b33eeacc393d103063234e8bc28" > models/qwen2.5-7b-instruct.revision
   ```

3. **Run the new-model smoke + paste JSON back to agent** so
   `_MODEL.md` can be updated:
   ```bash
   python scripts/04c_new_model_smoke.py \
       --model qwen2.5-7b-instruct \
       --model-dir models/qwen2.5-7b-instruct \
       --thinking-mode na \
       --repetition-penalty 1.0
   ```
   Output JSON lands at `runs/04c_new_model_smoke_qwen2.5-7b-instruct_<ts>.json`,
   ~5 KB self-contained. Copy contents into the next chat message and
   the agent will:
   - Add the §1 / §2 / §3 / §5 / §6 rows to `_MODEL.md`.
   - Add the per-model entry to `PROTOCOL` in
     `scripts/04b_protocol_smoke.py` and `MODEL_PROTOCOL` in
     `scripts/04_run_qa.py`.
   - Update `_rq1-plan.md` Burden 3 model set with confirmed
     `repetition_penalty` and `chat_template_kwargs`.

4. **Once `_MODEL.md` is updated and PROTOCOL dicts are in lockstep,
   add Qwen2.5 to the pilot orchestrator + re-run pilot for the new
   arch only**:
   ```bash
   ARCHS="qwen2.5-7b-instruct" bash scripts/run_pilot_p0.sh
   ```
   Existing Phi + Gemma pilot files are skipped (idempotent). Adds
   ~25 min wall (one small dense model, K=5 sampler, 25 questions × 4
   phases).

5. **After pilot: pull, run analyzer (CPU), commit:**
   ```bash
   python analysis/pilot_audit.py | tee analysis/pilot.out
   git add runs/v2_pilot_*/ analysis/pilot.out data/*_pilot.jsonl
   git commit -m "rq1 P0: pilot results across 3-arch pool (Phi+Gemma+Qwen2.5)"
   git push
   ```

### Schema version `rq1-pilot-v2` — extended capture for novel triggers (added 2026-05-13 evening)

P0 (rq1-pilot-v1) audited the published trigger pool. Post-P0
analysis (`analysis/p0_free_signals.py`, `_rq1-agent.md` P0.5) found
that **4 of 8 free post-hoc signals carry real cross-arch information**
that the published pool misses — most strikingly: `sec/token` (decode
slowdown) on Phi where DRAGIN is weakest. **This indicates the
published trigger pool is undercomplete; new architectural-aware
triggers are findable.**

P1 introduces 4 NOVEL trigger captures (none in published pool):

| # | field | model class | mechanism |
|---|---|---|---|
| N1 | `moe_routing_entropy` | MoE only | mean entropy over MoE expert-routing distribution per step |
| N2 | `prefix_answer_now` | all | KV-fork, prefill `Answer: `, greedy 10 tokens — "what would the model answer NOW" |
| N3 | `multi_layer_dragin` | all (esp. hybrid) | DRAGIN-RIND at {first, mid, last} full-attn layers |
| N4 | `attention_sink_frac` | any attn-bearing | fraction of attention to first-4 tokens (StreamingLLM sink) |

Code (committed-staged in 2026-05-13 evening session):
- `scripts/crv/sota_capture.py` — 4 new helpers + class.
- `scripts/04f_run_qa_sota.py` — `--schema-version=rq1-pilot-v2` flag wires them in. Graceful dense-model fallback for `output_router_logits`.
- `scripts/run_pilot_p1.sh` — sister orchestrator. Output at `runs/v2_pilot_v2_<arch>/`. Idempotent.

**Launch on the box (parallel split per parent):**

```bash
# Box A (any ≥40 GB; small-model box):
ARCHS="phi-4-mini-instruct qwen2.5-7b-instruct" bash scripts/run_pilot_p1.sh
# Box B (warm A800 with Gemma weights):
ARCHS="gemma-4-26b-a4b-it" bash scripts/run_pilot_p1.sh
# After both done:
git pull on each box → analyze pilot_audit-v2 (TBD; extend pilot_audit.py for v2 fields)
```

Wall: ~40 min (Box A small models serial) ‖ ~3 h (Box B Gemma) → **~3 h parallel**.

### Status of project trajectory after P0 + P0.5 + P1 design (2026-05-13 evening)

The pilot data has shifted the most likely paper trajectory away from
"DRAGIN-as-survivor" and toward **"the published trigger pool is
undercomplete; here are 8 candidate triggers (4 derived post-hoc from
existing schema; 4 captured in rq1-pilot-v2) that the literature
missed."** Detailed loose-ends + the reframe argument live in
`_rq1-agent.md` §"Loose ends to flag for next session."

**Burden priority shifts as of this session:**
- Burden 3 (cross-arch n=200) → **highest** (DRAGIN+lift on Gemma at n=25 is **0.89/0.94**; needs n=200 to confirm)
- Burden 4 (MuSiQue) → **high** (cross-arch length-correlation is dataset-heterogeneous; 3 datasets needed)
- Burden 5 (Bonferroni at k≥14, B=10 000) → **high** (signal pool grew from 11 to 14+; correction matters)
- Burden 2 (SeaKR K=20) → **medium-low** (K=5 proxy shows direction-of-effect; full K=20 confirms not discovers)
- Per-question-type stratification → **required** in headline tables

**The single most paper-defining experiment now**: Burden 3 cross-arch
n=200 with rq1-pilot-v2 schema + per-question-type stratification.
That gives the headline numbers + the cleanest dataset-specificity
evidence + the v2 NOVEL-trigger first-pass numbers in one campaign.

### NEW: Post P0.6 deep-exploration (2026-05-13 late evening)

**Discovery: Answer Instability Detection (AID) — novel trigger family.**

Deep exploratory analysis (`analysis/_explore_novel.py`, `_explore_deep_dive.py`,
`_explore_ans_flip_variants.py`) on the P0 traces found a signal that dominates
everything in the published trigger pool:

`flip_rate` = fraction of answer-bearing reasoning steps where K=5 boundary-
completion answers disagree with the previous step. AUROC for lift_em=+1:
**0.733 raw, 0.735 residualized, r(tokens)=0.008.**

This is: (a) highest single-signal AUROC for retrieval-helpfulness prediction,
(b) orthogonal to length (r<0.01), (c) orthogonal to DRAGIN (r=0.25),
(d) complementary per-arch (wins where DRAGIN is weakest: Qwen hot 0.867 vs 0.571),
(e) computable online from existing K=5 captures at zero extra inference cost,
(f) novel — no published trigger uses step-to-step answer stability.

**Paper trajectory shifted AGAIN:**
- ~~"Published pool is undercomplete" (descriptive)~~
- **NOW: "We introduce AID, a novel trigger family; under length-controlled
  evaluation across 3 archs × 2 datasets, it achieves highest AUROC and
  specifically complements DRAGIN where it is weakest."**

This makes the paper a CONTRIBUTION paper (novel trigger + methodology), not
just an evaluation paper.

**Revised burden priorities (post P0.6):**

| burden | priority | rationale |
|---|---|---|
| Burden 3 (cross-arch n=200, rq1-pilot-v1 schema) | **HIGHEST** | AID flip_rate is already computable from K=5 samples; n=200 gives paired-bootstrap power for DRAGIN vs flip_rate head-to-head |
| Burden 4 (MuSiQue n=200) | **HIGH** | 3rd dataset seals generalization; closes "benchmark-specific" critique |
| Burden 5 (Bonferroni B=10k, k≥15) | **HIGH** | signal pool now 15+; correction at k=15 is non-trivial |
| Burden 2 (SeaKR K=20) | **DROPPED** | K=5 proxy shows direction; compute better spent on MuSiQue |
| P1 novel triggers (N1-N4) | **insurance** | Gemma P1 running; Phi+Qwen deferred unless Gemma reveals something |
| Per-question-type stratification | **required** | architecture-heterogeneity finding demands it |

**The 2 remaining long runs (DECIDED 2026-05-13 late):**

1. **RUN 1 — n=200 cross-arch headline (Burden 3).** Phi+Gemma+Qwen2.5 ×
   HotpotQA+2Wiki, rq1-pilot-v1 schema (K=5), paired oracle re-decode. ~22h serial
   / ~10h with 2 boxes. Confirms flip_rate + DRAGIN at publishable power.
2. **RUN 2 — MuSiQue generalization (Burden 4).** Same 3 archs, n=200, v1 schema,
   closed+oracle. ~11h. 3rd dataset closes "benchmark-specific" critique.

**Pre-launch TODO for Run 1:**
- Add `flip_rate`, `late_flips`, `flip_count`, `ans_convergence` to
  `analysis/pilot_audit.py` SIGNAL_DEFS (post-hoc; no decode-side changes)
- Add `artifact_token_flag` diagnostic (Gemma-specific)
- Add per-question-type stratification to the report_cell output
- Verify `scripts/04_dataset_musique.py` exists or write it (for Run 2)
- Confirm the v2 500-question slices + 200-question calibration are on the box

**Pre-launch TODO for Run 2:**
- Build MuSiQue slice: `scripts/04_dataset_musique.py --n 200 --seed 20260601`
- Adapt oracle-decode for MuSiQue paragraph format
- Same capture pipeline as Run 1 (reuse scripts unchanged)

**AID signal requires NO schema changes** — it's derived post-hoc from the K=5
`samples[].text` field already captured in rq1-pilot-v1. The n=200 analyzer
(`pilot_audit.py`) just needs `flip_rate`, `late_flips`, and `flip_count` added
to SIGNAL_DEFS.

**Evaluated and rejected: H_diff (entropy growth) as trigger.** Per-cell AUROC
for incorrectness looks good (Gemma Hot 0.750, Wiki 0.718) but help_AUROC=0.497
(chance). It predicts wrong answers, not retrieval need. Filed under G3 (failure-
mode taxonomy) not G1 (trigger evaluation).

**Confirmed: Gemma artifact-token phenomenon.** 60-64% of Gemma traces contain
template-collapse tokens (`<turn|>`, `<end_of_turn>`). Last-step impact is small
(mean 6 tokens) because protocol stops at EOS/Answer:. Worth documenting as
measurement caveat; adding artifact-flag to the n=200 analyzer as diagnostic.

### Optional future addition: Llama-2-7B-Chat

If you decide later that direct comparability to the canonical 2024
trigger-paper backbone is worth the extra wall (license-form delay +
slow gated download), the steps are:

```bash
# (a) Accept license at https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
# (b) Download (gated, slow even on compute box):
conda run -n recollection python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='meta-llama/Llama-2-7b-chat-hf',
    revision='f5db02db724555f92da89c216ac04704f23d4590',
    local_dir='models/llama-2-7b-chat-hf',
    allow_patterns=['*.safetensors','*.json','*.model','*.txt','tokenizer*','LICENSE*'],
    ignore_patterns=['*.bin','*.pth'],
    max_workers=4,
)"
echo "f5db02db724555f92da89c216ac04704f23d4590" > models/llama-2-7b-chat-hf.revision

# Or, faster ungated mirror (same weights):
#   repo_id='NousResearch/Llama-2-7b-chat-hf'
#   revision='351844e75ed0bcbbe3f10671b3c808d2b83894ee'

# (c) Smoke + analyzer same as Qwen2.5 above.
```

Not part of the current campaign. Left here for one-step inclusion
later if reviewer feedback or scope expansion calls for it.

### Disk budget on the new box (after the model swap)

| weights present | size |
|---|---:|
| Phi-4-mini-instruct | ~7 GB |
| Gemma-4-26B-A4B-it | ~49 GB |
| Qwen3.6-35B-A3B (DELETED) | −67 GB |
| Qwen2.5-7B-Instruct (NEW) | ~14 GB |
| **net change** | **−53 GB → ≈70 GB total weights** |

Plus pilot traces (~30 MB), submodule clones (~4 GB), env (~5 GB),
data slices (~25 MB), HF cache (~5 GB). Comfortable on a 100 GB box.

---

## Compute-box handover — RQ1 P0 (new box, full bootstrap)

This is the **linear sequence** to take a fresh A800 box from
empty disk to running `bash scripts/run_pilot_p0.sh`. Assumes Cursor
running on the box. Total bootstrap: ~30–40 min wall (network-bound),
pilot decode: ~2 h, analysis: ~15 min.

### State on the new box at start

- No conda env, no pinned torch/transformers, no model weights,
  no v1/v2 trace files, no submodules cloned, no pilot slices built.
- HF cache may or may not be warm depending on the box's history.
- Old `runs/v2_exp1_*/`, `runs/phase4_*/`, `runs/v2_xarch_*/` from
  the prior box are **not present** — they live on the prior box.
  RQ1 P0 produces fresh traces that do not depend on them.

### Linear sequence (one block to copy-paste)

```bash
# 0. Repo + submodules
git clone <repo-url> recollection && cd recollection
git submodule update --init --recursive externalCodeAudits/

# 1. Env (~3-4 min); writes runs/env_lockfile.txt
bash scripts/setup.sh

# 2. Models (~5-15 min, network-bound; ~125 GB total for the 3 archs)
bash scripts/download_models.sh phase1.4

# 3. Sysinfo + per-arch protocol smoke; both should pass before any sweep
python scripts/sysinfo.py
python scripts/04b_protocol_smoke.py    # must report `[ok] all 3 models pass Option A protocol smoke`

# 4. Build the existing v2 calibration / test slices (cheap; HF download
#    dominates first time, ~30-60 s warm). These are referenced by the
#    P0 slice builder's --exclude-ids-from for disjointness.
python scripts/04_dataset.py
python scripts/04_dataset.py \
    --n 500 --seed 20260601 --split-seed 20260601 \
    --exclude-ids-from data/hotpotqa_distractor_200.jsonl \
    --all-test --name hotpotqa_distractor_500_v2test
python scripts/04_dataset_2wiki.py \
    --n 500 --seed 20260601 \
    --name 2wikimultihop_distractor_500_v2test

# 5. Run the pilot (~2 h; orchestrator hard-caps at 3 h).
#    Builds fresh n=25 pilot slices on first invocation.
bash scripts/run_pilot_p0.sh

# 6. Offline analysis (~15 min, no GPU).
python analysis/pilot_audit.py | tee analysis/pilot.out

# 7. Commit + push
git add runs/v2_pilot_*/ data/*_pilot.jsonl analysis/pilot.out
git commit -m "rq1 P0: pilot results, n=25 x 3 archs x 2 datasets"
git push
```

### What gets produced (committable artifacts)

| path | size estimate | why commit |
|---|---:|---|
| `data/hotpotqa_distractor_25_pilot.jsonl` | ~0.5 MB | reproducibility |
| `data/2wikimultihop_distractor_25_pilot.jsonl` | ~0.4 MB | reproducibility |
| `runs/v2_pilot_<arch>/{B,D1,D2,E1,E2}_*.jsonl` × 3 archs | ~27 MB total | pilot traces with rq1-pilot-v1 schema |
| `runs/v2_pilot_<arch>/checkpoints/slice_manifest.json` × 3 | <30 KB | sha256s + row counts (per small-checkpoint policy) |
| `runs/v2_pilot_<arch>/logs/*.log` × 3 | ~1 MB total | per-phase logs incl. wall-clock + sanity-gate output |
| `analysis/pilot.out` | <100 KB | per-cell numbers |

### What does NOT need to be present on the new box

- Old run dirs (`runs/v2_exp1_*`, `runs/phase4_*`, `runs/v2_xarch_*`).
- Old analysis outputs (`analysis/length_audit_v2.out`, `analysis/iter*.out`).
- Old draft / paperwork files.
- The legacy `_PLAN.md` / `_AGENT.md` / `_agent_v2.md` (those have been
  superseded by `_rq1-plan.md` + `_rq1-agent.md` for RQ1 work).

`analysis/length_audit_v2.py` is still useful as a reference for
`pilot_audit.py` but does not need to be runnable on the new box for
RQ1 P0 to land. (It will need the n=500 v2 traces to produce the
"Established first proof" table; if you want that table reproduced on
the new box, restore the v2 trace files from the prior box first.)

### Handover sanity check (run this before launching)

```bash
ls models/                                         # 3 dirs, each with *.safetensors
ls data/{hotpotqa_distractor_200,hotpotqa_distractor_500_v2test,2wikimultihop_distractor_500_v2test}.jsonl
git submodule status                               # 6 lines, each with a 40-char SHA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
test -x scripts/run_pilot_p0.sh && bash -n scripts/run_pilot_p0.sh && echo "orchestrator syntax OK"
test -f analysis/pilot_audit.py && python -c "import ast; ast.parse(open('analysis/pilot_audit.py').read())" && echo "analyzer syntax OK"
```

If any of those fails: stop; re-run the affected step from §"Linear
sequence" before invoking the pilot.

---

## Established first proof (the floor)

**Script:** `analysis/length_audit_v2.py`
**Output:** `analysis/length_audit_v2.out`
**Compute:** ~12 s under conda env `gzn`, no GPU.

11 parameter-free decode-time signals audited under raw AUROC,
length-residualized AUROC, paired bootstrap (B=2000) vs
`tokens_generated`. See `_rq1-agent.md` Experiment 0 for the headline
tables (HotpotQA + 2Wiki) and full numbers.

Established:

1. **Length is the dominant in-domain signal** — `tokens_generated`
   AUROC = 0.749 for `lift_em=+1` on HotpotQA-distractor.
2. **DRAGIN-RIND.max is the only positive-direction survivor** under
   length-residualization on both datasets (residual AUROC 0.616
   in-domain, 0.582 OOD; only signal with positive paired-margin point
   estimate on OOD).
3. **Three signals are paired-significantly worse than length**: ETC,
   SELFDOUBT-T1 HVR, variance_l2_window.
4. **One diagnostic counterexample**: 2Wiki `bridge_comparison`
   subset has inverse length-correctness correlation; internal
   signals gain under length-residualization there.

Honest scope: one model, two datasets in same task family, max
aggregator only, parameter-free family only. The four "not yet
established" gaps are the four reviewer-critiques in the next section.

---

## Open burdens / planned experiments

### Hostile-review playbook (anticipated EMNLP-short critiques)

1. **"Single architecture is too narrow."** RQ1 is methodological;
   one model is insufficient.
2. **"You didn't benchmark SeaKR (Su 2025) properly."** Headline
   `eigen_score` requires K=20 hidden-state sampling, not in on-disk
   audit.
3. **"`max` aggregator is a free parameter."** Mean / first-step / min
   variants might invert the survivor ranking.
4. **"Length-confound might be benchmark-specific."** The
   `bridge_comparison` exception is direct evidence of structural
   exceptions; reviewer will ask whether MuSiQue / Bamboogle / IIRC
   show the same pattern.

The burdens below address all four.

### Burden 1 — Re-apply audit extensions (no compute, ~30 min)

A prior session extended `analysis/length_audit_v2.py` from 11 → 19
signals (3 SeaKR ln-entropy proxies + 6 aggregator-robustness rows:
mean / first-step variants of `u_cons / dragin / ent`, plus
`ent.first`). Lost to a `git reset --hard`; re-applicable from session
memory. Prerequisite for everything below.

**Expected** (from prior session, not currently on disk):
- All three SeaKR ln-entropy proxies paired-significantly worse than
  `tokens_generated` in-domain.
- All `mean` and `step1` aggregator variants of u_cons / dragin / ent
  paired-significantly worse in-domain.
- DRAGIN-RIND retains survivor status across aggregators in raw AUROC
  (max > mean > step1); `dragin_rind.mean` shows OOD residualized
  P@5% = 0.391 vs base 0.166 — largest in OOD table.

**Decision rule:** if pattern reproduces, critique 3 ("aggregator-robust") closes.

### Burden 2 — SeaKR eigen_score (the actual signal, not a proxy)

**Why.** Hidden-state-geometry signal family; could break the
all-collapse pattern.

**Plan.**
- Re-decode 500+500 with K=20 sample completions per step + per-sample
  mid-stack hidden-state capture.
- Compute `eigen_score = log det(Σ + αI)` per step; aggregate by
  max + mean + first-step (3 audit rows).
- `SeaKR/SEAKR/reasoner.py:101-113` is the authoritative reference.

**Compute (Phi A800 80 GB):** ~12 h closed-book D × {Hot, 2Wiki}
combined. Hidden-state dump ~1 GB additional storage. Oracle
re-decode not required (eigen_score is closed-book; oracle file already
on disk).

**Decision rule (locked).**
- `eigen_score.{max,mean,first}` residualized AUROC > 0.62 on Hot
  OR > 0.60 on 2Wiki, with paired-bootstrap CI lower bound > 0 vs
  `tokens_generated` → **SeaKR breaks the pattern**; paper reports
  DRAGIN + SeaKR as two length-robust triggers.
- Both fail → **DRAGIN sole survivor across all 22 signals × 2
  datasets**; negative claim sharpens.

### Burden 3 — Cross-architecture replication

**Why.** Methodological contribution is universal; survivor claim is
currently single-arch. Closes critique 1.

**Plan.**
- Run v2 capture pipeline on Gemma-4-26B-A4B-it and
  Qwen3.6-35B-A3B-Instruct at n=200 per dataset, oracle re-decode same n.
- Frozen Option-A protocol per `_MODEL.md §3`.
- Re-run audit (Burden 1's script + Burden 2's eigen_score if landed).

**Compute (single A800):** Gemma D+E ~4–4.5 h; Qwen D+E ~5–5.5 h;
+eigen_score Gemma ~6 h, Qwen ~7 h. **Total: ~10 h baseline / ~22 h
with eigen_score.**

**Pre-registered decision rule (per arch ∈ {Gemma, Qwen}):**

| outcome | meaning |
|---|---|
| `dragin_rind_max.max` residualized AUROC ≥ 0.55 AND paired-margin point estimate vs `tokens_generated` ≥ 0 | **portable**: arch enters headline grid |
| residualized AUROC ∈ [0.50, 0.55] OR paired-margin CI crosses zero | **borderline**: descriptive only |
| residualized AUROC < 0.50 OR paired-margin point estimate < 0 | **negative**: cross-arch failure of DRAGIN |

Final headline depends on the matrix:
- **All 3 portable** → "DRAGIN-RIND is the only length-robust trigger across dense + MoE-sliding + MoE-hybrid-SSM at matched active-parameter scale."
- **2 of 3 portable** → "DRAGIN-RIND is length-robust on {Phi, X}; fails on {Y}; methodological contribution holds across all three."
- **Only Phi portable** → single-arch case study; protocol is the headline.

### Burden 4 — Benchmark-family generalization (MuSiQue, ~11 h)

**Plan.** Add **MuSiQue-Ans** (multi-hop, no distractor); n=200 per
arch per cohort, oracle re-decode. New slice builder
`scripts/04_dataset_musique.py` (shape after `04_dataset_2wiki.py`).

**Decision rule.** If MuSiQue replicates DRAGIN-as-survivor →
benchmark-family critique partially closed. If MuSiQue inverts →
headline narrows to "on multi-hop distractor benchmarks specifically."

**Compute:** Phi ~2 h, Gemma ~4 h, Qwen ~5 h.

### Burden 5 — Statistical hardening (no compute, ~5 min)

Re-run analyzer with **B=10,000** and Bonferroni-corrected α =
0.05 / 10 = 0.005 for the headline paired-margin test (k=10 published
triggers). CIs must clear ±99.5%.

### Burden 6 — Lift-conditional secondary table (no compute, ~30 min code)

`P(oracle helped | trigger fired) − P(oracle helped overall)` at fire
rates {5%, 10%, 20%}. The metric DRAGIN/FLARE/SELFDOUBT were
originally evaluated against. Bootstrap CIs.

### Burden 7 — DioR + Self-RAG citation discipline (writing only)

§2 paragraph distinguishing the parameter-free decode-time-signal
subfamily (audited) from the trained-classifier subfamily (DioR,
Self-RAG, INSIDE-EigenScore-with-trained-threshold). One paragraph + 3
citations.

### Burden 8 — External-method integration (the new triggers; the bulk of new compute)

Three external published triggers to add to the audit pool. From the
code-audit verdicts below:

| method | what to do | re-decode? | extra cost |
|---|---|---|---|
| **FLARE** | extend capture script to dump per-token `max_logprob`; aggregate to `1 − min(prob)` per step → max over steps | yes, all archs | bundled with Burden 3's xarch decode (no extra wall) |
| **SeaKR eigen_score** | port from `SeaKR/SEAKR/reasoner.py` + `vllm_uncertainty/` into our K=3 sampler scaffolding; bump K to 20; capture per-sample mid-stack hidden states | yes (this *is* Burden 2) | see Burden 2 |
| **SelfCheckGPT-NLI** | extend K=3 sampler to dump sample completion texts (currently only aggregate `u_cons` is dumped); offline DeBERTa-v3-large-MNLI scoring | yes, light (~5–10% wall overhead, bundled with FLARE/eigen_score re-decode) | offline NLI ~30 min/dataset on light GPU |
| **MBA-RAG** | per-arch BERT-base classifier retraining on calibration cohort; dataset-conditional reward signal | no (training-side, ~2 h GPU per cell) | 2 h × 3 archs × 2 datasets = ~12 h training |
| **Self-RAG-LLaMA-2-7B** | run released checkpoint on HotpotQA-500 / 2Wiki-500 slices as reference-model row | no (separate model) | ~2 h inference |

---

## STEP 2 — external code audit (kept vs dropped)

Repos cloned to `externalCodeAudits/`. For each, located the
trigger-signal source file, identified the exact computation, checked
whether it is faithfully reproducible on Phi/Gemma/Qwen with our
existing capture infrastructure (`scripts/04f_run_qa_sota.py`,
`scripts/crv/sota_capture.py`).

(`externalCodeAudits/SelfDoubt` is parent's personal scratch; ignored.)

| # | repo | trigger source file | signal | works on 3 archs as-is? | extra capture / training | verdict |
|---|---|---|---|---|---|---|
| 1 | **DRAGIN** | `src/generate.py:481` `AttnWeightRAG.modifier()` | per-token `attn × entropy × (tr−tl)`, fire if max > τ | already done | none | **KEEP — anchor / on-disk survivor** |
| 2 | **FLARE** | `src/templates.py:649` `truncate_at_prob(low)` | fire if any token's `prob ≤ low_prob` | yes if we re-decode | dump per-token `max_logprob` alongside per-token entropy | **KEEP — re-decode (~5 h Phi / ~9 h Gemma / ~11 h Qwen, bundled)** |
| 3 | **SeaKR** | `SEAKR/reasoner.py:101-113` + `vllm_uncertainty/` fork | `eigen_score = log det(Σ + αI)` over K=20 sample hidden states | NO out-of-the-box; vLLM fork is LLaMA-2-tied | port eigen_score into our K=3 sampler; bump K to 20; capture per-sample mid-stack hidden states | **KEEP — Burden 2 (~12 h Phi, ~24 h × MoE)** |
| 4 | **MBA-RAG** | `MAB/train_mab_single.py` | trained BERT-base bandit per question; outputs over 3 arms (no-RAG / single-RAG / multi-RAG) | needs per-arch retraining | train one BERT classifier per (arch × dataset); ~2 h GPU per cell | **KEEP — covers question-router subfamily (with explicit "trained per-arch" caveat)** |
| 5 | **Self-RAG** | `retrieval_lm/run_short_form.py:73-83` | logprob ratio of `[Retrieval]` vs `[No Retrieval]` reflection tokens; fire if > τ (default 0.5) | NO — released checkpoints are LLaMA-2-7B/13B only; reflection-token vocabulary requires fine-tuning | use released selfrag-LLaMA-2-7B as black-box reference model on our slices (separate row) | **KEEP AS REFERENCE ROW — different model, same protocol** |
| 6 | **SelfCheckGPT (NLI primary, Ngram secondary)** | `selfcheckgpt/modeling_selfcheck.py:328` `SelfCheckNLI` (line 295 for Ngram) | `mean_k prob_contradiction(sentence, sample_k)` from DeBERTa-v3-large-MNLI; K=3 sufficient | yes if we re-decode + dump K=3 sample texts | dump K=3 sample completion texts; offline NLI scoring | **KEEP — cheap capture extension, faithful, fills the canonical K-sample-consistency gap** |
| 7 | **DynamicRAG** | `inference.py` (no trigger; LLaMA-3-based reranker model) | post-retrieval reranker — selects which docs to keep, always retrieves first | scope-mismatched: trigger-firing rate is 100% by construction | n/a | **DROP from audit — cite in §2 as orthogonal "post-retrieval reranking subfamily"** |

### Final audit pool (combined with on-disk pool)

5 trigger rows + 1 reference-model row from external integration, plus
the on-disk pool already in `analysis/length_audit_v2.py`:

| row source | rows | notes |
|---|---|---|
| on-disk (length baselines + DRAGIN + ETC + HVR + entropy + CRV exploratory + u_cons exploratory) | 11 | currently committed; +8 from Burden 1 re-application = 19 |
| FLARE | 1 | Burden 8 |
| SeaKR eigen_score (×3 aggregators) | 3 | Burden 2 |
| SelfCheckGPT-NLI (+ Ngram secondary) | 2 | Burden 8 |
| MBA-RAG | 1 | Burden 8 |
| Self-RAG-LLaMA-2-7B reference | 1 (separate model) | Burden 8 |

**Combined headline audit:** ~26–28 distinct trigger-signal rows ×
2 datasets × 3 architectures + 1 reference model. Approximately the
largest length-controlled trigger audit reported in the dynamic-RAG
literature to date.

### Notes from the code audit

- **DRAGIN faithfulness** previously confirmed in `_agent_v2.md` #v2-8
  (max-over-heads attention reduction matches official
  `oneal2000/DRAGIN`; last-full-attention-layer routing on hybrid
  stacks is a documented principled extension).
- **FLARE** uses sentence-level retrieval based on per-token min-prob
  in the upcoming sentence; aggregation to trace-level for our audit
  is `1 − min(prob)` per step → max over steps.
- **SeaKR's `ln_entropy`, `perplexity`, `energy_score`** are secondary
  signals computable from greedy logprobs; the `eigen_score` is the
  load-bearing one and the only reason for the K=20 capture.
- **SelfCheckGPT-NLI** is recommended by the package's own README as
  the primary variant; SelfCheckGPT-MQAG is heavier (Q-A model + answer
  model) and skipped as a redundant cousin.
- **Self-RAG checkpoint** runs through vLLM with `skip_special_tokens`
  control; inference-only cost is low.
- **MBA-RAG training corpus** uses MuSiQue / HotpotQA / 2WikiMHQA dev
  splits for the bandit's reward signal; the same datasets are in our
  paired oracle-decode pipeline so per-arch retraining is data-compatible.

---

## Compute audit (consolidated, single A800 80 GB serial)

| burden | what | compute |
|---|---|---|
| 1 | re-apply audit extensions | 0 |
| 2 | SeaKR eigen_score on Phi | ~12 h |
| 3 | cross-arch baseline + eigen_score | ~22 h |
| 4 | MuSiQue across 3 archs | ~11 h |
| 5 | statistical hardening | 0 |
| 6 | lift-conditional table | 0 |
| 7 | DioR / Self-RAG citation | 0 |
| 8 | external integration (FLARE/SelfCheck bundled with 2+3; MBA training; Self-RAG ref) | ~14 h marginal (12 h MBA training + 2 h Self-RAG inference) |
| **total compute envelope** | | **~59 h** |

One-week compute window or three ~24-h sessions on one A800. All
high-leverage compute items (eigen_score, Gemma xarch, Qwen xarch,
MuSiQue, MBA training) are independent; with two boxes, envelope
halves to ~30 h wall.

**If compute is constrained to a single block, sequencing:**

1. **Highest information-per-compute single experiment:** Burden 2 +
   Burden 8-FLARE + Burden 8-SelfCheck (one Phi re-decode with
   K=20 + sample-text dump + per-token-logprob dump). One run, three
   methods integrated. ~12 h.
2. **Second:** Burden 3 — Gemma alone, n=200, baseline (no eigen_score
   yet). ~4.5 h.
3. **Third:** Burden 3 — Qwen alone, n=200, baseline. ~5.5 h.
4. **Fourth (defer if tight):** Burden 4 — MuSiQue across 3 archs.
5. **Fifth (defer):** Burden 8-MBA training, ~12 h.
6. **Sixth (defer):** Burden 8-Self-RAG-LLaMA-2-7B reference, ~2 h.

The first three together give a **headline-defining single-week
campaign**: 5 audited trigger families × 2 datasets × 2–3 archs.

---

## What `_rq1-plan.md` does NOT commit to

- A specific submission venue or deadline (parent has 6 months;
  this plan does not consume that headroom prematurely).
- A specific paper title (working title: *"A Length-Controlled
  Evaluation Protocol for Dynamic Retrieval Triggers"*; revisit after
  Burden 2 lands).
- A specific aggregator commitment for the headline DRAGIN row
  (Burden 1 will surface whether `max` or `mean` should be canonical
  for OOD reporting).
- A `u_cons` claim of any kind. Stays in the audit table for
  completeness but **never headline**, **never abstract**, **never
  load-bearing**. Exploratory throughout RQ1.
- A SeaKR-eigen_score outcome. Decision rule locked above; outcome
  open until compute lands.

These resolve in `_rq1-agent.md` entries as parent specifies them.

---

## Workflow contract for `_rq1-agent.md`

Experiment-first, not pre-registration-first. Each entry is one of:

- **Planned Experiment N — Setup.** Goal(s) served, model, dataset, n,
  knobs, scripts to invoke, expected wall clock, log paths.
- **Run Experiment N — Results.** Numbers, tables, raw JSON paths,
  anything weird. Both positive and negative results land with the
  same weight.

No success / failure / inconclusive criteria are committed before a
run; results are read on their merits. **Heavy logging at every layer
(per-trace, per-step, per-resample, per-sample) is mandatory** —
post-hoc analysis is always valuable, especially when an experiment
surfaces something the original framing didn't anticipate. The
cheapest re-decode is the one we never have to repeat.

Each experiment must explicitly tag which of the 3 goals it serves:

- **G1 (test hypothesis)** — produces a number that confirms or
  falsifies the survivor claim under residualization.
- **G2 (post-hoc readiness)** — adds a column to the trace schema /
  caches state that makes future analysis cheaper.
- **G3 (domain learning)** — surfaces a fact about a method or
  signal family that informs the discussion / future work.

Experiments serving 0 of the 3 goals do not get run.

Pilots from v1 are referenced by id only (e.g. v1 #20) but never
cited as evidence for RQ1 claims.

### Small-checkpoint commit policy

Per parent's directive (G2: max post-hoc readiness): commit derived
artifacts directly to git when small enough. Tier table:

| size | examples | policy |
|---|---|---|
| ≤ 10 MB | calibration JSONs, fitted thresholds, per-arch OLS coefficients, slice manifests + sha256s, fire-rate calibration tables, smoke validation JSONs, MBA-RAG classifier *head* (BERT base stripped) | **commit directly** to `runs/v2_<exp>/checkpoints/` |
| 10 – 500 MB | full DistilBERT bandit weights, per-trace bootstrap arrays, full pilot trace dumps with hidden-state capture | **commit by default**; flag in PR if size is unusual |
| > 500 MB | DeBERTa-v3-large-MNLI, Self-RAG-7B, base model weights | **HF download from pinned SHA only**; never commit |

Per-trace JSONL files inherit the size of their cohort: P0 pilot
(~27 MB total across all 3 archs × 4 phases) commits cleanly; the
n=500 v2 traces (~150 MB each) sit at the edge — kept on disk if
present from prior runs, but any new v2-scale captures should be
weighed against the tier table before committing.

Submodules (`externalCodeAudits/*`) pin **code** at exact SHAs the
audit was done against; they are not data and are not subject to
this policy.
