# _PLAN.md

> Build the smallest working version first, then add one capability per step.
> Targets: Mac (Apple Silicon, Metal) **and** Linux (NVIDIA, CUDA). Same code path.
> Each step ends with something we can **see** in the terminal.

---

## Guiding rules

- One concept per step. Split if it grows.
- Every step ends with a visible artifact: console output, a file, or a number.
- **Reproducibility is non-negotiable.** Pinned versions, scripted installs, recorded system info on every run.
- **Memory is tracked, not feared.** Mac has 32 GB; the goal is to *report* footprint at every phase, not to hit a ceiling.
- After each step, update `_AGENT.md` with what worked, what broke, what we changed.

---

## Repo layout (target)

```
recollection.local/
├── _PLAN.md                # this file (gitignored, local planning)
├── _AGENT.md               # experimental log (gitignored)
├── README.md
├── environment.yml         # conda env: python, cmake, pinned pip deps
├── scripts/
│   ├── setup.sh            # OS-detecting bootstrap (mac/metal vs linux/cuda)
│   ├── download_models.sh  # huggingface-cli, GGUF only
│   ├── sysinfo.py          # prints OS, RAM, GPU, backend, versions
│   └── 00_smoke.py ...     # phase scripts, numbered
├── models/                 # GGUF files (gitignored)
├── data/                   # corpora (gitignored)
└── runs/                   # JSONL logs per run (gitignored)
```

---

## Reproducibility contract (apply to every phase)

1. **Install path is scripted.** `scripts/setup.sh` detects Mac vs Linux+CUDA, builds the conda env from `environment.yml`, then installs `llama-cpp-python` with the correct `CMAKE_ARGS`. No manual installs.
2. **Versions are pinned.** `environment.yml` pins Python, `cmake`, and all pip deps in one file. `llama-cpp-python` version is pinned inside `setup.sh` (because its build flags differ per platform).
3. **Models are scripted.** `scripts/download_models.sh` uses `huggingface-cli` with explicit repo + filename. Never rely on browser downloads.
4. **Every run logs sysinfo.** `scripts/sysinfo.py` records OS, kernel, CPU, total/free RAM, GPU/Metal status, backend in use, Python and key package versions. Phase scripts call it at startup and write results to `runs/<timestamp>.json`.
5. **Every run logs memory.** Before model load, after model load, after first generation. Use `psutil` for RSS and (on Mac) `Llama` metadata for GGUF size on disk.

---

## Phase 0 — Environment + reproducibility scaffold

**Goal:** A fresh machine (Mac or NVIDIA Linux) can clone, run one script, and reach a working `llama-cpp` import.

- [x] **0.1** `scripts/setup.sh`: detects `uname -s` / `uname -m` / `nvidia-smi`; creates or updates the `recollection` conda env from `environment.yml`; installs `llama-cpp-python` with `-DGGML_METAL=on` on Apple Silicon, `-DGGML_CUDA=on` on Linux+NVIDIA, plain CPU otherwise.
- [x] **0.2** `environment.yml`: `python=3.11`, `cmake>=3.27`, `pkg-config`, plus pinned pip deps (`huggingface_hub`, `psutil`, `numpy`, `pandas`, `sentence-transformers`, `lancedb`, `tantivy`).
- [x] **0.3** `scripts/sysinfo.py`: returns + prints OS, arch, CPU, total/avail RAM, GPU (Metal chip via `sysctl` on Mac, `nvidia-smi` on Linux), Python version, package versions, current process RSS.
- [x] **0.4** `scripts/download_models.sh` using `huggingface-cli download`. Default = phase-0 model only. `bash scripts/download_models.sh all` adds the 7B. 
- [x] **0.5** `.gitignore` excludes `.venv/`, `models/`, `data/`, `runs/`, `__pycache__/`, `.DS_Store`.
- [x] **0.6** Ran `bash scripts/setup.sh`, `bash scripts/download_models.sh`, `python scripts/00_smoke.py`. Phi-3.5-mini answered at **30.9 tok/s**, 3.78 GB RSS, on Apple M4 / Metal. Numbers logged in `_AGENT.md`.

Phase 0 done. CUDA path stays untested for now but the install branch exists in `setup.sh`.

---

## Phase 1 — Single-model loop (CLI we can poke)

- [x] **1.1 + 1.2 + 1.3** `scripts/01_repl.py`: interactive loop with `--model` / `--n_ctx` / `--n_gpu_layers` / `--max_tokens` / `--temperature` / `--no-sysinfo` flags; per-turn `tokens/sec` and RSS delta; `:sys` command re-prints sysinfo. Smoke-tested with two piped prompts: ~30 tok/s, KV cache delta < 5 MB at `n_ctx=4096`.

---

## Phase 2 — Confidence signal (the gate's input)

- [x] **2.1** Schema recorded in `_AGENT.md`. Two findings: `logits_all=True` required at construction; `token_logprobs` is what we want (not `top_logprobs`). Calibration check on `"The capital of France is"` shows trailing-format tokens drag the mean down — gate cannot be a naive average.
- [x] **2.2** `scripts/confidence.py` — `mean_prob`, `min_prob`, `first_content_mean`, `summarize`. All three aggregators computed per turn.
- [x] **2.3** `01_repl.py` prints `mean / min / first_content_mean` per turn; `--show-tokens` dumps the per-token `(text, prob)` list. First read: `first_content_mean` separates easy vs hard cleanly (0.914 vs 0.720); `mean`/`min` don't.
- [x] **2.4** `scripts/02_calibration.py` ran 5 easy + 5 hard prompts in **raw completion mode**. Inverted `mean` / `min` and a +0.254 gap on `first_content_mean`. Records in `runs/phase2_calibration.jsonl`.
- [x] **2.5** Chat-template oversight caught and fixed. `scripts/02b_chat_calibration.py` re-ran the same prompts via `create_chat_completion`. **`mean` and `min` un-inverted** (gaps +0.030 and +0.119); `first_content_mean` saturates near 1.0 on easy in chat mode. New best single signal: **`min`**. Confidently-wrong H5 (cubic eqn) still slips through; reaffirms Phase 5 + Phase 6 are mandatory. Records in `runs/phase2_chat_calibration.jsonl`. `01_repl.py` switched to chat mode.

---

## Phase 3 — Two models, one gate

- [~] **3.0** Generator-swap calibration. **Partial on Mac (8/10 prompts), resumes on CUDA VM.** Download done (4.4 GB). Inspection done (`scripts/03_deepseek_inspect.py`): model emits empty `<think></think>` for easy prompts, blows past 1024 tokens for some hard prompts. Strategy locked: chat completion + score post-`</think>` only + treat `reasoning_complete=False` as a separate fire trigger. Running aggregate (4 easy, 2 hard): `min` gap **+0.224** (best so far across all configs). H4, H5 still to run on CUDA. See `_AGENT.md` for table.
- [ ] **3.1** Memory baseline: run `sysinfo` → load only generator (`DeepSeek-R1-Distill-Qwen-7B` GGUF Q4_K_M) → log RSS + Metal/VRAM usage if available.
- [ ] **3.2** Same for discriminator alone (Phi-4-mini Q4_K_M).
- [ ] **3.3** Both resident simultaneously. **Goal: a number, not a pass/fail.** On 32 GB Mac it should fit; we still report exactly what it costs.
- [ ] **3.4** `gated_inference(prompt, tau)`: generator first, if `avg_conf < tau` route `(prompt, generator_output)` to discriminator with a "verify or fix" instruction.
- [ ] **3.5** Print path taken: `[GENERATOR ONLY]` or `[GATE TRIGGERED -> DISCRIMINATOR]`.
- [ ] **3.6** Sweep `tau` over `{0.1, 0.5, 0.85, 0.95}` on the 10 prompts from 2.4. Record gate-trigger rate + memory + latency. Resolve here whether the paper's `theta = 0.1` is on raw probability or on something else.

---

## Phase 4 — Tiny retrieval (toy corpus)

- [ ] **4.1** `data/toy/` with 5 short `.txt` files containing distinct facts.
- [ ] **4.2** `scripts/04_ingest.py`: embed with `all-MiniLM-L6-v2`, write to LanceDB.
- [ ] **4.3** `scripts/04_search.py`: query → top-3 chunks with scores.
- [ ] **4.4** Wire retrieval into the generator prompt template. **Visible:** answers grounded in toy facts.

---

## Phase 5 — Confidence-gated RAG (paper's core, minimal version)

- [ ] **5.1** Pipeline: `query → retrieve → generate → measure conf → if low, discriminate (with same context)`.
- [ ] **5.2** Log every step to `runs/<timestamp>.jsonl`: query, retrieved ids, conf, gate decision, final answer, RSS at each stage.
- [ ] **5.3** Manual eval on 10 hand-written queries against the toy corpus.

---

## Phase 6 — Anti-memorization audit

- [ ] **6.1** Extract noun phrases / numeric tokens from the answer. If none appear in retrieved chunks, log `[MEMORIZED]`.
- [ ] **6.2** Run toy queries, count `[MEMORIZED]`.
- [ ] **6.3** Insert a deliberately-wrong fact into the toy corpus (e.g., "the capital of France is Berlin"). If the model still answers "Paris", `[MEMORIZED]` must fire. Sanity check.

---

## Phase 7 — Scale up corpus

- [ ] **7.1** Replace toy with 5 real papers from the bibliography.
- [ ] **7.2** Fixed 512-token chunks, 64-token overlap. Re-ingest.
- [ ] **7.3** Re-run Phase 5 on 5 single-chunk questions.
- [ ] **7.4** Then 5 multi-hop questions. Note where it breaks.

---

## Resume on CUDA — first actions in the VM

1. `bash scripts/setup.sh` (auto-detects `nvidia-smi`, pulls CUDA wheel for `llama-cpp-python==0.3.22`).
2. `bash scripts/download_models.sh all`.
3. `python scripts/sysinfo.py` — record GPU + VRAM baseline as a fresh "Phase 0 (CUDA)" entry in `_AGENT.md`.
4. `python scripts/03_deepseek_calibration.py` — finish all 10 prompts cleanly. Compare to Phi-3.5 chat-mode and partial Mac numbers in `_AGENT.md`.
5. Then proceed with Phase 3.1 → 3.6 in this file.

If `cu124` wheel is wrong for your driver: `LCPP_CUDA=cu121 bash scripts/setup.sh` (or whatever index abetlen exposes — check `pip index versions llama-cpp-python --index-url https://abetlen.github.io/llama-cpp-python/whl/cu121/`).

---

## Phase 8+ — Explicitly **not** doing yet

- MCTS rollouts (Phase 3 gate is the lightweight stand-in).
- PolarQuant / TurboQuant / KV-cache quantization (GGUF Q4_K_M is the prototype's quantization story).
- LongBench v2 / NeedleChain (only after Phase 7 is stable).
- CRAG baseline (Track A) — separate branch.
- 32k / 110k context windows (stay at 4096 until Phase 5 works).

---

## Definition of done for the prototype

One command (`scripts/setup.sh && scripts/download_models.sh && python scripts/05_pipeline.py "<question>"`) that on **both** Mac and Linux+CUDA prints:

1. System + memory baseline.
2. Retrieved chunks.
3. Generator answer + confidence.
4. Whether the gate fired and what the discriminator said.
5. Whether the answer was flagged `[MEMORIZED]`.
6. RSS deltas at each stage.

Everything else (MCTS, PolarQuant, full benchmarks) layers on after.
