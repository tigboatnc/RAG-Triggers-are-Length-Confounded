# _AGENT.md

> Experimental log + status. Read top-down: status block first, then chronological entries.
> Update after every step. One file, one source of truth.

---

## ⏸ HANDOFF (2026-05-09 — moving to CUDA VM)

Mac/Metal calibration on DeepSeek was throughput-limited (6–8 tok/s; 10-prompt calibration projects to >20 min wall time at 1024 max_tokens). Stopping here and resuming on CUDA.

### Resume in 3 commands inside the VM

```bash
git clone <this repo>
cd recollection.local
bash scripts/setup.sh                   # detects nvidia-smi, picks CUDA wheel automatically
bash scripts/download_models.sh all     # Phi-3.5-mini + DeepSeek-R1-Distill-Qwen-7B GGUF
conda run -n recollection --no-capture-output python -u scripts/03_deepseek_calibration.py
```

`setup.sh` auto-detects platform via `uname -s` + `nvidia-smi` and pulls the prebuilt CUDA wheel from `https://abetlen.github.io/llama-cpp-python/whl/${LCPP_CUDA:-cu124}`. Override CUDA target if needed: `LCPP_CUDA=cu121 bash scripts/setup.sh`.

### What's set up and reproducible

- `environment.yml`: `python=3.11`, pinned pip deps (`huggingface_hub==0.27.1`, `psutil==6.1.1`, `numpy==2.4.4`).
- `scripts/setup.sh`: pins `llama-cpp-python==0.3.22` (must be ≥0.3.11 for DeepSeek-R1 pre-tokenizer).
- `scripts/setup.sh --phase4` adds `pandas`, `sentence-transformers`, `lancedb`, `tantivy` for Phase 4 retrieval.
- `scripts/download_models.sh` (no arg) = Phi-3.5-mini only; `all` = both models.
- All scripts use the prebuilt wheel path (no source compile, no `cmake` needed).

### What's already done

- Phases 0, 1, 2.1–2.5 complete on Mac. Findings transfer to CUDA.
- Phase 3.0(a) DeepSeek inspection complete: model loads, `<think>...</think>` schema confirmed, post-think scoring strategy validated.
- Phase 3.0(b) calibration **8 of 10 prompts done** before stop. Partial table in entry below. Two prompts still to run on CUDA: H4 (made-up author summary), H5 (cubic equation).

### What we know about the gate going in

Using **chat completion** (`create_chat_completion`, NOT raw `__call__`) and scoring **post-`</think>` tokens only**:

| signal | role | provisional threshold |
| --- | --- | --- |
| `min` (post-think) | primary gate input | fire if < 0.40 |
| `reasoning_complete` | secondary signal — fail-to-close = high difficulty | fire if False |
| `first_content_mean` | diagnostic only (saturates near 1.0 on easy) | not used in gate |
| `mean` | diagnostic only (saturates) | not used in gate |

H5-style "confidently wrong" answers (Phi-3.5 hallucinated cubic roots at 0.886 mean) are **not catchable by confidence alone**. Mandatory downstream defenses: Phase 5 retrieval grounding + Phase 6 anti-memorization audit.

### CUDA-specific things to verify on first run

1. `nvidia-smi` is detected → setup picks `cuda` backend (look for `[ok] OS=Linux arch=x86_64 backend=cuda`).
2. `python -c "import llama_cpp; ..."` reports the same `0.3.22` and the model offloads to GPU (`n_gpu_layers=-1`).
3. **Re-run `scripts/sysinfo.py`** to capture the new GPU/VRAM baseline. Add the numbers to this file under a fresh "Phase 0 (CUDA)" header.
4. Throughput should jump from 6–8 tok/s (M4 Metal) to 30–80 tok/s on a modern NVIDIA card. Update the "watch" item in the Status block accordingly.
5. `LLAMA_CPP_VERSION=0.3.22` should have a `cu124` wheel; if your driver requires `cu121`, set `LCPP_CUDA=cu121` before running `setup.sh`.

### Files to look at first when resuming

| file | what it does |
| --- | --- |
| `_PLAN.md` | full phase plan with state per step |
| `_AGENT.md` | this file — log + decisions |
| `scripts/setup.sh` | bootstrap (idempotent) |
| `scripts/sysinfo.py` | baseline capture |
| `scripts/03_deepseek_inspect.py` | sanity check on DeepSeek |
| `scripts/03_deepseek_calibration.py` | the calibration we want to finish |
| `scripts/confidence.py` | three aggregators; handles raw + chat schemas |
| `scripts/01_repl.py` | chat-mode REPL with confidence per turn |
| `runs/phase2_chat_calibration.jsonl` | Phi-3.5 baseline numbers |

---

## Status (last updated 2026-05-09, mid Phase 3.0(b))

### Progress

| phase | step | state | artifact |
| --- | --- | --- | --- |
| 0 | env + reproducibility scaffold | done | `scripts/setup.sh`, `environment.yml` |
| 0 | sysinfo + memory tracking | done | `scripts/sysinfo.py` |
| 0 | model download | done | `scripts/download_models.sh` (Phi-3.5-mini Q4_K_M, 2.2 GB) |
| 0 | smoke test | done | `scripts/00_smoke.py` (30.9 tok/s, 3.78 GB RSS) |
| 1 | REPL + per-turn timing/RSS | done | `scripts/01_repl.py` |
| 2.1 | logprobs schema | done | `scripts/02_inspect_logprobs.py`, `runs/phase2_logprobs_schema.json` |
| 2.2 | confidence aggregators | done | `scripts/confidence.py` |
| 2.3 | confidence in REPL | done | `01_repl.py --show-tokens` |
| 2.4 | calibration (raw completion) | done | `runs/phase2_calibration.jsonl` |
| 2.5b | calibration (chat completion) | done | `runs/phase2_chat_calibration.jsonl` |
| 3.0 (a) | DeepSeek download + load + inspection | done (Mac) | `scripts/03_deepseek_inspect.py` |
| 3.0 (b) | DeepSeek 10-prompt chat-mode calibration | **partial: 8/10 done; H4, H5 pending** | `scripts/03_deepseek_calibration.py` |
| — | resume on CUDA VM | **next** | re-run setup + downloads + 03_deepseek_calibration |

### Current state

- **Generator**: Phi-3.5-mini-instruct Q4_K_M (placeholder; will be replaced by DeepSeek-R1-Distill-Qwen-7B at Phase 3.0).
- **Discriminator**: not yet pinned (Phi-4-mini-instruct GGUF — slot reserved in `download_models.sh`).
- **Call mode**: `create_chat_completion` (not raw `__call__`) — see Phase 2.5 entry.
- **Best confidence aggregator (chat mode, Phi-3.5)**: `min` (gap +0.119, no saturation). `mean` saturates near 1.0 on easy prompts; `first_content_mean` saturates worse. To re-test on DeepSeek.
- **Provisional gate**: fire if `min < 0.40` on the answer. Not committed; awaiting Phase 3.0 numbers.
- **Hardware**: Apple M4, 32 GB unified memory. Metal backend, prebuilt wheel.

### Open questions (parked)

| id | question | resolve in |
| --- | --- | --- |
| Q1 | Paper's `theta = 0.1` — what statistic? | 3.6 sweep |
| Q2 | RSS for both 7B + ~3.8B GGUFs resident at once | 3.3 |
| Q3 | Does DeepSeek's `<think>...</think>` block need separate scoring? | **resolved 3.0(a): yes — score post-`</think>` only; for easy prompts the think block is empty, for hard prompts it eats the entire budget** |
| Q4 | Does retrieval context (Phase 5) shift the confidence distribution? | re-run calibration after 5.x |

### Things to watch

- `logits_all=True` costs ~30% throughput (30 → ~22 tok/s on Phi-3.5-mini Q4_K_M, M4). Acceptable for prototype; revisit if Phase 5 latency hurts.
- `n_ctx=4096` cap stays until Phase 5 works. Going to 32k/110k before that explodes KV cache for no reason.
- Anti-memorization audit (Phase 6) needs a *deliberately wrong* fact in the toy corpus to confirm the flag actually fires; otherwise "no flag" and "broken flag" look the same.
- `setup.sh` pins `llama-cpp-python==0.3.22` (bumped from 0.3.5 in 3.0; older wheel rejected DeepSeek-R1's pre-tokenizer). To bump again: change `LLAMA_CPP_VERSION`, re-run.
- `environment.yml`'s numpy pin was a false promise: pip resolves `llama-cpp-python` after, which bumps numpy. Aligned to `numpy==2.4.4` (what 0.3.22 wants).

---

## 2026-05-09 — kickoff

- Read `draft_original.md`, `draft.md`, `README.md`, `extras/writingGuideline.md`.
- Wrote `_PLAN.md` with phases 0–7 + an explicit "not yet" list.
- Decisions:
  - Backend: `llama-cpp-python` for Mac/Metal **and** Linux/CUDA via the same code path. vLLM dropped (Mac-incompatible).
  - Quantization for prototype: GGUF `Q4_K_M`. PolarQuant / TurboQuant deferred until the gated pipeline runs end-to-end on a toy corpus.
  - Smallest model first (Phi-3.5-mini class) before the 7B. Faster iteration.
  - Memory is *reported*, not minimized. 32 GB Mac is comfortable; we still log RSS at every stage so we have a baseline before adding KV-cache tricks.

## 2026-05-09 — Phase 0 (env + reproducibility, model download, smoke)

Switched env tooling **venv → conda** mid-phase. Reasons: pins Python 3.11 itself (system was 3.9), no need for system `cmake`, single reproducibility artifact (`environment.yml`).

Two slow paths killed:

1. Source-compiling `llama-cpp-python` (5–10 min) → switched to **prebuilt Metal wheel** (`--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal`). Drops to ~45 s.
2. Eagerly installing `sentence-transformers` + `lancedb` (pulls torch, ~2 GB) → moved to `--phase4` flag on `setup.sh`. Phase-0 deps are just `huggingface_hub`, `psutil`, `numpy`.

`requirements.txt` deleted; `environment.yml` is the single source.

End-of-phase numbers (Apple M4, 32 GB):

| metric | value |
| --- | --- |
| env create + lcpp install | ~75 s |
| model download (Phi-3.5-mini Q4_K_M) | 26 s, 2.2 GB |
| model load (`n_ctx=4096`, `n_gpu_layers=-1`) | 10.98 s, +3.73 GB RSS |
| generation (smoke) | 96 tok in 3.11 s → **30.9 tok/s** |
| final RSS | 3.78 GB / 32 GB |

Sample output (`scripts/00_smoke.py`, prompt: "In one sentence, what is retrieval-augmented generation?"):

```
Retrieval-augmented generation (RAG) is a machine learning approach
that combines a large-scale retrieval system with a generative model
to produce more contextually relevant and informative text outputs by
accessing relevant documents or data sources during the generation
process.
```

(Already drifts into a `## Answer` block — first symptom of base-completion mode. We didn't recognize it yet.)

Benign warnings noted: `bf16` Metal kernels skipped on M4 (Q4_K_M doesn't use them); `n_ctx_per_seq (4096) < n_ctx_train (131072)` is the deliberate cap.

## 2026-05-09 — Phase 1 (REPL)

`scripts/01_repl.py` — one script for 1.1 + 1.2 + 1.3. Flags: `--model`, `--n_ctx`, `--n_gpu_layers`, `--max_tokens`, `--temperature`, `--no-sysinfo`. Inside the loop: `exit` / `:q` / Ctrl-D quit; `:sys` re-prints sysinfo.

Smoke (two piped prompts, `max_tokens=40`):

| turn | tok/s | rss delta |
| --- | --- | --- |
| 0 | 29.8 | +5 MB |
| 1 | 31.6 | +1 MB |

Cold load on a fresh process was 10.98 s (Phase 0); on the second process **0.20 s** because macOS page-cached the GGUF. Load times across runs aren't comparable unless we drop caches. Not chasing.

`.gitignore` extended: `build/`, `dist/`, `*.egg-info/`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `.ipynb_checkpoints`, `.env`, `.env.*`, `.vscode/`, `.idea/`, `*.swp`.

## 2026-05-09 — Phase 2.1 (logprobs schema)

`scripts/02_inspect_logprobs.py` ran one prompt with `logprobs=5` and dumped the actual response shape. Two real nuances surfaced.

### Nuance A — `logits_all=True` mandatory at construction

First attempt failed:

> `ValueError: logprobs is not supported for models created with logits_all=False`

In `llama-cpp-python` 0.3.5 the `Llama` object must be constructed with `logits_all=True` for `logprobs=N` at call time to populate `choices[0]['logprobs']`. The original draft snippet was right that this flag is needed, just for the wrong reason. **Cost:** full-vocab logits per generated token, not just the chosen one. ~30% throughput tax.

### Nuance B — actual schema (raw completion)

```text
response                                    # dict
├── id, object, created, model              # metadata
├── choices: [{
│     text, index, finish_reason,
│     logprobs: {
│       tokens:         list[str],
│       text_offset:    list[int],
│       token_logprobs: list[np.float32],     # logprob of CHOSEN token (this is what we want)
│       top_logprobs:   list[dict[str,float]] # top-N alternatives per position
│     }
│   }]
└── usage: {prompt_tokens, completion_tokens, total_tokens}
```

Two gotchas:

1. Values are `numpy.float32`, not Python `float`. JSON dump needs `default=float`.
2. `top_logprobs` is `list[dict[str, float]]` (token → logprob). **Not** OpenAI SDK shape.

### Calibration check (single prompt)

`"The capital of France is"`, `temperature=0.0`, `max_tokens=6`:

| pos | token | logprob | prob |
| --- | --- | --- | --- |
| 0 | ` Paris` | -0.090 | **0.914** |
| 1 | `.` | -0.251 | 0.778 |
| 2 | `\n` | -0.681 | 0.506 |
| 3 | `\n` | -0.792 | 0.453 |
| 4 | `Par` | -2.032 | 0.131 |
| 5 | `is` | -0.001 | 0.999 |

The actual answer (` Paris`) sits at 0.91. Trailing newlines + BPE re-spelling drag the mean to 0.63. We concluded "the gate cannot be a naive average over all generated tokens." (Correct conclusion, but for a deeper reason than we realized — see Phase 2.5.)

Artifact: `runs/phase2_logprobs_schema.json`.

## 2026-05-09 — Phase 2.2 / 2.3 (aggregators wired in)

`scripts/confidence.py`:

- `mean_prob` — average over all generated tokens.
- `min_prob` — lowest single-token probability.
- `first_content_mean` — mean over the first contiguous run of non-format tokens.
- `summarize(response)` returns all three + per-token `(text, prob)`.

`01_repl.py` updated: `Llama(..., logits_all=True)`, `logprobs=5`, prints `mean / min / first_content_mean` per turn, `--show-tokens` flag.

Throughput tax confirmed: 30.9 → ~22 tok/s with `logits_all=True`.

## 2026-05-09 — Phase 2.4 (calibration on raw completion — **WRONG MODE**)

`scripts/02_calibration.py` ran 5 easy + 5 hard prompts, deterministic, `max_tokens=40`. Records: `runs/phase2_calibration.jsonl`.

### Per-bucket separation (raw completion mode)

| aggregator | easy (lo / mid / hi) | hard (lo / mid / hi) | easy − hard gap |
| --- | --- | --- | --- |
| `mean` | 0.694 / 0.792 / 0.887 | 0.728 / 0.810 / 0.916 | **−0.018** (inverted) |
| `min` | 0.104 / 0.120 / 0.144 | 0.136 / 0.217 / 0.356 | **−0.098** (inverted) |
| `first_content_mean` | 0.914 / 0.951 / 0.985 | 0.414 / 0.697 / 0.886 | **+0.254** |

`mean` and `min` rated hard prompts **higher** than easy. We blamed format-pivot tokens. Correct mechanism, but we didn't yet realize *why* the model produced so many of them.

Sample (E1, raw mode): `'Paris. /  / Paris is the capital city of France, which is a country in Europe...'` — already drifting into a generic encyclopedia stub. E3 raw mode: `'William Shakespeare. /  /  / ## Instruction 2 (More Difficult) /  / **Instruction:** /  / Identify the author of the novel '1984...'` — straight-up training-data drift, the model is roleplaying as a textbook.

We provisionally crowned `first_content_mean` and tentatively chose `tau = 0.85`. **Both decisions were premature.**

## 2026-05-09 — Phase 2.5 (chat completion — **the real fix**)

### What I missed

I designed Phase 2's calibration around `llm(prompt)` — that's `Llama.__call__`, which is **base text completion**. Phi-3.5-mini-instruct is an instruction-tuned model; in base mode it autocompletes text rather than answering. The "Question:/Answer:" loops, the `## Instruction 2` headers, the encyclopedia drift — all symptoms of the wrong call mode.

The fix is `Llama.create_chat_completion(messages=[{"role": "user", ...}])`, which applies the model's chat template (Phi-3.5: `<|user|>...<|end|>\n<|assistant|>\n`). I should have caught this in Phase 0. The lesson goes into the "things to watch" list at the top.

### Test (`scripts/02b_chat_calibration.py`, identical 10 prompts, chat mode)

```
[E1] mean=0.882  min=0.262  first=1.000  | 40 tok @ 23.5 tok/s
     -> 'The capital of France is Paris. It is not only the largest city in
         the country but also serves as its political, cultura...'

[E3] mean=0.927  min=0.411  first=0.992  | 40 tok @ 23.2 tok/s
     -> "The play 'Hamlet' was written by William Shakespeare, an English
         playwright and poet who is widely regarded as one of th..."

[H2] mean=0.858  min=0.381  first=0.935  | 40 tok @ 23.8 tok/s
     -> 'Determining the exact population of Bhutan in 1992 requires access
         to historical demographic data. According to the Worl...'
```

Compare to raw mode E3: `"William Shakespeare. / ## Instruction 2 (More Difficult) / **Instruction:** / Identify the author of '1984..."`. Same model, same prompt, same `temperature=0.0`. Different *call*. The output is no longer drift-ridden; the model is actually responding. H2 even produces an honest hedge instead of a hallucinated "1,100,000".

### Per-bucket separation (chat mode)

| aggregator | easy (lo / mid / hi) | hard (lo / mid / hi) | easy − hard gap |
| --- | --- | --- | --- |
| `mean` | 0.882 / 0.938 / 0.975 | 0.858 / 0.909 / 0.967 | **+0.030** |
| `min` | 0.262 / 0.493 / 0.743 | 0.314 / 0.374 / 0.452 | **+0.119** |
| `first_content_mean` | 0.992 / 0.995 / 1.000 | 0.838 / 0.939 / 0.986 | **+0.056** |

Comparison to raw mode (gaps only):

| aggregator | raw mode gap | chat mode gap |
| --- | --- | --- |
| `mean` | −0.018 | **+0.030** |
| `min` | −0.098 | **+0.119** |
| `first_content_mean` | +0.254 | +0.056 |

Three structural changes when switching modes:

1. **`mean` and `min` flipped from inverted to correctly oriented.** The "inversion" was an artifact of base-completion, not a property of the model or the metrics.
2. **`min` is now the most useful single signal** — gap +0.119 and no saturation (easy spans 0.26–0.74). `mean` and `first_content_mean` saturate at ~1.0 on easy prompts, losing high-end discrimination.
3. **The model behaved better, not just looked better.** H2 went from confidently hallucinating a number to properly hedging. Chat mode surfaces the trained refusal/uncertainty behaviors that base completion suppresses.

### What didn't change

H5 (cubic equation) is still confidently wrong at `mean=0.967`, `first=0.981`. The "confident-wrong" failure mode is independent of completion mode. Phase 5 (retrieval grounding) and Phase 6 (anti-memorization) remain mandatory.

### Decisions

- `01_repl.py` → switched to `create_chat_completion`.
- 2.5a (stop tokens) cancelled — chat mode dominates that fix.
- `first_content_mean` heuristic stays in `confidence.py` for diagnostics, but loses primacy. Provisional gate: `min < 0.40`.
- Re-calibration on DeepSeek-R1-Distill-Qwen-7B is mandatory at Phase 3.0 — the saturation behavior, threshold, and best aggregator are model-specific.

Artifact: `runs/phase2_chat_calibration.jsonl`.

## 2026-05-09 — Phase 3.0(a): DeepSeek download + inspection

Downloaded `bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF :: Q4_K_M.gguf` (4.4 GB, 46 s).

### Setup-script drift caught & fixed

First load failed: `error loading model vocabulary: unknown pre-tokenizer type: 'deepseek-r1-qwen'`. The pinned `llama-cpp-python==0.3.5` predates upstream llama.cpp's DeepSeek-R1 support. Bumped to `0.3.22` (latest Metal wheel) in `setup.sh`. Also tightened `environment.yml`'s `numpy` pin to `2.4.4` (what 0.3.22 actually installs) so the file stops lying about reproducibility.

### Load + sample outputs

Cold load: 9.63 s, RSS 4.65 GB (vs Phi-3.5-mini 3.78 GB). After one 512-tok hard generation: RSS 5.99 GB.

**E1 — easy question, model self-skips reasoning:**

```
[E1] What is the capital of France?
[gen] 11 tok in 1.83s = 6.0 tok/s

--- raw output ---
<think>

</think>

The capital of France is Paris.

[whole] mean=0.981  min=0.801  first_content_mean=1.000  (11 tok)
[post ] mean=0.999  min=0.997                            (8 tok after </think>)
```

The think block is empty — DeepSeek-R1-Distill has a built-in early-exit. Compare Phi-3.5's chat-mode E1 (`min=0.262`): on easy facts DeepSeek is dramatically more confident.

**H5 — hard math, model never closes the think block:**

```
[H5] Solve for x: x^3 + 2x - 5 = 0. Round to two decimal places.
[gen] 512 tok in 63.29s = 8.1 tok/s

--- raw output (truncated, 1598 chars) ---
<think>
Okay, so I need to solve x^3 + 2x - 5 = 0. ...
First, I'll try the Rational Root Theorem...
Testing x=1: 1+2-5 = -2. Not zero.
Testing x=-1: -1-2-5 = -8. Not zero.
...
At x=1, f(1) = -2.
At x=2, f(2) = 7.
So between 1 and 2 ...
Let me check x=1.       <-- TRUNCATED HERE, no </think> ever emitted

[whole] mean=0.909  min=0.310                            (512 tok)
[post ] (no </think> found — model didn't close the block)
```

The model is reasoning correctly (the answer should be ≈1.33; it had bracketed [1, 2] and was about to bisect) but ran out of tokens. It **didn't hallucinate "x ≈ -1.71, 1.34, 2.37" the way Phi-3.5 did** — that's because Phi-3.5 was forced to commit to an answer in 40 tokens, while DeepSeek used those tokens to verify intermediate steps. **Reasoning models trade tokens for honesty.**

### Implications confirmed

1. **Score post-`</think>` only.** For E1 we get `min=0.997`; for H5 we get nothing useful from the whole-block aggregate (`min=0.310` is reasoning-process noise, not answer uncertainty).
2. **Calibration `max_tokens` needs to scale.** 512 isn't enough for hard prompts. Going to 1024 for the calibration run. Some hard prompts may still not finish; we'll record that as its own signal.
3. **Throughput is ~3× slower than Phi-3.5.** Calibration of 10 prompts at 6–8 tok/s × ~1024 tokens worst case is ~20 min. Acceptable one-time cost.
4. **"Reasoning incomplete" is a confidence signal.** If the model never reaches `</think>`, that's stronger evidence of difficulty than any logprob aggregate.

Artifact: full traces printed to stdout; numbers above are the record.

## 2026-05-09 — Phase 3.0(b): DeepSeek calibration, partial (stopped before CUDA move)

Ran `scripts/03_deepseek_calibration.py` on Mac/Metal. Stopped after 8 of 10 prompts because hard prompts at 6–8 tok/s × 1024 max_tokens project to >20 min wall time. Decision: move to CUDA, finish there.

### Partial calibration table (8 prompts; post-`</think>` scoring)

| tag | bucket | reasoning_complete | mean | min | first | wall (s) | answer (truncated) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | easy | ✓ | 0.999 | **0.997** | 0.999 | 1.95 | "The capital of France is Paris." |
| E2 | easy | ✓ | 0.911 | 0.503 | 0.994 | 96.53 | "Water boils at 100 degrees Celsius at sea level..." |
| E3 | easy | ✗ | – | – | – | 141.97 | overthought "Who wrote Hamlet?", never closed `</think>` |
| E4 | easy | ✓ | 0.851 | 0.317 | 0.990 | 69.16 | "The chemical symbol for gold is Au. Aurum..." |
| E5 | easy | ✓ | 0.948 | 0.344 | 0.944 | 90.69 | "There are seven continents..." |
| H1 | hard | ✗ | – | – | – | 141.20 | train-meeting math, never closed `</think>` |
| H2 | hard | ✓ | 0.869 | 0.238 | 0.842 | 6.85 | "approximately 600,000" (close to truth ≈1.7M, hedged) |
| H3 | hard | ✓ | 0.975 | 0.393 | 0.853 | 75.80 | listing primes 100–110 |
| H4 | hard | (not run) | – | – | – | – | run on CUDA |
| H5 | hard | (not run) | – | – | – | – | run on CUDA |

### Per-bucket separation (6 completed, 4 easy + 2 hard)

| aggregator | easy mid | hard mid | easy − hard gap |
| --- | --- | --- | --- |
| `mean` | 0.927 | 0.922 | +0.005 |
| **`min`** | **0.540** | **0.316** | **+0.224** |
| `first_content_mean` | 0.982 | 0.848 | +0.134 |

Compare across models:

| call mode + model | best aggregator | gap |
| --- | --- | --- |
| Phi-3.5 base completion | `first_content_mean` | +0.254 (after content-stripping hack) |
| Phi-3.5 chat completion | `min` | +0.119 |
| **DeepSeek-R1 chat + post-`</think>`** | **`min`** | **+0.224** |

### Findings

1. **`min` over post-`</think>` is the strongest single confidence signal yet.** Easy answers are surgical (E1: min=0.997 over 8 tokens); hard answers carry visible uncertainty (H2: min=0.238).
2. **`reasoning_complete` is a powerful binary signal.** E3 ("Who wrote Hamlet?") and H1 (train math) both blew past 1024 tokens without closing `</think>`. The model overthinks both trivia and word problems when it's unsure how to compress its reasoning. The gate should treat `reasoning_complete=False` as a strong "fire" trigger regardless of any aggregate.
3. **DeepSeek refuses to confidently hallucinate H2.** Phi-3.5 (raw mode) said "1,100,000" with high confidence; Phi-3.5 (chat) hedged; DeepSeek hedged with a number (600,000) that is in the right ballpark and explicitly recommends checking sources. Reasoning model + chat template = much better epistemic behavior on factual gaps.
4. **Throughput on M4 Metal: 6–8 tok/s.** Calibration of 10 hard prompts × 1024 tokens × ~7 tok/s ≈ 25 min worst case. Not viable for iteration. CUDA expected to be 4–10× faster.

### Setup-script drift fixed during 3.0(a)

- `LLAMA_CPP_VERSION=0.3.5` → `0.3.22` (DeepSeek-R1 pre-tokenizer support).
- `numpy==2.2.1` → `numpy==2.4.4` (matches what 0.3.22 installs; was a false pin before).

### Decision: move to CUDA

Code, configs, and `_PLAN.md` are unchanged. Setup.sh auto-detects CUDA via `nvidia-smi` and pulls the appropriate prebuilt wheel from abetlen's index. See HANDOFF block at top of this file.

Artifact: partial run logged in terminal; persistent record is the table above. JSONL was not written because the script was killed before completion. **First action on CUDA: re-run `scripts/03_deepseek_calibration.py` end-to-end so we have a clean `runs/phase3_deepseek_calibration.jsonl`.**
