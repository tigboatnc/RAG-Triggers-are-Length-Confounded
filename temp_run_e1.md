# temp_run_e1.md — Gemma E1 oracle decode on a fresh box

**Goal.** On a fresh Linux + NVIDIA box, install env, download Gemma weights, run the one oracle decode below at `n=200`, ship the resulting JSONL back. Wall budget: **~45 min** if weights are uncached, ~20 min if cached.

---

## 0. Inputs the user must fill in before launching

| variable | value |
|---|---|
| `REPO_URL` | (git URL of `recollection.local`) — or scp/tar a snapshot in |
| `COMMIT_SHA` | `044c7e1` (or whatever is current `HEAD` on the main box) |
| `HF_TOKEN` | a Hugging Face token with access to `google/gemma-4-26B-A4B-it` |
| `RETURN_DEST` | where to scp/rsync the final JSONL back (e.g. `user@main-box:/path/to/recollection.local/runs/v2_headline_gemma-4-26b-a4b-it/`) |

Box requirements: **Linux**, **NVIDIA GPU with ≥ 60 GB VRAM** (A100-80G or A800-80G), `nvidia-smi` works, ≥ 100 GB free disk, conda or miniforge available.

---

## 1. Clone + checkout

```bash
git clone "$REPO_URL" recollection.local
cd recollection.local
git checkout "$COMMIT_SHA"
```

---

## 2. Bootstrap the `recollection` conda env

The repo ships a one-shot setup script that creates the conda env (`recollection`), installs torch cu124, and smoke-tests CUDA. **Run it as-is; do not improvise.**

```bash
bash scripts/setup.sh
```

Success criterion: final line prints `[ok] setup complete.` and the smoke block prints `cuda_available=True` and a non-empty `device 0` line. If `cuda_available=False`, stop and report — do not try to run E1 on CPU.

Then activate:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate recollection
```

---

## 3. Download Gemma weights

Target dir (must match exactly — the run script reads from here):

```
models/gemma-4-26b-a4b-it/
```

HF repo: `google/gemma-4-26B-A4B-it`, pinned revision `462a98a12e28e2cbcfccaf78fe41e3e50235e6ae`. ~49 GB bf16.

```bash
export HF_TOKEN="$HF_TOKEN"   # token from step 0
mkdir -p models/gemma-4-26b-a4b-it

huggingface-cli download \
  google/gemma-4-26B-A4B-it \
  --revision 462a98a12e28e2cbcfccaf78fe41e3e50235e6ae \
  --local-dir models/gemma-4-26b-a4b-it \
  --local-dir-use-symlinks False
```

Sanity:

```bash
ls models/gemma-4-26b-a4b-it/*.safetensors | wc -l   # should be ≥ 1
du -sh models/gemma-4-26b-a4b-it                     # should be ~49 GB
```

If download is slow (< 50 MB/s), check `HF_TOKEN` is exported in the same shell — unauthenticated HF requests get rate-limited (this was the bottleneck on the main box).

---

## 4. Verify slice file is present

The slice is committed in the repo at:

```bash
ls -l data/hotpotqa_distractor_500_v2test.jsonl
wc -l data/hotpotqa_distractor_500_v2test.jsonl   # should be 500
```

If missing, **stop and report** — do not try to regenerate it (seed reproducibility matters; the main box already has the canonical slice).

---

## 5. Run E1 — Gemma × HotpotQA oracle decode, n=200

This is the entire compute job. One command. Expected wall: **~25-35 min** on A100-80G / A800-80G. No K=5 sampling, no hidden capture — it's much lighter than the closed-book phases.

```bash
mkdir -p runs/v2_headline_gemma-4-26b-a4b-it/logs

python scripts/04l_oracle_retrieve_decode.py \
  --model gemma-4-26b-a4b-it \
  --model-dir models/gemma-4-26b-a4b-it \
  --dataset data/hotpotqa_distractor_500_v2test.jsonl \
  --limit 200 \
  --out runs/v2_headline_gemma-4-26b-a4b-it/E1_oracle_hotpot.jsonl \
  2>&1 | tee runs/v2_headline_gemma-4-26b-a4b-it/logs/E1_oracle_hotpot.log
```

Watch for `[200]` near the end of stdout and a final line like:

```
[ok] wrote runs/v2_headline_gemma-4-26b-a4b-it/E1_oracle_hotpot.jsonl
     n=200  EM=<XX>/200  meanF1=<…>
```

---

## 6. Post-run sanity gates

All three must pass before shipping back:

```bash
# (a) Exactly 200 rows
wc -l runs/v2_headline_gemma-4-26b-a4b-it/E1_oracle_hotpot.jsonl
# expect: 200 runs/...

# (b) EM rate in a sane band (oracle Gemma should land roughly 50-70%, do not panic on exact value)
grep -E "EM=" runs/v2_headline_gemma-4-26b-a4b-it/logs/E1_oracle_hotpot.log | tail -1

# (c) Every row parses as JSON and has the expected keys
python - <<'PY'
import json
n_ok = 0
keys_needed = {"id", "em", "f1", "model_answer_extracted", "tokens_generated"}
with open("runs/v2_headline_gemma-4-26b-a4b-it/E1_oracle_hotpot.jsonl") as f:
    for line in f:
        r = json.loads(line)
        assert keys_needed <= set(r.keys()), f"missing keys in row {r.get('id')}"
        n_ok += 1
print(f"[gate] n={n_ok} rows parsed, all required keys present")
assert n_ok == 200, f"expected 200, got {n_ok}"
print("[gate] PASS")
PY
```

If any gate fails: paste the failing output, do **not** ship the file, stop and report.

---

## 7. Ship JSONL back

The output file is small (~few MB). Send back to the main box:

```bash
scp runs/v2_headline_gemma-4-26b-a4b-it/E1_oracle_hotpot.jsonl \
    runs/v2_headline_gemma-4-26b-a4b-it/logs/E1_oracle_hotpot.log \
    "$RETURN_DEST"
```

Or rsync, or `gh release upload`, or whatever the user provided as `RETURN_DEST`.

---

## 8. (Optional) E2 too, ~25 min more — strongly recommended

Since Gemma weights are loaded and warm, running E2 right after E1 is essentially free wall-time-wise. Worth doing.

```bash
python scripts/04l_oracle_retrieve_decode.py \
  --model gemma-4-26b-a4b-it \
  --model-dir models/gemma-4-26b-a4b-it \
  --dataset data/2wikimultihop_distractor_500_v2test.jsonl \
  --limit 200 \
  --out runs/v2_headline_gemma-4-26b-a4b-it/E2_oracle_wiki.jsonl \
  2>&1 | tee runs/v2_headline_gemma-4-26b-a4b-it/logs/E2_oracle_wiki.log

# Same three gates as step 6 on the new file, then scp back alongside E1.
```

---

## 9. Tear down

Once the user confirms receipt of E1 (and E2 if you ran it), the rental can be released. No state worth preserving on the rented box.

---

## Failure recovery cheat-sheet

| symptom | action |
|---|---|
| `nvidia-smi` missing | wrong box type — stop, report |
| `cuda_available=False` after `scripts/setup.sh` | driver mismatch; rerun setup, check torch wheel vs `nvidia-smi` driver version |
| HF download stalls / 401 | `HF_TOKEN` not exported in current shell, or token lacks Gemma access |
| OOM during decode | confirm GPU is 80GB; if 40GB, the run won't fit at bf16 — stop, report |
| `[200]` line never appears | check log tail; if dataset-load error or NaN, stop and report log |
| n=200 not exactly hit | usually a crash mid-decode; re-run the same E1 command, the script resumes by row count |
