"""Generates 2-exploreBaselines.ipynb — a primer on the SOTA baseline methods.
Run: python _exploration/build_baselines_notebook.py
"""
import json, nbformat as nbf
from pathlib import Path

NB_PATH = Path(__file__).parent / "2-exploreBaselines.ipynb"
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}

def md(src): nb.cells.append(nbf.v4.new_markdown_cell(src))
def code(src): nb.cells.append(nbf.v4.new_code_cell(src))

# ── Title ──
md("""# 🛡️ Baseline Methods Primer
## Understanding the SOTA "Retrieval Trigger" Detectors

This notebook explains the **five baseline methods** that our Geometry-Aware
Latent Gating paper compares against. Each method answers the same question:

> **"Should we interrupt the model's generation right now and retrieve
> new information?"**

They just answer it using very different signals.

### The Two Families

| Family | Signal Source | Methods |
|---|---|---|
| **Trained Token Detectors** | Special tokens learned during fine-tuning | Self-RAG, GRIP |
| **Text-Parsing / Post-hoc Detectors** | Surface text patterns or output statistics | DRAGIN, SELFDOUBT, ReaLM-RSUS |

Our method (**Latent Gating**) is a third family: it watches the model's
*hidden states* directly — no trained tokens, no text parsing.

---
""")

# ── Setup ──
code("""import json, re, sys, textwrap
from pathlib import Path
from IPython.display import display, Markdown, HTML
import pandas as pd

# Load bootstrap results if available
BOOTSTRAP_V2 = Path("runs/phase4_bootstrap_v2.json")
BOOTSTRAP_ADDONS = Path("runs/phase4_bootstrap_v2_addons.json")
if not BOOTSTRAP_V2.exists():
    BOOTSTRAP_V2 = Path("../runs/phase4_bootstrap_v2.json")
    BOOTSTRAP_ADDONS = Path("../runs/phase4_bootstrap_v2_addons.json")

bs_v2, bs_addons = {}, {}
if BOOTSTRAP_V2.exists():
    bs_v2 = json.loads(BOOTSTRAP_V2.read_text())
if BOOTSTRAP_ADDONS.exists():
    bs_addons = json.loads(BOOTSTRAP_ADDONS.read_text())
print("Bootstrap v2 loaded:", bool(bs_v2))
print("Bootstrap addons loaded:", bool(bs_addons))
""")

# ── The Core Problem ──
md("""## 🎯 The Core Problem: When to Retrieve?

During multi-hop question answering, an LLM generates reasoning step by step.
Sometimes it gets stuck, hallucinates, or goes off track. The question is:
**can we detect this in real-time and trigger a retrieval to get it back on track?**

```
User: "Who directed the film shot in the city where Einstein was born?"

Model generating:    Step 1 → Step 2 → Step 3 → ... → Answer
                                  ↑
                          "Hmm, I'm not sure about..."
                          Should we retrieve here? 🤔
```

Each baseline method proposes a different **signal** to detect this moment.
Let's walk through each one.
""")

# ═══════════════════════════════════════════════════════════════════════
# FAMILY 1: TRAINED TOKEN DETECTORS
# ═══════════════════════════════════════════════════════════════════════

md("""## 🏋️ Family 1: Trained Token Detectors

These methods **fine-tune the LLM itself** to emit special control tokens
that signal "I need to retrieve." The model learns *when* it's uncertain
during training.

> **Key trade-off:** Very powerful (the model itself decides), but requires
> expensive SFT/RL training per model. You can't just drop these into a
> new model — you need to retrain.

---

### 1️⃣ Self-RAG (Asai et al., 2023)

**Paper:** *Self-RAG: Learning to Retrieve, Generate, and Critique through
Self-Reflection* (`arXiv:2310.11511`)

**Core idea:** Fine-tune the LLM to emit four special "reflection tokens":
- `[Retrieve]` — "Do I need external info right now?" (Yes/No)
- `[IsRel]` — "Is the retrieved passage relevant?" (Relevant/Irrelevant)
- `[IsSup]` — "Does the passage support my claim?" (Fully/Partially/No)
- `[IsUse]` — "Is my overall response useful?" (1-5 scale)

**How it works:**
```
                    ┌─────────────────────────────────┐
                    │          Self-RAG LLM            │
                    │  (fine-tuned with reflection     │
                    │   tokens in the vocabulary)      │
                    └──────────┬──────────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
     [Retrieve]=Yes     [Retrieve]=No      [Retrieve]=Yes
            │                  │                  │
            ▼                  ▼                  ▼
     Fetch passages      Continue gen       Fetch passages
            │                                     │
            ▼                                     ▼
     [IsRel] check                          [IsRel] check
     [IsSup] check                          [IsSup] check
            │                                     │
            └──────────────┬──────────────────────┘
                           ▼
                    [IsUse] → Final score
                    Beam search over best continuations
```

**Training:** SFT on ~150K instruction-output pairs where a critic model
(GPT-4) inserted the reflection tokens at appropriate places.

**Why it's in Related Work, not our baseline pool:** Requires full SFT of
the target model. We can't reproduce it on our three models without the
training pipeline + data. Moved to §2 Related Work.
""")

md("""### 2️⃣ GRIP (Li et al., 2024)

**Paper:** *GRIP: Retrieval-Generation Unified by a Learned Control Token*

**Core idea:** Train a single `<retrieve>` control token into the model's
vocabulary via SFT (40K samples) + DAPO reinforcement learning (5K samples).
The model learns to emit `<retrieve>` at exactly the right moment.

**How it works:**
```
Input: "The director of Big Stone Gap is from..."

Model output:  "The director is... <retrieve> ...based in Greenwich Village"
                                      ↑
                        Trained classifier fires here.
                        Model has learned that this is
                        where it needs external knowledge.
```

**Key difference from Self-RAG:** GRIP uses a *single* token with a
lightweight classifier head, vs Self-RAG's four-token critique system.
GRIP's RL training (DAPO) also teaches the model to optimize *when* to
retrieve for downstream accuracy, not just *if*.

**Why it's in Related Work, not our baseline pool:** Like Self-RAG,
requires SFT + RL training per model — "reproducing it from forward-pass
quantities is categorically impossible" (`baseline_gates.py` line 42).

> **💡 Key insight for our paper:** Both Self-RAG and GRIP show that the
> *model itself* contains enough information to decide when to retrieve.
> Our latent gating approach asks: can we extract that same signal from
> the hidden states **without any training**?
""")

# ═══════════════════════════════════════════════════════════════════════
# FAMILY 2: TEXT-PARSING DETECTORS
# ═══════════════════════════════════════════════════════════════════════

md("""## 🔍 Family 2: Text-Parsing / Post-hoc Detectors

These methods work **without modifying the model**. They analyze the
model's outputs (text, token probabilities, attention patterns) to detect
uncertainty. This is the family our latent gate competes against directly.

---

### 3️⃣ DRAGIN (Su et al., 2024)

**Paper:** *DRAGIN: Dynamic Retrieval Augmented Generation based on the
Real-time Information Needs of LLMs* (`arXiv:2403.10081v3`)

**Core idea:** Combine three per-token signals into a single "RIND" score
that measures how much the model *needs* information at each token:

$$S_{RIND}(t_i) = H_i \\cdot a_{max}(i) \\cdot s_i$$

Where:
- **H_i** = next-token entropy (how uncertain is the model?)
- **a_max(i)** = max attention token i receives from later tokens
  (how much does the future context "look back" at this point?)
- **s_i** = stopword filter (0 for "the", "is", etc.; 1 for content words)

**Intuition:** A high RIND means "the model was uncertain at a content
word, AND later tokens kept looking back at it" — a strong signal that
the model got stuck on something important.
""")

code("""# DRAGIN's RIND score — let's see the building blocks from our codebase

display(Markdown("### DRAGIN Implementation in Our Codebase"))
display(Markdown(\"\"\"
Our implementation lives in two files:
- `scripts/crv/sota_capture.py` — `RindAccumulator` class (streaming capture)
- `scripts/crv/baseline_gates.py` — `DraginGate` class (thresholded decision)

**How RIND accumulates during generation:**
```
Step 1:  Token "The"     → s=0 (stopword)  → RIND = 0
         Token "director" → s=1, H=2.1, a_max=0.3 → RIND = 0.63
         Token "of"       → s=0 (stopword)  → RIND = 0

Step 2:  Token "Big"      → s=1, H=3.5, a_max=0.7 → RIND = 2.45  ← HIGH!
         Token "Stone"    → s=1, H=1.2, a_max=0.4 → RIND = 0.48
         Token "Gap"      → s=1, H=0.8, a_max=0.2 → RIND = 0.16

Per-step max RIND: Step 1 = 0.63, Step 2 = 2.45
If threshold = 1.5 → DRAGIN fires at Step 2
```

**Architecture-specific detail:** On hybrid models (Qwen 3.6), DRAGIN
reads attention from the *last full-attention layer* (not a DeltaNet layer,
which doesn't produce standard attention matrices). Our driver auto-selects
this via `_resolve_dragin_attn_layer` reading `config.layer_types`.
\"\"\"))
""")

md("""### 4️⃣ SELFDOUBT (Pandey et al., 2026)

**Paper:** *SELFDOUBT: Detecting Reasoning Errors via Hedge-Verify Ratio*
(`arXiv:2604.06389v1`)

**Core idea:** When a model is uncertain, it *hedges* ("maybe", "perhaps",
"I'm not sure") and when it's confident, it *verifies* ("let me check",
"actually", "correction:"). The ratio tells you the model's self-assessed
confidence.

$$HVR_t = \\frac{h_t}{v_t + 1}$$

Where:
- **h_t** = count of hedge markers in the step text
- **v_t** = count of verify markers in the step text
- **+1** = Laplace smoothing (avoids division by zero)

**Tiered implementation:**
- **Tier-1 (HVR only):** Pure regex matching — no GPU needed
- **Full:** HVR + verbalized confidence, z-score fused per model

**What the markers look like:**
""")

code("""# Show the actual hedge and verify patterns from our codebase
display(Markdown("### SELFDOUBT Marker Dictionaries"))

hedge_patterns = [
    "maybe", "perhaps", "possibly", "probably",
    "seems", "appears", "not sure/certain",
    "I'm not sure/certain", "could be wrong",
    "might be wrong/incorrect", "hmm,", "wait,", "well,"
]

verify_patterns = [
    "let me verify/check/confirm/re-read",
    "I need to/should verify/check/confirm",
    "double-check", "substitute back", "plug in",
    "correction:", "actually,", "on second thought", "scratch that"
]

df = pd.DataFrame({
    "Hedge Markers (uncertainty)": hedge_patterns + [""] * (len(verify_patterns) - len(hedge_patterns)),
    "Verify Markers (self-correction)": verify_patterns + [""] * (len(hedge_patterns) - len(verify_patterns)),
})
# Trim to matching length
max_len = max(len(hedge_patterns), len(verify_patterns))
df = pd.DataFrame({
    "🟡 Hedge (uncertainty)": (hedge_patterns + [""])[:max_len],
    "🟢 Verify (self-correction)": (verify_patterns + [""])[:max_len],
})
display(df)

display(Markdown(\"\"\"
**Example:**
```
Step text: "Hmm, maybe the director is John Smith. Let me check..."
           ^^^^  ^^^^^                              ^^^^^^^^^^^^
           hedge  hedge                              verify

h = 2, v = 1 → HVR = 2 / (1 + 1) = 1.0
If threshold = 0.5 → SELFDOUBT fires ✅
```
\"\"\"))
""")

md("""### 5️⃣ ReaLM-RSUS (Guo et al., 2026)

**Paper:** *ReaLM-Retrieve: Real-time Uncertainty-Aware Retrieval*

**Core idea:** Combine three *different* uncertainty channels into a
single score:

$$RSUS = \\alpha \\cdot U_{verb} + \\beta \\cdot U_{ent} + \\gamma \\cdot U_{cons}$$

With published weights: α=0.40, β=0.35, γ=0.25, threshold τ=0.65

The three channels:

| Channel | What it measures | How |
|---|---|---|
| **U_verb** (40%) | Verbalized confidence | Fork KV cache, ask "Rate your confidence 0-100", parse answer |
| **U_ent** (35%) | Entity grounding entropy | Extract named entities via spaCy NER, compute BM25 posterior over passages |
| **U_cons** (25%) | Continuation consistency | Sample K=3 continuations, compute pairwise cosine similarity via sentence-BERT |

**Key insight:** Each channel captures a *different* aspect of uncertainty:
- U_verb = "Does the model *think* it's uncertain?" (metacognition)
- U_ent = "Are the entities in this step well-grounded in the passages?" (factual)
- U_cons = "Would the model say something different if we re-rolled?" (stability)
""")

code("""display(Markdown(\"\"\"### ReaLM-RSUS: The Three Uncertainty Channels

```
                     ┌─────────────────────────┐
                     │    Current generation    │
                     │    state (KV cache)      │
                     └────┬────────┬────────┬───┘
                          │        │        │
               ┌──────────┘        │        └──────────┐
               ▼                   ▼                   ▼
     ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
     │   U_verb (40%)  │ │   U_ent (35%)   │ │  U_cons (25%)   │
     │                 │ │                 │ │                 │
     │ Fork cache →    │ │ spaCy NER on    │ │ Sample 3 cont-  │
     │ "Rate confidence│ │ step text →     │ │ inuations at    │
     │  0-100" →       │ │ BM25 posteriors │ │ T=0.6 →         │
     │ Parse number →  │ │ over passages → │ │ Sentence-BERT → │
     │ 1 - conf/100   │ │ Shannon entropy │ │ 1 - mean cosine │
     │ Restore cache   │ │ (normalised)   │ │ Restore cache   │
     └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
              │                   │                   │
              └───────────────────┼───────────────────┘
                                  ▼
                    RSUS = 0.40·U_verb + 0.35·U_ent + 0.25·U_cons
                    Fire if RSUS > 0.65
```

**Cost:** Most expensive baseline — requires 3 extra forward passes per
step (1 for U_verb, 3 for U_cons) plus NER + BM25 for U_ent. On our
A800 setup, estimated 12-15 hours for the full 200-question sweep.

**Important scope note:** We evaluate ReaLM-Retrieve's *detection*
mechanism (RSUS), not its full REINFORCE-trained intervention policy.
Reported as "ReaLM-RSUS (Thresholded)" in our paper.
\"\"\"))
""")

# ═══════════════════════════════════════════════════════════════════════
# BONUS: ETC
# ═══════════════════════════════════════════════════════════════════════

md("""### 📈 Bonus: ETC — Entropy Trend Constraint (Li et al., 2026)

**Paper:** *Modeling Uncertainty Trends for Adaptive Retrieval*

**Core idea:** Don't look at absolute entropy — look at the *curvature*
of the entropy trend. The second-order difference catches "turning points"
where uncertainty suddenly changes direction.

$$|\\Delta^2 H_t| = |H_{t+1} - 2H_t + H_{t-1}|$$

**Intuition:** If entropy was falling (model getting more confident) and
suddenly starts rising (model getting confused), that inflection point
is a stronger signal than just "entropy is high."

```
Entropy
  │    ╱╲
  │   ╱  ╲         ╱╲
  │  ╱    ╲       ╱  ╲
  │ ╱      ╲     ╱    ╲
  │╱        ╲   ╱      ╲
  │          ╲ ╱        ╲
  └──────────────────────── Steps
              ↑          ↑
         |D²H| high  |D²H| high
         (turning     (turning
          point)       point)
```
""")

# ═══════════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════

md("## 📊 Head-to-Head Comparison")

code("""methods = [
    {"Method": "Self-RAG", "Family": "Trained Token", "Signal": "4 reflection tokens",
     "Requires Training": "Yes (SFT 150K)", "GPU at Inference": "No extra",
     "In Our Baseline Pool": "No (Related Work)"},
    {"Method": "GRIP", "Family": "Trained Token", "Signal": "1 control token + classifier",
     "Requires Training": "Yes (SFT 40K + RL 5K)", "GPU at Inference": "No extra",
     "In Our Baseline Pool": "No (Related Work)"},
    {"Method": "DRAGIN", "Family": "Text-Parsing", "Signal": "RIND = entropy × attention × stopword",
     "Requires Training": "No", "GPU at Inference": "Yes (output_attentions)",
     "In Our Baseline Pool": "Yes (Phase 4.8)"},
    {"Method": "SELFDOUBT", "Family": "Text-Parsing", "Signal": "Hedge-to-Verify ratio",
     "Requires Training": "No", "GPU at Inference": "Tier-1: No / Full: Yes",
     "In Our Baseline Pool": "Yes (Phase 4.7/4.10)"},
    {"Method": "ReaLM-RSUS", "Family": "Text-Parsing", "Signal": "Weighted: verbal + entity + consistency",
     "Requires Training": "No", "GPU at Inference": "Yes (3 channels)",
     "In Our Baseline Pool": "Yes (Phase 4.9)"},
    {"Method": "ETC", "Family": "Text-Parsing", "Signal": "|Δ²H| entropy curvature",
     "Requires Training": "No", "GPU at Inference": "No extra",
     "In Our Baseline Pool": "Yes (Phase 4.7)"},
    {"Method": "Latent Gate (Ours)", "Family": "Hidden-State", "Signal": "CRV drift + variance",
     "Requires Training": "No", "GPU at Inference": "Yes (output_hidden_states)",
     "In Our Baseline Pool": "— (our method)"},
]
df = pd.DataFrame(methods).set_index("Method")
display(df)
""")

# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENTAL RESULTS
# ═══════════════════════════════════════════════════════════════════════

md("""## 📈 What Our Experiments Show (So Far)

Our Phase 4 showdown evaluates each gate on **200 HotpotQA multi-hop
questions** across the three model architectures. The metric is
**Δfrac** — how much does triggering retrieval at this gate's chosen
moment improve the final EM score compared to no retrieval?
""")

code("""# Show bootstrap results from the addons run
if bs_addons:
    display(Markdown("### Bootstrap Results (SELFDOUBT-Tier1 + ETC vs Latent)"))
    rows = []
    for arch, data in bs_addons.get("per_arch", {}).items():
        short_name = arch.split("-")[0].capitalize()
        if "gemma" in arch: short_name = "Gemma-4-26B"
        elif "qwen" in arch: short_name = "Qwen3.6-35B"
        elif "phi" in arch: short_name = "Phi-4-mini"

        margins = data.get("margins", {})
        lat_vs_etc = margins.get("latent_minus_etc", {})
        lat_vs_sd = margins.get("latent_minus_selfdoubt_t1", {})

        rows.append({
            "Model": short_name,
            "Latent Mode": data.get("latent_mode_chosen", "?"),
            "n": data.get("n_eligible", "?"),
            "SELFDOUBT-T1 fires": f"{data.get('observed_fire_rate', {}).get('selfdoubt_t1', 0):.0%}",
            "ETC fires": f"{data.get('observed_fire_rate', {}).get('etc', 0):.0%}",
            "Latent fires": f"{data.get('observed_fire_rate', {}).get('latent', 0):.0%}",
            "Latent−ETC Δ": f"{lat_vs_etc.get('point', 0):+.3f}" if lat_vs_etc.get('point') else "N/A",
            "P(latent>ETC)": f"{lat_vs_etc.get('p_gt_zero', 0):.3f}" if lat_vs_etc.get('p_gt_zero') else "N/A",
        })
    display(pd.DataFrame(rows).set_index("Model"))
else:
    display(Markdown("*Bootstrap results not available — run the Phase 4 sweep first.*"))

display(Markdown(\"\"\"
### Key Findings So Far

1. **SELFDOUBT-Tier1 is essentially silent** on our models — HVR threshold
   collapses to 0.0 on calibration. Modern instruct models under greedy
   decode rarely hedge. This is consistent with the SELFDOUBT paper's own
   finding that "traces with no hedging are correct 96% of the time."

2. **Latent gate significantly beats ETC on Qwen** (P=0.992, 95%-significant).
   This is the first 95%-clear win against a true post-decode baseline.

3. **Gemma trends positive** (latent +0.060 over ETC) but doesn't clear 95%
   at n=40.

4. **Phi remains an honest negative** — latent loses to ETC. Dense
   architectures may not produce the hidden-state geometry signals that
   MoE/hybrid models do.

5. **DRAGIN, ReaLM-RSUS, and SELFDOUBT-Full are pending** — these require
   GPU compute (attention capture, KV-cache forking, consistency sampling)
   and are tracked as Phase 4.8/4.9/4.10.
\"\"\"))
""")

# ═══════════════════════════════════════════════════════════════════════
# WHY OUR METHOD IS DIFFERENT
# ═══════════════════════════════════════════════════════════════════════

md("""## 🧬 Where Our Latent Gate Fits

```
Signal Source Spectrum:

  TRAINED TOKENS          TEXT SURFACE            HIDDEN STATES
  ─────────────          ─────────────          ──────────────
  Self-RAG, GRIP         SELFDOUBT, DRAGIN,     Latent Gate (Ours)
                         ReaLM-RSUS, ETC

  ✅ Most powerful        ✅ No training needed    ✅ No training needed
  ❌ Needs SFT/RL         ❌ Post-hoc (delayed)    ✅ Pre-token (earliest)
  ❌ Per-model training   ✅ Model-agnostic        ⚠️ Architecture-aware
                         ❌ Misses latent signals  ✅ Sees internal state
```

**The key insight:** Text-parsing methods see the model's output *after*
the uncertainty has already manifested as words. Our latent gate sees the
hidden-state drift *before* the uncertain token is even sampled — that's
the earliness advantage on MoE architectures (Gemma, Qwen).

**The honest caveat:** This advantage is **architecture-dependent**. On
dense models (Phi), the hidden-state geometry is "too clean" — drift and
variance don't separate well. The paper frames this as the
"geometry-aware" finding: different architectures need different detection
modes.

---

## 📚 Further Reading

- `scripts/crv/baseline_gates.py` — All gate implementations
- `scripts/crv/sota_capture.py` — DRAGIN/ReaLM capture utilities
- `scripts/04f_run_qa_sota.py` — Full SOTA sweep driver
- `_AGENT.md` entries #16, #17 — Experimental evolution
- `_PLAN.md` Phases 4.7–4.10 — Baseline rebuild roadmap
""")

# ── Write ──
nbf.write(nb, str(NB_PATH))
print(f"Notebook written to {NB_PATH} ({len(nb.cells)} cells)")
