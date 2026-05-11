"""Generates 1-exploreModels.ipynb from the JSON log produced by 1-exploreModels.py.
Run: python _exploration/build_notebook.py
"""
import json, nbformat as nbf
from pathlib import Path

NB_PATH = Path(__file__).parent / "1-exploreModels.ipynb"
LOG_PATH = Path(__file__).parent / "1-exploreModels_log.json"

nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}

def md(src): nb.cells.append(nbf.v4.new_markdown_cell(src))
def code(src): nb.cells.append(nbf.v4.new_code_cell(src))

# ── Title ──
md("""# 🔬 Model Exploration Primer
## Understanding Phi-4-mini, Gemma-4-26B-A4B & Qwen3.6-35B-A3B

This notebook reads the structured JSON log produced by `1-exploreModels.py` and
walks you through **what these three models are, how they differ, and why they
were chosen** for the Recollection research project.

### Prerequisite knowledge assumed
- Deep Learning fundamentals (loss, backprop, SGD)
- CNNs (convolution, pooling, feature maps)
- Basic awareness of what an LLM is

### What you'll learn
1. **Transformer decoder basics** — attention, KV cache, hidden states
2. **Dense vs MoE vs Hybrid** — three architectural paradigms
3. **Chat templates** — how each model expects its input
4. **Hidden-state geometry** — what the model's "internal representation" looks like
5. **Uncertainty signals** — how confident each architecture is, and why it matters

---
""")

# ── Setup cell ──
code("""import json, textwrap
from pathlib import Path
from IPython.display import display, Markdown, HTML
import pandas as pd

LOG = Path("1-exploreModels_log.json")
if not LOG.exists():
    LOG = Path("_exploration/1-exploreModels_log.json")

with open(LOG) as f:
    data = json.load(f)

models = data["models"]
print(f"Log generated at: {data['generated_at']}")
print(f"Models available: {list(models.keys())}")
""")

# ── Background section ──
md("""## 📖 Background: Key Transformer Concepts

### Hidden States
Each layer in a Transformer produces a **hidden-state vector** for every token in
the input.  Think of it as the model's internal representation of meaning at that
depth — early layers capture surface patterns (syntax, word identity), deeper layers
encode increasingly abstract concepts (reasoning, factual recall).  The size of
this vector is the **hidden dimension** (e.g. 2048, 3072).

### Attention
The core mechanism that lets tokens "talk to each other."  Each token computes a
weighted sum over all other tokens' representations — learning *which* other tokens
are relevant to its own meaning.

- **Full attention**: every token sees every other token — powerful but O(n²) in
  sequence length
- **Sliding window**: each token only sees the last W tokens — cheaper, but can
  miss long-range dependencies
- **Linear attention** (e.g. Gated DeltaNet): approximates full attention with
  O(n) cost by maintaining a compressed recurrent state — the most memory-efficient
  option, but an approximation

### KV Cache
During generation, previously computed Key/Value vectors are cached so they don't
need recomputing on each new token.  This cache **grows with every token generated**
— it's the main memory bottleneck for long outputs.

### Dense vs MoE
- **Dense model**: every parameter fires on every token (simple, predictable)
- **MoE (Mixture of Experts)**: each layer has many small "expert" sub-networks,
  and a tiny **router** picks the top-K experts per token — only those fire.
  This gives huge capacity (many experts = diverse knowledge) with low per-token compute.
""")

# ── Model overview table ──
md("## 🏗️ The Three Architectures at a Glance")
code("""rows = []
for key in ["phi", "gemma", "qwen"]:
    m = models[key]
    arch = m.get("p3_architecture", {})
    census = arch.get("param_census", {})
    rows.append({
        "Model": m["family"],
        "Architecture": m["arch_class"],
        "Total Params": f'{census.get("total_params",0)/1e9:.1f}B',
        "Hidden Dim": arch.get("hidden_size", "?"),
        "Layers": arch.get("num_hidden_layers", "?"),
        "Attn Heads": arch.get("num_attention_heads", "?"),
        "KV Heads": arch.get("num_key_value_heads", "?"),
        "Experts": arch.get("num_experts", "—"),
        "Vocab": f'{arch.get("vocab_size",0):,}',
        "Context": f'{arch.get("max_position_embeddings",0):,}',
        "VRAM (MB)": m.get("vram_after_load_mb", "?"),
        "Load (s)": m.get("load_time_s", "?"),
    })
df = pd.DataFrame(rows).set_index("Model")
display(df.T)
""")

# ── What makes each special ──
md("## 🌟 What Makes Each Model Special")
code("""for key in ["phi", "gemma", "qwen"]:
    m = models[key]
    display(Markdown(f"### {m['family']}"))
    display(Markdown(f"**Architecture class:** {m['arch_class']}"))
    display(Markdown(f"**Why it's special:** {m['what_makes_it_special']}"))
    display(Markdown(f"**Lineage:** {m['lineage']}"))
    comparable = ", ".join(m.get("comparable_models", []))
    display(Markdown(f"**Comparable open models:** {comparable}"))
    display(Markdown("---"))
""")

# ── Deep dive: Dense vs MoE vs Hybrid ──
md("""## 🧠 Deep Dive: Dense vs MoE vs Hybrid Linear

### 1. Dense (Phi-4-mini)
Every parameter participates in every forward pass. Simple, predictable,
but expensive per-parameter at scale.

```
Input → [Layer 1: ALL params] → [Layer 2: ALL params] → ... → [Layer 32] → Output
         Every neuron fires          Every neuron fires
```

### 2. Mixture of Experts / MoE (Gemma-4-26B)
Each layer has many "expert" sub-networks. A **router** picks the top-K
experts per token — only those experts fire. Total params are large (26B),
but active params per token are small (3.8B).

```
Input → [Attention] → [Router: pick 8 of 128 experts] → [Only 8 fire] → ...
         Full cost      Tiny gating network               ~6% of FFN params
```

**Trade-off**: Huge capacity (many experts = diverse knowledge), but routing
adds noise — the hidden state passes through *different* experts on different
tokens, making the internal representations less geometrically smooth.

### 3. Hybrid Linear MoE (Qwen3.6-35B)
Combines **three** innovations:
- **Gated DeltaNet** (linear attention): 3 out of every 4 layers use O(1)-per-token
  recurrent attention instead of O(n²) full attention
- **Full attention**: every 4th layer uses standard quadratic attention
- **MoE**: all layers use 256-expert MoE FFN (8 routed + 1 shared)

```
Input → [DeltaNet + MoE] → [DeltaNet + MoE] → [DeltaNet + MoE] → [Full Attn + MoE] → repeat ×10
         O(1) attention     O(1) attention     O(1) attention     O(n²) attention
         8/256 experts      8/256 experts      8/256 experts      8/256 experts
```

**Trade-off**: Extremely memory-efficient for long sequences (tiny KV cache
from linear attention), but the combination of routing noise AND recurrent-state
approximation makes hidden-state geometry the most complex of the three.
""")

# ── Chat templates ──
md("## 💬 Chat Templates: How Each Model Expects Its Input")
code("""for key in ["phi", "gemma", "qwen"]:
    m = models[key]
    ct = m.get("p2_chat_template", {})
    display(Markdown(f"### {m['family']}"))
    display(Markdown(f"**EOS token:** `{ct.get('eos_token', '?')}`  |  "
                     f"**Vocab size:** {ct.get('vocab_size', '?'):,}  |  "
                     f"**Template kwargs:** `{ct.get('chat_template_kwargs', {})}`"))
    rendered = ct.get("rendered_text", "")
    # Show rendered template with visible special tokens
    display(Markdown(f"```\\n{rendered}\\n```"))
    display(Markdown("---"))
""")

# ── Identity probe ──
md("## 🪪 P1: Identity — What Does Each Model Say About Itself?")
code("""for key in ["phi", "gemma", "qwen"]:
    m = models[key]
    p1 = m.get("p1_identity", {})
    display(Markdown(f"### {m['family']}"))
    display(Markdown(f"> {p1.get('response', 'N/A')}"))
    display(Markdown(f"*{p1.get('new_tokens',0)} tokens, {p1.get('tok_per_s',0)} tok/s*"))
    display(Markdown("---"))
""")

# ── Hidden state geometry ──
md("""## 📐 P4: Hidden-State Geometry — Looking Inside the Model

When a model processes text, each layer produces a **hidden-state vector** for
each token. This is analogous to a feature map in a CNN — it's the model's
internal representation of meaning at that depth.

Key questions:
- **How big is it?** (hidden dimension)
- **How does the L2 norm change across layers?** (does the representation grow/shrink?)
- **How many layers?** (depth of the processing pipeline)
""")
code("""import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
colors = {"phi": "#4285F4", "gemma": "#EA4335", "qwen": "#34A853"}

for idx, key in enumerate(["phi", "gemma", "qwen"]):
    m = models[key]
    hs = m.get("p4_hidden_states", {})
    norms = hs.get("l2_norms_per_layer", [])
    if not norms:
        continue
    ax = axes[idx]
    ax.plot(range(len(norms)), norms, color=colors[key], linewidth=2)
    ax.fill_between(range(len(norms)), norms, alpha=0.15, color=colors[key])
    ax.set_title(f"{m['family']}\\n{len(norms)-1} layers, dim={hs.get('hidden_dim','?')}", fontsize=11)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("L2 norm of hidden state")
    ax.grid(True, alpha=0.3)
    # Mark mid-stack
    mid = len(norms) // 2
    ax.axvline(mid, color="gray", linestyle="--", alpha=0.5, label=f"mid={mid}")
    ax.legend(fontsize=9)

plt.suptitle("Hidden-State L2 Norm Across Layers (single forward pass)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("_exploration/hidden_state_norms.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: _exploration/hidden_state_norms.png")
""")

# ── Uncertainty separation ──
md("""## 🎯 P5: Uncertainty Separation — Can the Model Tell Easy from Hard?

This is the core question of the Recollection project.  We give each model:
- A **predictable** prompt: *"The capital of France is"*  (answer is obvious: Paris)
- An **ambiguous** prompt: *"The next word in this random sequence is"*  (no clear answer)

We then measure two things:
1. **L2 norm** of the last hidden state — does the "magnitude" of the representation differ?
2. **Entropy** of the next-token distribution — how uncertain is the model about what comes next?

If entropy separates well but L2 norm doesn't, that tells us: **raw hidden-state
magnitude is NOT a good uncertainty signal**.  We need something more sophisticated
— which is exactly what leads to the "Geometry-Aware Latent Gating" approach.
""")
code("""rows = []
for key in ["phi", "gemma", "qwen"]:
    m = models[key]
    u = m.get("p5_uncertainty", {})
    pred = u.get("predictable", {})
    amb = u.get("ambiguous", {})
    sep = u.get("separation", {})
    rows.append({
        "Model": m["family"],
        "Pred. Top Token": pred.get("top_token", "?"),
        "Pred. Top Prob": pred.get("top_prob", "?"),
        "Pred. Entropy": pred.get("entropy_nats", "?"),
        "Pred. L2": pred.get("hidden_l2", "?"),
        "Amb. Top Token": amb.get("top_token", "?"),
        "Amb. Top Prob": amb.get("top_prob", "?"),
        "Amb. Entropy": amb.get("entropy_nats", "?"),
        "Amb. L2": amb.get("hidden_l2", "?"),
        "L2 Δ%": sep.get("l2_pct_delta", "?"),
        "Entropy Δ (nats)": sep.get("entropy_delta_nats", "?"),
        "Entropy Ratio": sep.get("entropy_ratio", "?"),
    })
df = pd.DataFrame(rows).set_index("Model")
display(df.T)
""")

code("""# Visual comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
labels = []
l2_pred, l2_amb = [], []
h_pred, h_amb = [], []

for key in ["phi", "gemma", "qwen"]:
    m = models[key]
    u = m.get("p5_uncertainty", {})
    labels.append(m["family"].split("(")[0].strip())
    l2_pred.append(u.get("predictable", {}).get("hidden_l2", 0))
    l2_amb.append(u.get("ambiguous", {}).get("hidden_l2", 0))
    h_pred.append(u.get("predictable", {}).get("entropy_nats", 0))
    h_amb.append(u.get("ambiguous", {}).get("entropy_nats", 0))

x = range(len(labels))
w = 0.35
ax1.bar([i - w/2 for i in x], l2_pred, w, label="Predictable", color="#4285F4")
ax1.bar([i + w/2 for i in x], l2_amb, w, label="Ambiguous", color="#EA4335")
ax1.set_ylabel("L2 Norm")
ax1.set_title("Hidden-State L2 Norm\\n(small Δ = poor signal)")
ax1.set_xticks(list(x)); ax1.set_xticklabels(labels)
ax1.legend(); ax1.grid(axis="y", alpha=0.3)

ax2.bar([i - w/2 for i in x], h_pred, w, label="Predictable", color="#4285F4")
ax2.bar([i + w/2 for i in x], h_amb, w, label="Ambiguous", color="#EA4335")
ax2.set_ylabel("Entropy (nats)")
ax2.set_title("Next-Token Entropy\\n(large Δ = good signal)")
ax2.set_xticks(list(x)); ax2.set_xticklabels(labels)
ax2.legend(); ax2.grid(axis="y", alpha=0.3)

plt.suptitle("Uncertainty Separation: L2 Norm vs Entropy", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("_exploration/uncertainty_separation.png", dpi=150, bbox_inches="tight")
plt.show()
print("Key insight: Entropy separates well, L2 norm barely moves → raw L2 is not useful for uncertainty detection")
""")

# ── Reasoning probe ──
md("## 🧩 P6: Reasoning Probe — How Does Each Model Think?")
code("""for key in ["phi", "gemma", "qwen"]:
    m = models[key]
    p6 = m.get("p6_reasoning", {})
    display(Markdown(f"### {m['family']}"))
    display(Markdown(f"**Prompt:** {repr(p6.get('response','')[:500])}"))
    display(Markdown(f"*{p6.get('new_tokens',0)} tokens in {p6.get('gen_time_s',0)}s "
                     f"({p6.get('tok_per_s',0)} tok/s)*"))
    display(Markdown("---"))
""")

# ── Performance comparison ──
md("## ⚡ Performance Comparison")
code("""rows = []
for key in ["phi", "gemma", "qwen"]:
    m = models[key]
    p6 = m.get("p6_reasoning", {})
    census = m.get("p3_architecture", {}).get("param_census", {})
    rows.append({
        "Model": m["family"],
        "VRAM (MB)": m.get("vram_after_load_mb", 0),
        "Load Time (s)": m.get("load_time_s", 0),
        "Reasoning tok/s": p6.get("tok_per_s", 0),
        "Total Params (B)": f'{census.get("total_params",0)/1e9:.1f}',
        "BF16 Size (GB)": census.get("approx_bf16_gb", 0),
    })
df = pd.DataFrame(rows).set_index("Model")
display(df)
""")

# ── Summary ──
md("""## 🎯 Summary & Key Takeaways

### The Three-Way Bracket
These models were chosen to represent three points on the 2026 architectural spectrum
**at matched active-parameter scale** (~3-4B active per token):

| | Phi-4-mini | Gemma-4-26B | Qwen3.6-35B |
|---|---|---|---|
| **Paradigm** | Dense | Sparse MoE | Hybrid Linear MoE |
| **Active params** | 3.8B (all) | 3.8B (of 25.2B) | ~3B (of 35B) |
| **Attention** | Full (quadratic) | 5:1 sliding+global | 3:1 DeltaNet+full |
| **KV cache growth** | Linear, moderate | Linear, steep | Linear, tiny |
| **Internal geometry** | Cleanest | MoE routing noise | Most complex |
| **Thinking mode** | None | Off by default | On by default (must disable) |

### Why This Matters for the Research
The project studies whether you can detect **reasoning trajectory failure** by
watching the model's hidden states drift during generation.  The key finding so far:

1. **L2 norm alone is useless** — it barely distinguishes easy from hard prompts
2. **Entropy works** — but it's a surface signal (output logits), not a latent one
3. **The solution**: track *drift patterns* in hidden-state geometry across generation
   steps, and the optimal detection mode differs per architecture ("geometry-aware")

### Next Steps
- Run the full Phase 4 sweep to see these architectural differences at scale
- Study the CRV (Confidence Representation Vector) signals in `scripts/02_crv_probe.py`
- Read `_PLAN.md` for the full experimental design
""")

# ── Write notebook ──
nbf.write(nb, str(NB_PATH))
print(f"Notebook written to {NB_PATH} ({len(nb.cells)} cells)")
