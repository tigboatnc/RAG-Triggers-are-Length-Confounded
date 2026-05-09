# recollection.local

> Local Recollection, RAG.
> Agentic reasoning under **KV-cache pressure** via confidence-gated computation.
> Co-design of inference algorithm + systems constraints.

## High Level

As agentic reasoning loops push context windows from a few thousand to tens of thousands of tokens — multi-hop retrieval, `<think>` chains, self-verification passes — **KV-cache memory becomes the binding constraint long before model weights do**. This is true on a 24 GB consumer GPU at 4 k context and equally true on an 80 GB datacenter card at 64 k context.

The recollection.local prototype tests one hypothesis: that the cheapest place to detect a model is about to hallucinate (and therefore needs corrective retrieval, more thinking budget, or a stop) is in its **hidden states at logical step boundaries** — not in its output tokens, not in `<think>` syntax, not in self-verification text.

## Three pain points this targets

1. **LIM** — Lost in the Middle, esp. at long agentic context
2. **RAG noise / fragmentation** — over-retrieval pollutes the working window
3. **KV-cache explosion** — the dominant cost of agentic methods at long context, and the one most directly attacked by gating *when* corrective retrieval fires

## Core ideas

### Confidence-gated discrimination

- Cuts off unnecessary reasoning paths
- Reasoning is conditional compute — not always-on
- Simple to implement, layers cleanly onto an existing generator

> *Don't think harder unless you're unsure.*

### Co-design (systems × algorithm)

- Quantization (NF4 / MXFP4) + KV compression
- Bounded MCTS over reasoning steps
- Retrieval gating from latent state, not output text

## Framing

- Core contribution = **confidence-gated reasoning** at logical step boundaries
- Supporting systems = quantization + retrieval choices that make the gate cheap to run
- Target regime = **agentic LLM reasoning where KV-cache pressure dominates total cost**, not single-shot prompting

## Things to push on

1. Is *confidence* reliable enough to gate reasoning?
   - LMs are often overconfident when wrong (esp. multi-hop)
   - Mitigations under exploration:
     1. Empirical correlation: confidence vs correctness
     2. Fallbacks: periodic forced retrievals, uncertainty smoothing

---

## Benchmarks

1. NeedleChain
2. LongBench v2
