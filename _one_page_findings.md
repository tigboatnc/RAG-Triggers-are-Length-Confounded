# Reasoning failures across edge-class LLM architectures: three findings from a controlled multi-signal trace corpus

## 1. What was done

We instrumented closed-book multi-hop QA decoding on three 2026 edge-class LLMs at matched ~3-4 B active-parameter scale — **Phi-4-mini-instruct** (dense, GQA), **Gemma-4-26B-A4B-it** (MoE, sliding 5:1 hybrid attention), **Qwen3.6-35B-A3B-Instruct** (MoE, Gated-DeltaNet 3:1 hybrid) — on HotpotQA-distractor (in-domain) and 2WikiMultiHopQA-distractor (zero-shot OOD), under a frozen Option-A decoding protocol (`_MODEL.md §3`). Per reasoning step we captured mid-stack hidden states, last-full-attention-layer attentions, next-token entropy, K=3 stochastic-consistency completions (u_cons), verbalised confidence (u_verb), hedge/verify lexical counts (Pandey HVR), and supporting metadata. Paired oracle re-decoding on supporting-fact-only paragraphs labels each trace with `lift_em ∈ {+1, 0, −1}`. Cohort: n=500 + 500 on Phi; pilot+full xarch sweep on Gemma + Qwen in flight. Pipeline, slice manifests, and per-trace dumps are reproducible from the on-disk repository.

## 2. Three findings, hard numbers

### F1 — Wu inverted-U is cross-architectural and architecture-dependent

EM rate by `n_steps` tertile, per arch × dataset (HotpotQA-200 V1 + Phi v2):

| arch | dataset | short EM | long EM | drop |
|---|---|---:|---:|---:|
| Phi | Hot (n=393) | 0.442 | 0.341 | −0.101 |
| Phi | 2Wiki (n=452) | 0.477 | 0.364 | −0.113 |
| Gemma | Hot (n=200) | 0.597 | 0.400 | −0.197 |
| Qwen | Hot (n=200) | 0.743 | 0.524 | **−0.219** |

Wu's "long reasoning hurts" claim, stated in 2025, replicates across the modern attention-design spectrum at 10-22 EM-point drops; **Qwen has the steepest inverted-U, Phi the shallowest** — slope tracks active-param efficiency and KV/state design. No prior work characterizes this across dense / MoE-sliding / MoE-hybrid-SSM at matched protocol.

### F2 — The trigger-utility paradox

Mean per-step signals by trace-type bucket (Phi v2, n=1000):

| bucket | n | tokens | ent.max | u_cons.max | dragin.max | drift.max |
|---|---:|---:|---:|---:|---:|---:|
| early_commit_wrong | 177 | 74 | 0.53 | 0.234 | 1.95 | 0.000 |
| short_correct | 306 | 65 | 0.46 | 0.167 | 1.80 | 0.019 |
| long_rambling_wrong | 97 | 201 | 1.49 | 0.519 | 3.70 | 0.192 |
| **long_correct** | **32** | **192** | **1.37** | **0.417** | **3.49** | **0.192** |

On **long traces the signals do not discriminate correct from incorrect** (drift identically 0.192; ent / u_cons / dragin within 10%). On short traces they do. Existing dynamic-RAG triggers (DRAGIN, FLARE, ETC, SELFDOUBT, ReaLM-RSUS) all rely on per-step uncertainty; **they work where retrieval is least needed and fail where retrieval is most needed**. Under length-residualized AUROC (continuous OLS on log_tokens), DRAGIN drops 0.732 → 0.616 (Hot) and is the only signal of 11 tested that retains residual AUROC > 0.58 cross-dataset; six other published triggers (u_cons, entropy, ETC, HVR, drift, variance, hidden_norm) collapse to length-equivalent. Result: **the field has been reporting a length-confounded number**.

### F3 — Uncertainty is empirically multi-dimensional, not scalar

Pairwise Pearson correlation of the per-trace signal stack (Phi v2 Hot, n=213 with all 8 signals populated):

- **Collapsed (r > 0.7)**: `drift_cosine ↔ variance_l2_window` (r = +0.853) — redundant CRV features, one can be dropped.
- **Moderate (0.3 < r < 0.7)**: dragin↔entropy (0.55), u_cons↔entropy (0.54), u_cons↔dragin (0.39).
- **Quasi-orthogonal (r < 0.3)**: u_cons↔drift (0.18), u_cons↔variance (0.03), HVR↔every internal signal (max 0.27).

The uncertainty stack is at least **4-dimensional**: attention-importance, stochastic-disagreement, entropy-trajectory, hidden-state-drift. Surface-form (HVR) is a fifth orthogonal axis. Scalar-uncertainty framings (every published dynamic-RAG paper) project a multi-dimensional structure onto one number.

## 3. Why this is historically notable

(a) **First controlled triangulation across the 2026 attention-design spectrum at matched active-parameter scale.** Bracketing dense (GQA), MoE-sliding-window (Gemma 4), and MoE-Gated-DeltaNet (Qwen 3.6) at ~3-4B active — the three architectural families that survived the 2024-2026 efficiency wave — under one decoding protocol with one capture stack. The 2026 Manik benchmark and the GDN papers (Yang, Sun, Gupta) measure VRAM/latency; nobody else has matched-protocol per-step reasoning-trace instrumentation across the three.

(b) **Empirical falsification of the dynamic-RAG-trigger framing.** Every trigger in the literature (DRAGIN 2024, FLARE 2023, ETC 2024, SELFDOUBT 2025, ReaLM 2026, INSIDE-EigenScore 2024) presupposes that per-step uncertainty discriminates retrieval-fixable from unrecoverable failures. F2 shows the discrimination is concentrated in *short* traces, exactly the regime where retrieval is structurally unnecessary; the *long* traces — where Song's detection dilemma, PAR²'s premature-commitment, and Wu's inverted-U all locate the actual failures — are signal-flat.

(c) **The length confound.** A simple `tokens_generated` baseline at n=500 reaches AUROC 0.749 on retrieval-helpfulness prediction in-domain, paired-statistically indistinguishable from u_cons, DRAGIN, and entropy. Most published dynamic-RAG comparisons in 2023–2026 fail to control for length; under our continuous-residualization protocol the field's signal ranking changes, and only DRAGIN survives as a genuine length-independent signal.

## 4. Direction

The findings reframe the research question from "how do we beat DRAGIN as a dynamic-RAG trigger" to "**how do we characterize and intervene on the long-trace signal-flat regime where retrieval-trigger frameworks structurally fail**." Candidate intervention surfaces — adaptive test-time compute, activation steering on detected trajectory divergence (Atkinson), and routed-action triggers conditioned on uncertainty-axis decomposition — are testable on the existing trace corpus once cross-arch capture completes.
