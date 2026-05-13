

# Support 
Here are 13 critical references from your provided sources that directly support and elaborate on your claims in the "GAP" section. I have focused entirely on validating your assertions regarding GRIP, ReaLM-Retrieve, the structural latency of post-decode triggers, and the fragility of post-hoc textual parsing.

### 1. Direct Evidence on GRIP's Post-Decode Bottleneck

**\cite{Li_RetrievalGenerationUnified}** **How it links, relevance:** This source directly supports your claim that GRIP relies on trained control tokens. GRIP embeds retrieval control into the generation process by requiring the model to autoregressively emit explicit tokens like `[RETRIEVE]`, `[INTERMEDIARY]`, and `[ANSWER]`. Because the framework relies on standard token-level decoding to trigger retrieval, it inherently forces the system to wait for a full decode step to evaluate these control signals, confirming your assertion that it is structurally late.

### 2. Direct Evidence on ReaLM-Retrieve's Post-Hoc Parsing

**\cite{Guo_WhenRetrieveAdaptive}** **How it links, relevance:** This paper precisely validates your description of ReaLM-Retrieve. The authors define their Reasoning Step Uncertainty Score (RSUS) as a combination of verbalized uncertainty ($U_{verb}$) and entity-based entropy ($U_{ent}$). To capture $U_{verb}$, the model prompts the reasoning engine at step boundaries to output a confidence rating between 0 and 100. To capture $U_{ent}$, the system extracts named entities using spaCy NER from the generated step. This proves your claim that ReaLM-Retrieve parses generated text post-hoc at reasoning boundaries, meaning the model has already committed to the reasoning trajectory before the trigger fires.

### 3. The "Structurally Late" Trigger Problem

**\cite{Li_ModelingUncertaintyTrends}** **How it links, relevance:** This source strongly supports your broader argument regarding the structural latency of existing RAG triggers. The authors explicitly identify a "delayed retrieval problem" in dynamic RAG systems that rely on token-level confidence. They observe that by the time a low-confidence token triggers the retrieval, the generation process has often already deviated from the correct path, resulting in delayed intervention and the propagation of errors.

**\cite{Zheng_LatentRAGLatent}** **How it links, relevance:** This paper reinforces the argument that generating explicit text before retrieving is a severe bottleneck. The authors note that agentic RAG methods incur substantial latency specifically due to the autoregressive generation of lengthy intermediate thoughts and subqueries in discrete language space. They demonstrate that thought and subquery generation accounts for approximately 90% of total inference latency, proving that waiting for decoded tokens before triggering retrieval is structurally inefficient.

**\cite{T_LatentAbstractionRetrieval}** **How it links, relevance:** This work further supports the critique of explicit text-based triggers by highlighting the computational overhead and latency of generating natural language queries at each step. The authors propose bypassing explicit token generation entirely by using the hidden representations of special tokens to trigger searches. This corroborates your transition to a pre-decode/hidden-state solution by highlighting the flaws of committing to textual tokens first.

**\cite{Su_DRAGINDynamicRetrieval}** **How it links, relevance:** This paper highlights the limitations of dynamic RAG methods that evaluate triggers only after generating a sentence or sequence of tokens. Existing methods evaluate uncertainty based on the probabilities of recently generated tokens, meaning the trigger only activates _after_ the model has attempted to formulate a response. This aligns perfectly with your claim that tokens are already committed before the trigger evaluates them.

**\cite{Jiang_ActiveRetrievalAugmented}** **How it links, relevance:** The FLARE framework detailed here exemplifies the post-decode trigger issue. It actively decides to retrieve only when the language model generates low-probability tokens during the formulation of a temporary next sentence. This supports your gap analysis by showing that standard dynamic RAG fundamentally relies on first committing to a provisional generation sequence before realizing retrieval is needed.

**\cite{Asai_SelfRAGLearning}** **How it links, relevance:** Self-RAG relies on an expanded model vocabulary to generate "reflection tokens" (like `[Retrieve]`) as an integral part of its autoregressive output. This provides additional evidence that state-of-the-art dynamic RAG systems treat retrieval triggers as standard generated text, forcing the system to wait for decoding and committing to tokens before external knowledge can intervene.

**\cite{Islam_OpenRAGEnhanced}** **How it links, relevance:** Similar to Self-RAG, Open-RAG uses reflection tokens paired with post-generation confidence scoring to determine retrieval necessity. Because the model must enforce a "no retrieval" condition to measure the confidence of an output sequence before deciding to actually retrieve, it is a distinctly post-hoc, post-decode evaluation mechanism.

### 4. Fragility of Verbalized Confidence & Post-Hoc Parsing

**\cite{Pandey_SELFDOUBTUncertainty}** **How it links, relevance:** This source provides excellent evidence to support your claim that ReaLM's reliance on verbalized confidence is "protocol-fragile". The authors note that verbalized confidence is often inconsistent across models and is notoriously miscalibrated on harder tasks or weaker models. This directly refutes the reliability of post-hoc verbalized self-evaluation, strengthening your argument against ReaLM-Retrieve's methodology.

**\cite{Chen_INSIDELLMsInternal}** **How it links, relevance:** This paper explicitly states that post-hoc semantic measurement on decoded language sentences is inferior to modeling internal states directly. It highlights that deriving sequence-level uncertainty from sophisticated, auto-regressive text outputs is a major challenge. This directly supports your proposal to abandon post-hoc parsing (like ReaLM-Retrieve) in favor of mid-stack hidden states.

**\cite{vanDijk_DetectingHallucinationsLarge}** **How it links, relevance:** This work refutes the assumption that output text or output logits are reliable indicators of model knowledge. The authors explain that because autoregressive models are trained to generate fluent continuations, they often produce overconfident falsehoods, meaning output-based uncertainty signals fail to distinguish confident errors from reliable knowledge. This supports your claim that waiting for committed text leads to hallucinated reasoning trajectories, and justifies shifting to internal signals.

**\cite{Kossen_SemanticEntropyProbes}** **How it links, relevance:** This source provides evidence that waiting for full text generation to evaluate uncertainty is unnecessary overhead. The authors demonstrate that uncertainty and semantic entropy can be accurately predicted _before_ generating any novel tokens by probing the model's hidden states with a single forward pass. This directly supports your proposed shift toward pre-decode hidden-state monitoring (your CRV adapter proposal) to catch hallucinations before tokens are committed.

# Refute 
\cite{Li_RetrievalGenerationUnified} **How it links, relevance:** This paper directly refutes your claim that GRIP's emission of a trained control token is a structural bottleneck that wastes compute. The authors of GRIP argue that internalizing retrieval as generative tokens (like `[RETRIEVE]`) is highly advantageous because it enables "end-to-end coordination without additional controllers or classifiers" and allows the model to autonomously plan multi-step retrieval directly within a single autoregressive trajectory, making it an adaptive and highly efficient feature rather than a flaw.

\cite{Guo_WhenRetrieveAdaptive} **How it links, relevance:** This paper contradicts your critique of ReaLM-Retrieve. While you claim its reliance on verbalized confidence and entity-coverage entropy is "post-hoc" and "protocol-fragile," the authors demonstrate that this Reasoning Step Uncertainty Score (RSUS) is highly robust. They show it adds merely 8% inference overhead while successfully reducing retrieval calls by 47% and establishing a new state-of-the-art efficiency-accuracy trade-off, explicitly validating verbalized uncertainty as a low-overhead and highly reliable signal.

\cite{Pandey_SELFDOUBTUncertainty} **How it links, relevance:** This source contradicts the premise that parsing generated text post-hoc is structurally flawed or fragile. The authors introduce SELFDOUBT, which parses reasoning traces for "hedging" markers (the Hedge-to-Verify Ratio). They demonstrate that traces with no hedging language are correct 96% of the time, serving as an emergent, high-precision confidence gate at _zero_ additional inference cost. This proves that post-decode text parsing can be extremely reliable and computationally superior to internal or pre-decode methods.

\cite{Testoni_MindGapBenchmarking} **How it links, relevance:** Further refuting the claim that post-decode textual features are fragile, this paper shows that extracting simple linguistic behavioral features from reasoning traces—such as self-posed questions, trace length, and the presence of "Wait" tokens—serves as a highly effective, lightweight, single-pass proxy for uncertainty. The authors prove that you do not need hidden states to get reliable uncertainty metrics; the generated text itself contains robust behavioral cues.

\cite{Li_ModelingUncertaintyTrends} **How it links, relevance:** This contradicts your claim that token-level, post-decode evaluations are inherently "structurally late." The authors propose the Entropy-Trend Constraint (ETC) which monitors the dynamic trends of token entropy during generation. They prove that relying on generated token probabilities can trigger retrieval _early and precisely_ (timely retrieval) without the delayed retrieval problem, avoiding the need to extract mid-stack hidden states.

\cite{Shapiro_HALTHallucinationAssessment} **How it links, relevance:** This paper contradicts your proposal to use mid-stack hidden states (your CRV adapter) by demonstrating that it is unnecessary. The HALT framework proves that you can achieve state-of-the-art hallucination detection purely by treating the top-20 token log-probabilities (a standard post-decode output) as a time series processed by a lightweight GRU. It argues that surface-level probability dynamics contain all the necessary signals, bypassing the need for white-box access to hidden layers.

\cite{Zheng_LatentRAGLatent} **How it links, relevance:** This source contradicts your claim that _existing_ RAG triggers are structurally late and rely solely on generated text tokens. The authors introduce LatentRAG, a framework that already successfully shifts reasoning and retrieval into the continuous latent space. It generates "latent tokens" for thoughts and subqueries directly from hidden states in a single forward pass, showing that pre-decode/latent triggers already exist in the literature.

\cite{T_LatentAbstractionRetrieval} **How it links, relevance:** Similar to LatentRAG, this paper contradicts the gap that existing methods rely heavily on explicit text-based retrieval. It proposes LAnR (Latent Abstraction for RAG), which bypasses discrete token generation entirely by using the internal representation space of the LLM as a latent query vector for retrieval, meaning the structural latency of explicit decoding has already been addressed by existing latent RAG frameworks.

\cite{Farquhar_Detectinghallucinationslarge} **How it links, relevance:** This contradicts your push away from post-decode evaluation by arguing that uncertainty _must_ be calculated at the level of semantic meaning. The authors show that post-decode generation sampling (Semantic Entropy) is critical because it accounts for the fact that a model can express the same correct idea in many different ways, something pre-decode hidden states or strict log-probs struggle to capture accurately.

\cite{Correa_EntropyGuidedLoop} **How it links, relevance:** This paper contradicts the assertion that post-decode triggers are wasteful. It introduces an Entropy-Guided Loop that uses a simple post-generation evaluation of perplexity, maximum token entropy, and low-confidence token counts to trigger a targeted refinement pass. The authors argue this is highly economical (cutting costs by 67% compared to full reasoning models) and effectively uses information the model _already_ computes, validating post-decode triggers as a highly practical middle ground.

\cite{Fu_MultipleChoiceQuestions} **How it links, relevance:** This source contradicts your proposal to rely heavily on internal/pre-decode metrics (like hidden state drift) during reasoning. The authors demonstrate that Chain-of-Thought reasoning causes LLMs to become artificially overconfident, systematically inflating their internal probability estimates even when the reasoning is flawed. This suggests that pre-decode or logprob-based signals might be inherently miscalibrated and unreliable during reasoning, validating the need for post-hoc textual evaluation instead.

\cite{Manakul_SelfCheckGPTZeroResource} **How it links, relevance:** This paper challenges the necessity of white-box hidden state extraction by showing that black-box, post-hoc consistency checking (SelfCheckGPT) is highly effective. It argues that evaluating the actual committed text through self-consistency is a robust way to detect hallucinations without ever needing to monitor internal drift or pre-decode activations.


- - - 

# Criticism 


1. \cite{Gupta_PersistentStateDataflowAccelerator} **How it links, relevance:** This source directly supports your claim regarding Qwen's Gated DeltaNets (GDN) solving the memory bottleneck. It explicitly notes that the hybrid Qwen3-Next architecture stacks GDN layers to replace the growing KV cache with a fixed-size recurrent state, drastically reducing the memory footprint to the point where the 2 MB GDN state can be held persistently on-chip.
    
2. \cite{Manik_Gemma4Phi4Qwen3Accuracy} **How it links, relevance:** This source validates your references to the 2026 architectures of Gemma 4, Phi-4, and Qwen3. It details a deployment-oriented benchmark measuring peak GPU memory usage (VRAM), latency, and FLOPs-per-token for these exact models (e.g., Gemma-4-E4B, Phi-4-reasoning), confirming that the focus in 2026 has shifted heavily toward evaluating practical operating points and memory efficiency in MoE vs. dense reasoning models.
    
3. \cite{Chen_ThinkDeepNotJustLong} **How it links, relevance:** This source strongly supports your hook's pivot to "cognitive cost" and "wasted tokens." It argues that increased generation length does not consistently correlate with accuracy and that excessive reasoning traces represent "overthinking." This directly backs your assertion that the bottleneck has shifted from memory to the computational waste of uninformative tokens during inference.
    
4. \cite{Wu_WhenMoreIsLess} **How it links, relevance:** This provides foundational evidence for the "wasted tokens" claim. The authors demonstrate that task accuracy follows an inverted U-shaped curve with Chain-of-Thought (CoT) length. It proves that excessively long reasoning trajectories degrade performance ("overthinking"), substantiating your claim that the primary cost is now the generation of wasted tokens along sub-optimal paths.
    
5. \cite{Zhang_HarnessingReasoningTrajectories} **How it links, relevance:** This source directly addresses "hallucinated reasoning trajectories." It highlights that large reasoning models (LRMs) generate long, seemingly coherent reasoning traces that can still culminate in hallucinated, incorrect answers because errors propagate through the trajectory. This perfectly sets up your argument that tracing and intervening in these trajectories is the new priority.
    
6. \cite{Yang_GatedDeltaNetworksImproving} **How it links, relevance:** This reinforces the first part of your hook regarding how 2026 architectures solve the memory bottleneck. It explains the mechanics of Gated Delta Networks (GDNs) unifying the delta rule with a gating mechanism to efficiently manage memory, allowing for linear-time complexity and eliminating the standard $O(N)$ KV cache scaling issues that plagued older Transformers.
    
7. \cite{Guo_WhenToRetrieveDuringReasoning} **How it links, relevance:** This source bridges the gap between your hook and the subsequent "GAP" section regarding structurally late RAG triggers. It notes that modern reasoning models (like DeepSeek-R1 and o1) generate 12,000–25,000 token chains, making post-decode or fixed-interval retrieval prohibitively expensive. It argues for step-level uncertainty detection during the reasoning chain, perfectly validating your critique of late-firing RAG triggers.
    
8. \cite{Atkinson_TetheredReasoningDecoupling} **How it links, relevance:** This source is highly relevant to your "PROPOSAL" section concerning hidden states and trajectory drift. It demonstrates that high-temperature "hallucination" is primarily a geometric trajectory divergence in the hidden-state representation space. It supports your proposed use of mid-stack hidden states (like your "CRV vector") to monitor drift and steer activations away from hallucination.
    
9. \cite{Zhang_100DaysAfterDeepSeekR1} **How it links, relevance:** This survey reinforces the shift in operational bottlenecks. It discusses how "overthinking" in reasoning language models leads to excessive reasoning tokens, which drastically raises operational and environmental costs. It directly supports the hook's premise that the cost bottleneck is now cognitive (wasted tokens) rather than purely memory-based.
    
10. \cite{Sun_FG2GDNEnhancingLongContext} **How it links, relevance:** Further corroborates the architectural shift mentioned in your hook. It details how Gated Delta Networks (GDN) and linear attention mechanisms reduce inference complexity from $O(L^2)$ to $O(L)$ by replacing softmax attention with matrix-valued hidden states, proving that the traditional memory bottlenecks are being engineered out of the latest models.
    
11. \cite{Fu_MultipleChoiceQuestionsReasoning} **How it links, relevance:** This provides additional backing for the "cognitive cost" argument. It explicitly states that while Chain-of-Thought (CoT) enhances reasoning, it leads to "overthinking" where models expend unnecessary computational resources on simple problems and follow incorrect reasoning paths.
    
12. \cite{Ainslie_GQATrainingGeneralized} **How it links, relevance:** This source provides the architectural foundation for your claim regarding Phi-4's use of Grouped-Query Attention (GQA). It explains how GQA reduces the size of the key-value cache compared to Multi-Head Attention, supporting your specific claim that dense models use GQA for O(N) efficiency.
    
13. \cite{Bocharnikov_KVCacheOffloading} **How it links, relevance:** This source can be used to contrast the old memory bottleneck with the new 2026 architectural solutions. It discusses KV cache offloading for models like Llama 3 and Qwen 3 to deal with context-intensive tasks, highlighting the severe memory constraints that newer architectures (like GDN) were specifically designed to bypass.


- - - - - 

# Verification of Hook


- **KV cache as the historical bottleneck (\cite{bocharnikov_kv_2026}):**
    - **Accurate:** Bocharnikov et al. explicitly state that the KV cache has become a "critical bottleneck for both latency and memory usage" and that prior work addressed this through "quantization" and "KV-cache offloading".
- **Gated Delta Networks replacing KV cache with a fixed-size recurrent state (\cite{yang_gated_2025,sun_fg2-gdn_2026}):**
    - **Accurate:** Yang et al. note that linear Transformers "reframe as a linear RNN with matrix-valued states" and utilize the delta rule to selectively update memory. Sun et al. state this reduces inference complexity "from $O(L^2)$ to $O(L)$".
- **GDN state remaining on-chip across decode steps (\cite{gupta_persistent-state_2026}):**
    - **Accurate:** Gupta et al. propose an FPGA accelerator specifically designed to hold "the full 2 MB recurrent state persistently in on-chip BRAM," eliminating off-chip memory round-trips and converting the workload to compute-bound.
- **Recent benchmarks shifting focus to peak VRAM and FLOPs (\cite{manik_gemma_2026}):**
    - **Accurate:** Manik and Wang benchmark Gemma-4, Phi-4, and Qwen3 variants, specifically emphasizing that deployability relies on "latency, throughput, and GPU memory" and tracking "peak GPU memory use, and an approximate floating-point operations (FLOPs)-per-token proxy".
- **GQA closing the gap on the dense side (\cite{ainslie_gqa_2023}):**
    - **Accurate:** Ainslie et al. confirm that Grouped-Query Attention drastically reduces "the size of the key-value cache and therefore amount of data that needs to be loaded" compared to Multi-Head Attention.
- **Tokens spent extending an already-wrong trajectory (\cite{chen_think_2026, wu_when_2025, fu_multiple_2026}):**
    - **Accurate:** Chen et al., Wu et al., and Fu et al. all extensively discuss the phenomenon of "overthinking," where models expend unnecessary computational resources on incorrect reasoning paths that ultimately degrade performance.
- **Growing share of inference cost on reasoning models (\cite{zhang_100_2025}):**
    - **Accurate:** Zhang et al. note that attacks or natural behaviors triggering "excessive reasoning" in LRMs are "raising operational and environmental costs".
- **Fluent hallucinated answers and error propagation (\cite{zhang_harnessing_2026}):**
    - **Accurate:** Zhang et al. state exactly this: "a single unsupported intermediate step can propagate through a long trajectory and culminate in a confident but wrong answer, while the surface form of the reasoning trace can remain persuasive".

_(Self-Correction/Minor Note):_ In your text, you refer to **"Qwen 3.6's hybrid Gated-DeltaNet stack"**. Ensure that "3.6" is not a typo. The sources (like Manik & Wang) evaluate Qwen3-8B and Qwen3-30B. If Qwen 3.6 is your own internal nomenclature or a specific sub-version you are testing in your empirical setup, that is fine, but double-check it against public release names.

---

### Thoughts on Narrative

**Strengths:**

- **The "Pivot" is Excellent:** The transition sentence—_"The cost ledger has therefore moved off the cache. What dominates instead is cognitive..."_—is incredibly strong. It executes a perfect academic pivot. You establish a solved problem (memory scaling) and immediately replace it with the new frontier (wasted reasoning tokens), creating a compelling gap for your paper to fill.
- **Empirical Grounding:** Backing up your literature review with your own measured data ($5{,}000$-token greedy decode) immediately establishes authority. It shows the reviewers that you aren't just summarizing papers; you have actively verified the architectural shift on the metal.
- **High-Density Writing:** The paragraph is packed with information but doesn't feel bloated. You successfully tie together hardware architecture, model benchmarks, and language model cognition in a single, cohesive flow.

**Areas for Improvement / Minor Tweaks:**

- **The `DynamicCache` Tangent:** The sentence _"Gemma's slope was the steepest of the three because the sliding window bounded compute, but the reference `DynamicCache` retained entries past the window, so memory grew unimpeded"_ is highly specific. While it explains the 221 KB/tok anomaly, it temporarily drags the narrative down into the weeds of a specific HuggingFace/implementation quirk right before your grand pivot.
    - _Suggestion:_ If word count is tight, consider moving this explanation to a footnote or the caption of `Figure 1`. This keeps the main text ruthlessly focused on the trajectory from "memory bottleneck" to "cognitive bottleneck."
- **Slight Phrasing Tweak:** Consider changing _"What dominates instead is cognitive: tokens spent..."_ to _"What dominates instead is the **cognitive cost**: tokens spent..."_ to explicitly link back to the "cost ledger" metaphor you established in the preceding sentence.