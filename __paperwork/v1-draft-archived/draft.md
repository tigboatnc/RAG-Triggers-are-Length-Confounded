# Draft 

## Introduction 
Small Language Models (SLMs) operating within constrained computational environments—specifically 6 to 20 gigabytes (GB) of Video RAM (VRAM)—can match or exceed the performance of massive, proprietary models such as GPT-4 or Claude 3 Opus

This parity is achieved not through sheer parameter scaling, but through the implementation of iterative, agentic Retrieval-Augmented Generation (RAG) architectures. When processing dense academic corpora of 50 to 100 complex papers, traditional single-pass RAG pipelines fail due to __noise injection, context fragmentation, and the inherent reasoning limitations of small-parameter models__. The paradigm must shift from `passive retrieval to active, self-correcting reasoning loops` 



## Bibliography 
2023 -> 2026 
1. Self-Reflective Retrieval-Augmented Generation (Self-RAG)
    - Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. arXiv preprint arXiv:2310.11511.
    - Short Summary 
        - 
    - Long Summary 
        - Traditional RAG architectures indiscriminately retrieve and incorporate a fixed number of passages regardless of necessity. This passive methodology often diminishes the versatility of  the language model, disrupts the coherence of the generated output, and introduces severe hallucinations when the retrieved context contradicts the model's internal logic or is irrelevant to the query. Asai et al. address this fatal flaw by introducing a framework that trains a single, arbitrary language model to adaptively retrieve passages on demand, generate responses, and rigorously reflect on its own outputs using specialized reflection tokens added to its vocabulary during the training phase.
        - The Self-RAG architecture operates through three distinct, iterative phases controlled entirely by these newly introduced tokens. First, in the Retrieve Phase, the model decodes a `` token to autonomously evaluate whether external information from the vector database is necessary to answer the current segment of the prompt. If its parametric knowledge is deemed sufficient, retrieval is bypassed entirely, saving compute cycles and preventing context pollution. If retrieval is required, the model executes a call to the external retrieval module. Crucially, Self-RAG allows for adaptive frequency, meaning the system can retrieve multiple times within a single generation loop or at various granularities depending on the complexity of the academic text being synthesized.
        - Second, during the Generate Phase, the model's behavior shifts based on the retrieval decision. It generates a continuation conditioned on the retrieved passages while simultaneously generating a critique token to assess if the retrieved documents are genuinely relevant to the query. Finally, the Critique Phase introduces self-evaluation. The model uses additional reflection tokens to evaluate whether the retrieved passages actively support the generated content, terminating pathways where the evidence contradicts the generated text. A final critique token evaluates the overall utility and quality of the generated response.
        - For a highly optimized local pipeline constrained by 6-20GB of VRAM, the most critical innovation of Self-RAG is its __enabling of segment-level beam search during inference__. By calculating the linear interpolation of desirable critique token probabilities at every step, the system identifies the top best continuations, effectively utilizing the 7B or 14B SLM as a highly controlled reasoning engine. This eliminates the need for external reinforcement learning reward models, keeping the VRAM footprint strictly within budget while elevating the model's factual accuracy to rival GPT-4.
2. Corrective Retrieval-Augmented Generation (CRAG)
    - Yan, S.-Q., Gu, J.-C., Zhu, Y., & Ling, Z.-H. (2024). Corrective Retrieval Augmented Generation. arXiv preprint arXiv:2401.15884.
    - Short Summary 
        - 
    - Long Summary 
        - While Self-RAG masterfully controls the generation side of the pipeline, Yan et al. identify that the retrieval mechanism itself remains exceptionally fragile. A low-quality retriever introduces substantial irrelevant information, which paralyzes SLMs and misleads them into generating hallucinations. To solve this, the authors propose Corrective Retrieval-Augmented Generation (CRAG), a plug-and-play methodology designed to self-correct retrieval results and maximize the utilization of retrieved academic documents.
        - The core of CRAG is the introduction of a lightweight retrieval evaluator. In their empirical setup, the authors utilize an evaluator with a mere 0.77B parameters (based on T5-large) to assess the quality of retrieved documents for a given query. Because this evaluator is so small, it reviews retrieval quality at an extremely low computational cost, making it perfect for a 20GB VRAM ceiling where the majority of memory must be reserved for the generative 7B-14B model. The evaluator outputs a confidence score for the retrieved context, which then triggers one of three strategic, agentic actions.
        - If the confidence score exceeds an upper threshold, the "Correct" action is triggered. However, rather than simply accepting the text, CRAG assumes the documents may still contain noise and subjects them to a "Decompose-then-Recompose" algorithm. Retrieved academic documents are fragmented into fine-grained knowledge strips, irrelevant strips are aggressively filtered out, and only the most critical insights are recomposed to form the final reference for the generator. If the score falls below a lower threshold, the "Incorrect" action discards the internal retrieved documents entirely and resorts to broad web searches to find alternative knowledge. If the score falls between the thresholds, the "Ambiguous" action combines both internal document refinement and external search results.
        - The empirical results of coupling CRAG with Self-RAG (referred to as Self-CRAG) demonstrate massive performance leaps for SLMs. On the complex Biography dataset, which requires long-form generation and deep factual consistency, Self-CRAG achieved a FactScore of 86.2, outperforming the standard Self-RAG implementation by a massive 5.0% margin when using a 7B parameter base model. Furthermore, ablation studies confirmed that removing the corrective actions dropped PopQA accuracy from 61.8% to 59.6%, proving that the self-correction mechanism—rather than the addition of external data—drives the performance.











