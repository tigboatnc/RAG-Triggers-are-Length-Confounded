# **Dynamic Retrieval-Augmented Generation (RAG): Methods, Pivots, SOTA, and Benchmark Datasets**

## 1. Introduction

Dynamic Retrieval-Augmented Generation (RAG) represents a rapidly evolving paradigm in large language models (LLMs), enabling systems to adaptively retrieve and integrate external knowledge during generation for complex, knowledge-intensive tasks. Recent years have seen a proliferation of dynamic RAG methods—such as FLARE, DRAGIN, DynamicRAG, Adaptive-RAG, Layered Query Retrieval, and agentic/graph-based approaches—each introducing new mechanisms for when and what to retrieve based on model uncertainty, query complexity, or real-time information needs  (Su et al.; Su et al. 12991-13013; Sun et al.; Hei et al.; Huang et al.; Sharma). The field has also witnessed significant pivots: from static pipelines to modular, policy-driven architectures; from unimodal to multimodal and agentic frameworks; and from single-hop to multi-hop and event-centric reasoning  (Gao et al.; Brown et al.; Singh et al.; Sun et al.; Liu et al.). Benchmarking practices have matured with the introduction of datasets like HotpotQA, Natural Questions (NQ), TriviaQA, 2WikiQA, MuSiQue, Dyn-VQA, ViDoSeek, CRAG, RGB, InfoDeepSeek, and others tailored for dynamic retrieval scenarios  (Su et al. 12991-13013; Li et al.; Sun et al.; Brown et al.; Sun et al.; Yang et al.; Guti'errez et al.). This review synthesizes the latest advances in dynamic RAG methods, their relevance in 2026 and beyond, state-of-the-art (SOTA) trends, and the datasets used for assessment.


**Figure 1:** Consensus meter: Are dynamic RAG methods SOTA for knowledge-intensive tasks?

## 2. Methods

A comprehensive search was conducted across over 170 million research papers in Consensus—including Semantic Scholar, PubMed, ArXiv, IEEE Xplore, Scopus, DBLP—using targeted queries on dynamic RAG methods (FLARE, DRAGIN, DynamicRAG), pivots/SOTA trends since 2023–2026+, evaluation datasets/benchmarks (HotpotQA etc.), agentic/multimodal/graph-based RAGs. In total: 16335132 papers were identified; after deduplication and relevance filtering 230 were screened; 135 were eligible; the top 50 most relevant papers were included in this review.

<search_strategy_diagram identifiedPapers="16335132" screenedPapers="230" eligiblePapers="135" includedPapers="50" />
**Figure 2:** Search strategy flow: identification to inclusion of top dynamic RAG papers.
Six unique search groups were executed to map the landscape of dynamic RAG methods and benchmarks.

## 3. Results

### 3.1 Landscape of Dynamic RAG Methods

Dynamic RAG approaches now span token-level uncertainty triggers (FLARE), self-attention-based query formulation (DRAGIN), RL-trained rerankers that learn k per query (DynamicRAG), classifier-based routing by complexity (Adaptive-RAG), semantic rule-based complexity layers (Layered Query Retrieval), agentic planning agents (Agentic RAG/AU-RAG/HM-RAG), graph-based retrievals (GraphRAG/DyG-RAG/KG²RAG/HippoRAG), multimodal extensions (mRAG/ViDoRAG/QA-Dragon/ImageRAG), parametric knowledge injection at test time (DyPRAG), and streaming/index-updating frameworks  (Su et al.; Su et al. 12991-13013; Tan et al.; Sun et al.; Jang and Li; Singh et al.; Hei et al.; Srinivas and Runkana; Sun et al.; Huang et al.; Zhang et al.; Guo et al.; Liu et al.; Ruangtanusak et al.).

Notable new methods include:
- **Self-RAG**: Reflection tokens trigger retrieval  (Su et al.)- **SeaKR**: Self-aware uncertainty quantifies token-level need  (Su et al.)- **DH-RAG**: Dynamic historical context for dialogue  (Zhang et al.)- **OmniSearch**: Self-adaptive planning agent for multimodal retrieval  (Li et al.)- **DoTA-RAG**: Dynamic routing/hybrid retrieval at web scale  (Ruangtanusak et al.)- **DioR**: Adaptive cognitive detection/contextual optimization  (Guo et al.)### 3.2 Pivots & State-of-the-Art Trends

Since ~2024 there has been a clear pivot from static retrieve-then-generate pipelines toward modular/adaptive stacks with:
- Uncertainty-triggered or policy-driven retrieval loops
- Hybrid dense/sparse/graph/memory retrievers
- Agentic orchestration/workflow planning
- Multimodality as a core feature
- Budget-aware tool use & provenance tracking
These advances are now considered SOTA for complex QA/multi-hop reasoning/long-form generation  (Gao et al.; Su et al.; Su et al. 12991-13013; Sun et al.; Brown et al.; Singh et al.; Srinivas and Runkana; Sun et al.).

### 3.3 Benchmark Datasets & Evaluation Practices

Dynamic RAG systems are assessed on a wide range of benchmarks:
- **Knowledge-intensive QA**: HotpotQA, Natural Questions (NQ), TriviaQA, FEVER
- **Multi-hop/Complex QA**: MuSiQue, 2WikiQA
- **Long-form Generation**: ASQA, ELI5
- **Temporal/Event Reasoning**: TimeQA, TempReason, ComplexTR
- **Multimodal/VQA**: Dyn-VQA  (Li et al.), ViDoSeek  (Wang et al. 9113-9134), ScienceQA/CrisisMMD  (Liu et al.), QA-Dragon/MM CRAG Challenge  (Jiang et al.)- **Agentic/Open-Web**: InfoDeepSeek  (Xi et al.), CRAG benchmark  (Yang et al.), RGB benchmark  (Chen et al. 17754-17762)Other datasets include PopQA/NarrativeQA/LV-Eval for memory/sense-making tasks; BioASQ/PubMedQA/MIRAGE for medical domains; FineWeb-10BT/WebOrganizer/DataMorgana for web-scale evaluation  (Li et al.; Wang et al. 9113-9134; Sun et al.; Brown et al.; Sun et al.; Yang et al.; Ruangtanusak et al.).

### 3.4 Comparative Performance & Efficiency

Dynamic/adaptive methods consistently outperform static baselines on multi-hop QA and long-form generation—e.g., DRAGIN/FLARE yield up to ~62% improvement over raw LLMs on multi-hop tasks; Layered Query Retrieval outperforms Adaptive-RAG by +9.4% accuracy/+16.14% F1 on HotpotQA; DoTA-RAG achieves >0.92 correctness at web scale with low latency; DyPRAG improves standard RAG by +6.54% via parametric fusion at test time  (Su et al.; Su et al. 12991-13013; Tan et al.; Sun et al.; Hei et al.; Huang et al.).

#### Results Timeline

results_timeline

**Figure 3:** Timeline of key dynamic RAG method publications from foundational work to recent SOTA advances. Larger markers indicate more citations.

#### Top Contributors

<top_contributors authors='@@json@@:[{"name": "Weihang Su", "citations": "5,7"}, {"name": "Jiashuo Sun", "citations": "13"}, {"name": "Penghao Zhao", "citations": "1"}]' journals='@@json@@:[{"name": "ArXiv", "citations": "1,3,7,9,10,11,13"}, {"name": "Applied Sciences", "citations": "22,27"}, {"name": "Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval", "citations": "5"}]' />

**Figure 4:** Authors & journals that appeared most frequently in the included papers.

## 4. Discussion

The research demonstrates that dynamic/adaptive RAG is now central to SOTA performance in knowledge-intensive NLP tasks—especially those requiring multi-step reasoning or robust grounding under uncertainty or domain shift  (Gao et al.; Su et al.; Su et al. 12991-13013). The field has moved decisively toward modularity (retriever/generator decoupling), policy-driven control loops (uncertainty triggers/classifiers/bandits/RL agents), hybrid retrievers (dense/sparse/graph/memory/multimodal), agentic orchestration/planning agents for workflow management across tools/data sources—and holistic benchmarking practices that combine quality/cost/safety metrics with provenance tracking  (Brown et al.).

However:
- Many benchmarks still focus on overlap metrics rather than holistic cost-quality-safety tradeoffs.
- Generalization under domain shift/evolving corpora remains brittle.
- Security/privacy/retrieval poisoning are emerging concerns.
Recent work calls for unified multilingual/multimodal benchmarks and end-to-end security frameworks as next steps toward reliable deployment at scale  (Brown et al.).

### Claims and Evidence Table

| Claim                                                                 | Evidence Strength                | Reasoning                                                                                                   | Papers                                  |
|-----------------------------------------------------------------------|----------------------------------|------------------------------------------------------------------------------------------------------------|-----------------------------------------|
| Dynamic/adaptive RAG is SOTA for complex QA/multi-hop reasoning       | Evidence strength: Strong (9/10) | Multiple studies show consistent gains over static baselines across diverse benchmarks                      |  (Su et al.), (Su et al. 12991-13013), (Sun et al.), (Hei et al.), (Huang et al.), (Sharma)|
| Modular/policy-driven architectures outperform monolithic pipelines    | Evidence strength: Strong (8/10) | Systematic reviews highlight shift toward modularity/hybrid retrievers/policy control                       |  (Gao et al.), (Brown et al.), (Singh et al.)|
| Multimodal/agentic extensions further improve robustness/flexibility  | Evidence strength: Moderate (7/10) | Surveys show multimodal/agentic systems excel in open-ended/dynamic environments                            |  (Abootorabi et al. 16776-16809), (Li et al.), (Wang et al. 9113-9134), (Singh et al.), (Xi et al.), (Liu et al.)|
| Evaluation remains overlap-heavy; cost/safety often neglected         | Evidence strength: Moderate (6/10) | Reviews note lack of holistic benchmarks combining quality/cost/safety                                      |  (Brown et al.), (Yang et al.)|
| Generalization under domain shift is still brittle                    | Evidence strength: Moderate (4/10) | Benchmarks reveal lower accuracy on rare/dynamic facts or evolving corpora                                  |  (Yang et al.), (Chen et al. 17754-17762)|
| Security/privacy/retrieval poisoning are underexplored                | Evidence strength: Weak (2/10) | Few studies address adversarial robustness or privacy-preserving retrieval                                  |  (Brown et al.)|

**Figure undefined:** Key claims and support evidence identified in these papers.

## 5. Conclusion

Dynamic/adaptive Retrieval-Augmented Generation has become foundational to state-of-the-art performance in knowledge-intensive NLP by enabling flexible integration of external information based on real-time needs or query complexity. The field continues to evolve rapidly with innovations in agentic orchestration/multimodality/policy-driven control—but faces ongoing challenges around holistic evaluation/generalization/security.

### Research Gaps

| Topic/Outcome                  | Multi-hop QA      | Long-form Gen     | Multimodal        | Temporal Reasoning   | Agentic Planning    |
|-------------------------------|-------------------|-------------------|-------------------|---------------------|---------------------|
| Static Baseline                | **8**   | **6**   | **2**   | **1**     | **GAP**     |
| Dynamic Triggered              | **12**  | **8**   | **4**   | **2**     | **1**     |
| Hybrid Dense/Sparse            | **10**  | **6**   | **2**   | **GAP**     | **GAP**     |
| Graph/Event-centric            | **6**   | **2**   | **1**   | **4**     | **GAP**     |
| Agentic/Multi-agent            | **2**   | **1**   | **4**   | **GAP**     | **6**     |

**Figure undefined:** Research gaps matrix showing coverage by method/outcome.

### Open Research Questions

Future research should focus on unified evaluation frameworks that combine quality/cost/safety metrics across modalities/domains; robust generalization under domain shift/evolving corpora; scalable security/privacy solutions; and deeper integration of agentic planning/workflow orchestration.

| Question                                                                                                 | Why                                                                                                         |
|----------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| **How can unified benchmarks better capture cost-quality-safety tradeoffs across modalities?**      | Current evaluations rarely combine accuracy with efficiency or safety—holistic metrics are needed for real-world deployment.         |
| **What architectures best support robust generalization under domain shift or evolving corpora?**    | Many SOTA systems degrade outside training domains—understanding generalization is critical for reliability at scale.                |
| **How can security/privacy be ensured against adversarial attacks or data leakage in dynamic RAG?**  | As adoption grows in sensitive domains (medicine/law/etc.), privacy-preserving retrieval becomes essential but is underexplored.      |

**Figure undefined:** Open questions table highlighting future research directions.

In summary: Dynamic/adaptive RAG is now central to SOTA NLP—yet major opportunities remain around unified evaluation frameworks/generalization/security as the field moves toward robust deployment at scale.
 
_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent._
 
## References
 
Abootorabi, Mohammad Mahdi, et al. "Ask in Any Modality: A Comprehensive Survey on Multimodal Retrieval-Augmented Generation." **, 2025, pp. 16776-16809. https://doi.org/10.48550/arxiv.2502.08826
 
Brown, Andrew, et al. "A Systematic Literature Review of Retrieval-Augmented Generation: Techniques, Metrics, and Challenges." *ArXiv*, vol. abs/2508.06401, 2025. https://doi.org/10.48550/arxiv.2508.06401
 
Chen, Jiawei, et al. "Benchmarking Large Language Models in Retrieval-Augmented Generation." **, 2023, pp. 17754-17762. https://doi.org/10.48550/arxiv.2309.01431
 
Gao, Yunfan, et al. "Retrieval-Augmented Generation for Large Language Models: A Survey." *ArXiv*, vol. abs/2312.10997, 2023.
 
Guo, Hanghui, et al. "DioR: Adaptive Cognitive Detection and Contextual Retrieval Optimization for Dynamic Retrieval-Augmented Generation." *ArXiv*, vol. abs/2504.10198, 2025. https://doi.org/10.48550/arxiv.2504.10198
 
Guti'errez, Bernal Jim'enez, et al. "From RAG to Memory: Non-Parametric Continual Learning for Large Language Models." *ArXiv*, vol. abs/2502.14802, 2025. https://doi.org/10.48550/arxiv.2502.14802
 
Hei, Zijian, et al. "DR-RAG: Applying Dynamic Document Relevance to Retrieval-Augmented Generation for Question-Answering." *ArXiv*, vol. abs/2406.07348, 2024. https://doi.org/10.48550/arxiv.2406.07348
 
Huang, Jie, et al. "Layered Query Retrieval: An Adaptive Framework for Retrieval-Augmented Generation in Complex Question Answering for Large Language Models." *Applied Sciences*, 2024. https://doi.org/10.3390/app142311014
 
Jang, Jisoo, and Wen-Syan Li. "AU-RAG: Agent-based Universal Retrieval Augmented Generation." *Proceedings of the 2024 Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region*, 2024. https://doi.org/10.1145/3673791.3698416
 
Jiang, Zhuohang, et al. "QA-Dragon: Query-Aware Dynamic RAG System for Knowledge-Intensive Visual Question Answering." *ArXiv*, vol. abs/2508.05197, 2025. https://doi.org/10.48550/arxiv.2508.05197
 
Li, Yangning, et al. "Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Planning Agent." *ArXiv*, vol. abs/2411.02937, 2024. https://doi.org/10.48550/arxiv.2411.02937
 
Liu, Pei, et al. "HM-RAG: Hierarchical Multi-Agent Multimodal Retrieval Augmented Generation." *Proceedings of the 33rd ACM International Conference on Multimedia*, 2025. https://doi.org/10.1145/3746027.3754761
 
Ruangtanusak, Saksorn, et al. "DoTA-RAG: Dynamic of Thought Aggregation RAG." *ArXiv*, vol. abs/2506.12571, 2025. https://doi.org/10.48550/arxiv.2506.12571
 
Sharma, C. "Retrieval-Augmented Generation: A Comprehensive Survey of Architectures, Enhancements, and Robustness Frontiers." *ArXiv*, vol. abs/2506.00054, 2025. https://doi.org/10.48550/arxiv.2506.00054
 
Singh, Aditi, et al. "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG." *ArXiv*, vol. abs/2501.09136, 2025. https://doi.org/10.48550/arxiv.2501.09136
 
Srinivas, Sakhinana Sagar, and Venkataramana Runkana. "Scaling Test-Time Inference with Policy-Optimized, Dynamic Retrieval-Augmented Generation via KV Caching and Decoding." *ArXiv*, vol. abs/2504.01281, 2025. https://doi.org/10.48550/arxiv.2504.01281
 
Su, Weihang, et al. "Dynamic and Parametric Retrieval-Augmented Generation." *Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval*, 2025. https://doi.org/10.1145/3726302.3731692
 
Su, Weihang, et al. "DRAGIN: Dynamic Retrieval Augmented Generation based on the Real-time Information Needs of Large Language Models." **, 2024, pp. 12991-13013. https://doi.org/10.48550/arxiv.2403.10081
 
Sun, Jiashuo, et al. "DynamicRAG: Leveraging Outputs of Large Language Model as Feedback for Dynamic Reranking in Retrieval-Augmented Generation." *ArXiv*, vol. abs/2505.07233, 2025. https://doi.org/10.48550/arxiv.2505.07233
 
Sun, Qingyun, et al. "DyG-RAG: Dynamic Graph Retrieval-Augmented Generation with Event-Centric Reasoning." *ArXiv*, vol. abs/2507.13396, 2025. https://doi.org/10.48550/arxiv.2507.13396
 
Tan, Yuqiao, et al. "Dynamic Parametric Retrieval Augmented Generation for Test-time Knowledge Enhancement." **, 2025.
 
Wang, Qiuchen, et al. "ViDoRAG: Visual Document Retrieval-Augmented Generation via Dynamic Iterative Reasoning Agents." **, 2025, pp. 9113-9134. https://doi.org/10.48550/arxiv.2502.18017
 
Xi, Yunjia, et al. "InfoDeepSeek: Benchmarking Agentic Information Seeking for Retrieval-Augmented Generation." *ArXiv*, vol. abs/2505.15872, 2025. https://doi.org/10.48550/arxiv.2505.15872
 
Yang, Xiao, et al. "CRAG - Comprehensive RAG Benchmark." *ArXiv*, vol. abs/2406.04744, 2024. https://doi.org/10.48550/arxiv.2406.04744
 
Zhang, Feiyuan, et al. "DH-RAG: A Dynamic Historical Context-Powered Retrieval-Augmented Generation Method for Multi-Turn Dialogue." *ArXiv*, vol. abs/2502.13847, 2025. https://doi.org/10.48550/arxiv.2502.13847
 
Zhang, Qinggang, et al. "A Survey of Graph Retrieval-Augmented Generation for Customized Large Language Models." *ArXiv*, vol. abs/2501.13958, 2025. https://doi.org/10.48550/arxiv.2501.13958
 
