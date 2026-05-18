

I have enough — here's the curated list, organized by relevance to your paper. I marked decision certainty honestly.

---

# OpenReview links for your reference papers

## Length-confound + evaluation-methodology lineage (the §1.2 / §2.2 cluster)

1. [ICLR 2026 Poster, Accept, **Addressing Pitfalls in the Evaluation of Uncertainty Estimation Methods for Natural Language Generation** — Ielanskyi et al.](https://openreview.net/forum?id=OxWnOV5q8w)
2. [ICLR 2025 Workshop (SCSL), earlier version of the above](https://openreview.net/forum?id=hbIpwrdfE1)
3. [SafeGenAI Workshop @ NeurIPS 2024, Poster, **On a Spurious Interaction between Uncertainty Scores and Answer Evaluation Metrics in Generative QA Tasks** — earlier version of Santilli's "Revisiting UQ Evaluation" (ACL 2025 short)](https://openreview.net/forum?id=jGtL0JFdeD)
4. [ACL ARR 2025 → EMNLP 2025 Main, Accept, **The Illusion of Progress: Re-evaluating Hallucination Detection in LLMs** — Janiak et al.](https://openreview.net/forum?id=izvcKVccHm)

## Step-level answer-trajectory primitives (the §1.3 / §2.3 cluster)

5. [ACL ARR → NAACL 2025 Findings, Accept, **Chain-of-Probe: Examining the Necessity and Accuracy of CoT Step-by-Step** — Wang Zezhong et al.](https://openreview.net/forum?id=CxiNICq0Rr)

## Stepwise / streaming uncertainty in CoT (the §2.4 cluster)

6. [ACL ARR 2026 January Cycle, **Under Review** (decision Aug 20, 2026), **Streaming Hallucination Detection in Long Chain-of-Thought Reasoning** — Lu et al.](https://openreview.net/forum?id=y46kwdz3oQ)
7. [ACL ARR 2026 January Cycle, **Under Review** (decision Aug 20, 2026), **Confidence over Time: Confidence Calibration with Temporal Logic for Large Language Model Reasoning** — Mao et al.](https://openreview.net/forum?id=GM0WK3dzDG)

## Dynamic-RAG trigger pool (§2.1)

8. [ICLR 2024 Oral, Accept, **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection** — Asai et al.](https://openreview.net/forum?id=hSyW5go0v8)
9. [ICLR 2024 Poster, Accept, **INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection** — Chen et al. (the EigenScore paper)](https://openreview.net/forum?id=Zj12nzlQbz)
10. [EMNLP 2023 (likely preprint-mirror), **Active Retrieval Augmented Generation (FLARE)** — Jiang et al.](https://openreview.net/forum?id=WLZX3et7VT)
11. [ICLR 2025, **Reject** (likely — paper subsequently accepted at ACL 2025), **SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation** — Yao et al.](https://openreview.net/forum?id=NhIaRz9Qf5)
12. [ICLR 2025, **Reject** (likely — paper subsequently accepted at ACL Findings 2025), **CtrlA: Adaptive Retrieval-Augmented Generation via Inherent Control** — Liu et al.](https://openreview.net/forum?id=QYvtX2XA8p)
13. [CoRR 2024 (earlier version of CtrlA), **Probe-Guided Control**](https://openreview.net/forum?id=xmXHD6hQov)
14. [ACL ARR → ACL 2025 Long, Accept, **DioR: Adaptive Cognitive Detection and Contextual Retrieval Optimization for Dynamic Retrieval-Augmented Generation** — Guo et al.](https://openreview.net/forum?id=SYC1AxRktG)
15. [ACL ARR → NAACL 2024, Accept, **Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity** — Jeong et al.](https://openreview.net/forum?id=eqlRvIeOyYzc)

## Adjacent (same trigger-evaluation framing as DTR, useful for reviewer-defense)

16. [ACL ARR submission, **To Retrieve or Not to Retrieve? Uncertainty Detection for Dynamic Retrieval Augmented Generation** — concurrent with Chen DTR, useful adjacent reference](https://openreview.net/forum?id=nTcpR9ls01)

## Not on OpenReview (for your awareness — these papers exist but the venues didn't use OpenReview)

These are heavily cited in your paper but the venues didn't go through OpenReview, so no reviews available:

- **Santilli "Revisiting UQ Evaluation"** ACL 2025 short — only the SafeGenAI workshop earlier version (#3 above) is on OpenReview
- **Vashurin "UNCERTAINTY-LINE"** EMNLP 2025 — went through ARR but I couldn't find the OpenReview ID
- **DRAGIN** ACL 2024 — not on OpenReview (ACL 2024 didn't use it)
- **Verbosity ≠ Veracity** ACL 2025 UncertaiNLP workshop — workshop paper, no OpenReview review
- **Datta "Decide Early"** ICML 2026 — arXiv only (or under-review without public reviews)
- **Chen "Dynamic Instability"** — arXiv only, no venue committed yet
- **Zhang ARS** — arXiv only (I checked `5kKH4nxSxB` — that's a different paper, "Evidence-Free Claim Verification")
- **Mallen 2023, IRCoT 2023, HotpotQA 2018, MuSiQue 2022** — too old / wrong venues for OpenReview

## What's most worth reading first

For pre-submission strategy:

1. **`OxWnOV5q8w` (Ielanskyi)** — accepted at ICLR 2026 Poster, which is the closest venue+lineage match to where your paper lives. Read the reviews to understand what reviewers in this space ask for.
2. **`izvcKVccHm` (Janiak)** — accepted at EMNLP 2025 main, the exact venue you're submitting to, exact lineage. Highest-signal reviews you can read.
3. **`y46kwdz3oQ` + `GM0WK3dzDG` (Lu + Mao)** — both ARR 2026 January submissions targeting EMNLP 2026; these are LITERALLY your direct competitors for the same cycle. The reviews (once visible) tell you exactly how reviewers in your slot are calibrated.
4. **`NhIaRz9Qf5` (SeaKR)** + **`QYvtX2XA8p` (CtrlA)** — both went ICLR 2025 → ACL 2025. Reading why ICLR rejected them is a cheap lesson about ICLR-vs-ACL reviewer differences for RAG-trigger papers.

For the rest, the reviews are useful background but not load-bearing.

