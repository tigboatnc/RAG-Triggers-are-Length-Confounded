# What's actually going on in this paper — for an intern

> A long, plain-language tour of dynamic RAG, the trigger zoo, the datasets, the `lift_em` pivot, what a model actually outputs, and why we needed `n=800` traces and 10,000 bootstrap samples to call anything significant.
>
> Every number and every example below is pulled from real log files in this repo. File paths shown next to claims.

---

## 1. Dynamic RAG: what it is and what it does

### The big picture

Normal RAG (retrieval-augmented generation) does this:

1. User asks a question.
2. System retrieves some documents from a database.
3. System stuffs the documents into the prompt and asks the LLM to answer.

**Dynamic** RAG adds one extra trick: instead of retrieving *once* up front, the system can pause mid-answer and decide "wait, I should look something up here" and then go fetch more documents on the fly. The decision-maker is called the **trigger**: a scalar function that looks at what the model is currently saying and outputs a number; if the number is high, fire retrieval.

The question this paper asks: **which trigger functions actually work? And do their reported wins survive a fair test?**

### A worked example, end to end

Here is a real trace from our headline experiment. Question from HotpotQA:

> *"Mummulgum is a stop on the Bruxner Highway before reaching the town on the banks of which river?"*
>
> **Gold answer:** Richmond  
> (`example_trace.json`, QID `5abd01205542992ac4f3819b`)

The model is Qwen2.5-7B-Instruct. It does **closed-book** decoding (no retrieval, just its own knowledge). It thinks step by step:

```
Step 1: Identify Mummulgum's location on the Bruxner Highway.
Step 2: Determine the next town downstream on the Richmond River.
Step 3: Confirm Casino is the town on the banks of the Richmond River.
Answer: Casino
```

So the model:

- Took 53 tokens.
- Produced 3 reasoning steps.
- Concluded "Casino".
- The gold answer is "Richmond".
- **EM (exact match) = 0** → wrong.

Notice: the model *mentions* "Richmond River" in Step 2, then somehow lands on "Casino" as the town. That kind of mid-reasoning slip is exactly what a good trigger should catch — if the system could pause after Step 1 and pull up a Wikipedia paragraph about Mummulgum, it would probably get the answer right.

When we then re-run the same question with the **gold supporting paragraphs** stuffed into the prompt (this is the "oracle" decode), the model answers "Richmond" correctly. So we know retrieval *would have* helped this trace. We label this trace `lift_em = +1` (more on this label in §3).

### Where does the trigger signal come from?

At each reasoning step boundary, the system records a lot of stuff about what the model is doing. Here is Step 1 of the trace above:

| field | value | what it means |
|---|---|---|
| `mean_token_entropy` | 0.449 | average uncertainty across all tokens in the step |
| `boundary_token_entropy` | 0.708 | uncertainty at the very last token |
| `dragin_rind_max` | 2.348 | DRAGIN's information-need score (high = retrieve!) |
| `u_cons` | 0.273 | how inconsistent K=5 stochastic re-samples of this step are (high = uncertain) |
| `per_token_max_logprob` | [0.70, 0.68, 1.00, ... 0.49 ...] | per-token confidence; low spots are where the model wasn't sure |

Plus we sample **K=5 alternative continuations** from this same boundary. For Step 2, the five samples were:

```
K=1: " Step 3: Confirm Casino is the town on the banks of the Richmond River. Answer: Casino"
K=2: " Step 3: Confirm Casino is the town on the banks of the Richmond River. Answer: Casino"
K=3: " Step 3: Confirm Casino is the town on the Richmond River. Answer: Casino"
K=4: " Step 3: Confirm Casino is the town on the Richmond River after Mummulgum. Answer: Richmond River"
K=5: " Step 3: Confirm Casino is the town on the banks of the Richmond River. Answer: Casino"
```

4 out of 5 samples say "Casino", 1 says "Richmond River". This is the kind of disagreement that the answer-trajectory signals (more on those in §2.5) look for. If the model can't even agree with itself, that's a clue retrieval might help.

Every trigger function we evaluate is a way of squeezing one of these per-step measurements (or the K=5 sample set) down to a single number per trace — and then we ask whether ranking traces by that number does a good job of identifying the traces where retrieval would have helped.

---

## 2. The trigger zoo: how each framework works

We audit **17 different signals** organized into **four families** (plus a length baseline). Here's what each one does in plain English, with a real number from the example trace where possible.

### 2.1 Length baselines (not really triggers — sanity checks)

- **`tokens_generated`** — just the number of tokens in the closed-book trace. (Our example: 53.)
- **`wall_clock_s`** — how long the decode took in seconds.
- **`n_steps`** — number of reasoning steps detected.

Why include these? Because if these "trivial" baselines are competitive with the published triggers, it tells us the published triggers might just be tracking length. **Spoiler: that turns out to be a real problem** — see §6.

### 2.2 Token-probability / logit-family triggers

These look at how confident the model is at each token. Low confidence = retrieve.

- **`ent.max`** ([Lin et al. 2024, "Generating with Confidence"]) — max token-level entropy over the trace. High entropy means the model's next-token distribution was flat and confused.

- **`flare.max`** ([FLARE; Jiang et al. EMNLP 2023]) — `1 − min(p)` where `p` is the predicted probability of the token actually chosen. If any token had probability < threshold, FLARE says retrieve. We report the score-level version (max over the trace) for apples-to-apples AUROC comparison.

- **`etc`** (entropy trajectory cumulant, second-order entropy change) — captures sudden entropy spikes.

These are computed straight from the model's logit outputs at decode time. Free with the forward pass.

### 2.3 Attention-conditioned triggers

These look at where the model is paying attention.

- **`dragin_rind_max`** ([DRAGIN; Su et al. ACL 2024]) — DRAGIN computes a "Real-time Information Need" (RIND) score per token by combining (a) token entropy, (b) self-attention magnitude over preceding context, and (c) a semantic-content mask. Tokens that are content words with high uncertainty *and* low attention to relevant context get a high RIND score. We take the max over the trace.

  In our example trace, Step 1 had `dragin_rind_max = 2.348`. DRAGIN fired on every step (`fire_rate = 1.0` — meaning the trigger would have said "retrieve!" at each boundary).

### 2.4 Multi-sample text-consistency triggers

These sample K alternative completions and measure how much they disagree.

- **`u_cons.max`** ([SelfCheckGPT-style; Manakul et al. EMNLP 2023]) — at each step boundary, sample K=5 continuations and measure their pairwise cosine *dis*similarity in some sentence-embedding space. High = the model would say different things if you re-rolled the dice. Means uncertain.

  In our example, Step 1 had `u_cons = 0.273` with `mean_pairwise_cos = 0.727` across the K=5 samples (decent but not perfect agreement).

- **`sc_ngram`** — same idea but on n-gram overlap of the K samples (cheap, no embedding model needed).

### 2.5 Hidden-state geometric triggers

These look at the model's internal activations.

- **`drift_cosine.max`** — cosine distance between this step's last-hidden-state and an early-trace reference.
- **`variance_l2_window.max`** — L2 spread of hidden states in a window.
- **`hidden_norm.range`** — span of hidden-state L2 norms across the trace.
- **`seakr_k5_proxy.logdet`** ([SeaKR; Yao et al. ACL 2025]) — at each step, sample K=20 (we use K=5 as a proxy) hidden-state vectors and compute the log-determinant of their Gram matrix. Large log-det = the model's hidden states are spread out across many directions = it's confused.
- **`hvr.max`** ([SELFDOUBT-T1; Kadavath et al. 2022]) — a hidden-state verifier.

### 2.6 The answer-trajectory family (this paper's main novelty class)

These are the new kid on the block, derived from [Datta et al. ICML 2026 — "LLMs decide early and explain later"]. The recipe:

1. At each reasoning-step boundary (we use spaCy sentence segmentation), sample K=5 alternative completions.
2. For each completion, extract its predicted answer using a regex on `"Answer: ..."`.
3. You now have a "trajectory" of K=5 majority answers, one per step: $\mathcal{M} = (m_1, m_2, \ldots, m_n)$.

From this trajectory we compute four trace-level numbers:

- **`flip_rate K=5`** — fraction of consecutive-step pairs where the majority answer disagreed. (Did the model change its mind mid-reasoning?)
- **`flip_rate K=1`** — same but using only the first stochastic sample at each step. Cheaper.
- **`late_flips`** — same as `flip_rate` but only counts flips in the last half of the trace (motivated by [Chen et al. 2026]'s finding that late instability is destructive while early instability is corrective).
- **`min_majority`** — `1 − min_t(max_count / K)`. The single worst step's within-step disagreement. (At which step did the K=5 samples agree the least?)
- **`ECT`** (early-commitment time, **our new signal**) — `t_commit / (n-1)` where `t_commit` is the earliest step from which the majority answer stays equal to the final answer for the rest of the trace. **Low ECT** = model committed early (closed-book trace will probably be right). **High ECT** = model deliberated until late (retrieval probably needed).

These signals operate on **answer semantics** (what the model would actually answer), not on token-probability or attention magnitude. That makes them different from everything else in the zoo.

---

## 3. The datasets

We use two multi-hop QA datasets. Both have the same format: each question requires combining information from 2+ Wikipedia paragraphs, and the dataset provides both the gold paragraphs ("supporting facts") and several distractors.

### 3.1 HotpotQA (in-domain)

[Yang et al. EMNLP 2018.] Two question types:
- **Bridge** questions: requires chaining facts ("X starred in Y, which is directed by Z; how old is Z?").
- **Comparison** questions: requires comparing two entities ("Was X or Y born earlier?").

A real record from `data/hotpotqa_distractor_500_v2test.jsonl`:

```
{
  "id": "5abd94525542992ac4f382d2",
  "question": "2014 S/S is the debut album of a South Korean boy group that was formed by who?",
  "answer": "YG Entertainment",
  "type": "bridge",
  "level": "hard",
  "paragraphs": [
    { "title": "List of awards and nominations received by Shinee",  <-- distractor
      "sentences": ["South Korean boy group Shinee have received...", ...] },
    { "title": "WINNER (band)",                                       <-- gold supporting
      "sentences": ["WINNER is a South Korean boy group formed in 2013 by YG Entertainment...", ...] },
    { "title": "2014 S/S",                                            <-- gold supporting
      "sentences": ["2014 S/S is the debut studio album of WINNER...", ...] },
    ... 7 more distractors ...
  ]
}
```

The model gets all ~10 paragraphs (gold + distractors) shoved into the prompt in the **closed-book** condition we use (which is a bit of a misnomer — it's actually closed-book *retrieval-wise*, but the distractor paragraphs are still in the prompt as part of the HotpotQA-distractor benchmark convention).

In the **oracle** condition, we strip out the 8 distractors and only feed in the 2 gold paragraphs.

### 3.2 2WikiMultiHopQA (out-of-domain)

[Ho et al. COLING 2020.] More question types — bridge, comparison, compositional, inference, bridge_comparison. Same structural format. We use this as the out-of-domain test set.

### 3.3 Sample slice

For each architecture (Qwen2.5-7B, Gemma-4-26B-A4B-it), each dataset (HotpotQA, 2WikiMHQA), we use **n=200 paired traces** per cell. Four cells, 800 total paired traces. Each "paired trace" is a (closed-book, oracle) pair on the same question — both must succeed (model produced a parseable answer) for the pair to count.

---

## 4. The assessment pivot: why we use `lift_em`, not `closed_em`

This is the single most important design decision in the paper. It deserves a careful walk-through.

### 4.1 The standard protocol (and what's wrong with it)

The standard way to evaluate a trigger is:

1. Run closed-book decoding on N questions.
2. Compute the trigger score for each trace.
3. Define the label as `closed_em = 0` (i.e., "the model got it wrong").
4. Compute AUROC: does the trigger rank wrong-answer traces above right-answer traces?

The intuition: triggers should fire on questions the model got wrong, because those are the ones retrieval might fix.

**The problem:** "the model got it wrong" is not the same as "retrieval would have helped". Some questions are just unanswerable from any reasonable document; the model would have been wrong regardless. A trigger that fires on those traces wastes compute (retrieves but doesn't help). A trigger that *doesn't* fire on those is actually being correct in a deployment sense.

### 4.2 The `lift_em` target

We define a paired-oracle label:

$$\texttt{lift\_em}(q) = \mathbb{1}[\mathrm{EM}(a_o) > \mathrm{EM}(a_c)]$$

That is: `lift_em = +1` if the oracle decode (with gold paragraphs) got the right answer **and** the closed-book decode got the wrong answer. Otherwise 0.

This label answers the operationally meaningful question: **on which traces would retrieval actually change the outcome from wrong to right?**

The trigger only needs to fire on the `lift_em = +1` traces. Firing on the `closed_em = 0, oracle_em = 0` traces (where the model is wrong but retrieval wouldn't help either) is wasted compute.

### 4.3 Real examples of helped pairs

Here are three actual helped pairs (`lift_em = +1`) from our Qwen2.5-7B × HotpotQA cell. Source: `runs/v2_headline_qwen2.5-7b-instruct/{D1_closed_hotpot,E1_oracle_hotpot}.jsonl`.

```
QID: 5abd01205542992ac4f3819b
  Q:      Mummulgum is a stop on the Bruxner Highway before reaching the town
          on the banks of which river?
  GOLD:   Richmond
  CLOSED: "Casino"     (em=0, tokens=53,  3 steps)
  ORACLE: "Richmond"   (em=1, tokens=95)

QID: 5ac1795f5542994ab5c67d5f
  Q:      Who is known for producing the documentary television series based
          on the book by science historian James Burke?
  GOLD:   James Burke
  CLOSED: "BBC Science and Features Department"  (em=0, tokens=117, 6 steps)
  ORACLE: "James Burke"                          (em=1, tokens=117)

QID: 5a83aaeb5542996488c2e483
  Q:      What studio album did Kanye West record with Roc-A-Fella Records
          and soul singer Dwele?
  GOLD:   My Beautiful Dark Twisted Fantasy
  CLOSED: "Graduation"                           (em=0, tokens=117, 4 steps)
  ORACLE: "My Beautiful Dark Twisted Fantasy"    (em=1, tokens=90)
```

These are the kind of traces where a good trigger should fire. Notice how the model often picks a *plausible but wrong* entity — "Casino" is a real town in the area, "Graduation" is a real Kanye album. The model is not flailing randomly; it is confidently wrong. A trigger that just looks at "is the model uncertain?" might miss these. A trigger that looks at "did the model's answer keep flipping during reasoning?" might catch them.

Out of 200 closed/oracle pairs on Qwen × HotpotQA, **31 were `lift_em = +1` helped pairs**. The full headline pool (Qwen + Gemma × HotpotQA + 2WikiMHQA, n=800) has **108 helped pairs**, base rate ≈ **13.5%**.

### 4.4 Why the paired design also defends us against the correctness-function critique

A separate critique in the recent literature (Janiak 2025, Ielanskyi 2025) is that EM (exact match) is a bad correctness function: it's overly strict on paraphrases (it would say "Tom Cruise" ≠ "the actor Tom Cruise"), and the field disagrees substantially on which correctness function to use.

Our paired design partially insulates against this: both the closed and the oracle arm use the *same* EM function. Any bias in EM (e.g., it's overly strict) affects both arms symmetrically. So a `lift_em = +1` label specifically catches traces where the *same* correctness function flipped from 0 to 1.

We also do a direct LLM-as-Judge check (Cohen's κ = 0.707 between EM and a GPT-4o-Mini judge; 8 of 9 survivors preserved under the judge label). See Appendix C of the paper.

---

## 5. Model outputs: what does a model actually produce, and what do we record?

Here's the structure of one full trace record (from `example_trace.json`, lightly abbreviated):

```json
{
  "qid": "5abd01205542992ac4f3819b",
  "closed": {
    "question": "Mummulgum is a stop on the Bruxner Highway...",
    "gold_answer": "Richmond",
    "model": "qwen2.5-7b-instruct",
    "prompt_tokens": 1132,
    "tokens_generated": 53,
    "wall_clock_s": 9.418,
    "n_steps": 3,
    "model_answer_extracted": "Casino",
    "em": 0,
    "f1": 0.0,

    "fire_counts": {           // how many steps each trigger said "retrieve!"
      "dragin": 3,             // DRAGIN fired on all 3 steps
      "etc": 0,                // entropy-trajectory said no
      "selfdoubt_t1": 0,
      "random": 0
    },
    "fire_rates": {            // fraction of steps that fired
      "dragin": 1.0,
      "etc": 0.0,
      "latent": 0.6667,
      "random": 0.2778
    },

    "steps": [
      { "step_idx": 1,
        "step_text": "Step 1: Identify Mummulgum's location...",
        "mean_token_entropy": 0.449,        // logit-family
        "boundary_token_entropy": 0.708,
        "dragin_rind_max": 2.348,           // attention-family
        "drift_cosine": 0.0,                // hidden-state-family
        "variance_l2_window": 0.0,
        "hidden_norm": 33.097,
        "u_cons": 0.273,                    // multi-sample text-consistency
        "per_token_max_logprob": [0.70, 0.68, 1.00, ...],
        "samples": [                        // K=5 boundary completions
          { "text": " Step 2: Determine the next town downstream...", "eos_hidden": [...] },
          { "text": " Step 2: Determine the next town after...", ... },
          ... 3 more ...
        ]
      },
      { "step_idx": 2, ... },
      { "step_idx": 3, ... }
    ]
  },
  "oracle": {
    "model_answer_extracted": "Richmond",
    "em": 1,
    ...  // (we don't extract triggers from the oracle trace; it's only for labeling)
  }
}
```

**Key things to internalize:**

1. The model's output isn't just a final answer — it's a *trace* with multiple reasoning steps, and we record everything for each step: entropies, hidden states, attention scores, K=5 alternative samples, etc.
2. A single trace produces ~10–20 different scalar signals (one per trigger we evaluate). They're all computed from the *closed-book* trace; the *oracle* trace is only used to compute the label.
3. The K=5 samples are expensive (they require K extra forward passes per step boundary). Most "free" signals (entropy, DRAGIN, hidden-state distances) come straight from the closed-book forward pass at no extra cost.

---

## 6. The final assessment: AUROC, 95% CI, Bonferroni, and why we need so many samples

This is the part most people get wrong, so worth taking slowly.

### 6.1 AUROC: the basic metric

For each signal $s$, we have a list of (signal value, label) pairs across all 800 paired traces. The label is `lift_em ∈ {0, 1}`. AUROC asks: **if I rank traces by the signal value, do the `lift_em = 1` traces tend to be ranked above the `lift_em = 0` traces?**

- AUROC = 1.0 → perfect discrimination.
- AUROC = 0.5 → random chance.
- AUROC = 0.0 → perfectly anti-predictive.

In our headline pool, the raw AUROCs look like this (from `analysis/moonshot_explore_3arch.out`, the n=800 pool):

```
signal                  rawAUC   resAUC   Δ vs tokens
tokens_generated         0.474    0.531   (baseline)
ECT                      0.674    0.691   +0.190 *
DRAGIN-RIND              0.634    0.640   +0.167 *
ent.max                  0.575    0.611   +0.137 *
min_majority             0.613    0.626   +0.149 *
flip_rate K=5            0.634    0.640   +0.135 *
SeaKR-K5.logdet          0.475    0.491   +0.018
seakr / hidden / etc...  (mostly ~0.5)
```

So ECT achieves raw AUROC 0.674, which is decent but not amazing. Why don't we just say "ECT wins, done"?

### 6.2 The length confound and why we residualize

Here's the awkward fact: `tokens_generated` (just the length of the trace) gets raw AUROC 0.474 on the headline pool. That's *below chance* — which means longer traces are *less* likely to be retrieval-helpful here (the Q×2Wiki and G×2Wiki cells have inverted length correlations).

But on other cells (and on the secondary `closed_em = 0` target), length is a strong predictor. Recent literature (Santilli 2025, Vashurin 2025, Janiak 2025, Zhang 2024, Ielanskyi 2025) shows that:

- LLMs systematically generate longer responses when uncertain ("verbosity compensation").
- Longer responses tend to be wrong more often on some QA benchmarks.
- Therefore, any signal that correlates with length will get partial AUROC credit even if it carries no actual uncertainty information.

A bunch of the published triggers (DRAGIN, FLARE, entropy-based scores) are correlated with trace length at Spearman ρ ∈ [+0.30, +0.45] (see Appendix F / `analysis/e7_spearman.out`). So when DRAGIN gets AUROC 0.634, the question is: how much of that is "DRAGIN is genuinely tracking retrieval-need" vs. "DRAGIN is tracking length, and length is tracking retrieval-need"?

**The fix:** regress each signal on `log(tokens + 1)` and use the OLS residual as the length-invariant signal. AUROC computed on the residual asks: *beyond what length explains*, does this signal still discriminate?

That's the `resAUC` column above. ECT goes from 0.674 raw → 0.691 residualized (it actually gets *better* once length is removed, because length is anti-correlated with the helpfulness target on the 2Wiki cells). DRAGIN goes from 0.634 → 0.640 (slight improvement). The hidden-state signals (drift_cosine, etc.) mostly stay at ~0.5 — they were never really discriminating in the first place; their raw AUROC numbers were length artifacts.

### 6.3 The paired-bootstrap test: is the difference real?

OK, ECT has residualized AUROC 0.691 vs. length's 0.531. The point estimate is +0.190. **Is that real, or could it be noise?**

We run a **paired bootstrap** with B=10,000 resamples. The procedure:

1. Take the 800 paired traces. Each trace has a signal-value, a length, and a `lift_em` label.
2. Resample 800 traces *with replacement* (some traces appear twice, some not at all).
3. On this bootstrap sample, compute the residualized AUROC of the signal *and* the AUROC of `tokens_generated` *on exactly the same resampled set*. The difference is one bootstrap estimate of Δ.
4. Repeat 10,000 times. You now have a distribution of 10,000 Δ values.
5. The 95% confidence interval is the [2.5%, 97.5%] percentiles of this distribution.

For ECT on the headline pool:

- Δ point estimate: +0.190
- 95% CI: [+0.093, +0.287]
- The lower bound is above 0, so we say "ECT is paired-significantly better than length at α=0.05".

From `analysis/e7_bonferroni.out`, the full headline table:

```
signal                  resAUC      Δ            95% CI               Bonferroni 99.67% CI   95%?  Bonf?
tokens_generated         0.531        —  (baseline, raw 0.474)
DRAGIN-RIND.max          0.640   +0.167  [+0.082,+0.249]   [+0.039,+0.286]      ✓     ✓✓
ent.max                  0.611   +0.137  [+0.050,+0.223]   [+0.001,+0.263]      ✓     ✓✓
min_majority             0.626   +0.149  [+0.054,+0.240]   [+0.004,+0.284]      ✓     ✓✓
ECT                      0.691   +0.190  [+0.093,+0.287]   [+0.047,+0.329]      ✓     ✓✓
flip_rate K=5            0.640   +0.135  [+0.037,+0.233]   [-0.012,+0.277]      ✓      —
flip_rate K=1            0.621   +0.116  [+0.016,+0.217]   [-0.030,+0.261]      ✓      —
u_cons.max               0.593   +0.120  [+0.028,+0.213]   [-0.017,+0.257]      ✓      —
etc (Δ²H max)            0.580   +0.107  [+0.018,+0.196]   [-0.026,+0.235]      ✓      —
M1 CABV                  0.641   +0.137  [+0.041,+0.229]   [-0.006,+0.273]      ✓      —
flare.max                0.553   +0.080  [-0.008,+0.165]   [-0.056,+0.207]      —      —
seakr_k5_proxy           0.491   +0.018  [-0.077,+0.112]   [-0.123,+0.161]      —      —
drift_cosine.max         0.493   +0.020  [-0.064,+0.101]   [-0.102,+0.142]      —      —
hidden_norm.range        0.499   +0.026  [-0.058,+0.108]   [-0.101,+0.146]      —      —
late_flips               0.586   +0.081  [-0.016,+0.175]   [-0.058,+0.219]      —      —
variance_l2_window       0.473   +0.000  [-0.091,+0.090]   [-0.136,+0.127]      —      —
hvr.max                  0.505   +0.032  [-0.089,+0.150]   [-0.147,+0.210]      —      —
```

**9 signals clear nominal 95% (✓). 4 clear Bonferroni (✓✓).**

### 6.4 Why Bonferroni? The multiple-testing problem

When you run a single test at α=0.05, you accept a 5% false-positive rate (5% chance of saying "this is real" when it's not). When you run **15 tests at α=0.05**, the chance that *at least one* of them gives a spurious "significant" result is much higher than 5%. With 15 independent tests, it's about 1 − (1 − 0.05)^15 ≈ **54%**. More than half!

Bonferroni's fix is simple: divide α by the number of tests. With k=15 signals, α becomes 0.05/15 = **0.0033** (the "99.67% CI" column). The criterion now is: the CI lower bound must be above 0 at the 99.67% level, not just 95%.

Bonferroni is conservative — it assumes the worst case (all tests are independent and you're trying to claim *at least one* is significant). Many real survivors that would pass with a smarter correction (like Benjamini-Hochberg FDR) get knocked out by Bonferroni. But it's bulletproof: if a signal survives Bonferroni, reviewers can't attack you on multiple-testing grounds.

Our four Bonferroni-survivors (ECT, DRAGIN-RIND, `min_majority`, `ent.max`) span four different signal families (answer-trajectory, attention, answer-trajectory, logit). This breadth matters: it shows the win isn't a fluke of one signal class.

### 6.5 Why do we need so many samples?

This is the part that bites every junior researcher. Two reasons.

**Reason 1: the base rate is tiny.**

Only 108 of our 800 paired traces are `lift_em = +1` (the "helped" cases). AUROC is fundamentally a comparison between the positive class and the negative class; the variance of an AUROC estimator goes as roughly 1 / sqrt(n_positive * n_negative). With only 108 positives, each AUROC point estimate already has a standard error of ~0.03-0.04, and the *difference* between two AUROCs has even more variance.

If we had n=200 traces (one cell), we'd have only ~27 positives. The standard error on Δ would be roughly twice as big. The 95% CIs would be roughly twice as wide. Almost nothing would clear.

**Reason 2: the differences we're chasing are small.**

The best signal (ECT) beats length by Δ = +0.190. That's modest in absolute terms. To detect a Δ of +0.10 reliably (say, with 80% statistical power at α=0.05), you need roughly **n = 4 × (1.96)² / (effect-size²)** for the relevant standard error formulation, which at our population sizes lands around n=500–800 positives. We have 108. Most of our marginal signals (`flip_rate K=1`, `u_cons.max`, etc.) are right on the edge — they clear nominal 95% but fail Bonferroni precisely because we don't have enough positives to nail them down with a tight enough CI.

This is also why the per-cell results are mostly silent (each cell has only ~20–35 positives, which is below the threshold where bootstrap CIs are reliable). Only the **pooled** results (n=800, 108 positives) are paired-significant.

**Reason 3: the bootstrap itself needs samples too.**

B=10,000 bootstrap resamples might sound like a lot, but it's set by the precision you want on the CI endpoints. With B=1,000, your 95% CI bounds have a standard error of ~0.002 from sampling noise alone, which is comparable to some of our smaller Δs. B=10,000 brings that down to ~0.0006, well below our effect sizes. Cheaper than re-running the experiment with more data.

### 6.6 The K=1 vs K=5 ablation, briefly

A separate question: do we need the K=5 multi-sample variant of the answer-trajectory triggers, or does K=1 work just as well? K=5 is 5× more expensive at inference time, so this matters for deployment.

From `analysis/a1_k1_vs_k5_r1_qwen.out`:

```
cell        n   pop  K1_raw  K5_raw   Δraw     Δraw 95% CI
qwen_hot    151  25   0.666   0.659  -0.006   [-0.062,+0.032]
qwen_wiki   154  22   0.579   0.582  +0.003   [-0.044,+0.056]
----------------------------------------------------------------
OVERALL     305  47   0.626   0.624  -0.002   [-0.040,+0.031]
```

Δ is essentially zero on both cells; the 95% CI for the overall Qwen pool is [-0.040, +0.031], comfortably straddling zero. Same picture on Gemma (Δ = +0.033, CI [-0.025, +0.092]).

**Verdict:** the K=5 variant offers no significant advantage on the `lift_em` target. We recommend K=1 for deployment because it's 5× cheaper at the same performance.

This is what we call a "regime-2" result — we pre-registered three possible outcomes (K=5 wins, K=5 ≈ K=1, K=1 wins) and this is the middle one. A null result is still useful! It saves anyone deploying these signals 80% of the compute.

---

## 7. The full picture in one paragraph

**Dynamic RAG needs a trigger — a function that says "retrieve now" mid-decode. The standard way to evaluate triggers is raw AUROC against a "model got it wrong" label, but recent work shows this is length-confounded (longer responses tend to be wrong, so anything correlated with length looks good). We re-evaluate 17 trigger signals on a deployment-relevant paired-oracle target (`lift_em = +1`: did retrieval flip a wrong answer to a right one?), residualizing each signal on log(tokens) to strip out length. On 800 paired traces across two open-weight chat models (Qwen2.5-7B dense, Gemma-4-26B-A4B-it MoE-hybrid) and two multi-hop QA datasets (HotpotQA, 2WikiMHQA), nine signals clear length at nominal 95% paired-bootstrap, four survive Bonferroni at k=15, and the strongest is ECT (Δ = +0.190 [+0.093, +0.287]), a forward-looking commitment-time signal in the answer-trajectory primitive class of Datta 2026. The K=5 multi-sample variant offers no advantage over K=1 at this task; deploy K=1. Under a GPT-4o-Mini semantic-equivalence judge instead of EM, ECT strengthens to Δ = +0.222 — the survivor set is not an artifact of any single correctness function.**

That's the paper.

---

## 8. Where to look in the repo

| Want to see... | Look at... |
|---|---|
| A full per-step trace with K=5 samples | `example_trace.json` |
| Closed-book decode log (Qwen × HotpotQA) | `runs/v2_headline_qwen2.5-7b-instruct/D1_closed_hotpot.jsonl` |
| Oracle decode log (same) | `runs/v2_headline_qwen2.5-7b-instruct/E1_oracle_hotpot.jsonl` |
| The headline survivor matrix output | `analysis/moonshot_explore_3arch.out` |
| The Bonferroni table | `analysis/e7_bonferroni.out` |
| The K=1 vs K=5 ablation log | `analysis/a1_k1_vs_k5_r1_qwen.out`, `..._gemma.out` |
| The Spearman length-vs-signal correlations | `analysis/e7_spearman.out` |
| The Uncertainty-Line reference replication | `analysis/uncertainty_line_audit_*.{out,json}` |
| The HotpotQA dataset slice | `data/hotpotqa_distractor_500_v2test.jsonl` |
| The forest plot for Figure 1 of the paper | `figs/survivor_forest.png` |
| The paper draft | `option1-lengthConfound/template.tex` |
| The corpus (long-form source-of-truth) | `option1-lengthConfound/option1-v2-corpus.tex` |

When in doubt, the corpus comment headers (top of `option1-v2-corpus.tex`) list every results-file path next to every claim in the paper.

