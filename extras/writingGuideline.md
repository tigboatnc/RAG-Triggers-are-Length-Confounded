- - - 
## Rules to follow 
> Dont mention these where they are being used but follow ! IMPORTANT

**#CA (compound adjectives)**
Hyphenate compound words that are used as an adjective.  For example, in "open-book exam", the phrase "open book" is used as an adjective for the noun "exam."  These hyphens are used much like parentheses; "open-book exam" is read like "((open book) exam), whereas "open book exam" is read "(open (book exam))", not the likely meaning.  Exception: no hyphen after a word ending in **ly**.

**#CN (citations are not nouns)**
Citation references are not nouns; they are parenthetical remarks.  Thus, it is not correct to write "In [13] they show that P=NP." You should write "A seminal paper proved P=NP [13]."  Or, "Jones and Smith show that P=NP [13]."  The reference can go at the end of the sentence (my preference) or at the end of the relevant phrase (sometimes better when multiple refs are cited in the same sentence).  See also #ET.

**#CQ (curly quotes)**
In LaTeX, quotation marks are written in pairs using backticks (to open quote) and forward ticks (to close quote), `like this' for single quotes and ``like this'' for double quotes. The double-quote character (") should not be used in LaTeX source, as it always formats as a right-curly double quote.

**#H (however)**
"However" should (usually) not begin a sentence: rewrite "However, I found that the red ball had been missing for weeks" as "I found, however, that the red ball had been missing for weeks".  In that usage, note "however" is surrounded by commas.  It is ok for it to be at the end of a sentence, however.  Above, I say "usually" because this is not a strict rule.

**#P (passive)**
Avoid the passive sentence structure.  It obscures the subject of the sentence, and leads to ambiguity.  For example, "The prototype was built" is a passive phrase whereas "We built a prototype" is an active phrase.  The active structure is almost always shorter, and clearer.

**#T (tense)**
- Scientific writing uses tense in specific ways.
	- It is common in scientific research papers to use the **first-person plural**, that is, to write "We developed a method..." rather than "I developed a method...", even on a single-authored paper.  The rationale I've heard is that the plural recognizes that science is normally a collaborative process.
	- When referring to your own experimental results, use the **past tense**.  Thus, "We _ran_ ten trials and the average execution _was_ 10.4 seconds." Past tense to say you did the work, and past tense to describe the result.
	- When writing about related work, describe their results in the **present tense**.  Thus, "Jones et al. _conducted_ a survey and _found_ that most Dartmouth students _wear_ green clothing."  Past tense to say they did the work, but present tense to describe their result.
	- When referring to other parts of your paper, use **present tense**.  That is, do not say, "This paper will discuss...", say, "This paper discusses...".  Similarly, say, "The argument above proves that..." rather than "The argument above proved that...".  Why?  Because, despite the fact that your reader is reading through the paper, over time, the paper stands complete, in the present.  The argument not only proved, but still proves....

**#WN (whether or not)**
It is rarely appropriate to say "whether or not"; usually you should just say "whether".  If you do use "whether or not", don't spread the words across the sentence.

**#NB (not... but)**

Avoid the "not just X, but Y" or "not only X, but also Y" sentence structure. This rhetorical device (amplification) is overused by LLMs to manufacture nuance. It often adds word count without adding value. Instead of saying "Our method is not just fast, but also accurate," simply say "Our method is fast and accurate" or "Our method achieves high accuracy with low latency."

**#ED (em-dashes)**

Restrict the use of em-dashes (—). While grammatically valid, LLMs overuse them to insert appositives or dramatic pauses in every paragraph. In formal scientific writing, parenthetical statements should usually go in parentheses (...), or the sentence should be split. If you see more than one em-dash per page, you are likely overusing them.

**#GS (grandstanding)**

Avoid subjective intensifiers and "hype" adjectives. Words like "meticulous," "paramount," "pivotal," "cutting-edge," and "unprecedented" are subjective. LLMs use them to sound authoritative; scientists avoid them to sound objective. Do not tell the reader your result is "remarkable"; show the data and let them decide.

**#TW (transition words)**

Do not start sentences with "Additionally," "Moreover," "Furthermore," or "In conclusion" unless absolutely necessary. LLMs use these to stitch independent thoughts together artificially. Good writing flows logically from one sentence to the next without needing a "glue" word at the start of every sentence.

**#BV (banned vocabulary)**

Avoid specific words that are statistical anomalies in AI writing but rare in human writing.

- **Delve:** Do not write "We delve into..." (Use "We investigate" or "We analyze").
- **Underscore/Highlight:** Do not write "This underscores the importance..." (Use "This demonstrates" or "This suggests").
- **Landscape/Realm:** Do not write "In the rapidly evolving landscape of..." (Be specific: "In recent mHealth studies...").
- **Nuance:** Avoid describing things as "nuanced" without explaining exactly _what_ the nuance is.

Here is a literature review summary structured to help you write Section 2.4, organized by the specific "beats" you requested. I have provided phrasing you can adapt directly into your draft, along with the specific citations from the provided source documents.

### 2.4 Deep Learning Approaches to Handle Time-to-Event Analysis

This section establishes the progression from traditional survival models to deep learning, then to Transformers, and finally to the specific PatchTST architecture relevant to your PRISM framework.

#### Beat 1: Survival and Time-to-Event Handled by Deep Learning

**Summary:** Traditional models like Cox Proportional Hazards often struggle with the complex, non-linear, and time-varying nature of mobile sensing data. Deep learning models have evolved to address this, moving from simple MLPs (DeepSurv) to architectures that handle competing risks (DeepHit) and longitudinal data (Dynamic-DeepHit). Recent work has specifically applied these concepts to notification management.

- **Phrasing:** "While traditional statistical methods like the Cox Proportional Hazards model have laid the groundwork for survival analysis, they are often limited by linearity assumptions and difficulties in modeling complex temporal dependencies. To address this, deep learning approaches have been introduced to capture non-linear relationships between covariates and hazard rates. Notably, DeepHit employs a multi-task neural network to directly learn the joint distribution of survival times and competing events, effectively handling right-censored data without relying on rigid parametric assumptions."
    - **Cite:** DeepHit learns the distribution of first hitting times directly and handles competing risks smoothly.
- **Phrasing:** "Building on this, Dynamic-DeepHit extends the architecture to longitudinal environments using Recurrent Neural Networks (RNNs). This approach is particularly relevant for mobile sensing, as it dynamically updates survival predictions as new measurements (e.g., biomarkers or sensor readings) become available, effectively creating a 'history' of user state."
    - **Cite:** Dynamic-DeepHit uses RNNs to encode longitudinal history into a context vector for dynamic risk prediction.
- **Phrasing:** "In the specific domain of mobile notification management, recent frameworks like DeepPvisit have utilized probabilistic deep learning to model the 'delta effect' or the time-to-visit after a notification is sent. By reparameterizing distributions like Weibull through deep neural networks, these models can better capture the latent, non-linear interactions between user context and engagement than standard regression baselines."
    - **Cite:** DeepPvisit uses deep learning to estimate time-to-event for user engagement, outperforming standard survival regression.

#### Beat 2: Transformers and Mobile Sensing

**Summary:** Transformers are replacing RNNs in sequential modeling due to their ability to capture long-range dependencies and process data in parallel. In mobile sensing (e.g., Human Activity Recognition), Transformers have shown superior robustness to data heterogeneity and distribution shifts compared to CNNs and LSTMs.

- **Phrasing:** "While RNNs and LSTMs have traditionally been the standard for sequential sensing tasks, they often struggle with long-term dependencies and sequential processing bottlenecks. Consequently, Transformer architectures have been adapted for mobile and wearable sensing. For instance, the Human Activity Recognition Transformer (HART) demonstrates that attention mechanisms can dynamically weight specific sensor segments (e.g., acceleration peaks) to improve classification accuracy, achieving high robustness even across heterogeneous device placements."
    - **Cite:** HART adapts Transformers for IMU sensing, using multi-headed self-attention to weight relevant signal segments dynamically.
- **Phrasing:** "Furthermore, Transformers are increasingly being applied directly to survival analysis (e.g., SurvFormer, BTSA). These models utilize self-attention to capture the interactions between covariates over time without the discretization errors common in MLP-based survival models. By processing sequences in parallel, these architectures offer a more efficient mechanism for modeling the complex, longitudinal evolution of user behavior found in mHealth datasets."
    - **Cite:** SurvFormer uses a Transformer decoder to capture inter-dependencies between temporal survival states; Transformer-based survival models outperform fully connected networks by modeling patient-specific survival distributions.

#### Beat 3: PatchTST and Sensing

**Summary:** A critical innovation for applying Transformers to time-series/sensing is "Patching." PatchTST (Patch Time Series Transformer) segments data into sub-series (patches) rather than individual time steps. This reduces computational complexity, preserves local semantic meaning (crucial for physical sensor data), and maintains channel independence, which prevents noise from one sensor corrupting others.

- **Phrasing:** "A significant challenge in applying Transformers to high-frequency mobile sensing data is the lack of semantic meaning in single time-steps and the quadratic complexity of attention mechanisms. The PatchTST architecture addresses this by segmenting time series into sub-series level 'patches' which serve as input tokens. This patching design retains local semantic information (e.g., a complete gesture or behavioral motif) that is often lost in point-wise attention."
    - **Cite:** Patching aggregates time steps into subseries-level patches to enhance locality and capture comprehensive semantic information.
- **Phrasing:** "Critically for multivariate mobile sensing, PatchTST enforces channel independence, where each sensor stream (e.g., accelerometer vs. battery level) is processed by the Transformer backbone in isolation with shared weights. This design has proven effective in sensing tasks such as wrist-motion recognition and wind power forecasting, as it prevents high-frequency noise from unstable sensors from corrupting the embeddings of robust signals, while significantly reducing memory usage to allow for longer historical context windows."
    - **Cite:** PatchTST reduces memory usage quadratically $O((L/S)^2)$ allowing for longer historical look-back windows; It has been successfully applied to motion recognition by capturing local semantic features and enhancing noise resilience; It is also effective in other time-series domains like wind power prediction by extracting local features.


