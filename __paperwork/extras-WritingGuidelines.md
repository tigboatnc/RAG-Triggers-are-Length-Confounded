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