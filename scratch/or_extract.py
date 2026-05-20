"""Extract review patterns relevant to user's paper themes:
1. Length / verbosity confound complaints
2. AUROC / evaluation protocol complaints
3. Novelty / incremental complaints
4. Dataset coverage complaints
5. Multiple-testing / statistical complaints
6. Baseline coverage complaints (LLM-as-judge, length baselines)
7. Writing / presentation complaints
"""
import json, re

d = json.load(open('scratch/or_reviews2.json'))

# Keyword buckets
BUCKETS = {
    'length_verbosity': [r'\blength\b', r'verbos', r'long(er)? (output|generation|response)', r'token count'],
    'auroc_eval': [r'\bAUROC\b', r'\bAUC\b', r'evaluation protocol', r'\bmetric\b', r'\bcorrelation\b', r'rank(ing)?'],
    'novelty_incremental': [r'\bnovelty\b', r'novel(?:\s|,|\.|-)', r'incremental', r'\bcontribution\b', r'closely related', r'difference(s)? (from|with|to)', r'prior work', r'similar to'],
    'datasets_coverage': [r'limited (datasets|models|tasks)', r'\bgeneraliz', r'one dataset', r'broader', r'more (datasets|models|tasks)', r'narrow'],
    'multitest_stats': [r'multiple (test|comparison)', r'Bonferroni', r'p[- ]value', r'significance', r'confidence interval', r'\bCI\b', r'\bsignifican'],
    'baseline_coverage': [r'baseline', r'compare (with|to|against)', r'missing (a |the )?compari', r'simple baseline', r'length(\s+)?baseline'],
    'writing_clarity': [r'\bwriting\b', r'clarity', r'unclear', r'hard to (read|follow|understand)', r'confusing', r'\bnotation\b', r'\bsection\b.*(unclear|hard)', r'presentation', r'\bfigure\b.*(unclear|hard)'],
    'reproducibility': [r'reproduc', r'release.*code', r'\bcode\b'],
    'llm_judge_critique': [r'LLM[- ]as[- ]judge', r'GPT-4', r'judge', r'\bbias\b'],
    'causal_mechanism': [r'why (is|does|the)', r'mechanism', r'causal', r'explain why', r'understand why'],
}

REGEX = {k: [re.compile(p, re.IGNORECASE) for p in pats] for k, pats in BUCKETS.items()}

def classify(text):
    if not text:
        return set()
    hits = set()
    for k, regs in REGEX.items():
        for r in regs:
            if r.search(text):
                hits.add(k)
                break
    return hits


SUMMARY = {}
for paper, info in d.items():
    SUMMARY[paper] = {
        'title': info['title'],
        'reviews_count': len(info['reviews']),
        'ratings': [r['rating'] for r in info['reviews']],
        'bucket_hits': {k: 0 for k in BUCKETS},
        'sample_quotes': {k: [] for k in BUCKETS},
    }
    for r in info['reviews']:
        text = '\n\n'.join([r.get('weaknesses',''), r.get('questions',''), r.get('strengths','')])
        hits = classify(text)
        for h in hits:
            SUMMARY[paper]['bucket_hits'][h] += 1

        # Extract some quotes per bucket from weaknesses
        wk = r.get('weaknesses','')
        if wk:
            sentences = re.split(r'(?<=[.!?])\s+', wk)
            for sent in sentences:
                for k, regs in REGEX.items():
                    if any(reg.search(sent) for reg in regs):
                        if len(SUMMARY[paper]['sample_quotes'][k]) < 3 and 50 <= len(sent) <= 350:
                            SUMMARY[paper]['sample_quotes'][k].append(sent.strip())

# Print summary table
print('\n\n# Bucket hits per paper (counts: how many of N reviews mention this concern)\n')
header = ['paper(reviews)'] + list(BUCKETS.keys())
print(' | '.join(['%-28s' % header[0]] + ['%-7s' % h[:7] for h in header[1:]]))
print('-' * 180)
for p, s in SUMMARY.items():
    row = [f"{p[:26]}({s['reviews_count']})"]
    for b in BUCKETS:
        row.append('%-7d' % s['bucket_hits'][b])
    print(' | '.join(['%-28s' % row[0]] + ['%-7s' % v for v in row[1:]]))

# Save full
with open('scratch/or_extracted.json', 'w') as f:
    json.dump(SUMMARY, f, indent=2)
print('\nSaved scratch/or_extracted.json')
