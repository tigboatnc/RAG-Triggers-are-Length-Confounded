#!/usr/bin/env python3
"""Fetch reviews focused on narrative-critique signal:
- What weaknesses (writing / framing / overclaim) do reviewers cite on QA/UQ/RAG short papers?
- What do accept-recommended reviewers cite as narrative strength?

Goal: extract patterns that an EMNLP 2026 reviewer would apply to a length-controlled
audit + answer-trajectory family + length-residualized AUROC paper.
"""

import json
import os
import re
import sys
from collections import defaultdict

import openreview

client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')

# Targeted papers in same neighborhood as ours
candidate_searches = [
    # length-confound family
    "length bias uncertainty quantification language model",
    "Vashurin uncertainty length invariant",
    "Santilli revisiting uncertainty quantification",
    # answer-trajectory / commitment-time family
    "answer trajectory step-level reasoning LLM",
    "decide early commitment large language model",
    "Chain of Probe necessity accuracy CoT",
    "dynamic instability LLM reasoning inference",
    "answer agreement representation shaping ICML",
    # dynamic RAG triggers
    "dynamic retrieval augmented generation trigger",
    "DRAGIN real time information needs",
    "active retrieval FLARE next token entropy",
    "SeaKR self aware knowledge retrieval",
    "INSIDE EigenScore internal states hallucination",
    "selective prediction retrieval question answering",
    # recent forecasted EMNLP / ARR short papers on related
    "length controlled evaluation question answering",
    "multi-hop QA retrieval trigger evaluation",
]

print("=" * 80)
print("STEP 1: Resolve forum IDs")
print("=" * 80)

resolved = {}
for q in candidate_searches:
    try:
        notes = client.search_notes(term=q, content={}, limit=4)
    except Exception as e:
        print(f"  Search ERROR ({q!r}): {e}", file=sys.stderr)
        continue
    for n_ in notes[:4]:
        title = n_.content.get('title', {})
        if isinstance(title, dict):
            title = title.get('value', '?')
        venue = n_.content.get('venue', {})
        if isinstance(venue, dict):
            venue = venue.get('value', '?')
        forum = n_.forum if hasattr(n_, 'forum') else n_.id
        if title and title != '?' and forum not in resolved:
            resolved[forum] = (title, venue, q)
            print(f"  + [{venue}] {title[:90]} | forum={forum}")
            break

print(f"\n\nTotal: {len(resolved)} forums")

print("\n" + "=" * 80)
print("STEP 2: Fetch reviews (only weakness/strengths/soundness/rating fields)")
print("=" * 80)


def fetch_reviews(forum_id):
    review_keys = [
        'summary', 'summary_of_strengths', 'summary_of_weaknesses',
        'strengths', 'weaknesses', 'soundness', 'excitement',
        'overall', 'rating', 'recommendation', 'overall_assessment',
        'reasons_to_accept', 'reasons_to_reject',
        'soundness_score', 'overall_score',
    ]
    out = []
    try:
        notes = client.get_notes(forum=forum_id)
        for n_ in notes:
            content = n_.content
            keys_found = [k for k in review_keys if k in content]
            if keys_found:
                rec = {'note_id': n_.id, 'forum': forum_id}
                for k in keys_found:
                    v = content[k]
                    if isinstance(v, dict):
                        v = v.get('value', '?')
                    rec[k] = v
                out.append(rec)
    except Exception as e:
        print(f"  ERROR {forum_id}: {e}", file=sys.stderr)
    return out


all_reviews = {}
for forum, (title, venue, query) in resolved.items():
    reviews = fetch_reviews(forum)
    if reviews:
        print(f"\n### [{venue}] {title[:90]}")
        print(f"    forum={forum}, {len(reviews)} reviews")
        all_reviews[forum] = {
            'title': title, 'venue': venue, 'query': query, 'reviews': reviews
        }

out_path = "/Users/sanegi/Documents/projects/recollection.local/.scratch/openreview_narrative_dump.json"
with open(out_path, 'w') as f:
    json.dump(all_reviews, f, indent=2, default=str)
print(f"\n\n[ok] Wrote {out_path} ({len(all_reviews)} forums)")

# ---- THEME EXTRACTION ----
print("\n\n" + "=" * 80)
print("STEP 3: Theme extraction across all reviews")
print("=" * 80)

# Heuristic theme buckets
themes = {
    'narrative/writing': [
        r'\b(?:writ(?:ing|ten)|present(?:ation|ed)|narrat(?:iv)?|clarity|clear|articulat|explain|describ|read(?:ability|s as)|confusing|hard to follow|not clear|unclear|story|flow|motivat)',
    ],
    'overclaim': [
        r'\b(?:overclaim|too\s+strong|first\s+(?:to)?|novel|seminal|paradigm|state.of.the.art|claim(?:s|ed)?\s+(?:are|is|too|that|too\s+strong)|orig)',
    ],
    'baseline-missing': [
        r'\b(?:baseline|compar(?:ison|e|ed)|missing|prior\s+work|previous|related\s+work)',
    ],
    'length-bias': [
        r'\b(?:length|verbos|token\s+count|short\s+vs\s+long|long(?:er)?\s+respons)',
    ],
    'eval/metric': [
        r'\b(?:metric|measure|score|exact\s+match|EM|F1|judge|correctness|AUROC|evaluation)',
    ],
    'sample-size/coverage': [
        r'\b(?:sample\s+size|coverage|generaliz|scope|breadth|limited|sufficient|enough)',
    ],
    'reproducibility': [
        r'\b(?:repli?cab|reproduc|code|implement|byte.faithful|details?)',
    ],
}

# Per-theme stats
theme_strengths = defaultdict(int)
theme_weaknesses = defaultdict(int)

# Collect all weakness texts
weak_texts = []
strength_texts = []
for forum, info in all_reviews.items():
    for r in info['reviews']:
        for k in ('weaknesses', 'summary_of_weaknesses', 'reasons_to_reject'):
            v = r.get(k, '')
            if isinstance(v, str) and v.strip():
                weak_texts.append(v)
                for theme, pats in themes.items():
                    for p in pats:
                        if re.search(p, v, re.IGNORECASE):
                            theme_weaknesses[theme] += 1
                            break
        for k in ('strengths', 'summary_of_strengths', 'reasons_to_accept'):
            v = r.get(k, '')
            if isinstance(v, str) and v.strip():
                strength_texts.append(v)
                for theme, pats in themes.items():
                    for p in pats:
                        if re.search(p, v, re.IGNORECASE):
                            theme_strengths[theme] += 1
                            break

print(f"\n  {'Theme':<25} {'# in W':<10} {'# in S':<10}")
print("  " + "-" * 50)
for theme in themes:
    print(f"  {theme:<25} {theme_weaknesses[theme]:<10} {theme_strengths[theme]:<10}")

# Print a few full weakness blocks for narrative-critique style
print("\n\n--- 6 EXAMPLE WEAKNESS BLOCKS (cleaned) ---")
for i, w in enumerate(weak_texts[:6]):
    print(f"\n  [#{i+1}]")
    for line in w.split('\n')[:30]:
        if line.strip():
            print(f"    {line[:180]}")
