#!/usr/bin/env python3
"""Targeted OpenReview review pull on the most relevant papers for our EMNLP 2026 short.

Domain coverage:
1. Length-confound family: Vashurin Uncertainty-Line, Janiak Illusion of Progress, Santilli Revisiting UQ, Ielanskyi UQ pitfalls
2. Dynamic RAG trigger pool: DRAGIN, FLARE, SeaKR (already done), Self-RAG (already done), INSIDE
3. Answer-trajectory family: Datta Decide Early, Wang Chain-of-Probe, Chen Dynamic Instability, Lu Streaming, Zhang ARS
4. New 2026: Zhang Long Commitment, Ballon Trajectory probing, DTR, CtrlA, QuCo-RAG, DioR, MIND

Goal: surface common reviewer concerns (weaknesses, soundness scores, rejection reasons)
to anticipate what an EMNLP 2026 reviewer would flag for our paper.
"""

import openreview
import json
import sys
import re
from collections import defaultdict

client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')


# search_notes returns a list of forum-IDs that match the term; we then get_notes(forum=fid)
def search_for_paper(query, n=5):
    """Return list of (title, venue, forum) tuples."""
    out = []
    try:
        notes = client.search_notes(term=query, content={}, limit=n)
        for n_ in notes[:n]:
            title = n_.content.get('title', {})
            if isinstance(title, dict):
                title = title.get('value', '?')
            venue = n_.content.get('venue', {})
            if isinstance(venue, dict):
                venue = venue.get('value', '?')
            forum = n_.forum if hasattr(n_, 'forum') else n_.id
            out.append((title, venue, forum))
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
    return out


def fetch_reviews(forum_id):
    """Return list of (review_dict) for a given forum, only review-style notes."""
    review_keys = ['summary', 'summary_of_strengths', 'summary_of_weaknesses',
                   'strengths', 'weaknesses', 'soundness', 'excitement',
                   'overall', 'rating', 'recommendation', 'overall_assessment',
                   'limitations', 'questions', 'questions_for_the_authors',
                   'main_review', 'reasons_to_accept', 'reasons_to_reject',
                   'reproducibility', 'soundness_score', 'overall_score',
                   'comments_suggestions_and_typos']
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
        print(f"  ERROR fetching {forum_id}: {e}", file=sys.stderr)
    return out


# Targeted forums for papers most relevant to our EMNLP 2026 short.
# IDs verified by web/openreview search. If a paper is at venue not on OpenReview
# (e.g., older ACL/NAACL pre-2024) we just skip — only OpenReview-submitted ones are here.
candidate_searches = [
    # length-confound family
    "UNCERTAINTY-LINE length invariant",
    "Vashurin uncertainty length invariant",
    "Santilli revisiting uncertainty quantification length",
    "Janiak illusion of progress hallucination",
    "Ielanskyi uncertainty estimation NLG pitfalls",
    "verbosity veracity verbosity compensation Zhang",
    # answer-trajectory family
    "Datta decide early explain later",
    "Chain of Probe necessity accuracy CoT step",
    "dynamic instability LLM reasoning Chen",
    "answer agreement representation shaping ICML",
    "streaming hallucination chain of thought Lu",
    # dynamic RAG trigger pool
    "DRAGIN real time information needs dynamic retrieval",
    "active retrieval augmented generation FLARE",
    "INSIDE EigenScore internal states detect hallucination",
    "Mallen when to trust language models retrieval popularity",
    # newer 2025-2026 dynamic RAG
    "DTR decide then retrieve dual path",
    "CtrlA adaptive retrieval probe",
    "DioR adaptive cognitive detection retrieval",
    "QuCo-RAG corpus uncertainty pretraining",
    "MIND token entropy attention multi-hop retrieval",
    # commitment-time / trajectory-probe (the new collisions)
    "when does language model commit pre-verbalization",
    "probing trajectories reasoning traces large language models",
    "point of no return deceptive commitment",
    "first commitment time decode answer",
]

print("=" * 80)
print("STEP 1: Resolve forum IDs for relevant papers")
print("=" * 80)

resolved = {}
for q in candidate_searches:
    print(f"\n--- Query: {q!r}")
    hits = search_for_paper(q, n=4)
    for title, venue, forum in hits:
        # heuristically pick the most relevant
        title_low = (title or "").lower()
        # use first hit unless null
        if title and title != '?':
            # dedupe
            if forum not in resolved:
                resolved[forum] = (title, venue, q)
                print(f"  + [{venue}] {title[:100]} | forum={forum}")
                # break after first decent hit per query to limit
                break
        else:
            print(f"  - (no title) | forum={forum}")

print(f"\n\nTotal forums resolved: {len(resolved)}")

print("\n" + "=" * 80)
print("STEP 2: Fetch reviews for each resolved forum")
print("=" * 80)

all_reviews = {}
for forum, (title, venue, query) in resolved.items():
    print(f"\n### {title[:90]} [{venue}] (forum={forum})")
    reviews = fetch_reviews(forum)
    if reviews:
        print(f"   {len(reviews)} review-like notes")
        all_reviews[forum] = {
            'title': title, 'venue': venue, 'query': query, 'reviews': reviews
        }
    else:
        print(f"   (no reviews)")

# Persist
out_path = "/Users/sanegi/Documents/projects/recollection.local/.scratch/openreview_targeted_dump.json"
with open(out_path, 'w') as f:
    json.dump(all_reviews, f, indent=2, default=str)
print(f"\n\n[ok] Wrote {out_path} ({len(all_reviews)} forums w/ reviews)")
