#!/usr/bin/env python3
"""Fetch reviews from relevant paper forums on OpenReview."""

import openreview
import sys
import json

client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')

# Candidate forums to pull reviews from
forums = {
    'DRAGIN-ACL2024': 'OCh112m6z2',
    'Unified Active Retrieval-EMNLP-F': 'y4MEChpv5L',
    'FLARE-EMNLP2023': 'WLZX3et7VT',
    'UNCERTAINTY-LINE-arxiv': 'GedoF7kQJ9',
    'First-Tokens-ICLR-WS': '0FOOrwSQ9E',
}

# Also: pure-text search to discover similar papers + their reviews on EMNLP/ARR/ICLR
additional_queries = [
    "FLARE active retrieval",
    "SelfCheckGPT hallucination",
    "SeaKR self-aware knowledge retrieval",
    "Self-RAG retrieval augmented",
    "verbosity compensation LLM",
    "selective prediction question answering",
    "calibration language model length",
]

for q in additional_queries:
    print(f"\n--- Query: {q!r} ---")
    try:
        notes = client.search_notes(term=q, content={}, limit=8)
        for n in notes[:8]:
            title = n.content.get('title', {})
            if isinstance(title, dict):
                title = title.get('value', '?')
            venue = n.content.get('venue', {})
            if isinstance(venue, dict):
                venue = venue.get('value', '?')
            forum = n.forum if hasattr(n, 'forum') else n.id
            print(f"  - {title[:90]} | venue={venue} | forum={forum}")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "="*60)
print("Fetching reviews for known forums...")
print("="*60)

for name, fid in forums.items():
    print(f"\n--- {name} (forum={fid}) ---")
    try:
        notes = client.get_notes(forum=fid)
        print(f"  Total notes in forum: {len(notes)}")
        for n in notes:
            inv = n.invitations if hasattr(n, 'invitations') else []
            if not isinstance(inv, list):
                inv = [inv]
            inv_str = ' | '.join(inv) if inv else (n.invitation if hasattr(n, 'invitation') else '?')
            content = n.content
            
            # Look for review-y stuff
            review_keys = ['review', 'summary', 'strengths', 'weaknesses', 'soundness', 
                           'excitement', 'overall', 'rating', 'recommendation', 'comments',
                           'limitations', 'questions', 'main_review']
            
            keys_found = [k for k in review_keys if k in content]
            if keys_found:
                print(f"\n  -- Note id={n.id} | invitations={inv_str}")
                for k in keys_found:
                    v = content[k]
                    if isinstance(v, dict):
                        v = v.get('value', '?')
                    if isinstance(v, str):
                        preview = v[:1500].replace('\n', ' ')
                        print(f"    [{k}]: {preview}")
                    else:
                        print(f"    [{k}]: {v}")
    except Exception as e:
        print(f"  ERROR: {e}")
