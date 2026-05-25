#!/usr/bin/env python3
"""Deep-fetch reviews for relevant paper forums on OpenReview."""

import openreview
import json

client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')

forums = {
    'FLARE-EMNLP2023': 'WLZX3et7VT',           # Active Retrieval Augmented Generation (FLARE)
    'SelfCheckGPT-EMNLP2023': 'RwzFNbJ3Ez',     # SelfCheckGPT
    'SeaKR-ICLR2025': 'NhIaRz9Qf5',             # SeaKR ICLR 2025 submission
    'Verbosity!=Veracity-ICLR2025': 'l49uZcEIcq', # Verbosity != Veracity
    'Self-RAG-ICLR2024': 'hSyW5go0v8',          # Self-RAG (try)
    'ToRetrieveOrNot-QUESTION': 'nTcpR9ls01',   # To Retrieve or Not? UQ detection for Dynamic RAG
}

# Try to find Self-RAG ICLR 2024 forum
for q in ['Self-RAG self-reflective', 'Self-RAG reflection', 'Asai Self-RAG']:
    print(f"\n--- Query: {q!r} ---")
    try:
        notes = client.search_notes(term=q, content={}, limit=5)
        for n in notes[:5]:
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

print("\n" + "="*80)
print("REVIEW EXTRACTION")
print("="*80)

for name, fid in forums.items():
    print(f"\n\n{'#'*80}")
    print(f"# {name} (forum={fid})")
    print('#'*80)
    try:
        notes = client.get_notes(forum=fid)
        for n in notes:
            inv = n.invitations if hasattr(n, 'invitations') else []
            if not isinstance(inv, list):
                inv = [inv]
            inv_str = ' | '.join(inv) if inv else '?'
            content = n.content
            
            review_keys = ['review', 'summary', 'summary_of_strengths', 'summary_of_weaknesses',
                           'strengths', 'weaknesses', 'soundness', 'excitement',
                           'overall', 'rating', 'recommendation', 'overall_assessment',
                           'limitations', 'questions', 'questions_for_the_authors',
                           'main_review', 'meta_review', 'paper_summary',
                           'reproducibility', 'comments', 'reasons_to_accept',
                           'reasons_to_reject', 'replicability_score',
                           'soundness_score', 'overall_score']
            keys_found = [k for k in review_keys if k in content]
            if keys_found:
                inv_short = inv_str.split('/-/')[-1] if '/-/' in inv_str else inv_str
                print(f"\n----- Note id={n.id} | inv={inv_short} -----")
                for k in keys_found:
                    v = content[k]
                    if isinstance(v, dict):
                        v = v.get('value', '?')
                    if isinstance(v, str):
                        # show full content
                        print(f"  [{k}]:")
                        for line in v.split('\n'):
                            if line.strip():
                                print(f"    {line[:200]}")
                    else:
                        print(f"  [{k}]: {v}")
    except Exception as e:
        print(f"  ERROR: {e}")
