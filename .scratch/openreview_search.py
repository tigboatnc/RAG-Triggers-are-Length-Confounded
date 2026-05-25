#!/usr/bin/env python3
"""Search OpenReview for reviews on papers about dynamic RAG, length confounds in UQ."""

import openreview
import sys
import json
import re

# API v2
try:
    client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
    print("API v2 client connected")
except Exception as e:
    print(f"API v2 init failed: {e}")
    client = None

# Search for ICLR-style venue reviews (often more public than EMNLP)
# Keywords from our paper
keyword_queries = [
    "DRAGIN dynamic retrieval",
    "active retrieval generation",
    "length bias uncertainty quantification LLM",
    "hallucination detection length",
    "retrieval trigger language model",
    "uncertainty estimation length-invariant",
    "answer trajectory LLM reasoning",
    "early commitment LLM",
]

print("\n" + "="*60)
print("Searching for related submissions via notes search...")
print("="*60)

if client:
    for q in keyword_queries:
        print(f"\n--- Query: {q!r} ---")
        try:
            # Use notes search
            notes = client.search_notes(term=q, content={}, limit=10)
            for n in notes[:10]:
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
