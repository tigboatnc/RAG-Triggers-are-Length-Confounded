"""Search ICLR (public reviews) and NeurIPS for UQ/RAG-trigger papers."""
import openreview, json, sys
client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')

TARGETS = [
    ('ICLR.cc/2024/Conference', 'ICLR24'),
    ('ICLR.cc/2025/Conference', 'ICLR25'),
    ('NeurIPS.cc/2024/Conference', 'NeurIPS24'),
    ('NeurIPS.cc/2025/Conference', 'NeurIPS25'),
    ('COLM/2024/Conference', 'COLM24'),
    ('COLM/2025/Conference', 'COLM25'),
]

KEYWORDS = [
    'uncertainty', 'length bias', 'length confound', 'length-control', 'verbosity',
    'selective prediction', 'hallucination detect',
    'retrieval-augmented', 'dynamic retrieval', 'retrieval trigger',
    'flare', 'dragin', 'selfcheck', 'seakr',
    'answer trajectory', 'answer switch', 'chain-of-probe', 'step-level uncertain',
    'self-rag', 'when to retrieve', 'adaptive retrieval', 'when not to', 'when retrieval',
]

def title_matches(title):
    t = (title or '').lower()
    return any(k in t for k in KEYWORDS)

results = {}
for venue, label in TARGETS:
    print(f'\n=== {label}: {venue} ===', file=sys.stderr)
    try:
        subs = list(client.get_all_notes(content={'venueid': venue}))
    except Exception as e:
        print(f'  ! {e}', file=sys.stderr)
        continue
    print(f'  {len(subs)} submissions', file=sys.stderr)
    hits = []
    for s in subs:
        title = s.content.get('title', {}).get('value', '')
        if title_matches(title):
            hits.append((s.id, s.number, title))
    print(f'  {len(hits)} keyword hits', file=sys.stderr)
    results[label] = hits

with open('scratch/or_hits2.json', 'w') as f:
    json.dump(results, f, indent=2)
for label, hits in results.items():
    print(f'\n--- {label} ({len(hits)} hits) ---')
    for sid, num, title in hits[:60]:
        print(f'  [{sid}] #{num}  {title[:140]}')
