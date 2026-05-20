"""Search EMNLP/ACL/NAACL/COLM/ICLR venues for papers similar to user's paper
and pull public reviews.

User paper themes:
- Length confound in UQ/hallucination evaluation
- Dynamic retrieval-augmented generation (FLARE, DRAGIN, SeaKR, SelfCheckGPT)
- Selective prediction / AUROC critique
- Step-level / answer-trajectory uncertainty
- Verbosity compensation
"""
import openreview, json, re, sys, time
client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')

# Venues to query (newest first; OpenReview-public reviews only)
TARGETS = [
    ('EMNLP/2024/Conference', 'EMNLP24'),
    ('EMNLP/2024/Workshop/BlackBoxNLP', 'BBNLP24'),
    ('EMNLP/2025/Workshop/UncertaiNLP', 'UncertaiNLP25'),
    ('aclweb.org/NAACL/2025/Conference', 'NAACL25'),
    ('aclweb.org/NAACL/2024/Conference', 'NAACL24'),
]

# Keyword filters — we care about UQ/length/RAG-trigger/answer-trajectory
KEYWORDS = [
    'uncertainty', 'length bias', 'length confound', 'length-control',
    'verbosity', 'selective prediction', 'hallucination detect',
    'retrieval-augmented', 'dynamic retrieval', 'retrieval trigger',
    'flare', 'dragin', 'selfcheck', 'seakr',
    'answer trajectory', 'answer switch', 'chain-of-probe', 'step-level',
    'AUROC', 'auc', 'calibration',
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

# Save title list for manual selection
with open('scratch/or_hits.json', 'w') as f:
    json.dump(results, f, indent=2)
for label, hits in results.items():
    print(f'\n--- {label} ({len(hits)} hits) ---')
    for sid, num, title in hits[:40]:
        print(f'  [{sid}] #{num}  {title[:120]}')
