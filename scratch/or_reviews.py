"""Fetch reviews for selected papers on OpenReview."""
import openreview, json, sys, time
client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')

TARGETS = [
    # (id, label, venue_hint)
    ('Iu8RytBaji', 'iclr25_lengthbias_reward',       'ICLR25 - Post-hoc Reward Calib: Length Bias'),
    ('J4xLuCt2kg', 'iclr25_uq_llm_eval',             'ICLR25 - Empirical Analysis of UQ in LLM Eval'),
    ('ztzZDzgfrh', 'iclr25_redeep',                  'ICLR25 - ReDeEP RAG hallucination'),
    ('hSyW5go0v8', 'iclr24_selfrag',                 'ICLR24 - Self-RAG'),
    ('Zj12nzlQbz', 'iclr24_inside',                  'ICLR24 - INSIDE'),
    ('gjeQKFxFpZ', 'iclr24_express_uq',              'ICLR24 - Can LLMs Express Uncertainty'),
    ('pTHfApDakA', 'iclr24_selfcheck',               'ICLR24 - SelfCheck'),
    ('j2wCrWmgMX', 'nips24_kernel_lang_entropy',     'NeurIPS24 - Kernel Language Entropy'),
    ('nfK0ZXFFSn', 'nips24_haloscope',               'NeurIPS24 - HaloScope'),
    ('YgJPQW0lkO', 'nips24_graph_uq_longform',       'NeurIPS24 - Graph UQ Long-form'),
    ('LOH6qzI7T6', 'nips24_semantic_density',        'NeurIPS24 - Semantic Density'),
    ('H1NGlLNaVC', 'nips25_cocoa',                   'NeurIPS25 - CoCoA MBR UQ'),
    ('sAFottNlra', 'nips25_signal_noise',            'NeurIPS25 - Signal and Noise in LM Eval'),
    ('KMbl8lg5Rv', 'nips25_inv_entropy',             'NeurIPS25 - Inv-Entropy'),
    ('IiEtQPGVyV', 'nips25_diversity_sampling',      'NeurIPS25 - Diversity-Steered UQ'),
    ('VwOYxPScxB', 'iclr25_hademif',                 'ICLR25 - HaDeMiF'),
    ('HSi4VetQLj', 'iclr25_sdlg',                    'ICLR25 - Semantically Diverse Generation'),
    ('IHp3vOVQO2', 'iclr25_instr_uq',                'ICLR25 - UQ in instruction following'),
    ('TDy5Ih78b4', 'iclr25_provence',                'ICLR25 - Provence context pruning'),
]

out = {}
for nid, label, hint in TARGETS:
    print(f'\n=== {label}: {hint} ({nid}) ===', file=sys.stderr)
    try:
        # Fetch the submission and all its replies
        notes = client.get_all_notes(forum=nid)
    except Exception as e:
        print(f'  ! get_all_notes failed: {e}', file=sys.stderr)
        continue
    sub = next((n for n in notes if n.id == nid), None)
    if not sub:
        print(f'  ! submission not found', file=sys.stderr)
        continue
    title = sub.content.get('title', {}).get('value', '')
    abstract = sub.content.get('abstract', {}).get('value', '')
    reviews = []
    for n in notes:
        inv = '/'.join(n.invitations) if isinstance(n.invitations, list) else str(n.invitations)
        # Pick "Official_Review" replies; skip rebuttals/meta-reviews/comments
        if 'Official_Review' in inv or 'official_review' in inv.lower():
            c = n.content
            review_obj = {
                'inv': inv,
                'rating': (c.get('rating') or {}).get('value') if isinstance(c.get('rating'), dict) else c.get('rating'),
                'confidence': (c.get('confidence') or {}).get('value') if isinstance(c.get('confidence'), dict) else c.get('confidence'),
                'summary': (c.get('summary') or {}).get('value', '') if isinstance(c.get('summary'), dict) else c.get('summary', ''),
                'strengths': (c.get('strengths') or {}).get('value', '') if isinstance(c.get('strengths'), dict) else c.get('strengths', ''),
                'weaknesses': (c.get('weaknesses') or {}).get('value', '') if isinstance(c.get('weaknesses'), dict) else c.get('weaknesses', ''),
                'questions': (c.get('questions') or {}).get('value', '') if isinstance(c.get('questions'), dict) else c.get('questions', ''),
                'soundness': (c.get('soundness') or {}).get('value', '') if isinstance(c.get('soundness'), dict) else c.get('soundness', ''),
                'presentation': (c.get('presentation') or {}).get('value', '') if isinstance(c.get('presentation'), dict) else c.get('presentation', ''),
                'contribution': (c.get('contribution') or {}).get('value', '') if isinstance(c.get('contribution'), dict) else c.get('contribution', ''),
            }
            reviews.append(review_obj)
    out[label] = {
        'id': nid,
        'title': title,
        'abstract': abstract,
        'reviews': reviews,
    }
    print(f'  -> {len(reviews)} reviews', file=sys.stderr)

with open('scratch/or_reviews.json', 'w') as f:
    json.dump(out, f, indent=2)
print('\nSaved scratch/or_reviews.json')
