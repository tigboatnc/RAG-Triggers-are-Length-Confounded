"""Fetch reviews more carefully: filter out rebuttals and only keep top-level reviews."""
import openreview, json, sys
client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')

TARGETS = [
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
    ('H1NGlLNaVC', 'nips25_cocoa',                   'NeurIPS25 - CoCoA'),
    ('sAFottNlra', 'nips25_signal_noise',            'NeurIPS25 - Signal and Noise'),
    ('KMbl8lg5Rv', 'nips25_inv_entropy',             'NeurIPS25 - Inv-Entropy'),
    ('IiEtQPGVyV', 'nips25_diversity_sampling',      'NeurIPS25 - Diversity-Steered UQ'),
    ('VwOYxPScxB', 'iclr25_hademif',                 'ICLR25 - HaDeMiF'),
    ('HSi4VetQLj', 'iclr25_sdlg',                    'ICLR25 - SDLG'),
    ('IHp3vOVQO2', 'iclr25_instr_uq',                'ICLR25 - UQ in instruction following'),
    ('TDy5Ih78b4', 'iclr25_provence',                'ICLR25 - Provence context pruning'),
]


def is_top_level_review(note, forum_id):
    """A real review: replyto = forum, invitation contains Official_Review, does not contain Rebuttal/Comment/Edit."""
    if note.replyto != forum_id:
        return False
    inv = '/'.join(note.invitations) if isinstance(note.invitations, list) else str(note.invitations)
    if 'Official_Review' not in inv:
        return False
    bad = ('Rebuttal', 'Comment', 'Confidential_Comment', 'Meta_Review', 'Author-')
    for b in bad:
        if b in inv:
            return False
    return True


def gv(c, k):
    v = c.get(k)
    if isinstance(v, dict):
        return v.get('value', '')
    return v if v is not None else ''


out = {}
for nid, label, hint in TARGETS:
    print(f'=== {label}: {hint} ({nid}) ===', file=sys.stderr)
    try:
        notes = client.get_all_notes(forum=nid)
    except Exception as e:
        print(f'  ! {e}', file=sys.stderr)
        continue
    sub = next((n for n in notes if n.id == nid), None)
    if not sub:
        continue
    title = gv(sub.content, 'title')
    abstract = gv(sub.content, 'abstract')
    reviews = []
    for n in notes:
        if not is_top_level_review(n, nid):
            continue
        c = n.content
        reviews.append({
            'rating': gv(c, 'rating'),
            'confidence': gv(c, 'confidence'),
            'soundness': gv(c, 'soundness'),
            'presentation': gv(c, 'presentation'),
            'contribution': gv(c, 'contribution'),
            'summary': gv(c, 'summary'),
            'strengths': gv(c, 'strengths'),
            'weaknesses': gv(c, 'weaknesses'),
            'questions': gv(c, 'questions'),
            # NeurIPS sometimes uses different keys
            'limitations': gv(c, 'limitations'),
        })
    out[label] = {'id': nid, 'title': title, 'abstract': abstract, 'reviews': reviews}
    print(f'  -> {len(reviews)} top-level reviews', file=sys.stderr)

with open('scratch/or_reviews2.json', 'w') as f:
    json.dump(out, f, indent=2)
print('Saved scratch/or_reviews2.json')
