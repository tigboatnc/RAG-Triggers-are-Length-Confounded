"""Print full weaknesses + questions sections for the most relevant papers."""
import json
d = json.load(open('scratch/or_reviews2.json'))

# Most relevant to the user's paper
PRIORITY = [
    'iclr25_lengthbias_reward',
    'iclr25_uq_llm_eval',
    'iclr25_sdlg',
    'iclr25_redeep',
    'iclr24_inside',
    'iclr24_selfrag',
    'iclr24_selfcheck',
    'nips24_kernel_lang_entropy',
    'nips24_semantic_density',
]

for p in PRIORITY:
    info = d.get(p)
    if not info:
        continue
    print(f'\n\n=========================================================')
    print(f'## {p}: {info["title"]}')
    print(f'   ratings: {[r.get("rating") for r in info["reviews"]]}')
    print(f'=========================================================')
    for i, r in enumerate(info['reviews']):
        print(f'\n--- R{i}  rating={r.get("rating")}  sound={r.get("soundness")}  pres={r.get("presentation")}  contrib={r.get("contribution")} ---')
        wk = r.get('weaknesses') or ''
        qs = r.get('questions') or ''
        if wk:
            print(f'WEAKNESSES:\n{wk[:2200]}')
        if qs:
            print(f'\nQUESTIONS:\n{qs[:1200]}')
