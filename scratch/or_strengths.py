"""Extract STRENGTHS sections from high-rated reviews to harvest the 'winning framing' vocabulary."""
import json, re
d = json.load(open('scratch/or_reviews2.json'))

# For each paper, find reviews rated >=6 and dump their strengths
def num(r):
    try:
        s = r.get('rating') or ''
        m = re.search(r'(\d+(?:\.\d+)?)', str(s))
        return float(m.group(1)) if m else 0
    except: return 0

print('\n# STRENGTHS SECTIONS FROM HIGH-RATED REVIEWS\n')
for paper, info in d.items():
    high = [r for r in info['reviews'] if num(r) >= 6]
    if not high:
        continue
    print(f'\n## {paper}: {info["title"][:90]}')
    for i, r in enumerate(high):
        st = (r.get('strengths') or '').strip()
        if st:
            print(f'\n--- R rating={r.get("rating")} sound={r.get("soundness")} pres={r.get("presentation")} contrib={r.get("contribution")} ---')
            print(f'STRENGTHS:\n{st[:1800]}')
