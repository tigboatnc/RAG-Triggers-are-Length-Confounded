import json
d = json.load(open('scratch/or_reviews.json'))
for k, v in d.items():
    print(f"\n## {k}: {v['title'][:90]}")
    print(f"   {len(v['reviews'])} reviews")
    for i, r in enumerate(v['reviews'][:3]):
        s = (r.get('summary') or '')[:130].replace('\n',' ')
        w = (r.get('weaknesses') or '')[:200].replace('\n',' ')
        print(f"   R{i} rating={r.get('rating')} sound={r.get('soundness')} pres={r.get('presentation')} contrib={r.get('contribution')}")
        print(f"      sum: {s}")
        print(f"      wk:  {w}")
