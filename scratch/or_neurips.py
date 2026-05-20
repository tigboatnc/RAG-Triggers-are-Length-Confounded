"""Inspect actual content keys NeurIPS uses for reviews."""
import openreview, json, sys
client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')

# Look at NeurIPS 24 Semantic Density
nid = 'LOH6qzI7T6'
notes = client.get_all_notes(forum=nid)
for n in notes:
    inv = '/'.join(n.invitations) if isinstance(n.invitations, list) else str(n.invitations)
    if 'Official_Review' in inv or 'official_review' in inv.lower():
        print(f'INV: {inv}')
        print(f'  content keys: {list(n.content.keys())}')
        # Print first 200 chars of each
        for k, v in n.content.items():
            if isinstance(v, dict) and 'value' in v:
                val = str(v['value'])[:80]
                print(f'    {k}: {val!r}')
            else:
                print(f'    {k}: {str(v)[:80]!r}')
        print('---')
        break
