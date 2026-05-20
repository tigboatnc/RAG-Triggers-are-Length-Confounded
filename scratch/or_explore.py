import openreview
client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
venues = client.get_group(id='venues').members
nlp_venues = [v for v in venues if any(k in v for k in ['EMNLP', 'ACL.org/ACL', 'NAACL']) and ('2024' in v or '2025' in v or '2026' in v)]
for v in nlp_venues[:60]:
    print(v)
