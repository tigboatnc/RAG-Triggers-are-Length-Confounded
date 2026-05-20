"""Print actual quotes from review weaknesses on relevant buckets."""
import json
d = json.load(open('scratch/or_extracted.json'))

# Print quotes for key papers
PRIORITY = [
    ('iclr25_lengthbias_reward', ['length_verbosity', 'baseline_coverage', 'novelty_incremental', 'llm_judge_critique']),
    ('iclr25_uq_llm_eval',        ['novelty_incremental', 'datasets_coverage', 'llm_judge_critique', 'multitest_stats']),
    ('iclr25_sdlg',               ['novelty_incremental', 'baseline_coverage', 'auroc_eval', 'causal_mechanism']),
    ('iclr25_redeep',             ['novelty_incremental', 'writing_clarity', 'baseline_coverage', 'causal_mechanism']),
    ('iclr24_selfrag',            ['novelty_incremental', 'multitest_stats', 'baseline_coverage']),
    ('iclr24_inside',             ['novelty_incremental', 'writing_clarity', 'baseline_coverage', 'auroc_eval']),
    ('iclr24_selfcheck',          ['multitest_stats', 'datasets_coverage', 'llm_judge_critique']),
    ('nips24_kernel_lang_entropy',['novelty_incremental', 'baseline_coverage', 'multitest_stats', 'auroc_eval']),
    ('nips24_semantic_density',   ['auroc_eval', 'novelty_incremental']),
    ('iclr25_hademif',            ['datasets_coverage', 'auroc_eval']),
    ('iclr25_provence',           ['auroc_eval', 'writing_clarity']),
]

for paper, buckets in PRIORITY:
    info = d.get(paper)
    if not info:
        continue
    print(f'\n\n========================================')
    print(f'## {paper}: {info["title"][:90]}')
    print(f'   ratings: {info["ratings"]}')
    print(f'========================================')
    for b in buckets:
        qs = info['sample_quotes'].get(b, [])
        if not qs:
            continue
        print(f'\n### [{b}] ({info["bucket_hits"][b]}/{info["reviews_count"]} reviews)')
        for q in qs:
            print(f'  > {q}')
