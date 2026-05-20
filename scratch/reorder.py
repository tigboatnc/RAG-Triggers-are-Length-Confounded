import re

with open('/Users/sanegi/Documents/projects/recollection.local/_exploration/websiteV2/index.html.bak', 'r') as f:
    html = f.read()

# 1. Replace nav
new_nav = """    <nav class="tab-bar">
        <div class="tab-container">
            <button class="tab-btn active" data-target="prior-works">Background</button>
            <button class="tab-btn" data-target="datasets">The Target</button>
            <button class="tab-btn" data-target="models">Models &amp; Boundaries</button>
            <button class="tab-btn" data-target="dynamic-rag">The Triggers</button>
            <button class="tab-btn" data-target="hypothesis">Hypotheses &amp; Stats</button>
            <button class="tab-btn" data-target="evaluation">Evaluation Matrix</button>
            <button class="tab-btn" data-target="llm-judge">LLM-as-Judge</button>
        </div>
    </nav>"""
html = re.sub(r'<nav class="tab-bar">.*?</nav>', new_nav, html, flags=re.DOTALL)

# Remove active classes from panels before splitting
html = html.replace('class="tab-panel active"', 'class="tab-panel"')

# Extract blocks function
def extract_block(text, start_str, end_str):
    start_idx = text.find(start_str)
    if start_idx == -1: return "", text
    end_idx = text.find(end_str, start_idx) + len(end_str)
    block = text[start_idx:end_idx]
    text = text[:start_idx] + text[end_idx:]
    return block, text

# Extract blocks from entire HTML first to be safe
elephant_block, html = extract_block(html, '<h3 style="margin-top: 32px;">The Length Confound (The Elephant in the Room)</h3>', 'statistically valid triggers.</p>\n                </div>')
c_parent_block, html = extract_block(html, '<h3 style="margin-top: 32px;">C. The Paired-Oracle Parent</h3>', 'from wrong to right?&rdquo;</p>\n                </div>')
b_trajectory_block, html = extract_block(html, '<h3 style="margin-top: 32px;">B. Answer-Trajectory Lineage (the new length-orthogonal signal class)</h3>', 'timing alone.</p>\n                    </div>\n                </div>')
d_bridge_block, html = extract_block(html, '<h3 style="margin-top: 32px;">D. What We Bridge</h3>', 'Bonferroni-robust at <code>k=15</code>.</p>\n                </div>')
k5_k1_text, html = extract_block(html, '<h3 style="margin-top: 24px;">K=5 vs K=1: The Multi-Sample Variant Is Null</h3>', 'table.</p>\n                </div>')
bootstrap_block, html = extract_block(html, '<h3 style="margin-top: 24px;">The Paired Bootstrap Procedure (B=10,000)</h3>', 'independent bootstraps would.</p>\n                </div>')
bonferroni_block, html = extract_block(html, '<h3 style="margin-top: 24px;">Why Bonferroni (and Why &alpha; = 0.0033)</h3>', 'fluke of one signal class.</p>\n                </div>')

# Split panels
parts = html.split('<div class="tab-panel"')
pre_main = parts[0]
panels = {}
for p in parts[1:]:
    # Find id
    m = re.search(r'id="([^"]+)"', p)
    if m:
        panels[m.group(1)] = '<div class="tab-panel"' + p

# Process prior-works (Background)
pw = panels['prior-works']
# Remove active class from prior-works if present, or add it
pw = pw.replace('class="tab-panel"', 'class="tab-panel active"')
# Append elephant block before last </div></div>
pw = pw.replace('</div>\n        </div>', '\n                ' + elephant_block + '\n\n                <div style="margin-top: 32px; padding: 16px; background: #f1f5f9; border-left: 4px solid #3b82f6;"><p style="margin:0;"><strong>To solve this, we first need a target that measures actual retrieval utility rather than just generic model error. &rarr; <a href="#" onclick="document.querySelector(\'[data-target=\\\'datasets\\\']\').click(); return false;" style="color: #2563eb; font-weight: bold; text-decoration: none;">Next: The Paired-Oracle Target</a></strong></p></div>\n            </div>\n        </div>')
panels['prior-works'] = pw

# Process datasets (The Target)
ds = panels['datasets']
ds = ds.replace('<h2>The Evaluation Cohorts</h2>', c_parent_block + '\n\n                <h2>The Evaluation Cohorts</h2>')
ds = ds.replace('</div>\n        </div>', '\n                <div style="margin-top: 32px; padding: 16px; background: #f1f5f9; border-left: 4px solid #3b82f6;"><p style="margin:0;"><strong>Now that we know exactly what we are predicting (<code>lift_em = +1</code>), we need to look at the traces we are extracting signals from. &rarr; <a href="#" onclick="document.querySelector(\'[data-target=\\\'models\\\']\').click(); return false;" style="color: #2563eb; font-weight: bold; text-decoration: none;">Next: Models &amp; Boundaries</a></strong></p></div>\n            </div>\n        </div>')
panels['datasets'] = ds

# Process models (Models & Boundaries)
md = panels['models']
md = md.replace('</div>\n        </div>', '\n                <div style="margin-top: 32px; padding: 16px; background: #f1f5f9; border-left: 4px solid #3b82f6;"><p style="margin:0;"><strong>With our multi-step traces and boundaries defined, what specific signals are we testing at these boundaries? &rarr; <a href="#" onclick="document.querySelector(\'[data-target=\\\'dynamic-rag\\\']\').click(); return false;" style="color: #2563eb; font-weight: bold; text-decoration: none;">Next: The Triggers</a></strong></p></div>\n            </div>\n        </div>')
panels['models'] = md

# Process dynamic-rag (The Triggers)
dr = panels['dynamic-rag']
dr = dr.replace('(<span style="color:#047857;">Bonferroni-robust</span>)', '(<span style="color:#047857;">Bonferroni-robust</span> <em style="font-size:0.85em; color:#64748b;">see Hypotheses tab</em>)')
dr = dr.replace('</p>\n\n                <div class="grid-layout"', '</p>\n\n                ' + b_trajectory_block + '\n\n                <div class="grid-layout"')
dr = dr.replace('</div>\n        </div>', '\n                <div style="margin-top: 32px; padding: 16px; background: #f1f5f9; border-left: 4px solid #3b82f6;"><p style="margin:0;"><strong>We have the target, the models, and the signals. How do we rigorously and statistically evaluate them? &rarr; <a href="#" onclick="document.querySelector(\'[data-target=\\\'hypothesis\\\']\').click(); return false;" style="color: #2563eb; font-weight: bold; text-decoration: none;">Next: Hypotheses &amp; Stats</a></strong></p></div>\n            </div>\n        </div>')
panels['dynamic-rag'] = dr

# Process hypothesis (Hypotheses & Stats)
hp = panels['hypothesis']
hp = hp.replace('<h3 style="margin-top: 24px;">The Three-Tier Survivor Structure (Pre-Registered)</h3>', bootstrap_block + '\n\n                ' + bonferroni_block + '\n\n                <h3 style="margin-top: 24px;">The Three-Tier Survivor Structure (Pre-Registered)</h3>')
hp = hp.replace('</div>\n        </div>', '\n                <div style="margin-top: 32px; padding: 16px; background: #f1f5f9; border-left: 4px solid #3b82f6;"><p style="margin:0;"><strong>With the statistical framework locked, let\'s look at the results. &rarr; <a href="#" onclick="document.querySelector(\'[data-target=\\\'evaluation\\\']\').click(); return false;" style="color: #2563eb; font-weight: bold; text-decoration: none;">Next: Evaluation Matrix</a></strong></p></div>\n            </div>\n        </div>')
panels['hypothesis'] = hp

# Process evaluation (Evaluation Matrix)
ev = panels['evaluation']
ev = ev.replace('<div class="trace-example">\n                    <table class="trace-table">', d_bridge_block + '\n\n                <div class="trace-example">\n                    <table class="trace-table">')
ev = ev.replace('<h3 style="margin-top: 24px;">The K=1 vs K=5 Ablation (Why We Recommend K=1 for Deployment)</h3>', k5_k1_text + '\n\n                <h3 style="margin-top: 24px;">The K=1 vs K=5 Ablation (Why We Recommend K=1 for Deployment)</h3>')
ev = ev.replace('</div>\n        </div>', '\n                <div style="margin-top: 32px; padding: 16px; background: #f1f5f9; border-left: 4px solid #3b82f6;"><p style="margin:0;"><strong>Are these results just an artifact of our Exact Match metric? &rarr; <a href="#" onclick="document.querySelector(\'[data-target=\\\'llm-judge\\\']\').click(); return false;" style="color: #2563eb; font-weight: bold; text-decoration: none;">Next: LLM-as-Judge</a></strong></p></div>\n            </div>\n        </div>')
panels['evaluation'] = ev

# Process llm-judge (remains unchanged)

# Reassemble in order
new_html = pre_main + \
           panels['prior-works'] + \
           panels['datasets'] + \
           panels['models'] + \
           panels['dynamic-rag'] + \
           panels['hypothesis'] + \
           panels['evaluation'] + \
           panels['llm-judge']

with open('/Users/sanegi/Documents/projects/recollection.local/_exploration/websiteV2/index.html', 'w') as f:
    f.write(new_html)

print("Done reorganizing DOM!")
