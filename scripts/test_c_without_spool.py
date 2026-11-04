import pandas as pd
from collections import defaultdict
import re

# Simulate selection in expansion_build_technical with 10019753 excluded
df_cand = pd.read_csv('expansion_wipo_v1/candidates/wipo_candidates.csv', encoding='utf-8')
df_var = pd.read_csv('expansion_wipo_v1/candidates/wipo_variants.csv', encoding='utf-8')

records = defaultdict(lambda: {'EN': [], 'RU': []})
for _, r in df_var.iterrows():
    cid = r['local_concept_id']
    rec_id = df_cand[df_cand['local_concept_id'] == cid]['concept_record_id'].iloc[0]
    rating = int(r['rating']) if pd.notna(r['rating']) else 0
    usage = str(r['usage']).strip().lower() if pd.notna(r['usage']) else ''
    if rating >= 3 and usage in ('allowed', 'standardized', 'preferred', ''):
        records[rec_id][r['lang']].append(r)

valid_primary_concepts = []
for rec_id, d in records.items():
    if not d['EN'] or not d['RU']: continue
    en_prim = [r for r in d['EN'] if bool(r.get('is_primary', False))]
    ru_prim = [r for r in d['RU'] if bool(r.get('is_primary', False))]
    if not en_prim or not ru_prim: continue
    en_prim.sort(key=lambda x: -int(x.get('rating') or 0))
    ru_prim.sort(key=lambda x: -int(x.get('rating') or 0))
    en_top = en_prim[0]
    ru_top = ru_prim[0]
    
    # Exclude 10019753
    if str(en_top['term_id']) == '10019753' or str(ru_top['term_id']) == '10019754':
        continue
        
    t_en = str(en_top['term_clean']).strip()
    t_ru = str(ru_top['term_clean']).strip()
    w_en = len(t_en.split())
    w_ru = len(t_ru.split())
    if (w_en == 1 and 2 <= w_ru <= 3) or (w_ru == 1 and 2 <= w_en <= 3):
        valid_primary_concepts.append((rec_id, en_top['term_id'], t_en, t_ru, int(en_top['rating']) + int(ru_top['rating'])))

print(f"Stratum C primary candidates without 10019753: {len(valid_primary_concepts)}")
for item in valid_primary_concepts:
    print(item)
