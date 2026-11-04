# -*- coding: utf-8 -*-
"""Inspect candidate pool for WIPO expansion."""
import json, re, html
from pathlib import Path
from collections import Counter
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "expansion_wipo_v1" / "raw" / "wipo"
PILOT_FILE = ROOT / "data" / "real" / "concepts_pilot.csv"

def clean_html(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<(?:p|div|br|tr|td|li|h[1-6])[^>]*>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"</(?:p|div|tr|td|li|h[1-6])>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

# Load existing baseline concepts to check duplicates
df_pilot = pd.read_csv(PILOT_FILE)
accepted_pilot = df_pilot[df_pilot["final_status"] == "accepted"]
existing_ru = set(accepted_pilot["term_ru_sg"].str.strip().str.lower())
existing_en = set(accepted_pilot["term_en_sg"].str.strip().str.lower())

unique_concepts = {}
for f in sorted(RAW_DIR.glob("*.json")):
    if f.name in ("search_log.jsonl", "search_budget.json"):
        continue
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        continue
    seed = f.stem.split("_")[0]
    for i, rec in enumerate(data.get("Record", [])):
        langs = {li.get("language"): li.get("termItems", []) for li in rec.get("languageItems", [])}
        if "EN" not in langs or "RU" not in langs:
            continue
        en_items, ru_items = langs["EN"], langs["RU"]
        if not en_items or not ru_items:
            continue
        tids = set()
        for ti in en_items + ru_items:
            tid = ti.get("termId") or ti.get("id")
            if tid:
                tids.add(str(tid))
        fp = frozenset(tids)
        if not fp:
            continue
        if fp not in unique_concepts:
            unique_concepts[fp] = {
                "en_items": en_items,
                "ru_items": ru_items,
                "rec_id": f"{f.name}#rec_{i}",
                "seeds": [seed],
                "subject_fields": rec.get("subjectField", "")
            }
        else:
            if seed not in unique_concepts[fp]["seeds"]:
                unique_concepts[fp]["seeds"].append(seed)

def pick_best(items):
    def sort_key(x):
        usage = (x.get("usage") or "").strip().lower()
        is_allowed = usage in ("allowed", "standardized", "preferred", "")
        try:
            r = int(x.get("rating") or 0)
        except Exception:
            r = 0
        return (not is_allowed, -r)
    return sorted(items, key=sort_key)[0]

candidates = []
for fp, c in unique_concepts.items():
    en_top = pick_best(c["en_items"])
    ru_top = pick_best(c["ru_items"])
    en_term = clean_html(en_top.get("term", ""))
    ru_term = clean_html(ru_top.get("term", ""))
    try:
        r_en = int(en_top.get("rating") or 0)
    except Exception:
        r_en = 0
    try:
        r_ru = int(ru_top.get("rating") or 0)
    except Exception:
        r_ru = 0
    
    w_en = len(en_term.split())
    w_ru = len(ru_term.split())
    
    if w_en == 1 and w_ru == 1:
        stratum = "A"
    elif 2 <= w_en <= 3 and 2 <= w_ru <= 3:
        stratum = "B"
    elif (w_en == 1 and 2 <= w_ru <= 3) or (w_ru == 1 and 2 <= w_en <= 3):
        stratum = "C"
    else:
        stratum = "Other"
        
    is_dup_pilot = (ru_term in existing_ru) or (en_term in existing_en)
    
    candidates.append({
        "en": en_term,
        "ru": ru_term,
        "w_en": w_en,
        "w_ru": w_ru,
        "r_en": r_en,
        "r_ru": r_ru,
        "stratum": stratum,
        "is_dup_pilot": is_dup_pilot,
        "rec_id": c["rec_id"],
        "seeds": ",".join(c["seeds"])
    })

df_cand = pd.DataFrame(candidates)
print(f"Total unique parsed concepts: {len(df_cand)}")
print("Stratum distribution:")
print(df_cand["stratum"].value_counts())

valid = df_cand[(df_cand["r_en"] >= 3) & (df_cand["r_ru"] >= 3) & (~df_cand["is_dup_pilot"])]
print("\nValid non-duplicate (rating >= 3) stratum distribution:")
print(valid["stratum"].value_counts())

print("\n--- GROUP C CANDIDATES ---")
c_valid = valid[valid["stratum"] == "C"]
for idx, r in c_valid.iterrows():
    print(f"  [{r['w_ru']}->{r['w_en']}] '{r['ru']}' <-> '{r['en']}' (rec: {r['rec_id']}, seeds: {r['seeds']})")

if __name__ == "__main__":
    pass
