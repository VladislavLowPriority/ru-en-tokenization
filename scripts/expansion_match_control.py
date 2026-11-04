# -*- coding: utf-8 -*-
"""Сопоставление бытового контроля для expansion_wipo_v1 (§14.4).

Правила протокола:
1. Совпадают A/B/C и число слов отдельно в RU и EN (exact_cell).
2. Разница длины без пробелов не больше 4 знаков в каждом языке (|d_chars| <= 4).
3. Минимизируется сумма абсолютных различий длин |d_RU| + |d_EN|.
4. Равенства разрешаются фиксированным порядком кандидатов.
5. Не более 3 понятий на контрольную лексическую семью.
6. Без повторного использования контроля (старые 76 зарезервированы).
7. Никакие токенные длины не читаются.
8. Несопоставленные понятия сохраняются в отдельной таблице причин.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "expansion_wipo_v1"
DATA_DIR = EXP / "data"
RAW_CTRL = EXP / "raw" / "control"

TECH_NEW_FILE = DATA_DIR / "concepts_new.csv"
TECH_COMB_FILE = DATA_DIR / "concepts_combined.csv"
BASE_MATCH_FILE = ROOT / "data" / "control" / "matching_pairs.csv"
BASE_CTRL_FILE = ROOT / "data" / "control" / "concepts_control.csv"
S1_FILE = RAW_CTRL / "control_candidates_s1.csv"
S2_FILE = RAW_CTRL / "control_candidates_s2.csv"
MANUAL_EXCL_FILE = RAW_CTRL / "manual_exclusions.csv"

# 1. Загрузка старых сопоставлений (76 пар)
df_base_match = pd.read_csv(BASE_MATCH_FILE)
reserved_ctrl_cand_ids = set(df_base_match["ctrl_cand_id"].astype(str))
print(f"Reserved baseline control candidate IDs: {len(reserved_ctrl_cand_ids)}")

# Граф семей контроля
class ControlFamilyTracker:
    def __init__(self):
        self.parent = {}
        
    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
        
    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra
            
    def add_concept(self, cid: str, keys: set[str]):
        for k in keys:
            self.union(f"c:{cid}", f"k:{k}")
            
    def family_size_with_candidate(self, cand_id: str, keys: set[str]) -> int:
        temp_parent = dict(self.parent)
        def tfind(x):
            if x not in temp_parent:
                temp_parent[x] = x
                return x
            if temp_parent[x] != x:
                temp_parent[x] = tfind(temp_parent[x])
            return temp_parent[x]
        def tunion(a, b):
            ra, rb = tfind(a), tfind(b)
            if ra != rb:
                temp_parent[rb] = ra
        for k in keys:
            tunion(f"c:{cand_id}", f"k:{k}")
        root = tfind(f"c:{cand_id}")
        return sum(1 for node in temp_parent if node.startswith("c:") and tfind(node) == root)

ctrl_fam = ControlFamilyTracker()

def extract_ctrl_keys(ru: str, en: str) -> set[str]:
    keys = set()
    ru_words = str(ru).lower().split()
    en_words = str(en).lower().split()
    if ru_words:
        if ru_words[0].endswith(("ый", "ий", "ой", "ая", "яя", "ое", "ее", "ые", "ие")) and len(ru_words) > 1:
            h_ru = ru_words[-1]
        else:
            h_ru = ru_words[0]
        keys.add(f"ru:{h_ru}")
    if en_words:
        keys.add(f"en:{en_words[-1]}")
    return keys

# Seed baseline control concepts into family tracker
for _, r in df_base_match.iterrows():
    cid = str(r["ctrl_cand_id"])
    keys = extract_ctrl_keys(r["ctrl_RU"], r["ctrl_EN"])
    ctrl_fam.add_concept(cid, keys)

# 2. Загрузка кандидатов контроля (S1 + S2)
manual_exclusions_pairs = set()
manual_exclusions_terms = set()
if MANUAL_EXCL_FILE.exists():
    for r in csv.DictReader(open(MANUAL_EXCL_FILE, encoding="utf-8")):
        en = str(r.get("term_en", "")).strip().lower()
        ru = str(r.get("term_ru", "")).strip().lower()
        if en and ru:
            manual_exclusions_pairs.add((en, ru))
        if en:
            manual_exclusions_terms.add(en)
        if ru:
            manual_exclusions_terms.add(ru)

baseline_ctrl_en = set(df_base_match["ctrl_EN"].astype(str).str.strip().str.lower())
baseline_ctrl_ru = set(df_base_match["ctrl_RU"].astype(str).str.strip().str.lower())

def chars_len(s: str) -> int:
    return len(str(s).replace(" ", ""))

ctrl_candidates = []
seen_ctrl_pairs = set()

for s_file, s_name in [(S1_FILE, "Concepticon"), (S2_FILE, "Wiktionary")]:
    if not s_file.exists():
        continue
    for r in csv.DictReader(open(s_file, encoding="utf-8")):
        cand_id = str(r.get("cand_id", "")).strip()
        if cand_id in reserved_ctrl_cand_ids:
            continue
        ru = str(r.get("RU", "")).strip().lower()
        en = str(r.get("EN", "")).strip().lower()
        if not ru or not en:
            continue
        if (en, ru) in manual_exclusions_pairs or en in manual_exclusions_terms or ru in manual_exclusions_terms:
            continue
        if en in baseline_ctrl_en or ru in baseline_ctrl_ru:
            continue
        rej = r.get("reject_reason", "")
        if rej and str(rej).strip():
            continue
        if (ru, en) in seen_ctrl_pairs:
            continue
        seen_ctrl_pairs.add((ru, en))
        
        w_ru = len(ru.split())
        w_en = len(en.split())
        if w_ru > 3 or w_en > 3:
            continue
            
        ctrl_candidates.append({
            "cand_id": cand_id,
            "source": s_name,
            "source_ref": r.get("source_ref", ""),
            "concepticon_id": r.get("concepticon_id", ""),
            "concepticon_gloss": r.get("concepticon_gloss", ""),
            "ctrl_RU": ru,
            "ctrl_EN": en,
            "w_ru": w_ru,
            "w_en": w_en,
            "chars_ru": chars_len(ru),
            "chars_en": chars_len(en),
            "exact_cell": f"{w_ru}_{w_en}",
            "semantic_category": r.get("semantic_category", ""),
            "ru_variants": r.get("ru_variants", "")
        })

print(f"Available fresh control candidates: {len(ctrl_candidates)}")
cand_by_cell = defaultdict(list)
for c in ctrl_candidates:
    cand_by_cell[c["exact_cell"]].append(c)

for cell, lst in sorted(cand_by_cell.items()):
    print(f"  Control cell {cell}: {len(lst)} candidates")

# 3. Загрузка технических понятий для сопоставления
# А: Новые 90 понятий
df_tech_new = pd.read_csv(TECH_NEW_FILE)

# Б: 14 непарных понятий из базового пилота (если есть)
df_tech_comb = pd.read_csv(TECH_COMB_FILE)
matched_baseline_tech_ids = set(df_base_match["tech_concept_id"].astype(str))
unpaired_baseline = df_tech_comb[
    (df_tech_comb["cohort"] != "wipo_expansion_v1") &
    (~df_tech_comb["concept_id"].astype(str).isin(matched_baseline_tech_ids))
].copy()
print(f"Unpaired baseline concepts to attempt: {len(unpaired_baseline)}")

# Greedy matching algorithm
CALIPER = 4  # Strict protocol caliper <= 4

def match_tech_list(tech_df: pd.DataFrame, is_repaired: bool) -> tuple[list[dict], list[dict]]:
    matched = []
    unmatched = []
    used_ctrl_en = set()
    used_ctrl_ru = set()
    
    # Sort cells by deficit (fewest available control candidates per tech concept)
    tech_rows = tech_df.to_dict("records")
    tech_cells = Counter(r["exact_cell"] for r in tech_rows)
    
    def cell_scarcity(r):
        cell = r["exact_cell"]
        avail = len(cand_by_cell.get(cell, []))
        demand = tech_cells[cell]
        ratio = avail / demand if demand > 0 else 999
        return (ratio, r["concept_id"])
        
    tech_sorted = sorted(tech_rows, key=cell_scarcity)
    
    for t in tech_sorted:
        cell = t["exact_cell"]
        t_chars_ru = chars_len(t["term_ru_sg"])
        t_chars_en = chars_len(t["term_en_sg"])
        
        eligible_cands = []
        for c in cand_by_cell.get(cell, []):
            if c["ctrl_EN"] in used_ctrl_en or c["ctrl_RU"] in used_ctrl_ru:
                continue
            d_ru_abs = abs(t_chars_ru - c["chars_ru"])
            d_en_abs = abs(t_chars_en - c["chars_en"])
            if d_ru_abs <= CALIPER and d_en_abs <= CALIPER:
                # Check family limit <= 3
                keys = extract_ctrl_keys(c["ctrl_RU"], c["ctrl_EN"])
                fsize = ctrl_fam.family_size_with_candidate(c["cand_id"], keys)
                if fsize <= 3:
                    score = d_ru_abs + d_en_abs
                    d_ru_signed = c["chars_ru"] - t_chars_ru
                    d_en_signed = c["chars_en"] - t_chars_en
                    eligible_cands.append((score, d_ru_signed, d_en_signed, c["cand_id"], c, keys))
                    
        if eligible_cands:
            # Pick best by minimum score, then candidate ID
            eligible_cands.sort(key=lambda x: (x[0], x[3]))
            best_score, d_ru_s, d_en_s, cand_id, best_c, best_keys = eligible_cands[0]
            
            # Remove candidate from pool and mark terms used
            cand_by_cell[cell].remove(best_c)
            used_ctrl_en.add(best_c["ctrl_EN"])
            used_ctrl_ru.add(best_c["ctrl_RU"])
            ctrl_fam.add_concept(cand_id, best_keys)
            
            match_row = {
                "tech_concept_id": t["concept_id"],
                "tech_cohort": t.get("cohort", "wipo_expansion_v1"),
                "tech_family_id": t["family_id"],
                "stratum": t["stratum"],
                "exact_cell": cell,
                "RU_words": t["w_ru_sg"],
                "EN_words": t["w_en_sg"],
                "tech_RU": t["term_ru_sg"],
                "tech_EN": t["term_en_sg"],
                "ctrl_cand_id": best_c["cand_id"],
                "ctrl_family_id": f"CFAM_{ctrl_fam.find(f'c:{cand_id}').replace(':', '_').replace('k_', '')}",
                "ctrl_RU": best_c["ctrl_RU"],
                "ctrl_EN": best_c["ctrl_EN"],
                "tech_RU_chars": t_chars_ru,
                "ctrl_RU_chars": best_c["chars_ru"],
                "tech_EN_chars": t_chars_en,
                "ctrl_EN_chars": best_c["chars_en"],
                "d_RU_chars": d_ru_s,
                "d_EN_chars": d_en_s,
                "score": best_score,
                "ctrl_source": best_c["source"],
                "ctrl_source_ref": best_c["source_ref"],
                "ctrl_category": best_c["semantic_category"],
                "ctrl_variants": best_c["ru_variants"],
                "is_historical_repaired": is_repaired
            }
            matched.append(match_row)
        else:
            unmatched.append({
                "tech_concept_id": t["concept_id"],
                "tech_cohort": t.get("cohort", "wipo_expansion_v1"),
                "stratum": t["stratum"],
                "exact_cell": cell,
                "tech_RU": t["term_ru_sg"],
                "tech_EN": t["term_en_sg"],
                "tech_RU_chars": t_chars_ru,
                "tech_EN_chars": t_chars_en,
                "reason": f"No eligible candidate in cell {cell} within caliper {CALIPER} and family<=3"
            })
            
    return matched, unmatched

# Match new technical concepts (90)
new_matches, new_unmatched = match_tech_list(df_tech_new, is_repaired=False)
print(f"\nNew concepts matching result: {len(new_matches)} matched, {len(new_unmatched)} unmatched.")

# Match unpaired historical concepts (14)
repaired_matches, repaired_unmatched = match_tech_list(unpaired_baseline, is_repaired=True)
print(f"Historical repair matching result: {len(repaired_matches)} matched, {len(repaired_unmatched)} unmatched.")

# Обновляем ctrl_family_id для всех найденных пар через общий граф семей ctrl_fam
for m in new_matches:
    cand_id = str(m["ctrl_cand_id"])
    fam_root = ctrl_fam.find(f"c:{cand_id}")
    m["ctrl_family_id"] = f"CFAM_{fam_root.replace(':', '_').replace('k_', '')}"

for m in repaired_matches:
    cand_id = str(m["ctrl_cand_id"])
    fam_root = ctrl_fam.find(f"c:{cand_id}")
    m["ctrl_family_id"] = f"CFAM_{fam_root.replace(':', '_').replace('k_', '')}"

# Save matching pairs tables
df_match_new = pd.DataFrame(new_matches)
df_match_new.to_csv(DATA_DIR / "matching_pairs_new.csv", index=False, encoding="utf-8")

# Combine all matched pairs:
# Base 76 matches + New matches + Historical repaired
base_matches_formatted = []
for _, r in df_base_match.iterrows():
    cand_id = str(r["ctrl_cand_id"])
    fam_root = ctrl_fam.find(f"c:{cand_id}")
    ctrl_fam_id = f"CFAM_{fam_root.replace(':', '_').replace('k_', '')}"
    base_matches_formatted.append({
        "tech_concept_id": r["tech_concept_id"],
        "tech_cohort": "doc_pilot_60" if not str(r["tech_concept_id"]).startswith("WP") else "wipo_pilot_30",
        "tech_family_id": r.get("tech_family_id", ""),
        "stratum": r["stratum"],
        "exact_cell": f"{r['RU_words']}_{r['EN_words']}",
        "RU_words": r["RU_words"],
        "EN_words": r["EN_words"],
        "tech_RU": r["tech_RU"],
        "tech_EN": r["tech_EN"],
        "ctrl_cand_id": r["ctrl_cand_id"],
        "ctrl_family_id": ctrl_fam_id,
        "ctrl_RU": r["ctrl_RU"],
        "ctrl_EN": r["ctrl_EN"],
        "tech_RU_chars": r["tech_RU_chars"],
        "ctrl_RU_chars": r["ctrl_RU_chars"],
        "tech_EN_chars": r["tech_EN_chars"],
        "ctrl_EN_chars": r["ctrl_EN_chars"],
        "d_RU_chars": r["d_RU_chars"],
        "d_EN_chars": r["d_EN_chars"],
        "score": abs(r["d_RU_chars"]) + abs(r["d_EN_chars"]),
        "ctrl_source": r.get("ctrl_source", ""),
        "ctrl_source_ref": r.get("ctrl_source_ref", ""),
        "ctrl_category": r.get("ctrl_category", ""),
        "ctrl_variants": r.get("ctrl_variants", ""),
        "is_historical_repaired": False
    })

all_matched_rows = base_matches_formatted + new_matches + repaired_matches
df_match_combined = pd.DataFrame(all_matched_rows)
df_match_combined.to_csv(DATA_DIR / "matching_pairs_combined.csv", index=False, encoding="utf-8")
print(f"Combined matched pairs saved: {len(df_match_combined)} (Baseline: 76, New: {len(new_matches)}, Repaired: {len(repaired_matches)})")

# Save unmatched concepts table
all_unmatched = new_unmatched + repaired_unmatched
df_unmatched = pd.DataFrame(all_unmatched)
df_unmatched.to_csv(DATA_DIR / "unmatched_technical.csv", index=False, encoding="utf-8")
print(f"Unmatched technical concepts: {len(df_unmatched)} saved to {DATA_DIR / 'unmatched_technical.csv'}")

# Build concepts_control_expanded.csv for measurement runner
ctrl_rows = []
for idx, r in enumerate(all_matched_rows, 1):
    ctrl_rows.append({
        "concept_id": f"CTRL_{r['ctrl_cand_id']}",
        "family_id": r["ctrl_family_id"],
        "split": "expansion",
        "stratum": r["stratum"],
        "domain": "everyday_control",
        "definition": f"Control concept from {r.get('ctrl_source', '')} ({r.get('ctrl_category', '')})",
        "term_ru_sg": r["ctrl_RU"],
        "term_ru_pl": "",
        "term_en_sg": r["ctrl_EN"],
        "term_en_pl": "",
        "head_ru": str(r["ctrl_RU"]).split()[0],
        "head_en": str(r["ctrl_EN"]).split()[-1],
        "source_ru": r["ctrl_source"],
        "source_en": r["ctrl_source"],
        "source_entry": r["ctrl_source_ref"],
        "edition": "2026-09-EXP",
        "access_date": "2026-09-20",
        "review_technical": "eligible_by_ai",
        "review_language": "eligible_by_ai",
        "final_status": "accepted",
        "decision_note": f"Matched to tech {r['tech_concept_id']}",
        "alternatives": "",
        "exclusion_reason": "",
        "selection_order": f"CTRL_{idx:03d}",
        "term_pattern_ru": "N" if r["RU_words"] == 1 else "Adj+N",
        "term_pattern_en": "N" if r["EN_words"] == 1 else "Adj+N",
        "changed_components_ru": 0,
        "changed_components_en": 0,
        "inflection_note_ru": "Control measured in SG only",
        "inflection_note_en": "Control measured in SG only",
        "pl_status": "not_applicable",
        "matched_tech_concept_id": r["tech_concept_id"],
        "matched_tech_family_id": r.get("tech_family_id", ""),
        "concepticon_id": r.get("concepticon_id", "")
    })
df_ctrl_expanded = pd.DataFrame(ctrl_rows)
df_ctrl_expanded.to_csv(DATA_DIR / "concepts_control_expanded.csv", index=False, encoding="utf-8")
print(f"Control expanded concepts saved: {len(df_ctrl_expanded)}")

# Matching quality statistics
quality = {
    "total_matched": len(df_match_combined),
    "baseline_matches": 76,
    "new_matches": len(new_matches),
    "historical_repaired_matches": len(repaired_matches),
    "total_unmatched": len(df_unmatched),
    "match_rate_combined": len(df_match_combined) / (len(df_match_combined) + len(df_unmatched)),
    "signed_mean_d_RU_chars": float(df_match_combined["d_RU_chars"].mean()),
    "signed_mean_d_EN_chars": float(df_match_combined["d_EN_chars"].mean()),
    "mean_abs_d_RU_chars": float(df_match_combined["d_RU_chars"].abs().mean()),
    "mean_abs_d_EN_chars": float(df_match_combined["d_EN_chars"].abs().mean()),
    "max_abs_d_RU_chars": int(df_match_combined["d_RU_chars"].abs().max()),
    "max_abs_d_EN_chars": int(df_match_combined["d_EN_chars"].abs().max()),
    "stratum_distribution": df_match_combined["stratum"].value_counts().to_dict(),
    "cohort_distribution": df_match_combined["tech_cohort"].value_counts().to_dict()
}
with open(DATA_DIR / "matching_quality.json", "w", encoding="utf-8") as f:
    json.dump(quality, f, ensure_ascii=False, indent=2)
print("Matching quality summary:")
print(json.dumps(quality, indent=2, ensure_ascii=False))
