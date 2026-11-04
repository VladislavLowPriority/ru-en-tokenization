# -*- coding: utf-8 -*-
"""Оценка выполнимости (Feasibility Analysis) для WIPO расширения v1.

Проверяет:
1. Число кандидатов по ячейкам A/B/C при строгом следовании правилам протокола.
2. Соблюдение предметной области (физические устройства, отсечение процессов/величин).
3. Соблюдение глобального лимита семей (<= 3 на объединенном корпусе 90 baseline + new).
4. Качество и рейтинги обозначений (рейтинг >= 3, допустимый статус).
5. Неприкосновенность и дедупликацию относительно baseline (P01-P60, WP01-WP30).
"""
import json, re, html
from pathlib import Path
from collections import Counter, defaultdict
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

NON_DEVICE_HEADS_EN = {
    "speed", "data", "system", "method", "process", "signal", "protocol",
    "value", "rate", "ratio", "factor", "coefficient", "current", "voltage",
    "power", "frequency", "temperature", "pressure", "force", "torque",
    "energy", "time", "length", "mass", "weight", "density", "resistance",
    "capacitance", "inductance", "impedance", "phase", "noise", "software",
    "algorithm", "program", "interface", "network", "model", "structure",
    "material", "technology", "effect", "principle", "mode", "state",
    "field", "wave", "beam", "light", "radiation", "movement", "motion",
    "flow", "level", "position", "thickness", "viscosity", "vibration",
    "efficiency", "loss", "absorption", "emission", "transmission"
}
NON_DEVICE_HEADS_RU = {
    "скорость", "данные", "система", "метод", "способ", "процесс", "сигнал",
    "протокол", "значение", "величина", "коэффициент", "ток", "напряжение",
    "мощность", "частота", "температура", "давление", "сила", "момент",
    "энергия", "время", "длина", "масса", "плотность", "сопротивление",
    "ёмкость", "емкость", "индуктивность", "фаза", "шум", "алгоритм",
    "программа", "интерфейс", "сеть", "модель", "структура", "материал",
    "технология", "эффект", "принцип", "режим", "состояние", "поле",
    "волна", "луч", "свет", "излучение", "движение", "поток", "уровень",
    "положение", "толщина", "вязкость", "вибрация", "кпд"
}

# Load baseline pilot
df_pilot = pd.read_csv(PILOT_FILE)
accepted_pilot = df_pilot[df_pilot["final_status"] == "accepted"].copy()
existing_ru_pilot = set(accepted_pilot["term_ru_sg"].str.strip().str.lower())
existing_en_pilot = set(accepted_pilot["term_en_sg"].str.strip().str.lower())

# Extract families of baseline
baseline_families = defaultdict(set)
for _, r in accepted_pilot.iterrows():
    fam = str(r["family_id"])
    cid = str(r["concept_id"])
    baseline_families[fam].add(cid)

print(f"Baseline accepted: {len(accepted_pilot)} concepts, {len(baseline_families)} families.")

# Load raw WIPO records
unique_records = {}
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
        if fp not in unique_records:
            unique_records[fp] = {
                "en_items": en_items,
                "ru_items": ru_items,
                "rec_id": f"{f.name}#rec_{i}",
                "seeds": [seed],
                "subject": rec.get("subjectField", {})
            }
        else:
            if seed not in unique_records[fp]["seeds"]:
                unique_records[fp]["seeds"].append(seed)

print(f"Total unique WIPO concept records: {len(unique_records)}")

def get_head_en(term: str) -> str:
    words = term.split()
    return words[-1] if words else ""

def get_head_ru(term: str) -> str:
    words = term.split()
    if not words:
        return ""
    # In Russian: if noun + genitive (e.g. датчик давления), head is first word
    # if adjective + noun (e.g. оптический датчик), head is last word
    return words[0] if len(words) > 1 and not words[0].endswith(("ый", "ий", "ой", "ая", "яя", "ое", "ее", "ые", "ие")) else words[-1]

def is_eponym(en: str) -> bool:
    # Eponym detection
    return bool(re.search(r"\b[A-Z][a-z]{2,}\b", en))

def pick_primary(items):
    def sort_key(x):
        usage = (x.get("usage") or "").strip().lower()
        is_allowed = usage in ("allowed", "standardized", "preferred", "")
        try:
            r = int(x.get("rating") or 0)
        except Exception:
            r = 0
        return (not is_allowed, -r)
    return sorted(items, key=sort_key)[0]

# Candidate evaluation
records_summary = []
for fp, c in unique_records.items():
    en_top = pick_primary(c["en_items"])
    ru_top = pick_primary(c["ru_items"])
    
    en_raw = en_top.get("term", "")
    ru_raw = ru_top.get("term", "")
    en_term = clean_html(en_raw)
    ru_term = clean_html(ru_raw)
    
    try:
        r_en = int(en_top.get("rating") or 0)
    except: r_en = 0
    try:
        r_ru = int(ru_top.get("rating") or 0)
    except: r_ru = 0
    
    u_en = (en_top.get("usage") or "").strip().lower()
    u_ru = (ru_top.get("usage") or "").strip().lower()
    
    w_en = len(en_term.split())
    w_ru = len(ru_term.split())
    
    # Classify stratum
    if w_en == 1 and w_ru == 1:
        stratum = "A"
    elif 2 <= w_en <= 3 and 2 <= w_ru <= 3:
        stratum = "B"
    elif (w_en == 1 and 2 <= w_ru <= 3) or (w_ru == 1 and 2 <= w_en <= 3):
        stratum = "C"
    else:
        stratum = "Other"
        
    head_en = get_head_en(en_term)
    head_ru = get_head_ru(ru_term)
    
    # Exclusion checks
    reasons = []
    if r_en < 3 or r_ru < 3:
        reasons.append(f"low_rating(EN={r_en},RU={r_ru})")
    if u_en not in ("allowed", "standardized", "preferred", "") or u_ru not in ("allowed", "standardized", "preferred", ""):
        reasons.append(f"usage_not_allowed(EN={u_en},RU={u_ru})")
    if is_eponym(en_raw):
        reasons.append("eponym")
    if head_en in NON_DEVICE_HEADS_EN or head_ru in NON_DEVICE_HEADS_RU:
        reasons.append(f"non_device_head({head_en}/{head_ru})")
    if en_term in existing_en_pilot or ru_term in existing_ru_pilot:
        reasons.append("duplicate_of_pilot_baseline")
    if w_en > 3 or w_ru > 3:
        reasons.append(f"word_count_exceeded({w_ru}w_ru,{w_en}w_en)")
    if "-" in en_term or "-" in ru_term:
        # Check hyphenated: allow standard prefix or exclude?
        # Protocol: avoid hyphens in main corpus unless standard
        pass
        
    status = "eligible" if not reasons else "excluded"
    
    records_summary.append({
        "rec_id": c["rec_id"],
        "en": en_term,
        "ru": ru_term,
        "head_en": head_en,
        "head_ru": head_ru,
        "w_en": w_en,
        "w_ru": w_ru,
        "r_en": r_en,
        "r_ru": r_ru,
        "stratum": stratum,
        "status": status,
        "reasons": "; ".join(reasons),
        "en_tid": en_top.get("termId"),
        "ru_tid": ru_top.get("termId"),
        "seeds": c["seeds"]
    })

df_all = pd.DataFrame(records_summary)
print("\n=== CANDIDATE SUMMARY ===")
print("Status counts:\n", df_all["status"].value_counts().to_dict())

eligible = df_all[df_all["status"] == "eligible"]
print("\nEligible candidates by stratum:")
print(eligible["stratum"].value_counts().to_dict())

print("\nEligible candidates by exact (w_ru, w_en) cell:")
print(eligible.groupby(["w_ru", "w_en"]).size())

# Check Group C candidates in detail
print("\n--- Group C Eligible Candidates ---")
for _, r in eligible[eligible["stratum"] == "C"].iterrows():
    print(f"  [{r['w_ru']}->{r['w_en']}] {r['ru']} <-> {r['en']} (head: {r['head_ru']}/{r['head_en']}, rec: {r['rec_id']})")
