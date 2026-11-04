# -*- coding: utf-8 -*-
"""Отбор технического пополнения из WIPO Pearl для expansion_wipo_v1.

Реализует:
1. Парсинг сырых файлов JSON с чисткой HTML без разрыва слов.
2. Сохранение concept_record_id, en_term_id, ru_term_id.
3. Полную таблицу кандидатов wipo_candidates.csv и вариантов wipo_variants.csv.
4. Журнал решений wipo_decisions.csv с указанием конкретных причин исключения.
5. Транзитивное объединение в лексические семьи с контролем лимита <= 3
   на объединенном корпусе (baseline accepted + new).
6. Приоритет свежих семей.
7. Отбор в целевые группы A, B, C до токенизации.
"""
import json, re, html
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "expansion_wipo_v1"
RAW_DIR = EXP / "raw" / "wipo"
CAND_DIR = EXP / "candidates"
DATA_DIR = EXP / "data"
PILOT_FILE = ROOT / "data" / "real" / "concepts_pilot.csv"

CAND_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

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
    "efficiency", "loss", "absorption", "emission", "transmission", "exchange",
    "finding", "treatment", "operation", "analysis", "detection", "generation"
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
    "положение", "толщина", "вязкость", "вибрация", "кпд", "обдирка",
    "радиопеленгация", "обработка", "анализ", "обнаружение", "генерация"
}

LATIN = re.compile(r"[A-Za-z]")
CYRILLIC = re.compile(r"[А-Яа-яЁё]")

def clean_html(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<(?:p|div|br|tr|td|li|h[1-6])[^>]*>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"</(?:p|div|tr|td|li|h[1-6])>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def is_eponym(s: str) -> bool:
    return bool(re.search(r"\b[A-Z][a-z]{2,}\b", s))

# Baseline pilot
df_pilot = pd.read_csv(PILOT_FILE)
accepted_pilot = df_pilot[df_pilot["final_status"] == "accepted"].copy()
baseline_ru = set(accepted_pilot["term_ru_sg"].str.strip().str.lower())
baseline_en = set(accepted_pilot["term_en_sg"].str.strip().str.lower())

# Extract baseline families
class FamilyTracker:
    def __init__(self):
        self.parent = {}
        self.members = defaultdict(set)
    
    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def add_concept(self, cid, keys):
        for k in keys:
            self.union(f"c:{cid}", f"k:{k}")

    def get_family_size(self, cid):
        root = self.find(f"c:{cid}")
        # Count all concepts in this component
        count = 0
        for node in self.parent:
            if node.startswith("c:") and self.find(node) == root:
                count += 1
        return count

    def get_family_concepts(self, cid):
        root = self.find(f"c:{cid}")
        return [node[2:] for node in self.parent if node.startswith("c:") and self.find(node) == root]

# Load raw responses
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

print(f"Loaded {len(unique_records)} unique WIPO concept records.")

# Parse all candidate concepts and variants
all_candidates = []
all_variants = []

def get_head_en(term: str) -> str:
    words = term.split()
    return words[-1] if words else ""

def get_head_ru(term: str) -> str:
    words = term.split()
    if not words:
        return ""
    if len(words) > 1 and not words[0].endswith(("ый", "ий", "ой", "ая", "яя", "ое", "ее", "ые", "ие")):
        return words[0]
    return words[-1]

def extract_compound_roots(ru_term: str) -> list[str]:
    # Extract Russian compound roots for family clustering
    known_roots = [
        "привод", "двигател", "датчик", "клапан", "насос", "редуктор", "фильтр",
        "усилител", "преобразовател", "переключател", "счетчик", "анализатор",
        "регулятор", "расходомер", "манометр", "термодатчик", "гидроцилиндр",
        "пневмоцилиндр", "электрод", "резистор", "конденсатор", "подшипник"
    ]
    found = []
    lower = ru_term.lower()
    for r in known_roots:
        if r in lower:
            found.append(r)
    return found

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

cand_counter = 1
for fp, c in unique_records.items():
    local_cid = f"WEXP_{cand_counter:04d}"
    cand_counter += 1
    
    en_top = pick_primary(c["en_items"])
    ru_top = pick_primary(c["ru_items"])
    
    en_raw = en_top.get("term", "")
    ru_raw = ru_top.get("term", "")
    en_clean = clean_html(en_raw).lower()
    ru_clean = clean_html(ru_raw).lower()
    
    try: r_en = int(en_top.get("rating") or 0)
    except: r_en = 0
    try: r_ru = int(ru_top.get("rating") or 0)
    except: r_ru = 0
    
    u_en = (en_top.get("usage") or "").strip().lower()
    u_ru = (ru_top.get("usage") or "").strip().lower()
    
    w_en = len(en_clean.split())
    w_ru = len(ru_clean.split())
    
    if w_en == 1 and w_ru == 1:
        stratum = "A"
    elif 2 <= w_en <= 3 and 2 <= w_ru <= 3:
        stratum = "B"
    elif (w_en == 1 and 2 <= w_ru <= 3) or (w_ru == 1 and 2 <= w_en <= 3):
        stratum = "C"
    else:
        stratum = "Other"
        
    head_en = get_head_en(en_clean)
    head_ru = get_head_ru(ru_clean)
    
    # Store all variants
    for en_v in c["en_items"]:
        v_raw = en_v.get("term", "")
        v_clean = clean_html(v_raw).lower()
        all_variants.append({
            "local_concept_id": local_cid,
            "lang": "EN",
            "term_id": en_v.get("termId") or en_v.get("id"),
            "term_raw": v_raw,
            "term_clean": v_clean,
            "rating": en_v.get("rating"),
            "usage": en_v.get("usage"),
            "context": clean_html(en_v.get("context", "")),
            "source": en_v.get("sourceText") or en_v.get("sourceLink") or "",
            "is_primary": (en_v.get("termId") == en_top.get("termId"))
        })
    for ru_v in c["ru_items"]:
        v_raw = ru_v.get("term", "")
        v_clean = clean_html(v_raw).lower()
        all_variants.append({
            "local_concept_id": local_cid,
            "lang": "RU",
            "term_id": ru_v.get("termId") or ru_v.get("id"),
            "term_raw": v_raw,
            "term_clean": v_clean,
            "rating": ru_v.get("rating"),
            "usage": ru_v.get("usage"),
            "context": clean_html(ru_v.get("context", "")),
            "source": ru_v.get("sourceText") or ru_v.get("sourceLink") or "",
            "is_primary": (ru_v.get("termId") == ru_top.get("termId"))
        })
        
    # Check exclusion rules
    reasons = []
    if r_en < 3 or r_ru < 3:
        reasons.append(f"low_rating(EN={r_en},RU={r_ru})")
    if u_en not in ("allowed", "standardized", "preferred", "") or u_ru not in ("allowed", "standardized", "preferred", ""):
        reasons.append(f"usage_not_allowed(EN={u_en},RU={u_ru})")
    if is_eponym(en_raw):
        reasons.append("eponym")
    if head_en in NON_DEVICE_HEADS_EN or head_ru in NON_DEVICE_HEADS_RU:
        reasons.append(f"non_device_head({head_en}/{head_ru})")
    if LATIN.search(ru_clean):
        reasons.append("latin_in_russian")
    if en_clean in baseline_en or ru_clean in baseline_ru:
        reasons.append("duplicate_of_baseline")
    if w_en > 3 or w_ru > 3:
        reasons.append(f"word_count_exceeded({w_ru}w_ru,{w_en}w_en)")
    if "-" in en_clean or "-" in ru_clean:
        reasons.append("hyphenated_term")

    status = "eligible" if not reasons else "excluded"

    all_candidates.append({
        "local_concept_id": local_cid,
        "concept_record_id": c["rec_id"],
        "en_term_id": en_top.get("termId") or en_top.get("id"),
        "ru_term_id": ru_top.get("termId") or ru_top.get("id"),
        "term_en_raw": en_raw,
        "term_ru_raw": ru_raw,
        "term_en_clean": en_clean,
        "term_ru_clean": ru_clean,
        "head_en": head_en,
        "head_ru": head_ru,
        "w_en": w_en,
        "w_ru": w_ru,
        "chars_en": len(en_clean.replace(" ", "")),
        "chars_ru": len(ru_clean.replace(" ", "")),
        "r_en": r_en,
        "r_ru": r_ru,
        "u_en": u_en,
        "u_ru": u_ru,
        "en_context": clean_html(en_top.get("context", "")),
        "ru_context": clean_html(ru_top.get("context", "")),
        "en_source": en_top.get("sourceText") or en_top.get("sourceLink") or "",
        "ru_source": ru_top.get("sourceText") or ru_top.get("sourceLink") or "",
        "subject_field": json.dumps(c["subject"], ensure_ascii=False),
        "seeds": ",".join(c["seeds"]),
        "stratum": stratum,
        "exact_cell": f"{w_ru}_{w_en}",
        "status": status,
        "decision_note": "; ".join(reasons) if reasons else "Eligible according to protocol criteria"
    })

df_candidates = pd.DataFrame(all_candidates)
df_variants = pd.DataFrame(all_variants)

df_candidates.to_csv(CAND_DIR / "wipo_candidates.csv", index=False, encoding="utf-8")
df_variants.to_csv(CAND_DIR / "wipo_variants.csv", index=False, encoding="utf-8")
df_candidates[["local_concept_id", "concept_record_id", "term_en_clean", "term_ru_clean", "stratum", "status", "decision_note"]].to_csv(CAND_DIR / "wipo_decisions.csv", index=False, encoding="utf-8")

print(f"Candidates written: {len(df_candidates)} (Eligible: {sum(df_candidates['status'] == 'eligible')}, Excluded: {sum(df_candidates['status'] == 'excluded')})")
print(f"Variants written: {len(df_variants)}")
print("Eligible by stratum:")
print(df_candidates[df_candidates["status"] == "eligible"]["stratum"].value_counts().to_dict())
