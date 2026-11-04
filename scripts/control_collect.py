# -*- coding: utf-8 -*-
"""Сбор кандидатов общеязыкового контроля из двух документальных источников.

Зачем это нужно. Главный вопрос исследования — есть ли у технической
терминологии **дополнительное** русско-английское различие в токенной длине
сверх того, которое есть у обычной лексики. Чтобы это проверить, нужна вторая
выборка: обычные слова и устойчивые выражения, у которых русское и английское
обозначение называют одно и то же понятие, и эта эквивалентность взята из
источника, а не придумана.

Два источника, оба открытые и воспроизводимые:

S1. **Concepticon, список Miklashevsky-2018-506** — 506 русских
    существительных из опубликованного нормирующего исследования, с
    английскими соответствиями, семантической категорией, конкретностью и
    частотностью. Каждая запись связана с понятием Concepticon, то есть
    соответствие подтверждено ещё и независимой привязкой.
    Отсюда берутся однословные пары (группа A).

S2. **Английский Викисловарь, таблицы переводов** — для устойчивых бытовых
    выражений из нескольких слов. Русское соответствие берётся из самой
    статьи, из шаблона перевода; исследователь ничего не переводит сам.
    Отсюда берутся пары из 2–3 слов и пары с разным числом слов в языках
    (группы B и C).

Что скрипт НЕ делает: не запускает токенизатор и не смотрит на длины в
токенах. Это требование порядка работ: состав контроля должен быть закрыт до
измерения, иначе выборку можно невольно подогнать под желаемый результат.

Запуск:
    .venv\\Scripts\\python.exe scripts\\control_collect.py --max-pages 40   # проба
    .venv\\Scripts\\python.exe scripts\\control_collect.py                  # полный сбор
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CTRL = ROOT / "Слова_данные" / "CONTROL"
RAW = CTRL / "raw"
RAW_WIKT = RAW / "wiktionary"
MIKL = RAW / "Miklashevsky-2018-506.tsv"
OUT_S1 = CTRL / "control_candidates_s1.csv"
OUT_S2 = CTRL / "control_candidates_s2.csv"
LOG = CTRL / "collect_log.json"

API = "https://en.wiktionary.org/w/api.php"
UA = "RuEnTokenizationStudy/1.0 (academic research; low volume)"
ACCESS_DATE = "2026-09-18"

# Бытовые предметные области. Технические категории (машины, инструменты,
# электроника, транспортные узлы) сюда намеренно не входят: контроль должен
# быть общеязыковым, иначе сравнение теряет смысл.
CATEGORIES = [
    "Category:en:Foods",
    "Category:en:Fruits",
    "Category:en:Vegetables",
    "Category:en:Beverages",
    "Category:en:Dishes",
    "Category:en:Clothing",
    "Category:en:Headwear",
    "Category:en:Footwear",
    "Category:en:Furniture",
    "Category:en:Rooms",
    "Category:en:Buildings",
    "Category:en:Home",
    "Category:en:Cooking",
    "Category:en:Tableware",
    "Category:en:Cookware and bakeware",
    "Category:en:Home appliances",
    "Category:en:Bedding",
    "Category:en:Hygiene",
    "Category:en:Domestic animals",
    "Category:en:Family",
    "Category:en:Body parts",
    "Category:en:Weather",
    "Category:en:Time",
    "Category:en:Roads",
    "Category:en:Shops",
]

# Головные слова технического корпуса и их английские пары: кандидат с такой
# вершиной не может считаться общеязыковым, даже если попал в бытовую
# категорию. Список подгружается из данных, здесь только ручные добавки.
MANUAL_TECH_HEADS_EN = {
    "sensor", "valve", "drive", "actuator", "motor", "bearing", "gauge",
    "pump", "switch", "relay", "transformer", "amplifier", "converter",
    "encoder", "controller", "cylinder", "gear", "battery", "fuse",
    "resistor", "electrode", "coil", "conveyor", "compressor", "nozzle",
}
MANUAL_TECH_HEADS_RU = {
    "датчик", "клапан", "привод", "двигатель", "подшипник", "насос",
    "реле", "трансформатор", "усилитель", "энкодер", "цилиндр", "редуктор",
    "аккумулятор", "предохранитель", "резистор", "электрод", "катушка",
    "конвейер", "компрессор", "форсунка", "муфта", "фильтр",
}

STRESS = "́̀"
ALLOWED_RU = re.compile(r"^[а-яёА-ЯЁ]+(?: [а-яёА-ЯЁ]+){0,2}$")
ALLOWED_EN = re.compile(r"^[a-z]+(?: [a-z]+){0,2}$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def api_get(params: dict, cache: Path | None) -> dict:
    """Запрос к API с кэшем на диске: повторный запуск не ходит в сеть."""
    if cache is not None and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    params = dict(params, format="json", formatversion="2")
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    last = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            time.sleep(1.0)
            return data
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(5.0 * (attempt + 1))
    raise RuntimeError(f"нет ответа от API: {last}")


def category_members(cat: str) -> list[str]:
    titles: list[str] = []
    cont = None
    page = 0
    while True:
        params = {"action": "query", "list": "categorymembers", "cmtitle": cat,
                  "cmlimit": "500", "cmnamespace": "0"}
        if cont:
            params["cmcontinue"] = cont
        slug = re.sub(r"[^A-Za-z0-9]+", "_", cat)
        data = api_get(params, RAW_WIKT / "cat" / f"{slug}_{page}.json")
        titles += [m["title"] for m in data.get("query", {}).get("categorymembers", [])]
        cont = data.get("continue", {}).get("cmcontinue")
        page += 1
        if not cont:
            return titles


def clean_ru(raw: str) -> str:
    s = unicodedata.normalize("NFC", raw)
    s = "".join(ch for ch in s if ch not in STRESS)
    s = s.replace("ё", "ё").strip()
    s = re.sub(r"\s+", " ", s)
    return s


TRANS_RU = re.compile(r"\{\{t\+?\*?\|ru\|([^|}]+)")
TOP = re.compile(r"\{\{trans-top(?:-also)?\|([^}]*)\}\}")


def parse_wikitext(title: str, text: str, revid) -> dict | None:
    """Разбор статьи: берутся переводы ТОЛЬКО из раздела существительного.

    Это существенно. В статье `foreign` есть и прилагательное, и
    существительное, а таблица переводов у прилагательного своя. Если брать
    первую попавшуюся таблицу в английском разделе, в выборку попадает
    «иностранный» — прилагательное, которое по §14.3 в контроль не годится.
    Поэтому английский раздел сначала режется по заголовкам частей речи, и
    дальше рассматривается только кусок, относящийся к существительному.
    """
    m = re.search(r"^==\s*English\s*==\s*$", text, flags=re.M)
    if not m:
        return None
    rest = text[m.end():]
    nxt = re.search(r"^==\s*[^=].*?\s*==\s*$", rest, flags=re.M)
    en_sec = rest[: nxt.start()] if nxt else rest

    # границы разделов частей речи внутри английской статьи
    heads = [(mm.start(), mm.group(1).strip())
             for mm in re.finditer(r"^===\s*([^=]+?)\s*===\s*$", en_sec, flags=re.M)]
    noun_chunks = []
    for i, (pos, name) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(en_sec)
        if name.lower() == "noun":
            noun_chunks.append(en_sec[pos:end])
    if not noun_chunks:
        return None

    ru: list[str] = []
    gloss = ""
    for chunk in noun_chunks:
        for block in re.split(r"\{\{trans-bottom\}\}", chunk):
            g = TOP.search(block)
            if not g:
                continue
            found = [clean_ru(x) for x in TRANS_RU.findall(block)]
            if found and not ru:
                gloss = g.group(1).strip()
                ru = found
            elif found:
                ru += found
    if not ru:
        return None
    seen, uniq = set(), []
    for x in ru:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return {"title": title, "revid": revid, "gloss": gloss, "ru": uniq}


BATCH = 50          # столько статей отдаёт один запрос к API


def fetch_batch(titles: list[str]) -> list[dict]:
    """Тексты сразу для полусотни статей.

    Это не ускорение ради ускорения: поштучные запросы упираются в ограничение
    частоты обращений, и загрузка тысячи статей занимает часы. Один запрос на
    полсотни заголовков и пауза между запросами укладываются в обычные правила
    работы с API.
    """
    slug = hashlib.sha256("|".join(titles).encode("utf-8")).hexdigest()[:16]
    data = api_get({"action": "query", "prop": "revisions", "rvprop": "content|ids",
                    "rvslots": "main", "titles": "|".join(titles)},
                   RAW_WIKT / "batch" / f"{slug}.json")
    out = []
    for page in data.get("query", {}).get("pages", []):
        if "revisions" not in page:
            continue
        rev = page["revisions"][0]
        text = rev.get("slots", {}).get("main", {}).get("content", "")
        rec = parse_wikitext(page["title"], text, rev.get("revid", ""))
        if rec:
            out.append(rec)
    return out


def tech_heads() -> tuple[set[str], set[str]]:
    en, ru = set(MANUAL_TECH_HEADS_EN), set(MANUAL_TECH_HEADS_RU)
    path = ROOT / "data" / "real" / "concepts_pilot.csv"
    if path.exists():
        for r in csv.DictReader(path.open(encoding="utf-8")):
            if r.get("head_en"):
                en.add(r["head_en"].strip().lower())
            if r.get("head_ru"):
                ru.add(r["head_ru"].strip().lower())
    return en, ru


def reject(en: str, ru: str, heads_en: set[str], heads_ru: set[str]) -> str:
    if not ALLOWED_EN.match(en):
        return "en_form"          # заглавные буквы, дефис, цифры, >3 слов
    if not ALLOWED_RU.match(ru):
        return "ru_form"
    if len(en.split()) > 3 or len(ru.split()) > 3:
        return "too_long"
    if en.split()[-1] in heads_en or ru.split()[-1] in heads_ru:
        return "technical_head"   # вершина совпадает с вершиной техтермина
    return ""


def build_s1(heads_en: set[str], heads_ru: set[str]) -> list[dict]:
    rows = []
    with MIKL.open(encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            en = (r["ENGLISH"] or "").strip().lower()
            ru = clean_ru(r["RUSSIAN"] or "").lower()
            why = reject(en, ru, heads_en, heads_ru)
            rows.append({
                "cand_id": "S1_" + r["NUMBER"],
                "source": "Concepticon Miklashevsky-2018-506",
                "source_ref": r["ID"],
                "concepticon_id": r["CONCEPTICON_ID"],
                "concepticon_gloss": r["CONCEPTICON_GLOSS"],
                "EN": en, "RU": ru,
                "EN_words": len(en.split()), "RU_words": len(ru.split()),
                "semantic_category": r["SEMANTIC_CATEGORY"],
                "concreteness": r["CONCRETENESS"],
                "source_freq": r["OBJECTIVE_WORD_FREQUENCY"],
                "ru_variants": "",
                "access_date": ACCESS_DATE,
                "reject_reason": why,
            })
    return rows


def build_s2(heads_en: set[str], heads_ru: set[str], max_pages: int | None) -> list[dict]:
    titles: dict[str, str] = {}
    for cat in CATEGORIES:
        for t in category_members(cat):
            titles.setdefault(t, cat)

    # Порядок обращения к источнику: сначала самые обиходные выражения.
    # Обиходность оценивается минимальной частотностью компонентов по
    # wordfreq — это признак самого слова, а не его длины в токенах, поэтому
    # порядок не может подогнать выборку под результат. Критерии включения
    # (§14.3) от порядка не зависят.
    from wordfreq import zipf_frequency

    def commonness(title: str) -> float:
        words = [w for w in title.split(" ") if w]
        return min((zipf_frequency(w, "en") for w in words), default=0.0)

    usable = [t for t in titles if ALLOWED_EN.match(t)]
    multi = sorted((t for t in usable if " " in t), key=commonness, reverse=True)
    single = sorted((t for t in usable if " " not in t), key=commonness, reverse=True)
    if max_pages:
        ordered = multi[: int(max_pages * 0.7)] + single[: int(max_pages * 0.3)]
    else:
        ordered = multi + single

    fetched: list[dict] = []
    for i in range(0, len(ordered), BATCH):
        chunk = ordered[i:i + BATCH]
        try:
            fetched += fetch_batch(chunk)
        except RuntimeError as exc:
            print(f"  пропуск партии {i // BATCH}: {exc}")
        if (i // BATCH) % 5 == 0:
            print(f"  загружено партий: {i // BATCH + 1}, пригодных статей: {len(fetched)}")

    rows = []
    for i, rec in enumerate(fetched, start=1):
        title = rec["title"]
        en = title.strip().lower()
        variants = rec["ru"]
        ru = variants[0].lower()
        why = reject(en, ru, heads_en, heads_ru)
        rows.append({
            "cand_id": f"S2_{i:04d}",
            "source": "English Wiktionary translation table",
            "source_ref": f"{title} (rev {rec['revid']})",
            "concepticon_id": "", "concepticon_gloss": rec["gloss"],
            "EN": en, "RU": ru,
            "EN_words": len(en.split()), "RU_words": len(ru.split()),
            "semantic_category": titles.get(title, "").replace("Category:en:", ""),
            "concreteness": "", "source_freq": "",
            "ru_variants": "|".join(v.lower() for v in variants[1:]),
            "access_date": ACCESS_DATE,
            "reject_reason": why,
        })
    return rows


def write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=0,
                    help="ограничение числа статей Викисловаря (проба)")
    args = ap.parse_args()

    CTRL.mkdir(parents=True, exist_ok=True)
    heads_en, heads_ru = tech_heads()

    s1 = build_s1(heads_en, heads_ru)
    s2 = build_s2(heads_en, heads_ru, args.max_pages or None)
    write(OUT_S1, s1)
    write(OUT_S2, s2)

    ok1 = [r for r in s1 if not r["reject_reason"]]
    ok2 = [r for r in s2 if not r["reject_reason"]]

    def profile(rows):
        prof: dict[str, int] = {}
        for r in rows:
            key = f"{r['RU_words']}-{r['EN_words']}"
            prof[key] = prof.get(key, 0) + 1
        return dict(sorted(prof.items(), key=lambda kv: -kv[1]))

    log = {
        "access_date": ACCESS_DATE,
        "s1_file": str(MIKL.relative_to(ROOT)),
        "s1_sha256": sha256(MIKL),
        "s1_total": len(s1), "s1_accepted": len(ok1),
        "s1_profile_RUw_ENw": profile(ok1),
        "s2_categories": CATEGORIES,
        "s2_pages_parsed": len(s2), "s2_accepted": len(ok2),
        "s2_profile_RUw_ENw": profile(ok2),
        "s2_with_ru_variants": sum(1 for r in ok2 if r["ru_variants"]),
        "max_pages": args.max_pages,
        "note": "токенизация на этом шаге не выполнялась",
    }
    LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"S1 (Concepticon): {len(ok1)} пригодных из {len(s1)}")
    print(f"   профиль слов RU-EN: {log['s1_profile_RUw_ENw']}")
    print(f"S2 (Викисловарь): {len(ok2)} пригодных из {len(s2)} разобранных статей")
    print(f"   профиль слов RU-EN: {log['s2_profile_RUw_ENw']}")
    print(f"   с альтернативными русскими обозначениями: {log['s2_with_ru_variants']}")
    print(f"журнал: {LOG.relative_to(ROOT)}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
