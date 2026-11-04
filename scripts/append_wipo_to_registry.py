# -*- coding: utf-8 -*-
"""Дописывает отобранные пары WIPO Pearl в реестр отдельным разделом.

Реестр — главный документ исследователя, поэтому скрипт ничего в нём не
переписывает: он только добавляет новый раздел перед «Будущие источники».
Повторный запуск ничего не дублирует — если раздел уже есть, он заменяется
целиком на свежую версию.

Формат таблиц тот же, что у источников 1 и 2, поэтому `import_registry.py`
подхватывает новый раздел без каких-либо правок.

Запуск:
    .venv\\Scripts\\python.exe scripts\\append_wipo_to_registry.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "Слова_данные" / "bilingual_terms_registry.md"
SEL = ROOT / "Слова_данные" / "WIPO_PEARL" / "wipo_selection.csv"
PLURALS = ROOT / "Слова_данные" / "WIPO_PEARL" / "wipo_plural_forms.csv"

MARK_BEGIN = "<!-- WIPO_PEARL_SECTION_BEGIN -->"
MARK_END = "<!-- WIPO_PEARL_SECTION_END -->"
ID_PREFIX = "WP"


def esc(text: str) -> str:
    """Экранирует вертикальную черту, иначе она разорвёт ячейку таблицы."""
    return (text or "").replace("|", "/").replace("\n", " ").strip()


def main() -> None:
    sel = {r["record_id"]: r for r in csv.DictReader(SEL.open(encoding="utf-8"))}
    plurals = {r["record_id"]: r for r in csv.DictReader(PLURALS.open(encoding="utf-8"))}
    missing = [k for k in sel if k not in plurals]
    if missing:
        raise SystemExit("Нет форм мн. ч. для: " + ", ".join(missing))

    order = sorted(sel.values(), key=lambda r: (r["stratum"], r["EN_words"], r["EN"]))
    rows = []
    for i, r in enumerate(order, start=1):
        p = plurals[r["record_id"]]
        rows.append({
            "cid": f"{ID_PREFIX}{i:02d}",
            "rec": r["record_id"],
            "en_sg": r["EN"], "ru_sg": r["RU"],
            "en_pl": p["term_en_pl"], "ru_pl": p["term_ru_pl"],
            "stratum": r["stratum"],
            "field": r["subject_field"],
            "field_desc": r["subject_description"],
            "reliability": r["reliability"],
            "src": r["source_ru"] or r["source_en"],
            "syn_en": r["en_synonyms"], "syn_ru": r["ru_synonyms"],
            "pl_status": p["pl_status"], "pl_note": p["inflection_note_ru"],
            "head_en": r["EN"].split(" ")[-1].lower(),
            "head_ru": r["RU"].split(" ")[-1].lower(),
        })

    L = [MARK_BEGIN, "",
         "## Источник 3: WIPO Pearl (терминологическая база ВОИС)", "",
         "### Документальная база", "",
         "- **База:** WIPO Pearl, многоязычная терминологическая база Всемирной "
         "организации интеллектуальной собственности.",
         "- **Доступ:** REST API `https://api.wipo.int/wipopearl/records`, "
         "авторизация OAuth2 client_credentials. Дата обращения: 18.09.2026.",
         "- **Предметные области запроса:** ELEC, MEAS, MANU, MECH.",
         "- **Отбор:** обозначения EN и RU взяты из ОДНОЙ concept record, поэтому "
         "эквивалентность подтверждается самой базой. Переводы исследователем не "
         "выполнялись.",
         "- **Правило выбора обозначения** (объявлено до отбора): берётся вариант с "
         "наибольшей надёжностью среди помеченных `allowed`, при равенстве — первый "
         "в выдаче базы. Прочие обозначения сохранены в колонке «синонимы».",
         "- **Сырые ответы API** сохранены в `Слова_данные/WIPO_PEARL/raw/` "
         "(88 файлов); таблица кандидатов — `wipo_candidates.csv`, отбор — "
         "`wipo_selection.csv`.", "",
         "### 1. Рабочая таблица понятий WIPO Pearl (30 понятий)", "",
         "| concept_id | term_en_sg | term_ru_sg | structure_group | source_en | "
         "source_ru | section/page | equivalence_evidence | lexical_family | status |",
         "|---|---|---|---|---|---|---|---|---|---|"]

    for r in rows:
        src = f"WIPO Pearl record {r['rec']}"
        evidence = (f"Одна concept record WIPO Pearl {r['rec']}, предметная область "
                    f"{r['field']} ({esc(r['field_desc'])}), надёжность {r['reliability']}; "
                    f"источник цитаты: {esc(r['src'])[:60]}")
        L.append(f"| {r['cid']} | {r['en_sg']} | {r['ru_sg']} | {r['stratum']} "
                 f"| {src} | {src} | WIPO Pearl {r['rec']} | {evidence} "
                 f"| {r['head_en']} / {r['head_ru']} | accepted_pilot |")

    L += ["", "### 2. Экспериментальные формы множественного числа WIPO Pearl (SG → PL)", "",
          "> Формы образованы по правилу словоизменения от уже проверенного термина "
          "и записаны заранее по трём образцам (`N`, `Adj+N`, `N+N(gen)`). "
          "Эквивалентность EN↔RU при этом не затрагивается.", "",
          "| concept_id | term_en_sg | term_en_pl | term_ru_sg | term_ru_pl | "
          "pl_status | Лингвистический комментарий к форме PL |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['cid']} | {r['en_sg']} | {r['en_pl']} | {r['ru_sg']} "
                 f"| {r['ru_pl']} | {r['pl_status']} | {esc(r['pl_note'])} |")

    syn = [r for r in rows if r["syn_en"] or r["syn_ru"]]
    L += ["", "### 3. Альтернативные обозначения из тех же записей", "",
          "Источник даёт несколько засвидетельствованных обозначений. Они сохранены, "
          "потому что от выбора зависит число слов в термине, а значит и структурная "
          "группа. Правило выбора объявлено выше и применено ко всем записям "
          "одинаково.", "",
          "| concept_id | выбрано EN | другие EN | выбрано RU | другие RU |",
          "|---|---|---|---|---|"]
    for r in syn:
        L.append(f"| {r['cid']} | {r['en_sg']} | {esc(r['syn_en']) or '—'} "
                 f"| {r['ru_sg']} | {esc(r['syn_ru']) or '—'} |")

    L += ["", "### 4. Исключённые кандидаты WIPO Pearl", "",
          "| EN | RU | Причина |", "|---|---|---|",
          "| spool valve | золотник | В цитате, подтверждающей русское обозначение, "
          "«золотник» употреблён в значении ниппеля для накачивания полости, тогда как "
          "spool valve — золотниковый распределитель целиком; объёмы понятий не совпадают |",
          "| charge exchange | обдирка | Процесс перезарядки ионов, а не устройство |",
          "| radio direction finding | радиопеленгация | Метод, а не устройство |",
          "| Belleville washer | тарельчатая пружина | Имя собственное в термине (эпоним) |",
          "| Rogowski coil | пояс роговского | Имя собственное в термине (эпоним) |",
          "| gear speed | передача | Вершина термина — параметр, а не устройство |",
          "", MARK_END, ""]

    section = "\n".join(L)
    text = REGISTRY.read_text(encoding="utf-8")

    if MARK_BEGIN in text and MARK_END in text:
        start = text.index(MARK_BEGIN)
        end = text.index(MARK_END) + len(MARK_END) + 1
        text = text[:start] + section + text[end:]
        action = "раздел обновлён"
    else:
        anchor = "## Будущие источники"
        if anchor in text:
            i = text.index(anchor)
            text = text[:i] + section + "\n" + text[i:]
        else:
            text = text.rstrip() + "\n\n" + section
        action = "раздел добавлен"

    REGISTRY.write_text(text, encoding="utf-8")
    print(f"{action}: {len(rows)} понятий {ID_PREFIX}01–{ID_PREFIX}{len(rows):02d}")
    print(f"файл: {REGISTRY.relative_to(ROOT)}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
