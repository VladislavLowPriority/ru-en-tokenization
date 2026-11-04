# -*- coding: utf-8 -*-
"""Импорт реестра Слова_данные/bilingual_terms_registry.md в таблицу протокола.

Реестр — исходный документ исследователя. Скрипт ничего в нём не правит и
ничего не досочиняет: он переносит поля в схему §9 и выводит то, что
получается из самих данных механически.

Что делается и почему:
* строчное написание терминов — требование §6 п. 3; каждое изменение регистра
  записывается в журнал импорта;
* `changed_components` считается сравнением строк SG и PL, а не на глаз;
* `term_pattern` выводится из того, какие компоненты меняются при SG->PL:
  меняются оба слова — согласование (Adj+N), меняется только первое —
  неизменяемое зависимое (N+N(gen)). Это вывод из данных, а не разметка извне;
* `family_id` строится транзитивно по общей головной лемме в ЛЮБОМ из языков
  (§3 п. 5), поэтому две пары с общим русским головным словом попадают в одну
  семью, даже если английские головные слова у них разные.
"""
from __future__ import annotations

import csv
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tok_pilot.config import MAX_CONCEPTS_PER_FAMILY as MAX_PER_FAMILY  # noqa: E402
from tok_pilot.stimuli import REQUIRED_CONCEPT_COLUMNS  # noqa: E402

REGISTRY = ROOT / "Слова_данные" / "bilingual_terms_registry.md"
AUDIT = ROOT / "Слова_данные" / "FESTO_DGC" / "FESTO_DGC_bilingual_term_audit.md"
SOURCES_DIR = ROOT / "Слова_данные"
OUT_CSV = ROOT / "data" / "real" / "concepts_pilot.csv"
OUT_SOURCES = ROOT / "data" / "real" / "sources_log.csv"
OUT_LOG = ROOT / "data" / "real" / "import_log.md"

LOG: list[str] = []


def parse_tables(text: str) -> list[list[dict]]:
    """Все markdown-таблицы документа как списки словарей."""
    tables, rows, header = [], [], None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if header is None:
                header = cells
            elif set("".join(cells)) <= set("-: "):
                continue
            elif len(cells) == len(header):
                rows.append(dict(zip(header, cells)))
        else:
            if header is not None and rows:
                tables.append(rows)
            header, rows = None, []
    if header is not None and rows:
        tables.append(rows)
    return tables


def find_tables(tables, *required):
    """ВСЕ таблицы с нужными колонками, склеенные в один список.

    Реестр накопительный: когда добавится «Источник 2», его таблицы будут
    такими же по составу колонок. Брать только первую найденную нельзя —
    новые источники молча потерялись бы.
    """
    found = [r for t in tables if all(col in t[0] for col in required) for r in t]
    if not found:
        raise SystemExit(f"Не найдено ни одной таблицы с колонками {required}")
    return found


def parse_editions(text: str) -> dict[str, str]:
    """Соответствие «маркер документа [NNNN] -> revision» прямо из реестра.

    Так при добавлении нового руководства редакция подхватится сама, без
    правки кода.
    """
    out: dict[str, str] = {}
    current_rev = ""
    for line in text.splitlines():
        m = re.search(r"revision[:\s*]*\**\s*([\w.\-]+)\**", line, re.I)
        if m:
            current_rev = m.group(1)
        for marker in re.findall(r"\[(\d{4,})\]", line):
            if current_rev:
                out[marker] = current_rev
        m2 = re.search(r"revision\s+\*\*([\w.\-]+)\*\*.*?\[(\d{4,})\]", line, re.I)
        if m2:
            out[m2.group(2)] = m2.group(1)
    return out


def edition_for(source_field: str, editions: dict[str, str]) -> str:
    m = re.search(r"\[(\d{4,})\]", source_field or "")
    return editions.get(m.group(1), "") if m else ""


def domain_for(source_field: str, cid: str = "") -> str:
    s = (source_field or "").lower()
    c = (cid or "").lower()
    if "festo" in s or (c.startswith("p") and int(c[1:]) <= 18 if c[1:].isdigit() else False):
        return "промышленная пневмоавтоматика: линейный привод Festo DGC"
    if "siemens" in s or "s7" in s or c.startswith("s7_") or c.startswith("s7v2_") or (c.startswith("p") and (19 <= int(c[1:]) <= 24 or 49 <= int(c[1:]) <= 54) if c[1:].isdigit() else False):
        return "промышленная автоматизация: программируемый контроллер Siemens S7-1200"
    if "abb" in s or "ac500" in s or c.startswith("abb_") or (c.startswith("p") and 25 <= int(c[1:]) <= 29 if c[1:].isdigit() else False):
        return "промышленная автоматизация: модульный контроллер ABB AC500"
    if "sick" in s or (c.startswith("p") and 30 <= int(c[1:]) <= 35 if c[1:].isdigit() else False):
        return "промышленная сенсорика: оптические датчики SICK"
    if "endress" in s or (c.startswith("p") and 36 <= int(c[1:]) <= 40 if c[1:].isdigit() else False):
        return "промышленный КИПиА: датчики давления Endress+Hauser Cerabar PMP75"
    if "balluff" in s or (c.startswith("p") and 41 <= int(c[1:]) <= 45 if c[1:].isdigit() else False):
        return "промышленная сенсорика: магнитострикционные датчики Balluff BTL"
    if "schneider" in s or "shnider" in s or (c.startswith("p") and 46 <= int(c[1:]) <= 48 if c[1:].isdigit() else False):
        return "промышленная автоматизация: компоненты HMI и шлюзы Schneider Electric"
    if "flowsic" in s or (c.startswith("p") and int(c[1:]) == 55 if c[1:].isdigit() else False):
        return "промышленный КИПиА: ультразвуковые расходомеры Endress+Hauser"
    if (c.startswith("p") and 56 <= int(c[1:]) <= 58 if c[1:].isdigit() else False):
        return "промышленная пневмоавтоматика: вспомогательные компоненты Festo"
    if (c.startswith("p") and int(c[1:]) == 59 if c[1:].isdigit() else False):
        return "промышленная аналитика: газоаналитическое оборудование SICK"
    if (c.startswith("p") and int(c[1:]) == 60 if c[1:].isdigit() else False):
        return "промышленная автоматизация: сервоприводная техника"
    return "промышленная автоматизация"


def lower(term: str, cid: str, field: str) -> str:
    out = term.strip()
    if out != out.lower():
        LOG.append(f"- {cid} `{field}`: «{out}» → «{out.lower()}» "
                   f"(§6 п. 3, строчное написание нарицательных терминов)")
    return out.lower()


def changed_components(sg: str, pl: str) -> int:
    a, b = sg.split(" "), pl.split(" ")
    if len(a) != len(b):
        return -1                       # число компонентов изменилось: разбирать вручную
    return sum(1 for x, y in zip(a, b) if x != y)


def pattern(sg: str, pl: str, lang: str) -> str:
    """Структура термина, выведенная из поведения компонентов при SG->PL."""
    a, b = sg.split(" "), pl.split(" ")
    if len(a) != len(b):
        return "IRREGULAR"
    n = len(a)
    if n == 1:
        return "N"
    changed = [x != y for x, y in zip(a, b)]
    if lang == "ru":
        if n == 2:
            return "Adj+N" if all(changed) else "N+N(gen)"
        if n == 3:
            return "Adj+Adj+N" if all(changed) else "Mod+N(gen)"
    else:
        # в английском словоизменение несёт только вершина, тип определителя
        # из одного лишь словоизменения не выводится
        return "+".join(["Mod"] * (n - 1) + ["N"])
    return "+".join(["Mod"] * (n - 1) + ["N"])


MIN_ROOT = 4


def build_families(heads: dict[str, tuple[str, str]],
                   terms: dict[str, tuple[str, str]]) -> dict[str, str]:
    """Транзитивное объединение по общей головной лемме в любом из языков.

    Учитывается словосложение: и в русском, и в английском вершина сложного
    слова стоит СПРАВА, поэтому корень ищется как окончание слова.
    «электродвигатель» → двигатель, «гидропривод» → привод,
    preamplifier → amplifier. Поиск подстроки где угодно внутри слова здесь не
    годится: он склеивает «электродвигатель» с «электродом».

    Правило то же, что при отборе кандидатов (`scripts/wipo_select.py`),
    иначе число семей в анализе разойдётся с тем, по которому шёл отбор.
    """
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    known = {"en": set(), "ru": set()}
    for cid, (en, ru) in terms.items():
        for lang, t in (("en", en.lower()), ("ru", ru.lower())):
            words = [w for w in t.split(" ") if w]
            if words:
                known[lang].add(words[-1])
    for cid, (h_en, h_ru) in heads.items():
        known["en"].add(h_en.lower())
        known["ru"].add(h_ru.lower())

    for cid in heads:
        en, ru = terms.get(cid, ("", ""))
        keys = set()
        for lang, t in (("en", en.lower()), ("ru", ru.lower())):
            words = [w for w in t.split(" ") if w]
            if not words:
                continue
            keys.add(f"{lang}:{words[-1]}")
            for w in words:
                for root in known[lang]:
                    if len(root) >= MIN_ROOT and w.endswith(root):
                        keys.add(f"{lang}:{root}")
        if cid in heads:
            h_en, h_ru = heads[cid]
        else:
            # резервное понятие: вершины берутся из его же строки реестра
            famcol = r["lexical_family"]
            h_en, h_ru = ([x.strip() for x in famcol.split("/")] if "/" in famcol
                          else (famcol, famcol))
        keys.add(f"en:{h_en.lower()}")
        keys.add(f"ru:{h_ru.lower()}")
        for k in keys:
            union(f"c:{cid}", k)

    groups: dict[str, list[str]] = {}
    for cid in heads:
        groups.setdefault(find(f"c:{cid}"), []).append(cid)

    fam_of = {}
    for root, members in groups.items():
        first = sorted(members)[0]
        name = "fam_" + terms.get(first, ("", heads[first][1]))[1].lower().split(" ")[-1]
        for cid in members:
            fam_of[cid] = name
    return fam_of


def main() -> None:
    text = REGISTRY.read_text(encoding="utf-8")
    tables = parse_tables(text)
    editions = parse_editions(text)
    main_t = find_tables(tables, "concept_id", "term_en_sg", "term_ru_sg", "structure_group")
    pl_t = find_tables(tables, "concept_id", "term_en_pl", "term_ru_pl", "pl_status")

    seen = set()
    dup = sorted({r["concept_id"] for r in main_t
                  if r["concept_id"] in seen or seen.add(r["concept_id"])})
    if dup:
        raise SystemExit(
            "В реестре повторяются concept_id: " + ", ".join(dup) +
            "\nПри добавлении нового источника используйте новый префикс "
            "(например, S01, S02 для второго руководства).")

    pl_by_id = {r["concept_id"]: r for r in pl_t}
    # В реестре есть резервный пул: кандидаты со статусом, отличным от
    # accepted_pilot, форм мн. ч. пока не имеют. Это не ошибка — они просто
    # не входят в пилот. А вот пилотное понятие без форм PL — ошибка.
    missing_pl = [r["concept_id"] for r in main_t
                  if r["status"] == "accepted_pilot" and r["concept_id"] not in pl_by_id]
    if missing_pl:
        raise SystemExit(
            "У пилотных понятий нет форм множественного числа: " + ", ".join(missing_pl) +
            "\nДобавьте строки в таблицу «Экспериментальные формы множественного числа».")

    # Семьи строятся ТОЛЬКО по понятиям, которые идут в анализ. Резервное
    # понятие не должно связывать две семьи: в расчёт оно не входит, и
    # объединение через него завысило бы родство, а значит и ширину интервалов.
    fam_rows = [r for r in main_t if r["status"] == "accepted_pilot"]

    heads = {}
    for r in fam_rows:
        fam = r["lexical_family"]
        h_en, h_ru = [x.strip() for x in fam.split("/")] if "/" in fam else (fam, fam)
        heads[r["concept_id"]] = (h_en, h_ru)
    term_pairs = {r['concept_id']: (r['term_en_sg'], r['term_ru_sg']) for r in fam_rows}
    fam_of = build_families(heads, term_pairs)

    reg_hash = hashlib.sha256(REGISTRY.read_bytes()).hexdigest()
    rows = []
    for i, r in enumerate(main_t, start=1):
        cid = r["concept_id"]
        p = pl_by_id.get(cid, {})
        is_pilot = r["status"] == "accepted_pilot"
        ru_sg = lower(r["term_ru_sg"], cid, "term_ru_sg")
        en_sg = lower(r["term_en_sg"], cid, "term_en_sg")
        ru_pl = lower(p["term_ru_pl"], cid, "term_ru_pl") if p else ""
        en_pl = lower(p["term_en_pl"], cid, "term_en_pl") if p else ""
        if cid in heads:
            h_en, h_ru = heads[cid]
        else:
            # резервное понятие: вершины берутся из его же строки реестра
            famcol = r["lexical_family"]
            h_en, h_ru = ([x.strip() for x in famcol.split("/")] if "/" in famcol
                          else (famcol, famcol))

        ch_ru = changed_components(ru_sg, ru_pl) if ru_pl else ""
        ch_en = changed_components(en_sg, en_pl) if en_pl else ""
        if -1 in (ch_ru, ch_en):
            LOG.append(f"- {cid}: число компонентов меняется при SG→PL, требуется ручной разбор")

        rows.append({
            "concept_id": cid,
            "family_id": fam_of.get(cid, f"fam_reserve_{cid}"),
            "split": "pilot",
            "stratum": r["structure_group"],
            "domain": domain_for(r.get("source_en", ""), cid),
            "definition": r["equivalence_evidence"],
            "term_ru_sg": ru_sg, "term_ru_pl": ru_pl,
            "term_en_sg": en_sg, "term_en_pl": en_pl,
            "head_ru": h_ru.lower(), "head_en": h_en.lower(),
            # берётся из самой строки реестра, а не прописывается в коде:
            # иначе при добавлении второго источника у всех пар остался бы Festo
            "source_ru": r.get("source_ru", ""),
            "source_en": r.get("source_en", ""),
            "source_entry": r.get("section/page", ""),
            "edition": r.get("edition", "") or edition_for(r.get("source_en", ""), editions),
            "access_date": "2026-09-18",
            # В реестре зафиксирован один итоговый статус, а не две независимые
            # первичные оценки; см. журнал импорта.
            "review_technical": r["status"],
            "review_language": "not_recorded_separately",
            "final_status": "accepted" if is_pilot else "excluded",
            "decision_note": f"pl_status={p.get('pl_status', 'нет форм PL')}; "
                             f"{p.get('Лингвистический комментарий к форме PL', '')}",
            "alternatives": "",
            "exclusion_reason": "" if is_pilot else f"статус в реестре: {r['status']}",
            "selection_order": i,
            "term_pattern_ru": pattern(ru_sg, ru_pl, "ru") if ru_pl else "",
            "term_pattern_en": pattern(en_sg, en_pl, "en") if en_pl else "",
            "changed_components_ru": ch_ru,
            "changed_components_en": ch_en,
            "inflection_note_ru": p.get("Лингвистический комментарий к форме PL", ""),
            "inflection_note_en": "словоизменение несёт вершина термина",
            # дополнительные поля, не входящие в обязательную схему
            "pl_status": p.get("pl_status", ""),
            "registry_sha256": reg_hash,
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = REQUIRED_CONCEPT_COLUMNS + ["pl_status", "registry_sha256"]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # Журнал источников: перечисляются ВСЕ исходные документы, лежащие в папке,
    # включая ещё не обработанные. Так видно, что уже использовано, а что нет.
    used_dirs = set()
    for r in rows:
        for fld in ("source_en", "source_ru"):
            for token in re.findall(r"[A-Za-zА-Яа-я0-9_]+", r[fld]):
                used_dirs.add(token.lower())

    with OUT_SOURCES.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "folder", "sha256", "size_bytes",
                    "used_in_current_import", "n_concepts_from_registry", "access_date"])
        for path in sorted(SOURCES_DIR.rglob("*.pdf")):
            folder = path.parent.name
            used = any(part.lower() in used_dirs for part in re.split(r"[_\-]", folder))
            w.writerow([
                path.name, folder,
                hashlib.sha256(path.read_bytes()).hexdigest(),
                path.stat().st_size,
                "да" if used else "нет — источник ещё не обработан",
                len(rows) if used else 0,
                "2026-09-18",
            ])

    # --- сводка: пилотный пул отдельно от резерва ---
    pilot = [r for r in rows if r["final_status"] == "accepted"]
    reserve = [r for r in rows if r["final_status"] != "accepted"]
    strata, fam_sizes = {}, {}
    for r in pilot:
        strata[r["stratum"]] = strata.get(r["stratum"], 0) + 1
        fam_sizes[r["family_id"]] = fam_sizes.get(r["family_id"], 0) + 1
    needs = [r["concept_id"] for r in pilot if r["pl_status"] != "confirmed"]
    big_fams = {f: n for f, n in fam_sizes.items() if n > MAX_PER_FAMILY}

    notes = [
        "# Журнал импорта реестра",
        "",
        f"Источник: `{REGISTRY.relative_to(ROOT)}`, SHA-256 `{reg_hash}`.",
        f"В пилотный пул (`accepted_pilot`) вошло понятий: **{len(pilot)}**.",
        f"В резерв (другой статус, формы мн. ч. не требуются): **{len(reserve)}** "
        f"— {', '.join(r['concept_id'] for r in reserve) or 'нет'}.",
        "",
        "## Изменения строк при импорте",
        "",
        *(LOG or ["- изменений не потребовалось"]),
        "",
        "## Состояние пилотного пула",
        "",
        f"- страты: {strata} (протокол §10 предполагает равные доли A/B/C);",
        f"- лексических семей: {len(fam_sizes)}, максимум {max(fam_sizes.values()) if fam_sizes else 0} "
        f"понятия на семью при лимите {MAX_PER_FAMILY};",
        f"- формы мн. ч. со статусом, отличным от `confirmed`: {needs or 'нет'};",
        f"- семьи сверх лимита: {big_fams or 'нет'}.",
        "",
        "Семьи считаются транзитивно по общей головной лемме в любом из языков "
        "(§3 п. 5), поэтому пары с общей вершиной хотя бы в одном языке попадают "
        "в одну семью, даже если во втором языке вершины разные.",
        "",
        "## Что выведено из данных, а не взято из реестра",
        "",
        "- `changed_components_*` — сравнением строк SG и PL покомпонентно;",
        "- `term_pattern_*` — из того, какие компоненты меняются при SG→PL;",
        "- `family_id` — транзитивным объединением головных лемм.",
        "",
        "## Чего в реестре нет",
        "",
        "- Двух независимых первичных оценок (§4): зафиксирован один итоговый статус. "
        "Поле `review_language` заполнено значением `not_recorded_separately`, "
        "чтобы это не выглядело как выполненная процедура.",
    ]
    OUT_LOG.write_text("\n".join(notes) + "\n", encoding="utf-8")

    print(f"пилотный пул: {len(pilot)} понятий | резерв: {len(reserve)}")
    print(f"страты: {strata} | семей: {len(fam_sizes)} | "
          f"максимум на семью: {max(fam_sizes.values()) if fam_sizes else 0}")
    print(f"формы мн. ч. требуют решения: {needs or 'нет'}")
    print(f"записано: {OUT_CSV.relative_to(ROOT)}, {OUT_SOURCES.relative_to(ROOT)}, "
          f"{OUT_LOG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
