# -*- coding: utf-8 -*-
"""Формы множественного числа для пар, отобранных из WIPO Pearl.

Что здесь происходит и чем это отличается от подбора самих терминов.

* **Эквивалентность EN↔RU** — утверждение о том, что два обозначения называют
  одно понятие. Его нельзя вывести из правил, оно берётся из источника: в
  нашем случае из одной concept record WIPO Pearl.
* **Формы числа** — словоизменение уже проверенного термина. Это регулярная
  морфология, и она образуется по правилу, а не извлекается из текста. Так же
  сделаны формы в существующем реестре.

Поэтому формы ниже записаны заранее, по трём образцам, а не генерируются
моделью на ходу:

  1. одиночное существительное: «золотник → золотники», slot → slots;
  2. «прилагательное + существительное»: меняются оба слова —
     «коленчатый вал → коленчатые валы», mounting bracket → mounting brackets;
  3. «существительное + зависимое в родительном падеже»: меняется только
     вершина — «источник света → источники света», не «светов».

Статус `needs_review` ставится там, где возможны два нормативных прочтения.
Для таких случаев есть отдельная проверка: `scripts/pl_sensitivity.py`
пересчитывает результат при каждом варианте и показывает фактическую разницу.

Итоговую языковую проверку по §4 протокола делает научный руководитель:
поле `verified_by` до этого момента остаётся незаполненным.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEL = ROOT / "Слова_данные" / "WIPO_PEARL" / "wipo_selection.csv"
OUT_CSV = ROOT / "Слова_данные" / "WIPO_PEARL" / "wipo_plural_forms.csv"
OUT_MD = ROOT / "Слова_данные" / "WIPO_PEARL" / "wipo_plural_forms.md"

# record_id -> (EN_pl, RU_pl, образец, статус, комментарий)
FORMS: dict[str, tuple[str, str, str, str, str]] = {
    # --- группа C -----------------------------------------------------------
    "8922805": ("baluns", "симметрирующие устройства", "Adj+N", "confirmed",
                "согласуемое определение меняется вместе с существительным"),
    "8822734": ("crankshafts", "коленчатые валы", "Adj+N", "confirmed",
                "согласование прилагательного; «валы» — литературная норма, «вала» отклонено"),
    "8826603": ("lamps", "источники света", "N+N(gen)", "confirmed",
                "зависимое «света» — вещественное существительное, мн. ч. не образует"),
    "10029194": ("preamplifiers", "предварительные усилители", "Adj+N", "confirmed",
                 "согласование; мягкая основа, мн. ч. на -и"),
    "10025490": ("atomizer nozzles", "форсунки", "N", "confirmed", "регулярное -и"),
    "8847816": ("counter electrodes", "противоэлектроды", "N", "confirmed", "регулярное -ы"),
    "8875687": ("current limiters", "токоограничители", "N", "confirmed",
                "мягкая основа, мн. ч. на -и"),
    "8840804": ("electric motors", "электродвигатели", "N", "confirmed",
                "мягкая основа, мн. ч. на -и"),
    "8862185": ("hydraulic pumps", "гидронасосы", "N", "confirmed", "регулярное -ы"),
    "8945106": ("memory resistors", "мемристоры", "N", "confirmed", "регулярное -ы"),
    "8889856": ("thermal fuses", "термопредохранители", "N", "confirmed",
                "мягкая основа, мн. ч. на -и"),
    "8890605": ("secondary batteries", "аккумуляторы", "N", "confirmed",
                "регулярное -ы; в английском чередование y/ies после согласной"),
    "10019350": ("planet gears", "сателлиты", "N", "confirmed", "регулярное -ы"),
    "10021064": ("reduction gears", "редукторы", "N", "confirmed",
                 "регулярное -ы; профессиональное «редуктора» отклонено"),
    "8942427": ("servo valves", "сервоклапаны", "N", "confirmed", "регулярное -ы"),
    "9004439": ("reed switches", "герконы", "N", "confirmed",
                "регулярное -ы; в английском орфографическое окончание -es после -ch"),
    "8855691": ("strain gauges", "тензодатчики", "N", "confirmed", "регулярное -и"),
    # --- группа A -----------------------------------------------------------
    "8826790": ("ammeters", "амперметры", "N", "confirmed", "регулярное -ы"),
    "8949044": ("amplifiers", "усилители", "N", "confirmed", "мягкая основа, мн. ч. на -и"),
    "8830484": ("autotransformers", "автотрансформаторы", "N", "confirmed",
                "регулярное -ы; профессиональное «трансформатора» отклонено"),
    "10020979": ("bearings", "подшипники", "N", "confirmed", "регулярное -и"),
    "9956741": ("cams", "кулачки", "N", "confirmed",
                "беглая гласная в основе: кулачок → кулачки"),
    "8819776": ("coils", "катушки", "N", "confirmed", "регулярное -и"),
    "8850062": ("compressors", "компрессоры", "N", "confirmed", "регулярное -ы"),
    "9015774": ("conveyors", "конвейеры", "N", "confirmed", "регулярное -ы"),
    "8942363": ("couplers", "ответвители", "N", "confirmed", "мягкая основа, мн. ч. на -и"),
    "8828706": ("durometers", "твердомеры", "N", "confirmed", "регулярное -ы"),
    "8986927": ("gyroscopes", "гироскопы", "N", "confirmed", "регулярное -ы"),
    "10027603": ("servomotors", "сервоприводы", "N", "confirmed",
                 "литературная норма «приводы»; профессиональное «привода» отклонено, "
                 "как и для P01 «линейные приводы»"),
    # --- группа B -----------------------------------------------------------
    "8946248": ("absolute encoders", "абсолютные энкодеры", "Adj+N", "confirmed",
                "согласование прилагательного"),
    "9942950": ("accumulating conveyors", "накопительные конвейеры", "Adj+N", "confirmed",
                "согласование прилагательного"),
    "9010317": ("aerodynamic bearings", "аэродинамические подшипники", "Adj+N", "confirmed",
                "согласование прилагательного"),
}


def changed_components(sg: str, pl: str) -> int:
    a, b = sg.split(" "), pl.split(" ")
    if len(a) != len(b):
        return -1
    return sum(1 for x, y in zip(a, b) if x != y)


def main() -> None:
    with SEL.open(encoding="utf-8") as f:
        sel = list(csv.DictReader(f))

    missing = [r["record_id"] for r in sel if r["record_id"] not in FORMS]
    if missing:
        raise SystemExit("Нет форм мн. ч. для record_id: " + ", ".join(missing))

    rows = []
    for r in sel:
        en_pl, ru_pl, pattern, status, note = FORMS[r["record_id"]]
        rows.append({
            "record_id": r["record_id"],
            "stratum": r["stratum"],
            "term_en_sg": r["EN"], "term_en_pl": en_pl,
            "term_ru_sg": r["RU"], "term_ru_pl": ru_pl,
            "pattern_ru": pattern,
            "changed_components_ru": changed_components(r["RU"], ru_pl),
            "changed_components_en": changed_components(r["EN"], en_pl),
            "pl_status": status,
            "inflection_note_ru": note,
            "inflection_note_en": "словоизменение несёт вершина термина",
            "verified_by": "",          # заполняет научный руководитель (§4)
            "source_record": "WIPO Pearl concept record " + r["record_id"],
        })

    cols = list(rows[0].keys())
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    by_pattern: dict[str, int] = {}
    for r in rows:
        by_pattern[r["pattern_ru"]] = by_pattern.get(r["pattern_ru"], 0) + 1

    lines = [
        "# Формы множественного числа для 30 пар из WIPO Pearl",
        "",
        "Формы образованы по правилу и записаны заранее, а не подобраны моделью",
        "на ходу. Три образца, по которым идёт словоизменение:",
        "",
        "| Образец | Что меняется | Пример |",
        "|---|---|---|",
        "| `N` | окончание существительного | золотник → золотники |",
        "| `Adj+N` | окончания обоих слов | коленчатый вал → коленчатые валы |",
        "| `N+N(gen)` | только вершина, зависимое остаётся | источник света → источники света |",
        "",
        "Распределение: " + ", ".join(f"{k} — {v}" for k, v in sorted(by_pattern.items())) + ".",
        "",
        "**Эквивалентность EN↔RU взята из записи WIPO Pearl и здесь не трогается.**",
        "Изменяется только грамматическая форма уже проверенной пары: лексема и",
        "тип понятия сохраняются, словообразование не используется.",
        "",
        "Колонка `verified_by` пуста: языковую проверку по §4 протокола выполняет",
        "научный руководитель. До этого формы считаются предложенными, а не принятыми.",
        "",
        "| record_id | группа | EN ед. → мн. | RU ед. → мн. | образец | изменилось слов (RU) | статус | комментарий |",
        "|---|:--:|---|---|---|:--:|---|---|",
    ]
    for r in sorted(rows, key=lambda x: (x["stratum"], x["term_en_sg"])):
        lines.append(
            f"| {r['record_id']} | {r['stratum']} "
            f"| {r['term_en_sg']} → {r['term_en_pl']} "
            f"| {r['term_ru_sg']} → {r['term_ru_pl']} "
            f"| {r['pattern_ru']} | {r['changed_components_ru']} "
            f"| {r['pl_status']} | {r['inflection_note_ru']} |")

    lines += [
        "",
        "## Спорные случаи",
        "",
        "Среди этих 30 пар конструкций с двояким прочтением нет: единственная",
        "пара образца `N+N(gen)` — «источник света», где зависимое слово",
        "вещественное и множественного числа не образует.",
        "",
        "Для уже имеющихся в корпусе спорных пар (P07, P16, P29) выбор варианта",
        "проверен измерением: см. `results/real/pl_sensitivity.json`.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"формы мн. ч.: {len(rows)} пар | по образцам: {by_pattern}")
    print(f"{OUT_CSV.relative_to(ROOT)}")
    print(f"{OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
