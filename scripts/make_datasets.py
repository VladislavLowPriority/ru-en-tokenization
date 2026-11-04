# -*- coding: utf-8 -*-
"""Создаёт две РАЗНЫЕ таблицы понятий.

1. data/real/concepts_pilot.csv  — пустой шаблон для настоящих терминов.
   Заполняется человеком из двуязычного терминологического источника
   с указанием записи, редакции и даты доступа (§3, §4 протокола).

2. data/demo/concepts_demo.csv   — служебный набор для проверки КОДА.
   Это не исследовательский материал: пары не проходили проверку
   эквивалентности, у них нет источника, и по ним нельзя делать
   никаких выводов о русском и английском языках.
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tok_pilot.stimuli import REQUIRED_CONCEPT_COLUMNS  # noqa: E402

DEMO_NOTE = "DEMO-FIXTURE: НЕ ПРОВЕРЕНО, НЕ ИССЛЕДОВАТЕЛЬСКИЕ ДАННЫЕ"

# stratum, family, ru_sg, ru_pl, en_sg, en_pl, head_ru, head_en,
# pattern_ru, pattern_en, changed_ru, changed_en, note_ru
DEMO = [
    ("A", "fam_sensor", "датчик", "датчики", "sensor", "sensors",
     "датчик", "sensor", "N", "N", 1, 1, "окончание мн. ч. -и"),
    ("A", "fam_valve", "клапан", "клапаны", "valve", "valves",
     "клапан", "valve", "N", "N", 1, 1, "окончание мн. ч. -ы"),
    ("A", "fam_relay", "реле", "реле", "relay", "relays",
     "реле", "relay", "N", "N", 0, 1, "несклоняемое: формы SG и PL совпадают"),
    ("A", "fam_drive", "привод", "приводы", "actuator", "actuators",
     "привод", "actuator", "N", "N", 1, 1, "окончание мн. ч. -ы"),
    ("A", "fam_controller", "контроллер", "контроллеры", "controller", "controllers",
     "контроллер", "controller", "N", "N", 1, 1, "окончание мн. ч. -ы"),
    ("A", "fam_converter", "преобразователь", "преобразователи", "converter", "converters",
     "преобразователь", "converter", "N", "N", 1, 1, "мягкая основа, мн. ч. -и"),

    ("B", "fam_sensor", "датчик давления", "датчики давления",
     "pressure sensor", "pressure sensors",
     "датчик", "sensor", "N+N(gen)", "N+N", 1, 1,
     "зависимое в род. п. не меняется"),
    ("B", "fam_sensor", "датчик температуры", "датчики температуры",
     "temperature sensor", "temperature sensors",
     "датчик", "sensor", "N+N(gen)", "N+N", 1, 1,
     "зависимое в род. п. не меняется"),
    ("B", "fam_valve", "регулирующий клапан", "регулирующие клапаны",
     "control valve", "control valves",
     "клапан", "valve", "Adj+N", "N+N", 2, 1,
     "согласуемое определение меняется вместе с существительным"),
    ("B", "fam_switch", "концевой выключатель", "концевые выключатели",
     "limit switch", "limit switches",
     "выключатель", "switch", "Adj+N", "N+N", 2, 1,
     "английское мн. ч. на -es"),
    ("B", "fam_powersupply", "блок питания", "блоки питания",
     "power supply", "power supplies",
     "блок", "supply", "N+N(gen)", "N+N", 1, 1,
     "английское мн. ч. с чередованием y/ies"),
    ("B", "fam_controller", "программируемый логический контроллер",
     "программируемые логические контроллеры",
     "programmable logic controller", "programmable logic controllers",
     "контроллер", "controller", "Adj+Adj+N", "Adj+N+N", 3, 1,
     "оба согласуемых определения меняются"),

    ("C", "fam_flowmeter", "расходомер", "расходомеры", "flow meter", "flow meters",
     "расходомер", "meter", "N", "N+N", 1, 1, "RU 1 слово, EN 2 слова"),
    ("C", "fam_gauge", "манометр", "манометры", "pressure gauge", "pressure gauges",
     "манометр", "gauge", "N", "N+N", 1, 1, "RU 1 слово, EN 2 слова"),
    ("C", "fam_drive", "сервопривод", "сервоприводы", "servo drive", "servo drives",
     "сервопривод", "drive", "N", "N+N", 1, 1, "RU 1 слово, EN 2 слова"),
    ("C", "fam_fuse", "плавкий предохранитель", "плавкие предохранители", "fuse", "fuses",
     "предохранитель", "fuse", "Adj+N", "N", 2, 1, "RU 2 слова, EN 1 слово"),
    ("C", "fam_stepper", "шаговый двигатель", "шаговые двигатели", "stepper", "steppers",
     "двигатель", "stepper", "Adj+N", "N", 2, 1, "RU 2 слова, EN 1 слово"),
    ("C", "fam_gland", "кабельный ввод", "кабельные вводы", "gland", "glands",
     "ввод", "gland", "Adj+N", "N", 2, 1, "RU 2 слова, EN 1 слово"),
]


def write_real_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(REQUIRED_CONCEPT_COLUMNS)


def write_demo(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REQUIRED_CONCEPT_COLUMNS)
        w.writeheader()
        for i, row in enumerate(DEMO, start=1):
            (stratum, family, ru_sg, ru_pl, en_sg, en_pl, head_ru, head_en,
             pat_ru, pat_en, ch_ru, ch_en, note_ru) = row
            w.writerow({
                "concept_id": f"D{i:02d}",
                "family_id": family,
                "split": "demo",
                "stratum": stratum,
                "domain": "промышленная автоматизация (служебный набор)",
                "definition": DEMO_NOTE + " — определение не выписывалось",
                "term_ru_sg": ru_sg, "term_ru_pl": ru_pl,
                "term_en_sg": en_sg, "term_en_pl": en_pl,
                "head_ru": head_ru, "head_en": head_en,
                "source_ru": DEMO_NOTE, "source_en": DEMO_NOTE,
                "source_entry": "", "edition": "", "access_date": "",
                "review_technical": "not_reviewed",
                "review_language": "not_reviewed",
                "final_status": "demo_only",
                "decision_note": "используется только для проверки работы кода",
                "alternatives": "", "exclusion_reason": "",
                "selection_order": i,
                "term_pattern_ru": pat_ru, "term_pattern_en": pat_en,
                "changed_components_ru": ch_ru, "changed_components_en": ch_en,
                "inflection_note_ru": note_ru,
                "inflection_note_en": "регулярное мн. ч. существительного-вершины",
            })


if __name__ == "__main__":
    write_real_template(ROOT / "data" / "real" / "concepts_pilot.csv")
    write_demo(ROOT / "data" / "demo" / "concepts_demo.csv")
    print("Шаблон реальных данных: data/real/concepts_pilot.csv (0 строк)")
    print(f"Служебный набор:        data/demo/concepts_demo.csv ({len(DEMO)} строк)")
