# -*- coding: utf-8 -*-
"""Проверка готовности данных. Запускать в любой момент, сколько угодно раз.

    .venv\\Scripts\\python.exe scripts\\check_data.py

Скрипт читает реестр, ничего в нём не меняет и пишет обычным языком,
что уже в порядке, а что мешает запускать эксперимент.
Никаких чисел про токены здесь нет — это проверка только самих данных.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from tok_pilot import config  # noqa: E402

CSV = ROOT / "data" / "real" / "concepts_pilot.csv"

OK, WARN, BAD = "  [в порядке]", "  [обратить внимание]", "  [мешает запуску]"
problems: list[str] = []
warnings_: list[str] = []


def say(level: str, text: str) -> None:
    print(f"{level} {text}")
    if level is BAD:
        problems.append(text)
    elif level is WARN:
        warnings_.append(text)


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    print("Проверка данных для исследования токенизации")
    print("=" * 46)

    section("1. Импорт реестра")
    sub_env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "import_registry.py")],
                       capture_output=True, text=True, encoding="utf-8", env=sub_env)
    if r.returncode != 0:
        say(BAD, "реестр не читается, импорт остановлен:")
        print("     " + (r.stderr or r.stdout).strip().replace("\n", "\n     "))
        summary()
        return
    for line in r.stdout.strip().splitlines():
        print("   " + line)
    say(OK, "реестр прочитан, таблица понятий обновлена")

    df = pd.read_csv(CSV, dtype=str, keep_default_na=False)
    pilot = df[df["final_status"] == "accepted"].copy()

    section("2. Сколько понятий")
    n = len(pilot)
    print(f"   в пилотном пуле: {n}")
    print(f"   в резерве:       {len(df) - n}")
    if n < 18:
        say(BAD, f"для пилота нужно не меньше 18 понятий, сейчас {n}")
    else:
        say(OK, f"понятий достаточно для пилота ({n} >= 18)")
    if n < 90:
        say(WARN, f"для основного эксперимента ориентир 90 понятий, сейчас {n} — "
                  f"нужно ещё примерно {90 - n}")

    section("3. Равномерность по структурным группам")
    counts = pilot["stratum"].value_counts().to_dict()
    for g in ("A", "B", "C"):
        print(f"   группа {g}: {counts.get(g, 0)}")
    lo, hi = min(counts.values()), max(counts.values())
    if counts.get("C", 0) == 0:
        say(BAD, "нет ни одного понятия группы C (разное число слов в языках)")
    elif hi >= 2 * max(lo, 1):
        say(WARN, "группы сильно неравные; протокол предполагает примерно поровну. "
                  "Это не ошибка, но по самой маленькой группе выводов не сделать")
    else:
        say(OK, "группы достаточно равные")

    section("4. Лексические семьи")
    fam = pilot.groupby("family_id").size()
    print(f"   семей: {fam.size}, максимум понятий в одной семье: {fam.max()}")
    over = fam[fam > config.MAX_CONCEPTS_PER_FAMILY]
    if len(over):
        say(BAD, f"лимит «не более {config.MAX_CONCEPTS_PER_FAMILY} понятий на семью» "
                 f"нарушен: {dict(over)}")
    else:
        say(OK, f"лимит не более {config.MAX_CONCEPTS_PER_FAMILY} понятий на семью соблюдён")
    if fam.size < config.FAMILY_TARGET_MAIN:
        say(WARN, f"для основного эксперимента нужно не меньше "
                  f"{config.FAMILY_TARGET_MAIN} семей, сейчас {fam.size}")

    section("5. Формы множественного числа")
    empty = pilot[(pilot["term_ru_pl"] == "") | (pilot["term_en_pl"] == "")]
    if len(empty):
        say(BAD, f"нет форм мн. ч. у: {', '.join(empty['concept_id'])}")
    same_ru = pilot[pilot["term_ru_sg"] == pilot["term_ru_pl"]]
    same_en = pilot[pilot["term_en_sg"] == pilot["term_en_pl"]]
    for label, sub in (("русских", same_ru), ("английских", same_en)):
        if len(sub):
            say(WARN, f"у {label} форм ед. и мн. ч. написаны одинаково "
                      f"({', '.join(sub['concept_id'])}) — если слово несклоняемое, "
                      f"это нормально, но лучше отметить это в комментарии")
    unconfirmed = pilot[pilot["pl_status"] != "confirmed"]
    if len(unconfirmed):
        say(WARN, f"формы мн. ч. помечены как спорные у: "
                  f"{', '.join(unconfirmed['concept_id'])} — нужно выбрать один вариант")
    if not len(empty) and not len(unconfirmed):
        say(OK, "формы множественного числа заполнены и не вызывают вопросов")

    section("6. Что протокол запрещает включать в термин")
    rules = {
        "цифры": lambda t: bool(re.search(r"\d", t)),
        "дефис": lambda t: "-" in t,
        "скобки или кавычки": lambda t: bool(re.search(r"[()«»\"']", t)),
        "заглавные буквы": lambda t: t != t.lower(),
        "латиница в русском термине": None,
        "больше 3 слов": lambda t: len(t.split(" ")) > 3,
    }
    found_any = False
    for name, test in rules.items():
        hits = []
        for _, row in pilot.iterrows():
            for fld in ("term_ru_sg", "term_ru_pl", "term_en_sg", "term_en_pl"):
                t = row[fld]
                if name == "латиница в русском термине":
                    bad = fld.startswith("term_ru") and re.search(r"[A-Za-z]", t)
                else:
                    bad = test(t)
                if bad:
                    hits.append(f"{row['concept_id']}:{fld}")
        if hits:
            found_any = True
            say(WARN if name == "заглавные буквы" else BAD,
                f"{name} — {', '.join(sorted(set(hits))[:6])}"
                + (" и др." if len(set(hits)) > 6 else ""))
    if not found_any:
        say(OK, "запрещённых элементов в терминах нет")

    section("7. Повторяющиеся термины")
    for fld, label in (("term_ru_sg", "русских"), ("term_en_sg", "английских")):
        d = pilot[pilot.duplicated(fld, keep=False)]
        if len(d):
            say(BAD, f"среди {label} терминов есть одинаковые: "
                     f"{', '.join(d['concept_id'])}")
    if not len(pilot[pilot.duplicated("term_ru_sg", keep=False)]) and \
       not len(pilot[pilot.duplicated("term_en_sg", keep=False)]):
        say(OK, "повторяющихся терминов нет")

    section("8. Происхождение каждой пары")
    for fld, label in (("source_ru", "источник русской формы"),
                       ("source_en", "источник английской формы"),
                       ("source_entry", "раздел или страница"),
                       ("definition", "обоснование эквивалентности")):
        empty_f = pilot[pilot[fld].str.strip() == ""]
        if len(empty_f):
            say(BAD, f"не заполнено поле «{label}» у: {', '.join(empty_f['concept_id'])}")
    say(OK, "поля происхождения проверены")

    section("9. Проверка двумя людьми (§4 протокола)")
    if (pilot["review_language"] == "not_recorded_separately").all():
        say(WARN, "в реестре один итоговый статус, а протокол просит две независимые "
                  "оценки: предметную и языковую, записанные порознь ДО токенизации. "
                  "Сейчас это ограничение, которое придётся указать в статье")
    else:
        say(OK, "независимые оценки записаны")

    summary()


def summary() -> None:
    print("\n" + "=" * 46)
    if problems:
        print(f"Мешает запуску: {len(problems)}")
        for p in problems:
            print(f"  - {p}")
    else:
        print("Ничего не мешает запустить эксперимент на этих данных.")
    if warnings_:
        print(f"\nСтоит обдумать, но запуску не мешает: {len(warnings_)}")
        for w in warnings_:
            print(f"  - {w}")
    print("\nКогда данные будут готовы, запустите:")
    print("  .venv\\Scripts\\python.exe scripts\\run_pilot.py --split real")


if __name__ == "__main__":
    main()
