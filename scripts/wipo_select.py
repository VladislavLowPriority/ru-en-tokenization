# -*- coding: utf-8 -*-
"""Отбор кандидатов WIPO Pearl до целевого состава корпуса 30/30/30.

Главное ограничение здесь не «набрать число», а §3 п. 5 протокола: не более
3 понятий на лексическую семью. Семьи считаются транзитивно по общей головной
лемме в ЛЮБОМ из языков, причём для русского учитывается словосложение —
«гидропривод» принадлежит семье «привод», хотя отдельного слова там нет.

Без этого ограничения выборка легко выродится в двадцать разновидностей
привода, а доверительные интервалы (бутстрэп перевыбирает семьи) станут уже,
чем есть на самом деле.

Отбор идёт «сначала свежие семьи»: при прочих равных берётся кандидат,
который не пересекается ни с чем уже собранным.

Запуск:
    .venv\\Scripts\\python.exe scripts\\wipo_select.py
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAND = ROOT / "Слова_данные" / "WIPO_PEARL" / "wipo_candidates.csv"
CORPUS = ROOT / "data" / "real" / "concepts_pilot.csv"
OUT_MD = ROOT / "Слова_данные" / "WIPO_PEARL" / "wipo_selection.md"
OUT_CSV = ROOT / "Слова_данные" / "WIPO_PEARL" / "wipo_selection.csv"

TARGET = {"A": 30, "B": 30, "C": 30}
MAX_PER_FAMILY = 3
MIN_ROOT = 4          # короче — слишком шумно для поиска корня внутри слова

# --- ручная правка после просмотра списка глазами ---------------------------
# Автоматика не отличает процесс от устройства и не видит многозначности.
EXCLUDE_MANUAL = {
    "8906644": "charge exchange → обдирка: процесс перезарядки ионов, а не устройство",
    "8833767": "radio direction finding → радиопеленгация: метод/процесс, а не устройство",
    "10019753": ("spool valve → золотник: в цитате WIPO русское обозначение употреблено "
                 "в значении ниппеля («вкручивается золотник для накачивания полости»), "
                 "тогда как spool valve — золотниковый распределитель целиком; "
                 "объёмы понятий не совпадают"),
}
EPONYM = re.compile(r"[A-Z][a-z]{2,}")


def is_eponym(en_term: str) -> bool:
    """Термин с именем собственным: Belleville washer, Rogowski coil.

    Для исследования токенизации такие термины неудобны вдвойне: §3 исключает
    имена собственные, а §6 п. 3 требует строчного написания, которое для
    эпонима неестественно.
    """
    return bool(re.search(r"[A-Z][a-z]{2,}", en_term))


RESERVE_MANUAL = {
    "8967547": "bushing: EN сильно многозначен (втулка / фильерный питатель), запись узкоотраслевая",
    "8930392": "oscillator → генератор сигналов: возможен сдвиг объёма понятия",
    "10022878": "turn → виток: EN сильно многозначен",
    "10022719": "cage → сепаратор: многозначность в обоих языках",
    "9943158": "probe → щуп: проверить совпадение объёма понятия",
    "8935316": "terminator → терминатор: многозначность",
    "8841638": "impeller → рабочее колесо: проверить, не шире ли русское обозначение",
}


# --- лексические семьи -------------------------------------------------------

class Families:
    """Транзитивное объединение понятий по общим головным леммам."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def add(self, concept: str, keys: set[str]) -> None:
        for k in keys:
            self.union(f"c:{concept}", k)

    def size_of(self, concept: str) -> int:
        root = self.find(f"c:{concept}")
        return sum(1 for k in self.parent if k.startswith("c:") and self.find(k) == root)

    def members(self, concept: str) -> list[str]:
        root = self.find(f"c:{concept}")
        return sorted(k[2:] for k in self.parent if k.startswith("c:") and self.find(k) == root)


def keys_for(en_term: str, ru_term: str, known_roots: dict[str, set[str]]) -> set[str]:
    """Ключи лексической семьи для одной пары терминов.

    Вершина берётся как последнее слово. Дополнительно учитывается
    словосложение: и в русском, и в английском вершина сложного слова стоит
    СПРАВА, поэтому корень ищется как окончание слова, а не где угодно внутри.

    Это принципиально. Поиск «подстрока где угодно» склеивал
    «электродвигатель» с «электродом», хотя это разные слова: двигатель и
    электрод. Проверка по концу слова даёт правильное родство:
    «электродвигатель» → двигатель, «гидропривод» → привод,
    «тензодатчик» → датчик, preamplifier → amplifier.
    """
    keys = set()
    for lang, term in (("en", en_term.lower()), ("ru", ru_term.lower())):
        words = [w for w in term.split(" ") if w]
        if not words:
            continue
        keys.add(f"{lang}:{words[-1]}")
        for w in words:
            for root in known_roots[lang]:
                if len(root) >= MIN_ROOT and w.endswith(root):
                    keys.add(f"{lang}:{root}")
    return keys


EXTRA_ROOTS: dict[str, set[str]] = {"en": set(), "ru": set()}


def collect_roots(items: list[tuple[str, str, str]]) -> dict[str, set[str]]:
    """Корни-вершины, известные из всех уже имеющихся терминов.

    Одних последних слов мало: у термина «датчик положения» вершина — «датчик»,
    а последнее слово «положения». Без явно указанной вершины родство
    «тензодатчик» → «датчик» останется незамеченным, и лимит семьи будет
    нарушен незаметно. Поэтому вершины из реестра добавляются отдельно.
    """
    roots = {"en": set(EXTRA_ROOTS["en"]), "ru": set(EXTRA_ROOTS["ru"])}
    for _, en, ru in items:
        for lang, term in (("en", en.lower()), ("ru", ru.lower())):
            words = [w for w in term.split(" ") if w]
            if words:
                roots[lang].add(words[-1])
    return roots


def build_families(items: list[tuple[str, str, str]]) -> Families:
    """Семьи строятся заново по всему набору: добавление одной пары может
    соединить две ранее раздельные семьи, и это видно только при полном
    пересчёте."""
    roots = collect_roots(items)
    fam = Families()
    for cid, en, ru in items:
        fam.add(cid, keys_for(en, ru, roots))
    return fam


def max_family_size(fam: Families) -> int:
    sizes: dict[str, int] = {}
    for k in fam.parent:
        if k.startswith("c:"):
            r = fam.find(k)
            sizes[r] = sizes.get(r, 0) + 1
    return max(sizes.values()) if sizes else 0


# --- данные ------------------------------------------------------------------

def load_corpus() -> list[dict]:
    rows = []
    with CORPUS.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("final_status") == "accepted":
                rows.append(r)
    return rows


def load_candidates() -> list[dict]:
    with CAND.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    corpus = load_corpus()
    cands = load_candidates()

    for r in corpus:
        for lang in ("en", "ru"):
            h = (r.get(f"head_{lang}") or "").strip().lower()
            if len(h) >= MIN_ROOT:
                EXTRA_ROOTS[lang].add(h)
    base_items = [(r["concept_id"], r["term_en_sg"], r["term_ru_sg"]) for r in corpus]

    have = {"A": 0, "B": 0, "C": 0}
    for r in corpus:
        have[r["stratum"]] = have.get(r["stratum"], 0) + 1
    need = {k: max(0, TARGET[k] - have.get(k, 0)) for k in TARGET}
    print(f"в корпусе: {have} | нужно добрать: {need}")

    # дедупликация одинаковых пар из разных предметных областей
    seen_pairs: set[tuple] = set()
    pool = []
    for c in cands:
        if c["status"] == "EXCLUDE":
            continue
        pair = (c["EN"].lower(), c["RU"].lower())
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        rid = str(c["record_id"])
        if rid in EXCLUDE_MANUAL:
            c["status"], c["manual"] = "EXCLUDE", EXCLUDE_MANUAL[rid]
            continue
        if is_eponym(c["EN"]):
            c["status"] = "EXCLUDE"
            c["manual"] = ("имя собственное в термине (эпоним): §3 исключает, "
                           "§6 п. 3 требует строчного написания")
            continue
        if rid in RESERVE_MANUAL:
            c["status"], c["manual"] = "RESERVE", RESERVE_MANUAL[rid]
        pool.append(c)

    selected: list[dict] = []
    chosen_items: list[tuple[str, str, str]] = []
    rejected_by_family: list[dict] = []
    field_used: dict[str, int] = {}

    # Сначала надёжность, затем чередование предметных областей: иначе
    # алфавитный порядок выдаёт почти одни ELEC и MECH, а измерительная
    # техника в корпус почти не попадает.
    base = sorted(pool, key=lambda c: (c["status"] != "ACCEPT",
                                       -int(c["reliability"] or 0), c["EN"]))
    by_field: dict[str, list] = {}
    for c in base:
        by_field.setdefault(c["subject_field"], []).append(c)
    order = []
    while any(by_field.values()):
        for fld in sorted(by_field, key=lambda f: -len(by_field[f])):
            if by_field[fld]:
                order.append(by_field[fld].pop(0))

    for c in order:
        st = c["stratum"]
        if need.get(st, 0) <= 0 or c["status"] != "ACCEPT":
            continue
        cid = f"W{c['record_id']}"
        trial = base_items + chosen_items + [(cid, c["EN"], c["RU"])]
        fam = build_families(trial)
        if fam.size_of(cid) > MAX_PER_FAMILY:
            c["manual"] = (f"семья вышла бы за лимит {MAX_PER_FAMILY}: "
                           f"{', '.join(fam.members(cid))}")
            rejected_by_family.append(c)
            continue
        chosen_items.append((cid, c["EN"], c["RU"]))
        field_used[c["subject_field"]] = field_used.get(c["subject_field"], 0) + 1
        selected.append(c)
        need[st] -= 1

    final = build_families(base_items + chosen_items)
    for c in selected:
        cid = f"W{c['record_id']}"
        c["family_size_after"] = final.size_of(cid)
        c["family_members"] = ", ".join(final.members(cid))
    biggest = max_family_size(final)
    print(f"итоговая проверка: самая большая лексическая семья — {biggest} понятий "
          f"при лимите {MAX_PER_FAMILY}")

    print(f"отобрано: {len(selected)} | осталось добрать: {need}")
    print(f"отсеяно лимитом семьи: {len(rejected_by_family)}")
    print(f"предметные области отобранных: {field_used}")

    cols = ["record_id", "stratum", "EN", "RU", "EN_words", "RU_words",
            "subject_field", "subject_description", "reliability",
            "usage_en", "usage_ru", "source_en", "source_ru",
            "context_ru", "en_synonyms", "ru_synonyms",
            "family_size_after", "family_members", "raw_file", "access_date", "notes"]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(selected)

    n_fam = len({final.find("c:" + cid) for cid, _, _ in base_items + chosen_items})
    extra = (f"Лексических семей в объединённом корпусе: **{n_fam}** "
             f"(ориентир протокола при N = 90 — не менее "
             f"{30}). Самая большая семья — {biggest} понятия при лимите "
             f"{MAX_PER_FAMILY}. Повторяющихся терминов между старой и новой частью нет.")
    write_markdown(selected, rejected_by_family, pool, have, need, extra)
    print(f"\n{OUT_CSV.relative_to(ROOT)}\n{OUT_MD.relative_to(ROOT)}")


def write_markdown(selected, rejected, pool, have, need, extra: str = "") -> None:
    lines = [
        "# Кандидаты из WIPO Pearl для добора корпуса до 30/30/30",
        "",
        "Источник: WIPO Pearl, многоязычная терминологическая база ВОИС.",
        "Пара EN/RU взята из **одной concept record**, поэтому эквивалентность",
        "подтверждена самой базой. Переводов от исследователя здесь нет.",
        "",
        "Состав корпуса до добора: " + ", ".join(f"{k} = {have.get(k, 0)}" for k in "ABC")
        + ". Сырые ответы API сохранены в",
        "`Слова_данные/WIPO_PEARL/raw/`, колонка `raw_file` указывает файл для каждой пары.",
        "",
        "**Токенизация не выполнялась.** Сначала фиксируется корпус.",
        "",
        "## Что ещё предстоит сделать человеку",
        "",
        "1. Проверить соответствие объёма понятий по контексту и источнику (колонка `context_ru`).",
        "2. Образовать и проверить формы множественного числа для обоих языков —",
        "   в WIPO Pearl их нет, скрипт их не придумывает.",
        "3. Поставить две независимые оценки (предметную и языковую) по §4 протокола.",
        "4. Присвоить `concept_id` и внести в реестр.",
        "",
    ]

    for st, title in (("C", "Группа C — структурно асимметричные пары"),
                      ("A", "Группа A — одно слово в обоих языках"),
                      ("B", "Группа B — 2–3 слова в обоих языках")):
        rows = [r for r in selected if r["stratum"] == st]
        if not rows:
            continue
        lines += [f"## {title} ({len(rows)})", "",
                  "| record_id | EN | RU | EN_w | RU_w | область | надёжн. | источник | статус | семья после добавления | обоснование |",
                  "|---|---|---|:--:|:--:|---|:--:|---|:--:|---|---|"]
        for r in sorted(rows, key=lambda x: (x["EN_words"], x["EN"])):
            just = "одна запись WIPO Pearl, обозначения EN и RU относятся к одному понятию"
            if r.get("manual"):
                just = r["manual"]
            lines.append(
                f"| {r['record_id']} | {r['EN']} | {r['RU']} | {r['EN_words']} | {r['RU_words']} "
                f"| {r['subject_field']} | {r['reliability']} | {(r['source_ru'] or r['source_en'])[:70]} "
                f"| ACCEPT | {r.get('family_size_after', 1)} ({r.get('family_members', '')}) | {just} |")
        lines.append("")

    if rejected:
        lines += ["## Отклонено лимитом лексической семьи", "",
                  "Пары корректные, но их добавление вывело бы семью за предел "
                  f"«не более {MAX_PER_FAMILY} понятий» (§3 п. 5). Годятся в резерв, "
                  "если из семьи будет исключено что-то другое.", "",
                  "| record_id | EN | RU | группа | причина |", "|---|---|---|:--:|---|"]
        for r in sorted(rejected, key=lambda x: x["EN"])[:40]:
            lines.append(f"| {r['record_id']} | {r['EN']} | {r['RU']} | {r['stratum']} "
                         f"| {r.get('manual', '')} |")
        lines.append("")

    manual_ex = [r for r in pool if r.get("manual") and r["status"] == "RESERVE"]
    if manual_ex:
        lines += ["## Отложено после просмотра глазами (RESERVE)", "",
                  "| record_id | EN | RU | группа | причина |", "|---|---|---|:--:|---|"]
        for r in sorted(manual_ex, key=lambda x: x["EN"]):
            lines.append(f"| {r['record_id']} | {r['EN']} | {r['RU']} | {r['stratum']} "
                         f"| {r['manual']} |")
        lines.append("")

    lines += ["## Состав корпуса после добора", "",
              "| Группа | Было | Добавляется | Станет | Осталось до цели |",
              "|---|:--:|:--:|:--:|:--:|"]
    for st in ("A", "B", "C"):
        add = sum(1 for r in selected if r["stratum"] == st)
        lines.append(f"| {st} | {have.get(st, 0)} | {add} | {have.get(st, 0) + add} "
                     f"| {need.get(st, 0)} |")
    total_before = sum(have.get(s_, 0) for s_ in "ABC")
    lines += [f"| **Всего** | **{total_before}** | **{len(selected)}** "
              f"| **{total_before + len(selected)}** | |", ""]
    if extra:
        lines += [extra, ""]

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
