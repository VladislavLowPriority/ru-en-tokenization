# -*- coding: utf-8 -*-
"""Пункт 5 плана: сопоставление контроля с технической выборкой (§14.4).

Смысл шага. Просто сравнить «термины» и «обычные слова» нельзя: термины
длиннее, чаще состоят из нескольких слов и по-другому устроены. Тогда любое
различие в токенах объяснялось бы длиной, а не тем, что нас интересует.
Поэтому к каждому техническому понятию подбирается бытовое понятие с тем же
числом слов в каждом языке, той же структурной группой и близкой длиной в
знаках.

Подбор идёт **до токенизации**. Ни одна длина в токенах здесь не читается.

Правила (объявлены в §14.4 протокола):

* совпадает структурная группа A/B/C и число слов отдельно в RU и EN;
* длина в знаках без пробелов различается не более чем на 4 в каждом языке;
* из подходящих берётся ближайший по сумме отклонений, при равенстве — первый
  по порядку источника;
* не более 3 понятий на лексическую семью контроля;
* одно понятие используется один раз.

Результат: `data/control/concepts_control.csv` в той же схеме, что и
технический корпус, и таблица качества сопоставления, которая публикуется
до измерения.

Запуск:
    .venv\\Scripts\\python.exe scripts\\control_select.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CTRL = ROOT / "Слова_данные" / "CONTROL"
S1 = CTRL / "control_candidates_s1.csv"
S2 = CTRL / "control_candidates_s2.csv"
TECH = ROOT / "data" / "real" / "concepts_pilot.csv"
OUT_DIR = ROOT / "data" / "control"
OUT_CONCEPTS = OUT_DIR / "concepts_control.csv"
OUT_MATCH = OUT_DIR / "matching_pairs.csv"
OUT_QUALITY = OUT_DIR / "matching_quality.csv"
OUT_LOG = OUT_DIR / "selection_log.json"
EXCLUSIONS = CTRL / "manual_exclusions.csv"

CALIPER = 4          # допуск по длине в знаках, объявлен до отбора
MAX_PER_FAMILY = 3
ACCESS_DATE = "2026-09-19"


def chars(term: str) -> int:
    return len(term.replace(" ", ""))


def stratum_of(w_ru: int, w_en: int) -> str:
    if w_ru == 1 and w_en == 1:
        return "A"
    if w_ru > 1 and w_en > 1:
        return "B"
    return "C"


def load_candidates() -> list[dict]:
    rows = []
    for path in (S1, S2):
        if not path.exists():
            continue
        for r in csv.DictReader(path.open(encoding="utf-8")):
            if r["reject_reason"]:
                continue
            w_ru, w_en = int(r["RU_words"]), int(r["EN_words"])
            if w_ru > 3 or w_en > 3:
                continue
            rows.append({
                "cand_id": r["cand_id"], "source": r["source"],
                "source_ref": r["source_ref"],
                "concepticon_id": r["concepticon_id"],
                "concepticon_gloss": r["concepticon_gloss"],
                "EN": r["EN"], "RU": r["RU"],
                "EN_words": w_en, "RU_words": w_ru,
                "EN_chars": chars(r["EN"]), "RU_chars": chars(r["RU"]),
                "stratum": stratum_of(w_ru, w_en),
                "semantic_category": r["semantic_category"],
                "ru_variants": r["ru_variants"],
                "access_date": r["access_date"],
            })
    # Снятие повторов. Повтором считается не только полностью совпавшая пара,
    # но и **одно и то же обозначение в любом из языков**: «журнальный столик»
    # пришёл дважды, как occasional table и как coffee table, — это два
    # обозначения одного понятия, а одно понятие входит в выборку один раз.
    seen_pair, seen_ru, seen_en, uniq = set(), set(), set(), []
    for r in sorted(rows, key=lambda x: x["cand_id"]):
        if (r["EN"], r["RU"]) in seen_pair or r["RU"] in seen_ru or r["EN"] in seen_en:
            continue
        seen_pair.add((r["EN"], r["RU"]))
        seen_ru.add(r["RU"])
        seen_en.add(r["EN"])
        uniq.append(r)

    # Проверенный вручную список нарушений §14.3: часть речи, повтор понятия и
    # предметный тип (орудие, производственное или медицинское помещение,
    # научное понятие). Список составлен просмотром отобранных пар и применён
    # к пулу кандидатов; решения принимались по языковым и предметным
    # основаниям, без обращения к числу токенов.
    excluded = {}
    if EXCLUSIONS.exists():
        for e in csv.DictReader(EXCLUSIONS.open(encoding="utf-8")):
            excluded[(e["term_en"], e["term_ru"])] = e
    kept = []
    for r in uniq:
        hit = excluded.get((r["EN"], r["RU"]))
        if hit:
            r["reject_reason"] = "manual::" + hit["rule"]
            continue
        kept.append(r)
    print(f"  снято повторов: {len(rows) - len(uniq)}; "
          f"исключено по проверке §14.3: {len(uniq) - len(kept)}")
    return kept


def load_tech() -> list[dict]:
    rows = []
    for r in csv.DictReader(TECH.open(encoding="utf-8")):
        if r["final_status"] != "accepted":
            continue
        rows.append({
            "concept_id": r["concept_id"], "family_id": r["family_id"],
            "stratum": r["stratum"],
            "RU": r["term_ru_sg"], "EN": r["term_en_sg"],
            "RU_words": len(r["term_ru_sg"].split(" ")),
            "EN_words": len(r["term_en_sg"].split(" ")),
            "RU_chars": chars(r["term_ru_sg"]), "EN_chars": chars(r["term_en_sg"]),
        })
    return rows


def family_key(r: dict) -> tuple[str, str]:
    """Семья контроля: по последнему слову в каждом языке.

    Для бытовых выражений это почти всегда вершина («стиральная машина» —
    машина, `washing machine` — machine). Правило огрубляет русские
    конструкции с родительным падежом, но для контроля оно используется только
    чтобы одно слово не заполнило выборку, а не для лингвистических выводов.
    """
    return (r["RU"].split(" ")[-1], r["EN"].split(" ")[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--caliper", type=int, default=CALIPER,
                    help="допуск по длине в знаках; основной анализ — 4 (§14.4)")
    ap.add_argument("--tag", default="",
                    help="суффикс имён файлов для дополнительной проверки")
    args = ap.parse_args()
    globals()["CALIPER"] = args.caliper
    if args.tag:
        for name in ("OUT_CONCEPTS", "OUT_MATCH", "OUT_QUALITY", "OUT_LOG"):
            path = globals()[name]
            globals()[name] = path.with_name(f"{path.stem}_{args.tag}{path.suffix}")

    cands = load_candidates()
    tech = load_tech()
    print(f"кандидатов контроля: {len(cands)}; технических понятий: {len(tech)}")

    pool: dict[tuple, list[dict]] = {}
    for c in cands:
        pool.setdefault((c["stratum"], c["RU_words"], c["EN_words"]), []).append(c)
    for v in pool.values():
        v.sort(key=lambda r: r["cand_id"])

    # Порядок подбора: сначала самые дефицитные ячейки, иначе редкие случаи
    # останутся без пары, а частые заберут всех кандидатов.
    need: dict[tuple, int] = {}
    for t in tech:
        need[(t["stratum"], t["RU_words"], t["EN_words"])] = \
            need.get((t["stratum"], t["RU_words"], t["EN_words"]), 0) + 1
    supply = {k: len(v) for k, v in pool.items()}
    tech_sorted = sorted(
        tech, key=lambda t: (supply.get((t["stratum"], t["RU_words"], t["EN_words"]), 0)
                             / max(need[(t["stratum"], t["RU_words"], t["EN_words"])], 1),
                             t["concept_id"]))

    used: set[str] = set()
    fam_count: dict[tuple, int] = {}
    pairs, unmatched = [], []
    for t in tech_sorted:
        cell = (t["stratum"], t["RU_words"], t["EN_words"])
        best, best_score = None, None
        for c in pool.get(cell, []):
            if c["cand_id"] in used:
                continue
            d_ru = abs(c["RU_chars"] - t["RU_chars"])
            d_en = abs(c["EN_chars"] - t["EN_chars"])
            if d_ru > CALIPER or d_en > CALIPER:
                continue
            if fam_count.get(family_key(c), 0) >= MAX_PER_FAMILY:
                continue
            score = d_ru + d_en
            if best_score is None or score < best_score:
                best, best_score = c, score
        if best is None:
            # Причина дефицита фиксируется точно: пустая ячейка, допуск или
            # лимит семьи. Без этого «не нашли» читается как «не существует».
            cell_cands = pool.get(cell, [])
            free = [c for c in cell_cands if c["cand_id"] not in used]
            not_capped = [c for c in free
                          if fam_count.get(family_key(c), 0) < MAX_PER_FAMILY]
            if not cell_cands:
                reason, nearest = "в ячейке нет ни одного кандидата", None
            elif not free:
                reason, nearest = "все кандидаты ячейки уже заняты другими парами", None
            elif not not_capped:
                reason, nearest = "оставшиеся кандидаты заблокированы лимитом семьи", None
            else:
                nearest = min(max(abs(c["RU_chars"] - t["RU_chars"]),
                                  abs(c["EN_chars"] - t["EN_chars"]))
                              for c in not_capped)
                reason = (f"ближайший кандидат отличается по длине на {nearest} знаков "
                          f"при допуске {CALIPER}")
            unmatched.append({**t, "reason": reason, "nearest_gap_chars": nearest,
                              "cell_candidates": len(cell_cands),
                              "cell_candidates_free": len(free)})
            continue
        used.add(best["cand_id"])
        fam_count[family_key(best)] = fam_count.get(family_key(best), 0) + 1
        pairs.append({
            "tech_concept_id": t["concept_id"], "tech_family_id": t["family_id"],
            "stratum": t["stratum"], "RU_words": t["RU_words"], "EN_words": t["EN_words"],
            "tech_RU": t["RU"], "tech_EN": t["EN"],
            "ctrl_cand_id": best["cand_id"], "ctrl_RU": best["RU"], "ctrl_EN": best["EN"],
            "tech_RU_chars": t["RU_chars"], "ctrl_RU_chars": best["RU_chars"],
            "tech_EN_chars": t["EN_chars"], "ctrl_EN_chars": best["EN_chars"],
            "d_RU_chars": best["RU_chars"] - t["RU_chars"],
            "d_EN_chars": best["EN_chars"] - t["EN_chars"],
            "ctrl_source": best["source"], "ctrl_source_ref": best["source_ref"],
            "ctrl_category": best["semantic_category"],
            "ctrl_variants": best["ru_variants"],
        })

    pairs.sort(key=lambda p: p["tech_concept_id"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_MATCH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
        w.writeheader()
        w.writerows(pairs)

    # --- таблица понятий контроля в схеме конвейера --------------------------
    by_cand = {c["cand_id"]: c for c in cands}
    fam_id: dict[tuple, str] = {}
    rows = []
    for i, p in enumerate(pairs, start=1):
        c = by_cand[p["ctrl_cand_id"]]
        key = family_key(c)
        fam_id.setdefault(key, f"cfam_{key[0]}")
        rows.append({
            "concept_id": f"C{i:03d}", "family_id": fam_id[key],
            "split": "control", "stratum": c["stratum"],
            "domain": "общеязыковая лексика: " + (c["semantic_category"] or "—"),
            "definition": c["concepticon_gloss"] or "",
            "term_ru_sg": c["RU"], "term_ru_pl": c["RU"],
            "term_en_sg": c["EN"], "term_en_pl": c["EN"],
            "head_ru": c["RU"].split(" ")[-1], "head_en": c["EN"].split(" ")[-1],
            "source_ru": c["source"], "source_en": c["source"],
            "source_entry": c["source_ref"], "edition": "",
            "access_date": c["access_date"],
            "review_technical": "not_applicable",
            "review_language": "not_recorded_separately",
            "final_status": "accepted",
            "decision_note": f"пара к {p['tech_concept_id']}; "
                             f"отклонение длин RU {p['d_RU_chars']:+d}, EN {p['d_EN_chars']:+d}",
            "alternatives": c["ru_variants"],
            "exclusion_reason": "",
            "selection_order": str(i),
            "term_pattern_ru": "", "term_pattern_en": "",
            "changed_components_ru": "0", "changed_components_en": "0",
            "inflection_note_ru": "формы множественного числа не образовывались (§14.6)",
            "inflection_note_en": "формы множественного числа не образовывались (§14.6)",
            "pl_status": "not_measured",
            "matched_tech_concept_id": p["tech_concept_id"],
            "matched_tech_family_id": p["tech_family_id"],
            "concepticon_id": c["concepticon_id"],
        })
    with OUT_CONCEPTS.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # --- качество сопоставления ---------------------------------------------
    cells: dict[tuple, list[dict]] = {}
    for p in pairs:
        cells.setdefault((p["stratum"], p["RU_words"], p["EN_words"]), []).append(p)
    quality = []
    for cell in sorted(need, key=lambda k: (k[0], k[1], k[2])):
        got = cells.get(cell, [])
        def mean(xs):
            return round(sum(xs) / len(xs), 2) if xs else None
        quality.append({
            "stratum": cell[0], "RU_words": cell[1], "EN_words": cell[2],
            "tech_needed": need[cell], "candidates_available": supply.get(cell, 0),
            "matched": len(got), "deficit": need[cell] - len(got),
            "tech_RU_chars_mean": mean([g["tech_RU_chars"] for g in got]),
            "ctrl_RU_chars_mean": mean([g["ctrl_RU_chars"] for g in got]),
            "tech_EN_chars_mean": mean([g["tech_EN_chars"] for g in got]),
            "ctrl_EN_chars_mean": mean([g["ctrl_EN_chars"] for g in got]),
            "mean_abs_d_RU": mean([abs(g["d_RU_chars"]) for g in got]),
            "mean_abs_d_EN": mean([abs(g["d_EN_chars"]) for g in got]),
        })
    with OUT_QUALITY.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(quality[0].keys()))
        w.writeheader()
        w.writerows(quality)

    log = {
        "access_date": ACCESS_DATE,
        "caliper_chars": CALIPER, "max_per_family": MAX_PER_FAMILY,
        "n_candidates": len(cands), "n_tech": len(tech),
        "n_matched": len(pairs), "n_unmatched": len(unmatched),
        "unmatched": unmatched,
        "control_families": len(fam_id),
        "max_family_size": max(fam_count.values()) if fam_count else 0,
        "sha256_candidates_s1": hashlib.sha256(S1.read_bytes()).hexdigest() if S1.exists() else "",
        "sha256_candidates_s2": hashlib.sha256(S2.read_bytes()).hexdigest() if S2.exists() else "",
        "note": "отбор выполнен без обращения к результатам токенизации",
    }
    OUT_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"сопоставлено пар: {len(pairs)} из {len(tech)}; без пары: {len(unmatched)}")
    print(f"семей контроля: {len(fam_id)}, максимум на семью: {log['max_family_size']}")
    for q in quality:
        print(f"  {q['stratum']} {q['RU_words']}-{q['EN_words']}: нужно {q['tech_needed']}, "
              f"кандидатов {q['candidates_available']}, подобрано {q['matched']}")
    print(f"\n{OUT_CONCEPTS.relative_to(ROOT)}\n{OUT_MATCH.relative_to(ROOT)}\n"
          f"{OUT_QUALITY.relative_to(ROOT)}\n{OUT_LOG.relative_to(ROOT)}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
