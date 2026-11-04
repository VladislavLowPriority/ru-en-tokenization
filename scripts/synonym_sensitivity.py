# -*- coding: utf-8 -*-
"""Пункт 7 плана: выбор обозначения как самостоятельная проверка (§14.7).

Вопрос простой. У части понятий источник даёт несколько засвидетельствованных
обозначений одного и того же предмета: «сервопривод» и «сервомотор»,
`memory resistor` и `memristor`. Мы выбрали по одному по объявленному правилу.
Насколько результат зависит от этого выбора?

Что здесь считается:

1. **Именованные сценарии** — заменить только английское обозначение, только
   русское, оба сразу (берётся первый засвидетельствованный вариант).
2. **Полный перебор** всех сочетаний вариантов: 147 456 комбинаций. Для каждой
   считается G. Отсюда точные границы и распределение разброса.
3. **Структурная группа** A/B/C: у части понятий вариант меняет число слов,
   значит и группу. Показано, сколько понятий переходит и как меняется состав.
4. **Лексические семьи**: если вариант меняет вершину, семьи пересобираются по
   тому же правилу (совпадение по концу слова), и проверяется предел «не более
   трёх понятий на семью».

Важное ограничение, которое видно уже из данных: альтернативы есть только у 12
из 90 понятий, и все 12 — из WIPO Pearl, где варианты записаны в самой базе.
Для 60 понятий из руководств альтернативы не выписывались, поэтому разброс
ниже — это разброс по той части корпуса, где варианты известны, а не по всему
корпусу.

Разброс из-за выбора обозначения публикуется ОТДЕЛЬНО от доверительного
интервала: это разные источники неопределённости, складывать их нельзя.

Запуск:
    .venv\\Scripts\\python.exe scripts\\synonym_sensitivity.py
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tok_pilot import backends, config, measure  # noqa: E402

CONCEPTS = ROOT / "data" / "real" / "concepts_pilot.csv"
SELECTION = ROOT / "Слова_данные" / "WIPO_PEARL" / "wipo_selection.csv"
OUT_JSON = ROOT / "results" / "real" / "synonym_sensitivity.json"
OUT_CSV = ROOT / "results" / "real" / "synonym_variants.csv"


def load_concepts() -> pd.DataFrame:
    df = pd.read_csv(CONCEPTS, dtype=str, keep_default_na=False)
    return df[df["final_status"] == "accepted"].reset_index(drop=True)


def load_variants(concepts: pd.DataFrame) -> list[dict]:
    """Варианты обозначений из записей WIPO Pearl, привязанные к concept_id.

    Привязка проверяется по совпадению самих обозначений: если EN и RU из
    таблицы отбора не находятся в корпусе, это ошибка, а не повод угадывать.
    """
    sel = list(csv.DictReader(SELECTION.open(encoding="utf-8")))
    by_terms = {(r["term_en_sg"], r["term_ru_sg"]): r["concept_id"]
                for _, r in concepts.iterrows()}
    out = []
    for r in sel:
        key = (r["EN"], r["RU"])
        if key not in by_terms:
            raise SystemExit(f"Пара из отбора не найдена в корпусе: {key}")
        en_alt = [x.strip() for x in r["en_synonyms"].split("|") if x.strip()]
        ru_alt = [x.strip() for x in r["ru_synonyms"].split("|") if x.strip()]
        if not en_alt and not ru_alt:
            continue
        out.append({
            "concept_id": by_terms[key],
            "record_id": r["record_id"],
            "en_chosen": r["EN"], "ru_chosen": r["RU"],
            "en_options": [r["EN"]] + en_alt,
            "ru_options": [r["RU"]] + ru_alt,
        })
    return out


def measure_strings(pairs: list[tuple[str, str, str]], bks: dict) -> pd.DataFrame:
    """pairs: (метка, язык, текст). Токены считаются тем же кодом, что и в пилоте."""
    rows = []
    for label, lang, text in pairs:
        for mode, inp in (("isolated_0", text), ("isolated_space", " " + text)):
            start = inp.index(text)
            rows.append({
                "stimulus_id": f"{label}|{lang}|sg|{mode}",
                "concept_id": label, "family_id": "-", "stratum": "-",
                "lang": lang, "number": "sg", "mode": mode, "template_id": "",
                "input_text": inp, "target_start": start,
                "target_end": start + len(text),
            })
    st = pd.DataFrame(rows)
    m = measure.measure(st, bks, run_id="synonyms")
    if (m["error_code"].fillna("") != "").any():
        raise SystemExit("Ошибка токенизации вариантов")
    if not m["decode_match"].all():
        raise SystemExit("Декодирование не восстановило исходную строку")
    return m


def _importer_families():
    """Берём ровно ту функцию семей, по которой собирался корпус.

    Правило шире буквального чтения §3: ключом становится не только головная
    лемма, но и последнее слово каждого термина. В русской конструкции
    «существительное + зависимое в родительном падеже» последнее слово —
    зависимое, поэтому «источник питания» и «розетка электропитания»
    оказываются одной семьёй. Правило объединяет больше пар, чем требуется,
    то есть интервалы получаются шире, а не уже. Здесь оно используется ради
    сопоставимости с отбором и основным расчётом; влияние самого правила
    проверяется отдельно в `scripts/hypotheses_extra.py`.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "import_registry", ROOT / "scripts" / "import_registry.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_families


def family_sizes(pairs: list[tuple[str, str, str]]) -> dict[str, int]:
    """pairs: (concept_id, EN, RU) -> размеры семей."""
    build = _importer_families()
    heads = {cid: (en.split(" ")[-1], ru.split(" ")[-1]) for cid, en, ru in pairs}
    terms = {cid: (en, ru) for cid, en, ru in pairs}
    fam_of = build(heads, terms)
    sizes: dict[str, int] = {}
    for cid in terms:
        f = fam_of[cid]
        sizes[f] = sizes.get(f, 0) + 1
    return sizes


def family_ok_combinations(concepts, by_cid, per_concept):
    """Какие сочетания вершин сохраняют ограничение «не более 3 на семью».

    Вершины меняет лишь часть вариантов, поэтому перебирать надо только их:
    для каждого сочетания вершин семьи пересобираются целиком, и сочетание
    считается допустимым, если ни одна семья не превысила трёх понятий.
    """
    base_pairs = [(c["concept_id"], c["term_en_sg"], c["term_ru_sg"])
                  for _, c in concepts.iterrows()]
    cache: dict[tuple, bool] = {}

    def check(sig: tuple) -> bool:
        if sig in cache:
            return cache[sig]
        repl = {cid: (en_head, ru_head) for cid, ru_head, en_head in
                [(c, r, e) for c, r, e in sig]}
        pairs = []
        for cid, en, ru in base_pairs:
            if cid in by_cid and cid in repl:
                opt = next(o for o in next(pc for pc in per_concept
                                           if pc["concept_id"] == cid)["options"]
                           if o["ru"].split(" ")[-1] == repl[cid][1]
                           and o["en"].split(" ")[-1] == repl[cid][0])
                en, ru = opt["en"], opt["ru"]
            pairs.append((cid, en, ru))
        sizes = family_sizes(pairs)
        ok = max(sizes.values()) <= 3
        cache[sig] = ok
        return ok

    return check


def main() -> None:
    concepts = load_concepts()
    variants = load_variants(concepts)
    print(f"понятий в корпусе: {len(concepts)}; с альтернативами: {len(variants)}")

    # Все строки, которые нужно измерить: выбранные обозначения всех понятий
    # плюс каждый вариант у понятий с альтернативами.
    to_measure: list[tuple[str, str, str]] = []
    for _, c in concepts.iterrows():
        to_measure.append((c["concept_id"] + "::base", "ru", c["term_ru_sg"]))
        to_measure.append((c["concept_id"] + "::base", "en", c["term_en_sg"]))
    for v in variants:
        for i, t in enumerate(v["ru_options"]):
            to_measure.append((f"{v['concept_id']}::ru{i}", "ru", t))
        for i, t in enumerate(v["en_options"]):
            to_measure.append((f"{v['concept_id']}::en{i}", "en", t))

    print("измерение вариантов...")
    bks = backends.load_all()
    m = measure_strings(to_measure, bks)

    # T для каждой (метка, язык, режим, токенизатор)
    tok_counts: dict[tuple, int] = {}
    for _, r in m.iterrows():
        tok_counts[(r["concept_id"], r["lang"], r["mode"], r["tokenizer_id"])] = int(r["n_tokens_input"])

    var_rows = []
    for v in variants:
        for lang, key in (("ru", "ru_options"), ("en", "en_options")):
            for i, text in enumerate(v[key]):
                for tid in config.TOKENIZERS:
                    var_rows.append({
                        "concept_id": v["concept_id"], "record_id": v["record_id"],
                        "lang": lang, "variant_index": i,
                        "designation": text, "is_chosen": i == 0,
                        "words": len(text.split(" ")),
                        "has_hyphen": "-" in text,
                        "tokenizer_id": tid,
                        "T_iso0": tok_counts[(f"{v['concept_id']}::{lang}{i}", lang, "isolated_0", tid)],
                        "T_isosp": tok_counts[(f"{v['concept_id']}::{lang}{i}", lang, "isolated_space", tid)],
                    })
    pd.DataFrame(var_rows).to_csv(OUT_CSV, index=False, encoding="utf-8")

    by_cid = {v["concept_id"]: v for v in variants}
    result = {
        "n_concepts": int(len(concepts)),
        "n_concepts_with_variants": len(variants),
        "n_variants_total": sum(len(v["en_options"]) - 1 + len(v["ru_options"]) - 1
                                for v in variants),
        "source_of_variants": "WIPO Pearl concept records (источники 1–2 вариантов не давали)",
        "note": "разброс по выбору обозначения не складывается с доверительным интервалом",
        "by_tokenizer": {},
    }

    for tid in config.TOKENIZERS:
        for mode, suffix in (("isolated_0", "isolated_0"), ("isolated_space", "isolated_space")):
            base_logs = {}
            for _, c in concepts.iterrows():
                cid = c["concept_id"]
                t_ru = tok_counts[(cid + "::base", "ru", mode, tid)]
                t_en = tok_counts[(cid + "::base", "en", mode, tid)]
                base_logs[cid] = np.log(t_ru / t_en)

            fixed = [lg for cid, lg in base_logs.items() if cid not in by_cid]
            fixed_sum = float(np.sum(fixed))
            n = len(base_logs)

            # варианты каждого понятия: (log-отношение, слов RU, слов EN, RU, EN)
            per_concept = []
            for v in variants:
                opts = []
                for i_ru, ru in enumerate(v["ru_options"]):
                    t_ru = tok_counts[(f"{v['concept_id']}::ru{i_ru}", "ru", mode, tid)]
                    for i_en, en in enumerate(v["en_options"]):
                        t_en = tok_counts[(f"{v['concept_id']}::en{i_en}", "en", mode, tid)]
                        opts.append({"log": float(np.log(t_ru / t_en)),
                                     "ru": ru, "en": en,
                                     "w_ru": len(ru.split(" ")), "w_en": len(en.split(" ")),
                                     "i_ru": i_ru, "i_en": i_en})
                per_concept.append({"concept_id": v["concept_id"], "options": opts})

            # 1. именованные сценарии
            def scenario(use_en_alt: bool, use_ru_alt: bool) -> float:
                s = fixed_sum
                for pc in per_concept:
                    v = by_cid[pc["concept_id"]]
                    i_en = 1 if (use_en_alt and len(v["en_options"]) > 1) else 0
                    i_ru = 1 if (use_ru_alt and len(v["ru_options"]) > 1) else 0
                    s += next(o["log"] for o in pc["options"]
                              if o["i_en"] == i_en and o["i_ru"] == i_ru)
                return float(np.exp(s / n))

            # 2. перебор сочетаний: отдельно по вариантам, которые не
            #    нарушают исходных требований к составу корпуса, и отдельно по
            #    всем засвидетельствованным вариантам. Смешивать их нельзя:
            #    вариант с дефисом или вариант, из-за которого семья
            #    разрастается сверх лимита, в исходный корпус не попал бы.
            for pc in per_concept:
                v = by_cid[pc["concept_id"]]
                for o in pc["options"]:
                    o["conforms_form"] = ("-" not in o["ru"] and "-" not in o["en"]
                                          and o["w_ru"] <= 3 and o["w_en"] <= 3)

            head_combo_ok = family_ok_combinations(concepts, by_cid, per_concept)

            def enumerate_G(only_conforming: bool):
                lists = []
                for pc in per_concept:
                    opts = [o for o in pc["options"]
                            if not only_conforming or o["conforms_form"]]
                    lists.append(opts or pc["options"][:1])
                vals = []
                for combo in itertools.product(*lists):
                    if only_conforming:
                        sig = tuple((pc["concept_id"], o["ru"].split(" ")[-1],
                                     o["en"].split(" ")[-1])
                                    for pc, o in zip(per_concept, combo))
                        if not head_combo_ok(sig):
                            continue
                    vals.append(np.exp((fixed_sum + sum(o["log"] for o in combo)) / n))
                return np.array(vals)

            all_G = enumerate_G(False)
            conf_G = enumerate_G(True)

            entry = {
                "n_combinations_all": int(all_G.size),
                "n_combinations_design_conforming": int(conf_G.size),
                "G_chosen": scenario(False, False),
                "G_en_alternative": scenario(True, False),
                "G_ru_alternative": scenario(False, True),
                "G_both_alternative": scenario(True, True),
                "design_conforming": {
                    "G_min": float(conf_G.min()), "G_max": float(conf_G.max()),
                    "G_median": float(np.median(conf_G)),
                },
                "all_attested": {
                    "G_min": float(all_G.min()), "G_max": float(all_G.max()),
                    "G_median": float(np.median(all_G)),
                },
            }
            for key, arr in (("design_conforming", conf_G), ("all_attested", all_G)):
                e = entry[key]
                e["spread_absolute"] = e["G_max"] - e["G_min"]
                e["spread_relative_pct"] = 100 * (e["G_max"] - e["G_min"]) / entry["G_chosen"]
            result["by_tokenizer"].setdefault(tid, {})[suffix] = entry

    # 3. структурная группа: не зависит от токенизатора
    def stratum_of(w_ru: int, w_en: int) -> str:
        if w_ru == 1 and w_en == 1:
            return "A"
        if w_ru > 1 and w_en > 1:
            return "B"
        return "C"

    group_changes = []
    for v in variants:
        cur = concepts.loc[concepts["concept_id"] == v["concept_id"]].iloc[0]
        base = stratum_of(len(v["ru_chosen"].split(" ")), len(v["en_chosen"].split(" ")))
        seen = set()
        for ru in v["ru_options"]:
            for en in v["en_options"]:
                seen.add(stratum_of(len(ru.split(" ")), len(en.split(" "))))
        if len(seen) > 1:
            group_changes.append({
                "concept_id": v["concept_id"],
                "en": v["en_chosen"], "ru": v["ru_chosen"],
                "stratum_declared": cur["stratum"], "stratum_base": base,
                "strata_possible": sorted(seen),
                "en_options": v["en_options"], "ru_options": v["ru_options"],
            })
    result["structure_group_depends_on_designation"] = {
        "n_concepts": len(group_changes), "details": group_changes,
    }

    # 4. семьи при крайних вариантах
    base_pairs = [(c["concept_id"], c["term_en_sg"], c["term_ru_sg"])
                  for _, c in concepts.iterrows()]
    base_sizes = family_sizes(base_pairs)
    alt_pairs = []
    for cid, en, ru in base_pairs:
        v = by_cid.get(cid)
        if v:
            en = v["en_options"][-1]
            ru = v["ru_options"][-1]
        alt_pairs.append((cid, en, ru))
    alt_sizes = family_sizes(alt_pairs)
    result["families"] = {
        "chosen_designations": {"n_families": len(base_sizes),
                                "max_size": max(base_sizes.values()),
                                "over_cap": sum(1 for s in base_sizes.values() if s > 3)},
        "last_variant_designations": {"n_families": len(alt_sizes),
                                      "max_size": max(alt_sizes.values()),
                                      "over_cap": sum(1 for s in alt_sizes.values() if s > 3)},
        "cap": 3,
    }

    # варианты с дефисом отмечаются: в основной корпус такие формы не берутся
    result["variants_with_hyphen"] = [
        {"concept_id": v["concept_id"], "designation": t}
        for v in variants for t in v["en_options"][1:] + v["ru_options"][1:] if "-" in t]

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    prim = result["by_tokenizer"][config.PRIMARY_TOKENIZER]["isolated_0"]
    print(f"\n{config.PRIMARY_TOKENIZER}, режим isolated_0:")
    print(f"  выбранные обозначения G = {prim['G_chosen']:.3f}")
    print(f"  замена EN / RU / обоих:  {prim['G_en_alternative']:.3f} / "
          f"{prim['G_ru_alternative']:.3f} / {prim['G_both_alternative']:.3f}")
    dc, al = prim["design_conforming"], prim["all_attested"]
    print(f"  варианты, не нарушающие исходных требований "
          f"({prim['n_combinations_design_conforming']} сочетаний): "
          f"{dc['G_min']:.3f} … {dc['G_max']:.3f} ({dc['spread_relative_pct']:.1f} %)")
    print(f"  все засвидетельствованные ({prim['n_combinations_all']}): "
          f"{al['G_min']:.3f} … {al['G_max']:.3f} ({al['spread_relative_pct']:.1f} %)")
    print(f"структурная группа зависит от обозначения у "
          f"{result['structure_group_depends_on_designation']['n_concepts']} понятий")
    print(f"семьи: {result['families']['chosen_designations']} -> "
          f"{result['families']['last_variant_designations']}")
    print(f"\n{OUT_JSON.relative_to(ROOT)}\n{OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
