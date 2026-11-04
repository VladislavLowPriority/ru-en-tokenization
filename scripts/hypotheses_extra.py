# -*- coding: utf-8 -*-
"""Пункт 8 плана: доработка H2 и H3 в ограниченном объёме.

Что здесь добавляется к уже посчитанному и зачем.

1. **H2 перестаёт быть одной средней цифрой.** Средняя разность в токенах
   ничего не говорит о том, как она получилась. Поэтому считается, у скольких
   понятий число токенов при переходе к множественному числу выросло,
   осталось прежним или уменьшилось, отдельно по языкам, и как это связано с
   числом изменившихся слов и с приростом длины в знаках.

2. **Прямое сравнение токенизаторов.** Фраза «у одного эффект значим, у
   другого нет» различия не доказывает: два интервала могут пересекаться при
   любой разнице. Поэтому считается парная разность эффектов между
   токенизаторами на одних и тех же понятиях, со своим интервалом.

3. **H3 — уточнение границ утверждения.** Сдвиг считается отдельно по каждому
   шаблону, и рядом показывается, у скольких понятий он в точности равен нулю.
   Ноль в среднем и ноль у каждого понятия — разные утверждения.

4. **Наглядный разбор.** Отношение длин RU/EN раскладывается на два множителя:
   во сколько раз русское обозначение длиннее в знаках и во сколько раз больше
   токенов приходится на знак. Это помогает понять наблюдение, но причиной
   не является и отдельным открытием не считается.

5. **Проверка правила лексических семей.** Корпус собирался по правилу, где
   ключом семьи становится не только головная лемма, но и последнее слово
   термина. Для русской конструкции «существительное + зависимое в родительном
   падеже» последнее слово — зависимое. Здесь показано, меняется ли оценка,
   если считать семьи строго по головным леммам.

Ничего не пересчитывается заново из текста: берутся сохранённые измерения.

Запуск:
    .venv\\Scripts\\python.exe scripts\\hypotheses_extra.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tok_pilot import config, metrics  # noqa: E402

WIDE = ROOT / "results" / "real" / "analysis_wide.csv"
CONCEPTS = ROOT / "data" / "real" / "concepts_pilot.csv"
CTX = ROOT / "results" / "real" / "context_token_comparisons.csv"
OUT_JSON = ROOT / "results" / "real" / "hypotheses_extra.json"
OUT_H2 = ROOT / "results" / "real" / "h2_by_concept.csv"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = pd.read_csv(WIDE)
    con = pd.read_csv(CONCEPTS, dtype=str, keep_default_na=False)
    con = con[con["final_status"] == "accepted"]
    cols = ["concept_id", "term_ru_sg", "term_en_sg", "term_ru_pl", "term_en_pl",
            "head_ru", "head_en", "term_pattern_ru", "term_pattern_en",
            "changed_components_ru", "changed_components_en", "pl_status"]
    wide = wide.merge(con[cols], on="concept_id", how="left", suffixes=("", "_c"))
    return wide, con


# --- 1. H2 по наблюдениям, а не одной цифрой ---------------------------------

def h2_breakdown(wide: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    rows = []
    for _, r in wide.iterrows():
        rows.append({
            "concept_id": r["concept_id"], "tokenizer_id": r["tokenizer_id"],
            "stratum": r["stratum"], "pattern_ru": r["term_pattern_ru"],
            "changed_ru": int(r["changed_components_ru"]),
            "changed_en": int(r["changed_components_en"]),
            "dT_ru": int(r["T_ru_pl_iso0"] - r["T_ru_sg_iso0"]),
            "dT_en": int(r["T_en_pl_iso0"] - r["T_en_sg_iso0"]),
            "dC_ru": int(r["C_ru_pl"] - r["C_ru_sg"]),
            "dC_en": int(r["C_en_pl"] - r["C_en_sg"]),
            "pl_status": r["pl_status"],
        })
    tab = pd.DataFrame(rows)
    tab["paired_diff"] = tab["dT_ru"] - tab["dT_en"]

    def direction(v: int) -> str:
        return "выросло" if v > 0 else ("без изменения" if v == 0 else "уменьшилось")

    out = {}
    for tid, d in tab.groupby("tokenizer_id"):
        counts_ru = d["dT_ru"].map(direction).value_counts().to_dict()
        counts_en = d["dT_en"].map(direction).value_counts().to_dict()
        cross = (d.assign(ru=d["dT_ru"].map(direction), en=d["dT_en"].map(direction))
                 .groupby(["ru", "en"]).size())
        by_pattern = {}
        for pat, dp in d.groupby("pattern_ru"):
            by_pattern[pat] = {
                "n": int(len(dp)),
                "mean_dT_ru": round(float(dp["dT_ru"].mean()), 3),
                "mean_dT_en": round(float(dp["dT_en"].mean()), 3),
                "mean_paired_diff": round(float(dp["paired_diff"].mean()), 3),
                "mean_dC_ru": round(float(dp["dC_ru"].mean()), 2),
                "mean_dC_en": round(float(dp["dC_en"].mean()), 2),
            }
        by_changed = {}
        for k, dk in d.groupby("changed_ru"):
            by_changed[int(k)] = {
                "n": int(len(dk)),
                "mean_dT_ru": round(float(dk["dT_ru"].mean()), 3),
                "mean_paired_diff": round(float(dk["paired_diff"].mean()), 3),
            }
        out[tid] = {
            "n_concepts": int(len(d)),
            "tokens_ru": counts_ru, "tokens_en": counts_en,
            "cross_ru_x_en": {f"RU {a} / EN {b}": int(v) for (a, b), v in cross.items()},
            "mean_char_growth_ru": round(float(d["dC_ru"].mean()), 2),
            "mean_char_growth_en": round(float(d["dC_en"].mean()), 2),
            "by_pattern_ru": by_pattern,
            "by_changed_components_ru": by_changed,
        }
    return out, tab


# --- 2. прямое сравнение токенизаторов --------------------------------------

def paired_tokenizer_contrast(wide: pd.DataFrame) -> dict:
    """Разность одного и того же показателя между двумя токенизаторами.

    Понятия те же самые, поэтому разность парная; интервал — тем же кластерным
    бутстрэпом по семьям, что и основные оценки.
    """
    res = {}
    tids = list(config.TOKENIZERS)
    for i, a in enumerate(tids):
        for b in tids[i + 1:]:
            da = wide[wide["tokenizer_id"] == a].set_index("concept_id")
            db = wide[wide["tokenizer_id"] == b].set_index("concept_id")
            common = da.index.intersection(db.index)
            merged = da.loc[common].join(db.loc[common], rsuffix="_b")
            merged = merged.reset_index()

            def m_diff(df: pd.DataFrame) -> float:
                ma = ((df["T_ru_pl_iso0"] - df["T_ru_sg_iso0"])
                      - (df["T_en_pl_iso0"] - df["T_en_sg_iso0"])).mean()
                mb = ((df["T_ru_pl_iso0_b"] - df["T_ru_sg_iso0_b"])
                      - (df["T_en_pl_iso0_b"] - df["T_en_sg_iso0_b"])).mean()
                return float(ma - mb)

            def g_ratio(df: pd.DataFrame) -> float:
                ga = np.exp(np.log(df["T_ru_sg_iso0"] / df["T_en_sg_iso0"]).mean())
                gb = np.exp(np.log(df["T_ru_sg_iso0_b"] / df["T_en_sg_iso0_b"]).mean())
                return float(ga / gb)

            key = f"{a} - {b}"
            res[key] = {
                "H2_M_difference": metrics.cluster_bootstrap(merged, m_diff),
                "H1_G_ratio": metrics.cluster_bootstrap(merged, g_ratio),
            }
    return res


# --- 3. H3 по шаблонам, с точным числом нулей --------------------------------

def h3_detail(wide: pd.DataFrame) -> dict:
    out = {}
    for tid, d in wide.groupby("tokenizer_id"):
        per_tpl = {}
        for tpl in config.ALL_TEMPLATES:
            vals = metrics.context_contrast_rows(d, [tpl])
            per_tpl[tpl] = {
                "estimate": metrics.cluster_bootstrap(d, metrics.make_stat_H3([tpl])),
                "n_concepts_exact_zero": int((vals == 0).sum()),
                "n_concepts": int(len(vals)),
                "min": float(vals.min()), "max": float(vals.max()),
            }
        out[tid] = per_tpl
    return out


# --- 4. разложение отношения длин -------------------------------------------

def decomposition(wide: pd.DataFrame) -> dict:
    out = {}
    for tid, d in wide.groupby("tokenizer_id"):
        def chars(df):
            return float(np.exp(np.log(df["C_ru_sg"] / df["C_en_sg"]).mean()))

        def tpc(df):
            r = (df["T_ru_sg_iso0"] / df["C_ru_sg"]) / (df["T_en_sg_iso0"] / df["C_en_sg"])
            return float(np.exp(np.log(r).mean()))

        g = metrics.cluster_bootstrap(d, metrics.stat_G)
        c = metrics.cluster_bootstrap(d, chars)
        t = metrics.cluster_bootstrap(d, tpc)
        out[tid] = {
            "G": g, "char_length_ratio": c, "tokens_per_char_ratio": t,
            "identity_check": round(c["point"] * t["point"] - g["point"], 12),
            "note": "G = (отношение длин в знаках) x (отношение числа токенов на знак); "
                    "это тождество, а не объяснение причины",
        }
    return out


# --- 5. правило лексических семей -------------------------------------------

def strict_head_families(con: pd.DataFrame) -> dict[str, str]:
    """Семьи строго по головным леммам (§3), без ключа «последнее слово»."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    heads = {"en": set(), "ru": set()}
    for _, r in con.iterrows():
        heads["en"].add(r["head_en"].lower())
        heads["ru"].add(r["head_ru"].lower())

    for _, r in con.iterrows():
        cid = "c:" + r["concept_id"]
        for lang in ("en", "ru"):
            h = r[f"head_{lang}"].lower()
            union(cid, f"{lang}:{h}")
            # словосложение: вершина сложного слова стоит справа
            for root in heads[lang]:
                if len(root) >= 4 and h.endswith(root) and h != root:
                    union(f"{lang}:{h}", f"{lang}:{root}")
    return {r["concept_id"]: find("c:" + r["concept_id"]) for _, r in con.iterrows()}


def family_rule_check(wide: pd.DataFrame, con: pd.DataFrame) -> dict:
    strict = strict_head_families(con)
    w = wide.copy()
    w["family_strict"] = w["concept_id"].map(strict)
    out = {
        "n_families_corpus_rule": int(w["family_id"].nunique()),
        "n_families_strict_head_rule": int(w["family_strict"].nunique()),
        "by_tokenizer": {},
    }
    for tid, d in w.groupby("tokenizer_id"):
        out["by_tokenizer"][tid] = {
            "G_corpus_rule": metrics.cluster_bootstrap(d, metrics.stat_G),
            "G_strict_head_rule": metrics.cluster_bootstrap(d, metrics.stat_G,
                                                            cluster="family_strict"),
        }
    return out


def main() -> None:
    wide, con = load()
    h2, h2_tab = h2_breakdown(wide)
    h2_tab.to_csv(OUT_H2, index=False, encoding="utf-8")

    ctx_note = {}
    if CTX.exists():
        ctx = pd.read_csv(CTX)
        ctx_note = {
            "n_comparisons": int(len(ctx)),
            "columns": list(ctx.columns)[:8],
        }

    result = {
        "source": "results/real/analysis_wide.csv (без повторной токенизации)",
        "H2_breakdown": h2,
        "tokenizer_contrasts": paired_tokenizer_contrast(wide),
        "H3_by_template": h3_detail(wide),
        "G_decomposition": decomposition(wide),
        "family_rule_check": family_rule_check(wide, con),
        "context_comparison_file": ctx_note,
        "scope_note": ("Утверждение о контексте относится только к проверенным "
                       "шаблонам A, B и диагностическому C, а не к языку вообще."),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float),
                        encoding="utf-8")

    prim = config.PRIMARY_TOKENIZER
    print(f"H2, {prim}: RU {h2[prim]['tokens_ru']}, EN {h2[prim]['tokens_en']}")
    print("Прямые разности эффекта H2 между токенизаторами:")
    for k, v in result["tokenizer_contrasts"].items():
        m = v["H2_M_difference"]
        print(f"  {k}: {m['point']:+.3f} [{m['ci_low']:+.3f}; {m['ci_high']:+.3f}]")
    d = result["G_decomposition"][prim]
    print(f"Разложение G ({prim}): длина в знаках {d['char_length_ratio']['point']:.3f} x "
          f"токенов на знак {d['tokens_per_char_ratio']['point']:.3f} = {d['G']['point']:.3f}")
    f = result["family_rule_check"]
    print(f"Семьи: правило корпуса {f['n_families_corpus_rule']}, "
          f"строго по вершинам {f['n_families_strict_head_rule']}")
    for tid, v in f["by_tokenizer"].items():
        a, b = v["G_corpus_rule"], v["G_strict_head_rule"]
        print(f"  {tid}: [{a['ci_low']:.3f}; {a['ci_high']:.3f}] -> "
              f"[{b['ci_low']:.3f}; {b['ci_high']:.3f}]")
    print(f"\n{OUT_JSON.relative_to(ROOT)}\n{OUT_H2.relative_to(ROOT)}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
