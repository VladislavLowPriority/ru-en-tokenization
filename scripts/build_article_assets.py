# -*- coding: utf-8 -*-
"""Сборка таблиц и рисунков статьи из зафиксированных результатов.

Скрипт **ничего не измеряет**. Он читает сохранённые JSON и CSV, строит
объекты, перечисленные в §4 `СКЕЛЕТ_СТАТЬИ.md`, и записывает манифест с
SHA-256 всех входов. Токенизатор вызывается один раз и только ради показа
готовых разбиений в таблице S1: идентификаторы токенов уже сохранены в
измерениях, здесь они лишь переводятся в читаемые строки.

Что собирается:

| Объект | Файл |
|---|---|
| Таблица 1 — материал | `article/tables/T1_material.*` |
| Таблица 2 — дополнительные проверки | `article/tables/T2_checks.*` |
| Таблица S1 — шесть примеров (§11.4) | `article/tables/S1_examples.*` |
| Таблица S2 — устойчивость | `article/tables/S2_robustness.*` |
| Рисунок 1 — R по срезам | `article/figures/fig1_R.*` |
| Рисунок 2 — G по A/B/C | `article/figures/fig2_G.*` |
| Рисунок S1 — SG→PL | `article/figures/figS1_sgpl.*` |
| Рисунок S2 — окружение | `article/figures/figS2_context.*` |

Правила оформления взяты из §4 скелета: единые цвета токенизаторов, три
формата (SVG, PDF, PNG 300 dpi), логарифмическая ось для отношений, никаких
объёмных эффектов и декоративных схем сегментации.

Запуск:
    .venv\\Scripts\\python.exe scripts\\build_article_assets.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EXP = ROOT / "expansion_wipo_v1"
OUT = ROOT / "article"
FIG = OUT / "figures"
TAB = OUT / "tables"

# Входные файлы. Любое число в статье должно прослеживаться до одного из них.
INPUTS = {
    "freeze": EXP / "plan" / "FREEZE.json",
    "combined_summary": EXP / "results" / "combined" / "summary.json",
    "new_summary": EXP / "results" / "new" / "summary.json",
    "sensitivity": EXP / "results" / "sensitivity" / "sensitivity_summary.json",
    "combined_wide": EXP / "results" / "combined" / "analysis_wide.csv",
    "combined_measurements": EXP / "results" / "combined" / "measurements.csv",
    "combined_validation": EXP / "results" / "combined" / "validation.json",
    "concepts_combined": EXP / "data" / "concepts_combined.csv",
    "pairs_combined": EXP / "data" / "matching_pairs_combined.csv",
    "unmatched": EXP / "data" / "unmatched_technical.csv",
    "matching_quality": EXP / "data" / "matching_quality.json",
    "baseline_R": ROOT / "results" / "control" / "estimates_R.json",
    "synonyms": ROOT / "results" / "real" / "synonym_sensitivity.json",
    "hypotheses_extra": ROOT / "results" / "real" / "hypotheses_extra.json",
}

TOKENIZERS = ["o200k_base", "cl100k_base", "qwen2.5-7b-instruct"]
TOK_LABEL = {"o200k_base": "o200k_base", "cl100k_base": "cl100k_base",
             "qwen2.5-7b-instruct": "Qwen2.5-7B"}
TOK_COLOR = {"o200k_base": "#1f4e79", "cl100k_base": "#b4531f",
             "qwen2.5-7b-instruct": "#3f6b34"}
STRATUM_COLOR = {"A": "#4477aa", "B": "#cc6677", "C": "#117733"}
LN_MARGIN = 0.095          # коридор практической эквивалентности из §14.1

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(key: str) -> dict:
    return json.loads(INPUTS[key].read_text(encoding="utf-8"))


def save_fig(fig, name: str) -> list[str]:
    FIG.mkdir(parents=True, exist_ok=True)
    made = []
    for ext, kw in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        p = FIG / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", **kw)
        made.append(str(p.relative_to(ROOT)))
    plt.close(fig)
    return made


# Запятая ставится только в самостоятельных числах: «v1.2» и «Qwen2.5-7B» —
# это обозначения версий, их трогать нельзя.
RU_DECIMAL = re.compile(r"(?<![A-Za-z\u0410-\u044f\d])(\d+)\.(\d+)(?![A-Za-z\u0410-\u044f])")


def save_table(df: pd.DataFrame, name: str, note: str = "") -> list[str]:
    """CSV — для машины (точка как разделитель), Markdown — для текста статьи.

    В русском тексте дробная часть отделяется запятой, поэтому в Markdown
    десятичные точки заменяются; в CSV они остаются, иначе файл перестанет
    читаться обычными средствами.
    """
    TAB.mkdir(parents=True, exist_ok=True)
    csv_path = TAB / f"{name}.csv"
    md_path = TAB / f"{name}.md"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    lines = ["| " + " | ".join(df.columns) + " |",
             "|" + "|".join(["---"] * len(df.columns)) + "|"]
    for _, r in df.iterrows():
        cells = ["" if pd.isna(v) else RU_DECIMAL.sub(r"\1,\2", str(v)) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    if note:
        lines += ["", note]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [str(csv_path.relative_to(ROOT)), str(md_path.relative_to(ROOT))]


def ci(d: dict, digits: int = 4) -> str:
    return f"{d['point']:.{digits}f} [{d['ci_low']:.{digits}f}; {d['ci_high']:.{digits}f}]"


# --- Таблица 1: материал -----------------------------------------------------

def table1() -> tuple[pd.DataFrame, str]:
    freeze = load_json("freeze")
    quality = load_json("matching_quality")
    comb = load_json("combined_summary")
    new = load_json("new_summary")
    base = load_json("baseline_R")

    sizes = freeze["sample_sizes"]
    pairs = pd.read_csv(INPUTS["pairs_combined"], dtype=str)
    unmatched = pd.read_csv(INPUTS["unmatched"], dtype=str)

    def strat(d: dict) -> str:
        return "/".join(str(d.get(s, 0)) for s in ("A", "B", "C"))

    base_mc = base["by_tokenizer"]["o200k_base"]
    new_mc = new["by_tokenizer"]["o200k_base"]["matched_control"]
    comb_mc = comb["by_tokenizer"]["o200k_base"]["matched_control"]

    base_pairs = pairs[pairs["tech_cohort"] != "wipo_expansion_v1"]
    new_pairs = pairs[pairs["tech_cohort"] == "wipo_expansion_v1"]

    rows = [
        {
            "Срез": "Исходный (фаза 1)",
            "Источники технических понятий": "руководства производителей, WIPO Pearl",
            "Технических понятий": 90,
            "A/B/C": "30/30/30",
            "Сопоставлений с контролем": base_mc["n_pairs"],
            "Покрытие, %": f"{100 * base_mc['n_pairs'] / 90:.1f}",
            "Семей техн. / контр.": f"{base_mc['n_families_tech']}/{base_mc['n_families_control']}",
            "Совместных кластеров": base_mc["n_clusters_joint"],
        },
        {
            "Срез": "Новые понятия (расширение v1.2)",
            "Источники технических понятий": "WIPO Pearl",
            "Технических понятий": sizes["technical_new_count"],
            "A/B/C": strat(sizes["technical_new_stratum"]),
            "Сопоставлений с контролем": new_mc["n_pairs"],
            "Покрытие, %": f"{100 * new_mc['n_pairs'] / sizes['technical_new_count']:.1f}",
            "Семей техн. / контр.": f"{new_mc['n_tech_families']}/{new_mc['n_ctrl_families']}",
            "Совместных кластеров": new_mc["n_joint_clusters"],
        },
        {
            "Срез": "Объединённый (v1.2)",
            "Источники технических понятий": "руководства производителей, WIPO Pearl",
            "Технических понятий": sizes["technical_combined_count"],
            "A/B/C": strat(sizes["technical_combined_stratum"]),
            "Сопоставлений с контролем": comb_mc["n_pairs"],
            "Покрытие, %": f"{100 * quality['match_rate_combined']:.1f}",
            "Семей техн. / контр.": f"{comb_mc['n_tech_families']}/{comb_mc['n_ctrl_families']}",
            "Совместных кластеров": comb_mc["n_joint_clusters"],
        },
    ]
    note = (
        f"Контроль — Concepticon (Miklashevsky-2018-506) и таблицы переводов "
        f"английского Викисловаря. Без контрольной пары остались "
        f"{len(unmatched)} технических понятий "
        f"(A/B/C = {strat(unmatched['stratum'].value_counts().to_dict())}); "
        f"средняя |разница| длин в сопоставленных парах "
        f"{quality['mean_abs_d_RU_chars']:.2f} знака в русском и "
        f"{quality['mean_abs_d_EN_chars']:.2f} в английском, допуск — 4 знака. "
        f"Совместный кластер — связная группа лексических семей обеих выборок; "
        f"это единица перевыбора, а не независимый источник данных."
    )
    assert len(base_pairs) + len(new_pairs) == comb_mc["n_pairs"]
    return pd.DataFrame(rows), note


# --- Таблица 2: дополнительные проверки --------------------------------------

def table2(context: dict) -> tuple[pd.DataFrame, str]:
    comb = load_json("combined_summary")
    syn = load_json("synonyms")["by_tokenizer"]["o200k_base"]["isolated_0"]
    rows = []
    for tid in TOKENIZERS:
        m = comb["by_tokenizer"][tid]["H2_M_all"]
        rows.append({
            "Проверка": f"Число: M, {TOK_LABEL[tid]}",
            "Срез": "объединённый v1.2",
            "N": m["n_concepts"], "Единица N": "понятий",
            "Значение и 95 % ДИ": f"{m['point']:+.3f} [{m['ci_low']:+.3f}; {m['ci_high']:+.3f}] токена",
        })
    rows.append({
        "Проверка": "Обозначение: диапазон G при замене засвидетельствованных вариантов",
        "Срез": "подвыборка фазы 1 (12 понятий с вариантами внутри корпуса 90)",
        "N": 12, "Единица N": "понятий с вариантами",
        "Значение и 95 % ДИ": (f"{syn['design_conforming']['G_min']:.3f} … "
                               f"{syn['design_conforming']['G_max']:.3f} "
                               f"(ширина {syn['design_conforming']['spread_relative_pct']:.1f} % от G)"),
    })
    rows.append({
        "Проверка": "Окружение: шаблоны A и B против режима с начальным пробелом",
        "Срез": "объединённый v1.2",
        "N": context["n_compared_AB"], "Единица N": "сравнений строк",
        "Значение и 95 % ДИ": (f"несовпадений {context['mismatch_AB']} "
                               f"(сверялись последовательности ID и границы токенов)"),
    })
    rows.append({
        "Проверка": "Окружение: диагностический шаблон C против режима без пробела",
        "Срез": "объединённый v1.2",
        "N": context["n_compared_C"], "Единица N": "сравнений строк",
        "Значение и 95 % ДИ": (f"несовпадений {context['mismatch_C']}; "
                               f"против режима с пробелом — "
                               f"{context['c_vs_isolated_space_mismatch']}"),
    })
    note = (
        "Каждая строка имеет собственный срез и собственное N; строки нельзя "
        "складывать. Диапазон по обозначениям получен на подвыборке первой фазы "
        "и не является проверкой всех 159 понятий. M — разность изменений RU и EN "
        "при переходе SG→PL, в токенах."
    )
    return pd.DataFrame(rows), note


# --- Таблица S1: шесть примеров по правилу §11.4 ------------------------------

def tableS1() -> tuple[pd.DataFrame, str]:
    wide = pd.read_csv(INPUTS["combined_wide"])
    concepts = pd.read_csv(INPUTS["concepts_combined"], dtype=str)
    meas = pd.read_csv(INPUTS["combined_measurements"],
                       usecols=["concept_id", "lang", "number", "mode",
                                "tokenizer_id", "target_tokens"])
    d = wide[wide["tokenizer_id"] == "o200k_base"].copy()
    d["g_sg"] = d["T_ru_sg_iso0"] / d["T_en_sg_iso0"]
    d["contrast"] = ((d["T_ru_pl_iso0"] - d["T_ru_sg_iso0"])
                     - (d["T_en_pl_iso0"] - d["T_en_sg_iso0"]))

    from tok_pilot import backends
    enc = backends.load_all()["o200k_base"]

    def pieces(cid: str, lang: str, term: str) -> str:
        """Показ разбиения: строка кодируется заново и сверяется с измерением.

        Сверка обязательна — иначе в таблицу примеров может попасть картинка,
        не совпадающая с тем, что реально посчитано. Байты токена
        декодируются с заменой: byte-level токен может разрезать букву, и это
        видно как символ замены, а не скрывается.
        """
        row = meas[(meas.concept_id == cid) & (meas.lang == lang)
                   & (meas.number == "sg") & (meas["mode"] == "isolated_0")
                   & (meas.tokenizer_id == "o200k_base")]
        saved = [t["id"] for t in json.loads(row.iloc[0]["target_tokens"])]
        e = enc.encode(term)
        if list(e.ids) != saved:
            raise SystemExit(f"Разбиение {cid}/{lang} не совпало с измерением")
        # Разделитель — точка по центру: вертикальная черта разорвала бы
        # ячейку Markdown-таблицы. Пробел внутри токена показан знаком
        # открытого прямоугольника, иначе «токен с ведущим пробелом»
        # неотличим от обычного.
        return " · ".join(
            b.decode("utf-8", errors="replace").replace(" ", "␣")
            for b in e.token_bytes)

    picked, rows = set(), []
    for stratum in ("A", "B", "C"):
        ds = d[d["stratum"] == stratum].sort_values("concept_id")
        median = ds["g_sg"].median()
        by_median = ds.assign(dist=(ds["g_sg"] - median).abs()).sort_values(
            ["dist", "concept_id"])
        by_contrast = ds.assign(ac=ds["contrast"].abs()).sort_values(
            ["ac", "concept_id"], ascending=[False, True])
        for kind, table in (("около медианы SG-отношения", by_median),
                            ("наибольший контраст SG→PL", by_contrast)):
            for _, r in table.iterrows():
                if r["concept_id"] in picked:
                    continue                      # «при совпадении брать следующую»
                picked.add(r["concept_id"])
                c = concepts[concepts["concept_id"] == r["concept_id"]].iloc[0]
                rows.append({
                    "Группа": stratum,
                    "Правило отбора": kind,
                    "concept_id": r["concept_id"],
                    "RU (ед. ч.)": c["term_ru_sg"],
                    "EN (ед. ч.)": c["term_en_sg"],
                    "T_RU / T_EN": f"{int(r['T_ru_sg_iso0'])} / {int(r['T_en_sg_iso0'])}",
                    "g": f"{r['g_sg']:.3f}",
                    "Δ SG→PL, RU−EN": f"{r['contrast']:+.0f}",
                    "Разбиение RU": pieces(r["concept_id"], "ru", c["term_ru_sg"]),
                    "Разбиение EN": pieces(r["concept_id"], "en", c["term_en_sg"]),
                })
                break
    note = (
        "Правило отбора объявлено в §11.4 протокола до расчёта: в каждой группе "
        "берётся понятие около медианы SG-отношения и понятие с наибольшим "
        "абсолютным контрастом SG→PL для o200k_base; при совпадении берётся "
        "следующее по порядку. Разбиения показаны для режима isolated_0; "
        "вертикальная черта — граница токенов. Это иллюстрации, а не типичные "
        "случаи."
    )
    return pd.DataFrame(rows), note


# --- Таблица S2: устойчивость -------------------------------------------------

def tableS2() -> tuple[pd.DataFrame, str]:
    comb = load_json("combined_summary")
    new = load_json("new_summary")
    base = load_json("baseline_R")
    rows = []
    for tid in TOKENIZERS:
        b = base["by_tokenizer"][tid]
        n = new["by_tokenizer"][tid]["matched_control"]
        c = comb["by_tokenizer"][tid]["matched_control"]
        rows += [
            {"Токенизатор": TOK_LABEL[tid], "Проверка": "R, исходный срез",
             "N сопоставлений": b["n_pairs"], "Значение и 95 % ДИ": ci(b["R_raw"])},
            {"Токенизатор": TOK_LABEL[tid], "Проверка": "R, только новые понятия",
             "N сопоставлений": n["n_pairs"], "Значение и 95 % ДИ": ci(n["R_raw"])},
            {"Токенизатор": TOK_LABEL[tid], "Проверка": "R, объединённый срез",
             "N сопоставлений": c["n_pairs"], "Значение и 95 % ДИ": ci(c["R_raw"])},
        ]
        for s in ("A", "B", "C"):
            rs = c["R_by_stratum"][s]
            rows.append({"Токенизатор": TOK_LABEL[tid],
                         "Проверка": f"R по группе {s}, объединённый срез",
                         "N сопоставлений": rs["n_concepts"],
                         "Значение и 95 % ДИ": ci(rs)})
        for key, label in (("R_adjusted_length_structure", "R с поправкой на длину и структуру"),
                           ("R_adjusted_frequency", "R с совместной поправкой на длину, структуру и частотность"),
                           ("R_isolated_sp", "R в режиме с начальным пробелом")):
            if key in c:
                v = c[key]
                point = v.get("point", v.get("R_adjusted"))
                rows.append({
                    "Токенизатор": TOK_LABEL[tid], "Проверка": label,
                    "N сопоставлений": v.get("n_concepts", v.get("n_pairs")),
                    "Значение и 95 % ДИ":
                        f"{point:.4f} [{v['ci_low']:.4f}; {v['ci_high']:.4f}]"})
    note = (
        "Все интервалы — кластерный бутстрэп, 5000 повторов, seed 20260918, "
        "единица перевыбора — совместный кластер семей обеих выборок. "
        "Поправки на длину, структуру и частотность действуют совместно: "
        "сдвиг нельзя приписывать одной только частотности. Строки с разными N "
        "относятся к разным подмножествам и не сопоставляются напрямую."
    )
    return pd.DataFrame(rows), note


# --- Рисунок 1: R по срезам ---------------------------------------------------

def figure1() -> list[str]:
    comb = load_json("combined_summary")
    new = load_json("new_summary")
    base = load_json("baseline_R")

    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.1), sharex=True)
    for ax, tid in zip(axes, TOKENIZERS):
        slices = [
            ("Объединённый", comb["by_tokenizer"][tid]["matched_control"]["R_raw"]),
            ("Только новые", new["by_tokenizer"][tid]["matched_control"]["R_raw"]),
            ("Исходный", base["by_tokenizer"][tid]["R_raw"]),
        ]
        ax.axvspan(np.exp(-LN_MARGIN), np.exp(LN_MARGIN), color="#dfe6ec", zorder=0)
        ax.axvline(1.0, color="#444444", lw=0.9, zorder=1)
        for y, (label, d) in enumerate(slices):
            ax.plot([d["ci_low"], d["ci_high"]], [y, y], color=TOK_COLOR[tid],
                    lw=1.6, solid_capstyle="butt", zorder=3)
            # 90 %-й интервал рисуется только там, где он сохранён: в расчётах
            # первой фазы его нет, и дорисовывать его нечем.
            if "ci_90_low" in d:
                ax.plot([d["ci_90_low"], d["ci_90_high"]], [y, y], color=TOK_COLOR[tid],
                        lw=4.0, alpha=0.35, solid_capstyle="butt", zorder=2)
            ax.plot([d["point"]], [y], "o", color=TOK_COLOR[tid], ms=4.5, zorder=4)
            ax.annotate(f"N={d['n_concepts']}", (d["ci_high"], y),
                        textcoords="offset points", xytext=(4, -3), fontsize=7,
                        color="#333333")
        ax.set_yticks(range(len(slices)))
        ax.set_yticklabels([s[0] for s in slices] if tid == TOKENIZERS[0] else [])
        ax.set_xscale("log")
        ax.set_xlim(0.78, 1.28)
        ax.set_xticks([0.8, 0.9, 1.0, 1.1, 1.2])
        ax.set_xticklabels(["0,8", "0,9", "1,0", "1,1", "1,2"])
        ax.set_ylim(-0.6, len(slices) - 0.4)
        ax.set_title(TOK_LABEL[tid])
        ax.grid(axis="y", visible=False)
    axes[1].set_xlabel("R — отношение RU/EN-разрыва терминов к разрыву контроля (лог. шкала)")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor="#dfe6ec", label="коридор практической эквивалентности "
                                         "[exp(-0,095); exp(0,095)]"),
        Line2D([0], [0], color="#555555", lw=1.6, label="95 % интервал"),
        Line2D([0], [0], color="#555555", lw=4.0, alpha=0.35,
               label="90 % интервал (там, где сохранён)"),
        Line2D([0], [0], color="#555555", marker="o", ls="none", ms=4.5,
               label="точечная оценка"),
    ]
    fig.tight_layout()
    # Легенда выносится под рисунок: внутри панели она перекрывает данные и
    # сжимает ось так, что подписи делений сливаются.
    fig.legend(handles=legend, frameon=False, fontsize=7, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.13), handlelength=1.8,
               columnspacing=1.4)
    return save_fig(fig, "fig1_R")


# --- Рисунок 2: G по группам --------------------------------------------------

def figure2() -> list[str]:
    comb = load_json("combined_summary")
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    strata = ["A", "B", "C"]
    offs = {"o200k_base": -0.22, "cl100k_base": 0.0, "qwen2.5-7b-instruct": 0.22}
    for tid in TOKENIZERS:
        by = comb["by_tokenizer"][tid]["H1_G_by_stratum"]
        xs, pts, los, his = [], [], [], []
        for i, s in enumerate(strata):
            d = by[s]
            xs.append(i + offs[tid])
            pts.append(d["point"]); los.append(d["ci_low"]); his.append(d["ci_high"])
        ax.vlines(xs, los, his, color=TOK_COLOR[tid], lw=1.6)
        ax.plot(xs, pts, "o", color=TOK_COLOR[tid], ms=5, label=TOK_LABEL[tid])
    ax.axhline(1.0, color="#444444", lw=0.9)
    ax.set_yscale("log")
    ax.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    ax.set_yticklabels(["1,0", "1,5", "2,0", "2,5", "3,0", "3,5"])
    ns = {s: comb["by_tokenizer"]["o200k_base"]["H1_G_by_stratum"][s]["n_concepts"]
          for s in strata}
    ax.set_xticks(range(len(strata)))
    ax.set_xticklabels([f"{s}\nN = {ns[s]}" for s in strata])
    ax.set_ylabel("G: отношение RU/EN")
    ax.set_xlabel("Структурная группа")
    ax.set_ylim(0.92, 3.8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return save_fig(fig, "fig2_G")


# --- Рисунок S1: SG→PL --------------------------------------------------------

def figureS1() -> list[str]:
    wide = pd.read_csv(INPUTS["combined_wide"])
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.2), sharex=True, sharey=True)
    for ax, tid in zip(axes, TOKENIZERS):
        d = wide[wide["tokenizer_id"] == tid].copy()
        d["dx"] = d["T_en_pl_iso0"] - d["T_en_sg_iso0"]
        d["dy"] = d["T_ru_pl_iso0"] - d["T_ru_sg_iso0"]
        grp = d.groupby(["stratum", "dx", "dy"]).size().reset_index(name="n")
        lim = [min(grp["dx"].min(), grp["dy"].min()) - 0.6,
               max(grp["dx"].max(), grp["dy"].max()) + 0.6]
        ax.plot(lim, lim, color="#888888", lw=0.8, ls="--", zorder=1)
        # Δ — целые числа, поэтому группы попадают ровно в одну точку и
        # закрывают друг друга. Разносим их постоянным смещением: положение
        # читается по подписям осей, смещение только разделяет группы.
        shift = {"A": (-0.13, 0.08), "B": (0.13, 0.08), "C": (0.0, -0.12)}
        for s in ("A", "B", "C"):
            g = grp[grp["stratum"] == s]
            ax.scatter(g["dx"] + shift[s][0], g["dy"] + shift[s][1],
                       s=8 + 9 * g["n"], alpha=0.85,
                       color=STRATUM_COLOR[s], edgecolor="white", linewidth=0.4,
                       label=s if tid == TOKENIZERS[0] else None, zorder=2)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(TOK_LABEL[tid])
        ax.set_xlabel("Δ токенов EN")
    axes[0].set_ylabel("Δ токенов RU")
    axes[0].legend(frameon=False, fontsize=7.5, title="Группа", title_fontsize=7.5,
                   loc="upper left")
    sample_sizes = [1, 5, 15, 30]
    size_handles = [
        plt.scatter([], [], s=8 + 9 * n, color="#666666", alpha=0.75,
                    edgecolors="white", linewidth=0.4)
        for n in sample_sizes
    ]
    fig.tight_layout()
    fig.legend(size_handles, [f"N = {n}" for n in sample_sizes],
               frameon=False, fontsize=7.5, ncol=4, title="Число понятий в точке:",
               title_fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, -0.14),
               columnspacing=1.8)
    return save_fig(fig, "figS1_sgpl")


# --- Рисунок S2: окружение ----------------------------------------------------

def figureS2() -> list[str]:
    import matplotlib.ticker as ticker
    wide = pd.read_csv(INPUTS["combined_wide"])
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2),
                             gridspec_kw={"width_ratios": [5, 3], "wspace": 0.35})

    comma_fmt = ticker.FuncFormatter(lambda x, pos: f"{x:.1f}".replace(".", ","))

    # панель 1: отношение по токенам ТЕРМИНА в разных условиях предъявления
    conds = [("isolated_0", "T_ru_sg_iso0", "T_en_sg_iso0", "изолированно"),
             ("isolated_space", "S_ru_sg_isosp", "S_en_sg_isosp", "с пробелом"),
             ("sentence_A", "S_ru_sg_sA", "S_en_sg_sA", "шаблон A"),
             ("sentence_B", "S_ru_sg_sB", "S_en_sg_sB", "шаблон B"),
             ("sentence_C", "S_ru_sg_sC", "S_en_sg_sC", "шаблон C")]
    for j, tid in enumerate(TOKENIZERS):
        d = wide[wide["tokenizer_id"] == tid]
        xs = np.arange(len(conds))
        ys = [float(np.exp(np.log(d[a] / d[b]).mean())) for _, a, b, _ in conds]
        # линия соединяет условия одного токенизатора: там, где значение не
        # меняется, отрезок виден как горизонталь — это и есть результат H3
        axes[0].plot(xs, ys, "-o", color=TOK_COLOR[tid], ms=5, lw=1.0,
                     label=TOK_LABEL[tid])
    axes[0].set_xticks(range(len(conds)))
    axes[0].set_xticklabels([c[3] for c in conds], fontsize=7.5, rotation=18,
                            ha="right")
    axes[0].set_ylabel("RU/EN по токенам термина (S)", labelpad=2)
    axes[0].yaxis.set_major_formatter(comma_fmt)
    axes[0].set_title("Токены самого термина (S)\n(SG, N = 159, геом. среднее)", fontsize=8)
    # Легенда общая и вынесена под рисунок: внутри левой панели она
    # перекрывает верхнюю линию, а справа — упирается во вторую панель.
    axes[0].grid(axis="x", visible=False)

    # панель 2: ПОЛНОЕ предложение — другой объект, отдельная панель
    sent = [("T_ru_sg_sA", "T_en_sg_sA", "шаблон A"),
            ("T_ru_sg_sB", "T_en_sg_sB", "шаблон B"),
            ("T_ru_sg_sC", "T_en_sg_sC", "шаблон C")]
    for j, tid in enumerate(TOKENIZERS):
        d = wide[wide["tokenizer_id"] == tid]
        xs = np.arange(len(sent))
        ys = [float(np.exp(np.log(d[a] / d[b]).mean())) for a, b, _ in sent]
        axes[1].plot(xs, ys, "-s", color=TOK_COLOR[tid], ms=5, lw=1.0)
    axes[1].set_xticks(range(len(sent)))
    axes[1].set_xticklabels([s[2] for s in sent], fontsize=7.5, rotation=18,
                            ha="right")
    axes[1].set_ylabel("RU/EN по предложению целиком (T)", labelpad=2)
    axes[1].yaxis.set_major_formatter(comma_fmt)
    axes[1].set_title("Предложение целиком (T): другой объект\n(SG, N = 159, геом. среднее)", fontsize=8)
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.16), columnspacing=2.2)
    return save_fig(fig, "figS2_context")


# --- проверка совпадения разбиений в контексте --------------------------------

def context_identity() -> dict:
    """Совпадают ли сами последовательности токенов термина, а не только их число.

    Сравниваются идентификаторы токенов и их байтовые границы относительно
    начала термина: режим `isolated_space` против каждого шаблона. Это та же
    проверка, что в аудите первой фазы, здесь — на объединённом корпусе.
    """
    meas = pd.read_csv(INPUTS["combined_measurements"],
                       usecols=["concept_id", "lang", "number", "mode",
                                "tokenizer_id", "target_tokens", "term_byte_start"])

    def signature(row) -> tuple:
        toks = json.loads(row["target_tokens"])
        start = int(row["term_byte_start"])
        return tuple((t["id"], t["span"][0] - start, t["span"][1] - start) for t in toks)

    meas["sig"] = meas.apply(signature, axis=1)
    key = ["concept_id", "lang", "number", "tokenizer_id"]
    refs = {"isolated_space": meas[meas["mode"] == "isolated_space"].set_index(key)["sig"],
            "isolated_0": meas[meas["mode"] == "isolated_0"].set_index(key)["sig"]}

    out: dict = {"pairs": {}}
    for mode in ("sentence_A", "sentence_B", "sentence_C"):
        sub = meas[meas["mode"] == mode].set_index(key)["sig"]
        for ref_name, ref in refs.items():
            joined = pd.concat([ref.rename("ref"), sub.rename("ctx")], axis=1).dropna()
            mismatch = int((joined["ref"] != joined["ctx"]).sum())
            out["pairs"][f"{mode}_vs_{ref_name}"] = {
                "n_compared": int(len(joined)), "n_mismatch": mismatch}
    joined = pd.concat([refs["isolated_0"].rename("a"),
                        refs["isolated_space"].rename("b")], axis=1).dropna()
    out["pairs"]["isolated_space_vs_isolated_0"] = {
        "n_compared": int(len(joined)),
        "n_mismatch": int((joined["a"] != joined["b"]).sum())}

    # Сводка для таблиц: A и B сравниваются с режимом «с пробелом», потому что
    # в этих шаблонах слева от термина стоит пробел; C — с режимом без пробела,
    # потому что там слева стоит открывающая скобка.
    ab = [out["pairs"]["sentence_A_vs_isolated_space"],
          out["pairs"]["sentence_B_vs_isolated_space"]]
    c = out["pairs"]["sentence_C_vs_isolated_0"]
    out["n_compared_AB"] = sum(x["n_compared"] for x in ab)
    out["mismatch_AB"] = sum(x["n_mismatch"] for x in ab)
    out["n_compared_C"] = c["n_compared"]
    out["mismatch_C"] = c["n_mismatch"]
    out["c_vs_isolated_space_mismatch"] = out["pairs"]["sentence_C_vs_isolated_space"]["n_mismatch"]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [k for k, p in INPUTS.items() if not p.exists()]
    if missing:
        raise SystemExit("Нет входных файлов: " + ", ".join(missing))

    print("[1/5] Проверка совпадения разбиений в контексте")
    ctx = context_identity()
    print(f"  шаблоны A и B: сравнений {ctx['n_compared_AB']}, "
          f"несовпадений {ctx['mismatch_AB']}")
    print(f"  шаблон C: сравнений {ctx['n_compared_C']}, "
          f"несовпадений {ctx['mismatch_C']}")

    print("[2/5] Таблицы")
    made = {}
    t1, n1 = table1(); made["T1_material"] = save_table(t1, "T1_material", n1)
    t2, n2 = table2(ctx); made["T2_checks"] = save_table(t2, "T2_checks", n2)
    s2, ns2 = tableS2(); made["S2_robustness"] = save_table(s2, "S2_robustness", ns2)

    print("[3/5] Таблица S1 (нужен токенизатор только для показа разбиений)")
    s1, ns1 = tableS1(); made["S1_examples"] = save_table(s1, "S1_examples", ns1)

    print("[4/5] Рисунки")
    made["fig1_R"] = figure1()
    made["fig2_G"] = figure2()
    made["figS1_sgpl"] = figureS1()
    made["figS2_context"] = figureS2()

    print("[5/5] Манифест")
    import importlib.metadata as md
    manifest = {
        "built_at_utc": pd.Timestamp.utcnow().isoformat(),
        "note": "Сборка оформления; измерения не выполнялись, результаты не менялись.",
        "inputs": {k: {"path": str(p.relative_to(ROOT)), "sha256": sha256(p)}
                   for k, p in INPUTS.items()},
        "outputs": made,
        "context_identity_check": ctx,
        "versions": {p: md.version(p) for p in ("matplotlib", "pandas", "numpy")},
        "equivalence_margin_ln": LN_MARGIN,
        "tokenizer_colors": TOK_COLOR,
    }
    (OUT / "assets_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nГотово. Объекты статьи:")
    for name, files in made.items():
        print(f"  {name}: " + ", ".join(files))
    print(f"  манифест: {(OUT / 'assets_manifest.json').relative_to(ROOT)}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
