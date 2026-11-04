# -*- coding: utf-8 -*-
"""Пункт 6 плана: измерение контроля и расчёт главного нового показателя R (§14).

R = G(техническая выборка) / G(общеязыковой контроль).

Что это значит простыми словами. G показывает, во сколько раз русское
обозначение длиннее английского в токенах. Такое различие есть и у обычных
слов — это известно из литературы. R отвечает на другой вопрос: **больше ли
это различие у терминов, чем у сопоставимой бытовой лексики**. R около единицы
означает, что ничего специфически терминологического в разрыве нет.

Порядок расчёта повторяет §14.5:

1. исходное сравнение на всех сопоставленных парах;
2. сравнение с учётом остаточной разницы длины и структуры (регрессия);
3. дополнительная поправка на частотность — только там, где частота известна
   для всех компонентов обоих языков.

Интервалы: перевыбираются **сопоставленные пары вместе с лексической семьёй
технического понятия**. Контрольное понятие всегда следует за своей парой,
поэтому связь, созданная сопоставлением, не теряется.

Множественное число контроля не измеряется (§14.6), поэтому H2 здесь нет.

Запуск:
    .venv\\Scripts\\python.exe scripts\\run_control.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tok_pilot import backends, config, manifest, measure, metrics, stimuli, validate  # noqa: E402

TECH_CONCEPTS = ROOT / "data" / "real" / "concepts_pilot.csv"
CTRL_CONCEPTS = ROOT / "data" / "control" / "concepts_control.csv"
TECH_WIDE = ROOT / "results" / "real" / "analysis_wide.csv"
OUT = ROOT / "results" / "control"


def json_dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")


def measure_control() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    concepts = stimuli.load_concepts(CTRL_CONCEPTS)
    concepts = concepts[concepts["final_status"] == "accepted"].copy()
    print(f"[1/5] Контрольных понятий: {len(concepts)}, "
          f"семей: {concepts['family_id'].nunique()}")

    st = stimuli.build_stimuli(concepts, numbers=("sg",))
    st = stimuli.add_frequencies(st)
    OUT.mkdir(parents=True, exist_ok=True)
    st.to_csv(OUT / "stimuli.csv", index=False, encoding="utf-8")
    print(f"[2/5] Строк: {len(st)}")

    bks = backends.load_all()
    manifest.save(manifest.build(bks, [CTRL_CONCEPTS], st, "control"), OUT / "manifest.json")

    print("[3/5] Измерение и повторный прогон")
    m1 = measure.measure(st, bks, run_id="run1")
    m2 = measure.measure(st, bks, run_id="run2")
    m1.to_csv(OUT / "measurements.csv", index=False, encoding="utf-8")

    checks = {
        "determinism": validate.check_determinism(m1, m2),
        "integrity": validate.check_integrity(m1),
        "isolated_consistency": validate.check_isolated_consistency(m1),
        "cross_boundary": validate.check_cross_boundary(m1).to_dict("records"),
        "freq_coverage": validate.check_freq_coverage(st).to_dict("records"),
    }
    for tid, be in bks.items():
        if be.backend == "hf":
            checks[f"hf_offsets::{tid}"] = validate.check_hf_offsets(be, st)
    json_dump(checks, OUT / "validation.json")
    print(f"  детерминированность: {checks['determinism']['identical']}, "
          f"декодирование: {checks['integrity']['decode_match_all']}")

    wide = metrics.build_wide(m1, st)
    wide = wide.merge(concepts[["concept_id", "matched_tech_concept_id",
                                "matched_tech_family_id"]],
                      on="concept_id", how="left")
    wide.to_csv(OUT / "analysis_wide.csv", index=False, encoding="utf-8")
    return wide, concepts, checks


def paired_frame(tech_wide: pd.DataFrame, ctrl_wide: pd.DataFrame, tid: str) -> pd.DataFrame:
    """Одна строка = сопоставленная пара «термин + бытовое понятие»."""
    t = tech_wide[tech_wide["tokenizer_id"] == tid].set_index("concept_id")
    c = ctrl_wide[ctrl_wide["tokenizer_id"] == tid].copy()
    c = c[c["matched_tech_concept_id"].isin(t.index)]
    rows = []
    for _, r in c.iterrows():
        tt = t.loc[r["matched_tech_concept_id"]]
        rows.append({
            "concept_id": r["matched_tech_concept_id"],   # ключ пары
            "family_id": tt["family_id"],                 # семья технического понятия
            "control_family_id": r["family_id"],          # семья контрольного понятия
            "control_concept_id": r["concept_id"],
            "stratum": tt["stratum"],
            "T_ru_tech": tt["T_ru_sg_iso0"], "T_en_tech": tt["T_en_sg_iso0"],
            "T_ru_ctrl": r["T_ru_sg_iso0"], "T_en_ctrl": r["T_en_sg_iso0"],
            "T_ru_tech_sp": tt["T_ru_sg_isosp"], "T_en_tech_sp": tt["T_en_sg_isosp"],
            "T_ru_ctrl_sp": r["T_ru_sg_isosp"], "T_en_ctrl_sp": r["T_en_sg_isosp"],
            "C_ru_tech": tt["C_ru_sg"], "C_en_tech": tt["C_en_sg"],
            "C_ru_ctrl": r["C_ru_sg"], "C_en_ctrl": r["C_en_sg"],
            "W_ru_tech": tt["W_ru_sg"], "W_en_tech": tt["W_en_sg"],
            "W_ru_ctrl": r["W_ru_sg"], "W_en_ctrl": r["W_en_sg"],
            "F_ru_tech": tt["F_ru_sg"], "F_en_tech": tt["F_en_sg"],
            "F_ru_ctrl": r["F_ru_sg"], "F_en_ctrl": r["F_en_sg"],
            "Fmiss_tech": tt["Fmiss_ru_sg"] + tt["Fmiss_en_sg"],
            "Fmiss_ctrl": r["Fmiss_ru_sg"] + r["Fmiss_en_sg"],
        })
    return add_clusters(pd.DataFrame(rows))


def add_clusters(pf: pd.DataFrame) -> pd.DataFrame:
    """Единица перевыбора: связная группа из семей обеих выборок.

    Сопоставление создаёт связи, которых нет ни в одной выборке по отдельности.
    Контрольные выражения тоже образуют семьи: пять пар имеют английскую
    вершину `room`, и эти пары привязаны к разным техническим семьям. Если
    перевыбирать только по семьям терминов, такие контрольные повторы
    считаются независимыми, хотя они не независимы.

    Поэтому строится граф: каждая сопоставленная пара соединяет свою
    техническую семью со своей контрольной. Кластером считается связная
    компонента этого графа, и перевыбираются кластеры целиком. Заранее
    неизвестно, станет интервал шире или уже: объединение уменьшает число
    независимых единиц, но меняет и состав повторов.
    """
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

    for _, r in pf.iterrows():
        union("T:" + str(r["family_id"]), "C:" + str(r["control_family_id"]))
    pf = pf.copy()
    pf["cluster_id"] = ["cl_" + find("T:" + str(f)) for f in pf["family_id"]]
    return pf


def stat_R(df: pd.DataFrame, suffix: str = "") -> float:
    lt = np.log(df[f"T_ru_tech{suffix}"] / df[f"T_en_tech{suffix}"])
    lc = np.log(df[f"T_ru_ctrl{suffix}"] / df[f"T_en_ctrl{suffix}"])
    return float(np.exp(lt.mean() - lc.mean()))


def stat_G_side(df: pd.DataFrame, side: str) -> float:
    return float(np.exp(np.log(df[f"T_ru_{side}"] / df[f"T_en_{side}"]).mean()))


def adjusted_R(df: pd.DataFrame, with_freq: bool, reps: int = config.BOOTSTRAP_REPS,
               seed: int = config.SEED) -> dict:
    """log(T_ru/T_en) = b0 + b_tech*[техническое] + b_C*log(C_ru/C_en) + b_W*(W_ru-W_en)
                        [+ b_F*(zF_ru - zF_en)]

    Обе выборки складываются в одну таблицу, признак «техническое» — двоичный.
    exp(b_tech) и есть R при выровненных длине, структуре и, во втором варианте,
    частотности. Это описательная поправка, а не причинная модель.
    """
    long = []
    for _, r in df.iterrows():
        for side in ("tech", "ctrl"):
            row = {
                # кластер — объединённая группа семей; если его нет (проверочные
                # данные), роль кластера играет семья технического понятия
                "pair_id": r["concept_id"],
                "family_id": r.get("cluster_id", r["family_id"]),
                "is_tech": 1.0 if side == "tech" else 0.0,
                "y": np.log(r[f"T_ru_{side}"] / r[f"T_en_{side}"]),
                "logC": np.log(r[f"C_ru_{side}"] / r[f"C_en_{side}"]),
                "dW": r[f"W_ru_{side}"] - r[f"W_en_{side}"],
                "F_ru": r[f"F_ru_{side}"], "F_en": r[f"F_en_{side}"],
                "fmiss": r["Fmiss_tech"] if side == "tech" else r["Fmiss_ctrl"],
            }
            long.append(row)
    L = pd.DataFrame(long)
    if with_freq:
        keep_pairs = L.groupby("pair_id")["fmiss"].max() == 0
        good = set(keep_pairs[keep_pairs].index)
        L = L[L["pair_id"].isin(good)]
        L = L[np.isfinite(L[["F_ru", "F_en"]]).all(axis=1)]

    def fit(sub: pd.DataFrame) -> float | None:
        cols = [np.ones(len(sub)), sub["is_tech"].to_numpy(float),
                sub["logC"].to_numpy(float), sub["dW"].to_numpy(float)]
        if with_freq:
            zr = (sub["F_ru"] - sub["F_ru"].mean()) / (sub["F_ru"].std(ddof=0) or 1)
            ze = (sub["F_en"] - sub["F_en"].mean()) / (sub["F_en"].std(ddof=0) or 1)
            cols.append((zr - ze).to_numpy(float))
        # Постоянный столбец несёт ноль сведений и делает систему вырожденной:
        # так бывает, например, когда в выборке все пары имеют одинаковое
        # число слов. Такой признак убирается, признак «техническое» — никогда.
        keep = [0, 1] + [i for i in range(2, len(cols)) if cols[i].std() > 0]
        X = np.column_stack([cols[i] for i in keep])
        y = sub["y"].to_numpy(float)
        if np.linalg.matrix_rank(X) < X.shape[1]:
            return None
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(beta[1])

    point = fit(L)
    rng = np.random.default_rng(seed)
    fams = L["family_id"].unique()
    idx_by_fam = [np.flatnonzero((L["family_id"] == f).to_numpy()) for f in fams]
    draws = []
    base = L.reset_index(drop=True)
    for _ in range(reps):
        pick = rng.integers(0, len(fams), size=len(fams))
        idx = np.concatenate([idx_by_fam[p] for p in pick])
        b = fit(base.take(idx))
        if b is not None:
            draws.append(b)
    draws = np.array(draws)
    enough = len(draws) >= 0.95 * reps
    lo, hi = (np.percentile(draws, [2.5, 97.5]) if len(draws) else (np.nan, np.nan))
    if not enough:
        # интервал по неполному набору повторов не публикуется (правило A1.2)
        lo = hi = np.nan
    return {
        "model": "log(T_ru/T_en) ~ is_tech + log(C_ru/C_en) + (W_ru-W_en)"
                 + (" + (zF_ru - zF_en)" if with_freq else ""),
        "n_pairs": int(L["pair_id"].nunique()),
        "n_families": int(len(fams)),
        "R_adjusted": float(np.exp(point)),
        "ci_low": float(np.exp(lo)), "ci_high": float(np.exp(hi)),
        "reps_used": int(len(draws)), "reps_requested": int(reps),
        "interval_published": bool(enough and point is not None), "seed": seed,
        "interpretation": "описательная поправка; exp(b) не является причинным эффектом",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="",
                    help="суффикс: прогон по дополнительному набору пар, а не по основному")
    args = ap.parse_args()
    if args.tag:
        globals()["CTRL_CONCEPTS"] = CTRL_CONCEPTS.with_name(
            f"{CTRL_CONCEPTS.stem}_{args.tag}{CTRL_CONCEPTS.suffix}")
        globals()["OUT"] = OUT.with_name(f"{OUT.name}_{args.tag}")

    if not CTRL_CONCEPTS.exists():
        raise SystemExit("Нет таблицы контроля: сначала scripts/control_select.py")
    ctrl_wide, ctrl_concepts, checks = measure_control()
    tech_wide = pd.read_csv(TECH_WIDE)

    print("[4/5] Расчёт R")
    results = {"threshold_practically_interesting": 0.10, "by_tokenizer": {}}
    for tid in config.TOKENIZERS:
        pf = paired_frame(tech_wide, ctrl_wide, tid)
        if pf.empty:
            continue
        CL = "cluster_id"
        entry = {
            "n_pairs": int(len(pf)),
            "n_families_tech": int(pf["family_id"].nunique()),
            "n_families_control": int(pf["control_family_id"].nunique()),
            "n_clusters_joint": int(pf[CL].nunique()),
            "cluster_unit": "связная группа семей обеих выборок (см. add_clusters)",
            "G_tech_matched": metrics.cluster_bootstrap(pf, lambda d: stat_G_side(d, "tech"), cluster=CL),
            "G_control_matched": metrics.cluster_bootstrap(pf, lambda d: stat_G_side(d, "ctrl"), cluster=CL),
            "R_raw": metrics.cluster_bootstrap(pf, stat_R, cluster=CL),
            "R_isolated_space": metrics.cluster_bootstrap(
                pf, lambda d: stat_R(d, "_sp"), cluster=CL),
            "R_raw_tech_families_only": metrics.cluster_bootstrap(pf, stat_R),
            "R_adjusted_length_structure": adjusted_R(pf, with_freq=False),
            "R_adjusted_plus_frequency": adjusted_R(pf, with_freq=True),
            "by_stratum": {},
        }
        for s, ds in pf.groupby("stratum"):
            entry["by_stratum"][s] = metrics.cluster_bootstrap(ds, stat_R, cluster=CL)
        # исходное R на том же подмножестве, что и частотная поправка:
        # иначе сдвиг смешивает смену состава пар и смену модели
        freq_ok = pf[(pf["Fmiss_tech"] == 0) & (pf["Fmiss_ctrl"] == 0)
                     & np.isfinite(pf[["F_ru_tech", "F_en_tech",
                                       "F_ru_ctrl", "F_en_ctrl"]]).all(axis=1)]
        entry["R_raw_on_frequency_subset"] = metrics.cluster_bootstrap(
            freq_ok, stat_R, cluster=CL)
        r = entry["R_raw"]
        entry["R_log_halfwidth"] = metrics.log_halfwidth(r)
        entry["R_meets_practical_threshold"] = bool(
            abs(np.log(r["point"])) >= np.log(1.10))
        entry["interval_excludes_1"] = bool(r["ci_low"] > 1 or r["ci_high"] < 1)
        results["by_tokenizer"][tid] = entry
        pf.to_csv(OUT / f"pairs_{tid}.csv", index=False, encoding="utf-8")

    json_dump(results, OUT / "estimates_R.json")

    rows = []
    for tid, e in results["by_tokenizer"].items():
        for name in ("G_tech_matched", "G_control_matched", "R_raw", "R_isolated_space",
                     "R_raw_tech_families_only", "R_raw_on_frequency_subset"):
            v = e[name]
            rows.append({"tokenizer_id": tid, "statistic": name, "point": v["point"],
                         "ci_low": v["ci_low"], "ci_high": v["ci_high"],
                         "n_pairs": e["n_pairs"],
                         "n_clusters": e["n_clusters_joint"]})
        for name in ("R_adjusted_length_structure", "R_adjusted_plus_frequency"):
            v = e[name]
            rows.append({"tokenizer_id": tid, "statistic": name, "point": v["R_adjusted"],
                         "ci_low": v["ci_low"], "ci_high": v["ci_high"],
                         "n_pairs": v["n_pairs"], "n_clusters": v["n_families"]})
    pd.DataFrame(rows).to_csv(OUT / "estimates_R_summary.csv", index=False, encoding="utf-8")

    print("[5/5] Готово")
    for tid, e in results["by_tokenizer"].items():
        r = e["R_raw"]
        old = e["R_raw_tech_families_only"]
        print(f"  {tid}: G_тех {e['G_tech_matched']['point']:.3f} / "
              f"G_контроль {e['G_control_matched']['point']:.3f} -> "
              f"R = {r['point']:.3f} [{r['ci_low']:.3f}; {r['ci_high']:.3f}] "
              f"({e['n_pairs']} пар, {e['n_clusters_joint']} кластеров; "
              f"только по семьям терминов было [{old['ci_low']:.3f}; {old['ci_high']:.3f}])")
    print(f"\n{(OUT / 'estimates_R.json').relative_to(ROOT)}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
