# -*- coding: utf-8 -*-
"""Проверки расчёта R и отбора контроля.

Смысл тех же проверок, что и в `test_counting.py`: убедиться, что показатель
считает то, что написано в §14, а не то, что случайно получилось из кода.
Здесь проверяются формулы и правила отбора, а не сами данные.

Запуск:
    .venv\\Scripts\\python.exe scripts\\test_control.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rc = load("run_control")
cs = load("control_select")
ok = 0


def check(cond: bool, what: str) -> None:
    global ok
    if not cond:
        raise AssertionError(what)
    ok += 1
    print(f"  ок: {what}")


print("1. Формула R")
df = pd.DataFrame({
    "concept_id": ["a", "b"], "family_id": ["f1", "f2"],
    "T_ru_tech": [4, 6], "T_en_tech": [2, 2],
    "T_ru_ctrl": [2, 3], "T_en_ctrl": [2, 1],
})
# tech: log2, log3 ; ctrl: log1, log3  -> R = exp(mean(log2,log3) - mean(log1,log3))
expected = np.exp((np.log(2) + np.log(3)) / 2 - (np.log(1) + np.log(3)) / 2)
check(abs(rc.stat_R(df) - expected) < 1e-12, "R равен отношению геометрических средних")
check(abs(rc.stat_G_side(df, "tech") - np.exp((np.log(2) + np.log(3)) / 2)) < 1e-12,
      "G технической стороны считается по тем же парам")

print("2. R не зависит от порядка строк")
check(abs(rc.stat_R(df.iloc[::-1]) - rc.stat_R(df)) < 1e-12, "перестановка строк ничего не меняет")

print("3. Одинаковые выборки дают R = 1")
same = df.copy()
same["T_ru_ctrl"], same["T_en_ctrl"] = same["T_ru_tech"], same["T_en_tech"]
check(abs(rc.stat_R(same) - 1.0) < 1e-12, "при совпадении выборок R равен единице")

print("4. Регрессия восстанавливает заложенный коэффициент")
rng = np.random.default_rng(7)
n = 60
logC = rng.normal(0, 0.3, n)
rows = []
for i in range(n):
    for side, is_tech in (("tech", 1.0), ("ctrl", 0.0)):
        dW = ((1 + i % 2) - 1) if side == "tech" else (1 - (1 + i % 3))
        y = 0.20 * is_tech + 0.5 * logC[i] + 0.03 * dW
        # подбираем длины и токены так, чтобы log(T_ru/T_en) был ровно y
        t_en, c_en = 10.0, 10.0
        rows.append((i, side, np.exp(y) * t_en, t_en, np.exp(logC[i]) * c_en, c_en))
pair = {}
for i, side, t_ru, t_en, c_ru, c_en in rows:
    pair.setdefault(i, {})
    pair[i][f"T_ru_{side}"] = t_ru
    pair[i][f"T_en_{side}"] = t_en
    pair[i][f"C_ru_{side}"] = c_ru
    pair[i][f"C_en_{side}"] = c_en
frame = pd.DataFrame([{
    "concept_id": f"p{i}", "family_id": f"f{i}", "cluster_id": f"cl{i}",
    "W_ru_tech": 1 + i % 2, "W_en_tech": 1, "W_ru_ctrl": 1, "W_en_ctrl": 1 + i % 3,
    "F_ru_tech": 3.0, "F_en_tech": 3.0, "F_ru_ctrl": 3.0, "F_en_ctrl": 3.0,
    "Fmiss_tech": 0, "Fmiss_ctrl": 0, **v} for i, v in pair.items()])
fit = rc.adjusted_R(frame, with_freq=False, reps=50)
check(abs(fit["R_adjusted"] - np.exp(0.20)) < 1e-6,
      "коэффициент при признаке «техническое» восстановлен точно")

print("5. Правила отбора")
check(cs.stratum_of(1, 1) == "A" and cs.stratum_of(2, 2) == "B"
      and cs.stratum_of(1, 2) == "C" and cs.stratum_of(2, 1) == "C",
      "структурные группы определяются по числу слов в обоих языках")
check(cs.chars("линейный привод") == 14, "длина считается без пробелов")
check(cs.family_key({"RU": "стиральная машина", "EN": "washing machine"})
      == ("машина", "machine"), "семья контроля берётся по последнему слову")

print("6. Контроль измеряется только в единственном числе")
from tok_pilot import stimuli  # noqa: E402

c = pd.DataFrame([{
    "concept_id": "C001", "family_id": "cfam", "split": "control", "stratum": "A",
    "domain": "быт", "definition": "",
    "term_ru_sg": "стол", "term_ru_pl": "стол",
    "term_en_sg": "table", "term_en_pl": "table",
    "head_ru": "стол", "head_en": "table",
    "source_ru": "s", "source_en": "s", "source_entry": "e", "edition": "",
    "access_date": "2026-09-19", "review_technical": "not_applicable",
    "review_language": "not_recorded_separately", "final_status": "accepted",
    "decision_note": "", "alternatives": "", "selection_order": "1",
    "term_pattern_ru": "", "term_pattern_en": "",
    "changed_components_ru": "0", "changed_components_en": "0",
    "inflection_note_ru": "", "inflection_note_en": "",
}])
st = stimuli.build_stimuli(c, numbers=("sg",))
check(set(st["number"]) == {"sg"}, "строки множественного числа не создаются")
check(len(st) == 2 * len(stimuli.config.ALL_TEMPLATES) + 2 * 2,
      "на понятие приходятся два языка x (2 изолированных режима + шаблоны)")

print("7. Кластеры перевыбора учитывают обе выборки")
pf = pd.DataFrame({
    "family_id": ["t1", "t2", "t3"],
    "control_family_id": ["c1", "c1", "c2"],
})
cl = rc.add_clusters(pf)["cluster_id"].tolist()
check(cl[0] == cl[1], "две технические семьи с общей семьёй контроля попадают в один кластер")
check(cl[2] != cl[0], "несвязанная пара остаётся отдельным кластером")

pf2 = pd.DataFrame({"family_id": ["t1", "t2"], "control_family_id": ["c1", "c2"]})
check(rc.add_clusters(pf2)["cluster_id"].nunique() == 2,
      "без общих семей контроля число кластеров равно числу семей терминов")

print("8. Совпадение с сохранённым результатом")
pairs_file = ROOT / "results" / "control" / "pairs_o200k_base.csv"
if pairs_file.exists():
    p = pd.read_csv(pairs_file)
    manual = np.exp(np.log(p["T_ru_tech"] / p["T_en_tech"]).mean()
                    - np.log(p["T_ru_ctrl"] / p["T_en_ctrl"]).mean())
    check(abs(manual - rc.stat_R(p)) < 1e-12, "R на сохранённых парах воспроизводится вручную")
else:
    print("  пропуск: прогон контроля ещё не выполнялся")

print(f"\nВсе проверки пройдены: {ok}")
