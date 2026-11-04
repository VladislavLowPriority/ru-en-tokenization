# -*- coding: utf-8 -*-
"""Чувствительность результатов к спорным формам множественного числа.

Зачем. Для конструкций «существительное + зависимое в родительном падеже»
русский допускает два прочтения: «экраны кабеля» (зависимое остаётся в ед. ч.,
мыслится тип детали) и «экраны кабелей» (дистрибутивное согласование).
Выбор варианта — лингвистическое решение, и его нельзя принимать по тому,
какой вариант удобнее для результата.

Что делает скрипт: пересчитывает величины, зависящие от множественного числа,
при каждом варианте и показывает фактическую разницу. Если разница мала,
выбор варианта перестаёт быть угрозой выводам — но это надо ИЗМЕРИТЬ, а не
предположить.

Запуск:
    .venv\\Scripts\\python.exe scripts\\pl_sensitivity.py
"""
from __future__ import annotations

import json
import re
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tok_pilot import backends, config, measure, metrics, stimuli  # noqa: E402

OUT = ROOT / "results" / "real" / "pl_sensitivity.json"
QUOTED = re.compile(r"«([^»]+)»")


def find_alternatives(concepts: pd.DataFrame) -> dict[str, str]:
    """Достаёт альтернативную форму мн. ч. из комментария к спорным парам.

    Альтернатива берётся из самого реестра, из текста в кавычках, а не
    придумывается здесь: скрипт лишь выбирает вариант с тем же числом слов,
    который не совпадает ни с записанной формой мн. ч., ни с формой ед. ч.
    """
    alts = {}
    for _, r in concepts.iterrows():
        if r.get("pl_status", "") == "confirmed":
            continue
        note = str(r.get("inflection_note_ru", ""))
        pl, sg = r["term_ru_pl"].strip(), r["term_ru_sg"].strip()
        for cand in QUOTED.findall(note):
            cand = cand.strip().lower()
            if cand in (pl, sg):
                continue
            if len(cand.split(" ")) != len(pl.split(" ")):
                continue
            alts[r["concept_id"]] = cand
            break
    return alts


def run_variant(concepts: pd.DataFrame, bks: dict) -> pd.DataFrame:
    st = stimuli.add_frequencies(stimuli.build_stimuli(concepts))
    m = measure.measure(st, bks, run_id="variant")
    return metrics.build_wide(m, st)


def main() -> None:
    src = config.DATA_REAL / "concepts_pilot.csv"
    concepts = stimuli.load_concepts(src)
    concepts = concepts[concepts["final_status"] == "accepted"].copy()

    alts = find_alternatives(concepts)
    if not alts:
        print("Спорных форм мн. ч. с указанной альтернативой не найдено.")
        return

    print("Спорные формы и их альтернативы (обе взяты из реестра):")
    for cid, alt in alts.items():
        cur = concepts.loc[concepts.concept_id == cid, "term_ru_pl"].iloc[0]
        print(f"   {cid}: «{cur}»  либо  «{alt}»")

    bks = backends.load_all()

    # Базовый вариант — формы, записанные в реестре.
    base_wide = run_variant(concepts, bks)

    # Все сочетания вариантов: спорных пар мало, перебор полный.
    results = {}
    combos = list(product(*[[(cid, "base"), (cid, "alt")] for cid in alts]))
    print(f"\nПересчёт для {len(combos)} сочетаний вариантов...")
    for combo in combos:
        label = ",".join(f"{cid}={which}" for cid, which in combo)
        mod = concepts.copy()
        for cid, which in combo:
            if which == "alt":
                mod.loc[mod.concept_id == cid, "term_ru_pl"] = alts[cid]
        wide = run_variant(mod, bks) if label != ",".join(
            f"{cid}=base" for cid in alts) else base_wide
        per_tok = {}
        for tid in config.TOKENIZERS:
            d = wide[wide.tokenizer_id == tid]
            if d.empty:
                continue
            # то же геометрическое среднее парных отношений, что и в H1,
            # но на формах множественного числа
            ratio = np.log(d["T_ru_pl_iso0"] / d["T_en_pl_iso0"])
            per_tok[tid] = {
                "H2_M": metrics.stat_M(d),
                "G_pl_iso0": float(np.exp(ratio.mean())),
            }
        results[label] = per_tok

    # Разброс по вариантам
    print("\nВлияние выбора варианта на величины, зависящие от мн. ч.:")
    summary = {}
    for tid in config.TOKENIZERS:
        vals_m = [r[tid]["H2_M"] for r in results.values() if tid in r]
        if not vals_m:
            continue
        spread = max(vals_m) - min(vals_m)
        summary[tid] = {
            "H2_M_min": min(vals_m), "H2_M_max": max(vals_m),
            "H2_M_spread_tokens": spread,
            "H2_M_base": results[",".join(f"{cid}=base" for cid in alts)][tid]["H2_M"],
        }
        print(f"   {tid:<22} M от {min(vals_m):+.4f} до {max(vals_m):+.4f} "
              f"| разброс {spread:.4f} токена")

    payload = {
        "n_disputed_concepts": len(alts),
        "alternatives": alts,
        "n_combinations": len(combos),
        "per_combination": results,
        "summary": summary,
        "note": ("Разброс показывает, насколько выбор спорной формы мн. ч. меняет "
                 "величину M. Он НЕ оправдывает отказ от лингвистического решения: "
                 "единое правило всё равно должно быть зафиксировано до основного прогона."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
