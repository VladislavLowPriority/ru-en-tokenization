# -*- coding: utf-8 -*-
"""Проверка чувствительности к исключению источников Festo.

**1. Влияние пилотных понятий.** Протокол (§10) требовал, чтобы понятия,
на которых отлаживался метод, не входили в основной анализ. Смысл требования
статистический: если решения о дизайне подгонялись под данные, а потом по тем
же данным считается результат, заявленная точность окажется завышенной.

Текущая реализация исключает все 21 понятие из источников Festo, а не
восстановленный хронологический список пилота. Близость оценок не доказывает
эквивалентность выборок и не отменяет отклонение от исходного дизайна.

Пересчёт при замене синонимов в этом скрипте НЕ реализован; он относится к
последующим пунктам плана. Ранее описание ошибочно обещало такую проверку.

Запуск:
    .venv\\Scripts\\python.exe scripts\\robustness.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tok_pilot import backends, config, measure, metrics, stimuli  # noqa: E402

OUT = ROOT / "results" / "real" / "robustness.json"


def analyse(concepts: pd.DataFrame, bks: dict) -> pd.DataFrame:
    st = stimuli.add_frequencies(stimuli.build_stimuli(concepts))
    m = measure.measure(st, bks, run_id="robust")
    return metrics.build_wide(m, st)


def estimates(wide: pd.DataFrame) -> dict:
    out = {}
    for tid in config.TOKENIZERS:
        d = wide[wide.tokenizer_id == tid]
        if d.empty:
            continue
        g = metrics.cluster_bootstrap(d, metrics.stat_G)
        out[tid] = {
            "G": g["point"], "ci": [g["ci_low"], g["ci_high"]],
            "n_concepts": g["n_concepts"], "n_families": g["n_families"],
            "M": metrics.stat_M(d),
        }
    return out


def main() -> None:
    src = config.DATA_REAL / "concepts_pilot.csv"
    concepts = stimuli.load_concepts(src)
    concepts = concepts[concepts["final_status"] == "accepted"].copy()

    # Источниковая проверка: Festo. Это не подтверждённый перечень пилота.
    is_pilot = concepts["source_en"].str.startswith("Festo")
    pilot = concepts[is_pilot]
    rest = concepts[~is_pilot]

    print(f"всего понятий: {len(concepts)}")
    print(f"   из источников Festo:     {len(pilot)}")
    print(f"   без Festo:               {len(rest)}")
    if rest.empty or len(rest) < 10:
        print("Слишком мало непилотных понятий для сравнения.")
        return

    bks = backends.load_all()
    full = estimates(analyse(concepts, bks))
    without = estimates(analyse(rest, bks))
    only_pilot = estimates(analyse(pilot, bks)) if len(pilot) >= 10 else {}

    print("\nОсновная величина G при разном составе выборки:")
    header = f"{'токенизатор':<22} {'вся выборка':<26} {'без Festo':<26} {'разница'}"
    print(header)
    print("-" * len(header))
    rows = {}
    for tid in full:
        a, b = full[tid], without.get(tid)
        if not b:
            continue
        diff = b["G"] - a["G"]
        overlap = not (a["ci"][1] < b["ci"][0] or b["ci"][1] < a["ci"][0])
        rows[tid] = {"full": a, "without_pilot": b, "delta_G": diff,
                     "intervals_overlap": overlap}
        print(f"{tid:<22} {a['G']:.3f} [{a['ci'][0]:.3f}; {a['ci'][1]:.3f}]   "
              f"{b['G']:.3f} [{b['ci'][0]:.3f}; {b['ci'][1]:.3f}]   {diff:+.3f}"
              f"{'' if overlap else '   ИНТЕРВАЛЫ НЕ ПЕРЕСЕКАЮТСЯ'}")

    payload = {
        "n_total": int(len(concepts)),
        "n_pilot": int(len(pilot)),
        "n_without_pilot": int(len(rest)),
        "selection_rule": "source_en starts with Festo; legacy pilot field names do not establish chronological pilot membership",
        "full_sample": full,
        "without_pilot": without,
        "pilot_only": only_pilot,
        "comparison": rows,
        "note": ("Исключаются все источники Festo по префиксу source_en, а не "
                 "хронологически восстановленный список пилота. Близость оценок "
                 "и пересечение интервалов не доказывают эквивалентность и не "
                 "отменяют требование разделить пилот и основной анализ. "
                 "Замены синонимов здесь не рассчитываются."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nсохранено: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
