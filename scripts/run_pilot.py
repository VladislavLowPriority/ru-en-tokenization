# -*- coding: utf-8 -*-
"""Полный прогон пилота: строки -> токенизация -> проверки -> оценки.

Запуск:
    python scripts/run_pilot.py --split demo
    python scripts/run_pilot.py --split real

Все промежуточные таблицы сохраняются, чтобы результат можно было проверить
построчно, а не поверить итоговому среднему.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tok_pilot import backends, config, manifest, measure, metrics, stimuli, validate  # noqa: E402


def json_dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")


def main(split: str) -> None:
    src = (config.DATA_REAL / "concepts_pilot.csv" if split == "real"
           else config.DATA_DEMO / "concepts_demo.csv")
    out = config.RESULTS / split
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/8] Понятия: {src}")
    concepts = stimuli.load_concepts(src)
    if split == "real":
        # В анализ идут только окончательно принятые пары (§10).
        concepts = concepts[concepts["final_status"] == "accepted"].copy()
    if concepts.empty:
        print("  ПУСТО. Реальные термины ещё не внесены — см. data/real/FILL_INSTRUCTIONS.md")
        json_dump({"status": "no_data", "source": str(src)}, out / "status.json")
        return
    print(f"  понятий: {len(concepts)}, семей: {concepts['family_id'].nunique()}")

    print("[2/8] Входные строки и частотность")
    st = stimuli.build_stimuli(concepts)
    st = stimuli.add_frequencies(st)
    st.to_csv(out / "stimuli.csv", index=False, encoding="utf-8")
    print(f"  строк: {len(st)}")

    print("[3/8] Загрузка токенизаторов")
    bks = backends.load_all()
    man = manifest.build(bks, [src], st, split)
    manifest.save(man, out / "manifest.json")
    for tid, m in man["tokenizers"].items():
        print(f"  {tid}: vocab={m['n_vocab']}")

    print("[4/8] Измерение (прогон 1)")
    m1 = measure.measure(st, bks, run_id="run1")
    m1.to_csv(out / "measurements.csv", index=False, encoding="utf-8")

    print("[5/8] Повторный прогон для проверки детерминированности")
    m2 = measure.measure(st, bks, run_id="run2")

    print("[6/8] Технические проверки протокола")
    ctx_pairs = validate.context_token_comparisons(m1, st)
    ctx_pairs.to_csv(out / "context_token_comparisons.csv", index=False, encoding="utf-8")
    checks = {
        "determinism": validate.check_determinism(m1, m2),
        "integrity": validate.check_integrity(m1),
        "isolated_consistency": validate.check_isolated_consistency(m1),
        "sample_frame": validate.check_sample_frame(concepts),
        "normalizer_effect": validate.check_normalizer_effect(bks, st).to_dict("records"),
        "freq_coverage": validate.check_freq_coverage(st).to_dict("records"),
        "cross_boundary": validate.check_cross_boundary(m1).to_dict("records"),
        "context_token_identity": validate.summarize_context_comparisons(ctx_pairs),
    }
    for tid, be in bks.items():
        if be.backend == "hf":
            checks[f"hf_offsets::{tid}"] = validate.check_hf_offsets(be, st)
    json_dump(checks, out / "validation.json")
    print(f"  детерминированность: {checks['determinism']['identical']}, "
          f"полное декодирование: {checks['integrity']['decode_match_all']}, "
          f"спецтокенов: {checks['integrity']['special_tokens_total']}")

    print("[7/8] Ручной просмотр разбиений")
    review_ids = (concepts.groupby("stratum")["concept_id"].head(2).tolist())
    frames = []
    for tid, be in bks.items():
        t = validate.manual_review_table(m1, st, be, review_ids)
        t.insert(0, "tokenizer_id", tid)
        frames.append(t)
    pd.concat(frames).to_csv(out / "manual_review.csv", index=False, encoding="utf-8")

    print("[8/8] Оценки и интервалы")
    wide = metrics.build_wide(m1, st)
    if "pl_status" in concepts.columns:
        wide = wide.merge(concepts[["concept_id", "pl_status"]], on="concept_id", how="left")
    wide.to_csv(out / "analysis_wide.csv", index=False, encoding="utf-8")

    estimates = {}
    for tid in config.TOKENIZERS:
        d = wide[wide["tokenizer_id"] == tid]
        if d.empty:
            continue
        # Формы мн. ч. со статусом, отличным от confirmed, не выбрасываются молча:
        # величины, зависящие от PL, считаются и со всеми понятиями, и без них.
        d_pl = d[d["pl_status"] == "confirmed"] if "pl_status" in d.columns else d
        e = {
            "H1_G": metrics.cluster_bootstrap(d, metrics.stat_G),
            "H1_G_isolated_space": metrics.cluster_bootstrap(d, metrics.stat_G_isosp),
            "H1_G_mode_gap": metrics.cluster_bootstrap(d, metrics.stat_G_mode_gap),
            "H1_mean_token_diff": metrics.cluster_bootstrap(d, metrics.stat_mean_diff_sg),
            "H1_ratio_of_sums": metrics.cluster_bootstrap(d, metrics.stat_ratio_of_sums),
            "H2_M": metrics.cluster_bootstrap(d, metrics.stat_M),
            "H3_context_shift_both": metrics.cluster_bootstrap(d, metrics.stat_H3),
            "fertility_ratio": metrics.cluster_bootstrap(d, metrics.stat_fertility_ratio),
            "space_effect_ru": metrics.cluster_bootstrap(
                d, lambda x: metrics.stat_space_effect(x, "ru")),
            "space_effect_en": metrics.cluster_bootstrap(
                d, lambda x: metrics.stat_space_effect(x, "en")),
            "sensitivity_regression": metrics.sensitivity_regression(d),
            "sensitivity_regression_available_components": metrics.sensitivity_regression(d, coverage="available"),
        }
        if len(d_pl) < len(d):
            e["H2_M_pl_confirmed_only"] = metrics.cluster_bootstrap(d_pl, metrics.stat_M)
            e["H3_context_shift_pl_confirmed_only"] = metrics.cluster_bootstrap(
                d_pl, metrics.stat_H3)
        for tpl in config.ALL_TEMPLATES:
            e[f"H3_context_shift_{tpl}"] = metrics.cluster_bootstrap(
                d, metrics.make_stat_H3([tpl]))
            e[f"sentence_ratio_{tpl}_sg"] = metrics.cluster_bootstrap(
                d, lambda x, t=tpl: metrics.stat_sentence_ratio(x, t, "sg"))
        for stratum, ds in d.groupby("stratum"):
            e[f"H1_G_stratum_{stratum}"] = metrics.cluster_bootstrap(ds, metrics.stat_G)
        e["H1_G_log_halfwidth"] = metrics.log_halfwidth(e["H1_G"])
        e["H1_G_meets_precision_target"] = bool(
            e["H1_G_log_halfwidth"] <= config.PRECISION_TARGET_LOG_HALFWIDTH)
        if len(d) < 90:
            e["precision_projection_N90"] = metrics.project_precision(
                e["H1_G"], target_families=config.FAMILY_TARGET_MAIN, target_concepts=90)
            e["precision_projection_N120"] = metrics.project_precision(
                e["H1_G"], target_families=40, target_concepts=120)
        estimates[tid] = e
    json_dump({"split": split, "primary_tokenizer": config.PRIMARY_TOKENIZER,
               "estimates": estimates}, out / "estimates.json")

    rows = []
    for tid, e in estimates.items():
        for name, v in e.items():
            if isinstance(v, dict) and "point" in v:
                rows.append({"tokenizer_id": tid, "statistic": name,
                             "point": v["point"], "ci_low": v["ci_low"],
                             "ci_high": v["ci_high"], "n_concepts": v["n_concepts"],
                             "n_families": v["n_families"]})
    pd.DataFrame(rows).to_csv(out / "estimates_summary.csv", index=False, encoding="utf-8")
    regression_rows = []
    for tid, e in estimates.items():
        for key in ("sensitivity_regression", "sensitivity_regression_available_components"):
            fit = e[key]
            for coefficient, value in fit.get("coefficients", {}).items():
                regression_rows.append({"tokenizer_id": tid, "analysis": key,
                                        "coefficient": coefficient, **value,
                                        "n_used": fit["n_used"], "n_families": fit["n_families"],
                                        "n_partial_used": fit["n_partial_used"],
                                        "reps_used": fit["reps_used"]})
    pd.DataFrame(regression_rows).to_csv(out / "regression_summary.csv", index=False, encoding="utf-8")
    print(f"Готово. Результаты в {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["demo", "real"], required=True)
    main(ap.parse_args().split)
