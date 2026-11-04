"""Технические проверки пилота (§10).

Это проверки МЕТОДА, а не поиск эффекта. Они должны выполняться до того, как
кто-либо посмотрит на величину G: иначе ошибка реализации будет истолкована
как языковое различие.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config
from .measure import term_byte_span


def check_determinism(m1: pd.DataFrame, m2: pd.DataFrame) -> dict:
    """Один и тот же вход должен давать те же ID при повторном запуске (§6 п.6)."""
    cols = ["stimulus_id", "tokenizer_id", "token_ids", "n_tokens_input", "n_tokens_target"]
    a = m1[cols].sort_values(cols[:2]).reset_index(drop=True)
    b = m2[cols].sort_values(cols[:2]).reset_index(drop=True)
    same = a.equals(b)
    diff = [] if same else a.compare(b).head(10).to_dict()
    return {"identical": bool(same), "n_rows": int(len(a)), "first_differences": diff}


def check_integrity(meas: pd.DataFrame) -> dict:
    """Полное декодирование совпадает, спецтокенов нет, ошибок нет."""
    return {
        "decode_match_all": bool(meas["decode_match"].all()),
        "decode_mismatch_rows": meas.loc[~meas["decode_match"], "stimulus_id"].tolist()[:20],
        "special_tokens_total": int(meas["special_token_count"].sum()),
        "errors": meas.loc[meas["error_code"].fillna("") != "",
                           ["stimulus_id", "tokenizer_id", "error_code"]].to_dict("records"),
        "zero_target_rows": meas.loc[meas["n_tokens_target"] == 0, "stimulus_id"].tolist()[:20],
    }


def check_cross_boundary(meas: pd.DataFrame) -> pd.DataFrame:
    """Доля контекстных измерений, где токен термина захватывает соседний текст.

    Порог §10: выше 5% — S плохо отделяется, и H3 остаётся только описательной.
    """
    ctx = meas[meas["mode"].str.startswith("sentence_")]
    g = ctx.groupby(["tokenizer_id", "lang"], as_index=False).agg(
        n=("cross_boundary", "size"), n_cross=("cross_boundary", "sum"))
    g["share"] = g["n_cross"] / g["n"]
    g["over_threshold"] = g["share"] > config.CROSS_BOUNDARY_ALERT
    return g


def check_normalizer_effect(backends: dict, stimuli: pd.DataFrame) -> pd.DataFrame:
    """Меняет ли встроенный normalizer наши строки (§6 п.7).

    Проверяется не декларация в конфиге, а фактическое совпадение
    склеенных байтов токенов с байтами исходной строки.
    """
    rows = []
    texts = stimuli["input_text"].unique()
    for tid, be in backends.items():
        n_changed = 0
        example = ""
        for t in texts:
            enc = be.encode(t)
            if b"".join(enc.token_bytes) != t.encode("utf-8"):
                n_changed += 1
                if not example:
                    example = t
        rows.append({"tokenizer_id": tid, "n_inputs": len(texts),
                     "n_text_changed": n_changed, "example": example})
    return pd.DataFrame(rows)


def check_hf_offsets(backend, stimuli: pd.DataFrame, limit: int | None = None) -> dict:
    """Сверка символьных offsets HF с независимым байтовым расчётом.

    Протокол требует проверить отображение offsets на исходную строку и не
    доверять им вслепую: у byte-level BPE один символ может делиться между
    токенами, и тогда символьные интервалы повторяются или схлопываются.
    """
    mismatches, repeated, n = [], 0, 0
    texts = stimuli[["input_text", "target_start", "target_end"]].drop_duplicates()
    if limit:
        texts = texts.head(limit)
    for _, r in texts.iterrows():
        text = r["input_text"]
        enc = backend.encode(text)
        n += 1
        if enc.hf_char_offsets is None:
            continue
        # 1) байтовый интервал, пересчитанный из символьного offset HF,
        #    должен совпасть с байтовым интервалом, накопленным по токенам
        for (cs, ce), (bs_tok, be_tok) in zip(enc.hf_char_offsets, enc.byte_spans):
            exp_bs = len(text[:cs].encode("utf-8"))
            exp_be = len(text[:ce].encode("utf-8"))
            if (exp_bs, exp_be) != (bs_tok, be_tok):
                mismatches.append({"text": text, "char": [cs, ce],
                                   "byte_from_char": [exp_bs, exp_be],
                                   "byte_from_tokens": [bs_tok, be_tok]})
        # 2) один и тот же символьный интервал у разных токенов
        seen = set()
        for o in enc.hf_char_offsets:
            if o in seen:
                repeated += 1
            seen.add(o)
    return {"n_inputs_checked": n, "n_offset_mismatches": len(mismatches),
            "n_repeated_char_intervals": repeated,
            "examples": mismatches[:5]}


def check_freq_coverage(stimuli: pd.DataFrame) -> pd.DataFrame:
    """Доля принятых пар, где частотные данные есть для всех компонентов SG."""
    sg = stimuli[(stimuli["number"] == "sg") & (stimuli["mode"] == "isolated_0")]
    rows = []
    for lang, sub in sg.groupby("lang"):
        full = (sub["freq_missing"] == 0).mean()
        rows.append({"lang": lang, "n_terms": len(sub),
                     "share_full_coverage": float(full),
                     "meets_threshold": bool(full >= config.FREQ_COVERAGE_MIN)})
    return pd.DataFrame(rows)


def check_sample_frame(concepts: pd.DataFrame) -> dict:
    """Состав выборки: страты, семьи, потолок понятий на семью (§3)."""
    fam = concepts.groupby("family_id").size()
    return {
        "n_concepts": int(len(concepts)),
        "n_families": int(fam.size),
        "max_concepts_per_family": int(fam.max()) if fam.size else 0,
        "family_cap_respected": bool(fam.max() <= config.MAX_CONCEPTS_PER_FAMILY)
        if fam.size else False,
        "by_stratum": concepts["stratum"].value_counts().to_dict(),
        "final_status": concepts["final_status"].value_counts().to_dict(),
    }


def manual_review_table(meas: pd.DataFrame, stimuli: pd.DataFrame,
                        backend, concept_ids: list[str]) -> pd.DataFrame:
    """Ручной просмотр всех режимов (§10 «Пробел и смещения»).

    Печатаются реальные ID токенов и их байтовые границы, а не декоративная
    картинка предполагаемого разбиения.
    """
    sub = stimuli[stimuli["concept_id"].isin(concept_ids)]
    rows = []
    for _, s in sub.iterrows():
        enc = backend.encode(s["input_text"])
        bs, be = term_byte_span(s["input_text"], int(s["target_start"]), int(s["target_end"]))
        pieces = []
        for i, ((ts, te), tb) in enumerate(zip(enc.byte_spans, enc.token_bytes)):
            inside = max(ts, bs) < min(te, be)
            mark = "*" if inside else " "
            pieces.append(f"{mark}{enc.ids[i]}:{tb!r}")
        rows.append({
            "concept_id": s["concept_id"], "lang": s["lang"], "number": s["number"],
            "mode": s["mode"], "input_text": s["input_text"],
            "term_bytes": f"[{bs},{be})",
            "T": len(enc.ids),
            "tokens": " | ".join(pieces),
        })
    return pd.DataFrame(rows)


def check_isolated_consistency(meas: pd.DataFrame) -> dict:
    """В режиме isolated_0 вся строка и есть термин: S обязан равняться T.

    Расхождение означало бы ошибку разметки границ или счёта, а не свойство языка.
    """
    iso = meas[meas["mode"] == "isolated_0"]
    bad = iso[iso["n_tokens_input"] != iso["n_tokens_target"]]
    # В isolated_space отдельный токен-пробел (если он есть) в S не входит,
    # поэтому S <= T; обратное было бы ошибкой.
    sp = meas[meas["mode"] == "isolated_space"]
    bad_sp = sp[sp["n_tokens_target"] > sp["n_tokens_input"]]
    return {
        "isolated_0_S_equals_T": bool(bad.empty),
        "isolated_0_violations": bad["stimulus_id"].tolist()[:20],
        "isolated_space_S_le_T": bool(bad_sp.empty),
        "isolated_space_violations": bad_sp["stimulus_id"].tolist()[:20],
    }


def context_token_comparisons(meas: pd.DataFrame, stimuli: pd.DataFrame) -> pd.DataFrame:
    """Сравнить каждый контекст с isolated_space по ID и границам, а не средним.

    Байтовые границы приводятся к началу термина. Сохраняются как полные
    интервалы пересекающих токенов (включая возможный ведущий пробел), так и
    интервалы, обрезанные по термину. Это различает смену токена и смену
    сегментации внутри самого термина. Пропуски и дубликаты не считаются успехом.
    """
    keys = ["concept_id", "lang", "number", "tokenizer_id"]
    selected = meas[meas["mode"].isin(["isolated_space"] + [f"sentence_{t}" for t in config.ALL_TEMPLATES])]
    if selected["error_code"].fillna("").ne("").any():
        raise ValueError("Context comparison contains failed measurements")
    if selected.duplicated(keys + ["mode"]).any():
        raise ValueError("Duplicate context/reference measurements")
    selected = selected.merge(stimuli[["stimulus_id", "target_text"]],
                              on="stimulus_id", how="left", validate="many_to_one")
    if selected["target_text"].isna().any():
        raise ValueError("Missing target text for context comparison")
    refs = {tuple(r[k] for k in keys): r for r in selected.to_dict("records") if r["mode"] == "isolated_space"}
    ctx = selected[selected["mode"].str.startswith("sentence_")]
    expected = {key + (f"sentence_{t}",) for key in refs for t in config.ALL_TEMPLATES}
    actual = {tuple(r[k] for k in keys) + (r["mode"],) for r in ctx.to_dict("records")}
    if not refs or actual != expected:
        raise ValueError("Incomplete context/reference pairing")

    def signature(r):
        hits = json.loads(r["target_tokens"])
        ids = json.loads(r["token_ids"])
        spans = json.loads(r["token_spans"])
        start, end = int(r["term_byte_start"]), int(r["term_byte_end"])
        if len(hits) != r["n_tokens_target"] or end - start != len(r["target_text"].encode("utf-8")):
            raise ValueError("Inconsistent target count or byte length")
        for hit in hits:
            if hit["id"] != ids[hit["i"]] or hit["span"] != spans[hit["i"]]:
                raise ValueError("Target tokens disagree with full encoding")
        token_ids = [h["id"] for h in hits]
        relative = [[h["span"][0] - start, h["span"][1] - start] for h in hits]
        clipped = [[max(0, a), min(end - start, b)] for a, b in relative]
        return token_ids, relative, clipped

    rows = []
    for r in ctx.to_dict("records"):
        ref = refs[tuple(r[k] for k in keys)]
        if r["target_text"] != ref["target_text"]:
            raise ValueError("Reference and context have different target text")
        ids, spans, clipped = signature(r)
        ref_ids, ref_spans, ref_clipped = signature(ref)
        rows.append({
            **{k: r[k] for k in keys}, "template_id": r["mode"].removeprefix("sentence_"),
            "stimulus_id": r["stimulus_id"], "reference_stimulus_id": ref["stimulus_id"],
            "target_text": r["target_text"],
            "count_equal": len(ids) == len(ref_ids), "ids_equal": ids == ref_ids,
            "relative_spans_equal": spans == ref_spans,
            "within_term_spans_equal": clipped == ref_clipped,
            "exact_match": ids == ref_ids and spans == ref_spans,
            "target_ids": json.dumps(ids), "reference_ids": json.dumps(ref_ids),
            "relative_spans": json.dumps(spans), "reference_relative_spans": json.dumps(ref_spans),
        })
    return pd.DataFrame(rows)


def summarize_context_comparisons(comparisons: pd.DataFrame) -> dict:
    checks = ("count_equal", "ids_equal", "relative_spans_equal", "within_term_spans_equal", "exact_match")
    groups = []
    for (tid, template), sub in comparisons.groupby(["tokenizer_id", "template_id"]):
        groups.append({"tokenizer_id": tid, "template_id": template, "n_comparisons": len(sub),
                       **{f"{c}_mismatches": int((~sub[c]).sum()) for c in checks}})
    ab = comparisons[comparisons["template_id"].isin(config.TEMPLATES)]
    return {"reference": "isolated_space target tokens, relative byte spans",
            "n_comparisons": len(comparisons), "n_comparisons_AB": len(ab),
            "AB_exact_match_all": bool(ab["exact_match"].all()) if len(ab) else False,
            "by_tokenizer_template": groups}
