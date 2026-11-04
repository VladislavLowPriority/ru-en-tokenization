"""Построение таблицы входных строк (§5, §9).

Из одной строки-понятия получается 2 языка x 2 числа x 4 режима = 16 строк.
Границы термина внутри предложения размечаются здесь, ДО токенизации, —
иначе счёт токенов термина зависел бы от того, что выдал токенизатор.
"""
from __future__ import annotations

import pandas as pd

from . import config
from .textio import prepare, sha256_text

REQUIRED_CONCEPT_COLUMNS = [
    "concept_id", "family_id", "split", "stratum", "domain", "definition",
    "term_ru_sg", "term_ru_pl", "term_en_sg", "term_en_pl",
    "head_ru", "head_en",
    "source_ru", "source_en", "source_entry", "edition", "access_date",
    "review_technical", "review_language", "final_status", "decision_note",
    "alternatives", "exclusion_reason", "selection_order",
    "term_pattern_ru", "term_pattern_en",
    "changed_components_ru", "changed_components_en",
    "inflection_note_ru", "inflection_note_en",
]


def load_concepts(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_CONCEPT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"В таблице понятий нет обязательных полей: {missing}")
    if df["concept_id"].duplicated().any():
        raise ValueError("Повторяющиеся concept_id")
    return df


def _locate(term: str, sentence: str) -> tuple[int, int]:
    """Найти термин в предложении. Неоднозначность — ошибка, а не догадка."""
    first = sentence.find(term)
    if first < 0:
        raise ValueError(f"Термин {term!r} не найден в {sentence!r}")
    if sentence.find(term, first + 1) >= 0:
        raise ValueError(f"Термин {term!r} встречается в {sentence!r} более одного раза")
    return first, first + len(term)


def build_stimuli(concepts: pd.DataFrame, numbers=("sg", "pl")) -> pd.DataFrame:
    """numbers ограничивает набор форм числа.

    Для общеязыкового контроля (§14.6) формы множественного числа не
    образуются, поэтому там строки строятся только для единственного числа.
    """
    rows = []
    for _, c in concepts.iterrows():
        for lang in ("ru", "en"):
            for number in numbers:
                raw_term = c[f"term_{lang}_{number}"]
                if not str(raw_term).strip():
                    raise ValueError(f"{c['concept_id']}: пустой term_{lang}_{number}")
                p = prepare(raw_term)
                term = p["text"]

                variants = {
                    "isolated_0": (term, None),
                    "isolated_space": (" " + term, None),
                }
                for tpl_id, tpl in config.ALL_TEMPLATES.items():
                    variants[f"sentence_{tpl_id}"] = (
                        tpl[lang][number].format(term=term), tpl_id
                    )

                for mode, (input_text, tpl_id) in variants.items():
                    start, end = _locate(term, input_text)
                    components = [w for w in term.split(" ") if w]
                    rows.append({
                        "stimulus_id": f"{c['concept_id']}|{lang}|{number}|{mode}",
                        "concept_id": c["concept_id"],
                        "family_id": c["family_id"],
                        "split": c["split"],
                        "stratum": c["stratum"],
                        "lang": lang,
                        "number": number,
                        # Русский термин остаётся в именительном падеже (§5);
                        # для английского категория падежа не размечается.
                        "case": "nom" if lang == "ru" else "",
                        "mode": mode,
                        "template_id": tpl_id or "",
                        "raw_text": raw_term,
                        "input_text": input_text,
                        "target_text": term,
                        "target_start": start,
                        "target_end": end,
                        "offset_unit": "char",
                        "word_count": len(components),
                        "char_count_no_spaces": len(term.replace(" ", "")),
                        "utf8_byte_count": len(term.encode("utf-8")),
                        "input_char_count": len(input_text),
                        "input_utf8_byte_count": len(input_text.encode("utf-8")),
                        "term_pattern": c[f"term_pattern_{lang}"],
                        "changed_components": c[f"changed_components_{lang}"],
                        "inflection_note": c[f"inflection_note_{lang}"],
                        "normalization_log": p["normalization_log"],
                        "text_sha256": sha256_text(input_text),
                    })
    return pd.DataFrame(rows)


def add_frequencies(stimuli: pd.DataFrame) -> pd.DataFrame:
    """Приблизительная частотность компонентов (§8).

    Это НЕ частота термина и НЕ частота в корпусе обучения токенизатора:
    wordfreq оценивает отдельные словоформы по своим спискам.
    """
    import importlib.metadata as _md

    from wordfreq import zipf_frequency

    wf_version = _md.version("wordfreq")

    freqs, proxies, missing = [], [], []
    for _, r in stimuli.iterrows():
        lang = r["lang"]
        stop = config.FREQ_STOPWORDS[lang]
        per_word, used, absent = [], [], 0
        for w in str(r["target_text"]).split(" "):
            z = zipf_frequency(w, lang, wordlist=config.WORDFREQ_WORDLIST)
            per_word.append(f"{w}={z:.2f}")
            if w.lower() in stop:
                continue
            if z == 0.0:
                absent += 1
            else:
                used.append(z)
        freqs.append("; ".join(per_word))
        proxies.append(sum(used) / len(used) if used else float("nan"))
        missing.append(absent)

    out = stimuli.copy()
    out["component_frequencies"] = freqs
    out["freq_proxy"] = proxies
    out["freq_missing"] = missing
    out["frequency_version"] = f"wordfreq=={wf_version}/{config.WORDFREQ_WORDLIST}"
    return out
