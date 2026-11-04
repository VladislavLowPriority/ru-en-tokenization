"""Измерение: сколько токенов во входе и сколько из них пересекают термин (§6).

T — все токены входной строки.
S — токены, пересекающие размеченный интервал термина; чисто пробельный токен
    в S не входит, а токен «пробел+буквы термина» входит.
"""
from __future__ import annotations

import json

import pandas as pd

from .backends import Encoded


def term_byte_span(input_text: str, start: int, end: int) -> tuple[int, int]:
    """Символьные границы термина -> байтовые, по фактической кодировке UTF-8."""
    return (len(input_text[:start].encode("utf-8")),
            len(input_text[:end].encode("utf-8")))


def _is_space_only(b: bytes) -> bool:
    return len(b) > 0 and all(x == 0x20 for x in b)


def count_target(enc: Encoded, bs: int, be: int) -> dict:
    """Счёт S и диагностика пересечения границ.

    Всё считается по байтам, поэтому корректно и тогда, когда byte-level токен
    разрезает кириллическую букву: такой токен пересекает термин и учитывается.
    """
    n, n_cross, hits = 0, 0, []
    for idx, ((ts, te), tb) in enumerate(zip(enc.byte_spans, enc.token_bytes)):
        if not (max(ts, bs) < min(te, be)):
            continue                      # нет пересечения с термином
        if _is_space_only(tb):
            continue                      # отдельный токен-пробел не принадлежит термину
        # байты токена, лежащие вне термина: слева от его начала и справа от конца
        left = tb[: max(0, bs - ts)]
        right = tb[max(0, be - ts):] if te > be else b""
        cross = any(x != 0x20 for x in left + right)
        n += 1
        n_cross += int(cross)
        hits.append({"i": idx, "id": enc.ids[idx], "span": [ts, te], "cross": cross})
    return {"n_tokens_target": n, "n_cross_boundary": n_cross, "target_tokens": hits}


def measure(stimuli: pd.DataFrame, backends: dict, run_id: str = "run1") -> pd.DataFrame:
    rows = []
    for _, s in stimuli.iterrows():
        text = s["input_text"]
        bs, be = term_byte_span(text, int(s["target_start"]), int(s["target_end"]))
        for tid, be_ in backends.items():
            error = ""
            try:
                enc = be_.encode(text)
                res = count_target(enc, bs, be)
            except Exception as e:
                rows.append({
                    "stimulus_id": s["stimulus_id"], "tokenizer_id": tid, "run_id": run_id,
                    "n_tokens_input": None, "n_tokens_target": None,
                    "error_code": f"{type(e).__name__}: {e}",
                })
                continue
            rows.append({
                "stimulus_id": s["stimulus_id"],
                "concept_id": s["concept_id"], "family_id": s["family_id"],
                "stratum": s["stratum"], "lang": s["lang"],
                "number": s["number"], "mode": s["mode"], "template_id": s["template_id"],
                "tokenizer_id": tid, "run_id": run_id,
                "n_tokens_input": len(enc.ids),
                "n_tokens_target": res["n_tokens_target"],
                "n_cross_boundary": res["n_cross_boundary"],
                "cross_boundary": res["n_cross_boundary"] > 0,
                "decode_match": enc.decode_match,
                "special_token_count": enc.special_token_count,
                "token_ids": json.dumps(enc.ids),
                "token_spans": json.dumps([list(x) for x in enc.byte_spans]),
                "target_tokens": json.dumps(res["target_tokens"], ensure_ascii=False),
                "term_byte_start": bs, "term_byte_end": be,
                "error_code": error,
            })
    return pd.DataFrame(rows)
