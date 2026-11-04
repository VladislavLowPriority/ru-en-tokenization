# -*- coding: utf-8 -*-
"""Проверка самой логики подсчёта S и пересечения границ.

Нули в основном прогоне ничего не доказывают, если детектор в принципе не
срабатывает. Здесь логика проверяется на заведомо известных случаях:
часть — искусственные последовательности байтов, часть — реальные строки,
для которых токенизатор действительно склеивает термин с соседним текстом.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tok_pilot.backends import Encoded, load_all, _spans  # noqa: E402
from tok_pilot.measure import count_target, term_byte_span  # noqa: E402

FAILED = []


def check(name, got, want):
    ok = got == want
    print(f"{'OK ' if ok else 'FAIL'} {name}: получено {got}, ожидалось {want}")
    if not ok:
        FAILED.append(name)


def synthetic(tokens: list[bytes], text: str, term: str):
    """Собрать Encoded вручную и посчитать S для известного разбиения."""
    assert b"".join(tokens) == text.encode("utf-8"), "тест задан неверно"
    enc = Encoded(ids=list(range(len(tokens))), token_bytes=tokens,
                  byte_spans=_spans(tokens), decode_match=True, special_token_count=0)
    start = text.index(term)
    bs, be = term_byte_span(text, start, start + len(term))
    return count_target(enc, bs, be)


print("--- искусственные разбиения (логика подсчёта) ---")

# 1. Термин целиком в одном токене.
r = synthetic([b"sensor"], "sensor", "sensor")
check("один токен на термин: S", r["n_tokens_target"], 1)
check("один токен на термин: пересечений", r["n_cross_boundary"], 0)

# 2. Отдельный токен-пробел перед термином в счёт термина НЕ входит.
r = synthetic([b" ", b"sensor"], " sensor", "sensor")
check("отдельный пробел не считается: S", r["n_tokens_target"], 1)

# 3. Токен 'пробел+буквы термина' входит и НЕ является пересечением границы.
r = synthetic([b" sensor"], " sensor", "sensor")
check("пробел внутри токена термина: S", r["n_tokens_target"], 1)
check("пробел внутри токена термина: пересечений", r["n_cross_boundary"], 0)

# 4. Токен захватил термин и точку — это пересечение границы.
r = synthetic([b"The", b" sensor."], "The sensor.", "sensor")
check("термин склеен с точкой: S", r["n_tokens_target"], 1)
check("термин склеен с точкой: пересечений", r["n_cross_boundary"], 1)

# 5. Токен захватил конец предыдущего слова и начало термина.
r = synthetic([b"These", b"nsor"], "Thesensor", "sensor")
check("захват предыдущего слова: S", r["n_tokens_target"], 2)
check("захват предыдущего слова: пересечений", r["n_cross_boundary"], 1)

# 6. Байтовый токен разрезает кириллическую букву на границе термина.
#    'я' = D1 8F; левый токен забирает первый байт следующей за термином буквы.
text = "датчикя"
tokens = [b"\xd0\xb4\xd0\xb0\xd1\x82", b"\xd1\x87\xd0\xb8\xd0\xba\xd1", b"\x8f"]
r = synthetic(tokens, text, "датчик")
check("разрезанная буква на границе: S", r["n_tokens_target"], 2)
check("разрезанная буква на границе: пересечений", r["n_cross_boundary"], 1)

# 7. Термин в начале строки, за ним слово без пробела внутри одного токена.
r = synthetic([b"sensorX"], "sensorX", "sensor")
check("термин и соседний текст в одном токене", r["n_cross_boundary"], 1)

print("\n--- реальные токенизаторы: срабатывает ли детектор на настоящих строках ---")
bks = load_all()
probes = [
    ("Проверяется датчик, привод.", "датчик"),
    ("The sensor, valve and drive.", "sensor"),
    ("датчик-преобразователь", "датчик"),
    ("Проверяется расходомер.", "расходомер"),
]
for tid, be in bks.items():
    for text, term in probes:
        enc = be.encode(text)
        start = text.index(term)
        bs, e_ = term_byte_span(text, start, start + len(term))
        r = count_target(enc, bs, e_)
        flag = "ПЕРЕСЕЧЕНИЕ" if r["n_cross_boundary"] else "чисто"
        print(f"  {tid:<22} {text!r:<34} S={r['n_tokens_target']} {flag}")

print("\nИтог:", "все проверки пройдены" if not FAILED else f"ПРОВАЛЫ: {FAILED}")
sys.exit(1 if FAILED else 0)
