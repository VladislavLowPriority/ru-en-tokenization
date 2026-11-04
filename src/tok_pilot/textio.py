"""Обработка исходных строк по §6 «Правила текста и специальных токенов».

Задача модуля — сделать так, чтобы ни один эффект не возник из-за невидимых
символов, «ё»→«е» или скрытой нормализации, и чтобы всякое изменение строки
было записано в журнал, а не сделано молча.
"""
from __future__ import annotations

import hashlib
import unicodedata

# Символы, которые внешне неотличимы от пробела или пусты, но меняют токенизацию.
SUSPECT_CHARS = {
    "\u00a0": "NBSP",
    "\u202f": "NARROW_NBSP",
    "\u2009": "THIN_SPACE",
    "\u200b": "ZERO_WIDTH_SPACE",
    "\u200e": "LRM",
    "\u200f": "RLM",
    "\ufeff": "BOM",
    "\u00ad": "SOFT_HYPHEN",
    "\u2011": "NON_BREAKING_HYPHEN",
    "\t": "TAB",
    "\n": "NEWLINE",
    "\r": "CR",
}


def scan_suspect(text: str) -> list[dict]:
    """Найти невидимые/нестандартные символы. Только выявление, без замены."""
    found = []
    for i, ch in enumerate(text):
        if ch in SUSPECT_CHARS:
            found.append({"pos": i, "char": repr(ch), "name": SUSPECT_CHARS[ch]})
        elif unicodedata.category(ch) in {"Cf", "Cc"} and ch != " ":
            found.append({"pos": i, "char": repr(ch), "name": unicodedata.category(ch)})
    return found


def prepare(raw: str) -> dict:
    """Из исходной строки получить исследовательскую, записав все изменения.

    Разрешено ровно три действия: NFC, снятие случайных краевых пробелов и
    схлопывание кратных внутренних U+0020. «ё» сохраняется, регистр не меняется.
    """
    log: list[str] = []
    suspects = scan_suspect(raw)
    if suspects:
        log.append(f"suspect_chars={suspects}")

    nfc = unicodedata.normalize("NFC", raw)
    if nfc != raw:
        log.append("NFC_changed")

    stripped = nfc.strip(" ")
    if stripped != nfc:
        log.append("edge_spaces_removed")

    collapsed = " ".join(stripped.split(" "))
    if collapsed != stripped:
        log.append("multiple_inner_spaces_collapsed")

    if "ё" in collapsed or "Ё" in collapsed:
        log.append("yo_preserved")

    return {
        "raw_text": raw,
        "text": collapsed,
        "normalization_log": "; ".join(log) if log else "",
        "has_suspect_chars": bool(suspects),
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
