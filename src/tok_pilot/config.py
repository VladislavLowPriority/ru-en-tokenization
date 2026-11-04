"""Неизменяемые настройки пилота.

Всё, что протокол требует «зафиксировать до анализа», собрано здесь, чтобы
никакой параметр не выбирался задним числом по виду результатов.
Ссылки на разделы — по protocol_ru_en_tokenization.md версии 1.0.
"""
from __future__ import annotations

from pathlib import Path

PROTOCOL_VERSION = "1.0"
PROTOCOL_FILE = "protocol_ru_en_tokenization.md"

# Единый seed протокола (§3 п.4, §8): отбор внутри страт и бутстрэп.
SEED = 20260918
BOOTSTRAP_REPS = 5000

ROOT = Path(__file__).resolve().parents[2]
DATA_REAL = ROOT / "data" / "real"
DATA_DEMO = ROOT / "data" / "demo"
RESULTS = ROOT / "results"

# --- Токенизаторы (§6) -------------------------------------------------------
# PRIMARY определён заранее и не переназначается по величине эффекта (§8).
PRIMARY_TOKENIZER = "o200k_base"

TOKENIZERS = {
    "o200k_base": {
        "backend": "tiktoken",
        "encoding": "o200k_base",
        "package": "tiktoken",
        "role": "основной, заранее выбранный объект H1",
    },
    "cl100k_base": {
        "backend": "tiktoken",
        "encoding": "cl100k_base",
        "package": "tiktoken",
        "role": "второй словарь той же библиотеки (не независимая репликация)",
    },
    "qwen2.5-7b-instruct": {
        "backend": "hf",
        "repo": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "acbd96531cda22292a3ceaa67e984955d3965282",
        "package": "transformers",
        "role": "готовая система другой линии разработки",
    },
}

# --- Режимы предъявления (§5) ---
# Решение D1 (§13 протокола, 18.09.2026): основной режим для H1 — isolated_0,
# isolated_space — анализ чувствительности. Обе оценки считаются и публикуются
# для каждого токенизатора; единая итоговая цифра эффекта не выбирается.
# Оба режима — изолированное предъявление: анализ термина внутри предложения
# выполняется отдельно по величине S в режимах sentence_*.
PRIMARY_MODE = "isolated_0"
SENSITIVITY_MODE = "isolated_space"

MODES = ("isolated_0", "isolated_space", "sentence_A", "sentence_B", "sentence_C")

# Шаблоны контекста. Пилот проверяет их естественность; после пилота они
# либо замораживаются, либо меняются целиком для всей выборки (§5, §10).
TEMPLATES = {
    "A": {
        "ru": {"sg": "Проверяется {term}.", "pl": "Проверяются {term}."},
        "en": {"sg": "The {term} is being tested.", "pl": "The {term} are being tested."},
    },
    "B": {
        "ru": {"sg": "Используется {term}.", "pl": "Используются {term}."},
        "en": {"sg": "The {term} is in use.", "pl": "The {term} are in use."},
    },
}

# Диагностический шаблон: слева от термина стоит НЕ пробел.
# Проверяет другое непосредственное окружение. Совпадение сегментации в A/B
# с isolated_space проверяется построчно, а не предполагается из названия BPE.
# Держится отдельно от TEMPLATES, чтобы не подменять
# заранее объявленную проверку H3 задним числом.
DIAGNOSTIC_TEMPLATES = {
    "C": {
        "ru": {"sg": "Проверяется следующее: ({term}).",
               "pl": "Проверяется следующее: ({term})."},
        "en": {"sg": "The following is tested: ({term}).",
               "pl": "The following is tested: ({term})."},
    },
}

ALL_TEMPLATES = {**TEMPLATES, **DIAGNOSTIC_TEMPLATES}

# --- Частотность (§8) --------------------------------------------------------
WORDFREQ_WORDLIST = "large"
# Предлоги в среднее по компонентам не входят (§8).
FREQ_STOPWORDS = {
    "ru": {"для", "с", "со", "под", "на", "в", "из", "по", "при", "к", "от", "о", "об"},
    "en": {"of", "for", "with", "in", "on", "at", "to", "by", "from"},
}

# --- Границы интерпретации (§10, §12) ---------------------------------------
PRECISION_TARGET_LOG_HALFWIDTH = 0.13976194237515863  # log(1.15)
CROSS_BOUNDARY_ALERT = 0.05      # >5% контекстных измерений — S плохо отделяется
EQUIVALENCE_MIN_ACCEPTED = 15    # из 18 пилотных пар
FREQ_COVERAGE_MIN = 0.80
FAMILY_TARGET_MAIN = 30
MAX_CONCEPTS_PER_FAMILY = 3
