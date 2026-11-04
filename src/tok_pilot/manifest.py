"""Технический манифест (§6): что именно было запущено и на каких данных.

Без него результат нельзя проверить: токенизаторы и пакеты обновляются, и
через полгода «o200k_base» без версии пакета уже не является фиксацией.
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .textio import sha256_file, sha256_text


def package_versions() -> dict:
    import importlib.metadata as md

    out = {}
    for name in ("tiktoken", "transformers", "tokenizers", "wordfreq",
                 "pandas", "numpy", "huggingface-hub", "regex"):
        try:
            out[name] = md.version(name)
        except Exception:
            out[name] = None
    return out


def build(backends: dict, data_files: list[Path], stimuli_df, split: str) -> dict:
    return {
        "protocol_version": config.PROTOCOL_VERSION,
        "protocol_sha256": sha256_file(config.ROOT / config.PROTOCOL_FILE),
        "split": split,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "seed": config.SEED,
        "bootstrap_reps": config.BOOTSTRAP_REPS,
        "primary_tokenizer": config.PRIMARY_TOKENIZER,
        "templates": config.TEMPLATES,
        "diagnostic_templates": config.DIAGNOSTIC_TEMPLATES,
        "analysis_revision": "audit-step1-complete-frequency-family-bootstrap-context-identity",
        "code_files": [{"path": str(p.relative_to(config.ROOT)), "sha256": sha256_file(p)}
                       for folder in ("src", "scripts")
                       for p in sorted((config.ROOT / folder).rglob("*.py"))],
        "requirements_sha256": sha256_file(config.ROOT / "requirements.lock"),
        "modes": list(config.MODES),
        "tokenizers": {tid: be.manifest() for tid, be in backends.items()},
        "data_files": [{"path": str(p.relative_to(config.ROOT)),
                        "sha256": sha256_file(p)} for p in data_files],
        "stimuli_table_sha256": sha256_text(
            stimuli_df.sort_values("stimulus_id").to_csv(index=False)),
        "n_stimuli": int(len(stimuli_df)),
    }


def save(man: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
