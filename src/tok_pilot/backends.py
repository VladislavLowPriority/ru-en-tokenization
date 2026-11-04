"""Запуск токенизаторов и восстановление границ токенов (§6).

Главный принцип: границы считаются по последовательности ИСХОДНЫХ БАЙТОВ
токенов. Протокол прямо запрещает складывать длины отдельно декодированных
токенов (byte-level токен может разрезать кириллическую букву пополам, и
такое декодирование даст символ замены вместо реальной длины).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .textio import sha256_file


@dataclass
class Encoded:
    ids: list[int]
    token_bytes: list[bytes]
    byte_spans: list[tuple[int, int]]
    decode_match: bool
    special_token_count: int
    hf_char_offsets: list[tuple[int, int]] | None = None


def _spans(token_bytes: list[bytes]) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for b in token_bytes:
        spans.append((pos, pos + len(b)))
        pos += len(b)
    return spans


class TiktokenBackend:
    backend = "tiktoken"

    def __init__(self, tokenizer_id: str, spec: dict):
        import tiktoken

        self.tokenizer_id = tokenizer_id
        self.spec = spec
        self.enc = tiktoken.get_encoding(spec["encoding"])
        self._special = set(self.enc._special_tokens.values())

    def encode(self, text: str) -> Encoded:
        # allowed_special=set(), disallowed_special='all' по умолчанию:
        # если в строке окажется текст спецтокена, будет ошибка, а не тихий спецтокен.
        ids = self.enc.encode(text)
        tb = [self.enc.decode_single_token_bytes(i) for i in ids]
        return Encoded(
            ids=ids,
            token_bytes=tb,
            byte_spans=_spans(tb),
            decode_match=(b"".join(tb) == text.encode("utf-8")),
            special_token_count=sum(1 for i in ids if i in self._special),
        )

    def manifest(self) -> dict:
        import hashlib
        import importlib.metadata as md

        ranks = self.enc._mergeable_ranks
        h = hashlib.sha256()
        for tok, rank in sorted(ranks.items(), key=lambda kv: kv[1]):
            h.update(rank.to_bytes(4, "big"))
            h.update(len(tok).to_bytes(4, "big"))
            h.update(tok)
        files = []
        cache = os.environ.get("TIKTOKEN_CACHE_DIR")
        if cache and Path(cache).is_dir():
            for p in sorted(Path(cache).iterdir()):
                if p.is_file():
                    files.append({"file": p.name, "sha256": sha256_file(p)})
        return {
            "tokenizer_id": self.tokenizer_id,
            "backend": "tiktoken",
            "encoding": self.spec["encoding"],
            "package_version": f"tiktoken=={md.version('tiktoken')}",
            "n_vocab": self.enc.n_vocab,
            "n_mergeable_ranks": len(ranks),
            "vocab_sha256": h.hexdigest(),
            "pre_tokenizer_regex": self.enc._pat_str,
            "special_tokens": sorted(self.enc._special_tokens),
            "normalizer": "нет (tiktoken не нормализует вход)",
            "cache_files": files,
            "role": self.spec["role"],
        }


class HFBackend:
    backend = "hf"

    def __init__(self, tokenizer_id: str, spec: dict):
        from transformers import AutoTokenizer

        self.tokenizer_id = tokenizer_id
        self.spec = spec
        self.tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
        if not self.tok.is_fast:
            raise RuntimeError("Нужен fast-токенизатор: иначе нет offsets")
        self.byte_decoder = self._build_byte_decoder()
        self._special = set(self.tok.all_special_ids)

    @staticmethod
    def _build_byte_decoder() -> dict[str, int]:
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

        return {v: k for k, v in bytes_to_unicode().items()}

    def encode(self, text: str) -> Encoded:
        enc = self.tok(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            padding=False,
        )
        ids = list(enc["input_ids"])
        pieces = self.tok.convert_ids_to_tokens(ids)
        tb = [bytes(self.byte_decoder[ch] for ch in p) for p in pieces]
        return Encoded(
            ids=ids,
            token_bytes=tb,
            byte_spans=_spans(tb),
            decode_match=(b"".join(tb) == text.encode("utf-8")),
            special_token_count=sum(1 for i in ids if i in self._special),
            hf_char_offsets=[tuple(o) for o in enc["offset_mapping"]],
        )

    def manifest(self) -> dict:
        import tokenizers
        import transformers

        files = []
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                     "merges.txt", "special_tokens_map.json"):
            try:
                from huggingface_hub import hf_hub_download

                p = hf_hub_download(self.spec["repo"], name, revision=self.spec["revision"])
                files.append({"file": name, "sha256": sha256_file(p), "local_path": p})
            except Exception as e:  # файла может не быть в снимке — это факт, а не ошибка
                files.append({"file": name, "sha256": None, "note": f"{type(e).__name__}"})

        bt = self.tok.backend_tokenizer
        spec = json.loads(bt.to_str())
        return {
            "tokenizer_id": self.tokenizer_id,
            "backend": "hf",
            "repo": self.spec["repo"],
            "revision": self.spec["revision"],
            "package_version": (f"transformers=={transformers.__version__}; "
                                f"tokenizers=={tokenizers.__version__}"),
            "n_vocab": len(self.tok),
            "vocab_size_attr": self.tok.vocab_size,
            "normalizer": spec.get("normalizer"),
            "pre_tokenizer": spec.get("pre_tokenizer"),
            "post_processor": spec.get("post_processor"),
            "decoder": spec.get("decoder"),
            "n_added_tokens": len(self.tok.get_added_vocab()),
            "special_tokens": self.tok.all_special_tokens,
            "files": files,
            "role": self.spec["role"],
        }


def load_all(ids: list[str] | None = None) -> dict:
    out = {}
    for tid, spec in config.TOKENIZERS.items():
        if ids and tid not in ids:
            continue
        out[tid] = (TiktokenBackend if spec["backend"] == "tiktoken" else HFBackend)(tid, spec)
    return out
