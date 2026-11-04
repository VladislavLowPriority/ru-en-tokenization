# -*- coding: utf-8 -*-
"""Сбор кандидатов русско-английских пар терминов из WIPO Pearl.

Зачем: WIPO Pearl — независимая терминологическая база, не связанная с
документацией производителей. Пара EN/RU берётся из ОДНОЙ concept record,
поэтому эквивалентность подтверждается самой базой, а не переводом
исследователя. Ничего не переводится и не досочиняется.

Что делает скрипт:
  1. ходит по списку «семейств» (sensor, valve, meter, ...);
  2. постранично забирает полные записи (limit/offset);
  3. сохраняет КАЖДЫЙ сырой ответ на диск — это происхождение данных;
  4. вытаскивает EN/RU обозначения, rating, usage, context, источник (WO-номер);
  5. применяет формальные правила отбора протокола;
  6. относит пару к структурной группе A/B/C;
  7. сверяет с уже собранным корпусом, чтобы не было повторов;
  8. пишет CSV и markdown-таблицу со статусами ACCEPT / RESERVE / EXCLUDE.

Токенизация здесь не выполняется: сначала фиксируется корпус.

Запуск:
    .venv\\Scripts\\python.exe scripts\\wipo_collect.py
    .venv\\Scripts\\python.exe scripts\\wipo_collect.py --seeds sensor valve
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / "src" / "api.env"
OUT_DIR = ROOT / "Слова_данные" / "WIPO_PEARL"
RAW_DIR = OUT_DIR / "raw"
CORPUS = ROOT / "data" / "real" / "concepts_pilot.csv"

API = "https://api.wipo.int/wipopearl/records"
TOKEN_URL = "https://oauth2.api.wipo.int/external/oauth/token"
SUBJECT_FIELDS = "ELEC,MEAS,MANU,MECH"
PAGE = 25
MAX_RECORDS_PER_SEED = 150
PAUSE = 0.35            # вежливая пауза между запросами

# Семейства для поиска. Список C-first: сначала то, где вероятнее асимметрия.
SEEDS = [
    "sensor", "detector", "transducer", "actuator", "encoder", "meter",
    "gauge", "switch", "valve", "drive", "motor", "relay", "controller",
    "regulator", "transformer", "coupler", "indicator", "transmitter",
    "limiter", "filter", "pump", "gear", "probe", "diode", "amplifier",
    "bearing", "connector", "capacitor", "resistor", "inductor", "fuse",
    "converter", "rectifier", "oscillator", "antenna", "electrode",
    "nozzle", "piston", "spring", "shaft", "clutch", "brake",
    "generator", "rotor", "stator", "winding", "contactor", "solenoid",
    "accelerometer", "gyroscope", "thermometer", "compressor", "turbine",
    "coil", "lamp", "battery", "membrane", "bushing", "seal", "gasket",
    "damper", "flange", "coupling", "spindle", "cam", "gripper",
    "conveyor", "hopper", "mixer", "heater", "cooler", "exchanger",
    "rheostat", "potentiometer", "thermostat", "manipulator", "impeller",
]


# --- доступ к API ------------------------------------------------------------

def read_env() -> dict:
    """Читает api.env. Значения секретов никуда не печатаются."""
    out = {}
    for line in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sep = ":" if ":" in line and ("=" not in line or line.index(":") < line.index("=")) else "="
        if sep not in line:
            continue
        k, v = line.split(sep, 1)
        key = re.sub(r"[^a-z]", "", k.lower())
        out[key] = v.strip().strip('"').strip("'")
    return out


def token_expiry(tok: str) -> float:
    try:
        payload = tok.split(".")[1]
        data = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        return float(data.get("exp", 0))
    except Exception:
        return 0.0


def get_token(env: dict) -> str:
    """Берёт токен из файла; если он истёк — запрашивает новый по client_credentials."""
    tok = env.get("accesstoken", "")
    if tok and token_expiry(tok) > time.time() + 120:
        return tok
    cid, secret = env.get("clientid"), env.get("clientsecret")
    if not (cid and secret):
        raise SystemExit("Токен истёк, а client_id/client_secret в api.env не найдены.")
    body = urllib.parse.urlencode({
        "grant_type": "client_credentials", "client_id": cid, "client_secret": secret,
    }).encode()
    req = urllib.request.Request(
        TOKEN_URL, data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=60) as r:
        new = json.loads(r.read())["access_token"]
    print("   получен новый токен (старый истёк)")
    return new


def api_get(token: str, params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url, headers={"Authorization": "Bearer " + token, "Accept": "application/json"})
    last = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(3 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            # обрыв связи или сбой прокси — пробуем ещё раз, а не роняем сбор
            last = e
            time.sleep(3 * (attempt + 1))
            continue
    raise RuntimeError(f"не удалось получить ответ после 5 попыток: {last}")


# --- разбор записей ----------------------------------------------------------

def clean(s: str) -> str:
    """Снимает HTML-подсветку совпадений, которую API вставляет в поле term."""
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = s.replace("&amp;", "&").replace("&nbsp;", " ").replace("&quot;", '"')
    return re.sub(r"\s+", " ", s).strip()


def pick_terms(record: dict, lang: str) -> list[dict]:
    """Все обозначения одного языка, отсортированные по надёжности."""
    items = []
    for li in record.get("languageItems", []):
        if li.get("language") != lang:
            continue
        for ti in li.get("termItems", []):
            term = clean(ti.get("term", ""))
            if not term:
                continue
            try:
                rating = int(ti.get("rating") or 0)
            except ValueError:
                rating = 0
            items.append({
                "term": term,
                "term_id": ti.get("termId") or ti.get("id"),
                "rating": rating,
                "usage": (ti.get("usage") or "").strip(),
                "context": clean(ti.get("context", ""))[:400],
                "source": ti.get("sourceText") or ti.get("sourceLink") or "",
            })
    items.sort(key=lambda x: (x["usage"] != "allowed", -x["rating"]))
    return items


# --- правила отбора протокола ------------------------------------------------

CYR = re.compile(r"[А-Яа-яЁё]")
LAT = re.compile(r"[A-Za-z]")

# Протокол требует физические устройства и компоненты. Если ВЕРШИНА термина —
# величина, параметр, процесс или программная сущность, это не наш объект.
NON_DEVICE_HEADS_EN = {
    "speed", "data", "system", "method", "process", "signal", "protocol",
    "value", "rate", "ratio", "factor", "coefficient", "current", "voltage",
    "power", "frequency", "temperature", "pressure", "force", "torque",
    "energy", "time", "length", "mass", "weight", "density", "resistance",
    "capacitance", "inductance", "impedance", "phase", "noise", "software",
    "algorithm", "program", "interface", "network", "model", "structure",
    "material", "technology", "effect", "principle", "mode", "state",
    "board", "circuit", "array", "matrix", "field", "wave", "beam", "light",
}
NON_DEVICE_HEADS_RU = {
    "скорость", "данные", "система", "метод", "способ", "процесс", "сигнал",
    "протокол", "значение", "величина", "коэффициент", "ток", "напряжение",
    "мощность", "частота", "температура", "давление", "сила", "момент",
    "энергия", "время", "длина", "масса", "плотность", "сопротивление",
    "ёмкость", "емкость", "индуктивность", "фаза", "шум", "алгоритм",
    "программа", "интерфейс", "сеть", "модель", "структура", "материал",
    "технология", "эффект", "принцип", "режим", "состояние", "поле",
    "волна", "луч", "свет", "схема", "матрица",
}


def reject_reason(term: str, lang: str) -> str | None:
    """Формальные запреты §3. Возвращает причину или None, если термин проходит."""
    if not term:
        return "пусто"
    if re.search(r"\d", term):
        return "цифры"
    if "-" in term or "–" in term:
        return "дефис"
    if re.search(r"[()\[\]/,;:«»\"']", term):
        return "скобки или пунктуация"
    words = term.split(" ")
    if len(words) > 3:
        return f"больше 3 слов ({len(words)})"
    for w in words:
        core = re.sub(r"[^\w]", "", w)
        if len(core) >= 2 and core.isupper():
            return f"аббревиатура «{w}»"
    if lang == "ru" and LAT.search(term):
        return "латиница в русском термине"
    if lang == "en" and CYR.search(term):
        return "кириллица в английском термине"
    head = words[-1].lower()
    if lang == "en" and head in NON_DEVICE_HEADS_EN:
        return f"вершина «{head}» — параметр или процесс, а не устройство"
    if lang == "ru" and head in NON_DEVICE_HEADS_RU:
        return f"вершина «{head}» — параметр или процесс, а не устройство"
    return None


def stratum_of(en_words: int, ru_words: int) -> str:
    if en_words == 1 and ru_words == 1:
        return "A"
    if 2 <= en_words <= 3 and 2 <= ru_words <= 3:
        return "B"
    if (en_words == 1 and 2 <= ru_words <= 3) or (2 <= en_words <= 3 and ru_words == 1):
        return "C"
    return "?"


# --- уже собранный корпус ----------------------------------------------------

def load_corpus() -> dict:
    """Термины и головные слова, которые уже в корпусе, — чтобы не повторяться."""
    if not CORPUS.exists():
        return {"en": set(), "ru": set(), "heads_en": {}, "heads_ru": {}}
    en, ru = set(), set()
    heads_en: dict[str, int] = {}
    heads_ru: dict[str, int] = {}
    with CORPUS.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("final_status") != "accepted":
                continue
            en.add(row["term_en_sg"].strip().lower())
            ru.add(row["term_ru_sg"].strip().lower())
            he = (row.get("head_en") or "").strip().lower()
            hr = (row.get("head_ru") or "").strip().lower()
            if he:
                heads_en[he] = heads_en.get(he, 0) + 1
            if hr:
                heads_ru[hr] = heads_ru.get(hr, 0) + 1
    return {"en": en, "ru": ru, "heads_en": heads_en, "heads_ru": heads_ru}


def family_hits(en_term: str, ru_term: str, corpus: dict) -> list[str]:
    """Пересечение с лексическими семьями корпуса (§3 п. 5).

    Проверка не только по последнему слову: русский язык образует сложные
    слова, и «тензодатчик» принадлежит семье «датчик», хотя отдельного слова
    «датчик» в нём нет. Поэтому дополнительно ищется вхождение корня.
    """
    out = []
    for lang, term, heads in (("EN", en_term.lower(), corpus["heads_en"]),
                              ("RU", ru_term.lower(), corpus["heads_ru"])):
        words = term.split(" ")
        for head, n in heads.items():
            if len(head) < 4:
                continue
            if head in words:
                out.append(f"{lang} «{head}» ({n} в корпусе, отдельное слово)")
            elif lang == "RU" and head in term:
                out.append(f"{lang} «{head}» ({n} в корпусе, внутри сложного слова)")
    return sorted(set(out))


# --- основной проход ---------------------------------------------------------

def collect(seeds: list[str]) -> list[dict]:
    env = read_env()
    token = get_token(env)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    seen_records: set[str] = set()
    rows: list[dict] = []
    failed: list[tuple] = []

    for seed in seeds:
        print(f"[{seed}]", end=" ", flush=True)
        # offset у этого API — НОМЕР СТРАНИЦЫ с 1, а размер страницы задаёт limit
        page, fetched, max_pages = 1, 0, None
        while True:
            params = {
                "term": seed, "sourceLanguages": "EN", "targetLanguages": "RU",
                "subjectFields": SUBJECT_FIELDS, "exactSearch": "false",
                "limit": PAGE, "offset": page,
            }
            try:
                data = api_get(token, params)
            except urllib.error.HTTPError as e:
                print(f"(HTTP {e.code})", end=" ")
                break
            except Exception as e:
                # семейство пропускается, но сбор в целом продолжается
                print(f"(пропущено: {type(e).__name__})", end=" ")
                failed.append((seed, page, str(e)[:120]))
                break
            raw_name = f"{seed}_p{page:02d}.json"
            (RAW_DIR / raw_name).write_text(
                json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

            if max_pages is None:
                try:
                    max_pages = int(data.get("maxPages") or 1)
                except ValueError:
                    max_pages = 1
            records = data.get("Record") or []
            if not records:
                break

            for rec in records:
                sf = rec.get("subjectField") or {}
                en_items, ru_items = pick_terms(rec, "EN"), pick_terms(rec, "RU")
                if not en_items or not ru_items:
                    continue
                en, ru = en_items[0], ru_items[0]
                key = f"{en['term_id']}|{ru['term_id']}"
                if key in seen_records:
                    continue
                seen_records.add(key)

                en_t, ru_t = en["term"], ru["term"].lower()
                bad_en, bad_ru = reject_reason(en_t, "en"), reject_reason(ru_t, "ru")
                enw, ruw = len(en_t.split(" ")), len(ru_t.split(" "))
                st = stratum_of(enw, ruw)
                rating = min(en["rating"], ru["rating"])

                notes = []
                status = "ACCEPT"
                if bad_en or bad_ru:
                    status = "EXCLUDE"
                    notes.append("; ".join(x for x in (
                        f"EN: {bad_en}" if bad_en else "", f"RU: {bad_ru}" if bad_ru else "") if x))
                elif st == "?":
                    status = "EXCLUDE"
                    notes.append(f"структура вне групп A/B/C (EN {enw}, RU {ruw})")
                else:
                    if en_t.lower() in corpus["en"] or ru_t in corpus["ru"]:
                        status = "EXCLUDE"
                        notes.append("термин уже есть в корпусе")
                    if rating < 3:
                        status = "RESERVE" if status == "ACCEPT" else status
                        notes.append(f"надёжность {rating} < 3")
                    if en["usage"] != "allowed" or ru["usage"] != "allowed":
                        status = "RESERVE" if status == "ACCEPT" else status
                        notes.append(f"usage EN={en['usage']}, RU={ru['usage']}")
                    hits = family_hits(en_t, ru_t, corpus)
                    if hits:
                        notes.append("пересечение с семьями корпуса: " + "; ".join(hits))

                rows.append({
                    "record_id": en["term_id"],
                    "ru_term_id": ru["term_id"],
                    "EN": en_t, "RU": ru_t,
                    "EN_words": enw, "RU_words": ruw,
                    "stratum": st,
                    "subject_field": sf.get("code", ""),
                    "subject_description": sf.get("description", ""),
                    "reliability": rating,
                    "usage_en": en["usage"], "usage_ru": ru["usage"],
                    "source_en": en["source"], "source_ru": ru["source"],
                    "context_ru": ru["context"],
                    "context_en": en["context"],
                    "en_synonyms": " | ".join(x["term"] for x in en_items[1:4]),
                    "ru_synonyms": " | ".join(x["term"].lower() for x in ru_items[1:4]),
                    "seed": seed,
                    "status": status,
                    "notes": "; ".join(n for n in notes if n),
                    "raw_file": f"WIPO_PEARL/raw/{raw_name}",
                    "access_date": stamp,
                })

            fetched += len(records)
            page += 1
            if page > (max_pages or 1) or fetched >= MAX_RECORDS_PER_SEED:
                break
            time.sleep(PAUSE)
        print(f"{fetched} записей", flush=True)
    if failed:
        print(f"\nне удалось получить: {len(failed)} запросов")
        for seed, page, err in failed[:10]:
            print(f"   {seed} стр. {page}: {err}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", default=None)
    args = ap.parse_args()
    seeds = args.seeds or SEEDS

    rows = collect(seeds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cols = list(rows[0].keys()) if rows else []
    out_csv = OUT_DIR / "wipo_candidates.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    by = {}
    for r in rows:
        by[(r["stratum"], r["status"])] = by.get((r["stratum"], r["status"]), 0) + 1
    print(f"\nвсего пар: {len(rows)}")
    for st in ("A", "B", "C", "?"):
        line = ", ".join(f"{s}={by.get((st, s), 0)}" for s in ("ACCEPT", "RESERVE", "EXCLUDE"))
        print(f"  группа {st}: {line}")
    print(f"\nтаблица: {out_csv.relative_to(ROOT)}")
    print(f"сырые ответы API: {RAW_DIR.relative_to(ROOT)} "
          f"({len(list(RAW_DIR.glob('*.json')))} файлов)")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
