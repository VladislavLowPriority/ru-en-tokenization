# -*- coding: utf-8 -*-
"""Сбор сырых ответов WIPO Pearl для каталога expansion_wipo_v1.

Соблюдает строгие правила ПРОМПТ_РАСШИРЕНИЕ_WIPO.md:
  1. Читает учетные данные из src/api.env в память, ни при каких условиях не логирует секреты.
  2. Запрашивает до 6 страниц (по 25 записей) на каждый сид из основного списка SEEDS
     и дополнительного резервного списка BACKUP_SEEDS.
  3. Использует локальный кэш: если файл уже существует, повторный сетевой запрос не делается.
  4. Сохраняет полный сырой JSON каждого ответа в expansion_wipo_v1/raw/wipo/.
  5. Ведёт структурированный журнал поиска search_log.jsonl и сводку SEARCH_BUDGET.json.
  6. Не производит токенизацию и не читает токенные длины.
"""
from __future__ import annotations

import base64
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
EXP_DIR = ROOT / "expansion_wipo_v1"
RAW_DIR = EXP_DIR / "raw" / "wipo"
LOG_FILE = RAW_DIR / "search_log.jsonl"
BUDGET_FILE = EXP_DIR / "plan" / "SEARCH_BUDGET.json"

API_URL = "https://api.wipo.int/wipopearl/records"
TOKEN_URL = "https://oauth2.api.wipo.int/external/oauth/token"
SUBJECT_FIELDS = "ELEC,MEAS,MANU,MECH"
PAGE_SIZE = 25
MAX_PAGES = 6
PAUSE = 0.40

# Основной список поисковых запросов из исходного SEEDS
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

# Резервный список запросов (не более 30 запросов, предметная область: промышленная автоматика/измерения/механика)
BACKUP_SEEDS = [
    "manifold", "cylinder", "fitting", "caliper", "tachometer", "pyrometer",
    "flowmeter", "manometer", "potentiometer", "inverter", "interrupter", "isolator",
    "interlock", "commutator", "distributor", "lubricator", "ejector", "separator",
    "atomizer", "calibrator", "densitometer", "micrometer", "anemometer", "spectrometer",
    "refractometer", "barometer", "altimeter", "dynamometer", "viscometer", "galvanometer"
]


def read_env() -> dict:
    out = {}
    if not ENV_FILE.exists():
        return out
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
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        new_token = json.loads(r.read())["access_token"]
    return new_token


def api_get(token: str, params: dict) -> dict:
    url = API_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url, headers={"Authorization": "Bearer " + token, "Accept": "application/json"}
    )
    last_err = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(2 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
            continue
    raise RuntimeError(f"WIPO API request failed after 5 attempts: {last_err}")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    env = read_env()
    token = None

    all_seeds = [("primary", s) for s in SEEDS] + [("backup", s) for s in BACKUP_SEEDS]
    # Дедупликация сидов с сохранением порядка
    seen_seeds = set()
    queue = []
    for queue_type, s in all_seeds:
        if s not in seen_seeds:
            seen_seeds.add(s)
            queue.append((queue_type, s))

    print(f"Total unique search seeds in queue: {len(queue)} (primary: {len(SEEDS)}, backup: {len(BACKUP_SEEDS)})")
    
    budget_records = []
    log_entries = []

    for idx, (qtype, seed) in enumerate(queue, 1):
        page = 1
        max_pages = 1
        seed_hits = 0
        seed_records = 0

        while page <= max_pages and page <= MAX_PAGES:
            out_file = RAW_DIR / f"{seed}_p{page:02d}.json"
            from_cache = False

            if out_file.exists():
                from_cache = True
                content = out_file.read_bytes()
                try:
                    data = json.loads(content)
                except Exception:
                    data = {}
                sha = sha256_bytes(content)
            else:
                if token is None:
                    token = get_token(env)
                params = {
                    "term": seed,
                    "sourceLanguages": "EN",
                    "targetLanguages": "RU",
                    "subjectFields": SUBJECT_FIELDS,
                    "exactSearch": "false",
                    "limit": PAGE_SIZE,
                    "offset": page,
                }
                try:
                    data = api_get(token, params)
                    content = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
                    out_file.write_bytes(content)
                    sha = sha256_bytes(content)
                    time.sleep(PAUSE)
                except Exception as ex:
                    print(f"  [ERROR] {seed} p{page}: {ex}")
                    break

            try:
                hits = int(data.get("hits") or 0)
            except (ValueError, TypeError):
                hits = 0
            try:
                max_p = int(data.get("maxPages") or 1)
            except (ValueError, TypeError):
                max_p = 1
            records_count = len(data.get("Record", []))
            
            if page == 1:
                max_pages = max_p
                seed_hits = hits
            seed_records += records_count

            log_entry = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "queue_type": qtype,
                "seed": seed,
                "page": page,
                "max_pages": max_pages,
                "hits": hits,
                "records_returned": records_count,
                "from_cache": from_cache,
                "filename": out_file.name,
                "sha256": sha
            }
            log_entries.append(log_entry)

            if records_count == 0 or page >= max_pages:
                break
            page += 1

        budget_records.append({
            "seed": seed,
            "queue_type": qtype,
            "hits": seed_hits,
            "pages_collected": page,
            "max_pages_available": max_pages,
            "total_records": seed_records
        })

        if idx % 10 == 0 or idx == len(queue):
            print(f"[{idx}/{len(queue)}] Processed seed '{seed}' ({qtype}): {seed_records} records, {page} pages.")

    # Сохраняем search_log.jsonl
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        for entry in log_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Сохраняем SEARCH_BUDGET.json
    budget_summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_seeds_planned": len(queue),
        "total_requests": len(log_entries),
        "cache_hits": sum(1 for e in log_entries if e["from_cache"]),
        "network_requests": sum(1 for e in log_entries if not e["from_cache"]),
        "seeds": budget_records
    }
    with open(BUDGET_FILE, "w", encoding="utf-8") as f:
        json.dump(budget_summary, f, ensure_ascii=False, indent=2)

    print(f"\nCollection finished successfully.")
    print(f"Requests logged: {len(log_entries)} (Cache: {budget_summary['cache_hits']}, Net: {budget_summary['network_requests']})")
    print(f"Log written to: {LOG_FILE}")
    print(f"Budget summary written to: {BUDGET_FILE}")


if __name__ == "__main__":
    main()
