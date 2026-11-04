# -*- coding: utf-8 -*-
"""Сбор дополнительных C-ориентированных терминов WIPO Pearl."""
import urllib.request, urllib.parse, json, time, re
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.wipo_collect import read_env, get_token, API, SUBJECT_FIELDS

ROOT = Path(__file__).resolve().parents[1]
raw_dir = ROOT / "expansion_wipo_v1" / "raw" / "wipo"
raw_dir.mkdir(parents=True, exist_ok=True)

env = read_env()
token = get_token(env)

queries = [
    'level gauge', 'moisture meter', 'gate valve', 'pneumatic valve',
    'hydraulic valve', 'terminal block', 'suction cup', 'heat exchanger',
    'spark gap', 'slip ring', 'guide rail', 'ball screw', 'thickness gauge'
]

print("Saving compound C terms to expansion_wipo_v1/raw/wipo...")
for q in queries:
    slug = re.sub(r"[^a-z0-9]+", "_", q.lower()).strip("_")
    out_file = raw_dir / f"{slug}_p01.json"
    if out_file.exists():
        print(f"  {q}: already exists ({out_file.name})")
        continue
    params = {
        "term": q,
        "sourceLanguages": "EN",
        "targetLanguages": "RU",
        "subjectFields": SUBJECT_FIELDS,
        "exactSearch": "false",
        "limit": 20,
        "offset": 1
    }
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + token, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
            out_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            hits = int(data.get("hits") or 0)
            recs = len(data.get("Record", []))
            print(f"  {q}: hits={hits}, records={recs} -> saved {out_file.name}")
        time.sleep(0.6)
    except Exception as e:
        print(f"  [ERROR] {q}: {e}")

print("Done.")
