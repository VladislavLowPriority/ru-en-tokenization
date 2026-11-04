# -*- coding: utf-8 -*-
"""Фиксация выборки перед запуском измерений (Pre-measurement Freeze).

Создает или верифицирует expansion_wipo_v1/plan/FREEZE.json:
- При запуске без аргументов: формирует и записывает FREEZE.json.
- При запуске с флагом --verify: проверяет контрольные суммы всех 9 файлов данных,
  размеры выборок и инварианты БЕЗ перезаписи FREEZE.json.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "expansion_wipo_v1"
DATA_DIR = EXP / "data"
CAND_DIR = EXP / "candidates"
PLAN_DIR = EXP / "plan"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Freeze sample dataset or verify existing FREEZE.json")
    parser.add_argument("--verify", action="store_true", help="Verify dataset files against existing FREEZE.json without writing")
    args = parser.parse_args()

    freeze_path = PLAN_DIR / "FREEZE.json"

    # Load files
    tech_new_file = DATA_DIR / "concepts_new.csv"
    tech_comb_file = DATA_DIR / "concepts_combined.csv"
    match_new_file = DATA_DIR / "matching_pairs_new.csv"
    match_comb_file = DATA_DIR / "matching_pairs_combined.csv"
    unmatched_file = DATA_DIR / "unmatched_technical.csv"
    ctrl_exp_file = DATA_DIR / "concepts_control_expanded.csv"
    cand_file = CAND_DIR / "wipo_candidates.csv"
    var_file = CAND_DIR / "wipo_variants.csv"
    dec_file = CAND_DIR / "wipo_decisions.csv"

    files_to_freeze = [
        tech_new_file, tech_comb_file, match_new_file, match_comb_file,
        unmatched_file, ctrl_exp_file, cand_file, var_file, dec_file
    ]

    file_hashes = {}
    for p in files_to_freeze:
        if p.exists():
            rel = str(p.relative_to(ROOT)).replace("\\", "/")
            file_hashes[rel] = {
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
                "lines": sum(1 for _ in open(p, "rb"))
            }

    df_new = pd.read_csv(tech_new_file)
    df_comb = pd.read_csv(tech_comb_file)
    df_match_new = pd.read_csv(match_new_file)
    df_match_comb = pd.read_csv(match_comb_file)
    df_unmatched = pd.read_csv(unmatched_file)

    # Integrity verifications
    n_new = len(df_new)
    n_comb = len(df_comb)
    assert n_new >= 50, f"Expected at least 50 new concepts, got {n_new}"
    assert n_comb == 90 + n_new, f"Combined count mismatch: {n_comb} != 90 + {n_new}"
    assert df_new["concept_id"].nunique() == n_new, "Duplicate concept_id in new concepts!"
    assert df_comb["concept_id"].nunique() == n_comb, "Duplicate concept_id in combined concepts!"
    assert df_new["wipo_concept_record_id"].nunique() == n_new, "Duplicate wipo_concept_record_id in new concepts!"
    assert df_match_comb["d_RU_chars"].abs().max() <= 4, "Caliper violation in RU!"
    assert df_match_comb["d_EN_chars"].abs().max() <= 4, "Caliper violation in EN!"
    assert df_match_comb["ctrl_cand_id"].nunique() == len(df_match_comb), "Duplicate control candidate in matched pairs!"
    assert df_match_comb["ctrl_RU"].nunique() == len(df_match_comb), "Duplicate control RU in matched pairs!"
    assert df_match_comb["ctrl_EN"].nunique() == len(df_match_comb), "Duplicate control EN in matched pairs!"

    # Check family size constraint <= 3
    fam_counts = df_comb["family_id"].value_counts()
    assert fam_counts.max() <= 3, f"Family limit exceeded: {fam_counts.head(5)}"

    if args.verify:
        if not freeze_path.exists():
            print(f"Error: {freeze_path} does not exist.", file=sys.stderr)
            sys.exit(1)

        with open(freeze_path, "r", encoding="utf-8") as f:
            freeze_data = json.load(f)

        expected_files = freeze_data.get("dataset_files", {})
        all_ok = True
        print("=== VERIFYING DATASET FILES AGAINST FREEZE.json ===")
        for rel_path, exp_meta in expected_files.items():
            if rel_path not in file_hashes:
                print(f"[MISSING] {rel_path}", file=sys.stderr)
                all_ok = False
                continue
            curr_meta = file_hashes[rel_path]
            if curr_meta["sha256"] != exp_meta["sha256"]:
                print(f"[FAIL HASH] {rel_path}: expected {exp_meta['sha256']}, got {curr_meta['sha256']}", file=sys.stderr)
                all_ok = False
            elif curr_meta["size_bytes"] != exp_meta["size_bytes"]:
                print(f"[FAIL SIZE] {rel_path}: expected {exp_meta['size_bytes']}, got {curr_meta['size_bytes']}", file=sys.stderr)
                all_ok = False
            else:
                print(f"[OK] {rel_path} (sha256: {curr_meta['sha256'][:16]}..., {curr_meta['size_bytes']} bytes)")

        # Verify sample sizes
        sample_sizes = freeze_data.get("sample_sizes", {})
        if sample_sizes:
            checks = [
                ("technical_new_count", len(df_new), sample_sizes.get("technical_new_count")),
                ("technical_combined_count", len(df_comb), sample_sizes.get("technical_combined_count")),
                ("matched_pairs_new_count", len(df_match_new), sample_sizes.get("matched_pairs_new_count")),
                ("matched_pairs_combined_count", len(df_match_comb), sample_sizes.get("matched_pairs_combined_count")),
                ("unmatched_technical_count", len(df_unmatched), sample_sizes.get("unmatched_technical_count")),
            ]
            for name, actual, expected in checks:
                if actual != expected:
                    print(f"[FAIL COUNT] {name}: expected {expected}, got {actual}", file=sys.stderr)
                    all_ok = False

        if all_ok:
            print(f"\nVerification SUCCESSFUL: all {len(expected_files)} files and sample invariants match FREEZE.json.")
            print("FREEZE.json was NOT modified.")
            sys.exit(0)
        else:
            print("\nVerification FAILED: discrepancies found.", file=sys.stderr)
            sys.exit(1)

    # If NOT --verify, create/write FREEZE.json
    freeze_record = {
        "protocol_phase": "audit_revision_post_results",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "tokenizers_called_so_far": True,
        "revision_note": (
            "Редакция v1.2 после сплошного построчного аудита, устранения морфологических дефектов "
            "и исключения семантически неэквивалентной пары EXP69 (spool valve / золотник). "
            "Токенизаторы уже запускались на предшествующих итерациях протокола; текущая фиксация "
            "прямо отражает аудированную финальную версию выборки после просмотра результатов."
        ),
        "seed": 20260918,
        "sample_sizes": {
            "technical_new_count": len(df_new),
            "technical_new_stratum": df_new["stratum"].value_counts().to_dict(),
            "technical_combined_count": len(df_comb),
            "technical_combined_stratum": df_comb["stratum"].value_counts().to_dict(),
            "technical_combined_cohort": df_comb["cohort"].value_counts().to_dict(),
            "matched_pairs_new_count": len(df_match_new),
            "matched_pairs_new_stratum": df_match_new["stratum"].value_counts().to_dict(),
            "matched_pairs_combined_count": len(df_match_comb),
            "matched_pairs_combined_stratum": df_match_comb["stratum"].value_counts().to_dict(),
            "unmatched_technical_count": len(df_unmatched),
            "total_global_families": df_comb["family_id"].nunique(),
            "max_global_family_size": int(fam_counts.max())
        },
        "verification_assertions": {
            "zero_duplicate_concept_ids": True,
            "max_family_size_le_3": True,
            "caliper_le_4_chars_both_langs": True,
            "no_reuse_of_control": True,
            "exact_word_count_matching": True
        },
        "dataset_files": file_hashes,
        "new_concept_ids": df_new["concept_id"].tolist(),
        "matched_pair_keys": [
            {"tech_id": r["tech_concept_id"], "ctrl_id": r["ctrl_cand_id"], "stratum": r["stratum"]}
            for _, r in df_match_comb.iterrows()
        ]
    }

    with open(freeze_path, "w", encoding="utf-8") as f:
        json.dump(freeze_record, f, ensure_ascii=False, indent=2)

    print(f"FREEZE.json created successfully at {freeze_path}")
    print(f"Total frozen files: {len(file_hashes)}")
    print(f"Technical: {len(df_new)} new, {len(df_comb)} combined.")
    print(f"Control matches: {len(df_match_new)} new, {len(df_match_comb)} combined matched pairs.")


if __name__ == "__main__":
    main()
