# -*- coding: utf-8 -*-
"""Модульные тесты новой логики расширения выборки (expansion_wipo_v1).

Проверяет требования раздела 11 ПРОМПТ_РАСШИРЕНИЕ_WIPO.md и устранение всех выявленных дефектов:
1. Очистка HTML: теги внутри слова не разрывают слово.
2. Различение concept_record_id, en_term_id, ru_term_id.
3. Транзитивное объединение семей через любой язык и соблюдение лимита <= 3.
4. Отсутствие повторного использования контроля и отсутствие воскресших исключений.
5. Соблюдение калипера длины <= 4 знака в обоих языках.
6. Математическая эквивалентность двух способов вычисления R.
7. Проблема 1: Корректная передача семей контроля (связанность coffee table / dressing table, отсутствие фиктивных CFAM_base_, ~110 совместных кластеров).
8. Проблема 2: Исключение не-устройств (нет функций, процессов, измерений, связей).
9. Проблема 3: Фильтр аббревиатур не бракует обычные английские слова (motor, valve, pump, laser, gear, lamp).
10. Проблема 4: Строго только первичные обозначения (is_primary == True, без variant shopping).
11. Верификация базовых референсных значений (M = -0.0667, H3 = 0.0).
"""
import unittest
import sys
import numpy as np
import pandas as pd
import json
import re
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXP = ROOT / "expansion_wipo_v1"
DATA_DIR = EXP / "data"
RAW_DIR = EXP / "raw"

class TestExpansionLogic(unittest.TestCase):

    def test_html_cleaning_preserves_internal_tags(self):
        def clean_html(s: str) -> str:
            s = re.sub(r"<(?:p|div|br|tr|td|li|h[1-6])[^>]*>", " ", s, flags=re.IGNORECASE)
            s = re.sub(r"</(?:p|div|tr|td|li|h[1-6])>", " ", s, flags=re.IGNORECASE)
            s = re.sub(r"<[^>]+>", "", s)
            s = html.unescape(s)
            s = re.sub(r"\s+", " ", s)
            return s.strip()

        self.assertEqual(clean_html("<p style='...'><b>encoder</b></p>"), "encoder")
        self.assertEqual(clean_html("opt<b>o</b>coupler"), "optocoupler")
        self.assertEqual(clean_html("серво<b>привод</b>"), "сервопривод")
        self.assertEqual(clean_html("pressure<br>sensor"), "pressure sensor")
        self.assertEqual(clean_html("<p>flow</p><p>meter</p>"), "flow meter")

    def test_r_formula_equivalence(self):
        tech_ru = np.array([4.0, 6.0, 3.0, 5.0])
        tech_en = np.array([2.0, 3.0, 1.0, 2.0])
        ctrl_ru = np.array([3.0, 5.0, 2.0, 4.0])
        ctrl_en = np.array([2.0, 2.0, 1.0, 2.0])

        d_i = np.log(tech_ru / tech_en) - np.log(ctrl_ru / ctrl_en)
        R1 = float(np.exp(np.mean(d_i)))

        G_tech = float(np.exp(np.mean(np.log(tech_ru / tech_en))))
        G_ctrl = float(np.exp(np.mean(np.log(ctrl_ru / ctrl_en))))
        R2 = float(G_tech / G_ctrl)

        self.assertAlmostEqual(R1, R2, places=9)
        self.assertAlmostEqual(R1, np.exp(d_i.mean()), places=9)

    def test_transitive_family_limit_combined(self):
        comb_file = DATA_DIR / "concepts_combined.csv"
        self.assertTrue(comb_file.exists(), "concepts_combined.csv not found")
        df = pd.read_csv(comb_file)
        self.assertEqual(len(df), 159, "Combined corpus must have exactly 159 concepts (90 base + 69 new)")
        
        fam_counts = df["family_id"].value_counts()
        self.assertLessEqual(fam_counts.max(), 3, f"Family limit <= 3 violated: max={fam_counts.max()}")

    def test_control_matching_invariants(self):
        match_file = DATA_DIR / "matching_pairs_combined.csv"
        self.assertTrue(match_file.exists(), "matching_pairs_combined.csv not found")
        df = pd.read_csv(match_file)
        
        for _, r in df.iterrows():
            w_ru_t = len(str(r["tech_RU"]).split())
            w_ru_c = len(str(r["ctrl_RU"]).split())
            w_en_t = len(str(r["tech_EN"]).split())
            w_en_c = len(str(r["ctrl_EN"]).split())
            self.assertEqual(w_ru_t, w_ru_c, f"RU word count mismatch in pair {r['tech_concept_id']}")
            self.assertEqual(w_en_t, w_en_c, f"EN word count mismatch in pair {r['tech_concept_id']}")

        self.assertLessEqual(df["d_RU_chars"].abs().max(), 4, "RU character caliper violated")
        self.assertLessEqual(df["d_EN_chars"].abs().max(), 4, "EN character caliper violated")
        self.assertEqual(df["ctrl_cand_id"].nunique(), len(df), "Duplicate control candidate IDs detected!")
        self.assertEqual(df["ctrl_RU"].str.strip().str.lower().nunique(), len(df), "Duplicate control RU detected!")
        self.assertEqual(df["ctrl_EN"].str.strip().str.lower().nunique(), len(df), "Duplicate control EN detected!")

    def test_problem1_control_families_transmission(self):
        """Проблема 1: Семьи контроля не должны иметь фиктивных CFAM_base_, а понятия с общими вершинами должны объединяться."""
        match_comb = pd.read_csv(DATA_DIR / "matching_pairs_combined.csv")

        # 1. Никаких фиктивных CFAM_base_
        base_fams = match_comb[match_comb["ctrl_family_id"].str.startswith("CFAM_base_")]
        self.assertEqual(len(base_fams), 0, f"Found {len(base_fams)} dummy CFAM_base_ family IDs!")

        # 2. Связывавание понятий с общей вершиной: coffee table и dressing table
        coffee = match_comb[match_comb["ctrl_EN"].str.lower() == "coffee table"]
        dressing = match_comb[match_comb["ctrl_EN"].str.lower() == "dressing table"]
        if len(coffee) > 0 and len(dressing) > 0:
            self.assertEqual(coffee.iloc[0]["ctrl_family_id"], dressing.iloc[0]["ctrl_family_id"],
                             "coffee table and dressing table must share ctrl_family_id!")

        # 3. Совместные кластеры ~110
        parent = {}
        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb: parent[rb] = ra

        comb_tech = pd.read_csv(DATA_DIR / "concepts_combined.csv").set_index("concept_id")
        for _, r in match_comb.iterrows():
            t_fam = comb_tech.loc[r["tech_concept_id"], "family_id"]
            c_fam = r["ctrl_family_id"]
            union("T:" + str(t_fam), "C:" + str(c_fam))

        clusters = set(find("T:" + str(comb_tech.loc[r["tech_concept_id"], "family_id"])) for _, r in match_comb.iterrows())
        self.assertLessEqual(len(clusters), 115, f"Joint clusters inflated: {len(clusters)}")
        self.assertGreaterEqual(len(clusters), 105, f"Joint clusters too low: {len(clusters)}")

    def test_problem2_no_non_devices(self):
        """Проблема 2: В технической выборке не должно быть математических функций, процессов, измерений, связей."""
        new_file = DATA_DIR / "concepts_new.csv"
        df_new = pd.read_csv(new_file)
        
        forbidden_substrings = [
            "cumulative distribution", "кумулятивная функция",
            "electricity metering", "измерение электропотребления",
            "capacitive coupling", "ёмкостная связь", "емкостная связь",
            "distribution function", "функция распределения",
            "induction heating", "индукционный нагрев",
            "spool valve", "золотник"
        ]
        for _, r in df_new.iterrows():
            en = str(r["term_en_sg"]).lower()
            ru = str(r["term_ru_sg"]).lower()
            for sub in forbidden_substrings:
                self.assertNotIn(sub, en, f"Non-device '{sub}' found in {r['concept_id']}")
                self.assertNotIn(sub, ru, f"Non-device '{sub}' found in {r['concept_id']}")

    def test_problem3_abbreviation_filter_accuracy(self):
        """Проблема 3: Фильтр аббревиатур не должен браковать обычные английские слова (motor, valve, pump, laser, gear, lamp)."""
        from scripts.expansion_build_technical import is_abbreviation

        # Обычные английские слова должны проходить (False)
        valid_words = ["motor", "valve", "pump", "laser", "relay", "diode", "prism", "gear", "lamp", "spindle"]
        for w in valid_words:
            self.assertFalse(is_abbreviation(w, f"<B>{w}</B>", "en"), f"Valid word '{w}' falsely flagged as abbreviation!")

        # Реальные аббревиатуры должны браковаться (True)
        bad_words_en = ["plc", "cnc", "pid", "hmi", "pcb", "ic", "led", "ram", "rom", "scada"]
        for w in bad_words_en:
            self.assertTrue(is_abbreviation(w, w.upper(), "en"), f"Real abbrev '{w}' not flagged!")

        bad_words_ru = ["плк", "чпу", "кип", "асу", "пзу", "озу", "уси", "гтд"]
        for w in bad_words_ru:
            self.assertTrue(is_abbreviation(w, w.upper(), "ru"), f"Real abbrev '{w}' not flagged!")

    def test_problem4_strict_primary_designations(self):
        """Проблема 4: Все новые понятия должны быть только основными обозначениями (is_primary == True)."""
        df_new = pd.read_csv(DATA_DIR / "concepts_new.csv")
        df_var = pd.read_csv(EXP / "candidates" / "wipo_variants.csv")
        
        for _, r in df_new.iterrows():
            en_tid = r["wipo_en_term_id"]
            ru_tid = r["wipo_ru_term_id"]
            
            en_match = df_var[df_var["term_id"] == en_tid]
            ru_match = df_var[df_var["term_id"] == ru_tid]
            
            if len(en_match) > 0:
                self.assertTrue(bool(en_match.iloc[0]["is_primary"]), f"EN term for {r['concept_id']} is not primary!")
            if len(ru_match) > 0:
                self.assertTrue(bool(ru_match.iloc[0]["is_primary"]), f"RU term for {r['concept_id']} is not primary!")

    def test_problem5_syntactic_head_devices_pass(self):
        """Проблема 1: Проверка вершинного фильтра — приборы с зависимыми существительными проходят, процессы бракуются."""
        from scripts.expansion_build_technical import extract_head_en, extract_head_ru, NON_DEV_EN, NON_DEV_RU

        # Валидные устройства с модификаторами величин должны проходить
        valid_pairs = [
            ("pressure sensor", "датчик давления"),
            ("voltage regulator", "регулятор напряжения"),
            ("temperature transmitter", "преобразователь температуры"),
            ("flow meter", "расходомер"),
        ]
        for en, ru in valid_pairs:
            h_en = extract_head_en(en)
            h_ru = extract_head_ru(ru)
            self.assertNotIn(h_en, NON_DEV_EN, f"Head '{h_en}' of '{en}' falsely rejected!")
            self.assertNotIn(h_ru, NON_DEV_RU, f"Head '{h_ru}' of '{ru}' falsely rejected!")

        # Процессы отсекаются по вершине
        bad_process_en = "induction heating"
        bad_process_ru = "индукционный нагрев"
        self.assertIn(extract_head_en(bad_process_en), NON_DEV_EN, "Process 'heating' must be rejected!")
        self.assertIn(extract_head_ru(bad_process_ru), NON_DEV_RU, "Process 'нагрев' must be rejected!")

    def test_problem6_plural_morphology(self):
        """Проблема 2: Проверка корректности форм множественного числа (виток -> витки, термоусаживающиеся трубы, тормоза)."""
        df_new = pd.read_csv(DATA_DIR / "concepts_new.csv")
        
        # 1. EXP04: виток -> витки
        v_row = df_new[df_new["term_ru_sg"] == "виток"]
        self.assertGreater(len(v_row), 0)
        self.assertEqual(v_row.iloc[0]["term_ru_pl"], "витки", "виток plural must be 'витки', not 'витоки'!")
        
        # 2. Тормоз -> тормоза
        b_row = df_new[df_new["term_ru_sg"] == "электромагнитный тормоз"]
        if len(b_row) > 0:
            self.assertEqual(b_row.iloc[0]["term_ru_pl"], "электромагнитные тормоза")
            
        # 3. Морфологическая функция для возвратных причастий
        from scripts.expansion_build_technical import make_plurals
        en_pl, ru_pl, *_ = make_plurals("heat shrink tubing", "термоусаживающаяся труба", 3, 2)
        self.assertEqual(ru_pl, "термоусаживающиеся трубы", "Reflexive participle must inflect to -иеся!")

    def test_freeze_manifest_hashes_exist(self):
        freeze_file = EXP / "plan" / "FREEZE.json"
        self.assertTrue(freeze_file.exists(), "FREEZE.json not found")
        data = json.loads(freeze_file.read_text(encoding="utf-8"))
        
        self.assertEqual(data["sample_sizes"]["technical_new_count"], 69)
        self.assertEqual(data["sample_sizes"]["technical_combined_count"], 159)
        self.assertEqual(data["sample_sizes"]["matched_pairs_combined_count"], 131)
        self.assertTrue(data.get("tokenizers_called_so_far", False), "tokenizers_called_so_far must be True for post-results revision!")
        self.assertEqual(data.get("protocol_phase"), "audit_revision_post_results")
        self.assertTrue(data["verification_assertions"]["max_family_size_le_3"])
        self.assertTrue(data["verification_assertions"]["caliper_le_4_chars_both_langs"])
        self.assertTrue(data["verification_assertions"]["no_reuse_of_control"])

    def test_bug5_baseline_estimates_truth(self):
        """Bug 5: Baseline estimates in results/real/estimates.json must match ground truth."""
        est_file = ROOT / "results" / "real" / "estimates.json"
        self.assertTrue(est_file.exists())
        est = json.loads(est_file.read_text(encoding="utf-8"))
        
        o200k = est["estimates"]["o200k_base"]
        m_point = o200k["H2_M"]["point"]
        self.assertAlmostEqual(m_point, -0.06666666666666667, places=4)
        
        h3_point = o200k["H3_context_shift_both"]["point"]
        self.assertEqual(h3_point, 0.0)


if __name__ == "__main__":
    unittest.main()
