"""Regression and context-audit checks with known answers; no network needed."""
from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tok_pilot import config, metrics, validate


def regression_data():
    rng = np.random.default_rng(73)
    n = 30
    f = rng.uniform(1, 6, (n, 2))
    c = np.linspace(-0.7, 1.1, n)
    w = rng.integers(-1, 2, n)
    z = (f - f.mean(axis=0)) / f.std(axis=0, ddof=1)
    y = 0.6 + 0.45 * c + 0.12 * w - 0.09 * (z[:, 0] - z[:, 1])
    return pd.DataFrame({
        "concept_id": [f"p{i}" for i in range(n)],
        "family_id": [f"f{i // 2}" for i in range(n)],
        "F_ru_sg": f[:, 0], "F_en_sg": f[:, 1],
        "Fmiss_ru_sg": 0, "Fmiss_en_sg": 0,
        "T_ru_sg_iso0": np.exp(y), "T_en_sg_iso0": 1.0,
        "C_ru_sg": np.exp(c) * 10, "C_en_sg": 10.0,
        "W_ru_sg": w + 2, "W_en_sg": 2,
    })


def context_data():
    rows, stimuli = [], []
    for j, mode in enumerate(["isolated_space"] + [f"sentence_{t}" for t in config.ALL_TEMPLATES]):
        start = 1 + 10 * j
        hits = [{"i": 0, "id": 101, "span": [start - 1, start + 1], "cross": False},
                {"i": 1, "id": 102, "span": [start + 1, start + 2], "cross": False}]
        rows.append({"stimulus_id": mode, "concept_id": "p1", "lang": "en", "number": "sg",
                     "tokenizer_id": "test", "mode": mode, "error_code": "",
                     "term_byte_start": start, "term_byte_end": start + 2,
                     "target_tokens": json.dumps(hits), "token_ids": "[101, 102]",
                     "token_spans": json.dumps([h["span"] for h in hits]), "n_tokens_target": 2})
        stimuli.append({"stimulus_id": mode, "target_text": "ab"})
    return pd.DataFrame(rows), pd.DataFrame(stimuli)


class RegressionTests(unittest.TestCase):
    def test_known_coefficients_and_reproducibility(self):
        d = regression_data()
        a = metrics.sensitivity_regression(d, reps=100)
        b = metrics.sensitivity_regression(d, reps=100)
        self.assertEqual(a, b)
        self.assertEqual(a["status"], "ok")
        np.testing.assert_allclose([a[k] for k in ("beta0", "beta_C", "beta_W", "beta_F")],
                                   [0.6, 0.45, 0.12, -0.09], atol=1e-12)
        self.assertEqual(a["reps_used"], 100)
        self.assertAlmostEqual(a["unadjusted_G_same_subset"]["point"], metrics.stat_G(d))

    def test_partial_and_missing_pairs_are_excluded_from_primary(self):
        d = regression_data()
        extra = d.iloc[:2].copy()
        extra["concept_id"] = ["partial", "missing"]
        extra["Fmiss_ru_sg"] = [1, 2]
        extra.loc[extra.index[1], "F_ru_sg"] = np.nan
        extra["T_ru_sg_iso0"] = 1e9
        combined = pd.concat([d, extra], ignore_index=True)
        a = metrics.sensitivity_regression(combined, reps=100)
        baseline = metrics.sensitivity_regression(d, reps=100)
        self.assertEqual(a["coefficients"], baseline["coefficients"])
        self.assertEqual(a["excluded_concept_ids"], ["partial", "missing"])
        self.assertEqual(a["n_partial_used"], 0)
        b = metrics.sensitivity_regression(combined, coverage="available", reps=100)
        self.assertEqual(b["n_used"], 31)
        self.assertEqual(b["n_complete"], 30)
        self.assertEqual(b["n_partial_used"], 1)
        self.assertNotAlmostEqual(b["beta0"], a["beta0"])

    def test_families_resampled_whole_and_with_multiplicity(self):
        d = regression_data()
        original = np.linalg.lstsq
        observed = []
        def observe(x, y, **kwargs):
            observed.append(x[:, 1].copy())
            return original(x, y, **kwargs)
        with patch.object(metrics.np.linalg, "lstsq", side_effect=observe):
            metrics.sensitivity_regression(d, reps=40)
        coordinates = np.log(d.C_ru_sg / d.C_en_sg).to_numpy()
        repeats_seen = False
        for draw in observed[1:]:
            counts = np.array([np.isclose(draw, c).sum() for c in coordinates])
            np.testing.assert_array_equal(counts[::2], counts[1::2])
            repeats_seen |= counts.max() > 1
        self.assertTrue(repeats_seen)

    def test_singular_model_is_not_reported_as_success(self):
        d = regression_data()
        d["W_ru_sg"] = d["W_en_sg"]
        a = metrics.sensitivity_regression(d, reps=20)
        self.assertEqual(a["status"], "rank_deficient")
        self.assertNotIn("coefficients", a)

    def test_zero_frequency_variance_is_not_silently_fitted(self):
        d = regression_data()
        d["F_ru_sg"] = 2
        self.assertEqual(metrics.sensitivity_regression(d, reps=20)["status"], "zero_frequency_variance")

    def test_duplicate_concepts_rejected(self):
        d = regression_data()
        with self.assertRaises(ValueError):
            metrics.sensitivity_regression(pd.concat([d, d.iloc[:1]]), reps=20)

    def test_failed_draws_are_counted_and_unreliable_interval_withheld(self):
        d = regression_data().iloc[:8].copy()
        d["family_id"] = ["a"] * 4 + ["b"] * 4
        d["F_ru_sg"] = [1.0] * 4 + [2.0] * 4
        a = metrics.sensitivity_regression(d, reps=100)
        self.assertEqual(a["status"], "insufficient_valid_bootstrap_draws")
        self.assertGreater(a["failed_draws"]["zero_frequency_variance"], 0)
        self.assertEqual(a["reps_used"] + sum(a["failed_draws"].values()), 100)
        self.assertNotIn("coefficients", a)


class ContextTests(unittest.TestCase):
    def test_absolute_offset_shift_does_not_change_identity(self):
        m, st = context_data()
        out = validate.context_token_comparisons(m, st)
        self.assertTrue(out.exact_match.all())
        self.assertEqual(len(out), 3)

    def test_same_count_different_ids_is_detected(self):
        m, st = context_data()
        hits = json.loads(m.loc[1, "target_tokens"])
        hits[0]["id"] = 999
        m.loc[1, "target_tokens"] = json.dumps(hits)
        m.loc[1, "token_ids"] = "[999, 102]"
        out = validate.context_token_comparisons(m, st).iloc[0]
        self.assertTrue(out.count_equal)
        self.assertFalse(out.ids_equal)
        self.assertFalse(out.exact_match)

    def test_same_ids_different_internal_boundaries_is_detected(self):
        m, st = context_data()
        hits = json.loads(m.loc[1, "target_tokens"])
        hits[0]["span"][1] -= 1
        hits[1]["span"][0] -= 1
        m.loc[1, "target_tokens"] = json.dumps(hits)
        m.loc[1, "token_spans"] = json.dumps([h["span"] for h in hits])
        out = validate.context_token_comparisons(m, st).iloc[0]
        self.assertTrue(out.ids_equal)
        self.assertFalse(out.within_term_spans_equal)

    def test_leading_space_is_distinguished_from_internal_segmentation(self):
        m, st = context_data()
        hits = json.loads(m.loc[1, "target_tokens"])
        hits[0]["span"][0] += 1
        m.loc[1, "target_tokens"] = json.dumps(hits)
        m.loc[1, "token_spans"] = json.dumps([h["span"] for h in hits])
        out = validate.context_token_comparisons(m, st).iloc[0]
        self.assertFalse(out.relative_spans_equal)
        self.assertTrue(out.within_term_spans_equal)

    def test_missing_context_cannot_pass_as_zero_mismatches(self):
        m, st = context_data()
        with self.assertRaises(ValueError):
            validate.context_token_comparisons(m.iloc[:-1], st)

    def test_mismatched_target_is_rejected(self):
        m, st = context_data()
        st.loc[1, "target_text"] = "cd"
        with self.assertRaises(ValueError):
            validate.context_token_comparisons(m, st)


if __name__ == "__main__":
    unittest.main(verbosity=2)
