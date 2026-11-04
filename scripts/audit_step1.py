"""Verify the frozen baseline and corrected outputs; rebuild historical subsets.

Run after run_pilot.py --split real. Historical subsets use CURRENT forms and
families; they must not be presented as recovered historical raw runs.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tok_pilot import metrics, validate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(baseline):
    baseline = baseline.resolve()
    frozen = json.loads((baseline / "SNAPSHOT_MANIFEST.json").read_text(encoding="utf-8"))
    for entry in frozen["files"]:
        assert digest(baseline / entry["path"]) == entry["sha256"], entry["path"]
    out = ROOT / "results" / "real"
    old = baseline / "results" / "real"
    concepts = pd.read_csv(ROOT / "data/real/concepts_pilot.csv")
    concepts = concepts[concepts.final_status.eq("accepted")].copy()
    unchanged = {}
    for name in ("stimuli.csv", "measurements.csv", "analysis_wide.csv"):
        a, b = pd.read_csv(old / name), pd.read_csv(out / name)
        added = sorted(set(b.columns) - set(a.columns))
        assert not set(a.columns) - set(b.columns), "An original column disappeared"
        if added:
            assert name == "analysis_wide.csv" and added == ["pl_status"], added
            expected = b.concept_id.map(concepts.set_index("concept_id").pl_status)
            pd.testing.assert_series_equal(b.pl_status, expected, check_names=False)
        pd.testing.assert_frame_equal(a, b[a.columns], check_exact=True)
        unchanged[name] = {"original_columns_identical": True, "added_columns": added, "n_rows": len(b),
                           "bytes_identical": digest(old / name) == digest(out / name)}
    assert digest(ROOT / "data/real/concepts_pilot.csv") == digest(baseline / "data/real/concepts_pilot.csv")
    p0 = (baseline / "protocol_ru_en_tokenization.md").read_text(encoding="utf-8")
    p1 = (ROOT / "protocol_ru_en_tokenization.md").read_text(encoding="utf-8")
    assert p1.startswith(p0.rstrip()), "Historical protocol changed instead of appending audit note"
    estimates = json.loads((out / "estimates.json").read_text(encoding="utf-8"))["estimates"]
    before = json.loads((old / "estimates.json").read_text(encoding="utf-8"))["estimates"]
    compared = 0
    for tid, e in before.items():
        for name, value in e.items():
            if name == "sensitivity_regression" or name.startswith("precision_projection_"):
                continue
            assert estimates[tid][name] == value, (tid, name)
            compared += 1
    w = pd.read_csv(out / "analysis_wide.csv")
    s = pd.read_csv(out / "stimuli.csv")
    m = pd.read_csv(out / "measurements.csv")
    context = validate.context_token_comparisons(m, s)
    coverage, regression = [], {}
    for tid, d in w.groupby("tokenizer_id"):
        complete = d.Fmiss_ru_sg.eq(0) & d.Fmiss_en_sg.eq(0) & d.F_ru_sg.notna() & d.F_en_sg.notna()
        available = d.F_ru_sg.notna() & d.F_en_sg.notna()
        if tid == "o200k_base":
            for _, row in d.iterrows():
                coverage.append({"concept_id": row.concept_id, "complete": bool(complete.loc[row.name]),
                                 "available_mean": bool(available.loc[row.name]),
                                 "missing_ru_components": row.Fmiss_ru_sg, "missing_en_components": row.Fmiss_en_sg})
        for policy, select in (("complete", complete), ("available", available)):
            fit = estimates[tid]["sensitivity_regression" if policy == "complete" else "sensitivity_regression_available_components"]
            a = d.loc[select]
            f = a[["F_ru_sg", "F_en_sg"]]
            z = (f - f.mean()) / f.std(ddof=1)
            x = np.array([np.ones(len(a)), np.log(a.C_ru_sg / a.C_en_sg),
                          a.W_ru_sg - a.W_en_sg, z.F_ru_sg - z.F_en_sg]).T
            # Independent linear-algebra route checks the reported point fit.
            reference = np.linalg.pinv(x) @ np.log(a.T_ru_sg_iso0 / a.T_en_sg_iso0)
            actual = np.array([fit[k] for k in ("beta0", "beta_C", "beta_W", "beta_F")])
            np.testing.assert_allclose(actual, reference, atol=1e-12, rtol=1e-12)
            assert fit["n_used"] == len(a) and fit["reps_used"] == 5000
            if policy == "available":
                np.testing.assert_allclose(actual, [before[tid]["sensitivity_regression"][k]
                                                   for k in ("beta0", "beta_C", "beta_W", "beta_F")], atol=1e-12)
            regression[f"{tid}:{policy}"] = {"n_used": len(a), "n_families": fit["n_families"],
                                           "matches_independent_pseudoinverse": True, "reps_used": fit["reps_used"]}
    slices = {}
    for n in (24, 40, 60, 90):
        ids = concepts.concept_id if n == 90 else concepts.loc[
            concepts.concept_id.str.fullmatch(r"P\d+") &
            (concepts.concept_id.str.extract(r"(\d+)", expand=False).astype(int) <= n), "concept_id"]
        assert len(ids) == n
        sample = w[w.concept_id.isin(ids)]
        c = concepts[concepts.concept_id.isin(ids)]
        pilot_ids = c.loc[c.source_en.str.startswith("Festo"), "concept_id"]
        per_tok = {}
        for tid, d in sample.groupby("tokenizer_id"):
            rest = d[~d.concept_id.isin(pilot_ids)]
            g = metrics.cluster_bootstrap(d, metrics.stat_G)
            per_tok[tid] = {"G": g, "M": metrics.stat_M(d),
                            "mode_ratio": metrics.stat_G_mode_gap(d),
                            "log_halfwidth": metrics.log_halfwidth(g),
                            "full_frequency_pairs": int((d.Fmiss_ru_sg.eq(0) & d.Fmiss_en_sg.eq(0)).sum()),
                            "n_excluded_festo": len(pilot_ids),
                            "without_festo_G": metrics.stat_G(rest) if len(rest) else None,
                            "delta_without_festo_G": metrics.stat_G(rest) - g["point"] if len(rest) else None}
        slices[str(n)] = {"concept_ids": ids.tolist(), "estimates": per_tok}
        print(f"Rebuilt subset {n}", flush=True)
    payload = {
        "baseline": str(baseline.relative_to(ROOT)), "snapshot_files_verified": len(frozen["files"]),
        "source_concepts_unchanged": True, "original_protocol_preserved_as_prefix": True,
        "unchanged_tables": unchanged, "unchanged_estimate_records": compared,
        "regression_verification": regression,
        "context_token_identity": validate.summarize_context_comparisons(context),
        "historical_subsets_note": "Reconstructed P01..P24/P40/P60 and all 90 from CURRENT corpus and families; not recovered original run snapshots. Historical rounded values are retained separately in НАХОДКИ.md.",
        "historical_subsets": slices,
        "scope": "Step 1 only. Human review, new control corpus and manuscript are deferred.",
    }
    (out / "audit_step1.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(coverage).merge(concepts[["concept_id", "term_ru_sg", "term_en_sg"]], on="concept_id").to_csv(
        out / "frequency_coverage_audit.csv", index=False, encoding="utf-8")
    # Refresh only the derived review list. Preserve existing human notes for
    # identical forms; do not create or infer any human review status.
    dest = ROOT / "data/real/sentences_review.csv"
    keys = ["concept_id", "lang", "number", "template_id", "input_text"]
    prior = pd.read_csv(dest).fillna("")
    fresh = s.loc[s["mode"].str.startswith("sentence_"), keys].copy()
    fresh = fresh.merge(prior[keys + ["naturalness", "reviewer_note"]], on=keys,
                        how="left", validate="one_to_one").fillna("")
    # No previously entered assessment may be lost on a changed phrase.
    noted = prior[prior.naturalness.ne("") | prior.reviewer_note.ne("")]
    if len(noted):
        retained = noted.merge(fresh[keys], on=keys, how="inner", validate="one_to_one")
        assert len(retained) == len(noted), "A reviewed phrase disappeared; retain it separately first"
    fresh.to_csv(dest, index=False, encoding="utf-8")
    # The full-run manifest records files as they existed at that run. Keep
    # it intact; this separate manifest captures the completed audit files.
    tracked = [ROOT / entry["path"] for entry in frozen["files"]]
    tracked += list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py"))
    tracked += list(ROOT.glob("*.md")) + list(out.glob("*.json")) + list(out.glob("*.csv"))
    completion = out / "audit_completion_manifest.json"
    previous = {e["path"]: e["sha256"] for e in frozen["files"]}
    changes = []
    for path in sorted(set(tracked)):
        if not path.is_file() or path == completion:
            continue
        rel = path.relative_to(ROOT).as_posix()
        sha = digest(path)
        if previous.get(rel) != sha:
            changes.append({"path": rel, "before_sha256": previous.get(rel), "after_sha256": sha})
    completion.write_text(json.dumps({
        "baseline_manifest_sha256": digest(baseline / "SNAPSHOT_MANIFEST.json"),
        "full_run_manifest_sha256": digest(out / "manifest.json"),
        "note": "Records final audit changes; does not replace the full-run manifest or frozen originals.",
        "changed_or_added_files": changes,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Audit passed; review list contains {len(fresh)} unassessed/context rows. See {out / 'audit_step1.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    main(parser.parse_args().baseline)
