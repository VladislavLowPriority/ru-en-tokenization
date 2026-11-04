# -*- coding: utf-8 -*-
"""Изолированный расчетный конвейер для expansion_wipo_v1.

Запуск:
    .venv\\Scripts\\python.exe scripts\\run_expansion.py

Строго изолирован от старых результатов:
- Читает данные из expansion_wipo_v1/data/
- Пишет результаты строго в expansion_wipo_v1/results/ (new, combined, control, sensitivity)
- Вычисляет H1 (G), H2 (M), H3 (контекст), H4 (R)
- Применяет кластерный бутстрэп (5000 итераций, seed 20260918)
- Оценивает точность и границы практической эквивалентности (|ln R| >= 0.095)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tok_pilot import backends, config, manifest, measure, metrics, stimuli, validate  # noqa: E402

EXP = ROOT / "expansion_wipo_v1"
DATA_DIR = EXP / "data"
RESULTS_DIR = EXP / "results"
RES_NEW = RESULTS_DIR / "new"
RES_COMB = RESULTS_DIR / "combined"
RES_CTRL = RESULTS_DIR / "control"
RES_SENS = RESULTS_DIR / "sensitivity"

for d in (RES_NEW, RES_COMB, RES_CTRL, RES_SENS):
    d.mkdir(parents=True, exist_ok=True)

SEED = 20260918
BOOTSTRAP_REPS = 5000
EQUIV_LN_R = 0.095
EQUIV_LOW = float(np.exp(-EQUIV_LN_R))
EQUIV_HIGH = float(np.exp(EQUIV_LN_R))

def json_dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


# --- 1. Измерение набора понятий ----------------------------------------------

def run_measurement_pipeline(concepts_path: Path, out_dir: Path, split_name: str, numbers=("sg", "pl")) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    print(f"\n[{split_name}] Загрузка понятий: {concepts_path}")
    concepts = stimuli.load_concepts(concepts_path)
    if "final_status" in concepts.columns:
        concepts = concepts[concepts["final_status"] == "accepted"].copy()
    print(f"  понятий: {len(concepts)}, семей: {concepts['family_id'].nunique()}")

    st = stimuli.build_stimuli(concepts, numbers=numbers)
    st = stimuli.add_frequencies(st)
    st.to_csv(out_dir / "stimuli.csv", index=False, encoding="utf-8")
    print(f"  стимулов (строк): {len(st)}")

    bks = backends.load_all()
    man = manifest.build(bks, [concepts_path], st, split_name)
    manifest.save(man, out_dir / "manifest.json")

    print(f"  измерение run1...")
    m1 = measure.measure(st, bks, run_id="run1")
    m1.to_csv(out_dir / "measurements.csv", index=False, encoding="utf-8")

    print(f"  проверка детерминированности run2...")
    m2 = measure.measure(st, bks, run_id="run2")

    checks = {
        "determinism": validate.check_determinism(m1, m2),
        "integrity": validate.check_integrity(m1),
        "isolated_consistency": validate.check_isolated_consistency(m1),
        "sample_frame": validate.check_sample_frame(concepts),
        "normalizer_effect": validate.check_normalizer_effect(bks, st).to_dict("records"),
        "freq_coverage": validate.check_freq_coverage(st).to_dict("records"),
        "cross_boundary": validate.check_cross_boundary(m1).to_dict("records"),
    }
    for tid, be in bks.items():
        if be.backend == "hf":
            checks[f"hf_offsets::{tid}"] = validate.check_hf_offsets(be, st)
    json_dump(checks, out_dir / "validation.json")
    print(f"  детерминированность: {checks['determinism']['identical']}, декодирование: {checks['integrity']['decode_match_all']}")

    wide = metrics.build_wide(m1, st)
    if "cohort" in concepts.columns:
        wide = wide.merge(concepts[["concept_id", "cohort"]].drop_duplicates(), on="concept_id", how="left")
    wide.to_csv(out_dir / "analysis_wide.csv", index=False, encoding="utf-8")
    return wide, concepts, checks


# --- 2. Сопоставление и кластеры ----------------------------------------------

def add_joint_clusters(pf: pd.DataFrame) -> pd.DataFrame:
    parent: dict[str, str] = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for _, r in pf.iterrows():
        union("T:" + str(r["family_id"]), "C:" + str(r["control_family_id"]))
    pf = pf.copy()
    pf["cluster_id"] = ["cl_" + find("T:" + str(f)) for f in pf["family_id"]]
    return pf


def build_paired_frame(tech_wide: pd.DataFrame, ctrl_wide: pd.DataFrame, match_df: pd.DataFrame, tid: str) -> pd.DataFrame:
    t_sub = tech_wide[tech_wide["tokenizer_id"] == tid].set_index("concept_id")
    c_sub = ctrl_wide[ctrl_wide["tokenizer_id"] == tid].copy()
    c_sub = c_sub.set_index("concept_id")

    rows = []
    for _, m in match_df.iterrows():
        t_id = str(m["tech_concept_id"])
        c_id = f"CTRL_{m['ctrl_cand_id']}"
        if t_id not in t_sub.index or c_id not in c_sub.index:
            continue
        tt = t_sub.loc[t_id]
        cc = c_sub.loc[c_id]
        
        rows.append({
            "concept_id": t_id,
            "family_id": tt["family_id"],
            "control_family_id": cc["family_id"],
            "control_concept_id": c_id,
            "stratum": tt["stratum"],
            "tech_cohort": m.get("tech_cohort", tt.get("cohort", "")),
            "exact_cell": m["exact_cell"],
            "T_ru_tech": tt["T_ru_sg_iso0"], "T_en_tech": tt["T_en_sg_iso0"],
            "T_ru_ctrl": cc["T_ru_sg_iso0"], "T_en_ctrl": cc["T_en_sg_iso0"],
            "T_ru_tech_sp": tt["T_ru_sg_isosp"], "T_en_tech_sp": tt["T_en_sg_isosp"],
            "T_ru_ctrl_sp": cc["T_ru_sg_isosp"], "T_en_ctrl_sp": cc["T_en_sg_isosp"],
            "C_ru_tech": tt["C_ru_sg"], "C_en_tech": tt["C_en_sg"],
            "C_ru_ctrl": cc["C_ru_sg"], "C_en_ctrl": cc["C_en_sg"],
            "W_ru_tech": tt["W_ru_sg"], "W_en_tech": tt["W_en_sg"],
            "W_ru_ctrl": cc["W_ru_sg"], "W_en_ctrl": cc["W_en_sg"],
            "F_ru_tech": tt["F_ru_sg"], "F_en_tech": tt["F_en_sg"],
            "F_ru_ctrl": cc["F_ru_sg"], "F_en_ctrl": cc["F_en_sg"],
            "Fmiss_tech": tt.get("Fmiss_ru_sg", 0) + tt.get("Fmiss_en_sg", 0),
            "Fmiss_ctrl": cc.get("Fmiss_ru_sg", 0) + cc.get("Fmiss_en_sg", 0),
        })
    df_pairs = pd.DataFrame(rows)
    if not df_pairs.empty:
        df_pairs = add_joint_clusters(df_pairs)
    return df_pairs


# --- 3. Статистические функции R ----------------------------------------------

def stat_R(df: pd.DataFrame, suffix: str = "") -> float:
    d = np.log(df[f"T_ru_tech{suffix}"] / df[f"T_en_tech{suffix}"]) - np.log(df[f"T_ru_ctrl{suffix}"] / df[f"T_en_ctrl{suffix}"])
    return float(np.exp(d.mean()))

def stat_G_side(df: pd.DataFrame, side: str, suffix: str = "") -> float:
    return float(np.exp(np.log(df[f"T_ru_{side}{suffix}"] / df[f"T_en_{side}{suffix}"]).mean()))

def run_adjusted_R(df: pd.DataFrame, with_freq: bool, reps: int = BOOTSTRAP_REPS, seed: int = SEED) -> dict:
    long = []
    for _, r in df.iterrows():
        for side in ("tech", "ctrl"):
            long.append({
                "pair_id": r["concept_id"],
                "family_id": r.get("cluster_id", r["family_id"]),
                "is_tech": 1.0 if side == "tech" else 0.0,
                "y": np.log(r[f"T_ru_{side}"] / r[f"T_en_{side}"]),
                "logC": np.log(r[f"C_ru_{side}"] / r[f"C_en_{side}"]),
                "dW": r[f"W_ru_{side}"] - r[f"W_en_{side}"],
                "F_ru": r[f"F_ru_{side}"], "F_en": r[f"F_en_{side}"],
                "fmiss": r["Fmiss_tech"] if side == "tech" else r["Fmiss_ctrl"],
            })
    L = pd.DataFrame(long)
    if with_freq:
        keep_pairs = L.groupby("pair_id")["fmiss"].max() == 0
        good = set(keep_pairs[keep_pairs].index)
        L = L[L["pair_id"].isin(good)]
        L = L[np.isfinite(L[["F_ru", "F_en"]]).all(axis=1)]

    def fit(sub: pd.DataFrame) -> float | None:
        cols = [np.ones(len(sub)), sub["is_tech"].to_numpy(float),
                sub["logC"].to_numpy(float), sub["dW"].to_numpy(float)]
        if with_freq:
            zr = (sub["F_ru"] - sub["F_ru"].mean()) / (sub["F_ru"].std(ddof=0) or 1)
            ze = (sub["F_en"] - sub["F_en"].mean()) / (sub["F_en"].std(ddof=0) or 1)
            cols.append((zr - ze).to_numpy(float))
        keep = [0, 1] + [i for i in range(2, len(cols)) if cols[i].std() > 0]
        X = np.column_stack([cols[i] for i in keep])
        y = sub["y"].to_numpy(float)
        if np.linalg.matrix_rank(X) < X.shape[1]:
            return None
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(beta[1])

    point = fit(L)
    rng = np.random.default_rng(seed)
    fams = L["family_id"].unique()
    idx_by_fam = [np.flatnonzero((L["family_id"] == f).to_numpy()) for f in fams]
    draws = []
    base = L.reset_index(drop=True)
    for _ in range(reps):
        pick = rng.integers(0, len(fams), size=len(fams))
        idx = np.concatenate([idx_by_fam[p] for p in pick])
        b = fit(base.take(idx))
        if b is not None:
            draws.append(b)
    draws = np.array(draws)
    enough = len(draws) >= 0.95 * reps
    lo, hi = (np.percentile(draws, [2.5, 97.5]) if len(draws) else (np.nan, np.nan))
    lo90, hi90 = (np.percentile(draws, [5.0, 95.0]) if len(draws) else (np.nan, np.nan))
    return {
        "model": "log(T_ru/T_en) ~ is_tech + log(C_ru/C_en) + (W_ru-W_en)" + (" + (zF_ru - zF_en)" if with_freq else ""),
        "n_pairs": int(L["pair_id"].nunique()),
        "n_clusters": int(len(fams)),
        "R_adjusted": float(np.exp(point)) if point is not None else None,
        "ci_low": float(np.exp(lo)) if enough else None,
        "ci_high": float(np.exp(hi)) if enough else None,
        "ci_90_low": float(np.exp(lo90)) if enough else None,
        "ci_90_high": float(np.exp(hi90)) if enough else None,
        "reps_used": int(len(draws)),
        "seed": seed
    }


# --- 4. Полный расчет среза H1-H4 ---------------------------------------------

def evaluate_slice(tech_wide: pd.DataFrame, ctrl_wide: pd.DataFrame, match_df: pd.DataFrame, slice_name: str, out_dir: Path) -> dict:
    print(f"\n==========================================")
    print(f" Расчет среза: {slice_name}")
    print(f"==========================================")

    res = {
        "slice_name": slice_name,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "equivalence_margin": {"ln_r_margin": EQUIV_LN_R, "r_low": EQUIV_LOW, "r_high": EQUIV_HIGH},
        "by_tokenizer": {}
    }

    tokenizers = ["o200k_base", "cl100k_base", "qwen2.5-7b-instruct"]

    for tid in tokenizers:
        tw = tech_wide[tech_wide["tokenizer_id"] == tid]
        pf = build_paired_frame(tech_wide, ctrl_wide, match_df, tid)
        
        print(f"\n--- Токенизатор: {tid} ---")
        # H1: G
        g_all = metrics.cluster_bootstrap(tw, metrics.stat_G, reps=BOOTSTRAP_REPS, seed=SEED)
        g_by_stratum = {}
        for s, sub in tw.groupby("stratum"):
            g_by_stratum[s] = metrics.cluster_bootstrap(sub, metrics.stat_G, reps=BOOTSTRAP_REPS, seed=SEED)
        print(f"  H1 G: {g_all['point']:.4f} [{g_all['ci_low']:.4f}; {g_all['ci_high']:.4f}]")

        # H2: M (SG -> PL difference)
        m_all = metrics.cluster_bootstrap(tw, metrics.stat_M, reps=BOOTSTRAP_REPS, seed=SEED)
        print(f"  H2 M: {m_all['point']:.4f} [{m_all['ci_low']:.4f}; {m_all['ci_high']:.4f}]")

        # H3: Context shift
        h3_all = metrics.cluster_bootstrap(tw, metrics.stat_H3, reps=BOOTSTRAP_REPS, seed=SEED)
        print(f"  H3 Shift: {h3_all['point']:.4f} [{h3_all['ci_low']:.4f}; {h3_all['ci_high']:.4f}]")

        # H4: R on matched pairs
        tok_entry = {
            "H1_G_tech_all": g_all,
            "H1_G_by_stratum": g_by_stratum,
            "H2_M_all": m_all,
            "H3_context_shift": h3_all,
            "matched_control": {}
        }

        if not pf.empty:
            CL = "cluster_id"
            n_pairs = len(pf)
            n_clusters = pf[CL].nunique()
            r_raw = metrics.cluster_bootstrap(pf, stat_R, cluster=CL, reps=BOOTSTRAP_REPS, seed=SEED)
            r_sp = metrics.cluster_bootstrap(pf, lambda d: stat_R(d, "_sp"), cluster=CL, reps=BOOTSTRAP_REPS, seed=SEED)
            g_tech_matched = metrics.cluster_bootstrap(pf, lambda d: stat_G_side(d, "tech"), cluster=CL, reps=BOOTSTRAP_REPS, seed=SEED)
            g_ctrl_matched = metrics.cluster_bootstrap(pf, lambda d: stat_G_side(d, "ctrl"), cluster=CL, reps=BOOTSTRAP_REPS, seed=SEED)
            
            # Equivalence test
            # 1. Conservative 95% check: [ci_low, ci_high] entirely inside [0.9094, 1.0997]
            equiv_established_95 = bool(r_raw["ci_low"] >= EQUIV_LOW and r_raw["ci_high"] <= EQUIV_HIGH)
            # 2. Standard TOST (alpha=0.05) 90% check: [ci_90_low, ci_90_high] entirely inside [0.9094, 1.0997]
            equiv_established_90 = bool(r_raw.get("ci_90_low", 0) >= EQUIV_LOW and r_raw.get("ci_90_high", 999) <= EQUIV_HIGH)
            effect_excluded = bool(r_raw["ci_high"] < EQUIV_HIGH or r_raw["ci_low"] > EQUIV_LOW)

            r_by_stratum = {}
            for s, ds in pf.groupby("stratum"):
                r_by_stratum[s] = metrics.cluster_bootstrap(ds, stat_R, cluster=CL, reps=BOOTSTRAP_REPS, seed=SEED)

            adj_len = run_adjusted_R(pf, with_freq=False, reps=BOOTSTRAP_REPS, seed=SEED)
            adj_freq = run_adjusted_R(pf, with_freq=True, reps=BOOTSTRAP_REPS, seed=SEED)

            tok_entry["matched_control"] = {
                "n_pairs": n_pairs,
                "n_tech_families": int(pf["family_id"].nunique()),
                "n_ctrl_families": int(pf["control_family_id"].nunique()),
                "n_joint_clusters": int(n_clusters),
                "G_tech_matched": g_tech_matched,
                "G_control_matched": g_ctrl_matched,
                "R_raw": r_raw,
                "R_isolated_sp": r_sp,
                "R_by_stratum": r_by_stratum,
                "R_adjusted_length_structure": adj_len,
                "R_adjusted_frequency": adj_freq,
                "equivalence_test": {
                    "margin_ln_r": EQUIV_LN_R,
                    "interval_limits": [EQUIV_LOW, EQUIV_HIGH],
                    "ci_95_limits": [r_raw["ci_low"], r_raw["ci_high"]],
                    "ci_95_inside_equivalence_zone": equiv_established_95,
                    "ci_90_limits": [r_raw.get("ci_90_low"), r_raw.get("ci_90_high")],
                    "ci_90_tost_inside_equivalence_zone": equiv_established_90,
                    "ci_inside_equivalence_zone": equiv_established_95
                }
            }
            print(f"  H4 R (raw, {n_pairs} pairs, {n_clusters} clusters): {r_raw['point']:.4f} [95% CI: {r_raw['ci_low']:.4f}; {r_raw['ci_high']:.4f} | 90% CI: {r_raw.get('ci_90_low', 0):.4f}; {r_raw.get('ci_90_high', 0):.4f}]")
            print(f"  H4 R (adj len/struct): {adj_len['R_adjusted']:.4f} [95% CI: {adj_len['ci_low']:.4f}; {adj_len['ci_high']:.4f}]")
            if adj_freq['R_adjusted'] is not None:
                print(f"  H4 R (adj freq, n={adj_freq['n_pairs']}): {adj_freq['R_adjusted']:.4f} [{adj_freq['ci_low']:.4f}; {adj_freq['ci_high']:.4f}]")
            print(f"  Эквивалентность в пределах 10% ({EQUIV_LOW:.4f}-{EQUIV_HIGH:.4f}): 95% консервативный={'ДА' if equiv_established_95 else 'НЕТ'}; 90% TOST (alpha=0.05)={'ДА' if equiv_established_90 else 'НЕТ'}")

        res["by_tokenizer"][tid] = tok_entry

    json_dump(res, out_dir / "summary.json")
    return res


# --- 5. Проверки устойчивости (Sensitivity) ------------------------------------

def run_sensitivity_analyses(tech_comb_wide: pd.DataFrame, ctrl_wide: pd.DataFrame, match_comb: pd.DataFrame) -> dict:
    print(f"\n==========================================")
    print(f" Запуск проверок устойчивости (Sensitivity)")
    print(f"==========================================")
    
    sens_out = {"evaluated_at_utc": datetime.now(timezone.utc).isoformat(), "checks": {}}
    tid = "o200k_base"
    tw = tech_comb_wide[tech_comb_wide["tokenizer_id"] == tid]
    
    # 1. Сравнение источников: WIPO vs Документация производителей
    # doc_pilot_60 vs wipo_pilot_30 vs wipo_expansion_v1
    sources = {}
    for cohort, sub in tw.groupby("cohort"):
        sources[cohort] = {
            "n_concepts": len(sub),
            "G": metrics.cluster_bootstrap(sub, metrics.stat_G, reps=BOOTSTRAP_REPS, seed=SEED),
            "M": metrics.cluster_bootstrap(sub, metrics.stat_M, reps=BOOTSTRAP_REPS, seed=SEED),
            "stratum_dist": sub["stratum"].value_counts().to_dict()
        }
    sens_out["checks"]["source_comparison"] = sources
    print("  [1/4] Сравнение источников (cohorts) рассчитано.")

    # 2. Прямые парные контрасты токенизаторов на объединенной выборке (общий бутстрэп)
    t_o200 = tech_comb_wide[tech_comb_wide["tokenizer_id"] == "o200k_base"].set_index("concept_id")
    t_cl100 = tech_comb_wide[tech_comb_wide["tokenizer_id"] == "cl100k_base"].set_index("concept_id")
    t_qwen = tech_comb_wide[tech_comb_wide["tokenizer_id"] == "qwen2.5-7b-instruct"].set_index("concept_id")

    paired_tok_rows = []
    for cid in t_o200.index:
        if cid in t_cl100.index and cid in t_qwen.index:
            r0 = t_o200.loc[cid]
            r1 = t_cl100.loc[cid]
            r2 = t_qwen.loc[cid]
            paired_tok_rows.append({
                "concept_id": cid,
                "family_id": r0["family_id"],
                "stratum": r0["stratum"],
                "G_o200": r0["T_ru_sg_iso0"] / r0["T_en_sg_iso0"],
                "G_cl100": r1["T_ru_sg_iso0"] / r1["T_en_sg_iso0"],
                "G_qwen": r2["T_ru_sg_iso0"] / r2["T_en_sg_iso0"],
            })
    df_tok_contrasts = pd.DataFrame(paired_tok_rows)
    
    # Delta log G
    def stat_delta_cl100_o200(d):
        return float(np.exp(np.log(d["G_cl100"]).mean() - np.log(d["G_o200"]).mean()))
    def stat_delta_qwen_o200(d):
        return float(np.exp(np.log(d["G_qwen"]).mean() - np.log(d["G_o200"]).mean()))

    sens_out["checks"]["tokenizer_contrasts"] = {
        "cl100_vs_o200_ratio": metrics.cluster_bootstrap(df_tok_contrasts, stat_delta_cl100_o200, reps=BOOTSTRAP_REPS, seed=SEED),
        "qwen_vs_o200_ratio": metrics.cluster_bootstrap(df_tok_contrasts, stat_delta_qwen_o200, reps=BOOTSTRAP_REPS, seed=SEED),
    }
    print("  [2/4] Прямые контрасты токенизаторов рассчитаны.")

    # 3. Режим пробелов: isolated_0 vs isolated_sp
    pf = build_paired_frame(tech_comb_wide, ctrl_wide, match_comb, tid)
    r_iso0 = metrics.cluster_bootstrap(pf, stat_R, cluster="cluster_id", reps=BOOTSTRAP_REPS, seed=SEED)
    r_isosp = metrics.cluster_bootstrap(pf, lambda d: stat_R(d, "_sp"), cluster="cluster_id", reps=BOOTSTRAP_REPS, seed=SEED)
    sens_out["checks"]["space_mode_contrast"] = {
        "R_isolated_0": r_iso0,
        "R_isolated_sp": r_isosp,
        "ratio_sp_to_iso0": float(r_isosp["point"] / r_iso0["point"])
    }
    print("  [3/4] Влияние начального пробела рассчитано.")

    # 4. Анализ ширины доверительного интервала (проверка гипотезы о сужении sqrt(K_old / K_new))
    # Старый срез: K_old = 62 кластера
    # Объединенный срез: K_new кластеров
    K_old = 62
    K_new = pf["cluster_id"].nunique()
    expected_ratio = float(np.sqrt(K_old / K_new))
    ci_old_log_width = np.log(1.1332576559) - np.log(0.9337718727)  # Baseline log CI width = 0.1935
    ci_new_log_width = np.log(r_iso0["ci_high"]) - np.log(r_iso0["ci_low"])
    actual_ratio = float(ci_new_log_width / ci_old_log_width)
    sens_out["checks"]["precision_gain"] = {
        "K_old_clusters": K_old,
        "K_new_clusters": K_new,
        "theoretical_ratio_sqrt_K": expected_ratio,
        "log_ci_width_baseline": float(ci_old_log_width),
        "log_ci_width_combined": float(ci_new_log_width),
        "actual_ratio_ci_width": actual_ratio,
        "ci_narrowing_percent": float((1.0 - actual_ratio) * 100)
    }
    print(f"  [4/4] Оценка точности: кластеров {K_old} -> {K_new}, ширина лог-ДИ изменилась на {sens_out['checks']['precision_gain']['ci_narrowing_percent']:.1f}%")

    json_dump(sens_out, RES_SENS / "sensitivity_summary.json")
    return sens_out


# --- 6. Главная функция -------------------------------------------------------

def main():
    print("=================================================================")
    print(" ЗАПУСК РАСЧЕТНОГО КОНВЕЙЕРА: expansion_wipo_v1")
    print("=================================================================")

    # 1. Измерение новых 90 технических понятий
    new_concepts_file = DATA_DIR / "concepts_new.csv"
    new_wide, new_concepts, new_checks = run_measurement_pipeline(
        new_concepts_file, RES_NEW, "expansion_new"
    )

    # 2. Измерение объединенных 180 понятий
    comb_concepts_file = DATA_DIR / "concepts_combined.csv"
    comb_wide, comb_concepts, comb_checks = run_measurement_pipeline(
        comb_concepts_file, RES_COMB, "expansion_combined"
    )

    # 3. Измерение бытового контроля (145 понятий)
    ctrl_concepts_file = DATA_DIR / "concepts_control_expanded.csv"
    ctrl_wide, ctrl_concepts, ctrl_checks = run_measurement_pipeline(
        ctrl_concepts_file, RES_CTRL, "control_expanded", numbers=("sg",)
    )

    # Загрузка таблиц сопоставления
    match_new_file = DATA_DIR / "matching_pairs_new.csv"
    match_comb_file = DATA_DIR / "matching_pairs_combined.csv"
    df_match_new = pd.read_csv(match_new_file)
    df_match_comb = pd.read_csv(match_comb_file)

    # Расчет среза: ТОЛЬКО НОВЫЕ ПОНЯТИЯ (90 technical, 69 matched pairs)
    new_slice_results = evaluate_slice(
        new_wide, ctrl_wide, df_match_new, "new_wipo_slice", RES_NEW
    )

    # Расчет среза: ОБЪЕДИНЕННАЯ ВЫБОРКА (180 technical, 145 matched pairs)
    comb_slice_results = evaluate_slice(
        comb_wide, ctrl_wide, df_match_comb, "combined_180_slice", RES_COMB
    )

    # Проверки устойчивости
    sens_results = run_sensitivity_analyses(comb_wide, ctrl_wide, df_match_comb)

    print("\n=================================================================")
    print(" ВСЕ ИЗМЕРЕНИЯ И РАСЧЕТЫ ЗАВЕРШЕНЫ УСПЕШНО!")
    print(f" Результаты сохранены в: {RESULTS_DIR}")
    print("=================================================================")


if __name__ == "__main__":
    main()
