"""Метрики и интервалы (§7, §8).

Единица наблюдения — понятие, а не строка. Языки, числа, шаблоны и
токенизаторы одного понятия связаны, поэтому используются парные величины,
а интервалы считаются кластерным бутстрэпом по лексическим семьям.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def build_wide(measurements: pd.DataFrame, stimuli: pd.DataFrame) -> pd.DataFrame:
    """Одна строка = понятие x токенизатор; колонки — T и S по условиям."""
    m = measurements.merge(
        stimuli[["stimulus_id", "word_count", "char_count_no_spaces",
                 "utf8_byte_count", "freq_proxy", "freq_missing"]],
        on="stimulus_id", how="left")
    m = m[m["error_code"].fillna("") == ""]

    keys = ["concept_id", "family_id", "stratum", "tokenizer_id"]
    out = m.groupby(keys, as_index=False).size().drop(columns="size")

    def put(col, sub, value_col):
        nonlocal out
        piece = sub.groupby(keys, as_index=False)[value_col].first().rename(
            columns={value_col: col})
        out = out.merge(piece, on=keys, how="left")

    for lang in ("ru", "en"):
        for number in ("sg", "pl"):
            iso0 = m[(m.lang == lang) & (m.number == number) & (m["mode"] == "isolated_0")]
            put(f"T_{lang}_{number}_iso0", iso0, "n_tokens_input")
            put(f"S_{lang}_{number}_iso0", iso0, "n_tokens_target")
            put(f"W_{lang}_{number}", iso0, "word_count")
            put(f"C_{lang}_{number}", iso0, "char_count_no_spaces")
            put(f"B_{lang}_{number}", iso0, "utf8_byte_count")
            put(f"F_{lang}_{number}", iso0, "freq_proxy")
            put(f"Fmiss_{lang}_{number}", iso0, "freq_missing")

            isosp = m[(m.lang == lang) & (m.number == number) & (m["mode"] == "isolated_space")]
            put(f"T_{lang}_{number}_isosp", isosp, "n_tokens_input")
            put(f"S_{lang}_{number}_isosp", isosp, "n_tokens_target")

            for tpl in config.ALL_TEMPLATES:
                sent = m[(m.lang == lang) & (m.number == number)
                         & (m["mode"] == f"sentence_{tpl}")]
                put(f"T_{lang}_{number}_s{tpl}", sent, "n_tokens_input")
                put(f"S_{lang}_{number}_s{tpl}", sent, "n_tokens_target")
                put(f"X_{lang}_{number}_s{tpl}", sent, "n_cross_boundary")
    return out


# --- точечные показатели -----------------------------------------------------

def stat_G(df: pd.DataFrame) -> float:
    """H1: геометрическое среднее парных отношений T_RU/T_EN, SG, isolated_0."""
    d = np.log(df["T_ru_sg_iso0"] / df["T_en_sg_iso0"])
    return float(np.exp(d.mean()))


def stat_mean_diff_sg(df: pd.DataFrame) -> float:
    """Средняя парная разность в токенах — для понятности рядом с G."""
    return float((df["T_ru_sg_iso0"] - df["T_en_sg_iso0"]).mean())


def stat_ratio_of_sums(df: pd.DataFrame) -> float:
    """Прикладное дополнение: отношение сумм. Сильнее взвешивает длинные термины."""
    return float(df["T_ru_sg_iso0"].sum() / df["T_en_sg_iso0"].sum())


def stat_M(df: pd.DataFrame) -> float:
    """H2: средняя парная разность изменений при SG->PL (в токенах)."""
    d_ru = df["T_ru_pl_iso0"] - df["T_ru_sg_iso0"]
    d_en = df["T_en_pl_iso0"] - df["T_en_sg_iso0"]
    return float((d_ru - d_en).mean())


def context_contrast_rows(df: pd.DataFrame, templates=None) -> pd.Series:
    """H3: сдвиг log(S_RU/S_EN) от isolated_space к предложению.

    Сначала равновзвешенное среднее по формам числа и шаблонам внутри понятия.
    """
    templates = templates or list(config.TEMPLATES)
    parts = []
    for number in ("sg", "pl"):
        base_n = np.log(df[f"S_ru_{number}_isosp"] / df[f"S_en_{number}_isosp"])
        for tpl in templates:
            ctx = np.log(df[f"S_ru_{number}_s{tpl}"] / df[f"S_en_{number}_s{tpl}"])
            parts.append(ctx - base_n)
    return sum(parts) / len(parts)


def stat_H3(df: pd.DataFrame, templates=None) -> float:
    return float(context_contrast_rows(df, templates).mean())


def make_stat_H3(templates):
    def f(df):
        return stat_H3(df, templates)
    return f


def stat_space_effect(df: pd.DataFrame, lang: str) -> float:
    """Диагностика ведущего пробела: T(isolated_space) - T(isolated_0), SG."""
    return float((df[f"T_{lang}_sg_isosp"] - df[f"T_{lang}_sg_iso0"]).mean())


def stat_fertility_ratio(df: pd.DataFrame) -> float:
    """Дополнительно (§7): отношение fertility RU/EN, геометрическое среднее."""
    f_ru = df["T_ru_sg_iso0"] / df["W_ru_sg"]
    f_en = df["T_en_sg_iso0"] / df["W_en_sg"]
    return float(np.exp(np.log(f_ru / f_en).mean()))


def stat_sentence_ratio(df: pd.DataFrame, tpl: str, number: str = "sg") -> float:
    """Отношение длин ПОЛНЫХ предложений — другой объект, чем S (§7)."""
    return float(np.exp(np.log(df[f"T_ru_{number}_s{tpl}"]
                               / df[f"T_en_{number}_s{tpl}"]).mean()))


# --- неопределённость --------------------------------------------------------

def cluster_bootstrap(df: pd.DataFrame, stat_fn, reps: int = config.BOOTSTRAP_REPS,
                      seed: int = config.SEED, cluster: str = "family_id") -> dict:
    """95% интервал перевыбором ЛЕКСИЧЕСКИХ СЕМЕЙ целиком (§8).

    Перевыбирается семья со всеми её понятиями и всеми их измерениями, что
    учитывает и связь словоформ, и повторение общего головного слова.
    """
    rng = np.random.default_rng(seed)
    base = df.reset_index(drop=True)
    fams = base[cluster].unique()
    # позиции строк каждой семьи: перевыбирается семья целиком, а не отдельное понятие
    idx_by_fam = [np.flatnonzero((base[cluster] == f).to_numpy()) for f in fams]
    point = stat_fn(base)
    draws = np.full(reps, np.nan)
    n_fam = len(fams)
    for r in range(reps):
        pick = rng.integers(0, n_fam, size=n_fam)
        idx = np.concatenate([idx_by_fam[p] for p in pick])
        try:
            draws[r] = stat_fn(base.take(idx))
        except Exception:
            draws[r] = np.nan
    ok = draws[~np.isnan(draws)]
    lo, hi = np.percentile(ok, [2.5, 97.5])
    lo90, hi90 = np.percentile(ok, [5.0, 95.0])
    return {
        "point": point, "ci_low": float(lo), "ci_high": float(hi),
        "ci_90_low": float(lo90), "ci_90_high": float(hi90),
        "n_concepts": int(df["concept_id"].nunique()),
        "n_families": int(len(fams)),
        "reps_used": int(len(ok)), "seed": seed,
    }


def log_halfwidth(res: dict) -> float:
    """Полуширина интервала на log-шкале — ориентир точности из §10."""
    return float((np.log(res["ci_high"]) - np.log(res["ci_low"])) / 2)


# --- проверка чувствительности (§8) -----------------------------------------

def sensitivity_regression(df: pd.DataFrame, coverage: str = "complete",
                           reps: int = config.BOOTSTRAP_REPS,
                           seed: int = config.SEED) -> dict:
    """d_i = b0 + bC*log(C_ru/C_en) + bW*(W_ru-W_en) + bF*(zF_ru-zF_en).

    Основная поправка использует только пары с частотой ВСЕХ компонентов.
    coverage='available' воспроизводит прежний отбор по непустому среднему,
    но явно помечает частично покрытые пары как дополнительную диагностику.
    В каждой бутстрэп-выборке семьи берутся целиком; стандартизация частот
    и все коэффициенты оцениваются заново. b0 не является чистым эффектом языка.
    """
    if coverage not in ("complete", "available"):
        raise ValueError("coverage must be complete or available")
    if reps < 1:
        raise ValueError("reps must be positive")
    if df["concept_id"].duplicated().any() or df["family_id"].isna().any():
        raise ValueError("Regression requires one row per concept and valid family IDs")
    available = np.isfinite(df[["F_ru_sg", "F_en_sg"]]).all(axis=1)
    complete = available & df["Fmiss_ru_sg"].eq(0) & df["Fmiss_en_sg"].eq(0)
    use = complete if coverage == "complete" else available
    sub = df.loc[use].reset_index(drop=True)
    result = {
        "coverage_policy": coverage,
        "n_total": int(len(df)), "n_used": int(len(sub)),
        "n_complete": int(complete.sum()),
        "n_partial_used": int((use & ~complete).sum()),
        "n_families": int(sub["family_id"].nunique()),
        "included_concept_ids": sub["concept_id"].tolist(),
        "excluded_concept_ids": df.loc[~use, "concept_id"].tolist(),
        "partial_concept_ids": df.loc[available & ~complete, "concept_id"].tolist(),
        "seed": seed, "reps_requested": reps,
        "standardization": "within each language on analysis subset; re-estimated in every family bootstrap draw",
        "interval_method": "percentile 95%; lexical-family bootstrap; equal concept weights in point estimate",
        "interpretation": "exp(beta0) is a model reference ratio, not G, not a causal/pure language effect; marginal ranges do not establish joint support",
    }
    if len(sub) < 6 or result["n_families"] < 2:
        return {**result, "status": "insufficient_data", "reps_used": 0}
    positive = sub[["T_ru_sg_iso0", "T_en_sg_iso0", "C_ru_sg", "C_en_sg"]].to_numpy(float)
    if not np.isfinite(positive).all() or (positive <= 0).any():
        raise ValueError("Token and character counts must be finite and positive")
    y = np.log(positive[:, 0] / positive[:, 1])
    log_c = np.log(positive[:, 2] / positive[:, 3])
    dw = (sub["W_ru_sg"] - sub["W_en_sg"]).to_numpy(float)
    freq = sub[["F_ru_sg", "F_en_sg"]].to_numpy(float)
    if not np.isfinite(dw).all():
        raise ValueError("Word counts must be finite")

    def fit(idx):
        f = freq[idx]
        scale = f.std(axis=0, ddof=1)
        if not np.isfinite(scale).all() or (scale == 0).any():
            return None, None, "zero_frequency_variance"
        z = (f - f.mean(axis=0)) / scale
        x = np.column_stack([np.ones(len(idx)), log_c[idx], dw[idx], z[:, 0] - z[:, 1]])
        beta, _, rank, _ = np.linalg.lstsq(x, y[idx], rcond=None)
        if rank < x.shape[1]:
            return None, x, "rank_deficient"
        if not np.isfinite(beta).all():
            return None, x, "nonfinite_fit"
        return beta, x, None

    beta, x, error = fit(np.arange(len(sub)))
    if error:
        return {**result, "status": error, "reps_used": 0}
    names = ("beta0", "beta_C", "beta_W", "beta_F")
    result.update({name: float(value) for name, value in zip(names, beta)})
    result.update({
        "range_logC": [float(log_c.min()), float(log_c.max())],
        "range_dW": [float(dw.min()), float(dw.max())],
        "range_dzF": [float(x[:, 3].min()), float(x[:, 3].max())],
        "design_rank": int(np.linalg.matrix_rank(x)),
        "design_condition_number": float(np.linalg.cond(x)),
        "frequency_mean": dict(zip(("ru", "en"), freq.mean(axis=0).tolist())),
        "frequency_sd": dict(zip(("ru", "en"), freq.std(axis=0, ddof=1).tolist())),
        "unadjusted_G_same_subset": cluster_bootstrap(sub, stat_G, reps=reps, seed=seed),
    })
    families = sub["family_id"].unique()
    groups = [np.flatnonzero(sub["family_id"].eq(f).to_numpy()) for f in families]
    rng = np.random.default_rng(seed)
    draws, failures = [], {}
    for _ in range(reps):
        chosen = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[j] for j in chosen])
        draw, _, error = fit(idx)
        if error:
            failures[error] = failures.get(error, 0) + 1
        else:
            draws.append(draw)
    result.update({"reps_used": len(draws), "failed_draws": failures})
    # Не выдаём интервал по скрыто урезанной совокупности успешных прогонов.
    if len(draws) < 0.95 * reps:
        return {**result, "status": "insufficient_valid_bootstrap_draws"}
    bounds = np.percentile(np.asarray(draws), [2.5, 97.5], axis=0)
    result["coefficients"] = {
        name: {"point": float(beta[j]), "ci_low": float(bounds[0, j]), "ci_high": float(bounds[1, j])}
        for j, name in enumerate(names)
    }
    result["reference_ratio_exp_beta0"] = {
        "point": float(np.exp(beta[0])),
        "ci_low": float(np.exp(bounds[0, 0])), "ci_high": float(np.exp(bounds[1, 0])),
    }
    result["status"] = "ok" if not failures else "ok_with_failed_draws"
    return result


def project_precision(res: dict, target_families: int, target_concepts: int) -> dict:
    """Ориентировочная ширина интервала при большем объёме (§10).

    Грубая экстраполяция 1/sqrt(числа семей): бутстрэп по кластерам управляется
    прежде всего их количеством. Малый пилот даёт ненадёжную оценку разброса,
    поэтому это ориентир для выбора N, а не обещание итоговой точности.
    """
    hw = log_halfwidth(res)
    scaled = hw * np.sqrt(res["n_families"] / target_families)
    return {
        "pilot_log_halfwidth": hw,
        "pilot_families": res["n_families"],
        "target_families": target_families,
        "target_concepts": target_concepts,
        "projected_log_halfwidth": float(scaled),
        "target_log_halfwidth": config.PRECISION_TARGET_LOG_HALFWIDTH,
        "meets_target": bool(scaled <= config.PRECISION_TARGET_LOG_HALFWIDTH),
        "note": "экстраполяция 1/sqrt(семей); пересчитать на реальных данных",
    }


def stat_G_isosp(df: pd.DataFrame) -> float:
    """То же, что stat_G, но по счёту токенов термина с ведущим пробелом.

    Нужна для прямого ответа на вопрос, насколько величина H1 зависит от
    выбора режима предъявления, а не от языка.
    """
    d = np.log(df["S_ru_sg_isosp"] / df["S_en_sg_isosp"])
    return float(np.exp(d.mean()))


def stat_G_mode_gap(df: pd.DataFrame) -> float:
    """Отношение G(isolated_space) к G(isolated_0): цена выбора режима."""
    return stat_G_isosp(df) / stat_G(df)
