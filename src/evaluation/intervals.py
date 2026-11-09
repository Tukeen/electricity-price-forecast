"""Prognozy przedziałowe: empiryczne kwantyle błędów z ostatnich `window` dób (wspólne dla wszystkich modeli).

Dla doby D przedział korzysta wyłącznie z błędów z dób < D (błąd doby D-1 jest znany przed prognozą doby D).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

LEVELS = (0.8, 0.9)


def empirical_intervals(pred: pd.DataFrame, true: pd.DataFrame, level: float,
                        window: int = 90, min_days: int = 60):
    """Zwraca (dolna, górna) granicę jako panele doby x 24 godziny; NaN, gdy za mało historii."""
    e = (pred - true.reindex(pred.index)).to_numpy()   # błąd = prognoza - wartość rzeczywista
    alpha = 1 - level
    lo = pd.DataFrame(np.nan, index=pred.index, columns=pred.columns)
    hi = lo.copy()
    p = pred.to_numpy()
    for i in range(min_days, len(pred)):
        past = e[max(0, i - window):i]
        past = past[~np.isnan(past).any(axis=1)]
        if len(past) < min_days:
            continue
        q_lo = np.quantile(past, alpha / 2, axis=0)
        q_hi = np.quantile(past, 1 - alpha / 2, axis=0)
        lo.iloc[i] = p[i] - q_hi       # y = pred - błąd
        hi.iloc[i] = p[i] - q_lo
    return lo, hi


def daily_interval_metrics(lo: pd.DataFrame, hi: pd.DataFrame, y: pd.DataFrame, level: float) -> pd.DataFrame:
    """Dobowe średnie: pokrycie, szerokość i wynik przedziałowy (Gneiting-Raftery)."""
    ok = lo.notna() & hi.notna() & y.notna()
    l, u, v = lo.where(ok), hi.where(ok), y.where(ok)
    alpha = 1 - level
    width = u - l
    miss = (l - v).clip(lower=0) + (v - u).clip(lower=0)
    score = width + (2 / alpha) * miss
    cover = ((v >= l) & (v <= u)).astype(float).where(ok)
    return pd.DataFrame({"coverage": cover.mean(axis=1), "width": width.mean(axis=1),
                         "score": score.mean(axis=1)})


def dm_scores(s1: pd.Series, s2: pd.Series, maxlags: int = 7):
    """Test Diebolda-Mariano na dobowych wynikach przedziałowych (DM < 0: s1 ma mniejszą stratę)."""
    d = (s1 - s2).dropna()
    res = sm.OLS(d.to_numpy(), np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(res.tvalues[0]), float(res.pvalues[0])