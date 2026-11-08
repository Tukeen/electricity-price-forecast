from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def errors(pred: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    return pred - price.loc[pred.index]


def mae(e: pd.DataFrame) -> float:
    return float(np.nanmean(np.abs(e.to_numpy())))


def rmse(e: pd.DataFrame) -> float:
    return float(np.sqrt(np.nanmean(e.to_numpy() ** 2)))


def me(e: pd.DataFrame) -> float:
    """Średni błąd (obciążenie): prognoza minus wartość rzeczywista."""
    return float(np.nanmean(e.to_numpy()))


def diebold_mariano(e1: pd.DataFrame, e2: pd.DataFrame, loss: str = "abs", maxlags: int = 7):
    """Test Diebolda-Mariano na dobowych stratach (średnia z 24 godzin), błędy standardowe HAC.

    Statystyka > 0 oznacza, że model 1 ma większą stratę (model 2 jest lepszy).
    Zwraca (statystyka, p dwustronne).
    """
    f = (lambda e: e.abs()) if loss == "abs" else (lambda e: e ** 2)
    d = (f(e1).mean(axis=1) - f(e2).mean(axis=1)).dropna()
    res = sm.OLS(d.to_numpy(), np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(res.tvalues[0]), float(res.pvalues[0])