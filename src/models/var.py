"""VAR(p) na 24 godzinach doby: OLS, asinh, okno kroczące, dummies dnia tygodnia (baza: poniedziałek)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.linear import Y_MARGIN, _asinh_params, _dummies, _standardize


def var_matrix(p: pd.DataFrame, lags: int) -> pd.DataFrame:
    parts = []
    for k in range(1, lags + 1):
        s = p.shift(k)
        s.columns = [f"p_l{k}_h{h}" for h in p.columns]
        parts.append(s)
    parts.append(_dummies(p.index, tuple(range(1, 7))))
    return pd.concat(parts, axis=1)


class VarEPF:
    def __init__(self, lags: int, window_days: int):
        self.lags, self.window_days = lags, window_days
        self.name = f"var{lags}"

    def predict(self, info: dict, day: pd.Timestamp) -> np.ndarray:
        price = info["price"]
        win = price.index[-self.window_days:]
        a, b = _asinh_params(price.loc[win].to_numpy().ravel())
        p = np.arcsinh((price.iloc[-(self.window_days + self.lags):] - a) / b)
        p = p.reindex(p.index.append(pd.DatetimeIndex([day])))

        X, Y = var_matrix(p, self.lags), p.loc[win]
        Xtr = X.loc[win]
        ok = Xtr.notna().all(axis=1) & Y.notna().all(axis=1)
        xday = X.loc[[day]]
        if ok.sum() < X.shape[1] + 5 or xday.isna().to_numpy().any():
            return np.full(24, np.nan)

        Z, z = _standardize(Xtr[ok], xday)
        A = np.column_stack([np.ones(len(Z)), Z])
        B = np.linalg.lstsq(A, Y[ok].to_numpy(), rcond=None)[0]
        yhat = B[0] + (z @ B[1:])[0]
        lo, hi = float(Y.min().min()), float(Y.max().max())
        yhat = np.clip(yhat, lo - Y_MARGIN, hi + Y_MARGIN)
        return a + b * np.sinh(yhat)