"""Modele z opóźnieniami: ARX (OLS, zestaw ekspercki) i LEAR (Lasso, Lago i in. 2021).

Osobny model na każdą godzinę doby, okno kroczące, transformacja asinh z parametrami z okna.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsIC

warnings.filterwarnings("ignore")
LAGS = (1, 2, 3, 7)
EXOG = ("load", "wind", "pv")


def _asinh_params(prices: np.ndarray) -> tuple[float, float]:
    a = float(np.median(prices))
    b = float(1.4826 * np.median(np.abs(prices - a)))
    return a, b


def _dummies(index: pd.DatetimeIndex, days: tuple[int, ...]) -> pd.DataFrame:
    dow = index.day_of_week
    return pd.DataFrame({f"dow{d}": (dow == d).astype(float) for d in days}, index=index)


def lear_matrix(p: pd.DataFrame, exog: dict, use_exog: bool) -> pd.DataFrame:
    """Wspólna macierz cech dla wszystkich godzin: ceny z D-1, D-2, D-3, D-7 (24 h każda),
    dzień tygodnia oraz (opcjonalnie) prognozy egzogeniczne na dobę D (24 h każda)."""
    parts = []
    for k in LAGS:
        s = p.shift(k)
        s.columns = [f"p_l{k}_h{h}" for h in p.columns]
        parts.append(s)
    parts.append(_dummies(p.index, tuple(range(7))))
    if use_exog:
        for name in EXOG:
            e = exog[name].reindex(p.index).copy()
            e.columns = [f"{name}_h{h}" for h in e.columns]
            parts.append(e)
    return pd.concat(parts, axis=1)


def arx_matrix(p: pd.DataFrame, exog: dict, h: int) -> pd.DataFrame:
    """Zestaw ekspercki dla godziny h: ceny z tej godziny (D-1, D-2, D-3, D-7), minimum z D-1,
    cena z godz. 23 w D-1, dummies pn/sb/nd, prognozy egzogeniczne dla godziny h doby D."""
    X = pd.DataFrame({f"p_l{k}": p[h].shift(k) for k in LAGS})
    X["p_min_l1"] = p.min(axis=1).shift(1)
    X["p_h23_l1"] = p[23].shift(1)
    X = X.join(_dummies(p.index, (0, 5, 6)))
    for name, e in exog.items():
        X[f"{name}_h"] = e[h].reindex(p.index)
    return X


def _standardize(Xtr: pd.DataFrame, xnew: pd.DataFrame):
    mu, sd = Xtr.mean(), Xtr.std(ddof=0).replace(0, 1.0)
    return ((Xtr - mu) / sd).to_numpy(), ((xnew - mu) / sd).to_numpy()


def _ols(Xtr: pd.DataFrame, ytr: pd.Series, xnew: pd.DataFrame) -> float:
    Z, z = _standardize(Xtr, xnew)
    A = np.column_stack([np.ones(len(Z)), Z])
    beta = np.linalg.lstsq(A, ytr.to_numpy(), rcond=None)[0]
    return float(beta[0] + (z @ beta[1:])[0])


class LinearEPF:
    def __init__(self, kind: str, use_exog: bool, window_days: int):
        assert kind in ("arx", "lear")
        self.kind, self.use_exog, self.window_days = kind, use_exog, window_days
        self.name = f"{kind}{'_x' if use_exog else ''}"

    def predict(self, info: dict, day: pd.Timestamp) -> np.ndarray:
        price = info["price"]
        win = price.index[-self.window_days:]
        a, b = _asinh_params(price.loc[win].to_numpy().ravel())

        hist = price.iloc[-(self.window_days + 7):]
        p = np.arcsinh((hist - a) / b)
        p = p.reindex(p.index.append(pd.DatetimeIndex([day])))  # wiersz D z NaN w miejscu nieznanych cen
        exog = {k: info[k] for k in EXOG} if self.use_exog else {}

        yhat = self._lear(p, exog, win, day) if self.kind == "lear" else self._arx(p, exog, win, day)
        return a + b * np.sinh(yhat)  # powrót do EUR/MWh

    def _lear(self, p, exog, win, day) -> np.ndarray:
        X = lear_matrix(p, exog, self.use_exog)
        Y = p.loc[win]
        Xtr = X.loc[win]
        ok = Xtr.notna().all(axis=1) & Y.notna().all(axis=1)
        Xtr, Ytr, xday = Xtr[ok], Y[ok], X.loc[[day]]
        if len(Xtr) < X.shape[1] + 10 or xday.isna().to_numpy().any():
            return np.full(24, np.nan)
        Z, z = _standardize(Xtr, xday)
        out = np.empty(24)
        for h in range(24):
            out[h] = LassoLarsIC(criterion="aic").fit(Z, Ytr[h].to_numpy()).predict(z)[0]
        return out

    def _arx(self, p, exog, win, day) -> np.ndarray:
        out = np.full(24, np.nan)
        for h in range(24):
            X = arx_matrix(p, exog, h)
            y = p.loc[win, h]
            Xtr = X.loc[win]
            ok = Xtr.notna().all(axis=1) & y.notna()
            xday = X.loc[[day]]
            if ok.sum() < X.shape[1] + 10 or xday.isna().to_numpy().any():
                continue
            out[h] = _ols(Xtr[ok], y[ok], xday)
        return out