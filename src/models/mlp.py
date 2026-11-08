"""Sieć neuronowa (MLP, sklearn): jeden model na wszystkie godziny, godzina i dzień tygodnia jako one-hot.

Okno kroczące, codzienne uczenie od nowa, asinh z parametrami z okna, średnia z kilku inicjalizacji.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from src.models.linear import EXOG, LAGS, Y_MARGIN, Z_CLIP, _asinh_params
from src.models.xgb import build_long

warnings.filterwarnings("ignore")
EXOG_COLS = ("load_h", "wind_h", "pv_h", "resid_load_h")
DEFAULT_PARAMS = dict(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                      validation_fraction=0.1, n_iter_no_change=10, max_iter=200)


def _design(X: pd.DataFrame) -> pd.DataFrame:
    hour = X["hour"].astype(pd.CategoricalDtype(range(24)))
    dow = X["dow"].astype(pd.CategoricalDtype(range(7)))
    onehot = pd.concat([pd.get_dummies(hour, prefix="h"), pd.get_dummies(dow, prefix="d")], axis=1).astype(float)
    return pd.concat([X.drop(columns=["hour", "dow"]), onehot], axis=1)


class MLPEPF:
    def __init__(self, use_exog: bool, window_days: int, n_seeds: int = 3, params: dict | None = None):
        self.use_exog, self.window_days, self.n_seeds = use_exog, window_days, n_seeds
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.name = "mlp_x" if use_exog else "mlp"

    def predict(self, info: dict, day: pd.Timestamp) -> np.ndarray:
        price = info["price"]
        win = price.index[-self.window_days:]
        a, b = _asinh_params(price.loc[win].to_numpy().ravel())
        p = np.arcsinh((price.iloc[-(self.window_days + max(LAGS)):] - a) / b)
        p = p.reindex(p.index.append(pd.DatetimeIndex([day])))
        exog = {k: info[k] for k in EXOG} if self.use_exog else {}
        X, y = build_long(p, exog, self.use_exog)
        D = _design(X)

        days = D.index.get_level_values("day")
        exog_cols = [c for c in D.columns if c in EXOG_COLS]
        core = [c for c in D.columns if c not in exog_cols]
        train = days.isin(win) & ~np.isnan(y) & D[core].notna().all(axis=1).to_numpy()
        xday = D.xs(day, level="day")
        if train.sum() < 24 * 30 or xday[core].isna().to_numpy().any():
            return np.full(24, np.nan)

        Dtr = D[train]
        mu, sd = Dtr.mean(), Dtr.std(ddof=0).replace(0, 1.0)
        Ztr = ((Dtr - mu) / sd).fillna(0.0).to_numpy()
        zday = ((xday - mu) / sd).clip(-Z_CLIP, Z_CLIP).fillna(0.0).to_numpy()
        ytr = y[train]
        preds = [MLPRegressor(random_state=s, **self.params).fit(Ztr, ytr).predict(zday)
                 for s in range(self.n_seeds)]
        yhat = np.clip(np.mean(preds, axis=0), ytr.min() - Y_MARGIN, ytr.max() + Y_MARGIN)
        return a + b * np.sinh(yhat)