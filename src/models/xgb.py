"""Gradient boosting (XGBoost): jeden model na wszystkie godziny, godzina jako cecha.

Okno kroczące, codzienne uczenie od nowa, asinh z parametrami z okna.
Drzewa przewidują wyłącznie wartości z zakresu celu z okna uczenia (brak ekstrapolacji).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from src.features.calendar import is_holiday

from src.models.linear import EXOG, LAGS, _asinh_params

warnings.filterwarnings("ignore")
DEFAULT_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, tree_method="hist", n_jobs=1, random_state=42, verbosity=0,
)


def _flat(panel: pd.DataFrame) -> np.ndarray:
    return panel.to_numpy(dtype=float).ravel()


def build_long(p: pd.DataFrame, exog: dict, use_exog: bool) -> tuple[pd.DataFrame, np.ndarray]:
    """Cechy w formacie długim (doba x godzina). Panel p (asinh) może zawierać wiersz doby D z NaN."""
    days = p.index
    n = len(days)
    cols = {"hour": np.tile(np.arange(24), n), "dow": np.repeat(days.day_of_week.to_numpy(), 24)}
    cols["hol"] = np.repeat(is_holiday(days), 24)
    for k in LAGS:
        cols[f"p_l{k}"] = _flat(p.shift(k))
    for name, agg in (("min", p.min(axis=1)), ("max", p.max(axis=1)), ("mean", p.mean(axis=1))):
        cols[f"p_{name}_l1"] = np.repeat(agg.shift(1).to_numpy(), 24)
    cols["p_h23_l1"] = np.repeat(p[23].shift(1).to_numpy(), 24)
    if use_exog:
        for k in EXOG:
            cols[f"{k}_h"] = _flat(exog[k].reindex(days))
        cols["resid_load_h"] = cols["load_h"] - cols["wind_h"] - cols["pv_h"]
    X = pd.DataFrame(cols, index=pd.MultiIndex.from_product([days, range(24)], names=["day", "hour_"]))
    return X, _flat(p)


class XGBoostEPF:
    def __init__(self, use_exog: bool, window_days: int, params: dict | None = None):
        self.use_exog, self.window_days = use_exog, window_days
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.name = "xgb_x" if use_exog else "xgb"

    def predict(self, info: dict, day: pd.Timestamp) -> np.ndarray:
        price = info["price"]
        win = price.index[-self.window_days:]
        a, b = _asinh_params(price.loc[win].to_numpy().ravel())

        p = np.arcsinh((price.iloc[-(self.window_days + max(LAGS)):] - a) / b)
        p = p.reindex(p.index.append(pd.DatetimeIndex([day])))
        exog = {k: info[k] for k in EXOG} if self.use_exog else {}
        X, y = build_long(p, exog, self.use_exog)

        train = X.index.get_level_values("day").isin(win) & ~np.isnan(y)
        if train.sum() < 24 * 30:
            return np.full(24, np.nan)
        model = XGBRegressor(**self.params).fit(X[train], y[train])
        yhat = model.predict(X.xs(day, level="day"))
        return a + b * np.sinh(yhat)