"""SARIMA / SARIMAX na szeregu godzinowym (asinh) z dummies dnia tygodnia.

sarima  = SARIMA + dummies dnia tygodnia
sarimax = j.w. + prognozy zapotrzebowania, wiatru i PV dla doby D (wartości z doby D, nie z przyszłości)
Parametry estymowane na początku bloku dób; w bloku stan filtra aktualizowany codziennie (extend).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.models.linear import EXOG, Y_MARGIN, Z_CLIP, _asinh_params

warnings.filterwarnings("ignore")


class SarimaEPF:
    def __init__(self, use_exog: bool, window_days: int, order=(2, 0, 1), seasonal=(1, 0, 1, 24), maxiter=100):
        self.use_exog, self.window_days = use_exog, window_days
        self.order, self.seasonal, self.maxiter = order, seasonal, maxiter
        self.name = "sarimax" if use_exog else "sarima"

    def _exog(self, panels: dict, days: pd.DatetimeIndex, n_train: int) -> np.ndarray:
        dow = days.day_of_week.to_numpy()
        X = np.column_stack([np.repeat((dow == d).astype(float), 24) for d in range(1, 7)])
        if self.use_exog:
            fund = np.column_stack([panels[k].reindex(days).to_numpy().ravel() for k in EXOG])
            tr = fund[: n_train * 24]
            mu, sd = np.nanmean(tr, axis=0), np.nanstd(tr, axis=0)
            sd[sd == 0] = 1.0
            fund = np.clip(np.nan_to_num((fund - mu) / sd, nan=0.0), -Z_CLIP, Z_CLIP)
            X = np.column_stack([X, fund])
        return X

    def _fit(self, y_tr, X_train, block_start):
        attempts = [
            ("podstawowy", self.order, self.seasonal, "lbfgs"),
            ("powell", self.order, self.seasonal, "powell"),
            ("prostszy rząd", (1, 0, 1), (1, 0, 1, 24), "lbfgs"),
        ]
        for label, order, seasonal, method in attempts:
            try:
                res = SARIMAX(y_tr, exog=X_train, order=order, seasonal_order=seasonal, trend="c",
                              concentrate_scale=True).fit(disp=False, maxiter=self.maxiter, method=method)
            except Exception as exc:
                print(f"  {self.name}: blok od {block_start}: wariant '{label}' nie powiódł się ({exc})")
                continue
            if label != "podstawowy":
                print(f"  {self.name}: blok od {block_start}: użyto wariantu '{label}'")
            return res
        return None

    def forecast_block(self, panels: dict, days: pd.DatetimeIndex) -> list[np.ndarray]:
        price = panels["price"]
        train_days = price.loc[: days[0] - pd.Timedelta(days=1)].index[-self.window_days:]
        a, b = _asinh_params(price.loc[train_days].to_numpy().ravel())
        y_tr = np.arcsinh((price.loc[train_days].to_numpy().ravel() - a) / b)
        lo, hi = y_tr.min() - Y_MARGIN, y_tr.max() + Y_MARGIN

        X = self._exog(panels, train_days.append(days), len(train_days))
        n = len(train_days) * 24
        res = self._fit(y_tr, X[:n], days[0].date())
        if res is None:
            return [np.full(24, np.nan) for _ in days]

        out = []
        for i, day in enumerate(days):
            Xd = X[n + i * 24: n + (i + 1) * 24]
            try:
                f = np.clip(np.asarray(res.forecast(steps=24, exog=Xd)), lo, hi)
                out.append(a + b * np.sinh(f))
                actual = np.arcsinh((price.loc[day].to_numpy() - a) / b)  # znane dopiero po dobie D
                res = res.extend(actual, exog=Xd)
            except Exception as exc:
                print(f"  {self.name}: błąd {day.date()}: {exc}")
                out.extend(np.full(24, np.nan) for _ in range(len(days) - len(out)))
                break
        return out