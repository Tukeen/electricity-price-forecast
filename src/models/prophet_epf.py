"""Prophet: szereg godzinowy (asinh), sezonowość dobowa i tygodniowa, opcjonalnie regresory egzogeniczne.
Codzienna reestymacja na oknie kroczącym (Prophet nie ma szybkiej aktualizacji stanu)."""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from src.models.linear import EXOG, Y_MARGIN, Z_CLIP, _asinh_params

warnings.filterwarnings("ignore")
for _name in ("cmdstanpy", "prophet"):
    logging.getLogger(_name).setLevel(logging.ERROR)


class ProphetEPF:
    def __init__(self, use_exog: bool, window_days: int, changepoint_prior_scale: float = 0.05):
        self.use_exog, self.window_days = use_exog, window_days
        self.cps = changepoint_prior_scale
        self.name = "prophet_x" if use_exog else "prophet"

    def _regressors(self, info: dict, win, day):
        tr, te = {}, {}
        for k in EXOG:
            x = info[k].reindex(win).to_numpy(dtype=float).ravel()
            xd = info[k].loc[day].to_numpy(dtype=float)
            mu, sd = np.nanmean(x), np.nanstd(x)
            sd = sd if sd > 0 else 1.0
            tr[k] = np.nan_to_num((x - mu) / sd, nan=0.0)
            te[k] = np.clip(np.nan_to_num((xd - mu) / sd, nan=0.0), -Z_CLIP, Z_CLIP)
        return tr, te

    def predict(self, info: dict, day: pd.Timestamp) -> np.ndarray:
        from prophet import Prophet  # import w procesie roboczym

        price = info["price"]
        win = price.index[-self.window_days:]
        a, b = _asinh_params(price.loc[win].to_numpy().ravel())
        hours = np.arange(24).astype("timedelta64[h]")
        ds_tr = (win.to_numpy().astype("datetime64[ns]")[:, None] + hours[None, :]).ravel()
        y_tr = np.arcsinh((price.loc[win].to_numpy().ravel() - a) / b)

        train = pd.DataFrame({"ds": ds_tr, "y": y_tr})
        future = pd.DataFrame({"ds": pd.Timestamp(day) + pd.to_timedelta(np.arange(24), unit="h")})

        m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True,
                    changepoint_prior_scale=self.cps, uncertainty_samples=0)
        if self.use_exog:
            tr, te = self._regressors(info, win, day)
            for k in EXOG:
                m.add_regressor(k, standardize=False)
                train[k], future[k] = tr[k], te[k]
        try:
            m.fit(train)
            yhat = m.predict(future)["yhat"].to_numpy()
        except Exception as exc:
            print(f"  {self.name}: błąd {pd.Timestamp(day).date()}: {exc}")
            return np.full(24, np.nan)
        yhat = np.clip(yhat, y_tr.min() - Y_MARGIN, y_tr.max() + Y_MARGIN)
        return a + b * np.sinh(yhat)