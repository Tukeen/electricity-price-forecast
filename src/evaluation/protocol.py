"""Protokół prognoz kroczących: dla doby D model widzi wyłącznie info_set(D)."""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from src.features.panels import info_set


def rolling_forecast(model, panels: dict, days, window_days: int, refit_every: int = 1) -> pd.DataFrame:
    """Zwraca panel prognoz (doby x 24 godziny). Model może mieć metodę fit(info, train_days)."""
    preds = []
    for i, day in enumerate(days):
        info = info_set(panels, day)
        if hasattr(model, "fit") and i % refit_every == 0:
            model.fit(info, info["price"].index[-window_days:])
        preds.append(np.asarray(model.predict(info, day), dtype=float))
    return pd.DataFrame(preds, index=pd.DatetimeIndex(days, name="delivery_date"), columns=range(24))


def save_forecasts(db_path, name: str, pred: pd.DataFrame, price: pd.DataFrame) -> None:
    key = ["delivery_date", "hour_local"]
    y_pred = pred.reset_index().melt(id_vars="delivery_date", var_name="hour_local", value_name="y_pred")
    y_true = (price.loc[pred.index].reset_index()
              .melt(id_vars="delivery_date", var_name="hour_local", value_name="y_true"))
    long = y_pred.merge(y_true, on=key)
    long.insert(0, "model", name)

    con = duckdb.connect(str(db_path))
    con.execute("""CREATE TABLE IF NOT EXISTS forecasts
                   (model VARCHAR, delivery_date DATE, hour_local INTEGER, y_pred DOUBLE, y_true DOUBLE)""")
    con.execute("DELETE FROM forecasts WHERE model = ?", [name])
    con.register("long_df", long)
    con.execute("""INSERT INTO forecasts
                   SELECT model, CAST(delivery_date AS DATE), CAST(hour_local AS INTEGER), y_pred, y_true
                   FROM long_df""")
    con.close()