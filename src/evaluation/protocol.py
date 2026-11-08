"""Protokół prognoz kroczących: dla doby D model widzi wyłącznie info_set(D)."""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import os
from joblib import Parallel, delayed

from src.features.panels import info_set


def _predict_chunk(model, panels, chunk):
    return [np.asarray(model.predict(info_set(panels, day), day), dtype=float) for day in chunk]


def rolling_forecast(model, panels: dict, days, window_days: int, refit_every: int = 1,
                     n_jobs: int = 1) -> pd.DataFrame:
    """Prognozy kroczące (doby x 24 h). Modele bez fit() liczone równolegle po dobach."""
    days = list(days)
    if hasattr(model, "fit") or n_jobs == 1:
        preds = []
        for i, day in enumerate(days):
            info = info_set(panels, day)
            if hasattr(model, "fit") and i % refit_every == 0:
                model.fit(info, info["price"].index[-window_days:])
            preds.append(np.asarray(model.predict(info, day), dtype=float))
            if (i + 1) % 250 == 0:
                print(f"  {getattr(model, 'name', 'model')}: {i + 1}/{len(days)} dób")
    else:
        n = n_jobs if n_jobs > 0 else (os.cpu_count() or 1)
        chunks = [c for c in np.array_split(np.arange(len(days)), n * 4) if len(c)]
        results = Parallel(n_jobs=n_jobs)(
            delayed(_predict_chunk)(model, panels, [days[i] for i in c]) for c in chunks
        )
        preds = [p for r in results for p in r]
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


def load_forecasts(db_path, name: str) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute("SELECT delivery_date, hour_local, y_pred FROM forecasts WHERE model = ?", [name]).df()
    con.close()
    if df.empty:
        return df
    df["delivery_date"] = pd.to_datetime(df["delivery_date"])
    return df.pivot(index="delivery_date", columns="hour_local", values="y_pred").sort_index()


def _block(model, panels, block):
    return model.forecast_block(panels, block)


def rolling_forecast_blocks(model, panels: dict, days, block_days: int, n_jobs: int) -> pd.DataFrame:
    """Modele estymowane raz na blok dób; bloki liczone równolegle."""
    days = list(days)
    blocks = [days[i:i + block_days] for i in range(0, len(days), block_days)]
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_block)(model, panels, pd.DatetimeIndex(b)) for b in blocks
    )
    preds = [p for r in results for p in r]
    return pd.DataFrame(preds, index=pd.DatetimeIndex(days, name="delivery_date"), columns=range(24))