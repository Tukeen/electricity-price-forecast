import numpy as np
import pandas as pd
 
from src.operational.panels import build_forecast_panels, exog_complete
from src.operational.run_daily import merge_log, needs_exog
 
 
def _hourly(days_price=4, days_exog=5):
    rows = []
    for d, day in enumerate(pd.date_range("2026-01-01", periods=days_exog)):
        for h in range(24):
            rows.append({"delivery_date": day, "hour_local": h,
                         "price_eur_mwh": 100.0 + h if d < days_price else np.nan,
                         "load_fc_mw": 20000.0, "wind_onshore_fc_mw": 3000.0, "pv_fc_mw": 0.0})
    return pd.DataFrame(rows)
 
 
def test_panels_get_one_extra_day_without_prices():
    panels, day = build_forecast_panels(_hourly())
    assert day == pd.Timestamp("2026-01-05")
    assert len(panels["price"]) == 5 and panels["price"].loc[day].isna().all()
    assert exog_complete(panels, day)
 
 
def test_exog_incomplete_when_a_forecast_is_missing():
    df = _hourly()
    df.loc[df["delivery_date"] == "2026-01-05", "wind_onshore_fc_mw"] = np.nan
    panels, day = build_forecast_panels(df)
    assert not exog_complete(panels, day)
 
 
def test_merge_log_keeps_first_forecast():
    old = pd.DataFrame({"run_utc": ["a"], "delivery_date": ["2026-01-05"], "model": ["lear_x"],
                        "hour_local": [0], "y_pred": [1.0]})
    new = pd.DataFrame({"run_utc": ["b", "b"], "delivery_date": ["2026-01-05"] * 2,
                        "model": ["lear_x", "xgb_x"], "hour_local": [0, 0], "y_pred": [2.0, 3.0]})
    out = merge_log(old, new)
    assert len(out) == 2 and out.loc[out["model"] == "lear_x", "y_pred"].iloc[0] == 1.0
 
 
def test_needs_exog():
    assert needs_exog("sarimax") and needs_exog("xgb_x") and not needs_exog("naive_mixed") and not needs_exog("lear")