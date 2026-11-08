import numpy as np
import pandas as pd

from src.features.panels import info_set
from src.models.xgb import XGBoostEPF, build_long


def _panel(n=120):
    idx = pd.date_range("2024-01-01", periods=n, freq="D", name="delivery_date")
    phase = np.arange(n) % 7
    base = 100 + 15 * np.sin(2 * np.pi * phase / 7)[:, None] + 3 * np.arange(24)[None, :]
    return pd.DataFrame(base, index=idx, columns=range(24))


def test_features_for_day_d_ignore_prices_of_day_d():
    p = np.arcsinh((_panel() - 100) / 20)
    day = p.index[100]
    q = p.copy()
    q.loc[day] = 123.0
    xa = build_long(p.loc[:day], {}, False)[0].xs(day, level="day")
    xb = build_long(q.loc[:day], {}, False)[0].xs(day, level="day")
    pd.testing.assert_frame_equal(xa, xb)


def test_xgb_learns_weekly_pattern():
    price = _panel()
    day = price.index[100]
    pred = XGBoostEPF(False, 60, dict(n_estimators=300, learning_rate=0.1)).predict(
        info_set({"price": price}, day), day)
    assert np.mean(np.abs(pred - price.loc[day].to_numpy())) < 6