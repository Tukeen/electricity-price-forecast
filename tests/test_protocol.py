import numpy as np
import pandas as pd

from src.evaluation.metrics import diebold_mariano
from src.features.panels import info_set
from src.models.naive import Naive


def _panels(n=14):
    idx = pd.date_range("2024-01-01", periods=n, freq="D", name="delivery_date")
    price = pd.DataFrame(np.arange(n)[:, None] * np.ones((1, 24)), index=idx)
    return {"price": price, "load": price * 0 + 1}


def test_info_set_hides_target_day_prices():
    day = pd.Timestamp("2024-01-10")
    info = info_set(_panels(), day)
    assert info["price"].index.max() == day - pd.Timedelta(days=1)
    assert info["load"].index.max() == day  # prognozy egzogeniczne na dobę D są dostępne


def test_naive_mixed_lags():
    p = _panels()
    mon, tue = pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-09")
    m = Naive("mixed")
    assert m.predict(info_set(p, mon), mon)[0] == 0   # poniedziałek: doba 01.01
    assert m.predict(info_set(p, tue), tue)[0] == 7   # wtorek: doba 08.01


def test_dm_detects_better_model():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    bad = pd.DataFrame(rng.normal(0, 2.0, (200, 24)), index=idx)
    good = pd.DataFrame(rng.normal(0, 1.0, (200, 24)), index=idx)
    stat, p = diebold_mariano(bad, good)
    assert stat > 0 and p < 0.01