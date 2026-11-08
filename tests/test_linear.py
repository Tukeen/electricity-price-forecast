import numpy as np
import pandas as pd

from src.features.panels import info_set
from src.models.linear import LinearEPF


def _periodic_panel(n=120):
    idx = pd.date_range("2024-01-01", periods=n, freq="D", name="delivery_date")
    phase = np.arange(n) % 7
    base = 100 + 15 * np.sin(2 * np.pi * phase / 7)[:, None] + 3 * np.arange(24)[None, :]
    return pd.DataFrame(base, index=idx, columns=range(24))


def test_arx_recovers_weekly_pattern():
    price = _periodic_panel()
    day = price.index[100]
    pred = LinearEPF("arx", False, 60).predict(info_set({"price": price}, day), day)
    assert np.allclose(pred, price.loc[day].to_numpy(), atol=1e-4)


def test_lear_returns_nan_when_too_few_rows():
    price = _periodic_panel()
    day = price.index[100]
    pred = LinearEPF("lear", False, 30).predict(info_set({"price": price}, day), day)
    assert pred.shape == (24,) and np.isnan(pred).all()