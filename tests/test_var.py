import numpy as np
import pandas as pd

from src.features.panels import info_set
from src.models.var import VarEPF


def test_var_recovers_weekly_pattern():
    idx = pd.date_range("2024-01-01", periods=120, freq="D", name="delivery_date")  # 2024-01-01 = poniedziałek
    phase = np.arange(120) % 7
    price = pd.DataFrame(100 + 15 * np.sin(2 * np.pi * phase / 7)[:, None] + 3 * np.arange(24)[None, :],
                         index=idx, columns=range(24))
    day = idx[100]
    pred = VarEPF(1, 60).predict(info_set({"price": price}, day), day)
    assert np.allclose(pred, price.loc[day].to_numpy(), atol=1e-3)