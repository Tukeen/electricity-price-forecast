import numpy as np
import pandas as pd

from src.features.panels import info_set
from src.models.mlp import MLPEPF


def test_mlp_learns_weekly_pattern():
    idx = pd.date_range("2024-01-01", periods=120, freq="D", name="delivery_date")
    phase = np.arange(120) % 7
    price = pd.DataFrame(100 + 15 * np.sin(2 * np.pi * phase / 7)[:, None] + 3 * np.arange(24)[None, :],
                         index=idx, columns=range(24))
    day = idx[100]
    pred = MLPEPF(False, 60, n_seeds=1).predict(info_set({"price": price}, day), day)
    assert np.mean(np.abs(pred - price.loc[day].to_numpy())) < 10   # test dymny: próg dobrany na oko