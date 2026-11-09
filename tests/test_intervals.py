import numpy as np
import pandas as pd

from src.evaluation.intervals import daily_interval_metrics, empirical_intervals


def _panels(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", name="delivery_date")
    true = pd.DataFrame(100.0, index=idx, columns=range(24))
    pred = true + rng.normal(0, 5, (n, 24))
    return pred, true


def test_empirical_intervals_have_nominal_coverage():
    pred, true = _panels()
    lo, hi = empirical_intervals(pred, true, 0.9)
    m = daily_interval_metrics(lo, hi, true, 0.9)
    assert 0.87 < m["coverage"].mean() < 0.93
    assert m["coverage"].notna().sum() == 340


def test_interval_for_day_does_not_use_its_own_error():
    pred, true = _panels()
    lo, hi = empirical_intervals(pred, true, 0.9)
    true2 = true.copy()
    true2.iloc[200] += 1000.0  # błąd doby 200 jest znany dopiero po tej dobie
    lo2, hi2 = empirical_intervals(pred, true2, 0.9)
    assert np.allclose(hi.iloc[200], hi2.iloc[200])
    assert not np.allclose(hi.iloc[201], hi2.iloc[201])