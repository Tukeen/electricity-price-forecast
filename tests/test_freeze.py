import numpy as np
import pandas as pd

from src.evaluation.freeze import data_fingerprint
from src.evaluation.report import holdout_groups


def _panels():
    idx = pd.date_range("2026-01-01", periods=10, freq="D", name="delivery_date")
    return {"price": pd.DataFrame(np.arange(240.0).reshape(10, 24), index=idx, columns=range(24)),
            "load": pd.DataFrame(1.0, index=idx, columns=range(24))}


def test_fingerprint_stable_and_sensitive():
    end = pd.Timestamp("2026-01-10")
    a = data_fingerprint(_panels(), end)
    assert a == data_fingerprint(_panels(), end)
    q = _panels()
    q["load"].iloc[3, 5] += 1e-6
    assert data_fingerprint(q, end) != a


def test_fingerprint_ignores_days_after_end():
    end = pd.Timestamp("2026-01-05")
    a = data_fingerprint(_panels(), end)
    q = _panels()
    q["price"].iloc[8] += 5.0
    assert data_fingerprint(q, end) == a


def test_holdout_groups_quarters_and_months():
    g = holdout_groups(pd.date_range("2026-01-01", "2026-12-31"))
    assert g["całość"].all() and g["Q1"].sum() == 90 and g["02"].sum() == 28 and g["12"].sum() == 31