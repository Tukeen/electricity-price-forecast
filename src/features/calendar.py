"""Święta ustawowe w Polsce (dni wolne od pracy) jako cecha kalendarzowa."""
from __future__ import annotations

from functools import lru_cache

import holidays
import numpy as np
import pandas as pd


@lru_cache(maxsize=None)
def _pl_holidays(years: tuple[int, ...]):
    return holidays.country_holidays("PL", years=list(years))


def is_holiday(index) -> np.ndarray:
    """1.0 dla dni będących świętem ustawowym w Polsce (wg pakietu holidays), 0.0 w pozostałych."""
    idx = pd.DatetimeIndex(index)
    cal = _pl_holidays(tuple(range(int(idx.year.min()), int(idx.year.max()) + 1)))
    return np.array([d.date() in cal for d in idx], dtype=float)