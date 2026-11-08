from __future__ import annotations

import numpy as np
import pandas as pd


class Naive:
    """daily: D-1 | weekly: D-7 | mixed: pn/sb/nd -> D-7, wt-pt -> D-1 (benchmark z literatury)."""

    def __init__(self, variant: str):
        self.variant = variant
        self.name = f"naive_{variant}"

    def predict(self, info: dict, day: pd.Timestamp) -> np.ndarray:
        if self.variant == "daily":
            lag = 1
        elif self.variant == "weekly":
            lag = 7
        else:
            lag = 7 if day.day_of_week in (0, 5, 6) else 1
        return info["price"].loc[day - pd.Timedelta(days=lag)].to_numpy()