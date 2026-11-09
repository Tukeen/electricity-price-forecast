"""LEAR_X z nieliniowymi cechami obciążenia rezydualnego (diagnostyka, nie model główny).

Do macierzy cech LEAR_X dodajemy dla każdej godziny doby D obciążenie rezydualne (zapotrzebowanie - wiatr - PV)
oraz 4 funkcje progowe max(0, r - k), gdzie k to kwantyle 20/40/60/80% obciążenia rezydualnego z okna uczącego.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.linear import LinearEPF, lear_matrix

KNOT_QUANTILES = (0.2, 0.4, 0.6, 0.8)


class LearNL(LinearEPF):
    def __init__(self, window_days: int):
        super().__init__("lear", True, window_days)
        self.name = "lear_nl"

    def _matrix(self, p, exog, win):
        X = lear_matrix(p, exog, True)
        r = (exog["load"] - exog["wind"] - exog["pv"]).reindex(p.index)
        knots = np.nanquantile(r.loc[win].to_numpy(), KNOT_QUANTILES)
        r.columns = [f"resid_h{h}" for h in r.columns]
        parts = [X, r]
        for q, k in zip(KNOT_QUANTILES, knots):
            hinge = (r - k).clip(lower=0)
            hinge.columns = [f"{c}_k{int(q * 100)}" for c in r.columns]
            parts.append(hinge)
        return pd.concat(parts, axis=1)