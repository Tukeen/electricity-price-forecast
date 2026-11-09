"""Wnioskowanie: poprawka Holma i bootstrap blokowy po dobach (H3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def holm(p: pd.Series) -> pd.Series:
    if len(p) <= 1:
        return p
    return pd.Series(multipletests(p.to_numpy(), method="holm")[1], index=p.index)


def block_indices(n: int, block: int, rng) -> np.ndarray:
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=nb)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def gain_difference(base_ml, x_ml, base_ec, x_ec, block: int = 7, B: int = 2000, seed: int = 42):
    """Delta = (MAE_x/MAE_base - 1)_ML - (MAE_x/MAE_base - 1)_ECON, bootstrap blokowy po dobach.

    Delta < 0: klasa ML zyskuje po dodaniu zmiennych egzogenicznych więcej.
    Zwraca (estymata, dolna 95%, górna 95%, udział prób bootstrap z Delta < 0).
    """
    arrs = [np.asarray(a, dtype=float) for a in (base_ml, x_ml, base_ec, x_ec)]
    n = len(arrs[0])

    def delta(idx):
        m = [a[idx].mean() for a in arrs]
        return (m[1] / m[0] - 1) - (m[3] / m[2] - 1)

    rng = np.random.default_rng(seed)
    point = delta(np.arange(n))
    boots = np.array([delta(block_indices(n, block, rng)) for _ in range(B)])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(point), float(lo), float(hi), float((boots < 0).mean())