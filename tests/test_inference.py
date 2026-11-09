import numpy as np
import pandas as pd

from src.evaluation.inference import block_indices, gain_difference, holm


def test_block_indices_cover_n_and_stay_in_range():
    idx = block_indices(100, 7, np.random.default_rng(0))
    assert len(idx) == 100 and idx.min() >= 0 and idx.max() <= 99


def test_gain_difference_detects_larger_ml_gain():
    rng = np.random.default_rng(1)
    base = 20 + rng.normal(0, 2, 600)
    ml_x = base * 0.70 * (1 + rng.normal(0, 0.02, 600))
    ec_x = base * 0.85 * (1 + rng.normal(0, 0.02, 600))
    point, lo, hi, prob = gain_difference(base, ml_x, base, ec_x, B=300)
    assert -0.20 < point < -0.10 and hi < 0 and prob > 0.99


def test_gain_difference_zero_when_same():
    base = np.linspace(10, 30, 200)
    x = base * 0.8
    point, lo, hi, _ = gain_difference(base, x, base, x, B=100)
    assert abs(point) < 1e-12 and lo <= 0 <= hi


def test_holm_is_not_smaller_than_raw():
    p = pd.Series([0.01, 0.02, 0.04])
    adj = holm(p)
    assert (adj >= p).all() and adj.max() <= 1