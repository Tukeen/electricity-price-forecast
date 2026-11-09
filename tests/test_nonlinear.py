import numpy as np
import pandas as pd

from src.models.linear_nl import LearNL


def _panels(n=60):
    idx = pd.date_range("2024-01-01", periods=n, freq="D", name="delivery_date")
    rng = np.random.default_rng(0)

    def mk(scale):
        return pd.DataFrame(rng.normal(100, 10, (n, 24)) * scale, index=idx, columns=range(24))

    return idx, mk(1.0), {"load": mk(1.0), "wind": mk(0.1), "pv": mk(0.1)}


def test_nl_matrix_has_expected_shape_and_nonnegative_hinges():
    idx, p, exog = _panels()
    X = LearNL(30)._matrix(p, exog, idx[-30:])
    hinge = [c for c in X.columns if "_k" in c]
    assert X.shape == (60, 176 + 24 * 5) and len(hinge) == 96
    assert (X[hinge] >= 0).all().all()


def test_nl_knots_use_only_window():
    idx, p, exog = _panels()
    m = LearNL(30)
    X1 = m._matrix(p, exog, idx[10:40])
    exog2 = {k: v.copy() for k, v in exog.items()}
    exog2["load"].iloc[50:] += 500.0  # poza oknem wyznaczającym węzły
    X2 = m._matrix(p, exog2, idx[10:40])
    cols = [c for c in X1.columns if "_k" in c]
    assert np.allclose(X1.loc[idx[10:40], cols], X2.loc[idx[10:40], cols])