import pandas as pd

from src.evaluation.tune import choose


def test_choose_keeps_default_unless_gain_over_one_percent():
    g = pd.DataFrame({"label": ["a", "b", "c"], "mae": [10.0, 9.95, 9.0]})
    assert choose(g)["label"] == "c"     # najlepsza poprawia o 10%
    g = pd.DataFrame({"label": ["a", "b"], "mae": [10.0, 9.95]})
    assert choose(g)["label"] == "a"     # zysk 0,5% < 1%: zostaje domyślna
    g = pd.DataFrame({"label": ["a", "b"], "mae": [float("nan"), 9.0]})
    assert choose(g)["label"] == "b"     # domyślna niewykonalna: bierzemy najlepszą