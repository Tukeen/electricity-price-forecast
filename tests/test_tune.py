import pandas as pd

from src.evaluation.tune import choose


def test_choose_prefers_simplest_within_tolerance():
    g = pd.DataFrame({"label": ["a", "b", "c"], "mae": [10.0, 9.95, 9.0]})
    assert choose(g)["label"] == "c"          # wyraźnie lepsza konfiguracja wygrywa
    g = pd.DataFrame({"label": ["a", "b"], "mae": [10.0, 9.95]})
    assert choose(g)["label"] == "a"          # różnica < 1%: zostaje pierwsza (najprostsza)