from src.evaluation.interpret import group_of


def test_group_of():
    assert group_of("p_l1_h5") == "ceny opóźnione"
    assert group_of("p_min_l1") == "ceny opóźnione"
    assert group_of("pv_h12") == "PV (prognoza)"
    assert group_of("wind_h3") == "wiatr (prognoza)"
    assert group_of("resid_load_h") == "obciążenie rezydualne"
    assert group_of("hol") == "kalendarz"