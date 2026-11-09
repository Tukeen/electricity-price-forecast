import src.evaluation.registry as reg


def test_registry_uses_selected_and_default(tmp_path, monkeypatch):
    p = tmp_path / "p.yaml"
    p.write_text("selected:\n  lear: w728\n  xgb: w728_d5_n600\n", encoding="utf-8")
    monkeypatch.setattr(reg, "PARAMS_PATH", p)
    assert reg.build_model("lear").window_days == 728
    assert reg.build_model("arx").window_days == 364          # brak wpisu: domyślna
    assert reg.build_model("xgb").params["n_estimators"] == 600

def test_is_changed_and_sens_spec(tmp_path, monkeypatch):
    p = tmp_path / "p.yaml"
    p.write_text("selected:\n  lear: w728\n  arx: w364\n", encoding="utf-8")
    monkeypatch.setattr(reg, "PARAMS_PATH", p)
    assert reg.is_changed("lear") and not reg.is_changed("arx") and not reg.is_changed("xgb")
    assert reg.build_model("lear", spec="sens").window_days == 364