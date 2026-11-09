"""Budowanie modeli z konfiguracją wybraną po strojeniu (config/model_params.yaml).
Brak wpisu = konfiguracja domyślna (pierwsza w siatce)."""
from __future__ import annotations

import yaml

from src.utils.config import PROJECT_ROOT

PARAMS_PATH = PROJECT_ROOT / "config" / "model_params.yaml"
FAMILY = {"arx": "linear", "arx_x": "linear", "lear": "linear", "lear_x": "linear",
          "xgb": "xgb", "xgb_x": "xgb", "mlp": "mlp", "mlp_x": "mlp",
          "prophet": "prophet", "prophet_x": "prophet", "sarima": "sarima", "sarimax": "sarima"}


def build_model(key: str):
    from src.evaluation.tune import specs  # import lokalny: tune importuje modele

    selected = {}
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH, encoding="utf-8") as f:
            selected = (yaml.safe_load(f) or {}).get("selected", {})
    for model, label, builder, _ in specs(FAMILY[key]):
        if model == key and (key not in selected or label == selected[key]):
            return builder()
    raise KeyError(f"Brak konfiguracji '{selected.get(key)}' dla modelu {key}")