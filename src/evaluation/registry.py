"""Budowanie modeli z konfiguracją wybraną po strojeniu (config/model_params.yaml).

spec="main": konfiguracja wybrana po strojeniu; spec="sens": konfiguracje domyślne (okno 364) jako analiza wrażliwości.
"""
from __future__ import annotations

import yaml

from src.utils.config import PROJECT_ROOT

PARAMS_PATH = PROJECT_ROOT / "config" / "model_params.yaml"
FAMILY = {"arx": "linear", "arx_x": "linear", "lear": "linear", "lear_x": "linear",
          "xgb": "xgb", "xgb_x": "xgb", "mlp": "mlp", "mlp_x": "mlp",
          "prophet": "prophet", "prophet_x": "prophet", "sarima": "sarima", "sarimax": "sarima"}


def _selected() -> dict:
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH, encoding="utf-8") as f:
            return (yaml.safe_load(f) or {}).get("selected", {})
    return {}


def build_model(key: str, spec: str = "main"):
    from src.evaluation.tune import specs  # import lokalny: tune importuje modele

    selected = _selected() if spec == "main" else {}
    for model, label, builder, _ in specs(FAMILY[key]):
        if model == key and (key not in selected or label == selected[key]):
            return builder()
    raise KeyError(f"Brak konfiguracji '{selected.get(key)}' dla modelu {key}")


def is_changed(key: str) -> bool:
    """Czy konfiguracja wybrana po strojeniu różni się od domyślnej (pierwszej w siatce)."""
    from src.evaluation.tune import specs

    sel = _selected().get(key)
    if sel is None:
        return False
    first = next(label for model, label, _, _ in specs(FAMILY[key]) if model == key)
    return sel != first