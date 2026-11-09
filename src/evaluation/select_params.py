"""Zapis wybranych konfiguracji do config/model_params.yaml (z reports/tables/tuning_results.csv)."""
import sys

import pandas as pd
import yaml

from src.evaluation.tune import OUT, choose, summarize
from src.utils.config import PROJECT_ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TARGET = PROJECT_ROOT / "config" / "model_params.yaml"


def main() -> None:
    done = pd.read_csv(OUT)
    selected = {model: str(choose(g)["label"]) for model, g in done.groupby("model", sort=False)}
    doc = {"walidacja": "2025",
           "reguła": "domyślna konfiguracja, chyba że najlepsza poprawia MAE o więcej niż 1%",
           "selected": selected}
    with open(TARGET, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)
    print(summarize(done).round(3).to_string(index=False))
    print(f"\nZapisano: {TARGET}")


if __name__ == "__main__":
    main()