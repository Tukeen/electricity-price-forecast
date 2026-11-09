"""Przewaga ML na dobach z ujemną ceną: podział według roku (2023-2025), obie specyfikacje."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.evaluation.metrics import diebold_mariano, errors, mae  # noqa: E402
from src.evaluation.protocol import load_forecasts  # noqa: E402
from src.features.panels import DB_PATH, load_panels  # noqa: E402

price = load_panels()["price"]
days = price.loc["2023-06-11":"2025-12-31"].index
neg = (price.reindex(days).min(axis=1) < 0).to_numpy()
MODELS = ("lear_x", "sarimax", "xgb_x", "mlp_x")

for table in ("forecasts", "forecasts_v2"):
    errs = {m: errors(load_forecasts(DB_PATH, m, table).reindex(days), price) for m in MODELS}
    rows = []
    for year in (2023, 2024, 2025):
        for label, flag in (("ujemna cena", True), ("pozostałe", False)):
            mask = (days.year == year) & (neg == flag)
            if mask.sum() < 5:
                continue
            maes = {m: mae(errs[m].loc[mask]) for m in MODELS}
            row = {"rok": year, "doby": label, "dni": int(mask.sum()), **{m: maes[m] for m in MODELS},
                   "xgb_x vs lear_x %": 100 * (maes["xgb_x"] / maes["lear_x"] - 1),
                   "xgb_x vs sarimax %": 100 * (maes["xgb_x"] / maes["sarimax"] - 1)}
            if mask.sum() >= 20:
                row["p DM xgb_x vs lear_x"] = diebold_mariano(errs["xgb_x"].loc[mask], errs["lear_x"].loc[mask])[1]
            rows.append(row)
    print(f"\n=== tabela {table} ===")
    print(pd.DataFrame(rows).round(3).to_string(index=False))