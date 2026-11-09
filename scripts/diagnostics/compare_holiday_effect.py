import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.features.calendar import is_holiday  # noqa: E402

con = duckdb.connect(str(ROOT / "data" / "processed" / "electricity.duckdb"), read_only=True)


def daily(table: str) -> pd.DataFrame:
    return con.sql(f"""
        SELECT model, delivery_date, avg(abs(y_pred - y_true)) AS mae
        FROM {table} WHERE delivery_date BETWEEN '2021-01-01' AND '2025-12-31'
        GROUP BY ALL""").df()


d = daily("forecasts_v1").merge(daily("forecasts"), on=["model", "delivery_date"],
                                suffixes=("_v1", "_v2")).dropna()
d["delivery_date"] = pd.to_datetime(d["delivery_date"])
d["hol"] = is_holiday(d["delivery_date"]) == 1.0

rows = []
for model, g in d.groupby("model"):
    if (g["mae_v1"] - g["mae_v2"]).abs().max() < 1e-9:
        continue  # model jeszcze nieprzeliczony
    h, n = g[g["hol"]], g[~g["hol"]]
    rows.append({"model": model, "MAE v1": g["mae_v1"].mean(), "MAE v2": g["mae_v2"].mean(),
                 "zmiana %": 100 * (g["mae_v2"].mean() / g["mae_v1"].mean() - 1),
                 "dni świąt": len(h),
                 "zmiana święta %": 100 * (h["mae_v2"].mean() / h["mae_v1"].mean() - 1),
                 "zmiana pozostałe %": 100 * (n["mae_v2"].mean() / n["mae_v1"].mean() - 1)})
print(pd.DataFrame(rows).round(2).to_string(index=False))