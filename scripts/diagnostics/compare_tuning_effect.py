"""Wpływ strojenia: prognozy sprzed (forecasts_v2, okno 364) i po (forecasts), MAE według lat."""
import sys
from pathlib import Path

import duckdb
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[2]
con = duckdb.connect(str(ROOT / "data" / "processed" / "electricity.duckdb"), read_only=True)


def daily(table: str) -> pd.DataFrame:
    return con.sql(f"""SELECT model, delivery_date, avg(abs(y_pred - y_true)) AS mae FROM {table}
                       WHERE delivery_date BETWEEN '2021-01-01' AND '2025-12-31' GROUP BY ALL""").df()


d = daily("forecasts_v2").merge(daily("forecasts"), on=["model", "delivery_date"],
                                suffixes=("_v2", "_tuned")).dropna()
d["year"] = pd.to_datetime(d["delivery_date"]).dt.year

rows = []
for model, g in d.groupby("model"):
    if (g["mae_v2"] - g["mae_tuned"]).abs().max() < 1e-9:
        continue  # model bez zmian
    row = {"model": model}
    for year, gy in g.groupby("year"):
        row[str(year)] = 100 * (gy["mae_tuned"].mean() / gy["mae_v2"].mean() - 1)
    row["całość"] = 100 * (g["mae_tuned"].mean() / g["mae_v2"].mean() - 1)
    rows.append(row)

print("Zmiana MAE po strojeniu, % (ujemna = poprawa)")
print(pd.DataFrame(rows).round(1).to_string(index=False))