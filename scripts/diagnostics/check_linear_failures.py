from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
con = duckdb.connect(str(ROOT / "data" / "processed" / "electricity.duckdb"), read_only=True)
daily = con.sql("""
    SELECT model, delivery_date, avg(abs(y_pred - y_true)) AS mae,
           max(y_pred) AS max_pred, min(y_pred) AS min_pred, max(y_true) AS max_true
    FROM forecasts
    WHERE model IN ('arx', 'arx_x', 'lear', 'lear_x')
      AND delivery_date BETWEEN '2024-01-01' AND '2025-12-31'
    GROUP BY ALL
""").df()
daily["year"] = pd.to_datetime(daily["delivery_date"]).dt.year

print("=== udział 10 najgorszych dób w sumie dobowego MAE ===")
for (model, year), g in daily.groupby(["model", "year"]):
    top = g.nlargest(10, "mae")
    print(f"{model:7s} {year}: MAE {g['mae'].mean():6.1f} | top10 = {top['mae'].sum() / g['mae'].sum():4.0%} "
          f"| max_pred {g['max_pred'].max():9.0f} | min_pred {g['min_pred'].min():9.0f}")

for model, year in (("arx_x", 2025), ("lear_x", 2024)):
    g = daily[(daily["model"] == model) & (daily["year"] == year)]
    print(f"\n=== {model} {year}: 10 najgorszych dób ===")
    print(g.nlargest(10, "mae").drop(columns=["model", "year"]).round(1).to_string(index=False))