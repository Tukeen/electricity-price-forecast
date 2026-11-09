from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
con = duckdb.connect(str(ROOT / "data" / "processed" / "electricity.duckdb"), read_only=True)


def daily(table: str) -> pd.Series:
    return con.sql(f"""SELECT delivery_date, avg(abs(y_pred - y_true)) AS mae FROM {table}
                       WHERE model = 'sarima' AND delivery_date BETWEEN '2021-03-26' AND '2021-04-22'
                       GROUP BY 1 ORDER BY 1""").df().set_index("delivery_date")["mae"]


t = pd.concat([daily("forecasts_v1").rename("MAE v1"), daily("forecasts").rename("MAE v2")], axis=1)
t["różnica"] = t["MAE v2"] - t["MAE v1"]
print(t.round(2).to_string())