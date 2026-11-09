from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
con = duckdb.connect(str(ROOT / "data" / "processed" / "electricity.duckdb"), read_only=True)


def block_mae(table: str, model: str) -> pd.Series:
    df = con.sql(f"""SELECT delivery_date, avg(abs(y_pred - y_true)) AS mae FROM {table}
                     WHERE model = '{model}' AND delivery_date BETWEEN '2021-01-01' AND '2025-12-31'
                     GROUP BY 1 ORDER BY 1""").df()
    d = pd.to_datetime(df["delivery_date"])
    start = d.dt.normalize().sub(pd.Timestamp("2021-01-01")).dt.days // 28
    return df.groupby(start)["mae"].mean().rename(model)


for model in ("sarima", "sarimax"):
    t = pd.concat([block_mae("forecasts_v1", model).rename("MAE v1"),
                   block_mae("forecasts", model).rename("MAE v2")], axis=1)
    t["różnica"] = t["MAE v2"] - t["MAE v1"]
    t["początek bloku"] = [(pd.Timestamp("2021-01-01") + pd.Timedelta(days=28 * i)).date() for i in t.index]
    print(f"\n=== {model}: 6 bloków o największym wzroście MAE ===")
    print(t.nlargest(6, "różnica").round(2).to_string())