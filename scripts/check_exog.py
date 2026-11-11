import duckdb, pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

tz = ZoneInfo("Europe/Warsaw")
con = duckdb.connect("data/processed/electricity.duckdb", read_only=True)
df = con.sql("SELECT dataset, series_key, max(timestamp_utc) AS last_utc "
             "FROM entsoe_raw GROUP BY ALL ORDER BY 1, 2").df()
con.close()

lines = [f"=== {datetime.now(tz):%Y-%m-%d %H:%M} czasu lokalnego ==="]
for r in df.itertuples():
    last = pd.Timestamp(r.last_utc).tz_localize("UTC").tz_convert(tz)
    lines.append(f"{r.dataset}/{r.series_key}: ostatni okres {last:%Y-%m-%d %H:%M}")
txt = "\n".join(lines)
print(txt)
with open("exog_log.txt", "a", encoding="utf-8") as f:
    f.write(txt + "\n")
