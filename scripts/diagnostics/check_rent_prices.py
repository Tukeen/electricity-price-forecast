from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
con = duckdb.connect(str(ROOT / "data" / "processed" / "electricity.duckdb"), read_only=True)

print("=== doby od 2026-10-20 ===")
print(con.sql("""
    SELECT delivery_date, count(*) AS h, round(avg(price_eur_mwh),1) AS mean_p,
           round(min(price_eur_mwh),1) AS min_p, round(max(price_eur_mwh),1) AS max_p
    FROM hourly WHERE delivery_date >= '2026-10-20' GROUP BY 1 ORDER BY 1
""").df().to_string(index=False))

print("\n=== 10 najwyższych średnich miesięcznych od 2020 ===")
print(con.sql("""
    SELECT strftime(timestamp_local, '%Y-%m') AS month, round(avg(price_eur_mwh),1) AS mean_p
    FROM hourly GROUP BY 1 ORDER BY 2 DESC LIMIT 10
""").df().to_string(index=False))

print("\n=== kwadranse z doby 2026-11-05 (UTC), bez agregacji ===")
print(con.sql("""
    SELECT timestamp_utc, value FROM entsoe_raw
    WHERE dataset = 'prices' AND timestamp_utc >= '2026-11-04 23:00' AND timestamp_utc < '2026-11-05 23:00'
    ORDER BY 1
""").df().to_string(index=False))