from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
con = duckdb.connect(str(ROOT / "data" / "processed" / "electricity.duckdb"), read_only=True)

print("=== miesięcznie: poziom i zmienność ceny godzinowej ===")
print(con.sql("""
    SELECT strftime(timestamp_local, '%Y-%m') AS month,
           round(avg(price_eur_mwh), 1)  AS mean_price,
           round(stddev(price_eur_mwh), 1) AS std_price,
           round(avg(abs(price_eur_mwh - lag_p)), 1) AS mean_abs_change_1h
    FROM (SELECT *, lag(price_eur_mwh) OVER (ORDER BY timestamp_utc) AS lag_p FROM hourly)
    WHERE timestamp_local >= '2024-07-01'
    GROUP BY 1 ORDER BY 1
""").df().to_string(index=False))

print("\n=== rozstęp kwadransów w obrębie godziny (okres 15-min) ===")
print(con.sql("""
    SELECT strftime(h, '%Y-%m') AS month, round(avg(rng), 1) AS avg_intrahour_range
    FROM (SELECT date_trunc('hour', timestamp_utc) AS h, max(value) - min(value) AS rng
          FROM entsoe_raw
          WHERE dataset = 'prices' AND resolution = 'PT15M' AND currency = 'EUR'
          GROUP BY h)
    GROUP BY 1 ORDER BY 1
""").df().to_string(index=False))