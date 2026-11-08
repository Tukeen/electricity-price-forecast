"""Benchmarki naiwne: ocena 2020-2025 (bez 2026) według reżimów i lat.

Uruchomienie: python -m src.evaluation.run_baselines
"""
from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import errors, mae, me, rmse
from src.evaluation.protocol import rolling_forecast, save_forecasts
from src.features.panels import DB_PATH, load_panels
from src.models.naive import Naive
from src.utils.config import PROJECT_ROOT, load_config

CFG = load_config()
DEV_END = pd.Timestamp(CFG["evaluation"]["validation"]["end"])


def main() -> None:
    panels = load_panels()
    price = panels["price"]
    days = price.loc[price.index[0] + pd.Timedelta(days=7): DEV_END].index

    models = [Naive("daily"), Naive("weekly"), Naive("mixed")]
    errs = {}
    for m in models:
        pred = rolling_forecast(m, panels, days, CFG["evaluation"]["window_days"])
        save_forecasts(DB_PATH, m.name, pred, price)
        errs[m.name] = errors(pred, price)

    regime = pd.Series("poza_regimem", index=days)
    for name, r in CFG["regimes"].items():
        regime[(days >= pd.Timestamp(r["start"])) & (days <= pd.Timestamp(r["end"]))] = name

    groups = {"całość": pd.Series(True, index=days)}
    groups.update({r: regime == r for r in CFG["regimes"]})
    groups.update({str(y): pd.Series(days.year == y, index=days) for y in sorted(set(days.year))})

    ref = errs["naive_mixed"]
    rows = []
    for gname, mask in groups.items():
        for name, e in errs.items():
            sel = e.loc[mask.to_numpy()]
            rows.append({"grupa": gname, "model": name, "dni": len(sel), "MAE": mae(sel),
                         "RMSE": rmse(sel), "ME": me(sel), "rMAE": mae(sel) / mae(ref.loc[mask.to_numpy()])})
    table = pd.DataFrame(rows)

    out = PROJECT_ROOT / "reports" / "tables" / "baselines.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, encoding="utf-8-sig")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()