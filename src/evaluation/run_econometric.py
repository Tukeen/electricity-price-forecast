"""SARIMA/SARIMAX i VAR, ocena 2021-2025.

python -m src.evaluation.run_econometric --models sarima sarimax --smoke     # test dymny (nic nie zapisuje)
python -m src.evaluation.run_econometric --models var1 var2 var3
python -m src.evaluation.run_econometric --models sarima sarimax --refit-every 28 --jobs 10
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import time

import pandas as pd

from src.evaluation.metrics import errors, mae
from src.evaluation.protocol import (load_forecasts, rolling_forecast, rolling_forecast_blocks,
                                     save_forecasts)
from src.evaluation.report import common_days, dm_table, summarize
from src.features.panels import DB_PATH, load_panels
from src.models.sarima import SarimaEPF
from src.models.var import VarEPF
from src.utils.config import PROJECT_ROOT, load_config

CFG = load_config()
EV = CFG["evaluation"]
KEYS = ["sarima", "sarimax", "var1", "var2", "var3"]
ALL = ["naive_daily", "naive_mixed", "arx", "arx_x", "lear", "lear_x", *KEYS]
PAIRS = [("sarima", "naive_mixed"), ("sarimax", "naive_mixed"), ("sarimax", "sarima"), ("sarima", "arx"),
         ("sarimax", "arx_x"), ("sarimax", "lear_x"), ("var1", "naive_mixed"), ("var2", "var1"),
         ("var3", "var2"), ("var2", "arx"), ("var1", "arx"), ("var1", "lear")]


def build(key: str):
    if key == "sarima":
        return SarimaEPF(False, EV["window_days"])
    if key == "sarimax":
        return SarimaEPF(True, EV["window_days"])
    return VarEPF(int(key[3:]), EV["window_days"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=KEYS, choices=KEYS)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--refit-every", type=int, default=28)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    panels = load_panels()
    price = panels["price"]
    days = price.loc[pd.Timestamp(EV["regime_analysis"]["start"]): pd.Timestamp(EV["validation"]["end"])].index

    for key in args.models:
        model = build(key)
        print(f"== {model.name} ==")
        t0 = time.time()
        if args.smoke:
            block = days[: args.refit_every]
            preds = model.forecast_block(panels, pd.DatetimeIndex(block)) if hasattr(model, "forecast_block") \
                else rolling_forecast(model, panels, block, EV["window_days"], n_jobs=1).to_numpy()
            pred = pd.DataFrame(list(preds), index=block, columns=range(24))
            ref = load_forecasts(DB_PATH, "naive_mixed").reindex(block)
            print(f"   {time.time() - t0:.0f} s | MAE {mae(errors(pred, price)):.2f} "
                  f"| naive_mixed {mae(errors(ref, price)):.2f} | dób z NaN: {int(pred.isna().any(axis=1).sum())}")
            continue
        if hasattr(model, "forecast_block"):
            pred = rolling_forecast_blocks(model, panels, days, args.refit_every, args.jobs)
        else:
            pred = rolling_forecast(model, panels, days, EV["window_days"], n_jobs=args.jobs)
        save_forecasts(DB_PATH, model.name, pred, price)
        print(f"   {time.time() - t0:.0f} s")

    if args.smoke:
        return
    errs = {}
    for name in ALL:
        pred = load_forecasts(DB_PATH, name)
        if not pred.empty:
            errs[name] = errors(pred.reindex(days), price)
    errs = common_days(errs)
    summary = summarize(errs, "naive_mixed", CFG["regimes"])
    dm = dm_table(errs, PAIRS, CFG["regimes"])
    out = PROJECT_ROOT / "reports" / "tables"
    summary.to_csv(out / "econometric_results.csv", index=False, encoding="utf-8-sig")
    dm.to_csv(out / "econometric_dm.csv", index=False, encoding="utf-8-sig")

    order = list(summary["grupa"].unique())
    for metric in ("MAE", "rMAE"):
        print(f"\n=== {metric} ===")
        print(summary.pivot(index="grupa", columns="model", values=metric).loc[order].round(3).to_string())
    print("\n=== Diebold-Mariano (DM < 0: A lepszy od B) ===")
    print(dm[dm["grupa"].isin(["całość", *CFG["regimes"]])].round(4).to_string(index=False))


if __name__ == "__main__":
    main()