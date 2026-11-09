"""Modele ML (XGBoost, MLP; z i bez zmiennych egzogenicznych), ocena 2021-2025 + raport zbiorczy.

python -m src.evaluation.run_ml --models mlp mlp_x --jobs 10
python -m src.evaluation.run_ml                         # tylko raport z zapisanych prognoz
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import time

import pandas as pd

from src.evaluation.metrics import errors, mae
from src.evaluation.protocol import load_forecasts, rolling_forecast, save_forecasts
from src.evaluation.report import common_days, dm_table, summarize
from src.features.panels import DB_PATH, load_panels
from src.models.mlp import MLPEPF
from src.models.xgb import XGBoostEPF
from src.models.prophet_epf import ProphetEPF
from src.utils.config import PROJECT_ROOT, load_config
from src.evaluation.registry import build_model

CFG = load_config()
EV = CFG["evaluation"]
SPECS = {"xgb": ("xgb", False), "xgb_x": ("xgb", True), "mlp": ("mlp", False), "mlp_x": ("mlp", True), "prophet": ("prophet", False), "prophet_x": ("prophet", True)}
ALL = ["naive_daily", "naive_mixed", "arx", "arx_x", "lear", "lear_x", "sarima", "sarimax", "var1",
       "xgb", "xgb_x", "mlp", "mlp_x", "prophet", "prophet_x"]
PAIRS = [("xgb", "naive_mixed"), ("xgb_x", "naive_mixed"), ("xgb_x", "xgb"),
         ("xgb", "arx"), ("xgb_x", "arx_x"), ("xgb", "lear"), ("xgb_x", "lear_x"),
         ("xgb", "sarima"), ("xgb_x", "sarimax"),
         ("mlp", "naive_mixed"), ("mlp_x", "naive_mixed"), ("mlp_x", "mlp"), ("mlp", "xgb"), ("mlp_x", "xgb_x"),
         ("mlp", "lear"), ("mlp_x", "lear_x"), ("mlp", "sarima"), ("mlp_x", "sarimax"),
         ("prophet", "naive_mixed"), ("prophet_x", "naive_mixed"), ("prophet_x", "prophet"),
         ("prophet", "lear"), ("prophet_x", "lear_x"), ("prophet", "sarima"), ("prophet_x", "sarimax"),
         ("prophet", "xgb"), ("prophet_x", "xgb_x")]
EXOG_PAIRS = [("arx", "arx_x"), ("lear", "lear_x"), ("sarima", "sarimax"), ("xgb", "xgb_x"), ("mlp", "mlp_x"), ("prophet", "prophet_x")]


def build(key: str):
    return build_model(key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[], choices=list(SPECS))
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()

    panels = load_panels()
    price = panels["price"]
    days = price.loc[pd.Timestamp(EV["regime_analysis"]["start"]): pd.Timestamp(EV["validation"]["end"])].index

    for key in args.models:
        model = build(key)
        print(f"== {model.name} ==")
        t0 = time.time()
        if args.smoke:
            block = days[: args.smoke]
            pred = rolling_forecast(model, panels, block, EV["window_days"], n_jobs=args.jobs)
            ref = load_forecasts(DB_PATH, "naive_mixed").reindex(block)
            print(f"   {time.time() - t0:.0f} s | MAE {mae(errors(pred, price)):.2f} | naive_mixed "
                  f"{mae(errors(ref, price)):.2f} | dób z NaN: {int(pred.isna().any(axis=1).sum())}")
            continue
        
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
    summary.to_csv(out / "ml_results.csv", index=False, encoding="utf-8-sig")
    dm.to_csv(out / "ml_dm.csv", index=False, encoding="utf-8-sig")

    order = list(summary["grupa"].unique())
    mae_tab = summary.pivot(index="grupa", columns="model", values="MAE").loc[order]
    gains = pd.DataFrame({f"{b} vs {a}": mae_tab[b] / mae_tab[a] - 1
                          for a, b in EXOG_PAIRS if a in mae_tab and b in mae_tab})
    gains.to_csv(out / "exog_gain.csv", encoding="utf-8-sig")

    print("\n=== MAE ===")
    print(mae_tab.round(3).to_string())
    print("\n=== rMAE (względem naive_mixed) ===")
    print(summary.pivot(index="grupa", columns="model", values="rMAE").loc[order].round(3).to_string())
    print("\n=== względna zmiana MAE po dodaniu zmiennych egzogenicznych (ujemna = poprawa; H3) ===")
    print((gains * 100).round(1).to_string())
    print("\n=== Diebold-Mariano (DM < 0: A lepszy od B) ===")
    print(dm[dm["grupa"].isin(["całość", *CFG["regimes"]])].round(4).to_string(index=False))


if __name__ == "__main__":
    main()