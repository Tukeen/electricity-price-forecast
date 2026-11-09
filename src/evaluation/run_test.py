"""Test końcowy na roku 2026 (jednorazowy, po zamrożeniu) oraz próba generalna.

python -m src.evaluation.run_test --dry-run --spec main --jobs 10      # próba na 2025-12-01..14
python -m src.evaluation.run_test --dry-run --spec sens --jobs 10
python -m src.evaluation.run_test --spec main --confirm-test --jobs 10     # TEST 2026
python -m src.evaluation.run_test --spec sens --confirm-test --jobs 10
python -m src.evaluation.run_test --spec main --confirm-test --report-only
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import time

import numpy as np
import pandas as pd

from src.evaluation.freeze import check_frozen
from src.evaluation.inference import gain_difference, holm
from src.evaluation.intervals import daily_interval_metrics, empirical_intervals
from src.evaluation.metrics import diebold_mariano, errors, mae
from src.evaluation.protocol import load_forecasts, rolling_forecast, rolling_forecast_blocks, save_forecasts
from src.evaluation.registry import build_model, is_changed
from src.evaluation.report import common_days, dm_table, summarize, holdout_groups
from src.features.panels import DB_PATH, load_panels
from src.models.linear_nl import LearNL
from src.models.naive import Naive
from src.models.var import VarEPF
from src.utils.config import PROJECT_ROOT, load_config

CFG = load_config()
EV = CFG["evaluation"]
TEST_START, TEST_END = pd.Timestamp(EV["test"]["start"]), pd.Timestamp(EV["test"]["end"])
DRY = (pd.Timestamp("2025-12-01"), pd.Timestamp("2025-12-14"))
CORE = ["naive_daily", "naive_mixed", "arx", "arx_x", "lear", "lear_x", "sarima", "sarimax", "var1",
        "xgb", "xgb_x", "mlp", "mlp_x", "prophet", "prophet_x"]
EXPLORATORY = ["lear_nl"]
ALL = CORE + EXPLORATORY
PRIMARY = [("xgb_x", "lear_x"), ("xgb_x", "sarimax"), ("xgb", "lear"), ("xgb", "sarima")]
SECONDARY = [("mlp_x", "lear_x"), ("mlp_x", "sarimax"), ("lear_x", "sarimax"), ("mlp", "lear"),
             ("xgb_x", "mlp_x"), ("lear_nl", "lear_x"), ("xgb_x", "lear_nl")]
NEG_PAIRS = [("xgb_x", "lear_x"), ("xgb_x", "sarimax"), ("mlp_x", "lear_x"), ("xgb", "lear")]
ML_X = {"xgb": "xgb_x", "mlp": "mlp_x"}
ECON_X = {"arx": "arx_x", "lear": "lear_x", "sarima": "sarimax"}
INTERVAL_MODELS = [m for m in CORE if m != "naive_daily"]
SHOW_MODELS = ["naive_mixed", "lear", "lear_x", "sarimax", "xgb", "xgb_x", "mlp_x", "lear_nl"]
HIST = {"main": ["forecasts", "forecasts_diag"], "sens": ["forecasts_v2", "forecasts", "forecasts_diag"]}
BLOCK_MODELS = {"sarima", "sarimax"}


def table_chain(spec: str, dry: bool) -> list[str]:
    if dry:
        return [f"forecasts_dryrun_{spec}"] + (["forecasts_dryrun_main"] if spec == "sens" else [])
    return ["forecasts_test_v2", "forecasts_test"] if spec == "sens" else ["forecasts_test"]


def changed(key: str) -> bool:
    if key.startswith("naive") or key == "var1":
        return False
    return is_changed("lear_x" if key == "lear_nl" else key)


def make(key: str, spec: str):
    if key.startswith("naive"):
        return Naive(key.split("_")[1])
    if key == "var1":
        return VarEPF(1, EV["window_days"])
    if key == "lear_nl":
        return LearNL(build_model("lear_x", spec).window_days)
    return build_model(key, spec)


def first_available(tables: list[str], key: str) -> pd.DataFrame:
    for t in tables:
        try:
            df = load_forecasts(DB_PATH, key, t)
        except Exception:
            continue
        if not df.empty:
            return df
    return pd.DataFrame()


def compute(spec, days, panels, jobs, models, dry) -> None:
    table = table_chain(spec, dry)[0]
    price = panels["price"]
    for key in models:
        if spec == "sens" and not changed(key):
            continue  # konfiguracja identyczna jak w głównej: prognozy współdzielone
        model = make(key, spec)
        print(f"== {key} ({spec}) ==", flush=True)
        t0 = time.time()
        if key in BLOCK_MODELS:
            pred = rolling_forecast_blocks(model, panels, days, 28, jobs)
        else:
            pred = rolling_forecast(model, panels, days, getattr(model, "window_days", EV["window_days"]), n_jobs=jobs)
        save_forecasts(DB_PATH, key, pred, price, table=table)
        print(f"   {time.time() - t0:.0f} s", flush=True)


def interval_metrics(key, spec, tabs, days, price, level, window=90) -> pd.DataFrame:
    test = first_available(tabs, key).reindex(days)
    hist = first_available(HIST[spec], key).loc[: days[0] - pd.Timedelta(days=1)].tail(window + 30)
    lo, hi = empirical_intervals(pd.concat([hist, test]), price, level, window=window)
    return daily_interval_metrics(lo.loc[days], hi.loc[days], price.reindex(days), level)


def report(spec, days, price, dry) -> None:
    tabs = table_chain(spec, dry)
    errs = {}
    for m in ALL:
        pred = first_available(tabs, m)
        if pred.empty:
            print(f"Brak prognoz: {m}")
            continue
        errs[m] = errors(pred.reindex(days), price)
    errs = common_days(errs)
    if "naive_mixed" not in errs:
        print("Brak benchmarku naive_mixed, raport przerwany.")
        return
    days = errs["naive_mixed"].index
    prefix = ("dryrun_" if dry else "test_") + spec
    out = PROJECT_ROOT / "reports" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n##### RAPORT ({'PRÓBA GENERALNA' if dry else 'TEST 2026'}), specyfikacja {spec}: "
          f"{len(days)} dób, {days[0].date()} - {days[-1].date()} #####")

    # 1. Metryki
    summary = summarize(errs, "naive_mixed", CFG["regimes"], group_fn=holdout_groups)
    summary.to_csv(out / f"{prefix}_results.csv", index=False, encoding="utf-8-sig")
    order = list(summary["grupa"].unique())
    cols = [m for m in ALL if m in errs]
    for metric in ("MAE", "rMAE"):
        print(f"\n=== {metric} (rMAE względem naive_mixed; lear_nl = model eksploracyjny) ===")
        print(summary.pivot(index="grupa", columns="model", values=metric).loc[order, cols].round(3).to_string())

    # 2. DM: całość i kwartały; Holm dla czterech testów potwierdzających na całym roku
    dm = dm_table(errs, PRIMARY + SECONDARY, CFG["regimes"], group_fn=holdout_groups)
    if dm.empty:
        print("\nDM: za mało dób (<30).")
    else:
        dm = dm[dm["grupa"].isin(["całość", "Q1", "Q2", "Q3", "Q4"])].copy()
        dm["rodzaj"] = [("potwierdzający" if (a, b) in PRIMARY else "drugorzędny/eksploracyjny")
                        for a, b in zip(dm["A"], dm["B"])]
        dm["p_holm"] = np.nan
        prim = dm[(dm["grupa"] == "całość") & (dm["rodzaj"] == "potwierdzający")]
        if len(prim):
            dm.loc[prim.index, "p_holm"] = holm(prim["p"]).to_numpy()
        dm.to_csv(out / f"{prefix}_dm.csv", index=False, encoding="utf-8-sig")
        print("\n=== Diebold-Mariano (DM < 0: A lepszy od B; p_holm tylko dla 4 testów potwierdzających, całość) ===")
        print(dm.round(4).to_string(index=False))

    # 3. Doby z ujemną ceną (opisowo)
    neg = (price.reindex(days).min(axis=1) < 0).to_numpy()
    rows = []
    for label, mask in (("doby z ujemną ceną", neg), ("pozostałe doby", ~neg)):
        if mask.sum() < 10:
            continue
        for a, b in NEG_PAIRS:
            if a in errs and b in errs:
                stat, p = diebold_mariano(errs[a].loc[mask], errs[b].loc[mask])
                rows.append({"podzbiór": label, "dni": int(mask.sum()), "para": f"{a} vs {b}",
                             "A/B - 1 [%]": 100 * (mae(errs[a].loc[mask]) / mae(errs[b].loc[mask]) - 1),
                             "DM": stat, "p": p})
    if rows:
        neg_tab = pd.DataFrame(rows)
        neg_tab.to_csv(out / f"{prefix}_negative_days.csv", index=False, encoding="utf-8-sig")
        print("\n=== Doby z ujemną ceną vs pozostałe (opisowo; podział po zrealizowanej cenie) ===")
        print(neg_tab.round(3).to_string(index=False))

    # 4. H3: różnica zysków z egzogenicznych, bootstrap blokowy
    if len(days) >= 30:
        daily = {m: errs[m].abs().mean(axis=1).to_numpy() for m in errs}
        rows = []
        for mb, mx in ML_X.items():
            for eb, ex in ECON_X.items():
                if all(k in daily for k in (mb, mx, eb, ex)):
                    d, lo_, hi_, prob = gain_difference(daily[mb], daily[mx], daily[eb], daily[ex])
                    rows.append({"ML": mx, "ECON": ex,
                                 "zysk ML [%]": 100 * (daily[mx].mean() / daily[mb].mean() - 1),
                                 "zysk ECON [%]": 100 * (daily[ex].mean() / daily[eb].mean() - 1),
                                 "Delta [p.p.]": 100 * d, "CI dolna": 100 * lo_, "CI górna": 100 * hi_,
                                 "P(Delta<0)": prob})
        h3 = pd.DataFrame(rows)
        h3.to_csv(out / f"{prefix}_h3_bootstrap.csv", index=False, encoding="utf-8-sig")
        print("\n=== H3: Delta = zysk ML - zysk ECON (Delta<0: ML zyskuje więcej) ===")
        print(h3.round(2).to_string(index=False))

    # 5. H4: przedziały 90% według miesięcy (empiryczne kwantyle z 90 dób, historia z 2025)
    frames = []
    for m in INTERVAL_MODELS:
        met = interval_metrics(m, spec, tabs, days, price, 0.9)
        met.insert(0, "model", m)
        met["miesiąc"] = days.month
        frames.append(met)
    iv = pd.concat(frames)
    iv.to_csv(out / f"{prefix}_intervals90_daily.csv", encoding="utf-8-sig")
    monthly = iv.groupby(["model", "miesiąc"])[["coverage", "width", "score"]].mean().reset_index()
    monthly.to_csv(out / f"{prefix}_intervals90_monthly.csv", index=False, encoding="utf-8-sig")
    for metric in ("coverage", "score"):
        tab = monthly.pivot(index="miesiąc", columns="model", values=metric)[[m for m in SHOW_MODELS if m in INTERVAL_MODELS]]
        print(f"\n=== Przedziały 90%, {metric} według miesięcy ===")
        print(tab.round(3).to_string())
    print("\nCałość:")
    print(iv.groupby("model")[["coverage", "width", "score"]].mean().round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", choices=["main", "sens"], default="main")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--models", nargs="+", default=ALL, choices=ALL)
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()

    panels = load_panels()
    price = panels["price"]
    if args.dry_run:
        days = price.loc[DRY[0]: DRY[1]].index
    else:
        if not args.confirm_test:
            print("To uruchamia test na roku 2026. Dodaj --confirm-test po zamrożeniu kodu i danych.")
            sys.exit(1)
        problems = check_frozen(panels, TEST_END)
        if problems:
            print("Test zablokowany:\n- " + "\n- ".join(problems))
            sys.exit(1)
        days = price.loc[TEST_START: min(TEST_END, price.index.max())].index
    print(f"Specyfikacja: {args.spec}; dób: {len(days)} ({days[0].date()} - {days[-1].date()})", flush=True)

    if not args.report_only:
        compute(args.spec, days, panels, args.jobs, args.models, args.dry_run)
    report(args.spec, days, price, args.dry_run)


if __name__ == "__main__":
    main()