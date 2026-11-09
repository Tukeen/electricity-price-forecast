"""Testy statystyczne do H1-H3 (okres 2021-2025, specyfikacja główna).

python -m src.evaluation.run_hypotheses
"""
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from src.evaluation.inference import gain_difference, holm
from src.evaluation.metrics import diebold_mariano, errors, mae
from src.evaluation.protocol import load_forecasts
from src.evaluation.report import common_days, dm_table, groups
from src.features.panels import DB_PATH, load_panels
from src.utils.config import PROJECT_ROOT, load_config

import argparse

CFG = load_config()
EV = CFG["evaluation"]
REGIMES = CFG["regimes"]
MODELS = ["arx", "arx_x", "lear", "lear_x", "sarima", "sarimax", "xgb", "xgb_x", "mlp", "mlp_x"]
PAIRS = [("xgb_x", "lear_x"), ("xgb_x", "sarimax"), ("mlp_x", "lear_x"), ("mlp_x", "sarimax"),
         ("lear_x", "sarimax"), ("xgb", "lear"), ("mlp", "lear"), ("xgb", "sarima")]
ML = {"xgb": "xgb_x", "mlp": "mlp_x"}
ECON = {"arx": "arx_x", "lear": "lear_x", "sarima": "sarimax"}
OUT = PROJECT_ROOT / "reports" / "tables"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="forecasts")
    args = ap.parse_args()
    suffix = "" if args.table == "forecasts" else f"_{args.table}"
    price = load_panels()["price"]
    days = price.loc[pd.Timestamp(EV["regime_analysis"]["start"]): pd.Timestamp(EV["validation"]["end"])].index
    errs = {m: errors(load_forecasts(DB_PATH, m, args.table).reindex(days), price) for m in MODELS}
    errs = common_days(errs)
    days = next(iter(errs.values())).index
    grp = groups(days, REGIMES)

    # 1. DM według lat i reżimów, p po poprawce Holma w obrębie (para, typ podziału)
    dm = dm_table(errs, PAIRS, REGIMES)
    dm["typ"] = np.where(dm["grupa"].str.isdigit(), "rok",
                         np.where(dm["grupa"].isin(list(REGIMES)), "reżim", "okres"))
    dm["p_holm"] = dm["p"]
    for _, idx in dm[dm["typ"] != "okres"].groupby(["A", "B", "typ"]).groups.items():
        dm.loc[idx, "p_holm"] = holm(dm.loc[idx, "p"]).to_numpy()
    dm.to_csv(OUT / f"hypotheses_dm{suffix}.csv", index=False, encoding="utf-8-sig")
    order = [f"{a} vs {b}" for a, b in PAIRS]
    for typ in ("rok", "reżim"):
        t = dm[dm["typ"] == typ].copy()
        t["para"] = t["A"] + " vs " + t["B"]
        t["wynik"] = t.apply(lambda r: f"{r['DM']:+.2f} ({r['p_holm']:.3f})", axis=1)
        print(f"\n=== DM (p po poprawce Holma w obrębie pary), podział: {typ}; DM<0: pierwszy model lepszy ===")
        print(t.pivot(index="para", columns="grupa", values="wynik").reindex(order).to_string())

    # 2. Doby z ujemną ceną w reżimie OZE (opisowo: podział po zrealizowanej cenie)
    neg = (price.reindex(days).min(axis=1) < 0).to_numpy()
    oze = grp["oze_ujemne"].to_numpy()
    subsets = {"doby z ujemną ceną": oze & neg, "pozostałe doby reżimu OZE": oze & ~neg}
    mae_rows, pair_rows = [], []
    for label, mask in subsets.items():
        mae_rows.append({"podzbiór": label, "dni": int(mask.sum()), **{m: mae(errs[m].loc[mask]) for m in MODELS}})
        for a, b in PAIRS:
            stat, p = diebold_mariano(errs[a].loc[mask], errs[b].loc[mask])
            ma, mb = mae(errs[a].loc[mask]), mae(errs[b].loc[mask])
            pair_rows.append({"podzbiór": label, "para": f"{a} vs {b}", "dni": int(mask.sum()),
                              "A/B - 1 [%]": 100 * (ma / mb - 1), "DM": stat, "p": p})
    pd.DataFrame(pair_rows).to_csv(OUT / f"hypotheses_negative_days{suffix}.csv", index=False, encoding="utf-8-sig")
    print("\n=== MAE w reżimie OZE: doby z ujemną ceną i pozostałe ===")
    print(pd.DataFrame(mae_rows).set_index("podzbiór").round(2).to_string())
    pairs_tab = pd.DataFrame(pair_rows)
    print("\n=== Różnica MAE [%] (A/B - 1) i DM; <0: A lepszy ===")
    print(pairs_tab.assign(wynik=pairs_tab.apply(lambda r: f"{r['A/B - 1 [%]']:+.1f}% (DM {r['DM']:+.2f}, p {r['p']:.3f})", axis=1))
          .pivot(index="para", columns="podzbiór", values="wynik").reindex(order).to_string())

    # 3. H3: czy zysk z egzogenicznych jest większy dla ML niż dla ekonometrii (bootstrap blokowy 7 dób)
    daily = {m: errs[m].abs().mean(axis=1).to_numpy() for m in MODELS}
    rows = []
    for g in ["całość", "2021-2024", *REGIMES]:
        mask = grp[g].to_numpy()
        for mb, mx in ML.items():
            for eb, ex in ECON.items():
                point, lo, hi, prob = gain_difference(daily[mb][mask], daily[mx][mask], daily[eb][mask], daily[ex][mask])
                rows.append({"grupa": g, "ML": mx, "ECON": ex,
                             "zysk ML [%]": 100 * (daily[mx][mask].mean() / daily[mb][mask].mean() - 1),
                             "zysk ECON [%]": 100 * (daily[ex][mask].mean() / daily[eb][mask].mean() - 1),
                             "Delta [p.p.]": 100 * point, "CI dolna": 100 * lo, "CI górna": 100 * hi,
                             "P(Delta<0)": prob})
    boot = pd.DataFrame(rows)
    boot.to_csv(OUT / f"hypotheses_h3_bootstrap{suffix}.csv", index=False, encoding="utf-8-sig")
    print("\n=== H3: Delta = zysk ML - zysk ECON po dodaniu zmiennych egzogenicznych (Delta<0: ML zyskuje więcej) ===")
    print(boot.round(2).to_string(index=False))


if __name__ == "__main__":
    main()