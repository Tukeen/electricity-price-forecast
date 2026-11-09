"""H4 (część 2): współczynniki LEAR_X i wkłady TreeSHAP dla XGB_X.

Okno uczące kończy się przed ostatnim dniem walidacji (rok 2026 nie jest używany).
python -m src.evaluation.interpret
"""
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LassoLarsIC
from xgboost import XGBRegressor

from src.evaluation.registry import build_model
from src.features.panels import info_set, load_panels
from src.models.linear import EXOG, LAGS, _asinh_params, _standardize, lear_matrix
from src.models.xgb import build_long
from src.utils.config import PROJECT_ROOT, load_config

CFG = load_config()
TABLES = PROJECT_ROOT / "reports" / "tables"
FIGURES = PROJECT_ROOT / "reports" / "figures"
HOURS = (3, 12, 19)  # noc, południe (dołek PV), wieczorny szczyt


def group_of(name: str) -> str:
    if name.startswith("p_"):
        return "ceny opóźnione"
    if name.startswith("load"):
        return "zapotrzebowanie (prognoza)"
    if name.startswith("wind"):
        return "wiatr (prognoza)"
    if name.startswith("pv"):
        return "PV (prognoza)"
    if name.startswith("resid"):
        return "obciążenie rezydualne"
    return "kalendarz"


def window_inputs(model, panels):
    day = pd.Timestamp(CFG["evaluation"]["validation"]["end"])
    info = info_set(panels, day)
    price = info["price"]
    win = price.index[-model.window_days:]
    a, b = _asinh_params(price.loc[win].to_numpy().ravel())
    hist = price.iloc[-(model.window_days + max(LAGS)):]
    p = np.arcsinh((hist - a) / b)
    p = p.reindex(p.index.append(pd.DatetimeIndex([day])))
    return day, win, p, {k: info[k] for k in EXOG}


def lear_coefficients(panels) -> pd.DataFrame:
    model = build_model("lear_x")
    day, win, p, exog = window_inputs(model, panels)
    X = lear_matrix(p, exog, True)
    Y, Xtr = p.loc[win], X.loc[win]
    ok = Xtr.notna().all(axis=1) & Y.notna().all(axis=1)
    Z, _ = _standardize(Xtr[ok], X.loc[[day]])
    coef = {h: pd.Series(LassoLarsIC(criterion="aic").fit(Z, Y[ok][h].to_numpy()).coef_, index=X.columns)
            for h in range(24)}
    return pd.DataFrame(coef)  # cechy x godziny (skala: standaryzowane cechy, asinh ceny)


def xgb_shap(panels) -> pd.Series:
    model = build_model("xgb_x")
    day, win, p, exog = window_inputs(model, panels)
    X, y = build_long(p, exog, True)
    train = X.index.get_level_values("day").isin(win) & ~np.isnan(y)
    Xtr = X[train]
    est = XGBRegressor(**model.params).fit(Xtr, y[train])
    contrib = est.get_booster().predict(xgb.DMatrix(Xtr), pred_contribs=True)
    return pd.Series(np.abs(contrib[:, :-1]).mean(axis=0), index=Xtr.columns)


def shares(values: pd.Series) -> pd.Series:
    g = values.abs().groupby(values.index.map(group_of)).sum()
    return (g / g.sum()).sort_values(ascending=False)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    panels = load_panels()

    coef = lear_coefficients(panels)
    coef.to_csv(TABLES / "interpret_lear_x_coefficients.csv", encoding="utf-8-sig")
    nonzero = (coef != 0).sum()
    print(f"LEAR_X: niezerowych współczynników na godzinę (z {len(coef)}): "
          f"min {nonzero.min()}, mediana {int(nonzero.median())}, max {nonzero.max()}")
    for h in HOURS:
        c = coef[h]
        top = c[c != 0].reindex(c[c != 0].abs().sort_values(ascending=False).index).head(10)
        print(f"\n--- LEAR_X, godzina {h}: 10 największych współczynników ---")
        print(top.round(3).to_string())
        print("udział |współczynników| wg grup:", shares(c).round(3).to_dict())
    mean_share = pd.concat([shares(coef[h]) for h in range(24)], axis=1).fillna(0).mean(axis=1)
    print("\nLEAR_X: średni udział |współczynników| wg grup (24 godziny):")
    print(mean_share.sort_values(ascending=False).round(3).to_string())

    imp = xgb_shap(panels)
    imp.sort_values(ascending=False).to_csv(TABLES / "interpret_xgb_x_shap.csv", encoding="utf-8-sig")
    print("\n--- XGB_X: średni |SHAP| (skala asinh) ---")
    print(imp.sort_values(ascending=False).head(12).round(4).to_string())
    print("udział wg grup:", shares(imp).round(3).to_dict())

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    c = coef[19]
    c = c.reindex(c.abs().sort_values(ascending=False).index).head(12)[::-1]
    ax[0].barh(c.index, c.values, color=["tab:red" if v < 0 else "tab:blue" for v in c.values])
    ax[0].set_title("LEAR_X, godzina 19: współczynniki (cechy standaryzowane)")
    s = imp.sort_values(ascending=False).head(12)[::-1]
    ax[1].barh(s.index, s.values)
    ax[1].set_title("XGB_X: średni |SHAP|")
    fig.tight_layout()
    fig.savefig(FIGURES / "05_interpretability.png", dpi=150)
    plt.close(fig)
    print(f"\nTabele: {TABLES}\nWykres: {FIGURES / '05_interpretability.png'}")


if __name__ == "__main__":
    main()