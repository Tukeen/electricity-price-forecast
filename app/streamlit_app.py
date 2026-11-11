"""Aplikacja Streamlit: prognozy cen energii na rynku dnia następnego w Polsce (praca magisterska, AGH)."""
from __future__ import annotations
 
import json
from datetime import timedelta
from pathlib import Path
 
import pandas as pd
import streamlit as st
 
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "app_data"
RESEARCH = DATA / "research"
LABELS = {
    "naive_mixed": "Naiwny (D-1 / D-7)", "arx": "ARX", "lear": "LEAR", "xgb": "XGBoost", "mlp": "MLP",
    "sarima": "SARIMA", "arx_x": "ARX + zm. zewnętrzne", "lear_x": "LEAR + zm. zewnętrzne",
    "xgb_x": "XGBoost + zm. zewnętrzne", "mlp_x": "MLP + zm. zewnętrzne", "sarimax": "SARIMAX",
}
GROUP_ORDER = ["całość", "2021-2024", "stabilny", "kryzys", "oze_ujemne", "2021", "2022", "2023", "2024", "2025"]
 
st.set_page_config(page_title="Prognozy cen energii na RDN w Polsce", layout="wide")
 
 
def label(model: str) -> str:
    return LABELS.get(model, model)
 
 
@st.cache_data(ttl=900)
def load_status() -> dict:
    p = DATA / "status.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
 
 
@st.cache_data(ttl=900)
def load_prices() -> pd.DataFrame:
    p = DATA / "prices_hourly.csv"
    return pd.read_csv(p, parse_dates=["delivery_date"]) if p.exists() else pd.DataFrame()
 
 
@st.cache_data(ttl=900)
def load_log() -> pd.DataFrame:
    p = DATA / "forecast_log.csv"
    return pd.read_csv(p, dtype={"delivery_date": str}) if p.exists() else pd.DataFrame()
 
 
def tab_forecast(status: dict, prices: pd.DataFrame, log: pd.DataFrame) -> None:
    if log.empty:
        st.info("Brak zapisanych prognoz. Pierwsza pojawi się po uruchomieniu automatyzacji.")
        return
    days = sorted(log["delivery_date"].unique(), reverse=True)
    day = st.selectbox("Doba dostawy", days, index=0)
    fc = log[log["delivery_date"] == day]
    wide = fc.pivot_table(index="hour_local", columns="model", values="y_pred").rename(columns=label)
    actual = prices[prices["delivery_date"] == pd.Timestamp(day)].groupby("hour_local")["price_eur_mwh"].mean()
    if len(actual) and actual.notna().any():
        wide["Cena rzeczywista RDN"] = actual
    st.line_chart(wide)
    st.caption("Oś pozioma: godzina doby dostawy (czas lokalny), oś pionowa: EUR/MWh. "
               "Godzina i cena to średnia z czterech kwadransów.")
    summary = pd.DataFrame({"średnia": wide.mean(), "minimum": wide.min(), "maksimum": wide.max()}).round(1)
    st.dataframe(summary, width="stretch")
    if status:
        with st.expander("Status ostatniego uruchomienia"):
            st.write(f"Uruchomienie (UTC): {status.get('run_utc')} | ostatnia doba ze znanymi cenami: "
                     f"{status.get('last_price_day')} | wersja kodu: {status.get('code_version')}")
            st.dataframe(pd.DataFrame(status.get("models", {})).T, width="stretch")
 
 
def tab_live(prices: pd.DataFrame, log: pd.DataFrame) -> None:
    if log.empty or prices.empty:
        st.info("Brak danych.")
        return
    actual = (prices.groupby(["delivery_date", "hour_local"])["price_eur_mwh"].mean().rename("actual").reset_index())
    log = log.assign(delivery_date=pd.to_datetime(log["delivery_date"]))
    j = log.merge(actual, on=["delivery_date", "hour_local"]).dropna(subset=["actual"])
    if j.empty:
        st.info("Brak dób z zapisaną prognozą i znaną ceną rzeczywistą.")
        return
    n_models = j.groupby("delivery_date")["model"].nunique()
    j = j[j["delivery_date"].isin(n_models[n_models == n_models.max()].index)]  # tylko doby z kompletem modeli
    j = j.assign(err=j["y_pred"] - j["actual"], abs_err=(j["y_pred"] - j["actual"]).abs())
    stats = j.groupby("model").agg(dni=("delivery_date", "nunique"), MAE=("abs_err", "mean"),
                                   RMSE=("err", lambda s: float((s ** 2).mean() ** 0.5)), ME=("err", "mean"))
    if "naive_mixed" in stats.index:
        stats["rMAE"] = stats["MAE"] / stats.loc["naive_mixed", "MAE"]
    st.dataframe(stats.rename(index=label).round(2), width="stretch")
    st.caption("Skuteczność prognoz zapisanych przed poznaniem cen. To uzupełniająca walidacja na żywo, "
               "nie wynik testu końcowego 2026.")
    daily = j.groupby(["delivery_date", "model"])["abs_err"].mean().unstack().rename(columns=label)
    st.line_chart(daily)
 
 
def tab_history(prices: pd.DataFrame) -> None:
    if prices.empty:
        st.info("Brak danych.")
        return
    daily = prices.groupby("delivery_date")["price_eur_mwh"].agg(["mean", "min", "max"]).dropna()
    last = daily.index.max().date()
    lo, hi = st.slider("Okres", min_value=daily.index.min().date(), max_value=last,
                       value=(last - timedelta(days=365), last))
    sel = daily.loc[pd.Timestamp(lo): pd.Timestamp(hi)]
    st.line_chart(sel.rename(columns={"mean": "średnia dobowa", "min": "minimum", "max": "maksimum"}))
    neg = prices.assign(miesiąc=prices["delivery_date"].dt.to_period("M").dt.to_timestamp(),
                        ujemna=prices["price_eur_mwh"] < 0).groupby("miesiąc")["ujemna"].sum()
    st.subheader("Godziny z ceną ujemną według miesięcy")
    st.bar_chart(neg)
 
 
FREEZE_TAG = "pre-test-2026"


def _csv(name: str) -> pd.DataFrame | None:
    p = RESEARCH / name
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None


def _freeze_note() -> str:
    info = RESEARCH / "frozen_data.json"
    if info.exists():
        try:
            d = json.loads(info.read_text(encoding="utf-8"))
            return f" Odcisk danych testowych: {str(d.get('fingerprint', d.get('sha256', '')))[:16] or 'zapisany w repozytorium'}."
        except Exception:
            return ""
    return ""


def _metric_table(df: pd.DataFrame, order: list[str], key: str) -> None:
    models = sorted(df["model"].unique(), key=lambda m: (not m.startswith("naive"), m))
    default = [m for m in ["naive_mixed", "lear", "lear_x", "sarimax", "xgb", "xgb_x", "mlp_x"] if m in models]
    chosen = st.multiselect("Modele", models, default=default, key=f"models_{key}")
    metric = st.radio("Miara", ["MAE", "rMAE", "RMSE"], horizontal=True, key=f"metric_{key}")
    if chosen:
        tab = df.pivot(index="grupa", columns="model", values=metric)[chosen]
        rows = [g for g in order if g in tab.index] or list(tab.index)
        st.dataframe(tab.loc[rows].round(3), width="stretch")


def _research_dev() -> None:
    df = _csv("ml_results.csv")
    if df is None:
        st.info("Tabele z wynikami okresu rozwojowego nie zostały jeszcze dodane do repozytorium.")
        return
    st.subheader("Okres rozwojowy 2021-2025 (rolling forecast, specyfikacja główna)")
    _metric_table(df, GROUP_ORDER, "dev")
    st.caption("MAE i RMSE w EUR/MWh; rMAE względem modelu naiwnego (D-1 / D-7). Rok 2025 posłużył do strojenia "
               "hiperparametrów, więc jego wyniki są zawyżone względem testu na danych nowych.")
    g = _csv("exog_gain.csv")
    if g is not None:
        st.subheader("Zmiana MAE po dodaniu zmiennych zewnętrznych [%]")
        st.dataframe((g.set_index(g.columns[0]) * 100).round(1), width="stretch")
    for name, title in (("hypotheses_h3_bootstrap.csv", "H3: różnica zysków z zm. zewnętrznych (bootstrap blokowy)"),
                        ("hypotheses_dm.csv", "Testy Diebolda-Mariano (okres rozwojowy)")):
        t = _csv(name)
        if t is not None:
            with st.expander(title):
                st.dataframe(t.round(3), width="stretch")


def _research_test() -> None:
    mains = {"Specyfikacja główna": "test_main", "Specyfikacja wrażliwości": "test_sens"}
    found = {label_: pre for label_, pre in mains.items() if (RESEARCH / f"{pre}_results.csv").exists()}
    if not found:
        st.info("Test końcowy na roku 2026 zostanie uruchomiony jednorazowo w styczniu 2027, po zamknięciu roku "
                "i zamrożeniu kodu oraz danych. Do tego czasu ta zakładka pozostaje pusta, a ocena modeli na danych "
                "z lat 2021-2025 jest w zakładce „Rozwój 2021-2025”.")
        return
    spec = st.radio("Specyfikacja", list(found), horizontal=True)
    pre = found[spec]
    res = _csv(f"{pre}_results.csv")
    st.subheader(f"Test końcowy 2026: {spec.lower()}")
    order = ["całość", "Q1", "Q2", "Q3", "Q4"] + [f"{m:02d}" for m in range(1, 13)]
    _metric_table(res, order, pre)
    st.caption("Grupy: cały rok, kwartały (Q1-Q4) i miesiące (01-12). rMAE względem modelu naiwnego (D-1 / D-7).")
    for suffix, title in (("dm", "Testy Diebolda-Mariano (p_holm dla czterech testów potwierdzających)"),
                          ("h3_bootstrap", "H3: bootstrap blokowy"),
                          ("negative_days", "Doby z ujemną ceną i pozostałe"),
                          ("intervals90_monthly", "H4: przedziały 90% według miesięcy")):
        t = _csv(f"{pre}_{suffix}.csv")
        if t is not None:
            with st.expander(title):
                st.dataframe(t.round(3), width="stretch")


def tab_research() -> None:
    st.warning(
        "**Wyniki zamrożone.** Ta zakładka zawiera wyniki analizowane w pracy magisterskiej. Nie są aktualizowane "
        "przez codzienną automatyzację: powstają jednorazowo i trafiają tu ręcznie. Kod badania jest zamrożony "
        f"tagiem git `{FREEZE_TAG}` (commit d3bbfa7), a test na roku 2026 uruchamiany jest raz, w styczniu 2027."
        + _freeze_note())
    dev, test = st.tabs(["Rozwój 2021-2025", "Test 2026"])
    with dev:
        _research_dev()
    with test:
        _research_test()


def tab_about() -> None:
    st.markdown("""
**Cel.** Porównanie metod ekonometrycznych (ARX, LEAR, SARIMA/SARIMAX) i uczenia maszynowego (XGBoost, MLP) w
prognozowaniu cen energii elektrycznej na polskim rynku dnia następnego, z uwzględnieniem reżimów rynkowych
(praca magisterska, AGH).
 
**Dane.** ENTSO-E Transparency Platform: ceny RDN, prognozy zapotrzebowania oraz wiatru i PV. Rozdzielczość godzinowa
(od 1.10.2025 średnia z czterech kwadransów).
 
**Prognoza operacyjna.** Rano, przed zamknięciem aukcji, wyznaczany jest profil 24 cen na kolejną dobę. W prognozie
operacyjnej działają wyłącznie modele bez zmiennych zewnętrznych, ponieważ prognozy wiatru i PV na dobę następną
ENTSO-E publikuje dopiero po południu (najpóźniej o 18:00), czyli po aukcji. Modele ze zmiennymi zewnętrznymi
(„+ zm. zewnętrzne” oraz SARIMAX) są oceniane w badaniu, przy założeniu, że takie prognozy są znane.
Pierwsza zapisana prognoza doby nie jest później zmieniana.
 
**Ograniczenia.** Projekt badawczy; prognozy nie stanowią porady inwestycyjnej. W prognozie operacyjnej SARIMA(X) jest
estymowany codziennie, a w badaniu raz na 28 dób.
""")
 
 
st.title("Prognozy cen energii elektrycznej na rynku dnia następnego (Polska)")
prices_df, log_df, status_df = load_prices(), load_log(), load_status()
t1, t2, t3, t4, t5 = st.tabs(["Prognoza", "Skuteczność na żywo", "Dane historyczne", "Wyniki badania", "O projekcie"])
with t1:
    tab_forecast(status_df, prices_df, log_df)
with t2:
    tab_live(prices_df, log_df)
with t3:
    tab_history(prices_df)
with t4:
    tab_research()
with t5:
    tab_about()