"""Aplikacja Streamlit: prognozy cen energii na rynku dnia następnego w Polsce (praca magisterska, AGH)."""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "app_data"
RESEARCH = DATA / "research"
TZ = "Europe/Warsaw"
FREEZE_TAG = "pre-test-2026"

ECON, ML, BENCH, OTHER, COMB = "ekonometria", "uczenie maszynowe", "benchmark", "inne", "kombinacja"
MODELS = {  # klucz: (nazwa, grupa, kolor)
    "naive_mixed": ("Naiwny (D-1 / D-7)", BENCH, "#9E9E9E"),
    "naive_daily": ("Naiwny (D-1)", BENCH, "#BDBDBD"),
    "arx": ("ARX", ECON, "#1F5FA8"), "arx_x": ("ARX + zm. zewn.", ECON, "#5B8FD0"),
    "lear": ("LEAR", ECON, "#0B3D91"), "lear_x": ("LEAR + zm. zewn.", ECON, "#4A78C2"),
    "sarima": ("SARIMA", ECON, "#3FA7D6"), "sarimax": ("SARIMAX", ECON, "#7CC6E8"),
    "var1": ("VAR(1)", ECON, "#2E7D9A"),
    "xgb": ("XGBoost", ML, "#D9822B"), "xgb_x": ("XGBoost + zm. zewn.", ML, "#F0A860"),
    "mlp": ("MLP", ML, "#B5432F"), "mlp_x": ("MLP + zm. zewn.", ML, "#E07A63"),
    "prophet": ("Prophet", OTHER, "#2E9E6E"), "prophet_x": ("Prophet + zm. zewn.", OTHER, "#6CC49A"),
    "lear_nl": ("LEAR nieliniowy", ECON, "#28508F"),
    "avg": ("Średnia modeli", COMB, "#6A3D9A"),
}
ACTUAL = "Cena rzeczywista"
GROUP_NAMES = {"całość": "cały okres 2021–2025", "2021-2024": "lata 2021–2024", "stabilny": "reżim stabilny",
               "kryzys": "reżim kryzysowy", "oze_ujemne": "reżim OZE / ceny ujemne"}
DAYS = ["poniedziałek", "wtorek", "środa", "czwartek", "piątek", "sobota", "niedziela"]

st.set_page_config(page_title="Prognozy cen energii na RDN w Polsce", layout="wide")


def name(m: str) -> str:
    return MODELS.get(m, (m, OTHER, "#777"))[0]


def group(m: str) -> str:
    return MODELS.get(m, (m, OTHER, "#777"))[1]


def color_scale(names: list[str]) -> alt.Scale:
    by_name = {v[0]: v[2] for v in MODELS.values()} | {ACTUAL: "#000000"}
    return alt.Scale(domain=names, range=[by_name.get(n, "#777777") for n in names])


def day_label(d) -> str:
    d = pd.Timestamp(d)
    return f"{DAYS[d.weekday()]} {d:%d.%m.%Y}"


def local_time(ts: str) -> str:
    try:
        return pd.Timestamp(ts).tz_convert(TZ).strftime("%d.%m.%Y %H:%M")
    except Exception:
        return str(ts)


# ---------------------------------------------------------------- dane
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
    if not p.exists():
        return pd.DataFrame()
    log = pd.read_csv(p, parse_dates=["delivery_date"])
    models = [m for m in log["model"].unique() if not m.startswith("naive")]
    avg = (log[log["model"].isin(models)].groupby(["delivery_date", "hour_local"], as_index=False)
           .agg(y_pred=("y_pred", "mean"), run_utc=("run_utc", "min")).assign(model="avg"))
    return pd.concat([log, avg], ignore_index=True)


def _csv(name_: str) -> pd.DataFrame | None:
    p = RESEARCH / name_
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None


@st.cache_data(ttl=900)
def verification(log: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Błędy godzinowe prognoz z dziennika dla dób ze znaną ceną i kompletem modeli."""
    if log.empty or prices.empty:
        return pd.DataFrame()
    actual = prices.groupby(["delivery_date", "hour_local"])["price_eur_mwh"].mean().rename("actual").reset_index()
    j = log.merge(actual, on=["delivery_date", "hour_local"]).dropna(subset=["actual"])
    if j.empty:
        return j
    n = j.groupby("delivery_date")["model"].nunique()
    j = j[j["delivery_date"].isin(n[n == n.max()].index)]
    return j.assign(err=j["y_pred"] - j["actual"], abs_err=(j["y_pred"] - j["actual"]).abs())


def ranking(ver: pd.DataFrame, last_days: int | None) -> pd.DataFrame:
    days = sorted(ver["delivery_date"].unique())
    if last_days:
        days = days[-last_days:]
    v = ver[ver["delivery_date"].isin(days)]
    daily = v.groupby(["delivery_date", "model"])["abs_err"].mean().unstack()
    wins = daily.drop(columns=["avg"], errors="ignore").idxmin(axis=1).value_counts()
    t = v.groupby("model").agg(MAE=("abs_err", "mean"), RMSE=("err", lambda s: float((s ** 2).mean() ** 0.5)),
                               ME=("err", "mean"))
    if "naive_mixed" in t.index:
        t["rMAE"] = t["MAE"] / t.loc["naive_mixed", "MAE"]
    t["dni najlepszy"] = wins.reindex(t.index).fillna(0).astype(int)
    t = t.sort_values("MAE")
    t.insert(0, "grupa", [group(m) for m in t.index])
    t.insert(0, "miejsce", range(1, len(t) + 1))
    t.attrs["n_days"] = len(days)
    return t


def best_model(ver: pd.DataFrame) -> tuple[str, str]:
    """Najlepszy pojedynczy model: najniższe MAE z ostatnich 30 dób (min. 7), inaczej wynik z badania."""
    if not ver.empty and ver["delivery_date"].nunique() >= 7:
        r = ranking(ver, 30).drop(index=["avg"], errors="ignore")
        return r.index[0], f"najniższy średni błąd (MAE) w ostatnich {r.attrs['n_days']} dobach"
    res = _csv("ml_results.csv")
    if res is not None:
        r = res[(res["grupa"] == "całość") & res["model"].isin(["arx", "lear", "sarima", "xgb", "mlp"])]
        if not r.empty:
            return r.sort_values("MAE")["model"].iloc[0], "najniższy MAE w badaniu (2021–2025), za mało dób na żywo"
    return "lear", "model domyślny"


# ---------------------------------------------------------------- wykres dobowy
def day_chart(fc: pd.DataFrame, actual: pd.Series | None, highlight: str, height: int = 380) -> alt.Chart:
    df = fc.assign(nazwa=fc["model"].map(name))[["hour_local", "nazwa", "y_pred"]].rename(columns={"y_pred": "cena"})
    if actual is not None and actual.notna().any():
        df = pd.concat([df, pd.DataFrame({"hour_local": actual.index, "nazwa": ACTUAL, "cena": actual.values})])
    order = [ACTUAL] if (df["nazwa"] == ACTUAL).any() else []
    order += [name(highlight), name("avg")] + sorted(set(df["nazwa"]) - {ACTUAL, name(highlight), name("avg")})
    order = [o for o in dict.fromkeys(order) if o in set(df["nazwa"])]
    df["szer"] = df["nazwa"].map(lambda n: 4.0 if n == ACTUAL else 3.2 if n == name(highlight) else 1.4)
    df["kreska"] = df["nazwa"].map(lambda n: "przerywana" if n == name("avg") else "ciągła")
    df["cena"] = df["cena"].round(1)
    base = alt.Chart(df).encode(
        x=alt.X("hour_local:Q", title="godzina doby dostawy (czas lokalny)", scale=alt.Scale(domain=[0, 23]),
                axis=alt.Axis(tickMinStep=1, values=list(range(0, 24)))),
        y=alt.Y("cena:Q", title="EUR/MWh"),
        color=alt.Color("nazwa:N", title=None, scale=color_scale(order), sort=order,
                        legend=alt.Legend(orient="bottom", columns=4)),
        strokeWidth=alt.StrokeWidth("szer:Q", legend=None, scale=None),
        strokeDash=alt.StrokeDash("kreska:N", legend=None,
                                  scale=alt.Scale(domain=["ciągła", "przerywana"], range=[[1, 0], [5, 3]])),
        tooltip=[alt.Tooltip("nazwa:N", title="model"), alt.Tooltip("hour_local:Q", title="godzina"),
                 alt.Tooltip("cena:Q", title="EUR/MWh")],
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#BBBBBB").encode(y="y:Q")
    return (zero + base.mark_line()).properties(height=height)


# ---------------------------------------------------------------- zakładki
def tab_forecast(status: dict, prices: pd.DataFrame, log: pd.DataFrame, ver: pd.DataFrame) -> None:
    if log.empty:
        st.info("Brak zapisanych prognoz.")
        return
    day = log["delivery_date"].max()
    fc = log[log["delivery_date"] == day]
    best, why = best_model(ver)
    run = local_time(fc["run_utc"].min())
    auction = (day - timedelta(days=1)).strftime("%d.%m.%Y")
    st.subheader(f"Prognoza cen na {day_label(day)}")
    st.markdown(f"Prognoza sporządzona **{run}** (czas lokalny), przed aukcją rozstrzygającą ceny tej doby "
                f"(**{auction}, 12:00**). Modele znały ceny do {(day - timedelta(days=1)):%d.%m.%Y} włącznie.")
    actual = prices[prices["delivery_date"] == day].groupby("hour_local")["price_eur_mwh"].mean()
    if actual.notna().any():
        st.info("Ceny tej doby są już znane – porównanie prognozy z rzeczywistością jest w zakładce „Weryfikacja”.")

    main = fc[fc["model"] == best].set_index("hour_local")["y_pred"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Średnia cena (prognoza)", f"{main.mean():.1f} EUR/MWh")
    c2.metric(f"Najdroższa godzina ({main.max():.0f} EUR/MWh)", f"{int(main.idxmax())}:00")
    c3.metric(f"Najtańsza godzina ({main.min():.0f} EUR/MWh)", f"{int(main.idxmin())}:00")
    c4.metric("Godziny z ceną ujemną", int((main < 0).sum()))
    st.caption(f"Wartości w kafelkach według prognozy głównej: **{name(best)}** ({group(best)}) – {why}.")

    all_models = list(dict.fromkeys([best, "avg"] + sorted(fc["model"].unique(), key=lambda m: (group(m), m))))
    chosen = st.multiselect("Modele na wykresie", all_models, default=all_models, format_func=name)
    st.altair_chart(day_chart(fc[fc["model"].isin(chosen)], actual if actual.notna().any() else None, best),
                    width="stretch")
    st.caption("Gruba linia: prognoza główna (najlepszy model). Przerywana: średnia modeli. "
               "Odcienie niebieskie: modele ekonometryczne, pomarańczowe i czerwone: uczenie maszynowe, szary: benchmark.")

    table = fc.pivot_table(index="hour_local", columns="model", values="y_pred")[all_models]
    table.columns = [name(m) for m in table.columns]
    table.index.name = "godzina"
    with st.expander("Tabela prognoz godzinowych [EUR/MWh]"):
        st.dataframe(table.round(1), width="stretch", height=880)


def tab_verify(ver: pd.DataFrame, prices: pd.DataFrame) -> None:
    if ver.empty:
        st.info("Brak dób z zapisaną prognozą i znaną ceną rzeczywistą.")
        return
    days = sorted(ver["delivery_date"].unique(), reverse=True)
    day = st.selectbox("Doba dostawy", days, index=0, format_func=day_label)
    v = ver[ver["delivery_date"] == day]
    actual = v.drop_duplicates("hour_local").set_index("hour_local")["actual"].sort_index()
    d = (v[~v["model"].isin(["avg", "naive_mixed", "naive_daily"])].groupby("model").agg(MAE=("abs_err", "mean"), ME=("err", "mean"))
         .sort_values("MAE"))
    best, worst = d.index[0], d.index[-1]
    run = local_time(v["run_utc"].min())
    st.markdown(f"Prognozy sporządzone **{run}**, cena ustalona na aukcji {(day - timedelta(days=1)):%d.%m.%Y}. "
                f"Średnia cena rzeczywista: **{actual.mean():.1f} EUR/MWh**.")
    c1, c2 = st.columns(2)
    c1.success(f"Najlepszy model tej doby (bez benchmarku): **{name(best)}** ({group(best)}), średni błąd {d.loc[best, 'MAE']:.1f} EUR/MWh")
    c2.error(f"Najsłabszy model tej doby (bez benchmarku): **{name(worst)}** ({group(worst)}), średni błąd {d.loc[worst, 'MAE']:.1f} EUR/MWh")
    st.altair_chart(day_chart(v, actual, best), width="stretch")
    st.caption("Czarna linia: cena rzeczywista. Gruba linia: najlepszy model tej doby. Przerywana: średnia modeli.")
    dd = (v.groupby("model").agg(MAE=("abs_err", "mean"), ME=("err", "mean")).sort_values("MAE"))
    dd.insert(0, "grupa", [group(m) for m in dd.index])
    dd.insert(0, "miejsce", range(1, len(dd) + 1))
    dd.index = [name(m) for m in dd.index]
    st.dataframe(dd.round(2), width="stretch")

    st.divider()
    st.subheader("Ranking modeli")
    period = st.radio("Okres", ["ostatnie 7 dób", "ostatnie 30 dób", "od początku"], horizontal=True, index=1)
    last = {"ostatnie 7 dób": 7, "ostatnie 30 dób": 30, "od początku": None}[period]
    r = ranking(ver, last)
    st.caption(f"Liczba dób w zestawieniu: {r.attrs['n_days']}. MAE: średni błąd bezwzględny w EUR/MWh (im mniej, "
               "tym lepiej); rMAE poniżej 1 oznacza model lepszy od naiwnego; „dni najlepszy”: w ilu dobach model miał "
               "najmniejszy błąd.")
    show = r.copy()
    show.index = [name(m) for m in show.index]
    st.dataframe(show.round(3), width="stretch")

    single = r.drop(index=["avg", "naive_mixed", "naive_daily"], errors="ignore")
    g = single.groupby("grupa")["MAE"]
    if {ECON, ML} <= set(g.groups):
        be, bm = single[single["grupa"] == ECON].index[0], single[single["grupa"] == ML].index[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Najlepszy model ekonometryczny", f"{name(be)} – MAE {single.loc[be, 'MAE']:.2f}")
        c2.metric("Najlepszy model uczenia maszynowego", f"{name(bm)} – MAE {single.loc[bm, 'MAE']:.2f}")
        lead = ECON if single.loc[be, "MAE"] < single.loc[bm, "MAE"] else ML
        diff = abs(single.loc[be, "MAE"] - single.loc[bm, "MAE"])
        c3.metric(f"Przewaga w tym okresie (o {diff:.2f} EUR/MWh)", lead)
        st.caption("Porównanie na żywo obejmuje tylko modele bez zmiennych zewnętrznych i krótki okres, dlatego "
                   "ma charakter poglądowy. Wnioski badawcze opierają się na zakładce „Wyniki badania”.")

    daily = ver.groupby(["delivery_date", "model"])["abs_err"].mean().reset_index()
    daily["MAE 7 dób"] = daily.groupby("model")["abs_err"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    daily["nazwa"] = daily["model"].map(name)
    names = sorted(daily["nazwa"].unique())
    ch = alt.Chart(daily).mark_line().encode(
        x=alt.X("delivery_date:T", title="doba dostawy", axis=alt.Axis(format="%d.%m", tickCount="day")), y=alt.Y("MAE 7 dób:Q", title="MAE, średnia z 7 dób [EUR/MWh]"),
        color=alt.Color("nazwa:N", title=None, scale=color_scale(names), legend=alt.Legend(orient="bottom", columns=4)),
        tooltip=["nazwa", alt.Tooltip("delivery_date:T", title="doba"), alt.Tooltip("MAE 7 dób:Q", format=".1f")],
    ).properties(height=300)
    st.altair_chart(ch, width="stretch")

    with st.expander("Ceny historyczne"):
        tab_history(prices)


def tab_history(prices: pd.DataFrame) -> None:
    daily = prices.groupby("delivery_date")["price_eur_mwh"].agg(["mean", "min", "max"]).dropna()
    if daily.empty:
        return
    last = daily.index.max().date()
    lo, hi = st.slider("Okres", min_value=daily.index.min().date(), max_value=last,
                       value=(last - timedelta(days=365), last))
    sel = daily.loc[pd.Timestamp(lo): pd.Timestamp(hi)]
    st.line_chart(sel.rename(columns={"mean": "średnia dobowa", "min": "minimum", "max": "maksimum"}))
    neg = prices.assign(miesiąc=prices["delivery_date"].dt.to_period("M").dt.to_timestamp(),
                        ujemna=prices["price_eur_mwh"] < 0).groupby("miesiąc")["ujemna"].sum()
    st.markdown("**Liczba godzin z ceną ujemną w miesiącu**")
    st.bar_chart(neg)


def _research_block(res: pd.DataFrame, groups: list[str], key: str) -> None:
    labels = {g: GROUP_NAMES.get(g, g) for g in groups if g in set(res["grupa"])}
    g = st.selectbox("Okres", list(labels), format_func=lambda x: labels[x], key=f"grp_{key}")
    r = res[(res["grupa"] == g) & (res["model"] != "naive_daily")].copy()
    r["nazwa"], r["grupa modelu"] = r["model"].map(name), r["model"].map(group)
    r = r.sort_values("MAE")
    colors = {ECON: "#1F5FA8", ML: "#D9822B", BENCH: "#9E9E9E", OTHER: "#2E9E6E"}
    bars = alt.Chart(r).mark_bar().encode(
        x=alt.X("MAE:Q", title="MAE [EUR/MWh]"),
        y=alt.Y("nazwa:N", sort=list(r["nazwa"]), title=None, axis=alt.Axis(labelLimit=220, labelOverlap=False)),
        color=alt.Color("grupa modelu:N", title=None, scale=alt.Scale(domain=list(colors), range=list(colors.values())),
                        legend=alt.Legend(orient="bottom")),
        tooltip=["nazwa", alt.Tooltip("MAE:Q", format=".2f"), alt.Tooltip("rMAE:Q", format=".3f"), "dni"],
    ).properties(height=26 * len(r))
    text = alt.Chart(r).mark_text(align="left", dx=4, fontSize=11).encode(
        x="MAE:Q", y=alt.Y("nazwa:N", sort=list(r["nazwa"])), text=alt.Text("MAE:Q", format=".2f"))
    st.altair_chart(bars + text, width="stretch")
    best_e = r[r["grupa modelu"] == ECON].iloc[0]
    best_m = r[r["grupa modelu"] == ML].iloc[0]
    c1, c2 = st.columns(2)
    c1.metric("Najlepszy model ekonometryczny", f"{best_e['nazwa']} – MAE {best_e['MAE']:.2f}")
    c2.metric("Najlepszy model uczenia maszynowego", f"{best_m['nazwa']} – MAE {best_m['MAE']:.2f}")
    t = r.set_index("nazwa")[["grupa modelu", "dni", "MAE", "RMSE", "ME", "rMAE"]]
    with st.expander("Tabela miar dokładności"):
        st.dataframe(t.round(3), width="stretch")


def _research_dev() -> None:
    res = _csv("ml_results.csv")
    if res is None:
        st.info("Brak tabel z wynikami okresu rozwojowego.")
        return
    st.markdown("Prognozy kroczące dla każdej doby lat 2021–2025, specyfikacja główna. Rok 2025 posłużył także "
                "do doboru konfiguracji modeli, dlatego ostateczną ocenę stanowi test na danych z 2026 r.")
    _research_block(res, ["całość", "stabilny", "kryzys", "oze_ujemne", "2021", "2022", "2023", "2024", "2025"], "dev")
    g = _csv("exog_gain.csv")
    if g is not None:
        with st.expander("Zmiana MAE po dodaniu zmiennych zewnętrznych [%] (hipoteza H3)"):
            g = g.set_index(g.columns[0])
            g.index = [GROUP_NAMES.get(i, i) for i in g.index]
            st.dataframe((g * 100).round(1), width="stretch")
    dm = _csv("hypotheses_dm.csv")
    if dm is not None:
        with st.expander("Test Diebolda–Mariano: czy różnica błędów jest istotna"):
            st.caption("Ujemna statystyka DM: model A ma mniejszy błąd niż B. p_holm < 0,05: różnica istotna.")
            dm = dm.assign(A=dm["A"].map(name), B=dm["B"].map(name), grupa=dm["grupa"].map(lambda x: GROUP_NAMES.get(x, x)))
            st.dataframe(dm.drop(columns=["typ"]).round(4), width="stretch", hide_index=True)
    h3 = _csv("hypotheses_h3_bootstrap.csv")
    if h3 is not None:
        with st.expander("H3: różnica zysków ze zmiennych zewnętrznych (bootstrap blokowy)"):
            h3 = h3.assign(ML=h3["ML"].map(name), ECON=h3["ECON"].map(name),
                           grupa=h3["grupa"].map(lambda x: GROUP_NAMES.get(x, x)))
            st.dataframe(h3.round(3), width="stretch", hide_index=True)


def _research_test() -> None:
    found = {lab: pre for lab, pre in {"Specyfikacja główna": "test_main", "Specyfikacja wrażliwości": "test_sens"}.items()
             if (RESEARCH / f"{pre}_results.csv").exists()}
    if not found:
        st.info("Wyniki testu na danych z 2026 r. pojawią się tutaj po zakończeniu roku.")
        return
    spec = st.radio("Specyfikacja", list(found), horizontal=True)
    pre = found[spec]
    res = _csv(f"{pre}_results.csv")
    order = ["całość", "Q1", "Q2", "Q3", "Q4"] + [f"{m:02d}" for m in range(1, 13)]
    _research_block(res, order, pre)
    for suffix, title in (("dm", "Test Diebolda–Mariano (cztery testy potwierdzające, korekta Holma)"),
                          ("h3_bootstrap", "H3: bootstrap blokowy"),
                          ("negative_days", "Doby z ceną ujemną i pozostałe (H2)"),
                          ("intervals90_monthly", "Przedziały prognoz 90% według miesięcy (H4)")):
        t = _csv(f"{pre}_{suffix}.csv")
        if t is not None:
            with st.expander(title):
                st.dataframe(t.round(3), width="stretch", hide_index=True)


def tab_research() -> None:
    st.info("**Wyniki z pracy magisterskiej.** Ta zakładka nie aktualizuje się codziennie: pokazuje wyniki badania "
            "uzyskane na ustalonych danych i wersji kodu (znacznik `" + FREEZE_TAG + "`). Uwzględnia także modele ze "
            "zmiennymi zewnętrznymi, które nie są stosowane w prognozie bieżącej.")
    dev, test = st.tabs(["Lata 2021–2025", "Test 2026"])
    with dev:
        _research_dev()
    with test:
        _research_test()


def tab_about(status: dict) -> None:
    st.markdown("""
**Cel.** Porównanie metod ekonometrycznych (ARX, LEAR, SARIMA) i uczenia maszynowego (XGBoost, MLP) w prognozowaniu
cen energii elektrycznej na polskim rynku dnia następnego (praca magisterska, AGH).

**Jak powstaje prognoza.**
1. Codziennie o 12:00 zamyka się aukcja rynku dnia następnego, a ok. 12:45 publikowane są ceny na kolejną dobę.
2. Po publikacji wyników automat pobiera dane z ENTSO-E i wyznacza prognozę 24 cen godzinowych dla doby, której
   ceny rozstrzygną się w **następnej** aukcji (dzień po jutrze). Modele znają ceny do doby poprzedzającej dostawę.
3. Prognoza trafia do dziennika i nie jest później zmieniana. Gdy cena zostanie ustalona, prognoza jest
   porównywana z rzeczywistością w zakładce „Weryfikacja”.

**Modele.** Naiwny (D-1 / D-7) jako punkt odniesienia; ekonometryczne: ARX, LEAR, SARIMA; uczenie maszynowe:
XGBoost, MLP; średnia modeli. W prognozie bieżącej nie są stosowane modele ze zmiennymi zewnętrznymi, ponieważ
prognozy wiatru i fotowoltaiki ENTSO-E publikuje dopiero po aukcji; są one oceniane w zakładce „Wyniki badania”.

**Miary.** MAE – średni błąd bezwzględny w EUR/MWh (im mniej, tym lepiej). RMSE – błąd średniokwadratowy, silniej
karze duże pomyłki. ME – średni błąd ze znakiem (dodatni: prognoza zawyżona). rMAE – MAE w relacji do prognozy
naiwnej (poniżej 1: model lepszy od naiwnego).

**Dane.** ENTSO-E Transparency Platform; ceny godzinowe (od 1.10.2025 średnia z czterech kwadransów).
Projekt badawczy; prognozy nie stanowią porady inwestycyjnej.
""")
    if status:
        with st.expander("Status ostatniego uruchomienia"):
            st.write(f"Uruchomienie: {local_time(status.get('run_utc', ''))} | ostatnia doba ze znanymi cenami: "
                     f"{status.get('last_price_day')} | wersja kodu: {status.get('code_version')}")
            s = pd.DataFrame(status.get("models", {})).T
            s.index = [name(m) for m in s.index]
            st.dataframe(s, width="stretch")


st.title("Prognozy cen energii elektrycznej na rynku dnia następnego w Polsce")
status_, prices_, log_ = load_status(), load_prices(), load_log()
ver_ = verification(log_, prices_)
t1, t2, t3, t4 = st.tabs(["Prognoza", "Weryfikacja", "Wyniki badania", "Jak to działa"])
with t1:
    tab_forecast(status_, prices_, log_, ver_)
with t2:
    tab_verify(ver_, prices_)
with t3:
    tab_research()
with t4:
    tab_about(status_)