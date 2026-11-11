# Rejestr decyzji i założeń (część praktyczna)

Dokument porządkuje decyzje podejmowane w krokach 1–10. Zasada naczelna: rok 2026 jest „sejfem”; żadna decyzja
o modelach, parametrach ani hipotezach nie opiera się na danych z 2026. Wszystkie analizy i strojenie dotyczą danych
do 31.12.2025. Wnioski z okresu rozwojowego (2021–2025) są wstępne do czasu testu na 2026.

*Uwaga redakcyjna (11.11.2026): poprzednia wersja pliku była zbiorem nieuporządkowanych fragmentów, w tym wklejonych
poleceń redakcyjnych. Treść pogrupowano tematycznie i oczyszczono bez zmiany ustaleń merytorycznych.*

## 1. Dane i przygotowanie (kroki 1–3)
- Agregacja 15→60 min jako średnia z 4 kwadransów (ceny 15-minutowe od 1.10.2025); niepełna godzina → brak wartości.
- Doba dostawy w czasie lokalnym (Europe/Warsaw). Doba = 24 wartości (wiersz na dobę, kolumny to godziny lokalne);
  zdublowaną godzinę jesienią uśredniamy, brakującą wiosną interpolujemy.
- Waluta EUR, start danych 2020-01-01 (lata 2017–2019 w PLN wykluczone).
- Pierwsza godzina 2020 bez danych egzogenicznych (start modeli od 2020-01-02); wiatr morski wyłączony; PV od 2020-04-10.
- EDA i testy na danych do 31.12.2025 (2026 tylko opisowo).
- Prognozy egzogeniczne z ENTSO-E (zapotrzebowanie, wiatr lądowy, PV): zakładamy znajomość prognozy na dobę D przy
  prognozowaniu doby D. Wiatr i PV na dobę następną mogą być publikowane po zamknięciu fixingu (ok. 12:00); w pracy
  traktujemy je jako przybliżenie prognoz dostępnych uczestnikom rynku z własnych źródeł. Obserwacja z 11.11.2026
  (godz. 15:00 lokalnie): ceny i prognoza obciążenia do 12.11, prognozy wiatru i PV tylko do 11.11. Godziny publikacji
  w trakcie pomiaru (exog_log.txt); opisać w pracy jako ograniczenie (zob. sekcja 12).

## 2. Reżimy, okno i schemat walidacji (kroki 4–5)
- Granice reżimów wiążemy z obserwowalnymi zdarzeniami: stabilny 2020-01-01 → 2021-08-31, kryzys 2021-09-01 →
  2023-06-10 (wzrost cen IX 2021), OZE z cenami ujemnymi od 2023-06-11 (pierwsza ujemna cena). Ostatnią granicę
  dałoby się uzasadnić statystycznie testem punktów zmiany (opcjonalnie w rozdziale 4).
- Okno kalibracji 364 dni jako główne w okresie rozwojowym. Przy oknie 2-letnim reżim stabilny (do VIII 2021) wypadałby
  z oceny, a to jedno z pytań badawczych. Przy 364 dniach ocena zaczyna się 2021-01-02 i obejmuje 8 miesięcy
  stabilności. Okno 728 dni jest analizą wrażliwości (od 2022), a po strojeniu wybrano je dla części modeli.
- Zbiór informacji: prognozując dobę D, widzimy ceny do doby D−1 i prognozy zapotrzebowania/OZE na dobę D.
- Transformacja asinh z parametrami (mediana, 1,4826·MAD) liczonymi osobno w każdym oknie kalibracji; parametry z całych
  danych (a = 94,6, b = 48,3) przeciekałyby informację z przyszłości. Powrót do EUR przez odwrotną transformację
  (w przybliżeniu mediana, nie średnia).
- Cechy budujemy osobno dla każdej klasy modeli, bo ich format się różni. Krok 5: panele danych, protokół, benchmarki.
- Okres analizy reżimów: 2021-01-01 do 2025-12-31 (1826 dób); reżimy: 243 / 648 / 935 dób.

## 3. Modele liniowe i ekonometryczne (kroki 6a–6b)
- LEAR używa prognoz egzogenicznych tylko z doby D (24 godziny), co jest uproszczeniem względem wersji z literatury.
- Wiersze z brakującym PV (przed 2020-04-10) są pomijane przy estymacji, więc wczesny 2021 ma mniej obserwacji.
- Jeśli okno ma za mało wierszy (< liczba cech + 10), doba jest pomijana, a raport liczy tylko doby wspólne dla
  wszystkich modeli.
- Przycinanie cech do ±5σ i predykcji do zakresu okna ±1 w skali asinh (identyczne dla wszystkich modeli) oraz równoległe
  liczenie prognoz. Poprawkę wprowadzono po obejrzeniu wyników okresu rozwojowego 2024–2025 (eksplozje numeryczne
  arx_x i lear_x), ale stosujemy ją identycznie do wszystkich modeli.
- SARIMA/SARIMAX: dummies dnia tygodnia w obu, estymacja raz na 28 dób z codzienną aktualizacją stanu filtra; braki PV w
  SARIMAX uzupełniane średnią okna. Wariant awaryjny SARIMAX przy błędzie numerycznym (kolejność prób: podstawowy,
  Powell, prostszy rząd (1,0,1)(1,0,1,24)); zapisywany jest blok, w którym go użyto (np. blok 2022-10-07).
- VAR z dummies, bez zmiennych egzogenicznych. VAR(1) wybrany spośród p = 1–3 na podstawie MAE i testu DM w okresie
  2021–2025 (rok 2026 nietknięty).
- ARX, LEAR, VAR i XGBoost uczone codziennie od nowa; SARIMA/SARIMAX raz na 28 dób (ograniczenie).
- Reguła raportowania: ML porównujemy z najlepszym modelem klasy ekonometrycznej (LEAR/SARIMAX), a nie tylko z SARIMA.
- Wyniki dla H3 z kroku 6a były wstępne do czasu ponownego liczenia po strojeniu.

## 4. Modele uczenia maszynowego (kroki 7a–7b)
- XGBoost: początkowo z domyślnymi hiperparametrami (strojenie w kroku 8).
- MLP: jeden model na wszystkie godziny, one-hot godzin i dni, dwie warstwy (64, 32), średnia z 3 inicjalizacji,
  early stopping na losowym podzbiorze wierszy treningowych.
- Prophet: codzienna reestymacja, bez sezonowości rocznej (okno 364 dni).
- Wnioski z 7a dotyczące H1–H3 wstępne (niestrojone hiperparametry, okres rozwojowy).
- Hipoteza o roli prognozy zapotrzebowania w wynikach pozostaje niezweryfikowana.
- Awaria WinError 6 (BrokenProcessPool) przy XGBoost/MLP: ponowienie przebiegu.

## 5. Cecha świąt (krok 7c)
- Cecha `hol` (święta ustawowe w Polsce) dodana do modeli; stare wyniki zachowane w tabeli forecasts_v1.
- W SARIMA po dodaniu cechy MAE w reżimie stabilnym wzrosło o 0,52 EUR/MWh, z czego ok. 86% pochodzi z jednego
  28-dobowego bloku estymacji (26.03–22.04.2021). Różnica nie jest skupiona w dobach świątecznych, tylko rozłożona na
  większość dób bloku, co wskazuje na zestaw parametrów estymowany raz na 28 dób. Wynik raportujemy bez zmian; wniosek o
  niższej jakości SARIMA względem benchmarku w reżimie stabilnym jest odporny na tę różnicę (rMAE 1,12 przed i 1,21 po).

## 6. Strojenie hiperparametrów i specyfikacje testu (krok 8a–8b)
- Walidacja to wyłącznie rok 2025; rok 2026 nie jest używany.
- Każdy wariant (z i bez egzogenicznych) stroimy osobno, żeby efekt egzogenicznych (H3) nie był zawyżony. Klasę
  ekonometryczną stroimy tak samo jak ML (długość okna dla ARX/LEAR, rząd dla SARIMA), żeby nie faworyzować ML.
- Reguła wyboru: konfiguracja domyślna (pierwsza w siatce), chyba że najlepsza poprawia MAE o więcej niż 1%
  (żeby nie dopasowywać się do szumu jednego roku; zysk poniżej ok. 1% to szum).
- Dwie specyfikacje testu: główna (config/model_params.yaml, okno 728 tam, gdzie wybrane) i wrażliwości (wszystkie modele
  z oknem 364; prognozy z tabeli forecasts_v2, już policzone). Obie liczymy na teście 2026.
- Ograniczenia interpretacji: LEAR vs ARX mają różne okna (728 vs 364), więc porównanie łączy estymator i okno.
  Różnice między blokami SARIMA(X) w strojeniu i w pełnym przebiegu (np. 15,63 vs 15,65 dla 2025) wynikają z innego
  początku bloków 28-dobowych.

## 7. Przedziały predykcji i interpretowalność (krok 8c)
- Przedziały: empiryczne kwantyle błędów z 90 dób, poziomy 80% i 90%, minimum 60 dób historii (pierwsze przedziały od
  początku marca 2021). Natywne przedziały parametryczne ARX i SARIMAX omawiane opisowo, nie testowane.
- RQ4 (interpretowalność): SARIMAX ma ok. 16 parametrów i MAE 17,20; ARX_X ok. 336 (14 na godzinę); LEAR_X ok. 1,2 tys.
  niezerowych (mediana 51 na godzinę z końca 2025); XGB_X 600 drzew głębokości 7 i MAE 15,20 (−11,6% wobec SARIMAX).
  Interpretacja: współczynniki wprost (ARX, SARIMAX), współczynniki mylące przy współliniowości (LEAR_X), SHAP post hoc
  (XGB_X). Poprawka Holma nie obejmuje całej tabeli (64 porównania), ale kluczowe wyniki mają p < 0,001.

## 8. Wyniki okresu rozwojowego 2021–2025 (wstępne; rok 2026 nietknięty)
- **H1.** Względem SARIMA(X) potwierdzona w każdym reżimie i w obu specyfikacjach: xgb_x wygrywa z sarimax (DM −5,28,
  MAE −9,3%), mlp_x także (DM −3,80); bez egzogenicznych xgb i mlp biją SARIMA (DM −6,89 i −8,67). W głównej
  specyfikacji przewaga jest o ok. 3,6 p.p. większa niż przy oknie 364, bo okno 728 pogorszyło SARIMAX. Względem najlepszego
  modelu ekonometrycznego (LEAR) częściowo: z egzogenicznymi tylko xgb_x bije lear_x (−5,0%, p = 0,0006), mlp_x nie
  (−2,0%, p = 0,16); bez egzogenicznych wygrywa LEAR (xgb o 5,8% gorszy, DM +4,93; mlp o 2,9%, DM +3,30).
- **Rozkład w czasie.** Przewaga xgb_x nad lear_x różni się MAE w kolejnych latach o +1,2% (2021), +0,3% (2022), −16,1%
  (2023), −11,3% (2024), +0,1% (2025); przy oknie 364: +2,4%, −4,4%, −14,3%, −4,2%, −0,5%. Rok 2023 jest silny w obu
  wersjach, 2024 zależy od okna, w 2021 i 2025 przewagi nie ma. Przyczyny 2023 nie ustalono (w tym roku pojawiły się
  pierwsze ceny ujemne; to tylko podejrzenie).
- **H2.** Niepotwierdzona dla kryzysu; dla cen ujemnych potwierdzona opisowo (zob. sekcja 9).
- **H3.** Potwierdzona względem ARX i LEAR (obie specyfikacje, całość i reżimy z wyjątkiem stabilnego); względem SARIMAX
  niepotwierdzona.
- LEAR_NL zamyka 44% luki w reżimie OZE (model eksploracyjny; mechanizm nie jest dowiedziony).

## 9. Doby z ujemną ceną (krok 8d–8e)
DM xgb_x vs lear_x, różnica MAE w % (ujemna = XGB_X lepszy); w nawiasach okno 364 dni:

| Rok | doby z ujemną ceną | pozostałe doby |
|---|---|---|
| 2023 | −23,8% (−18,6%), tylko 5 dób | −11,5% (−5,5%) |
| 2024 | −13,6% (−12,1%), 49 dób, p = 0,048 (<0,001) | −10,9% (−2,4%) |
| 2025 | −12,2% (−11,4%), 70 dób, p = 0,006 (0,005) | +4,1% (+2,9%), 295 dób |

- W 2025 roczna różnica jest zerowa (−0,1%), bo przewaga na 70 dobach z ujemną ceną kasuje się z gorszym wynikiem XGB_X
  na pozostałych 295 dobach.
- Przewaga na dobach z ujemną ceną jest odporna (ten sam kierunek i skala, −11 do −14%, w 2024 i 2025, w obu
  specyfikacjach, p ≤ 0,05); na pozostałych dobach niestabilna (zależy od okna w 2023–2024, odwraca się w 2025).
- Zastrzeżenia do zapisania w tekście: (a) doby wybieramy po zrealizowanej cenie, więc to warunkowanie na wyniku;
  (b) podział według roku powstał po obejrzeniu wyników zbiorczych, więc analiza jest post hoc, a p nie są korygowane;
  (c) doba z ujemną ceną to doba z bardzo niskim obciążeniem rezydualnym, więc nie da się odróżnić efektu znaku ceny od
  efektu niskiego obciążenia.
- Wersja H2, którą wyniki wspierają: przewaga ML nad klasą ekonometryczną występuje na dobach z ujemnymi cenami, a nie w
  kryzysie.

## 10. Harness testu, zamrożenie i próba generalna (krok 8f)
- Błędy pytest (34 passed, 2 errors) wynikały z nazwy `test_groups` w report.py (pytest traktował ją jako test) i z
  wzorca nazwy run_test.py; zmieniono na `holdout_groups` i dodano pytest.ini (`testpaths = tests`).
- Zakres zamrożenia zawężono do ścieżek wpływających na wyniki: src/models, src/features, src/evaluation, src/utils,
  config/settings.yaml, config/model_params.yaml. Poza zamrożeniem: src/data i src/analysis (dane zabezpiecza odcisk),
  aplikacja, automatyzacja i testy. **Nowe skrypty (np. scripts/thesis_outputs.py) umieszczamy poza zamrożonymi
  ścieżkami**, bo nowy plik w src/evaluation zablokowałby `run_test --confirm-test`.
- Oczekiwanie dla dób z ujemną ceną dopisano do protokołu (punkt 7e), bo wynika z wyników sekcji 9.
- Próba generalna (14 dób grudnia 2025): obie ścieżki działają (15 modeli plus lear_nl w main; w sens tylko modele ze
  zmienioną konfiguracją), raport się generuje; „DM: za mało dób” przy 14 dobach jest oczekiwane. Wyniki z 14 dób nic nie
  mówią o jakości modeli. Czasy: blok SARIMA 192 s (main) i 95 s (sens), SARIMAX 224 s i 108 s; szacunek pełnego 2026:
  ok. 10 min (SARIMA), ok. 25 min (SARIMAX), całość main ok. 1 godzina (szacunek).
- Mechanizm: tag `pre-test-2026` (commit d3bbfa7), odcisk danych docs/frozen_data.json (powstaje w styczniu 2027
  poleceniem `freeze --snapshot`), `run_test --confirm-test` odmawia uruchomienia, gdy stan zamrożony się różni.

## 11. Warstwa operacyjna (kroki 9–10, 11.11.2026)
- Aplikacja Streamlit (Community Cloud) czyta wyłącznie app_data/*.csv i status.json; workflow GitHub Actions
  `daily-forecast` (sekret ENTSOE_API_KEY) uruchamia ingest, build_hourly, run_daily i commituje app_data.
- Polityka modeli egzogenicznych: opcja B – dwa biegi dziennie (08:40 UTC: modele bez egzogenicznych; 10:40 UTC:
  `--only-missing`, modele _x i sarimax, jeśli prognozy dostępne). Pierwsza prognoza (doba, model) obowiązuje.
  Wstępna obserwacja wskazuje, że prognozy wiatru i PV mogą pojawiać się po aukcji, więc bieg 10:40 może nic nie policzyć;
  kod jest na to odporny. Decyzja o modelach _x w aplikacji po pomiarach (exog_log.txt).
- Rozdział warstw: badanie (zamrożone; zakładka „Wyniki badania”: Rozwój 2021–2025 i Test 2026) jest odrębne od
  aplikacji na żywo (Prognoza, Skuteczność na żywo, Dane historyczne). Wyniki na żywo nie służą do weryfikacji hipotez, a
  dni listopad–grudzień 2026 leżą w roku testowym: nie wolno zmieniać modeli na podstawie ich skuteczności.
- Horyzont prognoz: doba następna (D+1), zgodnie z rynkiem dnia następnego.
- W prognozie operacyjnej SARIMA(X) estymowany codziennie (w badaniu raz na 28 dób).
- Schemat testu 2026: prognoza krocząca (dla każdej doby ponowna estymacja na oknie), nie jednorazowy trening do 2025.
- Skrypt tabel i wykresów do rozdziału z wynikami: scripts/thesis_outputs.py (tylko czyta prognozy).

## 12. Ograniczenia i założenia do opisania w pracy
- Założenie o znajomości prognoz wiatru i PV dla doby D w chwili prognozowania (sekcje 1 i 11).
- Strojenie na jednym roku (2025); wynik strojenia zależy od roku (np. sarimax +12,4% w 2022, −3,0% w 2025), stąd dwie
  specyfikacje testu.
- Przedziały empiryczne niedoszacowane (niedopokrycie).
- Reżimy wyznaczone z góry (zdarzenia rynkowe), nie estymowane.
- Zmiana rozdzielczości cen na 15 minut od 1.10.2025 (uśrednienie do godziny); brak morskiej energii wiatrowej.
- LEAR_NL eksploracyjny; nie wchodzi do testów potwierdzających.
- **Do ujednolicenia:** granice reżimów w kodzie (stabilny do VIII 2021, kryzys IX 2021–VI 2023, OZE od VI 2023) różnią się od
  podziału w rozdziale 1 (stabilność 2021, kryzys 2022, OZE 2023–2026); w tekście przyjąć granice z config/settings.yaml
  albo świadomie zmienić konfigurację przed styczniem i przeliczyć analizy.