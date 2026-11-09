agregacja 15→60 min średnią z 4 kwadransów;
doba dostawy w czasie lokalnym;
waluta EUR, start danych 2020-01-01;
schemat walidacji (tabela z poprzedniej wiadomości);
reguła sejfu dla 2026;
transformacja asinh.
pierwsza godzina 2020 bez danych egzogenicznych (start modeli od 2020-01-02), wiatr morski wyłączony, PV od 2020-04-10
EDA i testy na danych do 31.12.2025 (2026 tylko opisowo) oraz transformacja asinh z parametrami z danych rozwojowych
Granice reżimów wiążemy z obserwowalnymi zdarzeniami: stabilny 2020-01-01 → 2021-08-31, kryzys 2021-09-01 → 2023-06-10, OZE z cenami ujemnymi od 2023-06-11. Ostatnią granicę dałoby się uzasadnić statystycznie testem punktów zmiany (opcjonalnie w 4.2).
Okno kalibracji 364 dni jako główne. Przy oknie 2-letnim reżim stabilny (do VIII 2021) wypadał całkowicie z oceny, a to jedno z pytań badawczych. Przy 364 dniach ocena zaczyna się 2021-01-02 i obejmuje 8 miesięcy stabilności. Okno 728 dni będzie analizą wrażliwości od 2022.
Parametry asinh liczymy w każdym oknie osobno. Parametry z całych danych (a=94,6, b=48,3) przeciekałyby informację z przyszłości.
Doba = 24 wartości (wiersz na dobę, kolumny to godziny lokalne). Zdublowaną godzinę jesienią uśredniamy, brakującą wiosną interpolujemy.
Cechy budujemy w krokach 6–7, osobno dla każdej klasy modeli, bo ich format się różni. Krok 5 to panele danych, protokół i benchmarki.
prognozując dobę D, widzimy ceny do doby D−1 i prognozy zapotrzebowania/OZE na dobę D
asinh z parametrami (mediana, 1,4826·MAD) liczonymi osobno w każdym oknie kalibracji;
powrót do EUR przez odwrotną transformację (to w przybliżeniu mediana, nie średnia);
LEAR używa prognoz egzogenicznych tylko z doby D (24 godziny), co jest uproszczeniem względem wersji z literatury;
wiersze z brakującym PV (przed 2020-04-10) są pomijane przy estymacji, więc wczesny 2021 ma mniej obserwacji;
jeśli okno ma za mało wierszy (< liczba cech + 10), doba jest pomijana i raport liczy tylko doby wspólne dla wszystkich modeli.
przycinanie cech do ±5σ i predykcji do zakresu okna ± 1 w skali asinh (identyczne dla wszystkich modeli) oraz równoległe liczenie prognoz
wyniki dla H3 z kroku 6a są wstępne do czasu ponownego liczenia
To wszystko dotyczy danych do 2025, więc sejf 2026 jest nietknięty.
Poprawkę wprowadziliśmy po obejrzeniu wyników w okresie rozwojowym 2024–2025, ale stosujemy ją identycznie do wszystkich modeli. Zapisz to w decisions.md.
Prognozy egzogeniczne z ENTSO-E. Z tego, co wiem, wiatr i PV na dobę następną mogą być publikowane po zamknięciu fixingu (ok. 12:00), dokładnie godzin nie sprawdzałem. W pracy traktujemy je jako przybliżenie prognoz dostępnych uczestnikom rynku z własnych źródeł. Wpisz to jako założenie w 3.1 i zweryfikuj godziny publikacji w dokumentacji ENTSO-E, bo rMAE poniżej 0,7 jest lepsze od typowych wyników z literatury.
definicje sarima i sarimax (dummies dnia tygodnia w obu), estymację raz na 28 dób z codzienną aktualizacją stanu, VAR z dummies bez zmiennych egzogenicznych oraz uzupełnianie braków PV średnią okna
VAR(1) wybrany spośród p = 1–3 na podstawie MAE i testu DM w okresie 2021–2025 (rok 2026 nietknięty),
ARX, LEAR, VAR i XGBoost uczone codziennie od nowa, SARIMA/SARIMAX raz na 28 dób z codzienną aktualizacją stanu filtra,
XGBoost z domyślnymi hiperparametrami (strojenie w kroku 8).
wariant awaryjny SARIMAX (kolejność prób: podstawowy, Powell, prostszy rząd (1,0,1)(1,0,1,24)) i w którym bloku został użyty,
specyfikację MLP: jeden model na wszystkie godziny, one-hot godzin i dni, dwie warstwy (64, 32), średnia z 3 inicjalizacji, early stopping na losowym podzbiorze wierszy treningowych,
ograniczenie: SARIMA(X) estymowana co 28 dób, pozostałe modele codziennie,
że wnioski z 7a dotyczące H1–H3 są wstępne (niestrojone hiperparametry, okres rozwojowy 2021–2025, rok 2026 nietknięty).
SARIMAX: blok 2022-10-07 z ponownej estymacji,
reguła raportowania: ML porównujemy z najlepszym modelem klasy ekonometrycznej (LEAR/SARIMAX), a nie tylko z SARIMA,
Prophet z codzienną reestymacją i bez sezonowości rocznej (okno 364 dni),
wnioski z H1–H3 są wstępne (niestrojone modele, okres rozwojowy 2021–2025, rok 2026 nietknięty).
hipoteza o roli prognozy zapotrzebowania, oznaczona jako niezweryfikowana.
W SARIMA po dodaniu cechy świątecznej MAE w reżimie stabilnym wzrosło o 0,52 EUR/MWh, z czego ok. 86% pochodzi z jednego 28-dobowego bloku estymacji (26.03–22.04.2021). Różnica nie jest skupiona w dobach świątecznych, tylko rozłożona na większość dób bloku, co wskazuje na zestaw parametrów estymowany raz na 28 dób. Wynik raportujemy bez zmian, a wniosek o niższej jakości SARIMA względem benchmarku w reżimie stabilnym jest odporny na tę różnicę (rMAE 1,12 przed i 1,21 po)
Walidacja to wyłącznie 2025. Rok 2026 nie jest używany.
Każdy wariant (z i bez egzogenicznych) stroimy osobno, żeby efekt egzogenicznych (H3) nie był zawyżony. Klasę ekonometryczną stroimy tak samo jak ML (długość okna dla ARX/LEAR, rząd dla SARIMA), żeby nie faworyzować ML.
Wybór: najprostsza konfiguracja (pierwsza w siatce) w granicach 1% MAE od najlepszej, żeby nie dopasowywać się do szumu jednego roku.
Wynik trzeba czytać ostrożnie: 2025 jest jednym rokiem, więc zysk poniżej ok. 1% to szum.
deklarujemy już teraz dwie specyfikacje:
główna: konfiguracje z config/model_params.yaml (okno 728 tam, gdzie wybrane),
wrażliwości: wszystkie modele z oknem 364 (prognozy z tabeli forecasts_v2, już policzone).
dwie specyfikacje (główna i wrażliwości, okno 364) i że obie policzymy na teście 2026,
tabelę z sekcji 2 jako obserwację z okresu rozwojowego, razem z uwagą, że wniosek jest wstępny, bo ML jeszcze nie przeliczono,
LEAR vs ARX: różne okna (728 vs 364), więc porównanie łączy estymator i okno,
awaria WinError 6 i ponowienie przebiegu XGBoost/MLP,
że różnice między blokami SARIMA(X) w strojeniu i w pełnym przebiegu (np. 15,63 vs 15,65 dla 2025) wynikają z innego początku bloków 28-dobowych.
H1 (ML lepsze od klasycznych): potwierdzona zależnie od porównania.
Względem SARIMA(X): tak, w każdym reżimie i w obu specyfikacjach. xgb_x wygrywa z sarimax (DM −5,28, MAE −9,3%) i mlp_x także (DM −3,80). Bez zmiennych egzogenicznych xgb i mlp biją SARIMA (DM −6,89 i −8,67). W głównej specyfikacji przewaga jest o ok. 3,6 punktu proc. większa niż w wersji z oknem 364, bo okno 728 pogorszyło SARIMAX.
Względem najlepszego modelu ekonometrycznego (LEAR): częściowo. Z egzogenicznymi tylko xgb_x bije lear_x (−5,0%, p=0,0006), a mlp_x nie (−2,0%, p=0,16). Bez egzogenicznych wygrywa LEAR: xgb jest od niego o 5,8% gorszy (DM +4,93), a mlp o 2,9% (DM +3,30).
Przewaga xgb_x nad lear_x jest skupiona w latach 2023–2024. MAE różni się w kolejnych latach o +1,2% (2021), +0,3% (2022), −16,1% (2023), −11,3% (2024), +0,1% (2025). Przy oknie 364 wynosi to +2,4%, −4,4%, −14,3%, −4,2% i −0,5%. Rok 2023 jest więc silny w obu wersjach, 2024 zależy od okna, a w 2021 i 2025 przewagi nie ma. Dlaczego akurat 2023, nie umiem wskazać. W tym roku pojawiły się pierwsze ceny ujemne, ale to tylko podejrzenie.
metodę przedziałów (empiryczne kwantyle błędów z 90 dób, poziomy 80% i 90%, minimum 60 dób historii, więc pierwsze przedziały od początku marca 2021) i ograniczenie, że natywne przedziały parametryczne ARX i SARIMAX są omówione opisowo, a nie testowane
szystko dotyczy okresu rozwojowego, a rok 2026 jest nietknięty. Poprawka Holma nie obejmuje całej tabeli (64 porównania), ale kluczowe wyniki mają p < 0,001. Dla RQ4 możesz dodać tabelę: SARIMAX ma ok. 16 parametrów (wynika ze specyfikacji, nie z biblioteki) i MAE 17,20, ARX_X ok. 336 (14 na godzinę), LEAR_X ok. 1,2 tys. niezerowych (mediana 51 na godzinę z końca 2025), XGB_X 600 drzew głębokości 7 i MAE 15,20 (−11,6% wobec SARIMAX). Interpretacja: współczynniki wprost (ARX, SARIMAX), współczynniki mylące przy współliniowości (LEAR_X), SHAP post hoc (XGB_X).
H1: potwierdzona względem SARIMA (odporna), a względem LEAR_X tylko w 2023. Bez zmiennych egzogenicznych LEAR jest istotnie lepszy od XGBoost w obu specyfikacjach w 2025 i reżimie OZE.
H2: niepotwierdzona dla kryzysu, a dla cen ujemnych potwierdzona opisowo. Do sprawdzenia zostaje niespójność z latami: gdyby ceny ujemne były przyczyną, przewaga XGB_X nad LEAR_X powinna być największa w 2025 (najwięcej ujemnych godzin), a rocznie jest tam zerowa. Skrypt poniżej rozstrzyga, czy przewaga na dobach z ceną ujemną występuje w każdym roku.
H3: potwierdzona względem ARX i LEAR (obie specyfikacje, całość i reżimy z wyjątkiem stabilnego), a względem SARIMAX niepotwierdzona.
3. Dobra z ujemną ceną według roku (H2)

DM xgb_x vs lear_x, różnica MAE w % (ujemna = XGB_X lepszy). W nawiasach okno 364 dni:

Rok	doby z ujemną ceną	pozostałe doby
2023	−23,8% (−18,6%), tylko 5 dób	−11,5% (−5,5%)
2024	−13,6% (−12,1%), 49 dób, p=0,048 (<0,001)	−10,9% (−2,4%)
2025	−12,2% (−11,4%), 70 dób, p=0,006 (0,005)	+4,1% (+2,9%), 295 dób
To wyjaśnia zagadkę z poprzedniej wiadomości. W 2025 roczna różnica jest zerowa (−0,1%), bo przewaga na 70 dobach z ujemną ceną (−12%) kasuje się z gorszym wynikiem XGB_X na pozostałych 295 dobach (+4%).
Przewaga na dobach z ujemną ceną jest odporna: ten sam kierunek i skala (−11 do −14%) w 2024 i 2025, w obu specyfikacjach okna, przy p ≤ 0,05. Na pozostałych dobach jest niestabilna: zależy od okna w 2023–2024 i odwraca się w 2025.
Zastrzeżenia, które trzeba zapisać w tekście:
Doby wybieramy po zrealizowanej cenie, więc to warunkowanie na wyniku.
Podział według roku powstał po obejrzeniu wyników zbiorczych, więc analiza jest post hoc i p nie są korygowane.
Doba z ujemną ceną to doba z bardzo niskim obciążeniem rezydualnym, więc nie da się odróżnić efektu znaku ceny od efektu niskiego obciążenia.
Wersja H2, którą wyniki wspierają: przewaga ML nad klasą ekonometryczną występuje na dobach z ujemnymi cenami, a nie w kryzysie. To pasuje do wyniku z 8e (LEAR_NL zamyka 44% luki w reżimie OZE), ale mechanizmu to nie dowodzi.
1. Dlaczego nie teraz
Błędy pytest (34 passed, 2 errors). Pytest bierze funkcję test_groups z report.py za test (nazwa zaczyna się od test_) i zgłasza brak fixture days. Dodatkowo run_test.py pasuje do wzorca *_test.py i też jest zbierany. Nie ma to wpływu na wyniki, ale testy muszą być czyste przed tagiem.
Zakres zamrożenia był za szeroki. freeze.py porównuje z tagiem cały src/ i config/. Kroki 9–10 (Streamlit, Actions, codzienne prognozy) prawdopodobnie dodadzą pliki w src/, a wtedy test 2026 odmówi uruchomienia z komunikatem „kod różni się od taga”. Zawężam zamrożenie do ścieżek, które wpływają na wyniki. Zmiany w src/data zabezpiecza odcisk danych.
Protokół nie ma jeszcze oczekiwania dla dób z ujemną ceną. Dopisujemy je przed tagiem, bo dopiero z wyników poniżej wynika, co tam widzieliśmy (sekcja 3).
2. Próba generalna

Obie ścieżki działają: 15 modeli plus lear_nl w main, w sens liczą się tylko modele ze zmienioną konfiguracją (reszta współdzieli prognozy), raport się generuje, a „DM: za mało dób” przy 14 dobach to oczekiwane zachowanie. Wyniki z 14 dób grudnia nic nie mówią o jakości modeli. Pomiar czasu: pojedynczy blok SARIMA zajął 192 s (main) i 95 s (sens), SARIMAX 224 s i 108 s. Z czasów pełnych przebiegów (SARIMA 47 min dla 66 bloków, SARIMAX 118 min) szacuję pełny rok 2026 (14 bloków) na ok. 10 i ok. 25 min dla SARIMA i SARIMAX, a całość main na około godzinę. To tylko szacunek.