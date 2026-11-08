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