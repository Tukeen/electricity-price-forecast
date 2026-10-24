
from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import truststore
from dotenv import load_dotenv


# Używaj systemowego magazynu certyfikatów Windows.
# Weryfikacja SSL pozostaje włączona.
truststore.inject_into_ssl()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "entsoe_prices"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("ENTSOE_API_KEY")
API_URL = "https://web-api.tp.entsoe.eu/api"
POLAND_EIC = "10YPL-AREA-----S"

# Początek zakresu badania. Później możemy go zmienić,
# jeśli analiza kompletności danych wskaże inną datę.
START_DATE = date(2015, 1, 1)

# Pobieramy tylko zakończone dni: data końcowa jest wyłączna.
END_DATE = datetime.now(timezone.utc).date()

OUTPUT_CSV = PROCESSED_DIR / "poland_day_ahead_prices.csv"


def local_name(tag: str) -> str:
    """Zwraca nazwę elementu XML bez przestrzeni nazw."""
    return tag.rsplit("}", 1)[-1]


def child_text(element: ET.Element, name: str) -> str | None:
    """Pobiera tekst bezpośredniego elementu potomnego."""
    for child in element:
        if local_name(child.tag) == name:
            return child.text
    return None


def parse_price_document(xml_content):
    """
    Parsuje dokument ENTSO-E A44.
    Dla krzywej A03 odtwarza pominięte pozycje,
    przenosząc cenę z poprzedniego przedziału.
    """
    root = ET.fromstring(xml_content)
    records = []

    for time_series in root.iter():
        if local_name(time_series.tag) != "TimeSeries":
            continue

        curve_type = child_text(time_series, "curveType")
        currency = child_text(time_series, "currency_Unit.name")
        unit = child_text(time_series, "price_Measure_Unit.name")

        for period in time_series.iter():
            if local_name(period.tag) != "Period":
                continue

            interval_start = None
            interval_end = None
            resolution = None
            points = []

            for element in period.iter():
                name = local_name(element.tag)

                if name == "timeInterval":
                    for child in element:
                        child_name = local_name(child.tag)
                        if child_name == "start":
                            interval_start = child.text
                        elif child_name == "end":
                            interval_end = child.text

                elif name == "resolution":
                    resolution = element.text

                elif name == "Point":
                    position = None
                    price = None

                    for child in element:
                        child_name = local_name(child.tag)
                        if child_name == "position":
                            position = int(child.text)
                        elif child_name == "price.amount":
                            price = float(child.text)

                    if position is not None and price is not None:
                        points.append((position, price))

            if not interval_start or not interval_end or not resolution:
                continue

            start = datetime.fromisoformat(
                interval_start.replace("Z", "+00:00")
            )
            end = datetime.fromisoformat(
                interval_end.replace("Z", "+00:00")
            )

            # Obsługujemy rozdzielczości PT15M, PT30M i PT60M.
            if resolution == "PT15M":
                step = timedelta(minutes=15)
            elif resolution == "PT30M":
                step = timedelta(minutes=30)
            elif resolution == "PT60M":
                step = timedelta(hours=1)
            else:
                print(f"Nieobsługiwana rozdzielczość: {resolution}")
                continue

            number_of_positions = int((end - start) / step)
            price_by_position = dict(points)
            previous_price = None

            for position in range(1, number_of_positions + 1):
                if position in price_by_position:
                    previous_price = price_by_position[position]

                elif curve_type == "A03" and previous_price is not None:
                    # W krzywej A03 pominięta pozycja
                    # zachowuje poprzednią cenę.
                    price_by_position[position] = previous_price

                else:
                    # Nie uzupełniamy braków w innych typach krzywych.
                    continue

                timestamp = start + (position - 1) * step

                records.append({
                    "timestamp_utc": timestamp,
                    "price_eur_mwh": price_by_position[position],
                    "resolution": resolution,
                    "currency": currency,
                    "unit": unit,
                    "curve_type": curve_type,
                })

    return records


def fetch_year(session: requests.Session, year: int) -> Path | None:
    """Pobiera dane dla roku, z możliwością pominięcia już pobranego pliku."""
    raw_file = RAW_DIR / f"prices_{year}.xml"

    if raw_file.exists():
        print(f"[{year}] Plik XML już istnieje — pomijam pobieranie.")
        return raw_file

    period_start = date(year, 1, 1)
    period_end = min(date(year + 1, 1, 1), END_DATE)

    if period_start >= period_end:
        return None

    params = {
        "documentType": "A44",
        "in_Domain": POLAND_EIC,
        "out_Domain": POLAND_EIC,
        "periodStart": period_start.strftime("%Y%m%d") + "0000",
        "periodEnd": period_end.strftime("%Y%m%d") + "0000",
    }

    # Token przesyłamy nagłówkiem, aby nie umieszczać go
    # w adresie URL ani w typowych komunikatach błędów.
    headers = {"SECURITY_TOKEN": API_KEY}

    for attempt in range(1, 4):
        try:
            response = session.get(
                API_URL,
                params=params,
                headers=headers,
                timeout=120,
            )

            if response.status_code != 200:
                print(f"[{year}] Błąd HTTP: {response.status_code}")
                if attempt < 3:
                    time.sleep(3 * attempt)
                    continue
                return None

            root = ET.fromstring(response.content)

            if local_name(root.tag) == "Acknowledgement_MarketDocument":
                message = "API zwróciło komunikat zamiast danych cenowych."
                for element in root.iter():
                    if local_name(element.tag) == "text" and element.text:
                        message = element.text
                        break
                print(f"[{year}] {message}")
                return None

            if local_name(root.tag) != "Publication_MarketDocument":
                print(f"[{year}] Nieoczekiwany typ dokumentu XML.")
                return None

            raw_file.write_bytes(response.content)
            print(f"[{year}] Zapisano surowe dane: {raw_file.name}")
            return raw_file

        except (requests.RequestException, ET.ParseError, OSError) as exc:
            # Nie wypisujemy pełnego wyjątku requests, ponieważ
            # może on zawierać szczegóły zapytania.
            print(f"[{year}] Próba {attempt}/3 nie powiodła się: "
                  f"{type(exc).__name__}")
            if attempt < 3:
                time.sleep(3 * attempt)

    return None


def load_all_raw_data() -> pd.DataFrame:
    """Przetwarza wszystkie zachowane dokumenty XML."""
    records = []

    for raw_file in sorted(RAW_DIR.glob("prices_*.xml")):
        print(f"Przetwarzanie {raw_file.name}...")
        xml_content = raw_file.read_bytes()
        records.extend(parse_price_document(xml_content))

    columns = [
        "timestamp_utc",
        "price_eur_mwh",
        "resolution",
        "currency",
        "unit",
    ]
    df = pd.DataFrame(records, columns=columns)

    if df.empty:
        return df

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], utc=True
    )
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    # Nie usuwamy automatycznie duplikatów: najpierw je raportujemy.
    duplicate_count = df.duplicated(
        subset=["timestamp_utc"], keep=False
    ).sum()

    if duplicate_count:
        print(
            f"UWAGA: {duplicate_count} rekordów należy do "
            "powtarzających się znaczników czasu."
        )

    return df


def main() -> None:
    if not API_KEY:
        raise RuntimeError(
            "Brak ENTSOE_API_KEY. Sprawdź plik .env w katalogu projektu."
        )

    if START_DATE >= END_DATE:
        raise RuntimeError("Nieprawidłowy zakres dat.")

    print(f"Zakres pobierania: {START_DATE} – {END_DATE} (koniec wyłączny)")
    print(f"Surowe pliki: {RAW_DIR}")
    print(f"CSV wynikowy: {OUTPUT_CSV}")

    with requests.Session() as session:
        for year in range(START_DATE.year, END_DATE.year + 1):
            fetch_year(session, year)
            time.sleep(1)

    df = load_all_raw_data()

    if df.empty:
        print("Nie znaleziono żadnych rekordów cenowych.")
        return

    # Na tym etapie zachowujemy rozdzielczość i jednostki ze źródła.
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print("\nPobieranie i przetwarzanie zakończone.")
    print(f"Liczba rekordów: {len(df):,}")
    print(f"Pierwszy znacznik czasu UTC: {df['timestamp_utc'].min()}")
    print(f"Ostatni znacznik czasu UTC: {df['timestamp_utc'].max()}")
    print("\nRozdzielczości:")
    print(df["resolution"].value_counts(dropna=False).to_string())
    print("\nWaluty:")
    print(df["currency"].value_counts(dropna=False).to_string())
    print("\nLiczba brakujących cen:", df["price_eur_mwh"].isna().sum())


if __name__ == "__main__":
    main()