
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
import truststore
from dotenv import load_dotenv

truststore.inject_into_ssl()

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("ENTSOE_API_KEY")
URL = "https://web-api.tp.entsoe.eu/api"

params = {
    "documentType": "A44",
    "in_Domain": "10YPL-AREA-----S",
    "out_Domain": "10YPL-AREA-----S",
    "periodStart": "202401040000",
    "periodEnd": "202401050000",
}

response = requests.get(
    URL,
    params=params,
    headers={"SECURITY_TOKEN": API_KEY},
    timeout=120,
)
print("HTTP:", response.status_code)

response.raise_for_status()
root = ET.fromstring(response.content)

def local_name(tag):
    return tag.rsplit("}", 1)[-1]

if local_name(root.tag) != "Publication_MarketDocument":
    print("Typ odpowiedzi:", local_name(root.tag))
    for element in root.iter():
        if local_name(element.tag) == "text" and element.text:
            print("Komunikat API:", element.text)
    raise SystemExit(1)

series_count = sum(
    1 for element in root.iter()
    if local_name(element.tag) == "TimeSeries"
)

point_count = sum(
    1 for element in root.iter()
    if local_name(element.tag) == "Point"
)

print("Liczba TimeSeries:", series_count)
print("Liczba Point:", point_count)
print("Rozmiar odpowiedzi XML:", len(response.content), "bajtów")




print("\nSzczegóły okresów:")

for series in root.iter():
    if local_name(series.tag) != "TimeSeries":
        continue

    curve_type = None
    for element in series.iter():
        if local_name(element.tag) == "curveType":
            curve_type = element.text
            break

    for period in series:
        if local_name(period.tag) != "Period":
            continue

        interval_start = None
        interval_end = None
        resolution = None
        positions = []

        for element in period.iter():
            name = local_name(element.tag)

            if name == "start":
                interval_start = element.text
            elif name == "end":
                interval_end = element.text
            elif name == "resolution":
                resolution = element.text
            elif name == "Point":
                for child in element:
                    if local_name(child.tag) == "position":
                        positions.append(child.text)

        print(
            f"curveType: {curve_type}, "
            f"start: {interval_start}, "
            f"end: {interval_end}, "
            f"resolution: {resolution}, "
            f"liczba jawnych punktów: {len(positions)}, "
            f"pozycje: {positions}"
        )