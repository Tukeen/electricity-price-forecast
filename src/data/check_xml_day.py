
from pathlib import Path
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[2]
xml_path = PROJECT_ROOT / "data" / "raw" / "entsoe_prices" / "prices_2015.xml"

def local_name(tag):
    return tag.rsplit("}", 1)[-1]

root = ET.parse(xml_path).getroot()

for ts in root.iter():
    if local_name(ts.tag) != "TimeSeries":
        continue

    for period in ts.iter():
        if local_name(period.tag) != "Period":
            continue

        start = None
        resolution = None
        points = []

        for element in period.iter():
            name = local_name(element.tag)

            if name == "start" and start is None:
                start = element.text
            elif name == "resolution":
                resolution = element.text
            elif name == "Point":
                position = None
                price = None

                for child in element.iter():
                    child_name = local_name(child.tag)
                    if child_name == "position":
                        position = child.text
                    elif child_name == "price.amount":
                        price = child.text

                points.append((position, price))

        if start and start.startswith("2015-01-04"):
            print(f"Start: {start}")
            print(f"Resolution: {resolution}")
            print(f"Liczba punktów: {len(points)}")
            print(f"Pozycje: {[p[0] for p in points]}")
            print(f"Ceny: {[p[1] for p in points]}")
            print("---")