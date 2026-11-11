import shutil
from pathlib import Path
 
ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "reports" / "tables", ROOT / "app_data" / "research"
FILES = ["ml_results.csv", "exog_gain.csv", "hypotheses_h3_bootstrap.csv", "hypotheses_dm.csv"]
FILES += [f"test_{spec}_{part}.csv" for spec in ("main", "sens")
          for part in ("results", "dm", "h3_bootstrap", "negative_days", "intervals90_monthly")]
 
DST.mkdir(parents=True, exist_ok=True)
for name in FILES:
    if (SRC / name).exists():
        shutil.copy2(SRC / name, DST / name)
        print("skopiowano", name)
    else:
        print("brak", name)

frozen = ROOT / "docs" / "frozen_data.json"      # odcisk danych testowych (powstaje w styczniu 2027)
if frozen.exists():
    shutil.copy2(frozen, DST / "frozen_data.json")
    print("skopiowano frozen_data.json")