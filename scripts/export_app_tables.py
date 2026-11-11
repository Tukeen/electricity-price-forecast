import shutil
from pathlib import Path
 
ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "reports" / "tables", ROOT / "app_data" / "research"
FILES = ["ml_results.csv", "exog_gain.csv", "test_main_results.csv", "test_sens_results.csv"]
 
DST.mkdir(parents=True, exist_ok=True)
for name in FILES:
    if (SRC / name).exists():
        shutil.copy2(SRC / name, DST / name)
        print("skopiowano", name)
    else:
        print("brak", name)