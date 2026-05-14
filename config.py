from pathlib import Path

TICKERS = ["MAR", "HLT"]

WRDS_USERNAME = "dzhuang"

START_DATE = "2026-04-01"
END_DATE = "2026-04-07"

SNAP_INTERVAL_MIN = 1

RETURN_WINDOW_20 = 20
OVERLAP_STEP_20 = 5

ROLLING_WINDOW_DAYS = 1
ROLLING_UPDATE_MINS = 60

PCA_CALIBRATION_HOURS = 3

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "output" / "figures"

for _d in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
