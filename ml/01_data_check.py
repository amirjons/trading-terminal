import glob
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent   # trading-terminal/
RAW = ROOT / "data" / "raw_m1"
OUT = ROOT / "data" / "eurusd_h1.csv"

files = sorted(glob.glob(str(RAW / "DAT_ASCII_EURUSD_M1_*.csv")))
print("ищу в:", RAW)
print("файлов:", len(files))
if not files:
    raise SystemExit("CSV не найдены")

parts = []
for f in files:
    p = pd.read_csv(f, sep=";", header=None,
                    names=["timestamp", "open", "high", "low", "close", "volume"])
    p["timestamp"] = pd.to_datetime(p["timestamp"], format="%Y%m%d %H%M%S")
    parts.append(p)

m1 = (pd.concat(parts)
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .set_index("timestamp"))
print("минутных строк:", len(m1))

# M1 -> H1
h1 = m1.resample("1h").agg({"open": "first", "high": "max", "low": "min",
                            "close": "last", "volume": "sum"})
h1 = h1.dropna(subset=["open", "close"]).reset_index()  # выходные выпадают

# проверки
bad = (h1.high < h1[["open", "close"]].max(axis=1)) | (h1.low > h1[["open", "close"]].min(axis=1))
print("битых свечей:", bad.sum())
h1 = h1[~bad].reset_index(drop=True)

print(h1.shape, h1.timestamp.min(), h1.timestamp.max())
print(h1[["open", "close"]].describe().loc[["min", "max"]])

gaps = h1.timestamp.diff()
print(gaps.value_counts().head(8))

h1.to_csv(OUT, index=False)
print("сохранено:", OUT)
