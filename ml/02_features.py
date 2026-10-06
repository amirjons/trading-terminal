import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
(ROOT / "models").mkdir(exist_ok=True)

FEATURES = ["o", "h", "l", "c", "rsi", "macd", "vol20", "roc10"]
WARMUP = 50  # первые строки, где индикаторы ещё не "прогрелись"


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Все признаки безразмерные: не зависят от абсолютного уровня цены."""
    prev_close = df.close.shift(1)
    f = pd.DataFrame({"timestamp": df.timestamp, "close": df.close})

    # log-returns относительно предыдущего close
    f["o"] = np.log(df.open / prev_close)
    f["h"] = np.log(df.high / prev_close)
    f["l"] = np.log(df.low / prev_close)
    f["c"] = np.log(df.close / prev_close)

    # RSI(14), приведён к диапазону [-1, 1]
    delta = df.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(50)
    f["rsi"] = (rsi - 50) / 50

    # MACD, нормированный на цену
    macd = df.close.ewm(span=12, adjust=False).mean() - df.close.ewm(span=26, adjust=False).mean()
    f["macd"] = macd / df.close

    # волатильность: std доходностей за 20 свечей
    f["vol20"] = f["c"].rolling(20).std()

    # скорость изменения цены (ROC) за 10 свечей
    f["roc10"] = np.log(df.close / df.close.shift(10))

    return f


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "eurusd_h1.csv", parse_dates=["timestamp"])

    # --- тест устойчивости к уровню цен (пригодится для защиты) ---
    a = make_features(df)[FEATURES].iloc[WARMUP:]
    scaled = df.copy()
    scaled[["open", "high", "low", "close"]] *= 1.15   # "EUR/USD ушёл на +15%"
    b = make_features(scaled)[FEATURES].iloc[WARMUP:]
    print("признаки инвариантны к масштабу цены:", np.allclose(a.values, b.values, atol=1e-8))

    # --- признаки, прогрев, split по времени ---
    f = make_features(df).iloc[WARMUP:].reset_index(drop=True)
    n = len(f)
    i1, i2 = int(n * 0.70), int(n * 0.85)
    f["split"] = "train"
    f.loc[i1:i2 - 1, "split"] = "val"
    f.loc[i2:, "split"] = "test"

    for name, part in f.groupby("split", sort=False):
        print(f"{name:5s} {len(part):6d} строк  {part.timestamp.min()} .. {part.timestamp.max()}"
              f"  цена {part.close.min():.3f}..{part.close.max():.3f}")

    # --- z-score: статистика ТОЛЬКО по train ---
    train = f[f.split == "train"]
    mean = train[FEATURES].mean()
    std = train[FEATURES].std()
    f[FEATURES] = ((f[FEATURES] - mean) / std).clip(-5, 5)

    scaler = {"features": FEATURES, "mean": mean.to_dict(), "std": std.to_dict(),
              "clip": 5, "warmup": WARMUP}
    (ROOT / "models" / "scaler.json").write_text(json.dumps(scaler, indent=2))
    f.to_csv(ROOT / "data" / "features.csv", index=False)
    print("сохранено: data/features.csv, models/scaler.json")
    print(f[FEATURES].describe().loc[["mean", "std", "min", "max"]].round(2))
