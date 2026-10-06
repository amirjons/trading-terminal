import numpy as np
import pandas as pd

WARMUP = 50


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    prev_close = df.close.shift(1)
    f = pd.DataFrame({"timestamp": df.timestamp, "close": df.close})
    f["o"] = np.log(df.open / prev_close)
    f["h"] = np.log(df.high / prev_close)
    f["l"] = np.log(df.low / prev_close)
    f["c"] = np.log(df.close / prev_close)

    delta = df.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(50)
    f["rsi"] = (rsi - 50) / 50

    macd = df.close.ewm(span=12, adjust=False).mean() - df.close.ewm(span=26, adjust=False).mean()
    f["macd"] = macd / df.close
    f["vol20"] = f["c"].rolling(20).std()
    f["roc10"] = np.log(df.close / df.close.shift(10))
    return f
