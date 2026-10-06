import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lstm_model import LSTMNet

COST = 0.00005          # ~0.5 пипса с каждой стороны, доля от цены
BARS_PER_YEAR = 6240    # часовых свечей в году на Форексе (24ч * 5дн * 52нед)

ck = torch.load(ROOT / "models" / "lstm.pt", map_location="cpu")
cfg = ck["config"]
L, FEATURES = cfg["lookback"], cfg["features"]
model = LSTMNet(len(FEATURES), cfg["hidden"], cfg["layers"], cfg["dropout"])
model.load_state_dict(ck["state_dict"])
model.eval()

f = pd.read_csv(ROOT / "data" / "features.csv", parse_dates=["timestamp"])
feats = f[FEATURES].values.astype(np.float32)
close, split = f.close.values, f.split.values
W = sliding_window_view(feats, (L, len(FEATURES)))[:, 0]


def probs(name):
    t = np.where(split == name)[0]
    t = t[t >= L]
    X = torch.tensor(W[t - L])
    with torch.no_grad():
        p = torch.cat([torch.sigmoid(model(X[i:i + 1024])[1]) for i in range(0, len(X), 1024)])
    return t, p.numpy()


def backtest(t, p, theta):
    ret = close[t] / close[t - 1] - 1
    pos, cur = np.zeros(len(t)), 0
    for i in range(len(t)):
        if p[i] > 0.5 + theta:
            cur = 1
        elif p[i] < 0.5 - theta:
            cur = -1
        pos[i] = cur                      # позиция на свече t решена по данным до t-1
    prev = np.concatenate([[0], pos[:-1]])
    pnl = pos * ret - COST * np.abs(pos - prev)
    eq = np.cumprod(1 + pnl)

    trades, start = [], None
    for i in range(len(pos)):
        if i == 0 or pos[i] != pos[i - 1]:
            if start is not None and pos[start] != 0:
                trades.append(np.prod(1 + pnl[start:i]) - 1)
            start = i
    if start is not None and pos[start] != 0:
        trades.append(np.prod(1 + pnl[start:]) - 1)

    sd = pnl.std()
    return {
        "return_%": (eq[-1] - 1) * 100,
        "sharpe": pnl.mean() / sd * np.sqrt(BARS_PER_YEAR) if sd > 0 else 0.0,
        "max_drawdown_%": (eq / np.maximum.accumulate(eq) - 1).min() * 100,
        "winrate_%": np.mean(np.array(trades) > 0) * 100 if trades else 0.0,
        "trades": len(trades),
        "buy_hold_%": (close[t[-1]] / close[t[0] - 1] - 1) * 100,
    }, eq


tv, pv = probs("val")
tt, pt = probs("test")
print("p(рост) на val: мин %.3f, медиана %.3f, макс %.3f" % (pv.min(), np.median(pv), pv.max()))

# подбор порога θ на VAL
best_theta, best_sharpe = 0.0, -1e9
print("\nθ      val_sharpe  val_return%  trades")
for theta in [0.0, 0.005, 0.01, 0.015, 0.02, 0.03]:
    m, _ = backtest(tv, pv, theta)
    print(f"{theta:<6} {m['sharpe']:>9.2f}  {m['return_%']:>10.2f}  {m['trades']:>6}")
    if m["trades"] >= 10 and m["sharpe"] > best_sharpe:
        best_theta, best_sharpe = theta, m["sharpe"]
print(f"\nвыбран θ = {best_theta} (по val)")

m, eq = backtest(tt, pt, best_theta)
print("\n=== TEST (комиссия 0.5 пипса с каждой стороны) ===")
for k, v in m.items():
    print(f"{k:16s} {v:.3f}" if isinstance(v, float) else f"{k:16s} {v}")

m0, _ = backtest(tt, pt, best_theta)
nc_pnl_note = "без комиссии результат был бы выше, см. COST"

(ROOT / "models" / "backtest_metrics.json").write_text(json.dumps({**m, "theta": best_theta}, indent=2))
(ROOT / "models" / "trade_config.json").write_text(json.dumps({"theta": best_theta}))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (ROOT / "docs").mkdir(exist_ok=True)
    plt.figure(figsize=(9, 4))
    plt.plot(f.timestamp.values[tt], eq, label="LSTM стратегия")
    bh = close[tt] / close[tt[0] - 1]
    plt.plot(f.timestamp.values[tt], bh, label="Buy & Hold", alpha=0.6)
    plt.title("Кривая капитала, test"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(ROOT / "docs" / "equity.png", dpi=130)
    print("график: docs/equity.png")
except ImportError:
    print("matplotlib не установлен: pip install matplotlib")
