import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
torch.manual_seed(42)
np.random.seed(42)

# ---------- конфиг ----------
LOOKBACKS = [24, 48, 96]
HIDDEN, LAYERS, DROPOUT = 64, 2, 0.2
LR, BATCH, MAX_EPOCHS, PATIENCE = 1e-3, 64, 30, 5

# ---------- данные ----------
sc = json.loads((ROOT / "models" / "scaler.json").read_text())
FEATURES = sc["features"]
C_MEAN, C_STD = sc["mean"]["c"], sc["std"]["c"]
f = pd.read_csv(ROOT / "data" / "features.csv", parse_dates=["timestamp"])
feats = f[FEATURES].values.astype(np.float32)
close = f.close.values
split = f.split.values
NF = len(FEATURES)


def build(L, name):
    """Окно t-L..t-1 -> свеча t. Целевая свеча строго внутри своего split."""
    W = sliding_window_view(feats, (L, NF))[:, 0]      # (n-L+1, L, NF)
    t = np.where(split == name)[0]
    t = t[t >= L]
    X = torch.tensor(W[t - L])
    y_reg = torch.tensor(feats[t, :4])                  # o,h,l,c следующей свечи (z-score)
    y_up = torch.tensor((np.log(close[t] / close[t - 1]) > 0).astype(np.float32))
    return t, X, y_reg, y_up


# ---------- модель ----------
class LSTMNet(nn.Module):
    def __init__(self, n_feat, hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 5))

    def forward(self, x):
        out, _ = self.lstm(x)
        o = self.head(out[:, -1])
        return o[:, :4], o[:, 4]          # регрессия OHLC, логит роста


huber, bce = nn.HuberLoss(), nn.BCEWithLogitsLoss()


def loss_fn(reg, logit, y_reg, y_up):
    return huber(reg, y_reg) + bce(logit, y_up)


@torch.no_grad()
def predict(model, X):
    model.eval()
    regs, logits = [], []
    for i in range(0, len(X), 1024):
        r, l = model(X[i:i + 1024])
        regs.append(r)
        logits.append(l)
    return torch.cat(regs), torch.cat(logits)


def metrics(t, reg, logit, y_up):
    """Метрики по восстановленной цене + точность направления."""
    prev = close[t - 1]
    real = close[t]
    ret = reg[:, 3].numpy() * C_STD + C_MEAN           # log-return close обратно из z-score
    pred = prev * np.exp(ret)

    def price_m(p):
        e = p - real
        return np.abs(e).mean(), np.sqrt((e ** 2).mean()), (np.abs(e) / real).mean() * 100

    mae, rmse, mape = price_m(pred)
    n_mae, n_rmse, n_mape = price_m(prev)               # наивный baseline
    up = y_up.numpy() > 0.5
    p_up = torch.sigmoid(logit).numpy()
    return {
        "MAE": mae, "RMSE": rmse, "MAPE%": mape,
        "naive_MAE": n_mae, "naive_RMSE": n_rmse, "naive_MAPE%": n_mape,
        "acc_cls": ((p_up > 0.5) == up).mean(),
        "acc_reg": ((ret > 0) == up).mean(),
        "majority": max(up.mean(), 1 - up.mean()),
    }


# ---------- обучение одной конфигурации ----------
def train_one(L):
    _, Xtr, ytr, utr = build(L, "train")
    _, Xva, yva, uva = build(L, "val")
    loader = DataLoader(TensorDataset(Xtr, ytr, utr), batch_size=BATCH, shuffle=True)
    # shuffle внутри train допустим: это перемешивание окон, split по времени не нарушен

    model = LSTMNet(NF)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, best_state, bad, hist = float("inf"), None, 0, []

    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        tl = 0.0
        for xb, yb, ub in loader:
            opt.zero_grad()
            r, l = model(xb)
            loss = loss_fn(r, l, yb, ub)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(xb)
        r, l = predict(model, Xva)
        vl = loss_fn(r, l, yva, uva).item()
        hist.append((ep, tl / len(Xtr), vl))
        print(f"  L={L} эпоха {ep:2d}  train {tl / len(Xtr):.4f}  val {vl:.4f}")
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print("  early stopping")
                break
    model.load_state_dict(best_state)
    return model, best, len(hist)


if __name__ == "__main__":
    rows, models = [], {}
    for L in LOOKBACKS:
        print(f"\n=== lookback {L} ===")
        t0 = time.time()
        model, vloss, epochs = train_one(L)
        t, Xva, yva, uva = build(L, "val")
        r, l = predict(model, Xva)
        m = metrics(t, r, l, uva)
        rows.append({"lookback": L, "epochs": epochs, "val_loss": vloss,
                     "val_MAE": m["MAE"], "val_naive_MAE": m["naive_MAE"],
                     "val_MAPE%": m["MAPE%"], "val_acc_cls": m["acc_cls"],
                     "val_acc_reg": m["acc_reg"], "val_majority": m["majority"],
                     "sec": time.time() - t0})
        models[L] = model

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n=== Сравнение lookback (VAL) ===")
    print(table.round(5).to_string(index=False))
    table.to_csv(ROOT / "models" / "experiments.csv", index=False)

    best_L = int(table.loc[table.val_loss.idxmin(), "lookback"])
    print(f"\nвыбран lookback = {best_L} (минимальный val_loss)")

    # тест: один раз, только для выбранной модели
    t, Xte, yte, ute = build(best_L, "test")
    r, l = predict(models[best_L], Xte)
    m = metrics(t, r, l, ute)
    print("\n=== TEST ===")
    for k, v in m.items():
        print(f"{k:12s} {v:.6f}")

    torch.save({"state_dict": models[best_L].state_dict(),
                "config": {"lookback": best_L, "hidden": HIDDEN, "layers": LAYERS,
                           "dropout": DROPOUT, "features": FEATURES}},
               ROOT / "models" / "lstm.pt")
    (ROOT / "models" / "test_metrics.json").write_text(json.dumps(m, indent=2))
    print("сохранено: models/lstm.pt, experiments.csv, test_metrics.json")
