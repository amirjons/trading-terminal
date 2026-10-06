import asyncio
import json
import os
import random
import sys
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
from features import WARMUP, make_features      # noqa: E402
from lstm_model import LSTMNet                  # noqa: E402

# ---------- настройки ----------
SPEED = float(os.getenv("SPEED", "1"))                    # свечей в секунду
REPLAY_FROM = os.getenv("REPLAY_FROM", "2025-02-13 20:00:00")   # начало test
START_BALANCE, NOTIONAL, HALF_SPREAD = 10_000.0, 10_000.0, 0.00005
BUF = 600

# ---------- модель ----------
ck = torch.load(ROOT / "models" / "lstm.pt", map_location="cpu")
cfg = ck["config"]
LOOKBACK, FEATURES = cfg["lookback"], cfg["features"]
model = LSTMNet(len(FEATURES), cfg["hidden"], cfg["layers"], cfg["dropout"])
model.load_state_dict(ck["state_dict"])
model.eval()
sc = json.loads((ROOT / "models" / "scaler.json").read_text())
MEAN, STD = pd.Series(sc["mean"])[FEATURES], pd.Series(sc["std"])[FEATURES]
tc = ROOT / "models" / "trade_config.json"
THETA = float(os.getenv("THETA", json.loads(tc.read_text())["theta"] if tc.exists() else 0.01))

HOLD = {"signal": "HOLD", "confidence": 0.5, "p_up": 0.5}
S = {"mode": "manual", "balance": START_BALANCE, "pos": None, "trades": [],
     "price": None, "cur": None, "closed": [], "signal": HOLD, "finished": False}
clients: set = set()


def predict_signal(raw):
    f = make_features(pd.DataFrame(list(raw))).iloc[WARMUP:]
    x = ((f[FEATURES] - MEAN) / STD).clip(-5, 5).values.astype(np.float32)[-LOOKBACK:]
    if len(x) < LOOKBACK or np.isnan(x).any():
        return HOLD
    with torch.no_grad():
        _, logit = model(torch.tensor(x)[None])
    p = float(torch.sigmoid(logit)[0])
    sig = "BUY" if p > 0.5 + THETA else "SELL" if p < 0.5 - THETA else "HOLD"
    return {"signal": sig, "confidence": round(max(p, 1 - p), 4), "p_up": round(p, 4)}


# ---------- торговый счёт ----------
def unreal():
    p = S["pos"]
    if not p or S["price"] is None:
        return 0.0
    exit_px = S["price"] - p["side"] * HALF_SPREAD
    return p["side"] * (exit_px - p["entry"]) / p["entry"] * NOTIONAL


def close_pos(price, ts):
    p = S["pos"]
    if not p:
        return
    exit_px = price - p["side"] * HALF_SPREAD
    pnl = p["side"] * (exit_px - p["entry"]) / p["entry"] * NOTIONAL
    S["balance"] += pnl
    S["trades"].append({"side": "BUY" if p["side"] == 1 else "SELL",
                        "entry": round(p["entry"], 5), "exit": round(exit_px, 5),
                        "pnl": round(pnl, 2), "open_time": p["time"], "close_time": ts})
    S["pos"] = None


def move_to(target, price, ts):
    """target: 1 лонг, -1 шорт. Общая логика для ручного и авто режимов."""
    p = S["pos"]
    if p and p["side"] == target:
        return
    if p:
        close_pos(price, ts)
    S["pos"] = {"side": target, "entry": price + target * HALF_SPREAD, "time": ts}


def account():
    u, p = unreal(), S["pos"]
    return {"mode": S["mode"], "balance": round(S["balance"], 2),
            "equity": round(S["balance"] + u, 2), "unrealized": round(u, 2),
            "total_pnl": round(sum(t["pnl"] for t in S["trades"]), 2),
            "n_trades": len(S["trades"]),
            "position": None if not p else {"side": "BUY" if p["side"] == 1 else "SELL",
                                            "entry": round(p["entry"], 5)},
            "trades": S["trades"][-30:][::-1]}


async def broadcast(msg):
    msg["signal"], msg["account"], msg["finished"] = S["signal"], account(), S["finished"]
    dead = []
    for ws in list(clients):
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


# ---------- эмулятор рынка ----------
async def replay():
    df = pd.read_csv(ROOT / "data" / "eurusd_h1.csv", parse_dates=["timestamp"])
    start = pd.Timestamp(REPLAY_FROM)
    cols = ["timestamp", "open", "high", "low", "close"]
    past, future = df[df.timestamp < start].tail(BUF), df[df.timestamp >= start]
    raw = deque(past[cols].to_dict("records"), maxlen=BUF)
    S["closed"] = [{"time": int(r.timestamp.value // 10**9), "open": float(r.open), "high": float(r.high),
                    "low": float(r.low), "close": float(r.close)} for r in past.tail(200).itertuples()]
    S["signal"] = predict_signal(raw)
    await asyncio.sleep(1)

    for r in future.itertuples():
        o, h, l, c = float(r.open), float(r.high), float(r.low), float(r.close)
        ts = int(r.timestamp.value // 10**9)
        ticks = [o] + ([h, l] if random.random() < 0.5 else [l, h]) + [c]
        hi, lo = o, o
        for px in ticks:                      # свеча собирается из тиков
            hi, lo = max(hi, px), min(lo, px)
            S["price"] = px
            S["cur"] = {"time": ts, "open": o, "high": hi, "low": lo, "close": px}
            await broadcast({"type": "tick", "candle": S["cur"]})
            await asyncio.sleep(1 / (SPEED * len(ticks)))

        S["closed"] = (S["closed"] + [dict(S["cur"])])[-1000:]
        raw.append({"timestamp": r.timestamp, "open": o, "high": h, "low": l, "close": c})
        S["signal"] = predict_signal(raw)
        if S["mode"] == "auto":               # бот торгует по сигналу модели
            if S["signal"]["signal"] == "BUY":
                move_to(1, c, ts)
            elif S["signal"]["signal"] == "SELL":
                move_to(-1, c, ts)
        await broadcast({"type": "close", "candle": S["cur"]})

    S["finished"] = True
    await broadcast({"type": "end"})


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(replay())
    yield
    task.cancel()


app = FastAPI(title="Trading terminal", lifespan=lifespan)


class Order(BaseModel):
    action: str       # buy | sell | close


class Mode(BaseModel):
    mode: str         # manual | auto


@app.get("/")
def index():
    return FileResponse(ROOT / "client" / "index.html")


@app.get("/history")
def history(n: int = 200):
    return {"candles": S["closed"][-n:], "cur": S["cur"], "signal": S["signal"], "account": account()}


@app.get("/account")
def get_account():
    return account()


@app.post("/mode")
def set_mode(m: Mode):
    if m.mode not in ("manual", "auto"):
        raise HTTPException(400, "mode: manual | auto")
    S["mode"] = m.mode
    return account()


@app.post("/order")
def order(o: Order):
    if S["mode"] != "manual":
        raise HTTPException(400, "Включён авто-режим")
    if S["price"] is None:
        raise HTTPException(400, "Нет цены, подожди секунду")
    ts, a = S["cur"]["time"], o.action.lower()
    if a == "buy":
        move_to(1, S["price"], ts)
    elif a == "sell":
        move_to(-1, S["price"], ts)
    elif a == "close":
        close_pos(S["price"], ts)
    else:
        raise HTTPException(400, "action: buy | sell | close")
    return account()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)
