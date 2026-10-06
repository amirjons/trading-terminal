# Торговый терминал с LSTM-прогнозированием (EUR/USD H1)

Клиент-серверная система: LSTM прогнозирует направление следующей часовой свечи EUR/USD,
сервер эмулирует поток котировок и раздаёт сигналы BUY/SELL/HOLD, веб-терминал (с мобильной
вёрсткой) рисует японские свечи и поддерживает ручную и автоматическую торговлю.

## Быстрый старт (≈ 5 минут, предобученная модель в репозитории)

```bash
git clone https://github.com/amirjons/trading-terminal.git
cd trading-terminal
docker compose up --build
```
Клиент: http://localhost:8000 · WebSocket потока: `ws://localhost:8000/ws`

Настройки в `docker-compose.yml`: `SPEED` (свечей в секунду), `REPLAY_FROM` (начало воспроизведения).

## Как подключиться к тестовому потоку
- `GET /history?n=200`: последние свечи, текущий сигнал, счёт
- `WS /ws`: сообщения `{type: tick|close, candle, signal, account}`
- `POST /order` `{"action": "buy|sell|close"}`: ручная сделка
- `POST /mode` `{"mode": "manual|auto"}`: режим торговли
- `GET /account`: баланс, позиция, история сделок, P&L

## Как обучить модель с нуля
```bash
pip install pandas numpy torch matplotlib
# 1. Скачать EUR/USD M1 (Generic ASCII) за 2020–2025 с histdata.com
#    и положить CSV в data/raw_m1/
python ml/01_data_check.py   # M1 -> H1, проверки -> data/eurusd_h1.csv
python ml/02_features.py     # признаки, split, z-score -> data/features.csv, models/scaler.json
python ml/03_train.py        # обучение, выбор lookback -> models/lstm.pt
python ml/04_backtest.py     # бэктест -> models/backtest_metrics.json, docs/equity.png
```
Готовые `data/eurusd_h1.csv` и веса `models/lstm.pt` уже в репозитории.

## Данные и предобработка
- EUR/USD, таймфрейм **H1**: меньше шума, чем на M1/M5, при этом ≈36 тыс. свечей за 6 лет.
- Признаки только относительные: log-returns open/high/low/close к предыдущему close, RSI, MACD/цена,
  волатильность, ROC. Они не зависят от абсолютного уровня цены (проверка в `02_features.py`:
  умножение цен на 1.15 не меняет признаки).
- Split строго по времени 70/15/15, z-score считается только по train (`models/scaler.json`).
- Volume не используется: у HistData объём нулевой.

## Модель
LSTM 2×64, dropout 0.2, две головы: регрессия OHLC (Huber) и вероятность роста (BCE),
Adam lr=1e-3, early stopping. Lookback 24/48/96 сравнивались на val (`models/experiments.csv`), выбран 48.

## Результаты (test, 2025-02-13 .. 2025-12-31)
| Метрика | Значение |
|---|---|
| MAE / naive MAE | 0.000732 / 0.000734 |
| Точность направления | 52.7% (мажоритарный класс 50.7%) |
| Доходность стратегии | −1.94% (Buy & Hold +12.3%) |
| Sharpe | −0.25 |
| Макс. просадка | −7.1% |
| Winrate | 57.5% (572 сделки) |

Преимущество модели по направлению (≈2 п.п.) меньше торговых издержек, поэтому стратегия
после комиссии 0.5 пипса с каждой стороны не прибыльна. Это согласуется с близостью рынка H1 к эффективному.

Скриншоты терминала и кривая капитала лежат в `docs/`.

## Структура
`ml/` обучение и бэктест · `server/` FastAPI-эмулятор · `client/` веб-терминал · `models/` веса и метрики · `data/` данные
