FROM python:3.11-slim
WORKDIR /app

COPY requirements-server.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-server.txt

COPY server/ server/
COPY ml/lstm_model.py ml/features.py ml/
COPY client/ client/
COPY models/ models/
COPY data/eurusd_h1.csv data/eurusd_h1.csv

EXPOSE 8000
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
