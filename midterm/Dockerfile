FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p ./data

COPY data/xgboost_model_threshold_0.28.pkl ./data/

COPY predict.py .

EXPOSE 9696

CMD ["gunicorn", "--bind", "0.0.0.0:9696", "--workers", "2", "--timeout", "120", "predict:app"]
