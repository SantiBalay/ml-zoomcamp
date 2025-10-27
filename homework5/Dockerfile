FROM agrigorev/zoomcamp-model:2025

WORKDIR /code

RUN pip install fastapi uvicorn scikit-learn

COPY data/pipeline_v1.bin /code/pipeline_v1.bin

COPY predict.py /code/predict.py

EXPOSE 9697

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9697"]
