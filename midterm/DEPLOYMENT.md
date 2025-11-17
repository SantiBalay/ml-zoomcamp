# Local Testing Guide

## Build and Run with Docker

### Build the Docker image
```bash
docker build -t bankruptcy-predictor .
```

### Run the container
```bash
docker run -p 9696:9696 bankruptcy-predictor
```

## Test the Service

### Health check
```bash
curl http://localhost:9696/health
```

### Make a prediction
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"ROA(C) before interest and depreciation before interest": 0.37, ...}'
```

### Use the test script
```bash
python test_service.py
```

## Notes
- Service runs on port 9696
- Model file must be present at `./data/xgboost_model_threshold_0.28.pkl`

