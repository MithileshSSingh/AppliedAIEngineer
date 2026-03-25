# P3: Dockerized ML API

## Objective
Build, containerize, and deploy a production-ready FastAPI ML service. The service predicts customer churn and includes health checks, validation, monitoring middleware, and API versioning.

## Architecture
```
Client Request
      ↓
 Load Balancer (nginx/ALB)
      ↓
 FastAPI (uvicorn, 2+ workers)
      ↓
 ML Model (GBM, loaded at startup)
      ↓
 Structured JSON Response
```

## Features
- Single and batch prediction endpoints
- Pydantic input validation
- Request/response logging middleware
- Prometheus metrics endpoint
- Health check for orchestration
- API versioning (v1/)
- Auto-generated Swagger docs at /docs

## Key Skills
- FastAPI with Pydantic v2
- Docker multi-stage builds
- uvicorn configuration
- Middleware and logging
- Load testing with locust

## Files
```
src/
├── main.py          — FastAPI app
├── model.py         — Model loading and prediction
├── schemas.py       — Pydantic request/response models
└── middleware.py    — Logging, metrics middleware
Dockerfile
docker-compose.yml
requirements.txt
tests/test_api.py
```

## How to Run
```bash
# Option 1: Local (no Docker)
pip install -r requirements.txt
python src/train_model.py    # Train and save model
uvicorn src.main:app --reload --port 8000

# Option 2: Docker
docker build -t churn-api .
docker run -p 8000:8000 churn-api

# Option 3: Docker Compose
docker-compose up
```

## Testing
```bash
# API test
pytest tests/
# Load test
locust -f tests/locustfile.py --headless -u 50 -r 10 --run-time 60s
```
