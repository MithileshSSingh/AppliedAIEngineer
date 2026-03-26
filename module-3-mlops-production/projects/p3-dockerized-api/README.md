# P3: Dockerized ML API

## Objective
Build, containerize, and test a production-ready FastAPI service that serves churn predictions with proper validation, health checks, and monitoring.

## Architecture
```
Client  →  FastAPI (uvicorn)  →  GBM Model  →  JSON Response
              │
              ├── /health         (readiness check)
              ├── /v1/predict     (single prediction)
              ├── /v1/predict/batch (batch prediction)
              ├── /model/info     (model metadata)
              └── /metrics        (latency stats)
```

## Key Skills
- FastAPI with Pydantic v2 validation
- Docker multi-stage builds
- API testing with TestClient
- Load testing with concurrent.futures
- Middleware for timing and logging

## Deliverables
1. `notebook.ipynb` — Build, test, and analyze the API
2. `src/main.py` — FastAPI application
3. `src/schemas.py` — Pydantic request/response models
4. `src/train_model.py` — Model training script
5. `Dockerfile` — Multi-stage production build
6. `docker-compose.yml` — Container orchestration

## Suggested Approach
**Week 12, Thursday-Friday:**
1. Train and save the churn model
2. Review the FastAPI app and schemas
3. Test all endpoints with TestClient
4. Build and run the Docker container
5. Load test and analyze latency

## How to Run

### Local (no Docker)
```bash
cd module-3-mlops-production/projects/p3-dockerized-api
python -m src.train_model
uvicorn src.main:app --reload
```

### Docker
```bash
python -m src.train_model   # Generate model.pkl first
docker compose up --build
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Notebook
```bash
jupyter lab notebook.ipynb
```
