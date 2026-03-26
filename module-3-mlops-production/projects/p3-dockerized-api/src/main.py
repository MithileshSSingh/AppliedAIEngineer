"""FastAPI churn prediction service.

Run locally:  uvicorn src.main:app --reload
Run in Docker: docker compose up
"""

import time
import pickle
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from collections import defaultdict

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import (
    CustomerFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# --- Globals ---
model_artifact = None
start_time = None
request_metrics = defaultdict(list)


def load_model() -> dict:
    """Load model artifact from disk."""
    model_path = Path(__file__).parent.parent / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run: python -m src.train_model"
        )
    with open(model_path, "rb") as f:
        return pickle.load(f)


def customer_to_array(c: CustomerFeatures) -> np.ndarray:
    """Convert a CustomerFeatures Pydantic model to a numpy array."""
    return np.array([[
        c.tenure_months,
        c.monthly_charges,
        c.total_charges,
        int(c.contract == "One year"),
        int(c.contract == "Two year"),
        int(c.internet_service == "Fiber optic"),
        int(c.internet_service == "No"),
        c.online_security,
        c.tech_support,
        c.senior_citizen,
        c.partner,
    ]])


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_artifact, start_time
    logger.info("Loading model...")
    model_artifact = load_model()
    start_time = time.time()
    logger.info(
        "Model v%s loaded (AUC=%.4f)",
        model_artifact["version"],
        model_artifact["metrics"]["auc"],
    )
    yield
    logger.info("Shutting down")


# --- App ---
app = FastAPI(
    title="Churn Prediction API",
    description="Production ML service for customer churn prediction.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Middleware ---
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - t0) * 1000
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"
    request_metrics["latencies"].append(elapsed_ms)
    request_metrics["total_requests"].append(1)
    return response


# --- Endpoints ---
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        model_loaded=model_artifact is not None,
        model_version=model_artifact["version"] if model_artifact else "N/A",
        uptime_seconds=round(time.time() - start_time, 1) if start_time else 0,
    )


@app.post("/v1/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    if model_artifact is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = customer_to_array(customer)
    model = model_artifact["model"]
    proba = float(model.predict_proba(X)[0, 1])

    return PredictionResponse(
        churn_probability=round(proba, 4),
        churn_prediction=proba >= 0.5,
        model_version=model_artifact["version"],
    )


@app.post("/v1/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    if model_artifact is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(request.customers) == 0:
        raise HTTPException(status_code=400, detail="Empty batch")

    X = np.vstack([customer_to_array(c) for c in request.customers])
    model = model_artifact["model"]
    probas = model.predict_proba(X)[:, 1]

    predictions = [
        PredictionResponse(
            churn_probability=round(float(p), 4),
            churn_prediction=p >= 0.5,
            model_version=model_artifact["version"],
        )
        for p in probas
    ]

    return BatchPredictionResponse(predictions=predictions, count=len(predictions))


@app.get("/model/info")
def model_info():
    if model_artifact is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "version": model_artifact["version"],
        "features": model_artifact["feature_names"],
        "metrics": model_artifact["metrics"],
    }


@app.get("/metrics")
def metrics():
    latencies = request_metrics["latencies"]
    total = len(request_metrics["total_requests"])
    return {
        "total_requests": total,
        "avg_latency_ms": round(np.mean(latencies), 2) if latencies else 0,
        "p50_latency_ms": round(np.percentile(latencies, 50), 2) if latencies else 0,
        "p95_latency_ms": round(np.percentile(latencies, 95), 2) if latencies else 0,
        "p99_latency_ms": round(np.percentile(latencies, 99), 2) if latencies else 0,
    }
