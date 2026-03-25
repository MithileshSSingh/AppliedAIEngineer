# P4: Full MLOps Pipeline

## Objective
Build a complete MLOps pipeline that integrates: MLflow tracking, model registry, A/B testing framework, and drift monitoring — all tied together in an automated workflow.

## Architecture
```
New Training Data
      ↓
 MLflow Experiment (track all runs)
      ↓
 Best Model → Model Registry (Staging)
      ↓
 A/B Test Framework (Staging vs Production)
      ↓
 Statistical Significance Check
      ↓
 Promote to Production (if significant improvement)
      ↓
 Drift Monitor (daily check, alert if PSI > 0.2)
```

## Key Skills
- MLflow experiment tracking and model registry
- A/B test design and analysis
- Statistical significance testing
- PSI-based drift detection
- Automated retraining triggers

## Deliverables
- `src/train.py` — Training with MLflow tracking
- `src/ab_test.py` — A/B test analysis functions
- `src/drift.py` — Drift detection utilities
- `notebook.ipynb` — Full pipeline orchestration

## How to Run
```bash
# Start MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts

# Run training
python src/train.py

# Open MLflow UI
open http://localhost:5000
```
