# P4: Full MLOps Pipeline

## Objective
Build a complete MLOps pipeline: experiment tracking with MLflow, model registry lifecycle, A/B testing framework, and drift detection.

## Architecture
```
Train (MLflow) → Register (Model Registry) → A/B Test → Monitor Drift → Retrain
```

## Key Skills
- MLflow experiment tracking and model registry
- A/B test design and statistical analysis (chi-squared, confidence intervals, effect size)
- Data drift detection (PSI, KS test)
- Retraining trigger logic

## Deliverables
1. `notebook.ipynb` — Full MLOps workflow
2. `src/train.py` — MLflow-tracked training
3. `src/ab_test.py` — A/B test simulation and analysis

## Suggested Approach
**Week 13, Thursday-Friday:**
1. Train 3 models with MLflow tracking
2. Register best model, transition through lifecycle
3. Simulate and analyze A/B test
4. Implement drift detection
5. Design retraining trigger

## How to Run
```bash
cd module-3-mlops-production/projects/p4-mlops-pipeline
jupyter lab notebook.ipynb
```
