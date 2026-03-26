# P2: Distributed Training at Scale

## Objective
Train ML models at scale using SageMaker patterns — demonstrate that the same training script works locally and in the cloud.

## Dataset
- Source: Synthetic data from src/generate_large_data.py
- Size: 100,000 rows, 20+ features
- Format: Parquet (primary), CSV (sample)

## Architecture
Large Dataset → Train/Val Split → Hyperparameter Search → Best Model → Model Artifact

## Key Skills
- SageMaker-compatible training scripts (Script Mode)
- Hyperparameter optimization
- Model selection and comparison
- Training artifact packaging

## Deliverables
1. notebook.ipynb — Training workflow
2. src/generate_large_data.py — Large dataset generation

## Suggested Approach
Week 11, Thursday-Friday:
1. Generate 100K-row dataset
2. Write SageMaker-compatible training script
3. Run local training baseline
4. Hyperparameter search
5. Package best model artifact

## How to Run
```bash
cd module-3-mlops-production/projects/p2-distributed-training
python src/generate_large_data.py
jupyter lab notebook.ipynb
```
