# P2: Distributed Training on SageMaker

## Objective
Scale model training using SageMaker's distributed training capabilities. Train the churn model on larger datasets using data parallelism, and track experiments with MLflow.

## Architecture
```
Large Dataset in S3 (partitioned by date)
         ↓
SageMaker Training Job
  - ml.m5.2xlarge × 2 instances (data parallel)
  - Custom training script with sagemaker.distributed
         ↓
Model artifacts → S3 → MLflow Registry
```

## Key Skills
- SageMaker distributed data parallelism
- Custom training containers
- Model checkpointing
- Multi-instance training configuration
- Cost optimization (Spot instances)

## Deliverables
1. `src/train.py` — Distributed training script
2. `src/generate_large_data.py` — Generate large dataset (100K rows)
3. `notebook.ipynb` — Launch and monitor training jobs

## Cost Estimate
- ml.m5.2xlarge × 2, 30 min: ~$0.19
- With Spot instances (70% discount): ~$0.06

## How to Run
```bash
python src/generate_large_data.py
python notebook.ipynb  # or use jupyter lab
```
