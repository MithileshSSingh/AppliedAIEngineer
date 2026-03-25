# P1: Cloud ML Data Pipeline

## Objective
Build a production-grade data pipeline that runs on AWS: raw data lands in S3, Lambda preprocesses it, SageMaker Processing runs feature engineering, and results feed into model training.

## Architecture
```
Local Data Generator
       ↓
   S3 (raw/)
       ↓
  Lambda Trigger
       ↓
  SageMaker Processing Job
       ↓
  S3 (processed/)
       ↓
  (Ready for Training)
```

## Key Skills
- boto3 for S3, Lambda, SageMaker
- SageMaker Processing Jobs
- Event-driven data pipelines
- IAM roles and policies
- CloudWatch logging

## Deliverables
1. `src/upload_raw.py` — Upload raw data to S3
2. `src/lambda_trigger.py` — Lambda function code
3. `src/processing_job.py` — SageMaker Processing job
4. `notebook.ipynb` — Orchestration and results

## How to Run
```bash
# Set up AWS credentials first
aws configure

# Upload data
python src/upload_raw.py

# Run processing job (requires SageMaker access)
python src/processing_job.py
```

## Cost Estimate
- S3: ~$0.01/month for this project's data
- SageMaker Processing: ~$0.05/run (ml.m5.large, ~5 min)
- Lambda: Free tier (1M invocations/month free)
