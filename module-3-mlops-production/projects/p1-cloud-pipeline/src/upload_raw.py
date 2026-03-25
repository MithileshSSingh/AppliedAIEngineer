"""Upload raw training data to S3.

Run this first to seed your S3 bucket with raw data.
"""

import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[4] / '.env')

AWS_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
BUCKET_NAME = os.getenv('S3_BUCKET', 'your-ml-pipeline-bucket')


def generate_churn_data(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic customer churn dataset."""
    np.random.seed(seed)

    customer_ids = [f'CUST-{i:06d}' for i in range(n)]
    tenure = np.random.exponential(24, n).clip(1, 72).astype(int)
    monthly_charges = np.random.normal(65, 25, n).clip(20, 150).round(2)
    total_charges = (tenure * monthly_charges + np.random.normal(0, 100, n)).clip(50).round(2)

    # Churn probability based on features
    churn_prob = 1 / (1 + np.exp(-(
        -2 + 0.05 * (monthly_charges - 65) - 0.03 * tenure
    )))
    churn = (np.random.random(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        'customer_id': customer_ids,
        'tenure_months': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.25, 0.20]),
        'payment_method': np.random.choice(['Credit card', 'Bank transfer', 'Electronic check', 'Mailed check'], n),
        'internet_service': np.random.choice(['Fiber optic', 'DSL', 'No'], n, p=[0.44, 0.34, 0.22]),
        'num_products': np.random.randint(1, 6, n),
        'support_calls': np.random.poisson(2, n),
        'churn': churn,
    })

    # Inject messiness
    missing_idx = np.random.choice(n, 200, replace=False)
    df.loc[missing_idx[:100], 'monthly_charges'] = np.nan
    df.loc[missing_idx[100:], 'num_products'] = np.nan

    return df


def upload_to_s3(df: pd.DataFrame, bucket: str, key: str) -> None:
    """Upload DataFrame to S3."""
    s3 = boto3.client('s3', region_name=AWS_REGION)

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer.getvalue(),
        ContentType='text/csv',
    )
    print(f"Uploaded {len(df)} rows → s3://{bucket}/{key}")


def main():
    print(f"Generating churn dataset...")
    df = generate_churn_data()

    print(f"Shape: {df.shape}")
    print(f"Churn rate: {df['churn'].mean():.1%}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    # Save locally first
    local_path = Path(__file__).parent.parent / 'data' / 'raw_churn.csv'
    df.to_csv(local_path, index=False)
    print(f"\nSaved locally: {local_path}")

    # Upload to S3 (if credentials available)
    try:
        date_str = datetime.now().strftime('%Y-%m-%d')
        s3_key = f'raw/churn/{date_str}/churn_data.csv'
        upload_to_s3(df, BUCKET_NAME, s3_key)
    except Exception as e:
        print(f"\nS3 upload skipped (AWS not configured): {e}")
        print("Data saved locally only.")


if __name__ == '__main__':
    main()
