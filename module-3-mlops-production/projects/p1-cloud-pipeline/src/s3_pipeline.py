"""S3 data pipeline utilities for the cloud pipeline project.

Provides a LocalS3Simulator for AWS-free development and pipeline
functions for data validation, preprocessing, and orchestration.
"""

import json
import time
import shutil
import hashlib
import tempfile
from io import BytesIO
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


class LocalS3Simulator:
    """Simulates AWS S3 using the local filesystem.

    Mirrors the boto3 S3 client API so you can develop and test
    S3-based pipelines without an AWS account.
    """

    def __init__(self, root_dir: str = None):
        self.root = Path(root_dir or tempfile.mkdtemp(prefix="local-s3-"))
        self.root.mkdir(parents=True, exist_ok=True)
        self._metadata = {}

    def create_bucket(self, Bucket: str, **kwargs) -> dict:
        bucket_path = self.root / Bucket
        bucket_path.mkdir(exist_ok=True)
        return {"Location": f"/{Bucket}"}

    def list_buckets(self) -> dict:
        buckets = []
        for p in sorted(self.root.iterdir()):
            if p.is_dir():
                buckets.append({
                    "Name": p.name,
                    "CreationDate": datetime.fromtimestamp(p.stat().st_ctime)
                })
        return {"Buckets": buckets}

    def put_object(self, Bucket: str, Key: str, Body: bytes, **kwargs) -> dict:
        if isinstance(Body, str):
            Body = Body.encode('utf-8')
        obj_path = self.root / Bucket / Key
        obj_path.parent.mkdir(parents=True, exist_ok=True)
        obj_path.write_bytes(Body)
        meta_key = f"{Bucket}/{Key}"
        self._metadata[meta_key] = {
            "ContentLength": len(Body),
            "LastModified": datetime.now(),
            "ETag": hashlib.md5(Body).hexdigest(),
        }
        return {"ETag": self._metadata[meta_key]["ETag"]}

    def get_object(self, Bucket: str, Key: str) -> dict:
        obj_path = self.root / Bucket / Key
        if not obj_path.exists():
            raise FileNotFoundError(f"NoSuchKey: {Key} in bucket {Bucket}")
        body = obj_path.read_bytes()
        return {
            "Body": BytesIO(body),
            "ContentLength": len(body),
            "LastModified": datetime.fromtimestamp(obj_path.stat().st_mtime),
        }

    def delete_object(self, Bucket: str, Key: str) -> dict:
        obj_path = self.root / Bucket / Key
        if obj_path.exists():
            obj_path.unlink()
        return {}

    def head_object(self, Bucket: str, Key: str) -> dict:
        obj_path = self.root / Bucket / Key
        if not obj_path.exists():
            raise FileNotFoundError(f"NoSuchKey: {Key}")
        stat = obj_path.stat()
        return self._metadata.get(f"{Bucket}/{Key}", {
            "ContentLength": stat.st_size,
            "LastModified": datetime.fromtimestamp(stat.st_mtime),
        })

    def list_objects_v2(self, Bucket: str, Prefix: str = "", MaxKeys: int = 1000) -> dict:
        bucket_path = self.root / Bucket
        if not bucket_path.exists():
            return {"Contents": [], "KeyCount": 0}
        objects = []
        for p in sorted(bucket_path.rglob("*")):
            if p.is_file():
                key = str(p.relative_to(bucket_path))
                if key.startswith(Prefix):
                    objects.append({
                        "Key": key,
                        "Size": p.stat().st_size,
                        "LastModified": datetime.fromtimestamp(p.stat().st_mtime),
                    })
        objects = objects[:MaxKeys]
        return {"Contents": objects, "KeyCount": len(objects)}

    def upload_file(self, Filename: str, Bucket: str, Key: str) -> None:
        with open(Filename, 'rb') as f:
            self.put_object(Bucket=Bucket, Key=Key, Body=f.read())

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None:
        response = self.get_object(Bucket=Bucket, Key=Key)
        Path(Filename).parent.mkdir(parents=True, exist_ok=True)
        with open(Filename, 'wb') as f:
            f.write(response['Body'].read())

    def cleanup(self):
        if self.root.exists():
            shutil.rmtree(self.root)


def validate_data(df: pd.DataFrame, config: dict = None) -> dict:
    """Validate a DataFrame against expected schema and quality rules.

    Returns a dict with status ('PASSED' or 'FAILED'), issues list,
    and detailed metrics.
    """
    config = config or {
        "required_columns": [
            "customer_id", "tenure_months", "monthly_charges", "churned"
        ],
        "max_null_rate": 0.05,
        "max_duplicate_rate": 0.01,
        "ranges": {
            "tenure_months": (0, 120),
            "monthly_charges": (0, 500),
        }
    }

    issues = []

    # Check required columns
    missing = set(config["required_columns"]) - set(df.columns)
    if missing:
        issues.append(f"Missing columns: {missing}")

    # Check null rates
    null_rates = df.isnull().mean()
    high_nulls = null_rates[null_rates > config["max_null_rate"]]
    if len(high_nulls) > 0:
        for col, rate in high_nulls.items():
            issues.append(f"High null rate in '{col}': {rate:.1%}")

    # Check duplicates
    dup_rate = df.duplicated().mean()
    if dup_rate > config["max_duplicate_rate"]:
        issues.append(f"Duplicate rate {dup_rate:.1%} exceeds {config['max_duplicate_rate']:.1%}")

    # Check value ranges
    for col, (lo, hi) in config.get("ranges", {}).items():
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min < lo or col_max > hi:
                issues.append(
                    f"'{col}' out of range [{lo}, {hi}]: "
                    f"actual [{col_min}, {col_max}]"
                )

    return {
        "status": "FAILED" if issues else "PASSED",
        "issues": issues,
        "rows": len(df),
        "columns": len(df.columns),
        "null_rate": float(df.isnull().mean().mean()),
        "duplicate_rate": float(dup_rate),
        "timestamp": datetime.now().isoformat(),
    }


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the churn dataset."""
    original_rows = len(df)

    # Remove duplicates
    df = df.drop_duplicates()

    # Impute nulls: median for numeric, mode for categorical
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Clip out-of-range values
    if 'tenure_months' in df.columns:
        df['tenure_months'] = df['tenure_months'].clip(0, 120)
    if 'monthly_charges' in df.columns:
        df['monthly_charges'] = df['monthly_charges'].clip(0, 500)

    print(f"Preprocessed: {original_rows} -> {len(df)} rows "
          f"(removed {original_rows - len(df)} duplicates)")
    return df


def run_pipeline(s3_client, bucket: str, raw_key: str) -> dict:
    """Run the full data pipeline: validate → preprocess → save.

    Returns a dict with pipeline status, timing, and output details.
    """
    pipeline_start = time.time()
    stages = {}

    # Stage 1: Download raw data
    t0 = time.time()
    response = s3_client.get_object(Bucket=bucket, Key=raw_key)
    df = pd.read_csv(BytesIO(response['Body'].read()))
    stages['download'] = time.time() - t0
    print(f"[1/4] Downloaded {raw_key}: {len(df)} rows")

    # Stage 2: Validate
    t0 = time.time()
    val_result = validate_data(df)
    stages['validate'] = time.time() - t0
    print(f"[2/4] Validation: {val_result['status']} ({len(val_result['issues'])} issues)")

    if val_result['status'] == 'FAILED':
        return {
            "status": "FAILED",
            "reason": val_result['issues'],
            "elapsed": time.time() - pipeline_start,
        }

    # Stage 3: Preprocess
    t0 = time.time()
    df_clean = preprocess(df)
    stages['preprocess'] = time.time() - t0
    print(f"[3/4] Preprocessed: {len(df_clean)} rows")

    # Stage 4: Save as Parquet
    t0 = time.time()
    output_key = raw_key.replace("raw/", "processed/").replace(".csv", ".parquet")
    buffer = BytesIO()
    df_clean.to_parquet(buffer, index=False)
    parquet_bytes = buffer.getvalue()
    s3_client.put_object(Bucket=bucket, Key=output_key, Body=parquet_bytes)
    stages['save'] = time.time() - t0
    print(f"[4/4] Saved: {output_key} ({len(parquet_bytes):,} bytes)")

    elapsed = time.time() - pipeline_start
    return {
        "status": "SUCCESS",
        "input_key": raw_key,
        "output_key": output_key,
        "input_rows": len(df),
        "output_rows": len(df_clean),
        "output_bytes": len(parquet_bytes),
        "stages": stages,
        "elapsed": elapsed,
        "timestamp": datetime.now().isoformat(),
    }
