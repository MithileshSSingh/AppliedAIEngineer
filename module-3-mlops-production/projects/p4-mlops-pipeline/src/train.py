"""MLflow-tracked training script for the churn MLOps pipeline."""

import argparse
import json
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


def generate_data(n: int = 8000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    tenure = np.random.exponential(24, n).clip(1, 72).astype(int)
    monthly = np.random.normal(65, 25, n).clip(20, 150)
    contract = np.random.choice([0, 1, 2], n, p=[0.55, 0.25, 0.20])
    calls = np.random.poisson(2, n)
    num_prod = np.random.randint(1, 6, n)
    total = tenure * monthly + np.random.normal(0, 200, n).clip(0)
    churn_prob = 1 / (1 + np.exp(-(-2 + 0.05*(monthly-65) - 0.03*tenure + 0.1*calls - 0.5*contract)))
    churn = (np.random.random(n) < churn_prob).astype(int)
    return pd.DataFrame({
        'tenure_months': tenure, 'monthly_charges': monthly, 'total_charges': total,
        'contract_encoded': contract, 'support_calls': calls, 'num_products': num_prod, 'churn': churn,
    })


def train_model(model, X_train, y_train, X_val, y_val, model_name: str, params: dict):
    """Train and log a model to MLflow."""
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.set_tag("model_type", model_name.split("-")[0])

        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)

        metrics = {
            "val_auc": roc_auc_score(y_val, y_prob),
            "val_f1": f1_score(y_val, y_pred),
            "val_ap": average_precision_score(y_val, y_prob),
            "val_precision": float((y_pred & y_val).sum() / max(y_pred.sum(), 1)),
            "val_recall": float((y_pred & y_val).sum() / max(y_val.sum(), 1)),
            "train_time_s": round(train_time, 3),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model", registered_model_name=None)

        if hasattr(model, 'feature_importances_'):
            fi = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
            fi_path = f"/tmp/fi_{model_name}.csv"
            fi.sort_values('importance', ascending=False).to_csv(fi_path, index=False)
            mlflow.log_artifact(fi_path, "feature_importances")

        run_id = mlflow.active_run().info.run_id
        print(f"  [{model_name:30s}] AUC={metrics['val_auc']:.4f}  F1={metrics['val_f1']:.4f}  run_id={run_id[:8]}")
        return run_id, metrics


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("churn-mlops-pipeline")

    print("Generating data...")
    df = generate_data(8000)
    feature_cols = [c for c in df.columns if c != 'churn']
    X = df[feature_cols]
    y = df['churn']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Churn rate: {y.mean():.2%}")

    print("\nTraining experiments:")
    models = [
        (GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
         "GBM-baseline", {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}),
        (GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
         "GBM-tuned", {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 5}),
        (RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
         "RF-200", {"n_estimators": 200, "max_depth": 10}),
        (LogisticRegression(C=1.0, max_iter=500, random_state=42),
         "LogReg-C1", {"C": 1.0}),
    ]

    run_results = []
    for model, name, params in models:
        run_id, metrics = train_model(model, X_train, y_train, X_val, y_val, name, params)
        run_results.append({'model': name, 'run_id': run_id, **metrics})

    results_df = pd.DataFrame(run_results).sort_values('val_auc', ascending=False)
    print("\nResults (sorted by AUC):")
    print(results_df[['model', 'val_auc', 'val_f1', 'train_time_s']].to_string(index=False))

    best = results_df.iloc[0]
    print(f"\nBest model: {best['model']} (AUC={best['val_auc']:.4f})")
    print(f"To register: mlflow.register_model(f'runs:/{best['run_id']}/model', 'churn-model')")

    results_df.to_csv('/tmp/training_results.csv', index=False)


if __name__ == '__main__':
    main()
