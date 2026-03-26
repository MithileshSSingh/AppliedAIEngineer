"""Train and save a churn prediction model for the API.

Run: python -m src.train_model
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report


FEATURE_NAMES = [
    "tenure_months", "monthly_charges", "total_charges",
    "contract_One year", "contract_Two year",
    "internet_Fiber optic", "internet_No",
    "online_security", "tech_support", "senior_citizen", "partner",
]


def generate_training_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic churn data for model training."""
    np.random.seed(seed)

    tenure = np.random.exponential(32, n).clip(1, 72).astype(int)
    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20]
    )
    internet = np.random.choice(
        ["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]
    )
    monthly = np.round(
        20 + 25 * (internet == "DSL") + 40 * (internet == "Fiber optic")
        + np.random.normal(0, 8, n),
        2,
    ).clip(18, 120)
    total = np.round(monthly * tenure * (1 + np.random.normal(0, 0.05, n)), 2)
    online_sec = np.where(internet == "No", 0, np.random.binomial(1, 0.35, n))
    tech_sup = np.where(internet == "No", 0, np.random.binomial(1, 0.30, n))
    senior = np.random.binomial(1, 0.16, n)
    partner = np.random.binomial(1, 0.48, n)

    logit = (
        -1.5 - 0.04 * tenure + 1.2 * (contract == "Month-to-month")
        - 0.8 * (contract == "Two year") + 0.02 * monthly
        + 0.5 * (internet == "Fiber optic") - 0.5 * online_sec
        - 0.4 * tech_sup + 0.3 * senior - 0.3 * partner
        + np.random.normal(0, 0.5, n)
    )
    churn = (np.random.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    return pd.DataFrame({
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "total_charges": total,
        "contract": contract,
        "internet_service": internet,
        "online_security": online_sec,
        "tech_support": tech_sup,
        "senior_citizen": senior,
        "partner": partner,
        "churned": churn,
    })


def prepare_features(df: pd.DataFrame) -> np.ndarray:
    """Convert DataFrame to feature array matching FEATURE_NAMES."""
    X = np.column_stack([
        df["tenure_months"].values,
        df["monthly_charges"].values,
        df["total_charges"].values,
        (df["contract"] == "One year").astype(int).values,
        (df["contract"] == "Two year").astype(int).values,
        (df["internet_service"] == "Fiber optic").astype(int).values,
        (df["internet_service"] == "No").astype(int).values,
        df["online_security"].values,
        df["tech_support"].values,
        df["senior_citizen"].values,
        df["partner"].values,
    ])
    return X


def train_and_save(output_path: str = None):
    """Train a GBM model and save to disk."""
    df = generate_training_data()
    X = prepare_features(df)
    y = df["churned"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("=== Churn Model Training Results ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC:  {roc_auc_score(y_test, y_proba):.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Retained', 'Churned'])}")

    # Save model artifact
    artifact = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "version": "1.0.0",
        "metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "auc": float(roc_auc_score(y_test, y_proba)),
        },
    }

    if output_path is None:
        output_path = str(Path(__file__).parent.parent / "model.pkl")

    with open(output_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\nModel saved to {output_path}")

    return artifact


if __name__ == "__main__":
    train_and_save()
