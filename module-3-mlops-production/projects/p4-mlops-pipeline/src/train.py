"""MLflow-tracked model training for the MLOps pipeline project."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not installed. Training will proceed without tracking.")


def generate_churn_data(n=5000, seed=42):
    """Generate synthetic churn data."""
    np.random.seed(seed)
    # Generate: tenure, monthly_charges, total_charges, contract (3 values),
    # internet_service (3 values), online_security, tech_support, senior, partner
    # Target: churned with logistic relationship

    tenure = np.random.exponential(32, n).clip(1, 72).astype(int)
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.25, 0.20])
    internet = np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.34, 0.44, 0.22])
    monthly = np.round(20 + 25*(internet=='DSL') + 40*(internet=='Fiber optic') + np.random.normal(0, 8, n), 2).clip(18, 120)
    total = np.round(monthly * tenure * (1 + np.random.normal(0, 0.05, n)), 2)
    online_sec = np.where(internet == 'No', 0, np.random.binomial(1, 0.35, n))
    tech_sup = np.where(internet == 'No', 0, np.random.binomial(1, 0.30, n))
    senior = np.random.binomial(1, 0.16, n)
    partner = np.random.binomial(1, 0.48, n)

    logit = (-1.5 - 0.04*tenure + 1.2*(contract=='Month-to-month') - 0.8*(contract=='Two year')
             + 0.02*monthly + 0.5*(internet=='Fiber optic') - 0.5*online_sec
             - 0.4*tech_sup + 0.3*senior - 0.3*partner + np.random.normal(0, 0.5, n))
    churn = (np.random.random(n) < 1/(1+np.exp(-logit))).astype(int)

    df = pd.DataFrame({
        'tenure_months': tenure, 'monthly_charges': monthly, 'total_charges': total,
        'contract': contract, 'internet_service': internet,
        'online_security': online_sec, 'tech_support': tech_sup,
        'senior_citizen': senior, 'partner': partner, 'churned': churn
    })
    return df


def prepare_features(df):
    """One-hot encode categoricals and return X, y."""
    X = pd.get_dummies(df.drop('churned', axis=1), columns=['contract', 'internet_service'], drop_first=True)
    y = df['churned'].values
    return X, y


def train_model(model, model_name, X_train, X_test, y_train, y_test, experiment_name="churn-experiment"):
    """Train a model and optionally log to MLflow."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'f1': round(f1_score(y_test, y_pred), 4),
        'auc': round(roc_auc_score(y_test, y_proba), 4),
    }

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=model_name):
            # Log hyperparameters
            params = model.get_params()
            for k, v in params.items():
                if v is not None and not callable(v):
                    mlflow.log_param(k, v)
            # Log metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            # Log model
            mlflow.sklearn.log_model(model, "model")

    print(f"{model_name:25s} | Acc={metrics['accuracy']:.4f} | F1={metrics['f1']:.4f} | AUC={metrics['auc']:.4f}")
    return model, metrics


def train_all_models(experiment_name="churn-experiment"):
    """Train LogReg, RF, GBM and return results."""
    df = generate_churn_data()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42),
    }

    results = {}
    best_auc = 0
    best_name = None
    best_model = None

    print(f"\n{'Model':25s} | {'Accuracy':8s} | {'F1':8s} | {'AUC':8s}")
    print("-" * 60)

    for name, model in models.items():
        trained_model, metrics = train_model(model, name, X_train, X_test, y_train, y_test, experiment_name)
        results[name] = {'model': trained_model, 'metrics': metrics}
        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            best_name = name
            best_model = trained_model

    print(f"\nBest model: {best_name} (AUC={best_auc:.4f})")
    return results, best_name, best_model, (X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    results, best_name, best_model, _ = train_all_models()
