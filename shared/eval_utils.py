"""Model evaluation helpers."""


def print_classification_report(y_true, y_pred, model_name: str = "Model"):
    """Print a formatted classification report."""
    from sklearn.metrics import classification_report, confusion_matrix
    print(f"\n{'='*50}")
    print(f"  {model_name} — Classification Report")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


def print_regression_report(y_true, y_pred, model_name: str = "Model"):
    """Print common regression metrics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    print(f"\n{'='*50}")
    print(f"  {model_name} — Regression Report")
    print(f"{'='*50}")
    print(f"  MAE:  {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"  R²:   {r2_score(y_true, y_pred):.4f}")
