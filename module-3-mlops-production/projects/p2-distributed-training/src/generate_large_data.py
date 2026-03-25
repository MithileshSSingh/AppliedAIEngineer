"""Generate a large churn dataset (100K rows) for distributed training practice."""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_large_churn_dataset(n: int = 100_000, seed: int = 42) -> pd.DataFrame:
    """Generate 100K customer records with churn labels."""
    np.random.seed(seed)

    # Core features
    tenure = np.random.exponential(24, n).clip(1, 72).astype(int)
    monthly_charges = np.random.normal(65, 25, n).clip(20, 150).round(2)
    total_charges = (tenure * monthly_charges + np.random.normal(0, 500, n)).clip(20).round(2)
    contract = np.random.choice([0, 1, 2], n, p=[0.55, 0.25, 0.20])
    internet = np.random.choice([0, 1, 2], n, p=[0.44, 0.34, 0.22])
    num_products = np.random.randint(1, 6, n)
    support_calls = np.random.poisson(2, n)
    payment_method = np.random.randint(0, 4, n)

    # Derived features
    avg_monthly = total_charges / tenure.clip(1)
    charge_ratio = monthly_charges / avg_monthly.clip(1)

    # Churn model
    log_odds = (-2.0
                + 0.05 * (monthly_charges - 65)
                - 0.04 * tenure
                + 0.15 * support_calls
                - 0.60 * contract
                - 0.30 * internet
                + 0.10 * (num_products - 3)
                + np.random.normal(0, 0.3, n))

    churn_prob = 1 / (1 + np.exp(-log_odds))
    churn = (np.random.random(n) < churn_prob).astype(int)

    # Add some missing values (~1%)
    df = pd.DataFrame({
        'customer_id': [f'C{i:07d}' for i in range(n)],
        'tenure_months': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract_encoded': contract,
        'internet_encoded': internet,
        'num_products': num_products,
        'support_calls': support_calls,
        'payment_method': payment_method,
        'avg_monthly_charges': avg_monthly.round(2),
        'charge_trend_ratio': charge_ratio.round(4),
        'churn': churn,
    })

    missing_idx = np.random.choice(n, int(n * 0.01), replace=False)
    df.loc[missing_idx[:len(missing_idx)//2], 'monthly_charges'] = np.nan
    df.loc[missing_idx[len(missing_idx)//2:], 'total_charges'] = np.nan

    return df


def main():
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)

    print("Generating 100K customer churn dataset...")
    df = generate_large_churn_dataset(100_000)

    print(f"Shape: {df.shape}")
    print(f"Churn rate: {df['churn'].mean():.2%}")
    print(f"Missing values: {df.isnull().sum().sum()}")

    # Split into train/test for SageMaker
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['churn'], random_state=42)

    train_df.to_csv(data_dir / 'train.csv', index=False)
    test_df.to_csv(data_dir / 'test.csv', index=False)

    print(f"\nSaved:")
    print(f"  Train: {len(train_df)} rows -> {data_dir / 'train.csv'}")
    print(f"  Test:  {len(test_df)} rows -> {data_dir / 'test.csv'}")


if __name__ == '__main__':
    main()
