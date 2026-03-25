"""Generate synthetic daily sales time series data."""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_sales_ts(seed: int = 42) -> pd.DataFrame:
    """Generate 3 years of daily sales for 3 product categories."""
    np.random.seed(seed)
    dates = pd.date_range('2022-01-01', '2024-12-31', freq='D')
    n = len(dates)

    categories = {
        'Electronics': {'base': 150, 'trend': 0.05, 'seasonal_amp': 40, 'noise': 12},
        'Clothing': {'base': 200, 'trend': 0.03, 'seasonal_amp': 60, 'noise': 15},
        'Food': {'base': 300, 'trend': 0.01, 'seasonal_amp': 20, 'noise': 10},
    }

    records = []
    for cat, params in categories.items():
        trend = params['base'] * (1 + params['trend'] * np.arange(n) / 365)
        yearly = params['seasonal_amp'] * np.sin(2 * np.pi * np.arange(n) / 365.25)
        weekly = 10 * np.sin(2 * np.pi * np.arange(n) / 7)

        # Holiday bumps (Black Friday area, Christmas)
        holiday_bump = np.zeros(n)
        for i, d in enumerate(dates):
            if d.month == 11 and d.day >= 20 and d.day <= 30:
                holiday_bump[i] = params['base'] * 0.5
            if d.month == 12 and d.day >= 15 and d.day <= 25:
                holiday_bump[i] = params['base'] * 0.8

        noise = np.random.normal(0, params['noise'], n)
        sales = (trend + yearly + weekly + holiday_bump + noise).clip(10).round(0)

        # Inject some missing values (~1%)
        mask = np.random.random(n) < 0.01
        sales[mask] = np.nan

        for i, date in enumerate(dates):
            records.append({
                'date': date,
                'category': cat,
                'sales': sales[i],
            })

    return pd.DataFrame(records)


def main():
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    df = generate_sales_ts()
    df.to_csv(data_dir / "daily_sales.csv", index=False)

    print(f"Generated {len(df)} rows")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Categories: {df['category'].unique().tolist()}")
    print(f"Missing values: {df['sales'].isna().sum()}")
    print(f"\nSaved to {data_dir / 'daily_sales.csv'}")


if __name__ == "__main__":
    main()
