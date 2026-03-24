"""Generate synthetic sales data for the Sales Pipeline project.

Run this if you don't want to download the Kaggle dataset.
Usage: python generate_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_sales_data(n_orders: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic sales dataset."""
    np.random.seed(seed)

    categories = {
        "Electronics": ["Laptop", "Monitor", "Tablet", "Phone", "Headset"],
        "Peripherals": ["Mouse", "Keyboard", "Webcam", "USB Hub", "Mousepad"],
        "Furniture": ["Desk", "Chair", "Bookshelf", "Filing Cabinet", "Lamp"],
        "Supplies": ["Paper", "Pens", "Notebooks", "Stapler", "Tape"],
    }

    base_prices = {
        "Laptop": 899, "Monitor": 399, "Tablet": 499, "Phone": 699, "Headset": 79,
        "Mouse": 29, "Keyboard": 69, "Webcam": 89, "USB Hub": 39, "Mousepad": 15,
        "Desk": 299, "Chair": 249, "Bookshelf": 149, "Filing Cabinet": 129, "Lamp": 49,
        "Paper": 12, "Pens": 8, "Notebooks": 15, "Stapler": 11, "Tape": 5,
    }

    base_costs = {k: v * np.random.uniform(0.4, 0.7) for k, v in base_prices.items()}

    regions = ["North", "South", "East", "West"]
    customer_ids = [f"C{i:04d}" for i in range(1, 201)]

    # Generate dates weighted toward recent months
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    date_weights = np.linspace(0.5, 1.5, len(dates))
    date_weights /= date_weights.sum()

    rows = []
    for i in range(n_orders):
        category = np.random.choice(list(categories.keys()),
                                     p=[0.25, 0.30, 0.20, 0.25])
        product = np.random.choice(categories[category])
        quantity = np.random.randint(1, 15)
        unit_price = base_prices[product] * np.random.uniform(0.85, 1.15)
        unit_cost = base_costs[product] * np.random.uniform(0.90, 1.10)
        discount = np.random.choice([0, 0, 0, 0.05, 0.10, 0.15, 0.20])

        rows.append({
            "order_id": f"ORD-{i+1:05d}",
            "date": np.random.choice(dates, p=date_weights),
            "customer_id": np.random.choice(customer_ids),
            "region": np.random.choice(regions),
            "category": category,
            "product": product,
            "quantity": quantity,
            "unit_price": round(unit_price, 2),
            "unit_cost": round(unit_cost, 2),
            "discount": discount,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = (df["unit_price"] * df["quantity"] * (1 - df["discount"])).round(2)
    df["cost"] = (df["unit_cost"] * df["quantity"]).round(2)
    df["profit"] = (df["revenue"] - df["cost"]).round(2)

    # Inject some realistic messiness
    # Missing values
    missing_idx = df.sample(frac=0.03, random_state=seed).index
    df.loc[missing_idx[:len(missing_idx)//2], "region"] = np.nan
    df.loc[missing_idx[len(missing_idx)//2:], "discount"] = np.nan

    # A few duplicate orders
    dupes = df.sample(10, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


if __name__ == "__main__":
    output_path = Path(__file__).parent.parent / "data" / "sales_data.csv"
    output_path.parent.mkdir(exist_ok=True)

    df = generate_sales_data()
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} rows → {output_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Missing values:\n{df.isnull().sum()}")
