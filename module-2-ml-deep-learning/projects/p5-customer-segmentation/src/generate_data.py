"""Generate synthetic customer data for segmentation."""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_customer_data(n_customers: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate customer data with hidden segments."""
    np.random.seed(seed)

    # 5 hidden segments
    segment_probs = [0.20, 0.25, 0.20, 0.20, 0.15]
    segments = np.random.choice(5, n_customers, p=segment_probs)

    # Segment profiles (Champions, Loyal, Promising, At Risk, Lost)
    profiles = {
        0: {'recency': (5, 3), 'frequency': (25, 5), 'monetary': (800, 150), 'tenure': (36, 8), 'name': 'Champions'},
        1: {'recency': (15, 8), 'frequency': (15, 4), 'monetary': (400, 100), 'tenure': (30, 10), 'name': 'Loyal'},
        2: {'recency': (20, 10), 'frequency': (8, 3), 'monetary': (200, 80), 'tenure': (12, 5), 'name': 'Promising'},
        3: {'recency': (60, 20), 'frequency': (10, 4), 'monetary': (350, 120), 'tenure': (24, 8), 'name': 'At Risk'},
        4: {'recency': (120, 30), 'frequency': (3, 2), 'monetary': (80, 40), 'tenure': (18, 10), 'name': 'Lost'},
    }

    data = []
    for i in range(n_customers):
        seg = segments[i]
        p = profiles[seg]

        recency = max(1, int(np.random.normal(*p['recency'])))
        frequency = max(1, int(np.random.normal(*p['frequency'])))
        monetary = max(10, round(np.random.normal(*p['monetary']), 2))
        tenure = max(1, int(np.random.normal(*p['tenure'])))

        # Derived features
        avg_order_value = round(monetary / frequency, 2)
        purchase_rate = round(frequency / max(tenure, 1), 2)

        # Behavioral features
        website_visits = max(0, int(np.random.normal(frequency * 3, frequency)))
        email_opens = max(0, int(np.random.normal(10 - seg * 2, 3)))
        support_tickets = max(0, int(np.random.poisson(1 + seg * 0.5)))

        data.append({
            'customer_id': f'C{i+1:04d}',
            'recency_days': recency,
            'frequency': frequency,
            'monetary': monetary,
            'tenure_months': tenure,
            'avg_order_value': avg_order_value,
            'purchase_rate': purchase_rate,
            'website_visits_30d': website_visits,
            'email_open_rate': round(np.random.beta(2 + (4 - seg), 2 + seg), 2),
            'support_tickets': support_tickets,
            'true_segment': profiles[seg]['name'],
        })

    return pd.DataFrame(data)


def main():
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    df = generate_customer_data()
    df.to_csv(data_dir / "customers.csv", index=False)

    print(f"Generated {len(df)} customers")
    print(f"\nTrue segment distribution:")
    print(df['true_segment'].value_counts())
    print(f"\nSaved to {data_dir / 'customers.csv'}")


if __name__ == "__main__":
    main()
