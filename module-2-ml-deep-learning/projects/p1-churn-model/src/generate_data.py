"""Generate synthetic telecom churn dataset.

Creates a realistic dataset with known relationships so the learner
can validate their model's feature importance findings.

Key relationships baked in:
- High monthly charges + short tenure → more churn
- No contract (month-to-month) → more churn
- Low engagement (support tickets, no online services) → more churn
- Senior citizens churn slightly more
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_churn_data(n_customers: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic churn dataset."""
    np.random.seed(seed)

    # --- Demographics ---
    senior = np.random.binomial(1, 0.16, n_customers)
    gender = np.random.choice(['Male', 'Female'], n_customers)
    partner = np.random.binomial(1, 0.48, n_customers)
    dependents = np.random.binomial(1, 0.30, n_customers)

    # --- Account Info ---
    tenure = np.random.exponential(scale=32, size=n_customers).clip(1, 72).astype(int)
    contract = np.random.choice(
        ['Month-to-month', 'One year', 'Two year'],
        n_customers,
        p=[0.55, 0.25, 0.20]
    )
    payment_method = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
        n_customers,
        p=[0.35, 0.20, 0.22, 0.23]
    )
    paperless = np.random.binomial(1, 0.60, n_customers)

    # --- Services ---
    phone_service = np.random.binomial(1, 0.90, n_customers)
    multiple_lines = (phone_service & np.random.binomial(1, 0.42, n_customers))
    internet_service = np.random.choice(
        ['DSL', 'Fiber optic', 'No'],
        n_customers,
        p=[0.34, 0.44, 0.22]
    )
    online_security = np.where(
        internet_service == 'No', 0,
        np.random.binomial(1, 0.35, n_customers)
    )
    online_backup = np.where(
        internet_service == 'No', 0,
        np.random.binomial(1, 0.40, n_customers)
    )
    tech_support = np.where(
        internet_service == 'No', 0,
        np.random.binomial(1, 0.30, n_customers)
    )
    streaming_tv = np.where(
        internet_service == 'No', 0,
        np.random.binomial(1, 0.45, n_customers)
    )
    streaming_movies = np.where(
        internet_service == 'No', 0,
        np.random.binomial(1, 0.45, n_customers)
    )

    # --- Charges ---
    # Base charge depends on services
    base_charge = 20 + np.random.normal(0, 3, n_customers)
    base_charge += phone_service * 20
    base_charge += multiple_lines * 10
    base_charge += (internet_service == 'DSL') * 25
    base_charge += (internet_service == 'Fiber optic') * 40
    base_charge += online_security * 10
    base_charge += online_backup * 8
    base_charge += tech_support * 10
    base_charge += streaming_tv * 12
    base_charge += streaming_movies * 12
    monthly_charges = np.round(base_charge.clip(18, 120), 2)

    total_charges = np.round(monthly_charges * tenure * (1 + np.random.normal(0, 0.05, n_customers)), 2)
    total_charges = total_charges.clip(monthly_charges)

    # --- Support Interactions ---
    support_tickets = np.random.poisson(lam=1.5, size=n_customers)
    support_tickets = np.where(tenure < 12, support_tickets + 1, support_tickets)

    # --- Churn (target) ---
    # Logistic model with known coefficients
    churn_logit = (
        -1.5                                          # base (overall ~20% churn)
        + 0.3 * senior                                # seniors churn more
        - 0.04 * tenure                               # longer tenure = less churn
        + 1.2 * (contract == 'Month-to-month')        # month-to-month = much more churn
        - 0.8 * (contract == 'Two year')              # two year = much less churn
        + 0.02 * monthly_charges                      # higher charges = more churn
        + 0.5 * (internet_service == 'Fiber optic')   # fiber = more churn (price?)
        - 0.5 * online_security                       # security = less churn
        - 0.4 * tech_support                          # support = less churn
        + 0.4 * (payment_method == 'Electronic check') # e-check = more churn
        + 0.1 * support_tickets                       # more tickets = more churn
        - 0.3 * partner                               # partner = less churn
        + np.random.normal(0, 0.5, n_customers)       # noise
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn = (np.random.random(n_customers) < churn_prob).astype(int)

    df = pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'gender': gender,
        'senior_citizen': senior,
        'partner': partner,
        'dependents': dependents,
        'tenure_months': tenure,
        'contract': contract,
        'payment_method': payment_method,
        'paperless_billing': paperless,
        'phone_service': phone_service,
        'multiple_lines': multiple_lines,
        'internet_service': internet_service,
        'online_security': online_security,
        'online_backup': online_backup,
        'tech_support': tech_support,
        'streaming_tv': streaming_tv,
        'streaming_movies': streaming_movies,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'support_tickets': support_tickets,
        'churned': churn,
    })

    # Inject some messiness for EDA practice
    # Missing values in total_charges (~2%)
    mask = np.random.random(n_customers) < 0.02
    df.loc[mask, 'total_charges'] = np.nan

    # A few duplicates (~0.5%)
    n_dupes = int(n_customers * 0.005)
    dupes = df.sample(n_dupes, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)

    return df


def main():
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    df = generate_churn_data()
    df.to_csv(data_dir / "churn_data.csv", index=False)

    print(f"Generated {len(df)} rows")
    print(f"Churn rate: {df['churned'].mean():.1%}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"\nSaved to {data_dir / 'churn_data.csv'}")


if __name__ == "__main__":
    main()
