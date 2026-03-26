"""Generate large synthetic dataset for distributed training practice.

Creates 100K rows with 20+ features to simulate scale that
justifies SageMaker / distributed training approaches.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_large_dataset(n_rows: int = 100_000, seed: int = 42) -> pd.DataFrame:
    """Generate a large synthetic churn dataset with many features.

    Features (24 total):
        Demographics (6): age_bucket, gender, senior_citizen, partner, dependents, region
        Account (6): tenure, contract, payment_method, paperless_billing,
                      monthly_charges, total_charges
        Services (8): phone_service, internet_type, online_security, online_backup,
                       tech_support, streaming_tv, streaming_movies, device_protection
        Behavioral (4): support_tickets, login_frequency, avg_session_minutes,
                         last_interaction_days
        Derived (2): charge_per_month, tenure_bucket
    Target: churned (binary)
    """
    rng = np.random.default_rng(seed)

    # ── Demographics (6) ──────────────────────────────────────────────
    age_bucket = rng.choice(
        ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
        size=n_rows,
        p=[0.12, 0.25, 0.22, 0.18, 0.13, 0.10],
    )
    gender = rng.choice(["Male", "Female"], size=n_rows)
    senior_citizen = np.where(
        np.isin(age_bucket, ["56-65", "65+"]),
        rng.choice([0, 1], size=n_rows, p=[0.3, 0.7]),
        rng.choice([0, 1], size=n_rows, p=[0.95, 0.05]),
    )
    partner = rng.choice([0, 1], size=n_rows, p=[0.48, 0.52])
    dependents = rng.choice([0, 1], size=n_rows, p=[0.70, 0.30])
    region = rng.choice(
        ["Northeast", "Southeast", "Midwest", "Southwest", "West"],
        size=n_rows,
        p=[0.20, 0.22, 0.18, 0.15, 0.25],
    )

    # ── Account info (6) ─────────────────────────────────────────────
    tenure = rng.exponential(scale=24, size=n_rows).clip(1, 72).astype(int)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_rows,
        p=[0.55, 0.25, 0.20],
    )
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n_rows,
        p=[0.35, 0.20, 0.22, 0.23],
    )
    paperless_billing = rng.choice([0, 1], size=n_rows, p=[0.40, 0.60])

    # Monthly charges — correlated with contract type
    base_charge = rng.normal(loc=65, scale=20, size=n_rows).clip(18, 120)
    contract_discount = np.where(
        contract == "Two year", -10,
        np.where(contract == "One year", -5, 0),
    )
    monthly_charges = np.round(base_charge + contract_discount, 2)

    # Total charges — derived from tenure * monthly with noise
    total_charges = np.round(
        tenure * monthly_charges * rng.uniform(0.9, 1.1, size=n_rows), 2
    )

    # ── Services (8) ─────────────────────────────────────────────────
    phone_service = rng.choice([0, 1], size=n_rows, p=[0.10, 0.90])
    internet_type = rng.choice(
        ["None", "DSL", "Fiber optic", "Cable"],
        size=n_rows,
        p=[0.08, 0.30, 0.35, 0.27],
    )
    has_internet = (internet_type != "None").astype(int)

    # Add-on services — only available when internet is active
    online_security = (has_internet & rng.choice([0, 1], size=n_rows, p=[0.50, 0.50])).astype(int)
    online_backup = (has_internet & rng.choice([0, 1], size=n_rows, p=[0.55, 0.45])).astype(int)
    tech_support = (has_internet & rng.choice([0, 1], size=n_rows, p=[0.55, 0.45])).astype(int)
    streaming_tv = (has_internet & rng.choice([0, 1], size=n_rows, p=[0.50, 0.50])).astype(int)
    streaming_movies = (has_internet & rng.choice([0, 1], size=n_rows, p=[0.50, 0.50])).astype(int)
    device_protection = (has_internet & rng.choice([0, 1], size=n_rows, p=[0.55, 0.45])).astype(int)

    # ── Behavioral (4) ───────────────────────────────────────────────
    support_tickets = rng.poisson(lam=1.5, size=n_rows)
    login_frequency = rng.poisson(lam=12, size=n_rows).clip(0, 60)
    avg_session_minutes = rng.exponential(scale=15, size=n_rows).round(1).clip(0.5, 120)
    last_interaction_days = rng.exponential(scale=30, size=n_rows).astype(int).clip(0, 365)

    # ── Derived (2) ──────────────────────────────────────────────────
    charge_per_month = np.where(
        tenure > 0, np.round(total_charges / tenure, 2), monthly_charges
    )
    tenure_bucket = pd.cut(
        tenure,
        bins=[0, 6, 12, 24, 48, 72],
        labels=["0-6m", "7-12m", "13-24m", "25-48m", "49-72m"],
    ).astype(str)

    # ── Target: churned (logistic model) ─────────────────────────────
    # Build a logit score from features that realistically drive churn
    logit = (
        -1.0                                                     # intercept
        + 0.8 * (contract == "Month-to-month").astype(float)     # short contract → churn
        - 0.5 * (contract == "Two year").astype(float)           # long contract → retain
        + 0.4 * (payment_method == "Electronic check").astype(float)
        + 0.005 * monthly_charges                                # higher bill → churn
        - 0.02 * tenure                                          # longer tenure → retain
        + 0.15 * support_tickets                                 # more tickets → churn
        - 0.3 * online_security                                  # security → retain
        - 0.25 * tech_support                                    # support → retain
        + 0.3 * senior_citizen                                   # seniors churn more
        - 0.01 * login_frequency                                 # engaged → retain
        + 0.005 * last_interaction_days                           # inactive → churn
        - 0.2 * partner                                          # partner → retain
        + 0.2 * paperless_billing                                # paperless → churn
        + rng.normal(0, 0.3, size=n_rows)                        # noise
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churned = (rng.random(n_rows) < churn_prob).astype(int)

    # ── Assemble DataFrame ───────────────────────────────────────────
    df = pd.DataFrame({
        # Demographics
        "age_bucket": age_bucket,
        "gender": gender,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "region": region,
        # Account
        "tenure": tenure,
        "contract": contract,
        "payment_method": payment_method,
        "paperless_billing": paperless_billing,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        # Services
        "phone_service": phone_service,
        "internet_type": internet_type,
        "online_security": online_security,
        "online_backup": online_backup,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "device_protection": device_protection,
        # Behavioral
        "support_tickets": support_tickets,
        "login_frequency": login_frequency,
        "avg_session_minutes": avg_session_minutes,
        "last_interaction_days": last_interaction_days,
        # Derived
        "charge_per_month": charge_per_month,
        "tenure_bucket": tenure_bucket,
        # Target
        "churned": churned,
    })

    return df


def main():
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    df = generate_large_dataset()
    df.to_parquet(data_dir / "large_churn_data.parquet", index=False)
    df.head(10_000).to_csv(data_dir / "large_churn_sample.csv", index=False)

    print(f"Generated {len(df):,} rows x {len(df.columns)} columns")
    print(f"Churn rate: {df['churned'].mean():.1%}")
    print(f"Saved to {data_dir}")


if __name__ == "__main__":
    main()
